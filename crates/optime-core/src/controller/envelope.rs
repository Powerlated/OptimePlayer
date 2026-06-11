//! The per-tick ADSR/LFO pass over the active notes (the audio-side envelope engine).

use super::lfo::{lfo_type, lfo_tick, LfoParams};
use super::volume::{calc_channel_volume, decibel_db};
use super::{ActiveNote, AdsrState, Controller, SynthConfig};

impl Controller {
    /// Runs one ADSR/LFO update pass over the active notes (mirrors the original exactly,
    /// including that at most one finished note is removed per tick).
    pub(super) fn process_active_notes(&mut self, config: &SynthConfig) {
        let mut index_to_delete: Option<usize> = None;
        let ticks = self.sequence.ticks_elapsed;

        for index in 0..self.active_notes.len() {
            let mut entry = self.active_notes[index];
            let t = entry.track_num;
            let si = entry.synth_instr_index;

            let (instr_start, instr_playing, looping, data_len, sample_t) = {
                let instr = self.synthesizers[t].instr(si);
                (
                    instr.start_time,
                    instr.playing,
                    instr.sample.looping,
                    instr.sample.data.len(),
                    instr.sample_t,
                )
            };

            if instr_start == entry.start_time && instr_playing {
                // Cut voices whose (non-looping) sample has run out.
                if !looping && sample_t > data_len as f64 {
                    index_to_delete = Some(index);
                    self.synthesizers[t].cut_instrument(si);
                }

                // Begin release once the note's scheduled duration elapses.
                if ticks >= entry.end_time
                    && !entry.from_keyboard
                    && entry.adsr_state != AdsrState::Release
                {
                    self.notes_on[t][entry.midi_note as usize] = 0;
                    entry.adsr_state = AdsrState::Release;
                }

                self.apply_lfo(&mut entry, config);
                self.apply_adsr(&mut entry, index, &mut index_to_delete);
            } else {
                index_to_delete = Some(index);
                self.notes_on[t][entry.midi_note as usize] = 0;
            }

            self.active_notes[index] = entry;
        }

        if let Some(i) = index_to_delete {
            self.active_notes.remove(i);
        }
    }

    /// The track-level decibel attenuation pokediamond folds into `chn->userDecay`:
    /// `DecibelSquareTable[volume] + DecibelSquareTable[expression] + DecibelSquareTable[master]`
    /// (`TrackUpdateChannel`). Master volume is per-track here rather than player-global, which is
    /// faithful for the common case where it stays 127 (a 0 dB no-op).
    pub(super) fn track_volume_db(&self, t: usize) -> i32 {
        let tr = &self.sequence.tracks[t];
        decibel_db(tr.volume) + decibel_db(tr.expression) + decibel_db(tr.master_volume)
    }

    /// LFO update for one note, ported faithfully (including the DS fixed-point math).
    fn apply_lfo(&mut self, entry: &mut ActiveNote, config: &SynthConfig) {
        let t = entry.track_num;
        let track = &self.sequence.tracks[t];
        let (lfo_depth, lfo_delay, lfo_type, lfo_speed, lfo_range) = (
            track.lfo_depth,
            track.lfo_delay,
            track.lfo_type,
            track.lfo_speed,
            track.lfo_range,
        );

        // Whether the LFO delay has elapsed (the phase advances this tick). pokediamond gates both
        // the value and the phase on a single `delayCounter`; see [`lfo_tick`].
        let delay_elapsed = entry.delay_counter >= lfo_delay;
        let params = LfoParams {
            depth: lfo_depth,
            delay: lfo_delay,
            lfo_type,
            speed: lfo_speed,
            range: lfo_range,
        };
        let lfo_value = lfo_tick(&params, &mut entry.lfo_counter, &mut entry.delay_counter);

        // pokediamond adds the LFO into whichever target it modulates. Pitch modulation is applied
        // only on ticks where the phase advances (delay elapsed); the value is in 1/64ths of a
        // semitone. Volume modulation is summed (in dB) into the channel volume by `apply_adsr`
        // via `lfo_vol_db`; it is naturally 0 during the LFO delay. (Pan LFO is not represented —
        // pan is a per-track stereo stage here, not per-voice.)
        entry.lfo_vol_db = if lfo_type == lfo_type::VOLUME {
            lfo_value as i32
        } else {
            0
        };
        if delay_elapsed && lfo_value != 0 && lfo_type == lfo_type::PITCH {
            self.synthesizers[t]
                .instr_mut(entry.synth_instr_index)
                .set_finetune_lfo(lfo_value as f64 / 64.0, config.tuning);
        }
    }

    /// ADSR envelope advance for one note.
    fn apply_adsr(
        &mut self,
        entry: &mut ActiveNote,
        index: usize,
        index_to_delete: &mut Option<usize>,
    ) {
        let t = entry.track_num;
        let si = entry.synth_instr_index;
        // pokediamond sums every contribution in the decibel domain before one conversion
        // (`SND_ExChannelMain`): velocity + envelope + userDecay (track volume + expression +
        // master) + volume-LFO. We fold the non-envelope terms into `extra_db` here.
        let extra_db = self.track_volume_db(t) + entry.lfo_vol_db;
        match entry.adsr_state {
            AdsrState::Attack => {
                entry.adsr_timer =
                    -(((-entry.attack_coefficient).wrapping_mul(entry.adsr_timer)) >> 8);
                self.synthesizers[t].instr_mut(si).volume =
                    calc_channel_volume(entry.velocity, entry.adsr_timer, extra_db);
                if entry.adsr_timer == 0 {
                    entry.adsr_state = AdsrState::Decay;
                }
            }
            AdsrState::Decay => {
                entry.adsr_timer -= entry.decay_coefficient;
                if entry.adsr_timer <= entry.sustain_level {
                    entry.adsr_timer = entry.sustain_level;
                    entry.adsr_state = AdsrState::Sustain;
                }
                self.synthesizers[t].instr_mut(si).volume =
                    calc_channel_volume(entry.velocity, entry.adsr_timer, extra_db);
            }
            AdsrState::Sustain => {
                self.synthesizers[t].instr_mut(si).volume =
                    calc_channel_volume(entry.velocity, entry.adsr_timer, extra_db);
            }
            AdsrState::Release => {
                // pokediamond cuts a channel only once its release envelope reaches the floor
                // (`SND_ChannelMain`: `envStatus == RELEASE && vol <= -723`, i.e. attenuation
                // <= -92544), uniformly for every channel type. The original OptimePlayer also
                // force-cut PSG-pulse voices immediately on release, which has no basis in
                // `SND_exChannel.c` and robs them of their release tail — so we don't.
                if entry.adsr_timer <= -92544 {
                    self.synthesizers[t].cut_instrument(si);
                    *index_to_delete = Some(index);
                    self.notes_on[t][entry.midi_note as usize] = 0;
                } else {
                    entry.adsr_timer -= entry.release_coefficient;
                    self.synthesizers[t].instr_mut(si).volume =
                        calc_channel_volume(entry.velocity, entry.adsr_timer, extra_db);
                }
            }
        }
    }
}

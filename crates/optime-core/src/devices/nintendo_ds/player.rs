//! [`NdsPlayer`]: the DS device player. Runs the SSEQ sequencer and the per-note ADSR/LFO
//! hardware model, and emits standardized [`SynthEvent`]s for the synthesis layer.

use std::sync::Arc;

use super::lfo::{lfo_tick, lfo_type, LfoParams};
use super::sequence::{Message, MessageType, Sequence};
use super::tables::SQUARE_WAVES;
use super::volume::{calc_channel_volume, decibel_db};
use super::{InstrumentType, Sdat};
use crate::devices::{SynthEvent, TickFeedback, VoiceId, VoicePitch};
use crate::sample::{decode_adpcm, decode_pcm16, decode_pcm8, Sample};
use crate::synth_controller::SynthConfig;
use crate::tuning::midi_note_to_hz;
use crate::util::{read_u16, read_u32, read_u8};
use crate::TRACK_COUNT;

/// ADSR envelope stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdsrState {
    Attack,
    Decay,
    Sustain,
    Release,
}

/// Per-note runtime state (all-`Copy` so the tick loop can work on a local copy).
#[derive(Debug, Clone, Copy)]
struct ActiveNote {
    track_num: usize,
    midi_note: u8,
    velocity: i32,
    voice: VoiceId,
    end_time: u32,
    adsr_state: AdsrState,
    adsr_timer: i32,
    from_keyboard: bool,
    lfo_counter: i32,
    /// Shared LFO delay/phase counter (pokediamond's single `SNDLfo::delayCounter`).
    delay_counter: i32,
    /// Volume-LFO contribution (dB) computed this tick, summed into the channel volume.
    lfo_vol_db: i32,
    // Resolved instrument coefficients for this note's region.
    attack_coefficient: i32,
    decay_coefficient: i32,
    sustain_level: i32,
    release_coefficient: i32,
}

/// The DS device player: SSEQ sequencer + decoded sample archives + the pokediamond note model.
///
/// One [`tick`](Self::tick) is one DS sequencer-timer period (every `CYCLES_PER_TICK` cycles of
/// the 33.51 MHz clock, ≈192 Hz); the BPM timer inside gates actual sequencer steps.
pub struct NdsPlayer {
    /// The running SSEQ interpreter.
    pub sequence: Sequence,
    instrument_bank: super::InstrumentBank,
    decoded_sample_archives: Vec<Option<Vec<Arc<Sample>>>>,
    squares: Vec<Arc<Sample>>,
    active_notes: Vec<ActiveNote>,
    /// Which track receives live keyboard input, if any (its sequenced notes are silenced).
    pub active_keyboard_track_num: Option<usize>,
    bpm_timer: u32,
    next_voice: VoiceId,
}

impl NdsPlayer {
    /// Binds sequence `sseq_id` from `sdat`, decoding the linked sample archives up front.
    /// Returns `None` if the sequence or its bank is missing.
    pub fn new(sdat: &Sdat, sseq_id: u32) -> Option<NdsPlayer> {
        let sseq_info = sdat.sseq_infos.get(sseq_id as usize)?.clone()?;
        let bank_id = sseq_info.bank as usize;
        let bank_info = sdat.sbnk_infos.get(bank_id)?.clone()?;
        let instrument_bank = sdat.instrument_banks.get(bank_id)?.clone()?;

        let sseq_file = sdat.file(sseq_info.file_id)?;
        let sseq_arc: Arc<[u8]> = Arc::from(sseq_file.to_vec());

        // Decode the up-to-four linked sample archives.
        let mut decoded_sample_archives: Vec<Option<Vec<Arc<Sample>>>> = vec![None; 4];
        for (i, &swar_id) in bank_info.swar_id.iter().enumerate() {
            let Some(Some(swar_info)) = sdat.swar_infos.get(swar_id as usize) else {
                continue;
            };
            let Some(swar_file) = sdat.file(swar_info.file_id) else {
                continue;
            };

            let sample_count = read_u32(swar_file, 0x38) as usize;
            let mut archive = Vec::with_capacity(sample_count);
            for j in 0..sample_count {
                let sample_offset = read_u32(swar_file, 0x3C + j * 4) as usize;

                let wav_type = read_u8(swar_file, sample_offset);
                let loop_flag = read_u8(swar_file, sample_offset + 1);
                let sample_rate_hdr = read_u16(swar_file, sample_offset + 2) as f64;
                let swar_loop_offset = read_u16(swar_file, sample_offset + 6) as i64;
                let swar_sample_length = read_u32(swar_file, sample_offset + 8) as i64;

                let data_len = ((swar_loop_offset + swar_sample_length) * 4) as usize;
                let start = sample_offset + 0xC;
                let sample_data = swar_file.get(start..start + data_len).unwrap_or(&[]);

                let (decoded, loop_point) = match wav_type {
                    0 => (decode_pcm8(sample_data), swar_loop_offset * 4),
                    1 => (decode_pcm16(sample_data), swar_loop_offset * 2),
                    2 => (decode_adpcm(sample_data), swar_loop_offset * 8 - 8),
                    _ => (Vec::new(), 0),
                };

                let mut sample =
                    Sample::new(decoded, 440.0, sample_rate_hdr, loop_flag != 0, loop_point);
                sample.sample_length = (swar_sample_length * 4) as usize;
                archive.push(Arc::new(sample));
            }
            decoded_sample_archives[i] = Some(archive);
        }

        // Build the eight PSG square-wave samples.
        let squares = SQUARE_WAVES
            .iter()
            .map(|wave| {
                let mut s = Sample::new(wave.to_vec(), 1.0, 8.0, true, 0);
                s.is_psg_square = true;
                Arc::new(s)
            })
            .collect();

        let data_offset = read_u32(&sseq_arc, 0x18);
        let sequence = Sequence::new(sseq_arc, data_offset, 1024);

        Some(NdsPlayer {
            sequence,
            instrument_bank,
            decoded_sample_archives,
            squares,
            active_notes: Vec::new(),
            active_keyboard_track_num: None,
            bpm_timer: 0,
            next_voice: 0,
        })
    }

    /// Sequencer steps executed (SSEQ ticks; the visualizer timeline).
    pub fn steps_elapsed(&self) -> u32 {
        self.sequence.ticks_elapsed
    }

    /// Current sequencer step rate in Hz: the ≈192 Hz hardware timer scaled by track 0's BPM.
    pub fn step_rate(&self) -> f64 {
        let base = crate::DS_CLOCK_RATE as f64 / crate::CYCLES_PER_TICK as f64;
        base * f64::from(self.sequence.tracks[0].bpm.max(1)) / 240.0
    }

    /// Advances note envelopes/LFOs, then runs any due sequencer steps and converts their
    /// messages into [`SynthEvent`]s.
    pub fn tick(
        &mut self,
        feedback: &mut TickFeedback,
        config: &SynthConfig,
        events: &mut Vec<SynthEvent>,
    ) {
        self.process_active_notes(feedback, config, events);
        feedback.ended_voices.clear();

        self.bpm_timer += self.sequence.tracks[0].bpm;
        while self.bpm_timer >= 240 {
            self.bpm_timer -= 240;

            // Report which tracks still have sounding/releasing channels, so the sequence can
            // honor pokediamond's `noteFinishWait` (stall after a zero-duration note).
            let mut track_has_channels = [false; TRACK_COUNT];
            for note in &self.active_notes {
                if let Some(slot) = track_has_channels.get_mut(note.track_num) {
                    *slot = true;
                }
            }
            self.sequence.tick(&track_has_channels);

            while let Some(msg) = self.sequence.message_buffer.pop() {
                self.handle_message(msg, config, events);
            }
        }
    }

    /// Applies one sequence message, translating it into standardized events.
    fn handle_message(&mut self, msg: Message, config: &SynthConfig, events: &mut Vec<SynthEvent>) {
        match msg.msg_type {
            MessageType::PlayNote => self.play_note(msg, config, events),
            MessageType::Jump => events.push(SynthEvent::Looped),
            MessageType::TrackEnded => {
                let any_active = self.sequence.tracks.iter().any(|tr| tr.active);
                if !any_active {
                    events.push(SynthEvent::Ended);
                }
            }
            MessageType::VolumeChange => {
                // No-op: track volume (with expression and master) is summed in the decibel
                // domain per tick by `track_volume_db`, matching pokediamond, rather than applied
                // as a separate linear mixer gain.
            }
            MessageType::PanChange => events.push(SynthEvent::TrackPan {
                track: msg.track_num,
                pan: msg.param0 as f64 / 128.0,
            }),
            MessageType::PitchBend => {
                let track = &self.sequence.tracks[msg.track_num];
                // `pitch_bend` is already a signed byte (pokediamond `par._s8`, set by `0xC4`);
                // scale by half the bend range, in 1/64 semitones.
                let pitch_bend = track.pitch_bend;
                let semitones = (pitch_bend as f64) * (track.pitch_bend_range as f64 / 2.0) / 64.0;
                events.push(SynthEvent::TrackDetune {
                    track: msg.track_num,
                    semitones,
                });
            }
            MessageType::InstrumentChange => {}
        }
    }

    /// Starts a note from a [`MessageType::PlayNote`] message.
    fn play_note(&mut self, msg: Message, config: &SynthConfig, events: &mut Vec<SynthEvent>) {
        let t = msg.track_num;
        // The active keyboard track is silenced by the sequence (live input drives it instead).
        if self.active_keyboard_track_num == Some(t) && !msg.from_keyboard {
            return;
        }

        let midi_note = msg.param0;
        let velocity = msg.param1;
        let duration = msg.param2 as u32;
        let program = self.sequence.tracks[t].program;

        // Resolve the region + sample.
        let Some(instrument) = self.instrument_bank.instruments.get(program) else {
            return;
        };
        let index = instrument.resolve_entry_index(midi_note as u8);
        let Some(region) = instrument.regions.get(index) else {
            return;
        };
        let is_psg_pulse = instrument.instrument_type() == Some(InstrumentType::PsgPulse);
        let archive_index = region.swar_info_id as usize;
        let sample_id = region.swav_info_id as usize;

        // Pick the sample and the pitch it represents.
        let (sample, sample_pitch_hz) = if is_psg_pulse {
            match self.squares.get(sample_id) {
                Some(s) => (s.clone(), s.frequency),
                None => return,
            }
        } else {
            let Some(Some(archive)) = self.decoded_sample_archives.get(archive_index) else {
                return;
            };
            let Some(sample) = archive.get(sample_id) else {
                return;
            };
            (
                sample.clone(),
                midi_note_to_hz(f64::from(region.note_number), config.tuning),
            )
        };

        let initial_volume = if region.attack_coefficient == 0 {
            calc_channel_volume(velocity, 0, self.track_volume_db(t))
        } else {
            0.0
        };

        let voice = self.next_voice;
        self.next_voice += 1;

        events.push(SynthEvent::NoteStarted {
            track: t,
            voice,
            key: midi_note as u8,
            keyboard: msg.from_keyboard,
            sample,
            pitch: VoicePitch::Midi {
                note: f64::from(midi_note),
                sample_pitch_hz,
            },
            volume: initial_volume,
            duration_ticks: (!msg.from_keyboard).then_some(duration),
        });

        self.active_notes.push(ActiveNote {
            track_num: t,
            midi_note: midi_note as u8,
            velocity,
            voice,
            end_time: self.sequence.ticks_elapsed + duration,
            adsr_state: AdsrState::Attack,
            adsr_timer: -92544,
            from_keyboard: msg.from_keyboard,
            lfo_counter: 0,
            delay_counter: 0,
            lfo_vol_db: 0,
            attack_coefficient: region.attack_coefficient,
            decay_coefficient: region.decay_coefficient,
            sustain_level: region.sustain_level,
            release_coefficient: region.release_coefficient,
        });
    }

    /// Triggers a note from live keyboard input on `track`. The note sounds until released.
    pub fn keyboard_note_on(
        &mut self,
        track: usize,
        note: u8,
        velocity: i32,
        duration: u32,
        config: &SynthConfig,
        events: &mut Vec<SynthEvent>,
    ) {
        if track >= TRACK_COUNT {
            return;
        }
        let msg = Message {
            from_keyboard: true,
            track_num: track,
            msg_type: MessageType::PlayNote,
            param0: i32::from(note),
            param1: velocity,
            param2: duration as i32,
            timestamp: self.sequence.ticks_elapsed,
        };
        self.play_note(msg, config, events);
    }

    /// Releases a previously-triggered keyboard note, moving it into its ADSR release stage.
    pub fn keyboard_note_off(&mut self, track: usize, note: u8, events: &mut Vec<SynthEvent>) {
        if track >= TRACK_COUNT {
            return;
        }
        for entry in self.active_notes.iter_mut() {
            if entry.track_num == track && entry.midi_note == note {
                entry.adsr_state = AdsrState::Release;
            }
        }
        events.push(SynthEvent::NoteReleased {
            track,
            key: note,
            keyboard: true,
        });
    }

    /// Runs one ADSR/LFO update pass over the active notes (mirrors the original engine exactly,
    /// including that at most one finished note is removed per tick).
    fn process_active_notes(
        &mut self,
        feedback: &TickFeedback,
        config: &SynthConfig,
        events: &mut Vec<SynthEvent>,
    ) {
        let mut index_to_delete: Option<usize> = None;
        let ticks = self.sequence.ticks_elapsed;

        for index in 0..self.active_notes.len() {
            let mut entry = self.active_notes[index];
            let t = entry.track_num;

            if feedback.is_ended(t, entry.voice) {
                // The synthesizer stopped this voice on its own (round-robin steal, or a
                // one-shot sample that ran out). Drop the bookkeeping.
                index_to_delete = Some(index);
                self.active_notes[index] = entry;
                continue;
            }

            // Begin release once the note's scheduled duration elapses.
            if ticks >= entry.end_time
                && !entry.from_keyboard
                && entry.adsr_state != AdsrState::Release
            {
                entry.adsr_state = AdsrState::Release;
                events.push(SynthEvent::NoteReleased {
                    track: t,
                    key: entry.midi_note,
                    keyboard: false,
                });
            }

            self.apply_lfo(&mut entry, config, events);
            self.apply_adsr(&mut entry, index, &mut index_to_delete, events);

            self.active_notes[index] = entry;
        }

        if let Some(i) = index_to_delete {
            self.active_notes.remove(i);
        }
    }

    /// The track-level decibel attenuation pokediamond folds into `chn->userDecay`:
    /// `DecibelSquareTable[volume] + DecibelSquareTable[expression] + DecibelSquareTable[master]`
    /// (`TrackUpdateChannel`).
    fn track_volume_db(&self, t: usize) -> i32 {
        let tr = &self.sequence.tracks[t];
        decibel_db(tr.volume) + decibel_db(tr.expression) + decibel_db(tr.master_volume)
    }

    /// LFO update for one note, ported faithfully (including the DS fixed-point math).
    fn apply_lfo(
        &mut self,
        entry: &mut ActiveNote,
        _config: &SynthConfig,
        events: &mut Vec<SynthEvent>,
    ) {
        let track = &self.sequence.tracks[entry.track_num];
        let params = LfoParams {
            depth: track.lfo_depth,
            delay: track.lfo_delay,
            lfo_type: track.lfo_type,
            speed: track.lfo_speed,
            range: track.lfo_range,
        };

        // Whether the LFO delay has elapsed (the phase advances this tick). pokediamond gates both
        // the value and the phase on a single `delayCounter`; see [`lfo_tick`].
        let delay_elapsed = entry.delay_counter >= params.delay;
        let lfo_value = lfo_tick(&params, &mut entry.lfo_counter, &mut entry.delay_counter);

        // pokediamond adds the LFO into whichever target it modulates. Pitch modulation is applied
        // only on ticks where the phase advances (delay elapsed); the value is in 1/64ths of a
        // semitone. Volume modulation is summed (in dB) into the channel volume by `apply_adsr`
        // via `lfo_vol_db`. (Pan LFO is not represented — pan is a per-track stereo stage here.)
        entry.lfo_vol_db = if params.lfo_type == lfo_type::VOLUME {
            lfo_value as i32
        } else {
            0
        };
        if delay_elapsed && lfo_value != 0 && params.lfo_type == lfo_type::PITCH {
            events.push(SynthEvent::VoiceDetune {
                track: entry.track_num,
                voice: entry.voice,
                semitones: lfo_value as f64 / 64.0,
            });
        }
    }

    /// ADSR envelope advance for one note.
    fn apply_adsr(
        &mut self,
        entry: &mut ActiveNote,
        index: usize,
        index_to_delete: &mut Option<usize>,
        events: &mut Vec<SynthEvent>,
    ) {
        let t = entry.track_num;
        // pokediamond sums every contribution in the decibel domain before one conversion
        // (`SND_ExChannelMain`): velocity + envelope + userDecay (track volume + expression +
        // master) + volume-LFO. We fold the non-envelope terms into `extra_db` here.
        let extra_db = self.track_volume_db(t) + entry.lfo_vol_db;
        let set_volume = |entry: &ActiveNote, events: &mut Vec<SynthEvent>| {
            events.push(SynthEvent::VoiceVolume {
                track: t,
                voice: entry.voice,
                volume: calc_channel_volume(entry.velocity, entry.adsr_timer, extra_db),
            });
        };
        match entry.adsr_state {
            AdsrState::Attack => {
                entry.adsr_timer =
                    -(((-entry.attack_coefficient).wrapping_mul(entry.adsr_timer)) >> 8);
                set_volume(entry, events);
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
                set_volume(entry, events);
            }
            AdsrState::Sustain => set_volume(entry, events),
            AdsrState::Release => {
                // pokediamond cuts a channel only once its release envelope reaches the floor
                // (`SND_ChannelMain`: `envStatus == RELEASE && vol <= -723`, i.e. attenuation
                // <= -92544), uniformly for every channel type.
                if entry.adsr_timer <= -92544 {
                    events.push(SynthEvent::VoiceStopped {
                        track: t,
                        voice: entry.voice,
                    });
                    *index_to_delete = Some(index);
                } else {
                    entry.adsr_timer -= entry.release_coefficient;
                    set_volume(entry, events);
                }
            }
        }
    }
}

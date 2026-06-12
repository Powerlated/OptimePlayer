//! Applying sequence [`Message`]s and starting notes (including live keyboard input).

use super::volume::calc_channel_volume;
use super::{ActiveNote, AdsrState, Controller, SynthConfig};
use crate::bank::InstrumentType;
use crate::sequence::{Message, MessageType};
use crate::tuning::midi_note_to_hz;
use crate::TRACK_COUNT;

impl Controller {
    /// Applies one sequence message.
    pub(super) fn handle_message(&mut self, msg: Message, config: &SynthConfig) {
        match msg.msg_type {
            MessageType::PlayNote => self.play_note(msg, config),
            MessageType::Jump => self.jumps += 1,
            MessageType::TrackEnded => {
                let any_active = self.sequence.tracks.iter().any(|tr| tr.active);
                if !any_active {
                    self.fading_start = true;
                }
            }
            MessageType::VolumeChange => {
                // No-op: track volume (with expression and master) is now summed in the decibel
                // domain per tick by `track_volume_db`, matching pokediamond, rather than applied
                // as a separate linear mixer gain. The synth's `volume` field stays at unity.
            }
            MessageType::PanChange => {
                self.synthesizers[msg.track_num].set_pan(msg.param0 as f64 / 128.0, config);
            }
            MessageType::PitchBend => {
                let track = &self.sequence.tracks[msg.track_num];
                // `pitch_bend` is already a signed byte (pokediamond `par._s8`, set by `0xC4`);
                // scale by half the bend range, in 1/64 semitones.
                let pitch_bend = track.pitch_bend;
                let finetune = (pitch_bend as f64) * (track.pitch_bend_range as f64 / 2.0) / 64.0;
                self.synthesizers[msg.track_num].set_finetune(finetune, config.tuning);
            }
            MessageType::InstrumentChange => {}
        }
    }

    /// Starts a note from a [`MessageType::PlayNote`] message.
    fn play_note(&mut self, msg: Message, config: &SynthConfig) {
        let t = msg.track_num;
        // The active keyboard track is silenced by the sequence (live input drives it instead).
        if self.active_keyboard_track_num == Some(t) && !msg.from_keyboard {
            return;
        }

        let midi_note = msg.param0;
        let velocity = msg.param1;
        let duration = msg.param2 as u32;
        let program = self.sequence.tracks[t].program;

        // Resolve the region + sample, copying out everything we need before we mutate `self`.
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
        let note_number = region.note_number;
        let attack_coefficient = region.attack_coefficient;
        let decay_coefficient = region.decay_coefficient;
        let sustain_level = region.sustain_level;
        let release_coefficient = region.release_coefficient;

        // Pick the sample and the pitch it represents.
        let (sample, sample_frequency) = if is_psg_pulse {
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
                midi_note_to_hz(f64::from(note_number), config.tuning),
            )
        };

        let initial_volume = if attack_coefficient == 0 {
            calc_channel_volume(velocity, 0, self.track_volume_db(t))
        } else {
            0.0
        };

        let synth_instr_index = self.synthesizers[t].play(
            sample,
            f64::from(midi_note),
            sample_frequency,
            initial_volume,
            self.sequence.ticks_elapsed,
            config.tuning,
        );

        self.notes_on[t][midi_note as usize] = 1;
        self.active_notes.push(ActiveNote {
            track_num: t,
            midi_note: midi_note as u8,
            velocity,
            synth_instr_index,
            start_time: self.sequence.ticks_elapsed,
            end_time: self.sequence.ticks_elapsed + duration,
            adsr_state: AdsrState::Attack,
            adsr_timer: -92544,
            from_keyboard: msg.from_keyboard,
            lfo_counter: 0,
            delay_counter: 0,
            lfo_vol_db: 0,
            attack_coefficient,
            decay_coefficient,
            sustain_level,
            release_coefficient,
        });
    }

    /// Triggers a note from live keyboard input on `track` (mirrors the legacy app's
    /// `sendMessage(true, PlayNote, ...)` path). The note sounds until released.
    pub fn play_keyboard_note(
        &mut self,
        track: usize,
        note: u8,
        velocity: i32,
        duration: u32,
        config: &SynthConfig,
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
        self.play_note(msg, config);
        self.notes_on_keyboard[track][note as usize] = 1;
    }

    /// Releases a previously-triggered keyboard note, moving it into its ADSR release stage.
    pub fn release_keyboard_note(&mut self, track: usize, note: u8) {
        if track >= TRACK_COUNT {
            return;
        }
        for entry in self.active_notes.iter_mut() {
            if entry.track_num == track && entry.midi_note == note {
                entry.adsr_state = AdsrState::Release;
            }
        }
        self.notes_on_keyboard[track][note as usize] = 0;
    }
}

//! The look-ahead [`FsVisController`]: a parallel sequence runner that drives visualizers
//! without producing audio.

use std::sync::Arc;

use crate::sdat::Sdat;
use crate::sequence::{Message, MessageType, Sequence};
use crate::util::{read_u32, CircularBuffer};
use crate::TRACK_COUNT;

/// A pitch-bend change observed by the look-ahead, in resolved semitones at a given tick.
#[derive(Debug, Clone, Copy)]
pub struct PitchBendEvent {
    /// Tick at which the bend took effect.
    pub timestamp: u32,
    /// Track the bend applies to.
    pub track: usize,
    /// Bend amount in semitones (`pitch_bend * range/2 / 64`).
    pub semitones: f32,
}

/// A parallel sequence runner used to drive look-ahead visualizers without producing audio.
///
/// It runs the same SSEQ as [`Controller`](super::Controller) but only tracks which notes are on,
/// and is advanced `run_ahead_ticks` ahead at construction.
pub struct FsVisController {
    /// The look-ahead sequence.
    pub sequence: Sequence,
    /// Recently triggered notes, newest last (capacity-bounded).
    pub active_notes: CircularBuffer<Message>,
    /// Recently observed pitch-bend changes, newest last (capacity-bounded).
    pub pitch_bends: CircularBuffer<PitchBendEvent>,
    bpm_timer: u32,
}

impl FsVisController {
    /// Builds a look-ahead controller for `sseq_id`, advanced `run_ahead_ticks` ticks.
    pub fn new(sdat: &Sdat, sseq_id: u32, run_ahead_ticks: u32) -> Option<FsVisController> {
        let info = sdat.sseq_infos.get(sseq_id as usize)?.clone()?;
        let file = sdat.file(info.file_id)?;
        let arc: Arc<[u8]> = Arc::from(file.to_vec());
        let data_offset = read_u32(&arc, 0x18);

        let mut ctrl = FsVisController {
            sequence: Sequence::new(arc, data_offset, 512),
            active_notes: CircularBuffer::new(2048),
            pitch_bends: CircularBuffer::new(2048),
            bpm_timer: 0,
        };
        for _ in 0..run_ahead_ticks {
            ctrl.tick();
        }
        Some(ctrl)
    }

    /// Advances the look-ahead sequence by one tick, recording note-on events.
    pub fn tick(&mut self) {
        self.bpm_timer += self.sequence.tracks[0].bpm;
        while self.bpm_timer >= 240 {
            self.bpm_timer -= 240;
            // The look-ahead visualizer has no channel state; pass all-false so zero-duration
            // notes advance immediately rather than stalling.
            self.sequence.tick(&[false; TRACK_COUNT]);

            while let Some(mut msg) = self.sequence.message_buffer.pop() {
                match msg.msg_type {
                    MessageType::PlayNote => {
                        if self.active_notes.is_full() {
                            self.active_notes.pop();
                        }
                        msg.timestamp = self.sequence.ticks_elapsed;
                        self.active_notes.insert(msg);
                    }
                    MessageType::PitchBend => {
                        // Resolve the current bend in semitones from the track's live state,
                        // matching the audio controller's `set_finetune` math.
                        let tr = &self.sequence.tracks[msg.track_num];
                        let semitones =
                            tr.pitch_bend as f32 * (tr.pitch_bend_range as f32 / 2.0) / 64.0;
                        if self.pitch_bends.is_full() {
                            self.pitch_bends.pop();
                        }
                        self.pitch_bends.insert(PitchBendEvent {
                            timestamp: self.sequence.ticks_elapsed,
                            track: msg.track_num,
                            semitones,
                        });
                    }
                    _ => {}
                }
            }
        }
    }
}

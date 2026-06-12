//! The look-ahead [`FsVisController`]: a parallel sequencer runner (no audio) that feeds
//! upcoming notes to visualizers, for any device.

use std::sync::Arc;

use crate::devices::gba::sequencer::{Mp2kOp, Mp2kSequencer};
use crate::devices::nintendo_ds::sequence::{MessageType, Sequence};
use crate::devices::SoundData;
use crate::util::{read_u32, CircularBuffer};
use crate::TRACK_COUNT;

/// A note observed by the look-ahead, on the sequencer-step timeline.
#[derive(Debug, Clone, Copy)]
pub struct VisNote {
    /// Track the note plays on.
    pub track: usize,
    /// MIDI key.
    pub key: u8,
    /// Velocity (0..=127).
    pub velocity: i32,
    /// Length in sequencer steps (0 = unknown / until released).
    pub duration: u32,
    /// Step at which the note starts.
    pub timestamp: u32,
}

/// The device-specific sequencer being run ahead.
enum Lookahead {
    /// DS: the bare SSEQ interpreter (no sample decoding needed).
    NintendoDs { sequence: Sequence, bpm_timer: u32 },
    /// GBA: the bare MP2K sequencer (channel ops other than notes are ignored).
    Gba {
        sequencer: Mp2kSequencer,
        ops: Vec<Mp2kOp>,
    },
}

/// A parallel sequencer runner used to drive look-ahead visualizers without producing audio.
pub struct FsVisController {
    inner: Lookahead,
    /// Recently triggered notes, newest last (capacity-bounded).
    pub notes: CircularBuffer<VisNote>,
}

impl FsVisController {
    /// Builds a look-ahead runner for song `song_id` of `data`.
    pub fn new(data: &SoundData, song_id: u32) -> Option<FsVisController> {
        let inner = match data {
            SoundData::NintendoDs(sdat) => {
                let info = sdat.sseq_infos.get(song_id as usize)?.clone()?;
                let file = sdat.file(info.file_id)?;
                let arc: Arc<[u8]> = Arc::from(file.to_vec());
                let data_offset = read_u32(&arc, 0x18);
                Lookahead::NintendoDs {
                    sequence: Sequence::new(arc, data_offset, 512),
                    bpm_timer: 0,
                }
            }
            SoundData::Gba(rom) => {
                let header = rom.song_header(song_id)?;
                Lookahead::Gba {
                    sequencer: Mp2kSequencer::new(rom.data.clone(), &header),
                    ops: Vec::new(),
                }
            }
        };
        Some(FsVisController {
            inner,
            notes: CircularBuffer::new(2048),
        })
    }

    /// Sequencer steps executed so far (matches the audio controller's `steps_elapsed`).
    pub fn steps_elapsed(&self) -> u32 {
        match &self.inner {
            Lookahead::NintendoDs { sequence, .. } => sequence.ticks_elapsed,
            Lookahead::Gba { sequencer, .. } => sequencer.steps,
        }
    }

    fn push_note(notes: &mut CircularBuffer<VisNote>, note: VisNote) {
        if notes.is_full() {
            notes.pop();
        }
        notes.insert(note);
    }

    /// Advances the look-ahead by one device tick, recording note-on events.
    pub fn tick(&mut self) {
        match &mut self.inner {
            Lookahead::NintendoDs {
                sequence,
                bpm_timer,
            } => {
                *bpm_timer += sequence.tracks[0].bpm;
                while *bpm_timer >= 240 {
                    *bpm_timer -= 240;
                    // The look-ahead has no channel state; pass all-false so zero-duration
                    // notes advance immediately rather than stalling.
                    sequence.tick(&[false; TRACK_COUNT]);

                    while let Some(msg) = sequence.message_buffer.pop() {
                        if msg.msg_type == MessageType::PlayNote {
                            Self::push_note(
                                &mut self.notes,
                                VisNote {
                                    track: msg.track_num,
                                    key: msg.param0 as u8,
                                    velocity: msg.param1,
                                    duration: msg.param2.max(0) as u32,
                                    timestamp: sequence.ticks_elapsed,
                                },
                            );
                        }
                    }
                }
            }
            Lookahead::Gba { sequencer, ops } => {
                ops.clear();
                sequencer.tick_frame(ops);
                for op in ops.drain(..) {
                    if let Mp2kOp::Note { track, note } = op {
                        Self::push_note(
                            &mut self.notes,
                            VisNote {
                                track,
                                key: note.midi_key,
                                velocity: i32::from(note.velocity),
                                duration: u32::from(note.gate),
                                timestamp: sequencer.steps,
                            },
                        );
                    }
                }
            }
        }
    }
}

//! The look-ahead [`FsVisController`]: a parallel sequencer runner (no audio) that feeds
//! upcoming notes to visualizers, for any device.

use std::sync::Arc;

use crate::devices::dse::{DseSequencer, SeqOp};
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
    /// Length in sequencer steps (0 = unknown / still held — a GBA tie awaiting its `EndTie`).
    pub duration: u32,
    /// Step at which the note starts.
    pub timestamp: u32,
}

/// A raw timeline event produced by stepping a device sequencer, before the note-length and
/// loop bookkeeping that [`FsVisController::tick`] (live) and [`FsVisController::overview`]
/// (whole-track) each apply in their own way.
enum VisEvent {
    /// A note started (`duration` = gate; 0 means a tie, resolved later by `EndTie`/`TrackEnded`).
    Note(VisNote),
    /// A GBA tie on `track` playing `key` was released — fixes that note's real length.
    EndTie { track: usize, key: u8 },
    /// A track ended (GBA `FINE`): releases any of its still-held ties.
    TrackEnded { track: usize },
    /// The song reached its loop point.
    Looped,
    /// The song ended (all tracks finished).
    Ended,
}

/// An open GBA tie awaiting its `EndTie`: the note's [`CircularBuffer`] serial (live) or its
/// index into the overview note list, plus the key it matches.
#[derive(Clone, Copy)]
struct OpenTie {
    track: usize,
    key: u8,
    /// `CircularBuffer` serial of the note (used by the live ring buffer).
    handle: u64,
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
    /// DSE: the bare SMDL sequencer. Notes carry their own concrete length (no ties).
    Dse {
        sequencer: DseSequencer,
        ops: Vec<SeqOp>,
    },
}

/// A parallel sequencer runner used to drive look-ahead visualizers without producing audio.
pub struct FsVisController {
    inner: Lookahead,
    /// Recently triggered notes, newest last (capacity-bounded).
    pub notes: CircularBuffer<VisNote>,
    /// Ties opened but not yet released, so their note bars get a real length once they end.
    open_ties: Vec<OpenTie>,
    /// Reused per-tick event scratch.
    scratch: Vec<VisEvent>,
}

/// The complete note timeline of a song, rendered once at load for the overview thumbnail.
pub struct SongOverview {
    /// Every note over one play-through (intro + first loop body, or to the end).
    pub notes: Vec<VisNote>,
    /// Total length of the timeline in sequencer steps.
    pub total_steps: u32,
    /// Tempo (musical BPM) over the timeline as `(step, bpm)` change points, in step order. The
    /// first entry is the starting tempo at step 0.
    pub tempos: Vec<(u32, f64)>,
}

impl FsVisController {
    /// Builds a look-ahead runner for song `song_id` of `data`.
    pub fn new(data: &SoundData, song_id: u32) -> Option<FsVisController> {
        let inner = build_lookahead(data, song_id)?;
        Some(FsVisController {
            inner,
            notes: CircularBuffer::new(2048),
            open_ties: Vec::new(),
            scratch: Vec::new(),
        })
    }

    /// Sequencer steps executed so far (matches the audio controller's `steps_elapsed`).
    pub fn steps_elapsed(&self) -> u32 {
        self.inner.steps_elapsed()
    }

    /// The current musical tempo in quarter-note BPM (DS: 48 steps/beat; GBA: 24).
    pub fn current_bpm(&self) -> f64 {
        self.inner.current_bpm()
    }

    fn push_note(notes: &mut CircularBuffer<VisNote>, note: VisNote) -> u64 {
        if notes.is_full() {
            notes.pop();
        }
        notes.insert(note);
        notes.last_serial().unwrap_or(0)
    }

    /// Advances the look-ahead by one device tick, recording note-on events and resolving the
    /// real length of any GBA ties that end this tick.
    pub fn tick(&mut self) {
        let mut events = std::mem::take(&mut self.scratch);
        events.clear();
        self.inner.step(&mut events);
        let now = self.steps_elapsed();
        for ev in events.drain(..) {
            match ev {
                VisEvent::Note(note) => {
                    let tie = note.duration == 0;
                    let track = note.track;
                    let key = note.key;
                    let handle = Self::push_note(&mut self.notes, note);
                    if tie {
                        // Bound the open-tie list: a tie whose note has already scrolled out of
                        // the ring buffer can never be resolved, so drop the oldest if it piles up.
                        if self.open_ties.len() >= self.notes.capacity() {
                            self.open_ties.remove(0);
                        }
                        self.open_ties.push(OpenTie { track, key, handle });
                    }
                }
                VisEvent::EndTie { track, key } => self.close_tie(track, key, now),
                VisEvent::TrackEnded { track } => self.close_track_ties(track, now),
                VisEvent::Looped | VisEvent::Ended => {}
            }
        }
        self.scratch = events;
    }

    /// Closes the oldest open tie on `track` matching `key`, setting its note's real length.
    fn close_tie(&mut self, track: usize, key: u8, now: u32) {
        if let Some(i) = self
            .open_ties
            .iter()
            .position(|t| t.track == track && t.key == key)
        {
            let tie = self.open_ties.swap_remove(i);
            if let Some(note) = self.notes.peek_mut_serial(tie.handle) {
                note.duration = now.saturating_sub(note.timestamp);
            }
        }
    }

    /// Closes every open tie on a track that ended.
    fn close_track_ties(&mut self, track: usize, now: u32) {
        let mut i = 0;
        while i < self.open_ties.len() {
            if self.open_ties[i].track == track {
                let tie = self.open_ties.swap_remove(i);
                if let Some(note) = self.notes.peek_mut_serial(tie.handle) {
                    note.duration = now.saturating_sub(note.timestamp);
                }
            } else {
                i += 1;
            }
        }
    }

    /// Renders the whole-track overview for song `song_id` of `data`: runs a fresh sequencer to
    /// the song's loop point or end (capped), collecting every note with its real length and the
    /// tempo changes along the way. Used to build the piano roll's pre-rendered overview bar.
    pub fn overview(data: &SoundData, song_id: u32) -> Option<SongOverview> {
        /// Safety cap so a song that neither loops nor ends can't spin forever.
        const MAX_STEPS: u32 = 200_000;

        let mut inner = build_lookahead(data, song_id)?;
        let mut notes: Vec<VisNote> = Vec::new();
        let mut open: Vec<OpenTie> = Vec::new(); // `handle` = index into `notes`
        let mut tempos: Vec<(u32, f64)> = Vec::new();
        let mut events: Vec<VisEvent> = Vec::new();
        let mut last_bpm = f64::NAN;

        let resolve = |open: &mut Vec<OpenTie>, notes: &mut [VisNote], idx: usize, now: u32| {
            let tie = open.swap_remove(idx);
            if let Some(note) = notes.get_mut(tie.handle as usize) {
                note.duration = now.saturating_sub(note.timestamp);
            }
        };

        loop {
            let step = inner.steps_elapsed();
            let bpm = inner.current_bpm();
            if (bpm - last_bpm).abs() > f64::EPSILON {
                tempos.push((step, bpm));
                last_bpm = bpm;
            }

            events.clear();
            inner.step(&mut events);
            let now = inner.steps_elapsed();
            let mut stop = false;
            for ev in events.drain(..) {
                match ev {
                    VisEvent::Note(note) => {
                        let tie = note.duration == 0;
                        let (track, key) = (note.track, note.key);
                        notes.push(note);
                        if tie {
                            open.push(OpenTie {
                                track,
                                key,
                                handle: (notes.len() - 1) as u64,
                            });
                        }
                    }
                    VisEvent::EndTie { track, key } => {
                        if let Some(i) = open.iter().position(|t| t.track == track && t.key == key)
                        {
                            resolve(&mut open, &mut notes, i, now);
                        }
                    }
                    VisEvent::TrackEnded { track } => {
                        while let Some(i) = open.iter().position(|t| t.track == track) {
                            resolve(&mut open, &mut notes, i, now);
                        }
                    }
                    VisEvent::Looped | VisEvent::Ended => stop = true,
                }
            }
            if stop || now >= MAX_STEPS {
                break;
            }
        }

        let total_steps = inner.steps_elapsed().max(1);
        // Any tie still held at the loop/end point runs to the boundary.
        for tie in &open {
            if let Some(note) = notes.get_mut(tie.handle as usize) {
                note.duration = total_steps.saturating_sub(note.timestamp);
            }
        }
        if tempos.is_empty() {
            tempos.push((0, inner.current_bpm()));
        }
        Some(SongOverview {
            notes,
            total_steps,
            tempos,
        })
    }
}

impl Lookahead {
    fn steps_elapsed(&self) -> u32 {
        match self {
            Lookahead::NintendoDs { sequence, .. } => sequence.ticks_elapsed,
            Lookahead::Gba { sequencer, .. } => sequencer.steps,
            Lookahead::Dse { sequencer, .. } => sequencer.ticks_elapsed,
        }
    }

    fn current_bpm(&self) -> f64 {
        match self {
            Lookahead::NintendoDs { sequence, .. } => f64::from(sequence.tracks[0].bpm),
            Lookahead::Gba { sequencer, .. } => f64::from(sequencer.tempo_i()),
            // Quarter-note BPM: the SMDL tempo is already musical when TPQN is the 48-step beat.
            Lookahead::Dse { sequencer, .. } => {
                f64::from(sequencer.bpm) * f64::from(sequencer.tpqn) / 48.0
            }
        }
    }

    /// Advances exactly one device tick, appending the raw timeline events it produced.
    fn step(&mut self, out: &mut Vec<VisEvent>) {
        match self {
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
                        match msg.msg_type {
                            MessageType::PlayNote {
                                note,
                                velocity,
                                duration,
                            } => out.push(VisEvent::Note(VisNote {
                                track: msg.track_num,
                                key: note as u8,
                                velocity,
                                duration: duration.max(0) as u32,
                                timestamp: sequence.ticks_elapsed,
                            })),
                            MessageType::Jump => out.push(VisEvent::Looped),
                            MessageType::TrackEnded
                                if !sequence.tracks.iter().any(|t| t.active) =>
                            {
                                out.push(VisEvent::Ended)
                            }
                            _ => {}
                        }
                    }
                }
            }
            Lookahead::Gba { sequencer, ops } => {
                ops.clear();
                sequencer.tick_frame(ops);
                for op in ops.drain(..) {
                    match op {
                        Mp2kOp::Note { track, note } => out.push(VisEvent::Note(VisNote {
                            track,
                            key: note.midi_key,
                            velocity: i32::from(note.velocity),
                            // gate 0 = tie: real length comes from the matching EndTie.
                            duration: u32::from(note.gate),
                            timestamp: sequencer.steps,
                        })),
                        Mp2kOp::EndTie { track, key } => out.push(VisEvent::EndTie { track, key }),
                        Mp2kOp::TrackEnded { track } => out.push(VisEvent::TrackEnded { track }),
                        Mp2kOp::Looped => out.push(VisEvent::Looped),
                        Mp2kOp::Finished => out.push(VisEvent::Ended),
                        Mp2kOp::GateTick { .. } => {}
                    }
                }
            }
            Lookahead::Dse { sequencer, ops } => {
                ops.clear();
                sequencer.seq_tick(ops);
                let now = sequencer.ticks_elapsed;
                for op in ops.drain(..) {
                    match op {
                        // DSE notes are fire-and-forget with a concrete length (never a tie), so
                        // clamp to >=1 to keep them off the GBA tie path.
                        SeqOp::NoteOn {
                            track,
                            key,
                            velocity,
                            duration,
                        } => out.push(VisEvent::Note(VisNote {
                            track,
                            key,
                            velocity: i32::from(velocity),
                            duration: duration.max(1),
                            timestamp: now,
                        })),
                        SeqOp::Looped => out.push(VisEvent::Looped),
                        _ => {}
                    }
                }
                if sequencer.ended {
                    out.push(VisEvent::Ended);
                }
            }
        }
    }
}

/// Builds the device-specific look-ahead sequencer for song `song_id`.
fn build_lookahead(data: &SoundData, song_id: u32) -> Option<Lookahead> {
    match data {
        SoundData::NintendoDs(sdat) => {
            let info = sdat.sseq_infos.get(song_id as usize)?.clone()?;
            let file = sdat.file(info.file_id)?;
            let arc: Arc<[u8]> = Arc::from(file.to_vec());
            let data_offset = read_u32(&arc, 0x18);
            Some(Lookahead::NintendoDs {
                sequence: Sequence::new(arc, data_offset, 512),
                bpm_timer: 0,
            })
        }
        SoundData::Gba(rom) => {
            let header = rom.song_header(song_id)?;
            Some(Lookahead::Gba {
                sequencer: Mp2kSequencer::new(rom.data.clone(), &header),
                ops: Vec::new(),
            })
        }
        SoundData::Dse(dse) => Some(Lookahead::Dse {
            sequencer: dse.make_sequencer(song_id)?,
            ops: Vec::new(),
        }),
    }
}

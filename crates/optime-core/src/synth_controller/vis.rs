//! The look-ahead [`FsVisController`]: a parallel player runner (no audio) that feeds upcoming
//! notes to visualizers, for any device.
//!
//! The look-ahead simply runs a second [`DevicePlayer`](crate::devices::DevicePlayer) headlessly
//! — ticking it with a neutral config and an empty [`TickFeedback`] — and extracts note events
//! from the standard [`SynthEvent`] stream. This means the visualizer sees the same notes the
//! audio player produces, with no device-specific logic here.

use crate::PerDeviceSettings;
use crate::devices::{DevicePlayer, SoundData, SynthEvent, TickFeedback};
use crate::util::CircularBuffer;

/// A note observed by the look-ahead, on the sequencer-step timeline.
#[derive(Debug, Clone, Copy)]
pub struct VisNote {
    /// Track the note plays on.
    pub track: usize,
    /// MIDI key.
    pub key: u8,
    /// Length in sequencer steps (0 = still held / tie — a note awaiting its release event).
    pub duration: u32,
    /// Step at which the note starts.
    pub timestamp: u32,
}

/// An open note awaiting its `NoteReleased` event, so its bar gets a real length once it ends.
#[derive(Clone, Copy)]
struct OpenNote {
    track: usize,
    key: u8,
    /// `CircularBuffer` serial of the corresponding `VisNote` (live ring buffer).
    handle: u64,
}

/// A parallel player runner used to drive look-ahead visualizers without producing audio.
pub struct FsVisController {
    player: Box<dyn DevicePlayer>,
    /// Recently triggered notes, newest last (capacity-bounded).
    pub notes: CircularBuffer<VisNote>,
    /// Notes opened but not yet released, so their bars get a real length once they end.
    open_notes: Vec<OpenNote>,
    /// Reused per-tick event scratch (avoid allocations in the hot path).
    events: Vec<SynthEvent>,
    feedback: TickFeedback,
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
    /// The device's sequencer steps per quarter-note beat (DS SSEQ: 48, GBA MP2K: 24). Constant for
    /// a song — a tempo change moves the step *rate* (steps per second), not this — so the musical
    /// bar grid is uniform in steps and needs no tempo map. `0.0` if the device reports no beat
    /// division.
    pub steps_per_beat: f64,
}

impl FsVisController {
    /// Builds a look-ahead runner for song `song_id` of `data`.
    pub fn new(data: &dyn SoundData, song_id: u32) -> Option<FsVisController> {
        let player = data.make_player(song_id)?;
        Some(FsVisController {
            player,
            notes: CircularBuffer::new(2048),
            open_notes: Vec::new(),
            events: Vec::new(),
            feedback: TickFeedback::default(),
        })
    }

    /// Sequencer steps executed so far (matches the audio controller's `steps_elapsed`).
    pub fn steps_elapsed(&self) -> u32 {
        self.player.steps_elapsed()
    }

    /// The current musical tempo in quarter-note BPM.
    pub fn current_bpm(&self) -> f64 {
        player_bpm(&*self.player)
    }

    fn push_note(notes: &mut CircularBuffer<VisNote>, note: VisNote) -> u64 {
        if notes.is_full() {
            notes.pop();
        }
        notes.insert(note);
        notes.last_serial().unwrap_or(0)
    }

    /// Advances the look-ahead by one device tick, recording note-on events and resolving the
    /// real length of any notes that are released this tick.
    pub fn tick(&mut self) {
        let config = PerDeviceSettings::neutral();
        let mut events = std::mem::take(&mut self.events);
        events.clear();
        self.player.tick(&mut self.feedback, &config, &mut events);
        self.feedback.ended_voices.clear();
        let now = self.player.steps_elapsed();
        for ev in events.drain(..) {
            match ev {
                SynthEvent::NoteStarted {
                    track,
                    key,
                    duration_ticks,
                    ..
                } => {
                    let duration = duration_ticks.unwrap_or(0);
                    let is_tie = duration == 0;
                    let handle = Self::push_note(
                        &mut self.notes,
                        VisNote {
                            track,
                            key,
                            duration,
                            timestamp: now,
                        },
                    );
                    if is_tie {
                        // Bound the open-note list: a note that has already scrolled out of
                        // the ring buffer can never be resolved, so drop the oldest if needed.
                        if self.open_notes.len() >= self.notes.capacity() {
                            self.open_notes.remove(0);
                        }
                        self.open_notes.push(OpenNote { track, key, handle });
                    }
                }
                SynthEvent::NoteReleased { track, key } => self.close_note(track, key, now),
                _ => {}
            }
        }
        self.events = events;
    }

    /// Closes the oldest open note on `track` matching `key`, setting its real length.
    fn close_note(&mut self, track: usize, key: u8, now: u32) {
        if let Some(i) = self
            .open_notes
            .iter()
            .position(|n| n.track == track && n.key == key)
        {
            let note = self.open_notes.swap_remove(i);
            if let Some(vis) = self.notes.peek_mut_serial(note.handle) {
                vis.duration = now.saturating_sub(vis.timestamp);
            }
        }
    }

    /// Renders the whole-track overview for song `song_id` of `data`: runs a fresh player to
    /// the song's loop point or end (capped), collecting every note with its real length and the
    /// tempo changes along the way. Used to build the piano roll's pre-rendered overview bar.
    pub fn overview(data: &dyn SoundData, song_id: u32) -> Option<SongOverview> {
        /// Safety cap so a song that neither loops nor ends can't spin forever.
        const MAX_STEPS: u32 = 200_000;

        let mut player = data.make_player(song_id)?;
        let config = PerDeviceSettings::neutral();
        let mut feedback = TickFeedback::default();
        let mut events: Vec<SynthEvent> = Vec::new();
        let mut notes: Vec<VisNote> = Vec::new();
        // `handle` is the index into `notes` for open notes.
        let mut open: Vec<OpenNote> = Vec::new();
        let mut tempos: Vec<(u32, f64)> = Vec::new();
        let mut last_bpm = f64::NAN;

        let resolve = |open: &mut Vec<OpenNote>, notes: &mut [VisNote], idx: usize, now: u32| {
            let entry = open.swap_remove(idx);
            if let Some(note) = notes.get_mut(entry.handle as usize) {
                note.duration = now.saturating_sub(note.timestamp);
            }
        };

        loop {
            let step = player.steps_elapsed();
            let bpm = player_bpm(&*player);
            if (bpm - last_bpm).abs() > f64::EPSILON {
                tempos.push((step, bpm));
                last_bpm = bpm;
            }

            events.clear();
            player.tick(&mut feedback, &config, &mut events);
            feedback.ended_voices.clear();
            let now = player.steps_elapsed();
            let mut stop = false;
            for ev in events.drain(..) {
                match ev {
                    SynthEvent::NoteStarted {
                        track,
                        key,
                        duration_ticks,
                        ..
                    } => {
                        let duration = duration_ticks.unwrap_or(0);
                        let is_tie = duration == 0;
                        notes.push(VisNote {
                            track,
                            key,
                            duration,
                            timestamp: now,
                        });
                        if is_tie {
                            open.push(OpenNote {
                                track,
                                key,
                                handle: (notes.len() - 1) as u64,
                            });
                        }
                    }
                    SynthEvent::NoteReleased { track, key } => {
                        if let Some(i) = open.iter().position(|n| n.track == track && n.key == key)
                        {
                            resolve(&mut open, &mut notes, i, now);
                        }
                    }
                    SynthEvent::Looped | SynthEvent::Ended => stop = true,
                    _ => {}
                }
            }
            if stop || now >= MAX_STEPS {
                break;
            }
        }

        let total_steps = player.steps_elapsed().max(1);
        // Any note still held at the loop/end point runs to the boundary.
        for entry in &open {
            if let Some(note) = notes.get_mut(entry.handle as usize) {
                note.duration = total_steps.saturating_sub(note.timestamp);
            }
        }
        if tempos.is_empty() {
            tempos.push((0, player_bpm(&*player)));
        }
        Some(SongOverview {
            notes,
            total_steps,
            tempos,
            steps_per_beat: player.steps_per_beat(),
        })
    }
}

/// A player's current musical tempo in quarter-note BPM (`step_rate * 60 / steps_per_beat`),
/// or `0.0` if the device reports no beat division.
fn player_bpm(player: &dyn DevicePlayer) -> f64 {
    let spb = player.steps_per_beat();
    if spb > 0.0 {
        player.step_rate() * 60.0 / spb
    } else {
        0.0
    }
}

use crate::PerDeviceSettings;
use crate::devices::{DevicePlayer, SoundData, SynthEvent, TickFeedback};
use crate::util::CircularBuffer;

#[derive(Debug, Clone, Copy)]
pub struct VisNote {
    pub track: usize,
    pub key: u8,
    pub duration: u32,
    pub timestamp: u32,
}

#[derive(Clone, Copy)]
struct OpenNote {
    track: usize,
    key: u8,
    handle: u64,
}

pub struct FsVisController {
    player: Box<dyn DevicePlayer>,
    pub notes: CircularBuffer<VisNote>,
    open_notes: Vec<OpenNote>,
    events: Vec<SynthEvent>,
    feedback: TickFeedback,
}

pub struct SongOverview {
    pub notes: Vec<VisNote>,
    pub total_steps: u32,
    pub tempos: Vec<(u32, f64)>,
    pub steps_per_beat: f64,
}

impl FsVisController {
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

    pub fn steps_elapsed(&self) -> u32 {
        self.player.steps_elapsed()
    }

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

    pub fn overview(data: &dyn SoundData, song_id: u32) -> Option<SongOverview> {
        const MAX_STEPS: u32 = 200_000;

        let mut player = data.make_player(song_id)?;
        let config = PerDeviceSettings::neutral();
        let mut feedback = TickFeedback::default();
        let mut events: Vec<SynthEvent> = Vec::new();
        let mut notes: Vec<VisNote> = Vec::new();
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

fn player_bpm(player: &dyn DevicePlayer) -> f64 {
    let spb = player.steps_per_beat();
    if spb > 0.0 {
        player.step_rate() * 60.0 / spb
    } else {
        0.0
    }
}

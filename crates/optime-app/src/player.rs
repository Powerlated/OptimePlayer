//! The state shared between the UI and the audio thread, including the playlist the audio thread owns and offline WAV rendering.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use optime_core::{
    LoopAndTransitionOptions, PerDeviceSettings, PlaybackEvent, SoundData, StreamResampler,
    SynthController,
};

use crate::annotation::{Bounce, ChordVoicer};
use crate::persisted::{RepeatMode, TrackRef};

pub struct PlaylistEntry {
    pub archive: Option<Arc<dyn SoundData>>,
    pub track: TrackRef,
}

pub enum PlaybackCommand {
    SetPlaylist {
        entries: Vec<PlaylistEntry>,
        index: usize,
    },
    Reorder {
        entries: Vec<PlaylistEntry>,
        index: usize,
    },
    PlayAt(usize),
    Next,
    Prev,
}

#[derive(Debug, PartialEq, Eq)]
pub enum AutoAdvance {
    Repeat,
    Play(usize),
    Stop,
}

#[derive(Default)]
pub struct Playback {
    pub entries: Vec<PlaylistEntry>,
    pub index: usize,
    pub repeat: RepeatMode,
    pub commands: VecDeque<PlaybackCommand>,
    pub pending: Option<usize>,
    pub status_gen: u64,
    pub stopped: bool,
    pub needs_ui: bool,
}

impl Playback {
    pub fn manual_step(&self, delta: isize) -> Option<usize> {
        let n = self.entries.len();
        if n == 0 {
            return None;
        }
        let base = self.pending.unwrap_or(self.index) as isize;
        Some((base + delta).rem_euclid(n as isize) as usize)
    }

    pub fn auto_advance(&self) -> AutoAdvance {
        let n = self.entries.len();
        if n == 0 {
            return AutoAdvance::Stop;
        }
        match self.repeat {
            RepeatMode::One => AutoAdvance::Repeat,
            RepeatMode::All => AutoAdvance::Play((self.index + 1) % n),
            RepeatMode::Off => {
                if self.index + 1 >= n {
                    AutoAdvance::Stop
                } else {
                    AutoAdvance::Play(self.index + 1)
                }
            }
        }
    }
}

pub struct AudioState {
    pub controller: Option<SynthController>,
    pub config: PerDeviceSettings,
    pub paused: bool,
    pub volume: f32,
    pub volume_smooth: f32,
    pub fade_gain: f32,
    pub manual_fade_active: bool,
    pub pause_gain: f32,
    pub sample_rate: f64,
    pub dsp_load: f32,
    pub voices: usize,
    pub high_comp_gr_db: f32,
    pub last_callback: f64,
    pub playback: Playback,
    pub bounce: BounceTransport,
    pub resampler: StreamResampler,
}

impl AudioState {
    pub(crate) fn new() -> Self {
        Self {
            controller: None,
            config: PerDeviceSettings::neutral(),
            paused: false,
            volume: 1.0,
            volume_smooth: 1.0,
            fade_gain: 1.0,
            manual_fade_active: false,
            pause_gain: 1.0,
            sample_rate: 48_000.0,
            dsp_load: 0.0,
            voices: 0,
            high_comp_gr_db: 0.0,
            last_callback: f64::NEG_INFINITY,
            playback: Playback::default(),
            bounce: BounceTransport::default(),
            resampler: StreamResampler::new(),
        }
    }
}

#[derive(Default)]
pub struct BounceTransport {
    pub active: bool,
    pub buffer: Option<Arc<Bounce>>,
    pub pos: usize,
    pub loop_range: Option<(usize, usize)>,
    pub playing: bool,
    pub chords: Option<ChordVoicer>,
    pub chords_on: bool,
}

impl BounceTransport {
    #[inline]
    pub fn next_frame(&mut self) -> (f32, f32) {
        let Some(b) = &self.buffer else {
            return (0.0, 0.0);
        };
        if let Some((start, end)) = self.loop_range
            && end > start
            && (self.pos < start || self.pos >= end)
        {
            self.pos = start;
        }
        let out = b.frame(self.pos);
        if self.playing && self.pos < b.frames() {
            self.pos += 1;
        }
        out
    }
}

pub type Shared = Arc<Mutex<AudioState>>;

pub fn new_shared() -> Shared {
    Arc::new(Mutex::new(AudioState::new()))
}

pub const EXPORT_SAMPLE_RATE: u32 = 32768;

pub fn render_to_samples(
    data: &dyn SoundData,
    song_id: u32,
    config: &PerDeviceSettings,
) -> Vec<(f32, f32)> {
    let sr = EXPORT_SAMPLE_RATE as f64;

    let Some(mut controller) = SynthController::new(sr, data, song_id) else {
        return Vec::new();
    };
    controller.set_loop_and_transition(LoopAndTransitionOptions::export());

    let mut out = Vec::new();
    let mut sample: u64 = 0;
    let max_samples = (sr * 480.0) as u64;

    const CHUNK_FRAMES: usize = 512;
    let mut buf = vec![0.0f32; 2 * CHUNK_FRAMES];

    while sample < max_samples {
        let n = CHUNK_FRAMES.min((max_samples - sample) as usize);
        let chunk = &mut buf[..2 * n];
        controller.fill(chunk, config);

        for frame in chunk.chunks_exact(2) {
            out.push((frame[0] * 0.5, frame[1] * 0.5));
            sample += 1;
        }
        if controller
            .take_messages()
            .any(|m| m == PlaybackEvent::Finished)
        {
            break;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: u32) -> PlaylistEntry {
        PlaylistEntry {
            archive: None,
            track: TrackRef {
                source: "s".into(),
                song_id: id,
                label: String::new(),
            },
        }
    }

    fn playback(n: u32, index: usize, repeat: RepeatMode) -> Playback {
        Playback {
            entries: (0..n).map(entry).collect(),
            index,
            repeat,
            ..Default::default()
        }
    }

    #[test]
    fn manual_step_wraps_both_ways() {
        assert_eq!(playback(3, 0, RepeatMode::Off).manual_step(1), Some(1));
        assert_eq!(playback(3, 0, RepeatMode::Off).manual_step(-1), Some(2));
        assert_eq!(playback(3, 2, RepeatMode::Off).manual_step(1), Some(0));
        assert_eq!(playback(3, 2, RepeatMode::One).manual_step(1), Some(0));
        assert_eq!(playback(0, 0, RepeatMode::All).manual_step(1), None);
    }

    #[test]
    fn manual_step_advances_from_queued_target() {
        let mut p = playback(4, 0, RepeatMode::All);
        p.pending = Some(1);
        assert_eq!(p.manual_step(1), Some(2));
    }

    #[test]
    fn auto_advance_per_repeat_mode() {
        assert_eq!(
            playback(3, 1, RepeatMode::One).auto_advance(),
            AutoAdvance::Repeat
        );
        assert_eq!(
            playback(3, 1, RepeatMode::All).auto_advance(),
            AutoAdvance::Play(2)
        );
        assert_eq!(
            playback(3, 2, RepeatMode::All).auto_advance(),
            AutoAdvance::Play(0)
        );
        assert_eq!(
            playback(3, 1, RepeatMode::Off).auto_advance(),
            AutoAdvance::Play(2)
        );
        assert_eq!(
            playback(3, 2, RepeatMode::Off).auto_advance(),
            AutoAdvance::Stop
        );
        assert_eq!(
            playback(0, 0, RepeatMode::All).auto_advance(),
            AutoAdvance::Stop
        );
    }
}

//! Shared state between the egui UI thread and the cpal audio callback, plus the offline
//! renderer used for WAV export.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use optime_core::{
    LoopAndTransitionOptions, PerDeviceSettings, PlaybackEvent, SoundData, StreamResampler,
    SynthController,
};

use crate::persisted::{RepeatMode, TrackRef};

/// One entry of the audio-thread-owned playlist. The id to decode is `track.song_id`.
pub struct PlaylistEntry {
    /// The decoded-on-demand source archive. `None` means the source isn't loaded yet (a
    /// cross-source queue track): the audio thread can't decode it and asks the UI to fetch.
    pub archive: Option<Arc<dyn SoundData>>,
    /// Identity for UI reconcile (highlight / recents / media session / share URL) and the song id.
    pub track: TrackRef,
}

/// One-shot intents the UI pushes for the audio callback to apply (in order).
pub enum PlaybackCommand {
    /// Replace the whole list and start at `index`.
    SetPlaylist {
        entries: Vec<PlaylistEntry>,
        index: usize,
    },
    /// Replace the list and point `index` at the currently-playing entry's new position, *without*
    /// restarting playback (used to re-order for a shuffle toggle).
    Reorder {
        entries: Vec<PlaylistEntry>,
        index: usize,
    },
    /// Jump to `index` within the current list and play it.
    PlayAt(usize),
    /// Skip forward one entry (wraps; ignores repeat mode — like a real Next button).
    Next,
    /// Skip back one entry (wraps).
    Prev,
}

/// What an automatic (natural end-of-song) advance resolves to, per repeat mode.
#[derive(Debug, PartialEq, Eq)]
pub enum AutoAdvance {
    /// Replay the current entry (RepeatMode::One).
    Repeat,
    /// Play this index next.
    Play(usize),
    /// Nothing left to play (RepeatMode::Off, past the end).
    Stop,
}

/// Audio-thread-owned playback state. The UI *sends* into it (entries + commands + level
/// `repeat`) and *reads back* `index`/`status_gen` to reflect advances the callback performed
/// on its own (which keeps working while the UI's repaint loop is frozen — e.g. a hidden tab).
#[derive(Default)]
pub struct Playback {
    /// The ordered list. Already shuffled (materialized once at toggle time) when shuffle is on,
    /// so the audio thread never needs an RNG — it just walks the list.
    pub entries: Vec<PlaylistEntry>,
    /// Index of the currently-playing (or last-played) entry.
    pub index: usize,
    /// Repeat mode; set by the UI on toggle / SetPlaylist.
    pub repeat: RepeatMode,
    /// UI → audio one-shots, drained by the callback each buffer.
    pub commands: VecDeque<PlaybackCommand>,
    /// Index awaiting a decode (set by a command or a resolved auto-advance); the callback
    /// decodes it once the output has faded to silence.
    pub pending: Option<usize>,
    /// Bumped whenever the callback changes `index` / stops / needs the UI, so the UI knows to
    /// reconcile its own visuals.
    pub status_gen: u64,
    /// End-of-queue reached under RepeatMode::Off (UI shows "End of queue.").
    pub stopped: bool,
    /// An advance landed on a `None`-archive entry; the UI must load that source and resend.
    pub needs_ui: bool,
}

impl Playback {
    /// Target index for a manual Next/Prev (`delta` ±1): plain wrap, never stops, regardless of
    /// repeat mode. `base` is the not-yet-applied target if one is already queued, so two quick
    /// Nexts advance by two.
    pub fn manual_step(&self, delta: isize) -> Option<usize> {
        let n = self.entries.len();
        if n == 0 {
            return None;
        }
        let base = self.pending.unwrap_or(self.index) as isize;
        Some((base + delta).rem_euclid(n as isize) as usize)
    }

    /// What a natural end-of-song advance does, per repeat mode.
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

/// State the audio callback pulls from and the UI mutates. Guarded by a [`Mutex`] so the two
/// sides can share it (single-threaded on web, two threads on native).
pub struct AudioState {
    /// The currently-playing controller, if any.
    pub controller: Option<SynthController>,
    /// Live synthesis configuration (tuning, stereo, track enables).
    pub config: PerDeviceSettings,
    /// When set, the callback emits silence but keeps the controller intact.
    pub paused: bool,
    /// User master volume target (0..=1); the callback smooths toward it (no zipper noise).
    pub volume: f32,
    /// The callback's smoothed volume state.
    pub volume_smooth: f32,
    /// Per-song fade-*in* gain: ramps up quickly after a new controller is installed for click-free
    /// song switches. The end-of-song / manual fade-*out* now lives in the controller's transition
    /// policy (see [`optime_core::LoopAndTransitionOptions`]).
    pub fade_gain: f32,
    /// Whether we've already requested the controller's quick manual fade for a pending transition
    /// (so it's requested exactly once per song switch, not re-armed every callback).
    pub manual_fade_active: bool,
    /// Pause ramp: eases toward 0 when paused and back to 1 on resume (no pause pops).
    pub pause_gain: f32,
    /// Device sample rate, for converting render time into DSP load.
    pub sample_rate: f64,
    /// Smoothed audio-callback load: render time / buffer real-time budget (1.0 = can't keep up).
    pub dsp_load: f32,
    /// Number of currently sounding synthesizer voices.
    pub voices: usize,
    /// Monotonic time (seconds) the audio callback last ran. Used on the web to detect a
    /// suspended/stalled `AudioContext` (iOS suspends it on background) so the stream can be
    /// rebuilt; `f64::NEG_INFINITY` until the first callback fires.
    pub last_callback: f64,
    /// Audio-thread-owned playlist + advancement (see [`Playback`]).
    pub playback: Playback,
    /// Final-stage resampler from the fixed engine render rate ([`crate::audio::ENGINE_SAMPLE_RATE_HZ`])
    /// to the device's actual output rate.
    pub resampler: StreamResampler,
}

impl AudioState {
    fn new() -> Self {
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
            last_callback: f64::NEG_INFINITY,
            playback: Playback::default(),
            resampler: StreamResampler::new(),
        }
    }
}

/// Handle to the shared audio state.
pub type Shared = Arc<Mutex<AudioState>>;

/// Creates a fresh shared audio state.
pub fn new_shared() -> Shared {
    Arc::new(Mutex::new(AudioState::new()))
}

/// The sample rate used for offline WAV rendering (the legacy renderer's fixed rate).
pub const EXPORT_SAMPLE_RATE: u32 = 32768;

/// Renders a song to interleaved stereo samples offline, looping twice then fading out, exactly
/// like the legacy `renderAndDownloadSeq`. The loop counting and the fade ramp live in the shared
/// [`SynthController`] fade policy ([`LoopAndTransitionOptions::export`]); this just applies the
/// 0.5 export headroom and stops when the controller reports the song finished.
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

    // Render through the block path in device-buffer-sized chunks.
    const CHUNK_FRAMES: usize = 512;
    let mut buf = vec![0.0f32; 2 * CHUNK_FRAMES];

    while sample < max_samples {
        let n = CHUNK_FRAMES.min((max_samples - sample) as usize);
        let chunk = &mut buf[..2 * n];
        controller.fill(chunk, config);

        // The controller has already applied the fade gain; just add the export headroom.
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
        // Manual stepping ignores repeat mode (a real Next button always moves).
        assert_eq!(playback(3, 2, RepeatMode::One).manual_step(1), Some(0));
        assert_eq!(playback(0, 0, RepeatMode::All).manual_step(1), None);
    }

    #[test]
    fn manual_step_advances_from_queued_target() {
        // Two quick Nexts before the first decodes should advance by two, not one.
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

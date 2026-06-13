//! Shared state between the egui UI thread and the cpal audio callback, plus the offline
//! renderer used for WAV export.

use std::sync::{Arc, Mutex};

use optime_core::{SoundData, SynthConfig, SynthController};

/// State the audio callback pulls from and the UI mutates. Guarded by a [`Mutex`] so the two
/// sides can share it (single-threaded on web, two threads on native).
pub struct AudioState {
    /// The currently-playing controller, if any.
    pub controller: Option<SynthController>,
    /// Live synthesis configuration (tuning, stereo, track enables).
    pub config: SynthConfig,
    /// When set, the callback emits silence but keeps the controller intact.
    pub paused: bool,
    /// User master volume target (0..=1); the callback smooths toward it (no zipper noise).
    pub volume: f32,
    /// The callback's smoothed volume state.
    pub volume_smooth: f32,
    /// Per-song gain: ramps up quickly after a new controller is installed (click-free song
    /// switches) and down per `fade_step` for the end-of-song fade.
    pub fade_gain: f32,
    /// Per-sample gain decrement while fading out (0.0 = not fading; the gain then rises to 1).
    pub fade_step: f32,
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
}

impl AudioState {
    fn new() -> Self {
        Self {
            controller: None,
            config: SynthConfig::default(),
            paused: false,
            volume: 1.0,
            volume_smooth: 1.0,
            fade_gain: 1.0,
            fade_step: 0.0,
            pause_gain: 1.0,
            sample_rate: 48_000.0,
            dsp_load: 0.0,
            voices: 0,
            last_callback: f64::NEG_INFINITY,
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
/// like the legacy `renderAndDownloadSeq`.
pub fn render_to_samples(data: &SoundData, song_id: u32, config: &SynthConfig) -> Vec<(f32, f32)> {
    const FADEOUT_LENGTH: f64 = 10.0;
    const LOOP_COUNT: u32 = 2;
    let sr = EXPORT_SAMPLE_RATE as f64;

    let Some(mut controller) = SynthController::new(sr, data, song_id) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    let mut sample: u64 = 0;
    let mut loop_count = 0u32;
    let mut fading_out = false;
    let mut fadeout_start_sample = 0u64;
    let max_samples = (sr * 480.0) as u64;

    // Render through the block path in device-buffer-sized chunks. The loop/fade flags only
    // change on sequencer ticks, so checking them once per chunk shifts the fade start by at
    // most one chunk — negligible against the two-second pre-fade grace below.
    const CHUNK_FRAMES: usize = 512;
    let mut buf = vec![0.0f32; 2 * CHUNK_FRAMES];

    'render: while sample < max_samples {
        let n = CHUNK_FRAMES.min((max_samples - sample) as usize);
        let chunk = &mut buf[..2 * n];
        controller.fill(chunk, config);

        if controller.jumps > 0 {
            controller.jumps = 0;
            loop_count += 1;
            if loop_count == LOOP_COUNT {
                controller.fading_start = true;
            }
        }

        if controller.fading_start {
            controller.fading_start = false;
            fading_out = true;
            fadeout_start_sample = sample + (sr * 2.0) as u64;
        }

        for frame in chunk.chunks_exact(2) {
            let mut fadeout_mul = 1.0f32;
            if fading_out && sample >= fadeout_start_sample {
                let fadeout_time = (sample - fadeout_start_sample) as f64 / sr;
                let ratio = fadeout_time / FADEOUT_LENGTH;
                fadeout_mul = (1.0 - ratio) as f32;
                if fadeout_mul <= 0.0 {
                    break 'render;
                }
            }
            out.push((frame[0] * 0.5 * fadeout_mul, frame[1] * 0.5 * fadeout_mul));
            sample += 1;
        }
    }

    out
}

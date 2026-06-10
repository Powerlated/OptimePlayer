//! Shared state between the egui UI thread and the cpal audio callback, plus the offline
//! renderer used for WAV export.

use std::sync::{Arc, Mutex};

use optime_core::{Controller, Sdat, SynthConfig};

/// State the audio callback pulls from and the UI mutates. Guarded by a [`Mutex`] so the two
/// sides can share it (single-threaded on web, two threads on native).
pub struct AudioState {
    /// The currently-playing controller, if any.
    pub controller: Option<Controller>,
    /// Live synthesis configuration (tuning, stereo, track enables).
    pub config: SynthConfig,
    /// When set, the callback emits silence but keeps the controller intact.
    pub paused: bool,
}

impl AudioState {
    fn new() -> Self {
        Self {
            controller: None,
            config: SynthConfig::default(),
            paused: false,
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

/// Renders an SSEQ to interleaved stereo samples offline, looping twice then fading out, exactly
/// like the legacy `renderAndDownloadSeq`.
pub fn render_to_samples(sdat: &Sdat, sseq_id: u32, config: &SynthConfig) -> Vec<(f32, f32)> {
    const FADEOUT_LENGTH: f64 = 10.0;
    const LOOP_COUNT: u32 = 2;
    let sr = EXPORT_SAMPLE_RATE as f64;

    let Some(mut controller) = Controller::new(sr, sdat, sseq_id) else {
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

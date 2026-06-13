//! cpal output stream that pulls stereo samples from the shared [`AudioState`]. Works on both
//! native (ALSA/CoreAudio/WASAPI) and web (WebAudio) targets.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::player::{new_shared, Shared};

/// Owns the live audio output stream and exposes the shared state the UI mutates.
pub struct AudioEngine {
    /// The cpal output stream, held alive for its duration. `Option` so it can be dropped
    /// (closing the web `AudioContext`) before a rebuild — see [`Self::rebuild`]. On native it's
    /// only kept alive, never read back.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    stream: Option<cpal::Stream>,
    /// The device sample rate; controllers must be built to match it.
    pub sample_rate: f64,
    /// Shared state pulled from by the audio callback.
    pub shared: Shared,
}

/// Opens the default output device and builds a started stereo f32 stream feeding `shared`.
/// Returns the stream and the device sample rate.
fn build_stream(shared: &Shared) -> Option<(cpal::Stream, f64)> {
    let host = cpal::default_host();
    let device = host.default_output_device()?;
    let supported = device.default_output_config().ok()?;
    let sample_rate = supported.sample_rate().0 as f64;
    let channels = supported.channels() as usize;
    let mut config: cpal::StreamConfig = supported.clone().into();
    config.buffer_size = cpal::BufferSize::Fixed(2048);

    if supported.sample_format() != cpal::SampleFormat::F32 {
        log::error!(
            "unsupported output sample format {:?}; expected f32",
            supported.sample_format()
        );
        return None;
    }

    let cb_shared = shared.clone();
    let err_fn = |e| log::error!("audio stream error: {e}");
    let stream = device
        .build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                write_audio(data, channels, &cb_shared);
            },
            err_fn,
            None,
        )
        .ok()?;
    stream.play().ok()?;
    Some((stream, sample_rate))
}

impl AudioEngine {
    /// Opens the default output device and starts a stereo f32 stream. Returns `None` if no
    /// device/format is available.
    pub fn new() -> Option<Self> {
        let shared = new_shared();
        let (stream, sample_rate) = build_stream(&shared)?;
        if let Ok(mut st) = shared.lock() {
            st.sample_rate = sample_rate;
        }
        Some(Self {
            stream: Some(stream),
            sample_rate,
            shared,
        })
    }

    /// (Re)starts the output stream — on the web this resumes a suspended `AudioContext`. Cheap;
    /// safe to call on every user interaction. Web-only recovery (see `keep_audio_alive`).
    #[cfg(target_arch = "wasm32")]
    pub fn resume(&self) {
        if let Some(stream) = &self.stream {
            let _ = stream.play();
        }
    }

    /// Seconds since the audio callback last ran (large when the stream is suspended/stalled).
    #[cfg(target_arch = "wasm32")]
    pub fn callback_age(&self) -> f64 {
        let last = self.shared.lock().map(|st| st.last_callback).unwrap_or(0.0);
        now_seconds() - last
    }

    /// Drops the current stream (closing its web `AudioContext`) and builds a fresh one over the
    /// same shared state, so playback continues seamlessly. Used to recover after iOS suspends
    /// the context on background. Returns `true` on success. Web-only recovery.
    #[cfg(target_arch = "wasm32")]
    pub fn rebuild(&mut self) -> bool {
        self.stream = None; // close the old context first (iOS limits live AudioContexts)
        match build_stream(&self.shared) {
            Some((stream, sample_rate)) => {
                self.sample_rate = sample_rate;
                if let Ok(mut st) = self.shared.lock() {
                    st.last_callback = now_seconds();
                }
                self.stream = Some(stream);
                true
            }
            None => false,
        }
    }
}

/// Monotonic clock in seconds (`Instant` is unavailable on wasm; use `performance.now()`).
fn now_seconds() -> f64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::OnceLock;
        use std::time::Instant;
        static START: OnceLock<Instant> = OnceLock::new();
        START.get_or_init(Instant::now).elapsed().as_secs_f64()
    }
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now() / 1000.0)
            .unwrap_or(0.0)
    }
}

/// Updates the DSP-load and voice-count meters after a buffer has been rendered.
fn update_meters(st: &mut crate::player::AudioState, t0: f64, frames: usize) {
    st.voices = st
        .controller
        .as_ref()
        .map(|c| c.synthesizers.iter().map(|s| s.active_voice_count()).sum())
        .unwrap_or(0);
    let budget = (frames as f64 / st.sample_rate.max(1.0)).max(1e-9);
    let load = ((now_seconds() - t0) / budget) as f32;
    st.dsp_load = st.dsp_load * 0.85 + load.clamp(0.0, 2.0) * 0.15;
}

/// The per-sample gain pipeline: smoothed master volume × song fade (in/out) × pause ramp.
/// All transitions are ramped inside the callback so no UI action produces a click.
struct GainRamp {
    volume: f32,
    volume_target: f32,
    /// One-pole smoothing fraction per sample (~10 ms time constant).
    volume_alpha: f32,
    song: f32,
    /// Downward fade step (end-of-song); when 0 the song gain rises instead (fade-in).
    song_down_step: f32,
    /// Upward fade-in step (~30 ms).
    song_up_step: f32,
    pause: f32,
    pause_target: f32,
    /// Pause ramp step (~25 ms).
    pause_step: f32,
}

impl GainRamp {
    fn new(st: &crate::player::AudioState) -> Self {
        let sr = st.sample_rate.max(1.0) as f32;
        Self {
            volume: st.volume_smooth,
            volume_target: st.volume,
            volume_alpha: (1.0 / (0.010 * sr)).min(1.0),
            song: st.fade_gain,
            song_down_step: st.fade_step,
            song_up_step: 1.0 / (0.030 * sr),
            pause: st.pause_gain,
            pause_target: if st.paused { 0.0 } else { 1.0 },
            pause_step: 1.0 / (0.025 * sr),
        }
    }

    /// Advances every ramp by one sample and returns the combined gain (incl. the 0.5 master).
    #[inline]
    fn next(&mut self) -> f32 {
        self.volume += (self.volume_target - self.volume) * self.volume_alpha;
        self.song = if self.song_down_step > 0.0 {
            (self.song - self.song_down_step).max(0.0)
        } else {
            (self.song + self.song_up_step).min(1.0)
        };
        self.pause = if self.pause < self.pause_target {
            (self.pause + self.pause_step).min(self.pause_target)
        } else {
            (self.pause - self.pause_step).max(self.pause_target)
        };
        0.5 * self.volume * self.song * self.pause
    }

    /// Writes the ramp state back into the shared audio state.
    fn store(self, st: &mut crate::player::AudioState) {
        st.volume_smooth = self.volume;
        st.fade_gain = self.song;
        st.pause_gain = self.pause;
    }
}

/// Fills the device buffer from the controller (or silence when idle / fully paused). While the
/// pause ramp is still easing out, the controller keeps rendering so the tail fades smoothly.
fn write_audio(data: &mut [f32], channels: usize, shared: &Shared) {
    let t0 = now_seconds();
    let Ok(mut guard) = shared.lock() else {
        data.fill(0.0);
        return;
    };
    let st = &mut *guard;
    // Heartbeat for the web stall detector (see `AudioEngine::callback_age`).
    st.last_callback = t0;
    if st.controller.is_none() || (st.paused && st.pause_gain <= 0.0) {
        data.fill(0.0);
        if !st.paused {
            st.pause_gain = 1.0;
        }
        st.volume_smooth = st.volume;
        st.dsp_load *= 0.85;
        st.voices = 0;
        return;
    }
    let mut ramp = GainRamp::new(st);
    let config = &st.config;
    let controller = st.controller.as_mut().unwrap();
    if channels == 2 {
        // The common case: render the whole device buffer through the block path.
        controller.fill(data, config);
        for frame in data.chunks_exact_mut(2) {
            let g = ramp.next();
            frame[0] *= g;
            frame[1] *= g;
        }
        ramp.store(st);
        update_meters(st, t0, data.len() / 2);
        return;
    }
    for frame in data.chunks_mut(channels.max(1)) {
        let (l, r) = controller.next_sample(config);
        let g = ramp.next();
        let (l, r) = (l * g, r * g);
        match frame.len() {
            0 => {}
            1 => frame[0] = (l + r) * 0.5,
            _ => {
                frame[0] = l;
                frame[1] = r;
                for s in &mut frame[2..] {
                    *s = 0.0;
                }
            }
        }
    }
    ramp.store(st);
    update_meters(st, t0, data.len() / channels.max(1));
}

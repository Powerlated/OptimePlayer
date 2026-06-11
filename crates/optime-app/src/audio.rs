//! cpal output stream that pulls stereo samples from the shared [`AudioState`]. Works on both
//! native (ALSA/CoreAudio/WASAPI) and web (WebAudio) targets.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::player::{new_shared, Shared};

/// Owns the live audio output stream and exposes the shared state the UI mutates.
pub struct AudioEngine {
    _stream: cpal::Stream,
    /// The device sample rate; controllers must be built to match it.
    pub sample_rate: f64,
    /// Shared state pulled from by the audio callback.
    pub shared: Shared,
}

impl AudioEngine {
    /// Opens the default output device and starts a stereo f32 stream. Returns `None` if no
    /// device/format is available.
    pub fn new() -> Option<Self> {
        let host = cpal::default_host();
        let device = host.default_output_device()?;
        let supported = device.default_output_config().ok()?;
        let sample_rate = supported.sample_rate().0 as f64;
        let channels = supported.channels() as usize;
        let mut config: cpal::StreamConfig = supported.clone().into();
        config.buffer_size = cpal::BufferSize::Fixed(2048);

        let shared = new_shared();
        if let Ok(mut st) = shared.lock() {
            st.sample_rate = sample_rate;
        }
        let cb_shared = shared.clone();

        if supported.sample_format() != cpal::SampleFormat::F32 {
            log::error!(
                "unsupported output sample format {:?}; expected f32",
                supported.sample_format()
            );
            return None;
        }

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

        Some(Self {
            _stream: stream,
            sample_rate,
            shared,
        })
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
        .map(|c| c.synthesizers.iter().map(|s| s.voice_count()).sum())
        .unwrap_or(0);
    let budget = (frames as f64 / st.sample_rate.max(1.0)).max(1e-9);
    let load = ((now_seconds() - t0) / budget) as f32;
    st.dsp_load = st.dsp_load * 0.85 + load.clamp(0.0, 2.0) * 0.15;
}

/// Fills the device buffer from the controller (or silence when idle/paused).
fn write_audio(data: &mut [f32], channels: usize, shared: &Shared) {
    let t0 = now_seconds();
    let Ok(mut guard) = shared.lock() else {
        data.fill(0.0);
        return;
    };
    let st = &mut *guard;
    if st.paused || st.controller.is_none() {
        data.fill(0.0);
        st.dsp_load *= 0.85;
        st.voices = 0;
        return;
    }
    let config = &st.config;
    let volume = st.volume;
    let mut gain = st.fade_gain;
    let step = st.fade_step;
    let controller = st.controller.as_mut().unwrap();
    if channels == 2 {
        // The common case: render the whole device buffer through the block path.
        controller.fill(data, config);
        for frame in data.chunks_exact_mut(2) {
            frame[0] *= 0.5 * volume * gain;
            frame[1] *= 0.5 * volume * gain;
            gain = (gain - step).max(0.0);
        }
        st.fade_gain = gain;
        update_meters(st, t0, data.len() / 2);
        return;
    }
    for frame in data.chunks_mut(channels.max(1)) {
        let (l, r) = controller.next_sample(config);
        let (l, r) = (l * 0.5 * volume * gain, r * 0.5 * volume * gain);
        gain = (gain - step).max(0.0);
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
    st.fade_gain = gain;
    update_meters(st, t0, data.len() / channels.max(1));
}

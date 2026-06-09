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
        config.buffer_size = cpal::BufferSize::Fixed(8192);

        let shared = new_shared();
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

/// Fills the device buffer from the controller (or silence when idle/paused).
fn write_audio(data: &mut [f32], channels: usize, shared: &Shared) {
    let Ok(mut guard) = shared.lock() else {
        data.fill(0.0);
        return;
    };
    let st = &mut *guard;
    if st.paused || st.controller.is_none() {
        data.fill(0.0);
        return;
    }
    let config = &st.config;
    let controller = st.controller.as_mut().unwrap();
    for frame in data.chunks_mut(channels.max(1)) {
        let (l, r) = controller.next_sample(config);
        let (l, r) = (l * 0.5, r * 0.5);
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
}

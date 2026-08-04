use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use optime_core::{
    InstrumentResampleMode, LoopAndTransitionOptions, PlaybackEvent, SynthController,
};

use crate::player::{AudioState, AutoAdvance, PlaybackCommand, Shared, new_shared};

pub const ENGINE_SAMPLE_RATE_HZ: f64 = 32768.0;

pub struct AudioEngine {
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    stream: Option<cpal::Stream>,
    pub sample_rate: f64,
    pub shared: Shared,
}

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

    #[cfg(target_arch = "wasm32")]
    pub fn resume(&self) {
        if let Some(stream) = &self.stream {
            let _ = stream.play();
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn callback_age(&self) -> f64 {
        let last = self.shared.lock().map(|st| st.last_callback).unwrap_or(0.0);
        now_seconds() - last
    }

    #[cfg(target_arch = "wasm32")]
    pub fn rebuild(&mut self) -> bool {
        self.stream = None;
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

fn update_meters(st: &mut crate::player::AudioState, t0: f64, frames: usize) {
    st.voices = st
        .controller
        .as_ref()
        .map(|c| c.active_voice_count())
        .unwrap_or(0);
    let budget = (frames as f64 / st.sample_rate.max(1.0)).max(1e-9);
    let load = ((now_seconds() - t0) / budget) as f32;
    st.dsp_load = st.dsp_load * 0.85 + load.clamp(0.0, 2.0) * 0.15;
    update_high_comp_meter(st, frames);
}

fn update_high_comp_meter(st: &mut crate::player::AudioState, frames: usize) {
    let target = st
        .controller
        .as_ref()
        .map(|c| c.high_band_gr_db() as f32)
        .unwrap_or(0.0);
    let held = st.high_comp_gr_db;
    if target <= held {
        st.high_comp_gr_db = target;
    } else {
        let dt = frames as f32 / st.sample_rate.max(1.0) as f32;
        let alpha = (dt / 0.200).min(1.0);
        st.high_comp_gr_db = held + (target - held) * alpha;
    }
}

fn step_playback(st: &mut AudioState) {
    const MANUAL_FADE_SECONDS: f64 = 0.040;

    while let Some(cmd) = st.playback.commands.pop_front() {
        match cmd {
            PlaybackCommand::SetPlaylist { entries, index } => {
                st.playback.entries = entries;
                st.playback.pending = Some(index);
            }
            PlaybackCommand::Reorder { entries, index } => {
                st.playback.entries = entries;
                st.playback.index = index;
            }
            PlaybackCommand::PlayAt(i) => st.playback.pending = Some(i),
            PlaybackCommand::Next => st.playback.pending = st.playback.manual_step(1),
            PlaybackCommand::Prev => st.playback.pending = st.playback.manual_step(-1),
        }
    }
    if st.playback.pending.is_some()
        && !st.manual_fade_active
        && let Some(controller) = &mut st.controller
    {
        controller.request_transition(MANUAL_FADE_SECONDS);
        st.manual_fade_active = true;
    }

    let finished = st
        .controller
        .as_mut()
        .is_some_and(|c| c.take_messages().any(|m| m == PlaybackEvent::Finished));

    let silent = st.controller.is_none() || finished;
    if !silent {
        return;
    }
    if st.playback.pending.is_none() && st.controller.is_some() && finished {
        match st.playback.auto_advance() {
            AutoAdvance::Repeat => st.playback.pending = Some(st.playback.index),
            AutoAdvance::Play(i) => st.playback.pending = Some(i),
            AutoAdvance::Stop => {
                st.controller = None;
                st.manual_fade_active = false;
                st.playback.stopped = true;
                st.playback.status_gen = st.playback.status_gen.wrapping_add(1);
                return;
            }
        }
    }
    let Some(target) = st.playback.pending.take() else {
        return;
    };
    let to_decode = st
        .playback
        .entries
        .get(target)
        .map(|e| (e.archive.clone(), e.track.song_id));
    let decoded = match to_decode {
        Some((Some(arc), song_id)) => SynthController::new(ENGINE_SAMPLE_RATE_HZ, &*arc, song_id),
        _ => None,
    };
    match decoded {
        Some(mut controller) => {
            controller.set_loop_and_transition(LoopAndTransitionOptions::live());
            st.controller = Some(controller);
            st.resampler.reset();
            st.playback.index = target;
            st.manual_fade_active = false;
            st.fade_gain = 0.0;
            st.playback.stopped = false;
            st.playback.needs_ui = false;
        }
        None => {
            st.controller = None;
            st.manual_fade_active = false;
            if target < st.playback.entries.len() {
                st.playback.index = target;
                st.playback.needs_ui = true;
            }
        }
    }
    st.playback.status_gen = st.playback.status_gen.wrapping_add(1);
}

struct GainRamp {
    volume: f32,
    volume_target: f32,
    volume_alpha: f32,
    song: f32,
    song_up_step: f32,
}

impl GainRamp {
    fn new(st: &crate::player::AudioState) -> Self {
        let sr = st.sample_rate.max(1.0) as f32;
        Self {
            volume: st.volume_smooth,
            volume_target: st.volume,
            volume_alpha: (1.0 / (0.010 * sr)).min(1.0),
            song: st.fade_gain,
            song_up_step: 1.0 / (0.030 * sr),
        }
    }

    #[inline]
    fn next(&mut self) -> f32 {
        self.volume += (self.volume_target - self.volume) * self.volume_alpha;
        self.song = (self.song + self.song_up_step).min(1.0);
        0.5 * self.volume * self.song
    }

    fn store(self, st: &mut crate::player::AudioState) {
        st.volume_smooth = self.volume;
        st.fade_gain = self.song;
    }
}

struct PauseRamp {
    gain: f32,
    target: f32,
    step: f32,
}

impl PauseRamp {
    fn new(st: &crate::player::AudioState, rate: f32) -> Self {
        Self {
            gain: st.pause_gain,
            target: if st.paused { 0.0 } else { 1.0 },
            step: 1.0 / (0.025 * rate.max(1.0)),
        }
    }

    #[inline]
    fn next(&mut self) -> f32 {
        self.gain = if self.gain < self.target {
            (self.gain + self.step).min(self.target)
        } else {
            (self.gain - self.step).max(self.target)
        };
        self.gain
    }

    fn store(self, st: &mut crate::player::AudioState) {
        st.pause_gain = self.gain;
    }
}

fn write_audio(data: &mut [f32], channels: usize, shared: &Shared) {
    let t0 = now_seconds();
    let Ok(mut guard) = shared.lock() else {
        data.fill(0.0);
        return;
    };
    let st = &mut *guard;
    st.last_callback = t0;
    if st.bounce.active {
        mix_annotation(data, channels, st, t0);
        return;
    }
    step_playback(st);
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
    let device_rate = st.sample_rate as f32;
    let mut ramp = GainRamp::new(st);
    let mut pause = PauseRamp::new(st, device_rate);
    let config = &st.config;
    let controller = st.controller.as_mut().unwrap();
    let resampler = &mut st.resampler;
    resampler.set(
        ENGINE_SAMPLE_RATE_HZ as f32,
        device_rate,
        InstrumentResampleMode::SincSampleNyquist { half_taps: 8 },
    );
    const CHUNK: usize = 256;
    let mut scratch_l = [0.0f32; CHUNK];
    let mut scratch_r = [0.0f32; CHUNK];
    if channels == 2 {
        for block in data.chunks_mut(2 * CHUNK) {
            let n = block.len() / 2;
            resampler.process(&mut scratch_l[..n], &mut scratch_r[..n], &mut |l, r| {
                controller.render(l, r, config)
            });
            for ((out, &l), &r) in block
                .chunks_exact_mut(2)
                .zip(&scratch_l[..n])
                .zip(&scratch_r[..n])
            {
                let g = ramp.next() * pause.next();
                out[0] = l * g;
                out[1] = r * g;
            }
        }
        ramp.store(st);
        pause.store(st);
        update_meters(st, t0, data.len() / 2);
        return;
    }
    let frames_per_call = channels.max(1);
    for block in data.chunks_mut(frames_per_call * CHUNK) {
        let n = block.len() / frames_per_call;
        resampler.process(&mut scratch_l[..n], &mut scratch_r[..n], &mut |l, r| {
            controller.render(l, r, config)
        });
        for ((frame, &l), &r) in block
            .chunks_mut(frames_per_call)
            .zip(&scratch_l[..n])
            .zip(&scratch_r[..n])
        {
            let g = ramp.next() * pause.next();
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
    }
    ramp.store(st);
    pause.store(st);
    update_meters(st, t0, data.len() / frames_per_call);
}

fn mix_annotation(data: &mut [f32], channels: usize, st: &mut crate::player::AudioState, t0: f64) {
    let device_rate = st.sample_rate as f32;
    let mut ramp = GainRamp::new(st);
    let mut pause = PauseRamp::new(st, ENGINE_SAMPLE_RATE_HZ as f32);
    let resampler = &mut st.resampler;
    let bounce = &mut st.bounce;
    resampler.set(
        ENGINE_SAMPLE_RATE_HZ as f32,
        device_rate,
        InstrumentResampleMode::SincSampleNyquist { half_taps: 8 },
    );

    const CHUNK: usize = 256;
    let mut scratch_l = [0.0f32; CHUNK];
    let mut scratch_r = [0.0f32; CHUNK];
    let frames_per_call = channels.max(1);
    for block in data.chunks_mut(frames_per_call * CHUNK) {
        let n = block.len() / frames_per_call;
        resampler.process(
            &mut scratch_l[..n],
            &mut scratch_r[..n],
            &mut |out_l, out_r| {
                for (out_l, out_r) in out_l.iter_mut().zip(out_r.iter_mut()) {
                    let (l, r) = bounce.next_frame();
                    let p = pause.next();
                    let (mut l, mut r) = (l * p, r * p);
                    if let Some(c) = &mut bounce.chords
                        && bounce.chords_on
                    {
                        let (cl, cr) = c.next_frame();
                        l += cl;
                        r += cr;
                    }
                    (*out_l, *out_r) = (l, r);
                }
            },
        );
        for ((frame, &l), &r) in block
            .chunks_mut(frames_per_call)
            .zip(&scratch_l[..n])
            .zip(&scratch_r[..n])
        {
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
    }
    ramp.store(st);
    pause.store(st);
    update_meters(st, t0, data.len() / frames_per_call);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotation::ChordVoicer;
    use crate::annotation::model::{Chord, Quality};

    fn annotating() -> AudioState {
        let instrument = crate::annotation::chord_voice::embedded_piano();
        let mut st = AudioState::new();
        st.sample_rate = 48_000.0;
        st.volume = 1.0;
        st.volume_smooth = 1.0;
        st.fade_gain = 1.0;
        st.bounce.active = true;
        st.bounce.buffer = None;
        st.bounce.chords = Some(ChordVoicer::new(instrument));
        st.bounce.chords_on = true;
        st
    }

    fn peak(data: &[f32]) -> f32 {
        data.iter().fold(0.0f32, |a, b| a.max(b.abs()))
    }

    fn c_major() -> Chord {
        Chord {
            root: 0,
            quality: Quality::Major,
            quality_uncertain: false,
        }
    }

    #[test]
    fn chords_sound_before_the_bounce_is_rendered() {
        let mut st = annotating();
        st.bounce.chords.as_mut().unwrap().strike((0, c_major()));
        let mut data = vec![0.0f32; 2 * 4096];
        mix_annotation(&mut data, 2, &mut st, 0.0);
        assert!(
            peak(&data) > 0.01,
            "no chord audio with buffer=None (peak {})",
            peak(&data)
        );
    }

    #[test]
    fn chords_sound_while_paused_but_the_song_does_not() {
        let mut st = annotating();
        st.paused = true;
        st.pause_gain = 0.0;
        st.bounce.chords.as_mut().unwrap().strike((0, c_major()));
        let mut data = vec![0.0f32; 2 * 4096];
        mix_annotation(&mut data, 2, &mut st, 0.0);
        assert!(
            peak(&data) > 0.01,
            "a paused transport silenced the chord audition (peak {})",
            peak(&data)
        );
    }

    #[test]
    fn the_idle_mixer_is_silent() {
        let mut st = annotating();
        let mut data = vec![0.0f32; 2 * 4096];
        mix_annotation(&mut data, 2, &mut st, 0.0);
        assert_eq!(peak(&data), 0.0, "the idle annotation mixer must be silent");
    }
}

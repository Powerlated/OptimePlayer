//! cpal output stream that pulls stereo samples from the shared [`AudioState`]. Works on both
//! native (ALSA/CoreAudio/WASAPI) and web (WebAudio) targets.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use optime_core::{
    InstrumentResampleMode, LoopAndTransitionOptions, PlaybackEvent, SynthController,
};

use crate::player::{AudioState, AutoAdvance, PlaybackCommand, Shared, new_shared};

/// Fixed rate the engine always renders at, emulating the hardware DACs' 32768 Hz output. A
/// [`optime_core::StreamResampler`] then converts this to the device's actual output rate in
/// [`write_audio`] (a final resampling stage not exposed in the UI), so device rate never touches
/// synthesis.
pub const ENGINE_SAMPLE_RATE_HZ: f64 = 32768.0;

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
        .map(|c| c.active_voice_count())
        .unwrap_or(0);
    let budget = (frames as f64 / st.sample_rate.max(1.0)).max(1e-9);
    let load = ((now_seconds() - t0) / budget) as f32;
    st.dsp_load = st.dsp_load * 0.85 + load.clamp(0.0, 2.0) * 0.15;
    update_high_comp_meter(st, frames);
}

/// Smooths the high-band-compressor gain-reduction meter: instant catch on deeper reduction,
/// ~200 ms one-pole release back toward 0. The compressor's own detector is already attack/release
/// smoothed per sample; this peak-hold-with-release makes the per-buffer reading readable at UI
/// frame rate.
fn update_high_comp_meter(st: &mut crate::player::AudioState, frames: usize) {
    let target = st
        .controller
        .as_ref()
        .map(|c| c.high_band_gr_db() as f32)
        .unwrap_or(0.0);
    let held = st.high_comp_gr_db;
    if target <= held {
        // Deeper-than-held sample: peaks read straight through.
        st.high_comp_gr_db = target;
    } else {
        // Release toward the held-then-0 target with a ~200 ms one-pole time constant, advanced
        // by the buffer's wall-clock length (`frames / sample_rate` seconds).
        let dt = frames as f32 / st.sample_rate.max(1.0) as f32;
        let alpha = (dt / 0.200).min(1.0);
        st.high_comp_gr_db = held + (target - held) * alpha;
    }
}

/// Drives the audio-thread-owned playlist: drains UI commands, triggers the end-of-song fade,
/// and — once the output has faded to silence — advances the index and decodes+installs the next
/// song. Runs every callback before rendering, so advancement keeps working while the UI's
/// repaint loop is frozen (e.g. a hidden browser tab). Decoding here is intentional: it only
/// happens at a transition while the output is already silent, so the brief stall is inaudible.
fn step_playback(st: &mut AudioState) {
    // ~40 ms fade-out before a manual transition, so the lazy decode lands in silence.
    const MANUAL_FADE_SECONDS: f64 = 0.040;

    // (a) Drain UI commands. Manual intents take precedence over a natural-end fade.
    while let Some(cmd) = st.playback.commands.pop_front() {
        match cmd {
            PlaybackCommand::SetPlaylist { entries, index } => {
                st.playback.entries = entries;
                st.playback.pending = Some(index);
            }
            PlaybackCommand::Reorder { entries, index } => {
                // Swap the list under the still-playing song; no transition.
                st.playback.entries = entries;
                st.playback.index = index;
            }
            PlaybackCommand::PlayAt(i) => st.playback.pending = Some(i),
            PlaybackCommand::Next => st.playback.pending = st.playback.manual_step(1),
            PlaybackCommand::Prev => st.playback.pending = st.playback.manual_step(-1),
        }
    }
    // A manual transition interrupting a playing song asks the controller for a quick fade-out —
    // once per switch (it overrides any in-progress end-of-song fade with the faster ramp).
    if st.playback.pending.is_some()
        && !st.manual_fade_active
        && let Some(controller) = &mut st.controller
    {
        controller.request_transition(MANUAL_FADE_SECONDS);
        st.manual_fade_active = true;
    }

    // (b) The end-of-song fade (loop-count / FINE) and our requested manual fade both live in the
    //     controller now; it reports `Finished` once it has faded to silence.
    let finished = st
        .controller
        .as_mut()
        .is_some_and(|c| c.take_messages().any(|m| m == PlaybackEvent::Finished));

    // (c) Perform a transition once we're silent (faded out, or nothing playing yet).
    let silent = st.controller.is_none() || finished;
    if !silent {
        return;
    }
    // Resolve a natural-end advance if no manual target is queued and a song actually finished.
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
    // Clone out what the decode needs so the entries borrow ends before we mutate `st`.
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
            st.fade_gain = 0.0; // fade the new song in
            st.playback.stopped = false;
            st.playback.needs_ui = false;
        }
        None => {
            // Entry missing or its source isn't loaded (cross-source): go silent, ask the UI.
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

/// The per-sample gain pipeline: smoothed master volume × song fade-in. The end-of-song / manual
/// fade-*out* now lives in the controller (it has already been applied to the samples this
/// multiplies). All transitions are ramped inside the callback so no UI action pops.
///
/// Pause is deliberately *not* here — see [`PauseRamp`].
struct GainRamp {
    volume: f32,
    volume_target: f32,
    /// One-pole smoothing fraction per sample (~10 ms time constant).
    volume_alpha: f32,
    /// Song fade-*in* level (0 → 1 after a new controller is installed); never fades out here.
    song: f32,
    /// Upward fade-in step (~30 ms).
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

    /// Advances every ramp by one sample and returns the combined gain (incl. the 0.5 master).
    #[inline]
    fn next(&mut self) -> f32 {
        self.volume += (self.volume_target - self.volume) * self.volume_alpha;
        self.song = (self.song + self.song_up_step).min(1.0);
        0.5 * self.volume * self.song
    }

    /// Writes the ramp state back into the shared audio state.
    fn store(self, st: &mut crate::player::AudioState) {
        st.volume_smooth = self.volume;
        st.fade_gain = self.song;
    }
}

/// The pause fade (~25 ms), on its own ramp rather than folded into [`GainRamp`].
///
/// It is separate because the two callers need it in different places. The live path applies it with
/// everything else, per output sample. The annotation mixer applies it to the prerendered song
/// *alone* — a chord audition must sound while the transport is stopped — and therefore has to apply
/// it before the chords are summed in, which happens at engine rate ahead of the output resampler.
/// Hence the explicit `rate`: the ramp steps per sample of whichever bus it rides, keeping the same
/// ~25 ms wall-clock length either way.
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
    // Annotation owns the output for as long as the mode is on: its mixer runs continuously, so a
    // right-clicked chord sounds while the bounce is still rendering and while the transport is
    // stopped. Once the bounce exists the same mixer plays it under the chords, which is what keeps
    // the two synchronous — one frame of song and one frame of chord leave together. The playlist
    // stays idle throughout (no fade, no decode, no advancement).
    if st.bounce.active {
        mix_annotation(data, channels, st, t0);
        return;
    }
    // Advance the playlist (fade trigger + lazy decode) before rendering; keeps working while
    // the UI is frozen. May install a new controller.
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
    // The live path wants pause folded in with everything else, per output sample.
    let mut pause = PauseRamp::new(st, device_rate);
    let config = &st.config;
    let controller = st.controller.as_mut().unwrap();
    let resampler = &mut st.resampler;
    // Cheap per docs: only rebuilds sinc tables on a half-taps change, never disturbs position.
    resampler.set(
        ENGINE_SAMPLE_RATE_HZ as f32,
        device_rate,
        InstrumentResampleMode::SincSampleNyquist { half_taps: 8 },
    );
    // The block resampler fills a scratch buffer of engine→device frames per chunk, pulling
    // engine-rate audio from the controller a block at a time — so the whole synthesis chain runs
    // blocked, never a sample at a time. The per-frame gain ramp is then applied while writing into
    // the interleaved output.
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

/// The annotation mixer: two buses — the prerendered [`Bounce`] and the chord audition — summed at
/// [`ENGINE_SAMPLE_RATE_HZ`] and pushed through the output resampler as one.
///
/// Runs for the whole of annotation mode ([`BounceTransport::active`]), not just once a bounce
/// exists. That is what lets a right-clicked chord sound while the render is still in progress and
/// while the transport is stopped: with no buffer the song bus is simply silent and the chord bus
/// plays alone. Once the bounce arrives the two are summed *before* the resampler — at the same
/// engine rate, through the same resampler, so they cannot drift apart — which is what "synchronous"
/// means here.
///
/// The buses differ in one gain: [`PauseRamp`] applies to the song only, since auditioning a chord
/// is a deliberate act that must sound whether or not the transport is rolling. There is no fade
/// policy and no playlist: a looped bar must repeat bit-identically.
///
/// Uses the general channel-mapping loop rather than `write_audio`'s stereo fast path; annotation is
/// a maintainer tool and never runs in the hot playback case.
fn mix_annotation(data: &mut [f32], channels: usize, st: &mut crate::player::AudioState, t0: f64) {
    let device_rate = st.sample_rate as f32;
    let mut ramp = GainRamp::new(st);
    // Pause rides the engine-rate song bus, because that is where it has to be applied: ahead of
    // the point the chords are summed in.
    let mut pause = PauseRamp::new(st, ENGINE_SAMPLE_RATE_HZ as f32);
    // Disjoint field borrows: the resampler and the transport are separate members of `st`.
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
        // The two buses are summed frame by frame inside the block the resampler asks for. Both
        // sides are inherently per-frame here — one reads a prerendered buffer, the other steps
        // per-frame chord envelopes — and annotation is a maintainer tool that never runs in the
        // hot playback case.
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

    /// A state in annotation mode with the embedded piano voicer armed and nothing rendered yet.
    fn annotating() -> AudioState {
        let instrument = crate::annotation::chord_voice::embedded_piano();
        let mut st = AudioState::new();
        st.sample_rate = 48_000.0;
        st.volume = 1.0;
        st.volume_smooth = 1.0;
        st.fade_gain = 1.0;
        st.bounce.active = true;
        st.bounce.buffer = None; // still rendering
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

    /// The mixer runs for the whole of annotation mode, not just once a bounce exists: a chord
    /// right-clicked while the multi-second render is still going has to sound.
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

    /// A chord audition is a deliberate act and skips the pause ramp — right-clicking a bar while
    /// the transport is stopped is the main way this tool gets used.
    #[test]
    fn chords_sound_while_paused_but_the_song_does_not() {
        let mut st = annotating();
        st.paused = true;
        st.pause_gain = 0.0; // fully faded out
        st.bounce.chords.as_mut().unwrap().strike((0, c_major()));
        let mut data = vec![0.0f32; 2 * 4096];
        mix_annotation(&mut data, 2, &mut st, 0.0);
        assert!(
            peak(&data) > 0.01,
            "a paused transport silenced the chord audition (peak {})",
            peak(&data)
        );
    }

    /// The flip side: with nothing struck and nothing rendered, the mixer is silent — running
    /// always must not mean humming always.
    #[test]
    fn the_idle_mixer_is_silent() {
        let mut st = annotating();
        let mut data = vec![0.0f32; 2 * 4096];
        mix_annotation(&mut data, 2, &mut st, 0.0);
        assert_eq!(peak(&data), 0.0, "the idle annotation mixer must be silent");
    }
}

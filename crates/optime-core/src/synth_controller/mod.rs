//! [`SynthController`]: the device-agnostic synthesis runtime.
//!
//! It owns the per-track voice pools, the master clock, and the note-grid bookkeeping for
//! visualizers. All musical intelligence lives in the device player
//! ([`DevicePlayer`](crate::devices::DevicePlayer)); the controller drives it one tick at a
//! time and applies the standardized [`SynthEvent`] stream it produces to the voices.
//!
//! - [`config`] — the [`DelaySmoothing`]/[`HighShelf`]/[`PopSmoothing`] option types that make up
//!   the [`PerDeviceSettings`](crate::PerDeviceSettings) config struct the controller consumes.
//! - [`vis`] — the look-ahead [`FsVisController`] for visualizers.
//!
//! ## The intermediate mixer
//!
//! The controller owns two sets of per-track synthesizers: the output set, which renders at the
//! output rate, and the mixer set, which renders at the config's `mixer_sample_rate`. With
//! `use_mixer` set, sampled (non-PSG) voices play on the mixer set and PSG voices on
//! the output set; the controller *routes* each set's stereo audio to the final mix — the output
//! set straight through, the mixer set via the [`Bank`], which upsamples the mixer-rate bus to the
//! output rate. With the mixer off, every voice plays on the output set, so the result is
//! bit-identical to the single-set engine. The [`Bank`] holds only the resampler; it does not own
//! the synthesizers — audio flows *from* the synthesizers *to* the bank.

mod config;
pub mod messages;
mod vis;

pub use config::{DEFAULT_POP_SLEW_SECONDS, DelaySmoothing, HighShelf, PopSmoothing};
pub use vis::{FsVisController, SongOverview, VisNote};

use crate::devices::{DevicePlayer, SoundData, SynthEvent, TickFeedback, VoiceId};
use crate::dsp::biquad_filter::BiquadFilter;
use crate::dsp::resample::StreamResampler;
use crate::synth::MAX_BLOCK;
use crate::waveform::{Frame, InstrumentResampleMode, Sample};
use crate::{PerDeviceSettings, TRACK_COUNT, WaveformSynthesizer};

/// PSG crunch-compensation low-pass — a cascade of identical RBJ low-pass biquad sections fit
/// (MATLAB, `scripts/fit_compensation.m`, fed by `examples/mixer_resample_response.rs`) to the
/// measured nearest→crunch spectral power loss the mixer-to-output crunch imposes on real Emerald
/// DirectSound. It is specified as a **cutoff (Hz) + Q + cascade order** rather than baked
/// z-coefficients so it is rebuilt at whatever output rate the device runs at — the knee stays at a
/// fixed absolute frequency instead of drifting with the sample rate. Measured content-weighted
/// loss ≈ −0.05 dB; the audible effect is a high-frequency rolloff (≈ −3 dB at the cutoff). Order 6
/// is the diminishing-returns sweet spot: knee-region fit error falls 3.6→3.0→2.7 dB from order
/// 2→4→6, with order 8 buying only a further ~0.2 dB.
const PSG_COMP_ORDER: usize = 6;
const PSG_COMP_CUTOFF_HZ: f64 = 14_534.8;
const PSG_COMP_Q: f64 = 0.707;

/// Builds the PSG crunch-compensation low-pass for `sample_rate` (rebuilt on any rate change so the
/// knee tracks a fixed frequency in Hz).
fn psg_comp_filter(sample_rate: f64) -> BiquadFilter {
    BiquadFilter::low_pass(PSG_COMP_ORDER, sample_rate, PSG_COMP_CUTOFF_HZ, PSG_COMP_Q)
}

/// Whether the PSG crunch-compensation filter should run: the option is on, the mixer is engaged,
/// and the mixer-to-output stage is the (DirectSound-darkening) output-Nyquist crunch.
fn psg_comp_active(config: &PerDeviceSettings) -> bool {
    config.psg_crunch_compensation
        && config.use_mixer
        && matches!(
            config.mixer_resample_mode(),
            InstrumentResampleMode::SincOutputNyquist { .. }
        )
}

/// Which device voice currently owns a pool slot.
#[derive(Debug, Clone, Copy)]
struct SlotOwner {
    voice: VoiceId,
    key: u8,
}

/// `slot_owner[track][pool_slot]` — which device voice occupies each synthesizer voice in a set.
type SlotOwners = Vec<Vec<Option<SlotOwner>>>;

/// Builds an idle set of per-track synthesizers at `sample_rate` and its empty slot bookkeeping.
fn new_synth_set(sample_rate: f64) -> (Vec<WaveformSynthesizer>, SlotOwners) {
    let synths: Vec<_> = (0..TRACK_COUNT)
        .map(|_| WaveformSynthesizer::new(sample_rate, 16))
        .collect();
    let slot_owner = synths.iter().map(|s| vec![None; s.voice_count()]).collect();
    (synths, slot_owner)
}

/// The pool slot owned by `voice` on `track` within a set, if any.
fn find_slot(slot_owner: &SlotOwners, track: usize, voice: VoiceId) -> Option<usize> {
    slot_owner[track]
        .iter()
        .position(|o| o.is_some_and(|o| o.voice == voice))
}

/// Advances every voice in a set by one sample and returns the enabled-track stereo sum.
fn render_set(
    synths: &mut [WaveformSynthesizer],
    enables: &[bool],
    config: &PerDeviceSettings,
) -> Frame {
    let (mut l, mut r): Frame = (0.0, 0.0);
    for (synth, &enabled) in synths.iter_mut().zip(enables) {
        synth.next_sample(config);
        if enabled {
            l += synth.val_l;
            r += synth.val_r;
        }
    }
    (l, r)
}

/// Cuts one-shot samples in a set that ran out (only the synthesizer knows their playback
/// position), clearing their note-grid cells and reporting each ending to the device as feedback.
fn cut_finished(
    synths: &mut [WaveformSynthesizer],
    slot_owner: &mut SlotOwners,
    notes_on: &mut [[u8; 128]],
    feedback: &mut TickFeedback,
) {
    for (t, synth) in synths.iter_mut().enumerate() {
        for slot in 0..slot_owner[t].len() {
            let Some(owner) = slot_owner[t][slot] else {
                continue;
            };
            let instr = synth.instr(slot);
            let ran_out =
                !instr.waveform.looping && instr.sample_t > instr.waveform.data.len() as f32;
            if ran_out || !instr.playing {
                synth.cut_instrument(slot);
                slot_owner[t][slot] = None;
                notes_on[t][owner.key as usize] = 0;
                feedback.ended_voices.push((t, owner.voice));
            }
        }
    }
}

/// The intermediate mixer bank: a stereo mix bus the controller routes the mixer-set audio into.
///
/// It owns only the resampler (and its rate bookkeeping), **not** the synthesizers — the
/// controller renders the mixer set and routes its summed bus here, where it is upsampled from the
/// mixer rate to the output rate.
struct Bank {
    resampler: StreamResampler,
    /// The sample rate the feeding (mixer-set) synthesizers run at.
    rate: f64,
    /// Whether the bank was engaged on the previous render call (to reset on a fresh enable).
    was_active: bool,
}

impl Bank {
    fn new(rate: f64) -> Self {
        Self {
            resampler: StreamResampler::new(),
            rate,
            was_active: false,
        }
    }

    /// Marks the bank idle so the next enable starts the resampler from a clean ring.
    fn disable(&mut self) {
        self.was_active = false;
    }

    /// Reconfigures the resampler from `config` for output `out_rate`, once per render call.
    /// Resets on a fresh enable. Returns `Some(rate)` when the feeding synthesizers must be
    /// re-targeted to a new rate (the bank doesn't own them, so the controller does it).
    fn prepare(&mut self, config: &PerDeviceSettings, out_rate: f64) -> Option<f64> {
        if !self.was_active {
            self.resampler.reset();
            self.was_active = true;
        }
        let mixer_rate = f64::from(config.mixer_sample_rate);
        let rate_change = (self.rate != mixer_rate).then(|| {
            self.rate = mixer_rate;
            self.rate
        });
        // The mixer/output rates are frequencies (kept `f64` here); the resampler works in `f32`.
        self.resampler.set(
            self.rate as f32,
            out_rate as f32,
            config.mixer_resample_mode(),
        );
        rate_change
    }

    /// One upsampled stereo sample, pulling the mixer-rate bus from `render` on demand.
    fn route(&mut self, render: &mut impl FnMut() -> Frame) -> Frame {
        self.resampler.next(render)
    }
}

/// How a bound song loops and when it fades out — the single home for the end-of-song fade
/// policy that every renderer (live playback and the offline exporters) shares.
///
/// The consumer hands this to [`SynthController::set_loop_and_transition`]; the controller then
/// counts sequence loops, begins the linear fade-out when the policy says so, applies the fade
/// gain to its own output, and pumps [`PlaybackEvent`]s the consumer drains with
/// [`SynthController::take_messages`]. The default ([`LoopAndTransitionOptions::none`]) never
/// fades, so a controller that is just rendered (visualizers, tests) is unaffected.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoopAndTransitionOptions {
    /// Begin the end-of-song fade after the sequence has looped this many times (`None` = never).
    pub loops_before_fade: Option<u32>,
    /// At a loop point, also begin the fade if playback has by then run past this many seconds
    /// (`None` = no cap). Evaluated only on a loop, so the fade still starts at a musical boundary;
    /// it bounds songs whose loop is long enough to outlast [`Self::loops_before_fade`] iterations.
    pub too_long_after_loop_threshold_seconds: Option<f64>,
    /// Also begin the fade when the sequence signals it has ended (a `FINE`/end-of-song event).
    pub fade_on_end: bool,
    /// Hold full gain this long (seconds) after the fade is triggered before the ramp starts.
    pub grace_seconds: f64,
    /// Linear fade-out duration in seconds.
    pub fade_seconds: f64,
}

impl LoopAndTransitionOptions {
    /// No auto-fade: full gain forever (the controller's default).
    pub const fn none() -> Self {
        Self {
            loops_before_fade: None,
            too_long_after_loop_threshold_seconds: None,
            fade_on_end: false,
            grace_seconds: 0.0,
            fade_seconds: 0.0,
        }
    }

    /// The offline-export policy: fade after one loop, a 2 s grace, then a 3 s fade. A 90 s
    /// after-loop cap is also set so that, if `loops_before_fade` is raised, an overlong track
    /// still fades at the first loop past 90 s.
    pub const fn export() -> Self {
        Self {
            loops_before_fade: Some(0),
            too_long_after_loop_threshold_seconds: Some(90.0),
            fade_on_end: true,
            grace_seconds: 2.0,
            fade_seconds: 3.0,
        }
    }

    /// The live-playback policy: two loops (or end), no grace, a 3 s fade.
    pub const fn live() -> Self {
        Self {
            loops_before_fade: Some(2),
            too_long_after_loop_threshold_seconds: None,
            fade_on_end: true,
            grace_seconds: 0.0,
            fade_seconds: 3.0,
        }
    }
}

impl Default for LoopAndTransitionOptions {
    fn default() -> Self {
        Self::none()
    }
}

/// High-level playback events the controller pumps to its consumer (drained via
/// [`SynthController::take_messages`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PlaybackEvent {
    /// The sequence completed a loop.
    Looped,
    /// The end-of-song fade-out began (auto-triggered by the policy, or requested).
    TransitionStarted,
    /// The fade-out reached silence; the song is finished and the consumer should advance/stop.
    Finished,
}

/// Owns the loop-count → fade-out policy and the per-output-sample fade gain. Built from
/// [`LoopAndTransitionOptions`] and advanced one output sample at a time inside the render path,
/// so the linear ramp lives in exactly one place.
struct Transition {
    opts: LoopAndTransitionOptions,
    sample_rate: f64,
    loops_seen: u32,
    /// Output samples rendered since the song started, advanced once per sample in [`Self::advance`].
    /// Backs the `too_long_after_loop_threshold_seconds` length cap.
    total_samples: u64,
    /// `None` until the fade is triggered; then counts output samples since the trigger.
    elapsed: Option<u64>,
    /// Output samples of full-gain grace before the ramp starts.
    grace_samples: u64,
    /// Fade-out duration in seconds (the ramp denominator; matches the legacy `FADEOUT_LENGTH`).
    fade_seconds: f64,
    /// Latched once the gain has reached zero (so `Finished` is emitted exactly once).
    finished: bool,
}

impl Transition {
    fn new(sample_rate: f64, opts: LoopAndTransitionOptions) -> Self {
        Self {
            opts,
            sample_rate,
            loops_seen: 0,
            total_samples: 0,
            elapsed: None,
            grace_samples: (sample_rate * opts.grace_seconds) as u64,
            fade_seconds: opts.fade_seconds,
            finished: false,
        }
    }

    /// Re-arms the policy from scratch (no fade in progress, loop count reset).
    fn set_opts(&mut self, opts: LoopAndTransitionOptions) {
        *self = Self::new(self.sample_rate, opts);
    }

    /// Begins the configured fade now (idempotent: a fade already in progress is left alone).
    fn trip(&mut self, messages: &mut Vec<PlaybackEvent>) {
        if self.elapsed.is_none() {
            self.elapsed = Some(0);
            messages.push(PlaybackEvent::TransitionStarted);
        }
    }

    /// Begins an immediate fade of `fade_seconds` (no grace), overriding any in-progress fade with
    /// the new ramp — the live path's quick fade before a manual song switch.
    fn request(&mut self, fade_seconds: f64, messages: &mut Vec<PlaybackEvent>) {
        self.grace_samples = 0;
        self.fade_seconds = fade_seconds;
        self.elapsed = Some(0);
        self.finished = false;
        messages.push(PlaybackEvent::TransitionStarted);
    }

    /// A sequence loop occurred: count it, pump `Looped`, and trip the fade if the policy is met —
    /// either the loop count or the playback-length cap has been reached.
    fn on_loop(&mut self, messages: &mut Vec<PlaybackEvent>) {
        self.loops_seen += 1;
        messages.push(PlaybackEvent::Looped);
        let loops_met = self
            .opts
            .loops_before_fade
            .is_some_and(|n| self.loops_seen >= n);
        let too_long = self
            .opts
            .too_long_after_loop_threshold_seconds
            .is_some_and(|s| self.total_samples as f64 / self.sample_rate > s);
        if loops_met || too_long {
            self.trip(messages);
        }
    }

    /// The sequence signalled its end: trip the fade if the policy fades on end.
    fn on_end(&mut self, messages: &mut Vec<PlaybackEvent>) {
        if self.opts.fade_on_end {
            self.trip(messages);
        }
    }

    /// The fade gain for the current output sample (1.0 until triggered / during grace).
    fn gain(&self) -> f32 {
        match self.elapsed {
            None => 1.0,
            Some(n) => {
                if n < self.grace_samples {
                    1.0
                } else {
                    // Bit-for-bit the legacy ramp: `1 - (k/sr)/fade_seconds` in f64, then to f32.
                    let k = (n - self.grace_samples) as f64;
                    (1.0 - (k / self.sample_rate) / self.fade_seconds) as f32
                }
            }
        }
    }

    /// Returns the gain for this output sample and advances by one, emitting `Finished` the first
    /// time the ramp reaches silence (then holding at zero).
    fn advance(&mut self, messages: &mut Vec<PlaybackEvent>) -> f32 {
        self.total_samples += 1;
        match self.elapsed {
            None => 1.0,
            Some(n) => {
                let g = self.gain();
                if g <= 0.0 {
                    if !self.finished {
                        self.finished = true;
                        messages.push(PlaybackEvent::Finished);
                    }
                    return 0.0;
                }
                self.elapsed = Some(n + 1);
                g
            }
        }
    }
}

/// The synthesis runtime: voice pools + master clock, driven by a device player.
pub struct SynthController {
    sample_rate: f64,
    /// The device player generating the event stream.
    pub player: Box<dyn DevicePlayer>,
    /// The output-rate synthesizers (PSG voices, and every voice when the mixer is off).
    synths: Vec<WaveformSynthesizer>,
    slot_owner: SlotOwners,
    /// The mixer-rate synthesizers (sampled voices when the config's `use_mixer` is set).
    mixer_synths: Vec<WaveformSynthesizer>,
    mixer_slot_owner: SlotOwners,
    /// The mixer bus the mixer set's audio is routed into (owns the resampler, not the synths).
    bank: Bank,
    /// `notes_on[track][note]` is 1 while a sequence note sounds (drives the visualizer).
    pub notes_on: Vec<[u8; 128]>,
    /// The loop-count → fade-out policy and the per-sample fade gain it applies to the output.
    transition: Transition,
    /// High-level [`PlaybackEvent`]s pumped to the consumer (drained by [`Self::take_messages`]).
    messages: Vec<PlaybackEvent>,
    /// Synth-side voice endings reported back to the device on the next tick.
    feedback: TickFeedback,
    /// Reusable event buffer.
    events: Vec<SynthEvent>,
    timer: f64,
    /// Master high-shelf EQ on the final mixed output (left/right), and the params it's built for.
    shelf_l: BiquadFilter,
    shelf_r: BiquadFilter,
    shelf_params: Option<HighShelf>,
    /// PSG crunch-compensation biquads (left/right), and whether they ran on the previous sample
    /// (so the state is cleared on a fresh enable to avoid a transient).
    psg_comp_l: BiquadFilter,
    psg_comp_r: BiquadFilter,
    psg_comp_was_active: bool,
}

impl SynthController {
    /// Binds song `song_id` from `data` for playback at `sample_rate`.
    ///
    /// Returns `None` if the song is missing or malformed.
    pub fn new(sample_rate: f64, data: &dyn SoundData, song_id: u32) -> Option<SynthController> {
        let player = data.make_player(song_id)?;
        let mixer_rate = 48_000.0;
        let (synths, slot_owner) = new_synth_set(sample_rate);
        let (mixer_synths, mixer_slot_owner) = new_synth_set(mixer_rate);
        Some(SynthController {
            sample_rate,
            player,
            synths,
            slot_owner,
            mixer_synths,
            mixer_slot_owner,
            bank: Bank::new(mixer_rate),
            notes_on: vec![[0u8; 128]; TRACK_COUNT],
            transition: Transition::new(sample_rate, LoopAndTransitionOptions::none()),
            messages: Vec::new(),
            feedback: TickFeedback::default(),
            events: Vec::new(),
            timer: 0.0,
            shelf_l: BiquadFilter::high_shelf(2, sample_rate, 4000.0, 0.707, 0.0),
            shelf_r: BiquadFilter::high_shelf(2, sample_rate, 4000.0, 0.707, 0.0),
            shelf_params: None,
            psg_comp_l: psg_comp_filter(sample_rate),
            psg_comp_r: psg_comp_filter(sample_rate),
            psg_comp_was_active: false,
        })
    }

    /// Applies the master high-shelf EQ to one final stereo sample (a transparent pass when the
    /// shelf is disabled / 0 dB), reconfiguring the biquads when the parameters change.
    #[inline]
    fn master_filter(&mut self, l: Sample, r: Sample, config: &PerDeviceSettings) -> Frame {
        let hs = config.shelf;
        if !hs.is_active() {
            return (l, r);
        }
        if self.shelf_params != Some(hs) {
            let order = (hs.order.max(2)) & !1; // even, ≥ 2
            if self.shelf_l.num_cascade() * 2 != order {
                self.shelf_l = BiquadFilter::high_shelf(
                    order,
                    self.sample_rate,
                    hs.cutoff_hz,
                    hs.q,
                    hs.gain_db,
                );
                self.shelf_r = BiquadFilter::high_shelf(
                    order,
                    self.sample_rate,
                    hs.cutoff_hz,
                    hs.q,
                    hs.gain_db,
                );
            } else {
                self.shelf_l
                    .set_high_shelf(self.sample_rate, hs.cutoff_hz, hs.q, hs.gain_db);
                self.shelf_r
                    .set_high_shelf(self.sample_rate, hs.cutoff_hz, hs.q, hs.gain_db);
            }
            self.shelf_params = Some(hs);
        }
        (self.shelf_l.transform(l), self.shelf_r.transform(r))
    }

    /// Colours one PSG-bus (output-set) stereo sample with the crunch-compensation low-pass when
    /// active, matching the high-frequency darkening the mixer-to-output crunch gives DirectSound.
    /// A transparent pass otherwise; the biquad state is cleared on the inactive→active edge so a
    /// fresh enable starts clean.
    #[inline]
    fn psg_compensate(&mut self, l: Sample, r: Sample, config: &PerDeviceSettings) -> Frame {
        if !psg_comp_active(config) {
            self.psg_comp_was_active = false;
            return (l, r);
        }
        if !self.psg_comp_was_active {
            self.psg_comp_l.reset_state();
            self.psg_comp_r.reset_state();
            self.psg_comp_was_active = true;
        }
        (self.psg_comp_l.transform(l), self.psg_comp_r.transform(r))
    }

    /// The audio sample rate this controller renders at.
    #[inline]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Total voices sounding across both synthesizer sets (drives the app's DSP-load / voice stats).
    pub fn active_voice_count(&self) -> usize {
        let count = |synths: &[WaveformSynthesizer]| -> usize {
            synths.iter().map(|s| s.active_voice_count()).sum()
        };
        count(&self.synths) + count(&self.mixer_synths)
    }

    /// Sets the loop-count → fade-out policy (re-arming it from scratch: no fade in progress, loop
    /// count reset). See [`LoopAndTransitionOptions`].
    pub fn set_loop_and_transition(&mut self, opts: LoopAndTransitionOptions) {
        self.transition.set_opts(opts);
    }

    /// Begins an immediate fade-out of `fade_seconds` (no grace), overriding any in-progress fade —
    /// the live path's quick fade before a manual song switch. Emits [`PlaybackEvent::Finished`]
    /// once the gain reaches silence, like an auto-triggered fade.
    pub fn request_transition(&mut self, fade_seconds: f64) {
        self.transition.request(fade_seconds, &mut self.messages);
    }

    /// Drains the [`PlaybackEvent`]s pumped since the last call (loops / transition start / finish).
    pub fn take_messages(&mut self) -> std::vec::Drain<'_, PlaybackEvent> {
        self.messages.drain(..)
    }

    /// The current end-of-song fade gain (1.0 until the fade is triggered / during its grace).
    pub fn fade_gain(&self) -> f32 {
        self.transition.gain()
    }

    /// Changes the output sample rate at any time, re-targeting every voice and filter. A no-op
    /// when the rate is unchanged.
    ///
    /// The master clock reads `sample_rate` live (see [`Self::next_sample`]), so only the derived
    /// state needs refreshing: the output set's synthesizers (voices, Haas delays, crossover) and
    /// the master high-shelf, whose biquads are rebuilt against the new rate on next use. The mixer
    /// set runs at its own rate and is independent; the bank picks up the new output rate on the
    /// next render (see [`Self::prepare_mixer`]).
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        if sample_rate == self.sample_rate {
            return;
        }
        self.sample_rate = sample_rate;
        for synth in &mut self.synths {
            synth.set_sample_rate(sample_rate);
        }
        self.shelf_params = None;
        // Rebuild the PSG compensation low-pass so its knee stays at a fixed frequency in Hz.
        self.psg_comp_l = psg_comp_filter(sample_rate);
        self.psg_comp_r = psg_comp_filter(sample_rate);
        self.psg_comp_was_active = false;
    }

    /// Sequencer steps executed (the visualizer timeline position).
    pub fn steps_elapsed(&self) -> u32 {
        self.player.steps_elapsed()
    }

    /// Current sequencer step rate (steps per second at the current tempo).
    pub fn step_rate(&self) -> f64 {
        self.player.step_rate()
    }

    /// The current musical tempo in quarter-note beats per minute, derived from the live step
    /// rate and the device's steps-per-beat. Tracks tempo changes as the song plays.
    pub fn current_bpm(&self) -> f64 {
        let spb = self.player.steps_per_beat();
        if spb > 0.0 {
            self.player.step_rate() * 60.0 / spb
        } else {
            0.0
        }
    }

    /// Reconfigures the mixer bank (and re-rates the mixer set when needed) from `config`, once per
    /// render call.
    fn prepare_mixer(&mut self, config: &PerDeviceSettings) {
        if !config.use_mixer {
            self.bank.disable();
            return;
        }
        if let Some(rate) = self.bank.prepare(config, self.sample_rate) {
            for synth in &mut self.mixer_synths {
                synth.set_sample_rate(rate);
            }
        }
    }

    /// One upsampled stereo sample routed from the mixer set through the bank: the resampler pulls
    /// mixer-rate samples (each a fresh advance + stereo mix of the mixer set) only as its read
    /// window consumes them.
    fn route_mixer(&mut self, config: &PerDeviceSettings) -> Frame {
        let mixer_synths = &mut self.mixer_synths;
        let enables = &config.track_enables;
        let mut render = || render_set(mixer_synths, enables, config);
        self.bank.route(&mut render)
    }

    /// Advances the device master clock and returns one mixed stereo sample.
    ///
    /// This is the single place where the hardware tick math lives: the device clock is
    /// accumulated per output sample and the player is ticked every `cycles_per_tick` cycles.
    pub fn next_sample(&mut self, config: &PerDeviceSettings) -> (f32, f32) {
        self.prepare_mixer(config);
        self.timer += self.player.clock_rate();
        let threshold = self.player.cycles_per_tick() * self.sample_rate;
        while self.timer >= threshold {
            self.timer -= threshold;
            self.tick(config);
        }

        let (val_l, val_r) = render_set(&mut self.synths, &config.track_enables, config);
        // The output set is PSG-only when the mixer is engaged; darken it to match DirectSound.
        let (mut val_l, mut val_r) = self.psg_compensate(val_l, val_r, config);
        if config.use_mixer {
            let (ml, mr) = self.route_mixer(config);
            val_l += ml;
            val_r += mr;
        }
        let (val_l, val_r) = self.master_filter(val_l, val_r, config);
        // The end-of-song fade is applied here, once per output sample, so it lives in one place.
        let g = self.transition.advance(&mut self.messages);
        (val_l * g, val_r * g)
    }

    /// Fills `out` with interleaved stereo (L, R, L, R, …) samples.
    ///
    /// Renders in blocks between device ticks (voice parameters only change on ticks), so each
    /// voice runs one tight loop per block instead of re-deriving its setup per sample. The clock
    /// is advanced with the same per-sample additions as [`Self::next_sample`], so the output is
    /// bit-identical to calling that in a loop.
    pub fn fill(&mut self, out: &mut [f32], config: &PerDeviceSettings) {
        self.prepare_mixer(config);
        let threshold = self.player.cycles_per_tick() * self.sample_rate;
        let clock = self.player.clock_rate();
        let frames = out.len() / 2;
        let mut acc_l: [Sample; MAX_BLOCK] = [0.0; MAX_BLOCK];
        let mut acc_r: [Sample; MAX_BLOCK] = [0.0; MAX_BLOCK];

        let mut frame = 0;
        while frame < frames {
            // First sample of the block: advance the clock and run any due ticks (mirroring
            // `next_sample`'s ordering: the tick fires before the sample is synthesized).
            self.timer += clock;
            while self.timer >= threshold {
                self.timer -= threshold;
                self.tick(config);
            }
            // Extend the block with tick-free samples, advancing the clock identically.
            let max_n = (frames - frame).min(MAX_BLOCK);
            let mut n = 1;
            while n < max_n && self.timer + clock < threshold {
                self.timer += clock;
                n += 1;
            }

            // The output set renders the whole block at once; the mixer set is pulled per output
            // sample (its consumption is data-dependent), matching `next_sample` bit-for-bit.
            acc_l[..n].fill(0.0);
            acc_r[..n].fill(0.0);
            for (synth, &enabled) in self.synths.iter_mut().zip(&config.track_enables) {
                synth.render_block(config, n, &mut acc_l, &mut acc_r, enabled);
            }
            let block_out = &mut out[2 * frame..2 * (frame + n)];
            for (i, frame_out) in block_out.chunks_exact_mut(2).enumerate() {
                // The block-rendered output set is the PSG bus when the mixer is engaged.
                let (mut l, mut r) = self.psg_compensate(acc_l[i], acc_r[i], config);
                if config.use_mixer {
                    let (ml, mr) = self.route_mixer(config);
                    l += ml;
                    r += mr;
                }
                let (l, r) = self.master_filter(l, r, config);
                // Per-sample fade gain (see `next_sample`), keeping the block path bit-identical.
                let g = self.transition.advance(&mut self.messages);
                frame_out[0] = l * g;
                frame_out[1] = r * g;
            }
            frame += n;
        }

        // Odd trailing f32 (half a frame): render one more stereo sample, keep its left channel.
        if out.len() % 2 == 1 {
            let (l, _) = self.next_sample(config);
            out[out.len() - 1] = l;
        }
    }

    /// Renders the isolated sampled (DirectSound/SWAR) mixer bus at the config's `mixer_sample_rate`
    /// into interleaved stereo `out`, with **no** mixer→output resampling and **no** PSG voices.
    ///
    /// This is an offline-analysis hook: it exposes the exact stereo signal the
    /// [`StreamResampler`] consumes in the mixer-to-output stage, so the resampler's transfer
    /// function (e.g. nearest vs output-Nyquist crunch) can be measured on real DirectSound content
    /// in isolation from the PSG path. Requires the config's `use_mixer`; the device clock is run
    /// at the mixer rate so the captured bus is at `mixer_sample_rate`.
    pub fn fill_mixer_bus(&mut self, out: &mut [f32], config: &PerDeviceSettings) {
        assert!(config.use_mixer, "fill_mixer_bus requires config.use_mixer");
        // Retargets the mixer set to `mixer_sample_rate` (the bank's resampler config it also sets
        // is unused here — we read the mixer bus directly).
        self.prepare_mixer(config);
        let rate = f64::from(config.mixer_sample_rate);
        let threshold = self.player.cycles_per_tick() * rate;
        let clock = self.player.clock_rate();
        let frames = out.len() / 2;
        for frame in 0..frames {
            self.timer += clock;
            while self.timer >= threshold {
                self.timer -= threshold;
                self.tick(config);
            }
            let (l, r) = render_set(&mut self.mixer_synths, &config.track_enables, config);
            // Advance the output (PSG) set too so one-shot endings / voice-steal feedback stay
            // consistent with a normal render; its audio is discarded.
            let _ = render_set(&mut self.synths, &config.track_enables, config);
            out[2 * frame] = l;
            out[2 * frame + 1] = r;
        }
    }

    /// One device tick: report synth-side voice endings, advance the device, apply its events.
    pub fn tick(&mut self, config: &PerDeviceSettings) {
        // One-shot samples that ran out are cut here — only the synthesizer knows their playback
        // position — and reported to the device as feedback (both sets).
        cut_finished(
            &mut self.synths,
            &mut self.slot_owner,
            &mut self.notes_on,
            &mut self.feedback,
        );
        cut_finished(
            &mut self.mixer_synths,
            &mut self.mixer_slot_owner,
            &mut self.notes_on,
            &mut self.feedback,
        );

        let mut events = std::mem::take(&mut self.events);
        self.player.tick(&mut self.feedback, config, &mut events);
        for event in events.drain(..) {
            self.apply_event(event, config);
        }
        self.events = events;
    }

    /// Locates the set and pool slot owning device voice `voice` on `track`: `true` ⇒ the mixer
    /// set, `false` ⇒ the output set.
    fn locate(&self, track: usize, voice: VoiceId) -> Option<(bool, usize)> {
        if let Some(slot) = find_slot(&self.slot_owner, track, voice) {
            return Some((false, slot));
        }
        find_slot(&self.mixer_slot_owner, track, voice).map(|slot| (true, slot))
    }

    /// Mutable access to one synthesizer set and its slot bookkeeping: `true` ⇒ mixer, `false` ⇒
    /// output.
    #[inline]
    fn set_mut(&mut self, mixer: bool) -> (&mut Vec<WaveformSynthesizer>, &mut SlotOwners) {
        if mixer {
            (&mut self.mixer_synths, &mut self.mixer_slot_owner)
        } else {
            (&mut self.synths, &mut self.slot_owner)
        }
    }

    /// Applies one standardized device event to the voice pools.
    fn apply_event(&mut self, event: SynthEvent, config: &PerDeviceSettings) {
        match event {
            SynthEvent::NoteStarted {
                track,
                voice,
                key,
                waveform,
                pitch,
                volume,
                duration_ticks: _,
            } => {
                // Sampled (non-PSG) voices play on the mixer set when it's engaged; PSG voices and
                // everything in direct mode play on the output set.
                let mixer = config.use_mixer && !waveform.is_psg_square;
                let (synths, slot_owner) = self.set_mut(mixer);
                // Device volumes arrive in `f64` (dB-domain/square-law envelope math); narrow to the
                // sample width here, at the device→synth boundary.
                let slot = synths[track].play(waveform, pitch, volume as Sample, config);
                if let Some(old) = slot_owner[track][slot].replace(SlotOwner { voice, key }) {
                    // Round-robin steal: the previous occupant is gone; tell the device.
                    self.notes_on[track][old.key as usize] = 0;
                    self.feedback.ended_voices.push((track, old.voice));
                }
                self.notes_on[track][key as usize] = 1;
            }
            SynthEvent::VoiceVolume {
                track,
                voice,
                volume,
            } => {
                if let Some((mixer, slot)) = self.locate(track, voice) {
                    self.set_mut(mixer).0[track].instr_mut(slot).volume = volume as Sample;
                }
            }
            SynthEvent::VoicePitch {
                track,
                voice,
                pitch,
            } => {
                if let Some((mixer, slot)) = self.locate(track, voice) {
                    self.set_mut(mixer).0[track]
                        .instr_mut(slot)
                        .set_pitch(pitch, config.tuning());
                }
            }
            SynthEvent::VoiceDetune {
                track,
                voice,
                semitones,
            } => {
                if let Some((mixer, slot)) = self.locate(track, voice) {
                    self.set_mut(mixer).0[track]
                        .instr_mut(slot)
                        .set_finetune_lfo(semitones, config.tuning());
                }
            }
            SynthEvent::VoiceStopped { track, voice } => {
                if let Some((mixer, slot)) = self.locate(track, voice) {
                    let (synths, slot_owner) = self.set_mut(mixer);
                    let owner = slot_owner[track][slot].take();
                    synths[track].stop_instrument(slot, config.pop_smoothing());
                    if let Some(owner) = owner {
                        self.notes_on[track][owner.key as usize] = 0;
                    }
                }
            }
            SynthEvent::NoteReleased { track, key } => {
                self.notes_on[track][key as usize] = 0;
            }
            // Track-level pan/detune apply to both sets: a track's voices may be split across them.
            SynthEvent::TrackPan {
                track,
                pan_vol_l,
                pan_vol_r,
            } => {
                self.synths[track].set_pan(pan_vol_l, pan_vol_r, config);
                self.mixer_synths[track].set_pan(pan_vol_l, pan_vol_r, config);
            }
            SynthEvent::TrackDetune { track, semitones } => {
                self.synths[track].set_finetune(semitones, config.tuning());
                self.mixer_synths[track].set_finetune(semitones, config.tuning());
            }
            SynthEvent::Looped => self.transition.on_loop(&mut self.messages),
            SynthEvent::Ended => self.transition.on_end(&mut self.messages),
        }
    }
}

#[cfg(test)]
mod transition_tests {
    use super::*;

    const SR: f64 = 32_768.0;

    /// The shared fade ramp reproduces the legacy `1 - (k/sr)/fade` shape, with a full-gain grace
    /// before it, and emits `Finished` exactly once when it reaches silence.
    #[test]
    fn export_ramp_matches_legacy_math() {
        let opts = LoopAndTransitionOptions::export();
        let mut t = Transition::new(SR, opts);
        let mut msgs = Vec::new();

        // Export fades after a single loop: `Looped`, then `TransitionStarted`.
        t.on_loop(&mut msgs);
        assert_eq!(
            msgs,
            [PlaybackEvent::Looped, PlaybackEvent::TransitionStarted]
        );
        msgs.clear();

        let grace = (SR * opts.grace_seconds) as u64; // 65_536
        let fade = (SR * opts.fade_seconds) as u64; // 98_304
        let mut finished_at = None;
        for n in 0..(grace + fade + 4) {
            let g = t.advance(&mut msgs);
            let expected = if n < grace {
                1.0f32
            } else {
                (1.0 - ((n - grace) as f64 / SR) / opts.fade_seconds) as f32
            };
            if g <= 0.0 {
                finished_at.get_or_insert(n);
            } else {
                assert_eq!(g, expected, "gain mismatch at sample {n}");
            }
        }
        assert_eq!(finished_at, Some(grace + fade), "fade ends at grace+fade");
        assert_eq!(
            msgs,
            [PlaybackEvent::Finished],
            "exactly one Finished is pumped"
        );
    }

    /// The `too_long_after_loop_threshold_seconds` cap trips the fade at the next loop once playback
    /// has run past the threshold, even before `loops_before_fade` loops have elapsed.
    #[test]
    fn too_long_after_loop_threshold_trips_before_loop_count() {
        let opts = LoopAndTransitionOptions {
            loops_before_fade: Some(4),
            too_long_after_loop_threshold_seconds: Some(1.0),
            ..LoopAndTransitionOptions::none()
        };
        let mut t = Transition::new(SR, opts);
        let mut msgs = Vec::new();

        // A loop under the threshold: counted, but no fade (1 of 4 loops, < 1 s elapsed).
        t.on_loop(&mut msgs);
        assert_eq!(msgs, [PlaybackEvent::Looped]);
        msgs.clear();

        // Render just over one second of output, then loop again: the length cap trips the fade
        // even though only two of four loops have elapsed.
        for _ in 0..=(SR as u64) {
            t.advance(&mut Vec::new());
        }
        t.on_loop(&mut msgs);
        assert_eq!(
            msgs,
            [PlaybackEvent::Looped, PlaybackEvent::TransitionStarted]
        );
    }

    /// A requested transition fades immediately (no grace) over its own duration.
    #[test]
    fn requested_transition_has_no_grace() {
        let mut t = Transition::new(SR, LoopAndTransitionOptions::none());
        let mut msgs = Vec::new();
        t.request(0.040, &mut msgs);
        assert_eq!(msgs, [PlaybackEvent::TransitionStarted]);
        // First sample is still full gain; it falls immediately after.
        assert_eq!(t.advance(&mut Vec::new()), 1.0);
        assert!(t.advance(&mut Vec::new()) < 1.0);
    }

    /// The default policy never fades, so a plain render is unaffected.
    #[test]
    fn none_policy_never_fades() {
        let mut t = Transition::new(SR, LoopAndTransitionOptions::none());
        let mut msgs = Vec::new();
        t.on_loop(&mut msgs);
        t.on_end(&mut msgs);
        assert_eq!(msgs, [PlaybackEvent::Looped]); // Looped reported, but no transition
        for _ in 0..1000 {
            assert_eq!(t.advance(&mut Vec::new()), 1.0);
        }
    }
}

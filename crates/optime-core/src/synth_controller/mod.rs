//! [`SynthController`]: the device-agnostic synthesis runtime.
//!
//! It owns the per-track voice pools, the master clock, and the note-grid bookkeeping for
//! visualizers. All musical intelligence lives in the device player
//! ([`DevicePlayer`](crate::devices::DevicePlayer)); the controller drives it one tick at a
//! time and applies the standardized [`SynthEvent`] stream it produces to the voices.
//!
//! - [`config`] — the [`SynthConfig`] options struct.
//! - [`vis`] — the look-ahead [`FsVisController`] for visualizers.
//!
//! ## The intermediate mixer
//!
//! The controller owns two sets of per-track synthesizers: the output set, which renders at the
//! output rate, and the mixer set, which renders at [`SynthConfig::mixer_sample_rate`]. With
//! [`SynthConfig::use_mixer`] set, sampled (non-PSG) voices play on the mixer set and PSG voices on
//! the output set; the controller *routes* each set's stereo audio to the final mix — the output
//! set straight through, the mixer set via the [`Bank`], which upsamples the mixer-rate bus to the
//! output rate. With the mixer off, every voice plays on the output set, so the result is
//! bit-identical to the single-set engine. The [`Bank`] holds only the resampler; it does not own
//! the synthesizers — audio flows *from* the synthesizers *to* the bank.

mod config;
pub mod messages;
mod vis;

pub use config::{DelaySmoothing, HighShelf, PopSmoothing, SynthConfig};
pub use vis::{FsVisController, SongOverview, VisNote};

use crate::devices::{DevicePlayer, SoundData, SynthEvent, TickFeedback, VoiceId};
use crate::dsp::biquad_filter::BiquadFilter;
use crate::dsp::resample::StreamResampler;
use crate::sample::InstrumentResampleMode;
use crate::synth::MAX_BLOCK;
use crate::{SampleSynthesizer, TRACK_COUNT};

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
fn psg_comp_active(config: &SynthConfig) -> bool {
    config.psg_crunch_compensation
        && config.use_mixer
        && matches!(
            config.mixer_resample,
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
fn new_synth_set(sample_rate: f64) -> (Vec<SampleSynthesizer>, SlotOwners) {
    let synths: Vec<_> = (0..TRACK_COUNT)
        .map(|_| SampleSynthesizer::new(sample_rate, 16))
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
    synths: &mut [SampleSynthesizer],
    enables: &[bool],
    config: &SynthConfig,
) -> (f64, f64) {
    let (mut l, mut r) = (0.0, 0.0);
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
    synths: &mut [SampleSynthesizer],
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
            let ran_out = !instr.sample.looping && instr.sample_t > instr.sample.data.len() as f64;
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
    fn prepare(&mut self, config: &SynthConfig, out_rate: f64) -> Option<f64> {
        if !self.was_active {
            self.resampler.reset();
            self.was_active = true;
        }
        let rate_change = (self.rate != config.mixer_sample_rate).then(|| {
            self.rate = config.mixer_sample_rate;
            self.rate
        });
        self.resampler
            .set(self.rate, out_rate, config.mixer_resample);
        rate_change
    }

    /// One upsampled stereo sample, pulling the mixer-rate bus from `render` on demand.
    fn route(&mut self, render: &mut impl FnMut() -> (f64, f64)) -> (f64, f64) {
        self.resampler.next(render)
    }
}

/// The synthesis runtime: voice pools + master clock, driven by a device player.
pub struct SynthController {
    sample_rate: f64,
    /// The device player generating the event stream.
    pub player: DevicePlayer,
    /// The output-rate synthesizers (PSG voices, and every voice when the mixer is off).
    synths: Vec<SampleSynthesizer>,
    slot_owner: SlotOwners,
    /// The mixer-rate synthesizers (sampled voices when [`SynthConfig::use_mixer`] is set).
    mixer_synths: Vec<SampleSynthesizer>,
    mixer_slot_owner: SlotOwners,
    /// The mixer bus the mixer set's audio is routed into (owns the resampler, not the synths).
    bank: Bank,
    /// `notes_on[track][note]` is 1 while a sequence note sounds (drives the visualizer).
    pub notes_on: Vec<[u8; 128]>,
    /// Count of sequence loops seen (used by callers to detect loop points).
    pub jumps: u32,
    /// Set when the song has ended and should fade out.
    pub fading_start: bool,
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
    pub fn new(sample_rate: f64, data: &SoundData, song_id: u32) -> Option<SynthController> {
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
            jumps: 0,
            fading_start: false,
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
    fn master_filter(&mut self, l: f64, r: f64, config: &SynthConfig) -> (f64, f64) {
        let hs = config.high_shelf;
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
    fn psg_compensate(&mut self, l: f64, r: f64, config: &SynthConfig) -> (f64, f64) {
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
        let count = |synths: &[SampleSynthesizer]| -> usize {
            synths.iter().map(|s| s.active_voice_count()).sum()
        };
        count(&self.synths) + count(&self.mixer_synths)
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
    fn prepare_mixer(&mut self, config: &SynthConfig) {
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
    fn route_mixer(&mut self, config: &SynthConfig) -> (f64, f64) {
        let mixer_synths = &mut self.mixer_synths;
        let enables = &config.track_enables;
        let mut render = || render_set(mixer_synths, enables, config);
        self.bank.route(&mut render)
    }

    /// Advances the device master clock and returns one mixed stereo sample.
    ///
    /// This is the single place where the hardware tick math lives: the device clock is
    /// accumulated per output sample and the player is ticked every `cycles_per_tick` cycles.
    pub fn next_sample(&mut self, config: &SynthConfig) -> (f32, f32) {
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
        (val_l as f32, val_r as f32)
    }

    /// Fills `out` with interleaved stereo (L, R, L, R, …) samples.
    ///
    /// Renders in blocks between device ticks (voice parameters only change on ticks), so each
    /// voice runs one tight loop per block instead of re-deriving its setup per sample. The clock
    /// is advanced with the same per-sample additions as [`Self::next_sample`], so the output is
    /// bit-identical to calling that in a loop.
    pub fn fill(&mut self, out: &mut [f32], config: &SynthConfig) {
        self.prepare_mixer(config);
        let threshold = self.player.cycles_per_tick() * self.sample_rate;
        let clock = self.player.clock_rate();
        let frames = out.len() / 2;
        let mut acc_l = [0.0f64; MAX_BLOCK];
        let mut acc_r = [0.0f64; MAX_BLOCK];

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
                frame_out[0] = l as f32;
                frame_out[1] = r as f32;
            }
            frame += n;
        }

        // Odd trailing f32 (half a frame): render one more stereo sample, keep its left channel.
        if out.len() % 2 == 1 {
            let (l, _) = self.next_sample(config);
            out[out.len() - 1] = l;
        }
    }

    /// Renders the isolated sampled (DirectSound/SWAR) mixer bus at
    /// [`SynthConfig::mixer_sample_rate`] into interleaved stereo `out`, with **no** mixer→output
    /// resampling and **no** PSG voices.
    ///
    /// This is an offline-analysis hook: it exposes the exact stereo signal the
    /// [`StreamResampler`] consumes in the mixer-to-output stage, so the resampler's transfer
    /// function (e.g. nearest vs output-Nyquist crunch) can be measured on real DirectSound content
    /// in isolation from the PSG path. Requires [`SynthConfig::use_mixer`]; the device clock is run
    /// at the mixer rate so the captured bus is at `mixer_sample_rate`.
    pub fn fill_mixer_bus(&mut self, out: &mut [f32], config: &SynthConfig) {
        assert!(config.use_mixer, "fill_mixer_bus requires config.use_mixer");
        // Retargets the mixer set to `mixer_sample_rate` (the bank's resampler config it also sets
        // is unused here — we read the mixer bus directly).
        self.prepare_mixer(config);
        let rate = config.mixer_sample_rate;
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
            out[2 * frame] = l as f32;
            out[2 * frame + 1] = r as f32;
        }
    }

    /// One device tick: report synth-side voice endings, advance the device, apply its events.
    pub fn tick(&mut self, config: &SynthConfig) {
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
    fn set_mut(&mut self, mixer: bool) -> (&mut Vec<SampleSynthesizer>, &mut SlotOwners) {
        if mixer {
            (&mut self.mixer_synths, &mut self.mixer_slot_owner)
        } else {
            (&mut self.synths, &mut self.slot_owner)
        }
    }

    /// Applies one standardized device event to the voice pools.
    fn apply_event(&mut self, event: SynthEvent, config: &SynthConfig) {
        match event {
            SynthEvent::NoteStarted {
                track,
                voice,
                key,
                sample,
                pitch,
                volume,
                duration_ticks: _,
            } => {
                // Sampled (non-PSG) voices play on the mixer set when it's engaged; PSG voices and
                // everything in direct mode play on the output set.
                let mixer = config.use_mixer && !sample.is_psg_square;
                let (synths, slot_owner) = self.set_mut(mixer);
                let slot = synths[track].play(sample, pitch, volume, config);
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
                    self.set_mut(mixer).0[track].instr_mut(slot).volume = volume;
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
                        .set_pitch(pitch, config.tuning);
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
                        .set_finetune_lfo(semitones, config.tuning);
                }
            }
            SynthEvent::VoiceStopped { track, voice } => {
                if let Some((mixer, slot)) = self.locate(track, voice) {
                    let (synths, slot_owner) = self.set_mut(mixer);
                    let owner = slot_owner[track][slot].take();
                    synths[track].stop_instrument(slot, config.pop_smoothing);
                    if let Some(owner) = owner {
                        self.notes_on[track][owner.key as usize] = 0;
                    }
                }
            }
            SynthEvent::NoteReleased { track, key } => {
                self.notes_on[track][key as usize] = 0;
            }
            // Track-level pan/detune apply to both sets: a track's voices may be split across them.
            SynthEvent::TrackPan { track, pan } => {
                self.synths[track].set_pan(pan, config);
                self.mixer_synths[track].set_pan(pan, config);
            }
            SynthEvent::TrackDetune { track, semitones } => {
                self.synths[track].set_finetune(semitones, config.tuning);
                self.mixer_synths[track].set_finetune(semitones, config.tuning);
            }
            SynthEvent::Looped => self.jumps += 1,
            SynthEvent::Ended => self.fading_start = true,
        }
    }
}

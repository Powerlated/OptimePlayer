//! The synth controller: drives the device player on the master clock and renders its events through the signal chain.

mod config;
pub mod messages;
mod reverb;
mod vis;

pub use config::{
    DEFAULT_POP_SLEW_SECONDS, DelaySmoothing, Exciter, HighBandCompressor, HighShelf, PopSmoothing,
};
pub use vis::{FsVisController, SongOverview, VisNote};

use crate::devices::{DevicePlayer, SoundData, SynthEvent, TickFeedback, VoiceId};
use crate::dsp::biquad_filter::BiquadFilter;
use crate::dsp::block::{self, MAX_BLOCK};
use crate::dsp::exciter::ExciterStage;
use crate::dsp::high_band_compressor::HighBandCompressorStage;
use crate::dsp::resample::{DefaultResampler, Resampler, StreamResampler};
use crate::waveform::{InstrumentResampleMode, Sample};
use crate::{PerDeviceSettings, TRACK_COUNT, WaveformSynthesizer};
use reverb::Reverb;

const PSG_COMP_ORDER: usize = 6;
const PSG_COMP_CUTOFF_HZ: f64 = 14_534.8;
const PSG_COMP_Q: f64 = 0.707;

fn psg_comp_filter(sample_rate: f64) -> BiquadFilter {
    BiquadFilter::low_pass(PSG_COMP_ORDER, sample_rate, PSG_COMP_CUTOFF_HZ, PSG_COMP_Q)
}

fn psg_comp_active(config: &PerDeviceSettings) -> bool {
    config.psg_crunch_compensation
        && config.use_mixer
        && matches!(
            config.mixer_resample_mode(),
            InstrumentResampleMode::SincOutputNyquist { .. }
        )
}

#[derive(Debug, Clone, Copy)]
struct SlotOwner {
    voice: VoiceId,
    key: u8,
}

type SlotOwners = Vec<Vec<Option<SlotOwner>>>;

fn new_synth_set<R: Resampler>(sample_rate: f64) -> (Vec<WaveformSynthesizer<R>>, SlotOwners) {
    let synths: Vec<_> = (0..TRACK_COUNT)
        .map(|_| WaveformSynthesizer::with_resampler(sample_rate, 16))
        .collect();
    let slot_owner = synths.iter().map(|s| vec![None; s.voice_count()]).collect();
    (synths, slot_owner)
}

fn find_slot(slot_owner: &SlotOwners, track: usize, voice: VoiceId) -> Option<usize> {
    slot_owner[track]
        .iter()
        .position(|o| o.is_some_and(|o| o.voice == voice))
}

fn render_set_block<R: Resampler>(
    synths: &mut [WaveformSynthesizer<R>],
    enables: &[bool],
    config: &PerDeviceSettings,
    out_l: &mut [Sample],
    out_r: &mut [Sample],
) {
    let n = block::stereo_len(out_l, out_r);
    out_l.fill(0.0);
    out_r.fill(0.0);
    for (synth, &enabled) in synths.iter_mut().zip(enables) {
        synth.render_block(config, n, out_l, out_r, enabled);
    }
}

fn bitcrush_sample(x: Sample, bits: u32) -> Sample {
    let bits = bits.clamp(1, 16);
    let scale = (1i64 << (bits - 1)) as Sample;
    let code = (x * scale).floor() as i64;
    let sign = 1i64 << (bits - 1);
    let mask = (1i64 << bits) - 1;
    let wrapped = ((code & mask) ^ sign) - sign;
    wrapped as Sample / scale
}

fn bitcrush_block(block: &mut [Sample], bits: u32) {
    for x in block.iter_mut() {
        *x = bitcrush_sample(*x, bits);
    }
}

fn cut_finished<R: Resampler>(
    synths: &mut [WaveformSynthesizer<R>],
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

struct ChainScratch {
    acc_l: [Sample; MAX_BLOCK],
    acc_r: [Sample; MAX_BLOCK],
    mix_l: [Sample; MAX_BLOCK],
    mix_r: [Sample; MAX_BLOCK],
    high_l: [Sample; MAX_BLOCK],
    high_r: [Sample; MAX_BLOCK],
    gain: [f32; MAX_BLOCK],
}

impl ChainScratch {
    fn new() -> Self {
        Self {
            acc_l: [0.0; MAX_BLOCK],
            acc_r: [0.0; MAX_BLOCK],
            mix_l: [0.0; MAX_BLOCK],
            mix_r: [0.0; MAX_BLOCK],
            high_l: [0.0; MAX_BLOCK],
            high_r: [0.0; MAX_BLOCK],
            gain: [0.0; MAX_BLOCK],
        }
    }
}

struct Bank<R: Resampler> {
    resampler: StreamResampler<R>,
    rate: f64,
    was_active: bool,
}

impl<R: Resampler> Bank<R> {
    fn new(rate: f64) -> Self {
        Self {
            resampler: StreamResampler::with_resampler(),
            rate,
            was_active: false,
        }
    }

    fn disable(&mut self) {
        self.was_active = false;
    }

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
        self.resampler.set(
            self.rate as f32,
            out_rate as f32,
            config.mixer_resample_mode(),
        );
        rate_change
    }

    fn route_block(
        &mut self,
        out_l: &mut [Sample],
        out_r: &mut [Sample],
        render: &mut impl FnMut(&mut [Sample], &mut [Sample]),
    ) {
        self.resampler.process(out_l, out_r, render);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoopAndTransitionOptions {
    pub loops_before_fade: Option<u32>,
    pub too_long_after_loop_threshold_seconds: Option<f64>,
    pub fade_on_end: bool,
    pub grace_seconds: f64,
    pub fade_seconds: f64,
}

impl LoopAndTransitionOptions {
    pub const fn none() -> Self {
        Self {
            loops_before_fade: None,
            too_long_after_loop_threshold_seconds: None,
            fade_on_end: false,
            grace_seconds: 0.0,
            fade_seconds: 0.0,
        }
    }

    pub const fn export() -> Self {
        Self {
            loops_before_fade: Some(0),
            too_long_after_loop_threshold_seconds: Some(90.0),
            fade_on_end: true,
            grace_seconds: 2.0,
            fade_seconds: 3.0,
        }
    }

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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PlaybackEvent {
    Looped,
    TransitionStarted,
    Finished,
}

struct Transition {
    opts: LoopAndTransitionOptions,
    sample_rate: f64,
    loops_seen: u32,
    total_samples: u64,
    elapsed: Option<u64>,
    grace_samples: u64,
    fade_seconds: f64,
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

    fn set_opts(&mut self, opts: LoopAndTransitionOptions) {
        *self = Self::new(self.sample_rate, opts);
    }

    fn trip(&mut self, messages: &mut Vec<PlaybackEvent>) {
        if self.elapsed.is_none() {
            self.elapsed = Some(0);
            messages.push(PlaybackEvent::TransitionStarted);
        }
    }

    fn request(&mut self, fade_seconds: f64, messages: &mut Vec<PlaybackEvent>) {
        self.grace_samples = 0;
        self.fade_seconds = fade_seconds;
        self.elapsed = Some(0);
        self.finished = false;
        messages.push(PlaybackEvent::TransitionStarted);
    }

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

    fn on_end(&mut self, messages: &mut Vec<PlaybackEvent>) {
        if self.opts.fade_on_end {
            self.trip(messages);
        }
    }

    fn gain(&self) -> f32 {
        match self.elapsed {
            None => 1.0,
            Some(n) => {
                if n < self.grace_samples {
                    1.0
                } else {
                    let k = (n - self.grace_samples) as f64;
                    (1.0 - (k / self.sample_rate) / self.fade_seconds) as f32
                }
            }
        }
    }

    fn advance_block(&mut self, out: &mut [f32], messages: &mut Vec<PlaybackEvent>) {
        self.total_samples += out.len() as u64;
        if self.elapsed.is_none() {
            out.fill(1.0);
            return;
        }
        for slot in out.iter_mut() {
            let g = self.gain();
            if g <= 0.0 {
                if !self.finished {
                    self.finished = true;
                    messages.push(PlaybackEvent::Finished);
                }
                *slot = 0.0;
                continue;
            }
            self.elapsed = self.elapsed.map(|n| n + 1);
            *slot = g;
        }
    }

    #[cfg(test)]
    fn advance(&mut self, messages: &mut Vec<PlaybackEvent>) -> f32 {
        let mut out = [0.0];
        self.advance_block(&mut out, messages);
        out[0]
    }
}

pub struct SynthController<R: Resampler = DefaultResampler> {
    sample_rate: f64,
    pub player: Box<dyn DevicePlayer>,
    synths: Vec<WaveformSynthesizer<R>>,
    slot_owner: SlotOwners,
    mixer_synths: Vec<WaveformSynthesizer<R>>,
    mixer_slot_owner: SlotOwners,
    bank: Bank<R>,
    pub notes_on: Vec<[u8; 128]>,
    transition: Transition,
    messages: Vec<PlaybackEvent>,
    feedback: TickFeedback,
    events: Vec<SynthEvent>,
    timer: f64,
    shelf_l: BiquadFilter,
    shelf_r: BiquadFilter,
    shelf_params: Option<HighShelf>,
    psg_comp_l: BiquadFilter,
    psg_comp_r: BiquadFilter,
    psg_comp_was_active: bool,
    high_comp_psg: HighBandCompressorStage,
    high_comp_sampled: HighBandCompressorStage,
    high_comp_psg_was_active: bool,
    high_comp_sampled_was_active: bool,
    exciter: ExciterStage,
    exciter_was_active: bool,
    reverb: Reverb,
}

impl SynthController<DefaultResampler> {
    pub fn new(sample_rate: f64, data: &dyn SoundData, song_id: u32) -> Option<Self> {
        Self::with_resampler(sample_rate, data, song_id)
    }
}

impl<R: Resampler> SynthController<R> {
    pub fn with_resampler(sample_rate: f64, data: &dyn SoundData, song_id: u32) -> Option<Self> {
        let player = data.make_player(song_id)?;
        let mixer_rate = 48_000.0;
        let (synths, slot_owner) = new_synth_set(sample_rate);
        let (mixer_synths, mixer_slot_owner) = new_synth_set(mixer_rate);
        Some(Self {
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
            high_comp_psg: HighBandCompressorStage::new(sample_rate),
            high_comp_sampled: HighBandCompressorStage::new(sample_rate),
            high_comp_psg_was_active: false,
            high_comp_sampled_was_active: false,
            exciter: ExciterStage::new(sample_rate),
            exciter_was_active: false,
            reverb: Reverb::new(),
        })
    }

    fn master_filter_block(
        &mut self,
        l: &mut [Sample],
        r: &mut [Sample],
        config: &PerDeviceSettings,
    ) {
        let hs = config.shelf;
        if !hs.is_active() {
            return;
        }
        if self.shelf_params != Some(hs) {
            let order = (hs.order.max(2)) & !1;
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
        self.shelf_l.transform_block(l);
        self.shelf_r.transform_block(r);
    }

    fn psg_compensate_block(
        &mut self,
        l: &mut [Sample],
        r: &mut [Sample],
        config: &PerDeviceSettings,
    ) {
        if !psg_comp_active(config) {
            self.psg_comp_was_active = false;
            return;
        }
        if !self.psg_comp_was_active {
            self.psg_comp_l.reset_state();
            self.psg_comp_r.reset_state();
            self.psg_comp_was_active = true;
        }
        self.psg_comp_l.transform_block(l);
        self.psg_comp_r.transform_block(r);
    }

    fn compress_psg_high_band_block(
        &mut self,
        l: &mut [Sample],
        r: &mut [Sample],
        config: &PerDeviceSettings,
        high_l: &mut [Sample],
        high_r: &mut [Sample],
    ) {
        let hbc = &config.high_band_compress;
        if !config.use_mixer || !hbc.is_active_psg() {
            self.high_comp_psg_was_active = false;
            return;
        }
        if !self.high_comp_psg_was_active {
            self.high_comp_psg.reset_state();
            self.high_comp_psg_was_active = true;
        }
        self.high_comp_psg
            .process_block(l, r, hbc.params(), high_l, high_r);
    }

    fn compress_sampled_high_band_block(
        &mut self,
        l: &mut [Sample],
        r: &mut [Sample],
        config: &PerDeviceSettings,
        high_l: &mut [Sample],
        high_r: &mut [Sample],
    ) {
        let hbc = &config.high_band_compress;
        if !config.use_mixer || !hbc.is_active_sampled() {
            self.high_comp_sampled_was_active = false;
            return;
        }
        if !self.high_comp_sampled_was_active {
            self.high_comp_sampled.reset_state();
            self.high_comp_sampled_was_active = true;
        }
        self.high_comp_sampled
            .process_block(l, r, hbc.params(), high_l, high_r);
    }

    fn excite_block(
        &mut self,
        l: &mut [Sample],
        r: &mut [Sample],
        config: &PerDeviceSettings,
        high_l: &mut [Sample],
        high_r: &mut [Sample],
    ) {
        if !config.exciter.is_active() {
            self.exciter_was_active = false;
            return;
        }
        if !self.exciter_was_active {
            self.exciter.reset_state();
            self.exciter_was_active = true;
        }
        self.exciter
            .process_block(l, r, config.exciter.params(), high_l, high_r);
    }

    #[inline]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    pub fn high_band_gr_db(&self) -> f64 {
        let psg = if self.high_comp_psg_was_active {
            self.high_comp_psg.last_gr_db()
        } else {
            0.0
        };
        let sampled = if self.high_comp_sampled_was_active {
            self.high_comp_sampled.last_gr_db()
        } else {
            0.0
        };
        psg.min(sampled)
    }

    pub fn active_voice_count(&self) -> usize {
        let count = |synths: &[WaveformSynthesizer<R>]| -> usize {
            synths.iter().map(|s| s.active_voice_count()).sum()
        };
        count(&self.synths) + count(&self.mixer_synths)
    }

    pub fn set_loop_and_transition(&mut self, opts: LoopAndTransitionOptions) {
        self.transition.set_opts(opts);
    }

    pub fn request_transition(&mut self, fade_seconds: f64) {
        self.transition.request(fade_seconds, &mut self.messages);
    }

    pub fn take_messages(&mut self) -> std::vec::Drain<'_, PlaybackEvent> {
        self.messages.drain(..)
    }

    pub fn fade_gain(&self) -> f32 {
        self.transition.gain()
    }

    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        if sample_rate == self.sample_rate {
            return;
        }
        self.sample_rate = sample_rate;
        for synth in &mut self.synths {
            synth.set_sample_rate(sample_rate);
        }
        self.shelf_params = None;
        self.psg_comp_l = psg_comp_filter(sample_rate);
        self.psg_comp_r = psg_comp_filter(sample_rate);
        self.psg_comp_was_active = false;
        self.high_comp_psg.set_sample_rate(sample_rate);
        self.high_comp_psg_was_active = false;
        self.high_comp_sampled.set_sample_rate(sample_rate);
        self.high_comp_sampled_was_active = false;
        self.exciter.set_sample_rate(sample_rate);
        self.exciter_was_active = false;
    }

    pub fn steps_elapsed(&self) -> u32 {
        self.player.steps_elapsed()
    }

    pub fn step_rate(&self) -> f64 {
        self.player.step_rate()
    }

    pub fn current_bpm(&self) -> f64 {
        let spb = self.player.steps_per_beat();
        if spb > 0.0 {
            self.player.step_rate() * 60.0 / spb
        } else {
            0.0
        }
    }

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
        self.reverb.set_rate(self.bank.rate);
    }

    fn route_mixer_block(
        &mut self,
        out_l: &mut [Sample],
        out_r: &mut [Sample],
        config: &PerDeviceSettings,
    ) {
        let mixer_synths = &mut self.mixer_synths;
        let reverb = &mut self.reverb;
        let enables = &config.track_enables;
        let reverb_on = config.mp2k_reverb;
        let bitcrush = config.bitcrush_mixer.then_some(config.bitcrush_bits);
        let mut render = |l: &mut [Sample], r: &mut [Sample]| {
            render_set_block(mixer_synths, enables, config, l, r);
            reverb.process_block(l, r, reverb_on);
            if let Some(bits) = bitcrush {
                bitcrush_block(l, bits);
                bitcrush_block(r, bits);
            }
        };
        self.bank.route_block(out_l, out_r, &mut render);
    }

    pub fn render(
        &mut self,
        out_l: &mut [Sample],
        out_r: &mut [Sample],
        config: &PerDeviceSettings,
    ) {
        debug_assert_eq!(out_l.len(), out_r.len());
        let frames = out_l.len();
        self.prepare_mixer(config);
        let threshold = self.player.cycles_per_tick() * self.sample_rate;
        let clock = self.player.clock_rate();
        let mut scratch = ChainScratch::new();

        let mut frame = 0;
        while frame < frames {
            self.timer += clock;
            while self.timer >= threshold {
                self.timer -= threshold;
                self.tick(config);
            }
            let max_n = (frames - frame).min(MAX_BLOCK);
            let mut n = 1;
            while n < max_n && self.timer + clock < threshold {
                self.timer += clock;
                n += 1;
            }

            let (acc_l, acc_r) = (&mut scratch.acc_l[..n], &mut scratch.acc_r[..n]);
            render_set_block(
                &mut self.synths,
                &config.track_enables,
                config,
                acc_l,
                acc_r,
            );

            self.psg_compensate_block(acc_l, acc_r, config);
            let (high_l, high_r) = (&mut scratch.high_l, &mut scratch.high_r);
            self.compress_psg_high_band_block(acc_l, acc_r, config, high_l, high_r);

            if config.use_mixer {
                let (mix_l, mix_r) = (&mut scratch.mix_l[..n], &mut scratch.mix_r[..n]);
                self.route_mixer_block(mix_l, mix_r, config);
                self.compress_sampled_high_band_block(mix_l, mix_r, config, high_l, high_r);
                for (acc, &m) in acc_l.iter_mut().zip(mix_l.iter()) {
                    *acc += m;
                }
                for (acc, &m) in acc_r.iter_mut().zip(mix_r.iter()) {
                    *acc += m;
                }
            }
            self.excite_block(acc_l, acc_r, config, high_l, high_r);
            self.master_filter_block(acc_l, acc_r, config);

            let gain = &mut scratch.gain[..n];
            self.transition.advance_block(gain, &mut self.messages);
            for ((o, &v), &g) in out_l[frame..].iter_mut().zip(acc_l.iter()).zip(gain.iter()) {
                *o = v * g;
            }
            for ((o, &v), &g) in out_r[frame..].iter_mut().zip(acc_r.iter()).zip(gain.iter()) {
                *o = v * g;
            }
            frame += n;
        }
    }

    pub fn next_sample(&mut self, config: &PerDeviceSettings) -> (f32, f32) {
        let (mut l, mut r) = ([0.0], [0.0]);
        self.render(&mut l, &mut r, config);
        (l[0], r[0])
    }

    pub fn fill(&mut self, out: &mut [f32], config: &PerDeviceSettings) {
        let mut buf_l: [Sample; MAX_BLOCK] = [0.0; MAX_BLOCK];
        let mut buf_r: [Sample; MAX_BLOCK] = [0.0; MAX_BLOCK];
        for chunk in out.chunks_mut(2 * MAX_BLOCK) {
            let n = chunk.len() / 2;
            self.render(&mut buf_l[..n], &mut buf_r[..n], config);
            for ((frame_out, &l), &r) in chunk.chunks_exact_mut(2).zip(&buf_l[..n]).zip(&buf_r[..n])
            {
                frame_out[0] = l;
                frame_out[1] = r;
            }
            if chunk.len() % 2 == 1 {
                let (l, _) = self.next_sample(config);
                *chunk.last_mut().expect("odd-length chunk is non-empty") = l;
            }
        }
    }

    pub fn fill_mixer_bus(&mut self, out: &mut [f32], config: &PerDeviceSettings) {
        assert!(config.use_mixer, "fill_mixer_bus requires config.use_mixer");
        self.prepare_mixer(config);
        let rate = f64::from(config.mixer_sample_rate);
        let threshold = self.player.cycles_per_tick() * rate;
        let clock = self.player.clock_rate();
        let frames = out.len() / 2;
        let (mut bus_l, mut bus_r): ([Sample; MAX_BLOCK], [Sample; MAX_BLOCK]) =
            ([0.0; MAX_BLOCK], [0.0; MAX_BLOCK]);
        let (mut sink_l, mut sink_r): ([Sample; MAX_BLOCK], [Sample; MAX_BLOCK]) =
            ([0.0; MAX_BLOCK], [0.0; MAX_BLOCK]);

        let mut frame = 0;
        while frame < frames {
            self.timer += clock;
            while self.timer >= threshold {
                self.timer -= threshold;
                self.tick(config);
            }
            let max_n = (frames - frame).min(MAX_BLOCK);
            let mut n = 1;
            while n < max_n && self.timer + clock < threshold {
                self.timer += clock;
                n += 1;
            }

            let enables = &config.track_enables;
            render_set_block(
                &mut self.mixer_synths,
                enables,
                config,
                &mut bus_l[..n],
                &mut bus_r[..n],
            );
            render_set_block(
                &mut self.synths,
                enables,
                config,
                &mut sink_l[..n],
                &mut sink_r[..n],
            );
            for ((frame_out, &l), &r) in out[2 * frame..2 * (frame + n)]
                .chunks_exact_mut(2)
                .zip(&bus_l[..n])
                .zip(&bus_r[..n])
            {
                frame_out[0] = l;
                frame_out[1] = r;
            }
            frame += n;
        }
    }

    pub fn tick(&mut self, config: &PerDeviceSettings) {
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

    fn locate(&self, track: usize, voice: VoiceId) -> Option<(bool, usize)> {
        if let Some(slot) = find_slot(&self.slot_owner, track, voice) {
            return Some((false, slot));
        }
        find_slot(&self.mixer_slot_owner, track, voice).map(|slot| (true, slot))
    }

    #[inline]
    fn set_mut(&mut self, mixer: bool) -> (&mut Vec<WaveformSynthesizer<R>>, &mut SlotOwners) {
        if mixer {
            (&mut self.mixer_synths, &mut self.mixer_slot_owner)
        } else {
            (&mut self.synths, &mut self.slot_owner)
        }
    }

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
                let mixer = config.use_mixer && !waveform.is_psg_square;
                let (synths, slot_owner) = self.set_mut(mixer);
                let slot = synths[track].play(waveform, pitch, volume as Sample, config);
                if let Some(old) = slot_owner[track][slot].replace(SlotOwner { voice, key }) {
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
            SynthEvent::ReverbAmount { amount } => self.reverb.set_amount(amount),
            SynthEvent::Looped => self.transition.on_loop(&mut self.messages),
            SynthEvent::Ended => self.transition.on_end(&mut self.messages),
        }
    }
}

#[cfg(test)]
mod transition_tests {
    use super::*;

    const SR: f64 = 32_768.0;

    #[test]
    fn export_ramp_matches_legacy_math() {
        let opts = LoopAndTransitionOptions::export();
        let mut t = Transition::new(SR, opts);
        let mut msgs = Vec::new();

        t.on_loop(&mut msgs);
        assert_eq!(
            msgs,
            [PlaybackEvent::Looped, PlaybackEvent::TransitionStarted]
        );
        msgs.clear();

        let grace = (SR * opts.grace_seconds) as u64;
        let fade = (SR * opts.fade_seconds) as u64;
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

    #[test]
    fn too_long_after_loop_threshold_trips_before_loop_count() {
        let opts = LoopAndTransitionOptions {
            loops_before_fade: Some(4),
            too_long_after_loop_threshold_seconds: Some(1.0),
            ..LoopAndTransitionOptions::none()
        };
        let mut t = Transition::new(SR, opts);
        let mut msgs = Vec::new();

        t.on_loop(&mut msgs);
        assert_eq!(msgs, [PlaybackEvent::Looped]);
        msgs.clear();

        for _ in 0..=(SR as u64) {
            t.advance(&mut Vec::new());
        }
        t.on_loop(&mut msgs);
        assert_eq!(
            msgs,
            [PlaybackEvent::Looped, PlaybackEvent::TransitionStarted]
        );
    }

    #[test]
    fn requested_transition_has_no_grace() {
        let mut t = Transition::new(SR, LoopAndTransitionOptions::none());
        let mut msgs = Vec::new();
        t.request(0.040, &mut msgs);
        assert_eq!(msgs, [PlaybackEvent::TransitionStarted]);
        assert_eq!(t.advance(&mut Vec::new()), 1.0);
        assert!(t.advance(&mut Vec::new()) < 1.0);
    }

    #[test]
    fn none_policy_never_fades() {
        let mut t = Transition::new(SR, LoopAndTransitionOptions::none());
        let mut msgs = Vec::new();
        t.on_loop(&mut msgs);
        t.on_end(&mut msgs);
        assert_eq!(msgs, [PlaybackEvent::Looped]);
        for _ in 0..1000 {
            assert_eq!(t.advance(&mut Vec::new()), 1.0);
        }
    }
}

#[cfg(test)]
mod bitcrush_tests {
    use super::bitcrush_sample;

    fn m4a_8bit(x: f32) -> f32 {
        let code = (x * 128.0).floor() as i64;
        f32::from(code as i8) / 128.0
    }

    #[test]
    fn truncates_to_signed_levels() {
        assert_eq!(bitcrush_sample(0.0, 8), 0.0);
        assert_eq!(bitcrush_sample(0.5, 8), 0.5);
        assert_eq!(bitcrush_sample(0.5 + 0.9 / 128.0, 8), 0.5);
        assert_eq!(bitcrush_sample(-0.001, 8), -1.0 / 128.0);
    }

    #[test]
    fn wraps_on_overflow_like_hardware() {
        assert_eq!(bitcrush_sample(1.0, 8), -1.0);
        assert_eq!(bitcrush_sample(1.5, 8), -0.5);
        assert_eq!(bitcrush_sample(2.0, 8), 0.0);
        assert_eq!(bitcrush_sample(-1.0, 8), -1.0);
        assert_eq!(bitcrush_sample(-1.5, 8), 0.5);
    }

    #[test]
    fn matches_m4a_8bit_oracle_across_range() {
        for i in -400..=400 {
            let x = i as f32 / 128.0;
            assert_eq!(bitcrush_sample(x, 8), m4a_8bit(x), "x = {x}");
        }
    }

    #[test]
    fn lower_depth_truncates_on_a_coarser_grid() {
        assert_eq!(bitcrush_sample(0.1, 4), 0.0);
        assert_eq!(bitcrush_sample(0.3, 4), 0.25);
    }
}

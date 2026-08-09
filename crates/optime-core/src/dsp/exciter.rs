//! The top-end exciter: generates harmonics above a crossover by driving that band through a
//! saturating waveshaper, and adds only what the shaper invented back into the signal.
//!
//! It exists to replace an excitation the chain used to get for free. Reconstructing the
//! intermediate mixer bus as a staircase throws images above that bus's Nyquist, and those images
//! are what gave the sampled voices their air — bought at the price of a `sinc` null at the mixer
//! rate and a great deal of harshness nobody chose. Reconstructing cleanly removes both and leaves
//! the top octave empty; this stage refills it deliberately. That is why the caller runs it on the
//! sampled bus alone: PSG voices bypass the mixer entirely, so they never had staircase images to
//! replace, and exciting them would be adding harshness rather than trading it away.
//!
//! The topology is additive rather than a band replacement. The high band is taken with a
//! high-pass, driven through a saturating curve, and what the drive added to it — the harmonics and
//! nothing else — is high-passed again and summed into the dry input at `amount`. The dry path is
//! never split, so there is no crossover-summing error to answer for.
//!
//! The curve is `tanh(drive·(x + bias))`, offset so it still passes through the origin. Plain `tanh`
//! is an odd function, and an odd nonlinearity produces only odd harmonics — the third and fifth,
//! which sit a musical twelfth and two octaves plus a third above the fundamental and are the
//! harmonics ears read as harsh. Displacing the input along the curve makes it asymmetric, and
//! asymmetry is what produces even harmonics: the second is an octave, which reads as loudness
//! rather than grit. `bias` is therefore the knob that trades harshness for warmth at equal
//! brightness, and it exists because a tuner asked to add air with `bias` fixed at zero can only
//! add the harsh kind.
//!
//! A waveshaper applied sample by sample is a memoryless nonlinearity, and the harmonics it creates
//! above Nyquist fold back as inharmonic aliases that no later filter can separate. This stage
//! suppresses them with first-order antiderivative antialiasing: instead of evaluating the shaper
//! at each sample, it evaluates the shaper's antiderivative at consecutive samples and divides by
//! their difference, which is the average of the shaper over the segment between them rather than
//! its value at a point. That average is what a band-limited reconstruction of the shaped signal
//! would have produced, so the aliases arrive attenuated instead of at full strength. When two
//! consecutive samples are too close for that quotient to survive cancellation, the shaper is
//! evaluated at their midpoint, which is the limit the quotient is approaching; the quotient runs
//! in f64 so that guard can sit low enough to never bite audibly.
//!
//! Averaging is also why the shaper reports a `linear_reference` alongside its output, and why the
//! harmonics are the difference between the two rather than against the band that went in. An
//! antialiased shaper is a half-sample average even when the shaper it is averaging is the
//! identity, so subtracting the undelayed band would leave a first difference — a bright, entirely
//! artificial residual that survives at zero drive and is largest exactly where this stage works.
//! The reference is the same average taken of the identity, so the difference is saturation and
//! nothing else, and both `amount = 0` and `drive → 0` come out transparent.

use crate::dsp::biquad_filter::BiquadFilter;
use crate::dsp::block;
use crate::waveform::{Frame, Sample};

const Q: f64 = core::f64::consts::FRAC_1_SQRT_2;
const SPLIT_ORDER: usize = 4;
const HARMONIC_ORDER: usize = 2;
const ADAA_MIN_STEP: f64 = 1.0e-5;
const SEED_CROSSOVER_HZ: f64 = 3000.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExciterParams {
    pub crossover_hz: f64,
    pub drive: Sample,
    pub bias: Sample,
    pub amount: Sample,
}

const LN_COSH_SERIES_LIMIT: f64 = 1.0;

#[inline]
fn ln_cosh(u: f64) -> f64 {
    let a = u.abs();
    if a < LN_COSH_SERIES_LIMIT {
        let half_sinh = (0.5 * a).sinh();
        (2.0 * half_sinh * half_sinh).ln_1p()
    } else {
        a + (-2.0 * a).exp().ln_1p() - core::f64::consts::LN_2
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Curve {
    drive: f64,
    bias: f64,
    offset: f64,
}

impl Curve {
    fn new(drive: Sample, bias: Sample) -> Self {
        let (drive, bias) = (f64::from(drive), f64::from(bias));
        Self {
            drive,
            bias,
            offset: (bias * drive).tanh() / drive,
        }
    }

    #[inline]
    fn shape(&self, x: f64) -> f64 {
        ((x + self.bias) * self.drive).tanh() / self.drive - self.offset
    }

    #[inline]
    fn antiderivative(&self, x: f64) -> f64 {
        ln_cosh((x + self.bias) * self.drive) / (self.drive * self.drive) - x * self.offset
    }
}

struct AntialiasedShaper {
    curve: Curve,
    prev_x: f64,
    prev_antiderivative: f64,
}

impl AntialiasedShaper {
    fn new(drive: Sample, bias: Sample) -> Self {
        let curve = Curve::new(drive, bias);
        Self {
            curve,
            prev_x: 0.0,
            prev_antiderivative: curve.antiderivative(0.0),
        }
    }

    fn set_curve(&mut self, drive: Sample, bias: Sample) {
        let curve = Curve::new(drive, bias);
        if curve == self.curve {
            return;
        }
        self.curve = curve;
        self.prev_antiderivative = curve.antiderivative(self.prev_x);
    }

    fn reset_state(&mut self) {
        self.prev_x = 0.0;
        self.prev_antiderivative = self.curve.antiderivative(0.0);
    }

    #[inline]
    fn process(&mut self, x: Sample) -> ShapedSample {
        let x = f64::from(x);
        let antiderivative = self.curve.antiderivative(x);
        let step = x - self.prev_x;
        let midpoint = 0.5 * (x + self.prev_x);
        let shaped = if step.abs() > ADAA_MIN_STEP {
            (antiderivative - self.prev_antiderivative) / step
        } else {
            self.curve.shape(midpoint)
        };
        self.prev_x = x;
        self.prev_antiderivative = antiderivative;
        ShapedSample {
            shaped: shaped as Sample,
            linear_reference: midpoint as Sample,
        }
    }
}

struct ShapedSample {
    shaped: Sample,
    linear_reference: Sample,
}

pub struct ExciterStage {
    sample_rate: f64,
    params: Option<ExciterParams>,
    split_l: BiquadFilter,
    split_r: BiquadFilter,
    harmonic_l: BiquadFilter,
    harmonic_r: BiquadFilter,
    shaper_l: AntialiasedShaper,
    shaper_r: AntialiasedShaper,
}

impl ExciterStage {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            params: None,
            split_l: BiquadFilter::high_pass(SPLIT_ORDER, sample_rate, SEED_CROSSOVER_HZ, Q),
            split_r: BiquadFilter::high_pass(SPLIT_ORDER, sample_rate, SEED_CROSSOVER_HZ, Q),
            harmonic_l: BiquadFilter::high_pass(HARMONIC_ORDER, sample_rate, SEED_CROSSOVER_HZ, Q),
            harmonic_r: BiquadFilter::high_pass(HARMONIC_ORDER, sample_rate, SEED_CROSSOVER_HZ, Q),
            shaper_l: AntialiasedShaper::new(1.0, 0.0),
            shaper_r: AntialiasedShaper::new(1.0, 0.0),
        }
    }

    fn configure(&mut self, p: ExciterParams) {
        if self.params == Some(p) {
            return;
        }
        if self.params.map(|q| q.crossover_hz) != Some(p.crossover_hz) {
            self.split_l
                .set_high_pass(self.sample_rate, p.crossover_hz, Q);
            self.split_r
                .set_high_pass(self.sample_rate, p.crossover_hz, Q);
            self.harmonic_l
                .set_high_pass(self.sample_rate, p.crossover_hz, Q);
            self.harmonic_r
                .set_high_pass(self.sample_rate, p.crossover_hz, Q);
            self.split_l.reset_state();
            self.split_r.reset_state();
            self.harmonic_l.reset_state();
            self.harmonic_r.reset_state();
        }
        self.shaper_l.set_curve(p.drive, p.bias);
        self.shaper_r.set_curve(p.drive, p.bias);
        self.params = Some(p);
    }

    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        if sample_rate == self.sample_rate {
            return;
        }
        self.sample_rate = sample_rate;
        let previous = self.params.take();
        let crossover = previous.map_or(SEED_CROSSOVER_HZ, |p| p.crossover_hz);
        self.split_l = BiquadFilter::high_pass(SPLIT_ORDER, sample_rate, crossover, Q);
        self.split_r = BiquadFilter::high_pass(SPLIT_ORDER, sample_rate, crossover, Q);
        self.harmonic_l = BiquadFilter::high_pass(HARMONIC_ORDER, sample_rate, crossover, Q);
        self.harmonic_r = BiquadFilter::high_pass(HARMONIC_ORDER, sample_rate, crossover, Q);
        if let Some(p) = previous {
            self.configure(p);
        }
    }

    pub fn reset_state(&mut self) {
        self.split_l.reset_state();
        self.split_r.reset_state();
        self.harmonic_l.reset_state();
        self.harmonic_r.reset_state();
        self.shaper_l.reset_state();
        self.shaper_r.reset_state();
    }

    pub fn process_block(
        &mut self,
        l: &mut [Sample],
        r: &mut [Sample],
        params: ExciterParams,
        high_l: &mut [Sample],
        high_r: &mut [Sample],
    ) {
        self.configure(params);
        let n = block::stereo_len(l, r);
        let (high_l, high_r) = (&mut high_l[..n], &mut high_r[..n]);
        high_l.copy_from_slice(l);
        high_r.copy_from_slice(r);
        self.split_l.transform_block(high_l);
        self.split_r.transform_block(high_r);

        harmonics_of(high_l, &mut self.shaper_l);
        harmonics_of(high_r, &mut self.shaper_r);
        self.harmonic_l.transform_block(high_l);
        self.harmonic_r.transform_block(high_r);

        for (dry, &harmonic) in l.iter_mut().zip(high_l.iter()) {
            *dry += params.amount * harmonic;
        }
        for (dry, &harmonic) in r.iter_mut().zip(high_r.iter()) {
            *dry += params.amount * harmonic;
        }
    }

    #[inline]
    pub fn process(&mut self, input: Frame, params: ExciterParams) -> Frame {
        let (mut l, mut r) = ([input.0], [input.1]);
        let (mut high_l, mut high_r) = ([0.0], [0.0]);
        self.process_block(&mut l, &mut r, params, &mut high_l, &mut high_r);
        (l[0], r[0])
    }
}

fn harmonics_of(band: &mut [Sample], shaper: &mut AntialiasedShaper) {
    for x in band.iter_mut() {
        let sample = shaper.process(*x);
        *x = sample.shaped - sample.linear_reference;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::block::{MAX_BLOCK, TEST_BLOCK_LENGTHS, test_signal};

    const SR: f64 = 48_000.0;

    fn params(drive: Sample, amount: Sample) -> ExciterParams {
        ExciterParams {
            crossover_hz: 2000.0,
            drive,
            bias: 0.0,
            amount,
        }
    }

    fn amplitude_at(signal: &[Sample], hz: f64, rate: f64) -> f64 {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (n, &x) in signal.iter().enumerate() {
            let phase = core::f64::consts::TAU * hz * n as f64 / rate;
            re += f64::from(x) * phase.cos();
            im -= f64::from(x) * phase.sin();
        }
        2.0 * (re * re + im * im).sqrt() / signal.len() as f64
    }

    #[test]
    fn zero_amount_is_transparent() {
        let mut stage = ExciterStage::new(SR);
        let signal = test_signal(512);
        let (mut got_l, mut got_r) = (signal.clone(), signal.clone());
        let (mut high_l, mut high_r) = ([0.0; MAX_BLOCK], [0.0; MAX_BLOCK]);
        for (l, r) in got_l.chunks_mut(64).zip(got_r.chunks_mut(64)) {
            stage.process_block(l, r, params(4.0, 0.0), &mut high_l, &mut high_r);
        }
        assert_eq!(got_l, signal);
        assert_eq!(got_r, signal);
    }

    #[test]
    fn vanishing_drive_is_transparent() {
        let mut stage = ExciterStage::new(SR);
        let signal = test_signal(2048);
        let mut worst = 0.0f32;
        for &x in &signal {
            let (l, _) = stage.process((x, x), params(1.0e-3, 1.0));
            worst = worst.max((l - x).abs());
        }
        assert!(worst < 1.0e-4, "vanishing drive deviated by {worst}");
    }

    #[test]
    fn process_block_matches_per_sample() {
        let p = params(6.0, 0.75);
        for n in TEST_BLOCK_LENGTHS {
            let signal = test_signal(4 * n);
            let right: Vec<Sample> = signal.iter().map(|x| -0.4 * x).collect();

            let mut blocked = ExciterStage::new(SR);
            let (mut high_l, mut high_r) = ([0.0; MAX_BLOCK], [0.0; MAX_BLOCK]);
            let (mut got_l, mut got_r) = (signal.clone(), right.clone());
            for (l, r) in got_l.chunks_mut(n).zip(got_r.chunks_mut(n)) {
                blocked.process_block(l, r, p, &mut high_l, &mut high_r);
            }

            let mut per_sample = ExciterStage::new(SR);
            let (mut want_l, mut want_r) = (Vec::new(), Vec::new());
            for (&l, &r) in signal.iter().zip(&right) {
                let (l, r) = per_sample.process((l, r), p);
                want_l.push(l);
                want_r.push(r);
            }

            assert_eq!((got_l, got_r), (want_l, want_r), "block length {n}");
        }
    }

    #[test]
    fn the_shaper_adds_harmonics_the_input_did_not_have() {
        const LEN: usize = 4800;
        let tone = 5_000.0;
        let mut shaper = AntialiasedShaper::new(8.0, 0.0);
        let shaped: Vec<Sample> = (0..LEN)
            .map(|n| {
                let x = (core::f64::consts::TAU * tone * n as f64 / SR).sin() as Sample * 0.8;
                shaper.process(x).shaped
            })
            .collect();
        let fundamental = amplitude_at(&shaped, tone, SR);
        let third = amplitude_at(&shaped, 3.0 * tone, SR);
        assert!(
            third > 0.02 * fundamental,
            "third harmonic {third} too weak"
        );
    }

    #[test]
    fn antiderivative_antialiasing_rejects_the_folded_harmonics() {
        const LEN: usize = 4800;
        const DRIVE: f64 = 8.0;
        let tone = 5_000.0;
        let source: Vec<Sample> = (0..LEN)
            .map(|n| (core::f64::consts::TAU * tone * n as f64 / SR).sin() as Sample * 0.8)
            .collect();

        let mut shaper = AntialiasedShaper::new(DRIVE as Sample, 0.0);
        let antialiased: Vec<Sample> = source.iter().map(|&x| shaper.process(x).shaped).collect();
        let naive: Vec<Sample> = source
            .iter()
            .map(|&x| Curve::new(DRIVE as Sample, 0.0).shape(f64::from(x)) as Sample)
            .collect();

        let alias_energy = |signal: &[Sample]| -> f64 {
            [3_000.0, 7_000.0, 13_000.0, 23_000.0]
                .iter()
                .map(|&hz| amplitude_at(signal, hz, SR))
                .sum()
        };
        let (got, want) = (alias_energy(&antialiased), alias_energy(&naive));
        assert!(
            got < 0.5 * want,
            "antialiased aliases {got} did not beat naive {want}"
        );
    }

    #[test]
    fn bias_is_what_creates_even_harmonics() {
        const LEN: usize = 4800;
        let tone = 5_000.0;
        let source: Vec<Sample> = (0..LEN)
            .map(|n| (core::f64::consts::TAU * tone * n as f64 / SR).sin() as Sample * 0.5)
            .collect();

        let second_harmonic = |bias: Sample| -> f64 {
            let mut shaper = AntialiasedShaper::new(8.0, bias);
            let shaped: Vec<Sample> = source.iter().map(|&x| shaper.process(x).shaped).collect();
            amplitude_at(&shaped, 2.0 * tone, SR)
        };

        let symmetric = second_harmonic(0.0);
        let asymmetric = second_harmonic(0.3);
        assert!(
            symmetric < 1.0e-3,
            "an odd curve produced a second harmonic of {symmetric}"
        );
        assert!(
            asymmetric > 20.0 * symmetric.max(1.0e-6),
            "bias produced only {asymmetric} of second harmonic against {symmetric}"
        );
    }

    #[test]
    fn a_biased_curve_still_passes_through_the_origin() {
        let curve = Curve::new(8.0, 0.4);
        assert!(
            curve.shape(0.0).abs() < 1.0e-12,
            "shape(0) = {}",
            curve.shape(0.0)
        );
    }

    #[test]
    fn set_sample_rate_preserves_the_crossover() {
        let mut stage = ExciterStage::new(SR);
        let p = params(6.0, 1.0);
        let _ = stage.process((0.0, 0.0), p);
        stage.set_sample_rate(96_000.0);
        let mut settled = 0.0f32;
        for n in 0..40_000 {
            let x = (core::f64::consts::TAU * 100.0 * n as f64 / 96_000.0).sin() as Sample * 0.4;
            let (l, _) = stage.process((x, x), p);
            if n > 30_000 {
                settled = settled.max((l - x).abs());
            }
        }
        assert!(
            settled < 0.02,
            "a tone far below the crossover was excited by {settled}"
        );
    }
}

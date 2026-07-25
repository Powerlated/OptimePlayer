//! Stereo-linked feed-forward compressor with analog-style one-pole attack/release.
//!
//! Port of NAudio's `EnvelopeDetector` + `AttRelEnvelope` + `SimpleCompressor` (themselves from
//! ChunkWare SimpleComp v1.10). The reference's `Soundgoodizer` leaves `Ratio = 1.0` set, so its
//! per-band "compressors" do no work; this port uses the conventional ratio convention
//! (`R:1` with `R ≥ 1`), so the gain-reduction slope is `(R − 1) / R` and a 2:1 ratio halves the
//! dB above threshold — actual compression.

use crate::waveform::Sample;

/// DC offset added before log/exp to dodge `log(0)` and keep the envelope state out of the
/// denormal pit (a serious x86 perf killer). Matches the reference's `DC_OFFSET = 1.0E-25`.
const DC_OFFSET: f64 = 1.0e-25;

/// One-pole smoothing detector: `coeff = exp(-1 / (0.001 · ms · sr))`. State slews toward the
/// input by a fixed fraction per sample; the time constant is the standard "analog-style" one.
#[derive(Debug, Clone, Copy)]
struct EnvelopeDetector {
    coeff: f64,
}

impl EnvelopeDetector {
    fn new(ms: f64, sample_rate: f64) -> Self {
        Self {
            coeff: Self::coef(ms, sample_rate),
        }
    }

    fn set_time(&mut self, ms: f64, sample_rate: f64) {
        self.coeff = Self::coef(ms, sample_rate);
    }

    fn coef(ms: f64, sample_rate: f64) -> f64 {
        debug_assert!(ms > 0.0 && sample_rate > 0.0);
        (-1.0 / (0.001 * ms * sample_rate)).exp()
    }

    #[inline]
    fn run(self, input: f64, state: f64) -> f64 {
        input + self.coeff * (state - input)
    }
}

/// A simple feed-forward compressor with a stereo-linked sidechain.
///
/// Holds the attack/release detectors and the running envelope (dB above threshold). One
/// [`Self::process`] call handles both channels of a stereo sample: the sidechain is the louder of
/// the two, and the resulting gain is applied to both — so a peak on one channel ducks the other,
/// preserving the stereo image instead of letting one side pull away.
#[derive(Debug, Clone)]
pub struct SimpleCompressor {
    attack: EnvelopeDetector,
    release: EnvelopeDetector,
    /// Smoothed dB above threshold, held above `DC_OFFSET` to keep the detector out of denormals.
    env_db: f64,
    threshold_db: f64,
    ratio: f64,
    makeup_db: f64,
}

impl SimpleCompressor {
    /// Builds a new compressor at the given rate. The time-constants and dB params can be changed
    /// later via [`Self::set_params`]; the envelope state starts idle.
    pub fn new(
        attack_ms: f64,
        release_ms: f64,
        sample_rate: f64,
        threshold_db: f64,
        ratio: f64,
        makeup_db: f64,
    ) -> Self {
        Self {
            attack: EnvelopeDetector::new(attack_ms, sample_rate),
            release: EnvelopeDetector::new(release_ms, sample_rate),
            env_db: DC_OFFSET,
            threshold_db,
            ratio,
            makeup_db,
        }
    }

    /// Reconfigures everything (time constants + dB params). The envelope state is preserved, so
    /// changing a slider mid-playback doesn't cause an audible jump.
    pub fn set_params(
        &mut self,
        attack_ms: f64,
        release_ms: f64,
        sample_rate: f64,
        threshold_db: f64,
        ratio: f64,
        makeup_db: f64,
    ) {
        self.attack.set_time(attack_ms, sample_rate);
        self.release.set_time(release_ms, sample_rate);
        self.threshold_db = threshold_db;
        self.ratio = ratio;
        self.makeup_db = makeup_db;
    }

    /// Resets the envelope to its idle state (call on the inactive→active edge so a fresh enable
    /// doesn't duck from stale envelope state).
    pub fn reset_state(&mut self) {
        self.env_db = DC_OFFSET;
    }

    /// The current smoothed gain reduction in dB (≤ 0, excludes makeup). Derived from the running
    /// envelope: `-(env_db − DC_OFFSET) · (R − 1)/R`, the same `over_db · slope` shape `process`
    /// applies, but read straight off the detector state instead of the last sample — so it stays
    /// valid between renders and reflects the envelope's attack/release smoothing.
    pub fn last_reduction_db(&self) -> f64 {
        let over_db = self.env_db - DC_OFFSET;
        -over_db * (self.ratio - 1.0) / self.ratio
    }

    /// Compresses a block of consecutive stereo samples in place. Returns the gain reduction in dB
    /// applied to the last sample (negative = attenuation, positive = makeup boost beyond the
    /// reduction), or `self.makeup_db` for an empty block.
    ///
    /// The envelope feeds back into itself, so the samples are still walked one at a time; what the
    /// block form saves is re-reading the threshold, ratio, makeup and detector coefficients from
    /// the struct on every sample. They are constant across a block because they only change on a
    /// device tick, and a block never spans one.
    pub fn process_block(&mut self, l: &mut [Sample], r: &mut [Sample]) -> f64 {
        debug_assert_eq!(l.len(), r.len());
        let (attack, release) = (self.attack, self.release);
        let (threshold_db, makeup_db) = (self.threshold_db, self.makeup_db);
        // Conventional compressor slope: output above threshold = overdB / R, so the gain
        // reduction is overdB − overdB/R = overdB · (R − 1) / R. At R = 1 the slope is 0 (unity);
        // as R → ∞ it approaches 1 (hard limit at the threshold).
        let slope = (self.ratio - 1.0) / self.ratio;
        let mut env_db = self.env_db;
        let mut gr_db = makeup_db;

        for (l, r) in l.iter_mut().zip(r.iter_mut()) {
            // Sidechain: link channels by the louder of the two (the standard max-link convention).
            let link = (f64::from(*l)).abs().max(f64::from(*r).abs()) + DC_OFFSET;
            let key_db = 20.0 * link.log10();

            // dB above threshold (clamped to 0 — no expansion below the knee).
            let mut over_db = key_db - threshold_db;
            if over_db < 0.0 {
                over_db = 0.0;
            }
            // Add the DC offset before the envelope so the detector floor sits above the denormal pit.
            over_db += DC_OFFSET;

            // Attack on a rise, release on a fall (the dB-domain envelope is one-pole either way).
            env_db = if over_db > env_db {
                attack.run(over_db, env_db)
            } else {
                release.run(over_db, env_db)
            };
            let over_db = env_db - DC_OFFSET;

            gr_db = -over_db * slope + makeup_db;
            let gr_lin = (10f64).powf(gr_db / 20.0) as Sample;
            *l *= gr_lin;
            *r *= gr_lin;
        }

        self.env_db = env_db;
        gr_db
    }

    /// Compresses one stereo sample in place. A one-sample [`Self::process_block`].
    #[inline]
    pub fn process(&mut self, l: &mut Sample, r: &mut Sample) -> f64 {
        let (mut lb, mut rb) = ([*l], [*r]);
        let gr_db = self.process_block(&mut lb, &mut rb);
        (*l, *r) = (lb[0], rb[0]);
        gr_db
    }
}

#[cfg(test)]
#[allow(unused_assignments)]
mod tests {
    use super::*;

    /// A block of any length must give bit-identical results to compressing one stereo sample at a
    /// time, envelope and all.
    #[test]
    fn process_block_matches_per_sample() {
        use crate::dsp::block::{TEST_BLOCK_LENGTHS, test_signal};

        for n in TEST_BLOCK_LENGTHS {
            // Scaled well above the -6 dBFS threshold so the detector spends the run attacking and
            // releasing rather than sitting at the floor.
            let signal: Vec<Sample> = test_signal(4 * n).iter().map(|x| x * 3.0).collect();
            // The right channel is quieter and inverted, so the stereo-linked sidechain has to pick
            // a genuine per-sample maximum rather than seeing two identical channels.
            let right: Vec<Sample> = signal.iter().map(|x| -0.4 * x).collect();
            let make = || SimpleCompressor::new(2.0, 50.0, 48_000.0, -6.0, 4.0, 1.5);

            let mut blocked = make();
            let (mut got_l, mut got_r) = (signal.clone(), right.clone());
            for (l, r) in got_l.chunks_mut(n).zip(got_r.chunks_mut(n)) {
                blocked.process_block(l, r);
            }

            let mut per_sample = make();
            let (mut want_l, mut want_r) = (signal.clone(), right.clone());
            for (l, r) in want_l.iter_mut().zip(want_r.iter_mut()) {
                per_sample.process(l, r);
            }

            assert_eq!((got_l, got_r), (want_l, want_r), "block length {n}");
        }
    }

    /// At unity ratio the compressor must be a pure makeup gain, regardless of input level —
    /// pins the `gr = −overdB · (R − 1) / R` shape at the R = 1 boundary.
    #[test]
    fn unity_ratio_is_pure_makeup() {
        let mut c = SimpleCompressor::new(2.0, 50.0, 48_000.0, -6.0, 1.0, 0.0);
        for amp in [0.1_f32, 0.5, 1.0, 2.0] {
            // Drive the same amplitude long enough for the envelope to settle; the last iteration's
            // output is the assertion sample. `let mut` without an init avoids the unused-init
            // warning (the loop immediately assigns before reading).
            let mut l: Sample = amp;
            let mut r: Sample = amp;
            for _ in 0..5_000 {
                l = amp;
                r = amp;
                c.process(&mut l, &mut r);
            }
            assert!((l - amp).abs() < 1e-3, "amp {amp}: output {l} != {amp}");
        }
    }

    /// 2:1 ratio above a 0 dBFS threshold: +12 dB input should settle at +6 dB out (linear ×2 →
    /// input 4.0 → output 2.0). The classic 2:1 curve.
    #[test]
    fn two_to_one_halves_overthreshold() {
        let mut c = SimpleCompressor::new(0.5, 200.0, 48_000.0, 0.0, 2.0, 0.0);
        let mut l: Sample = 4.0;
        let mut r: Sample = 4.0;
        for _ in 0..20_000 {
            l = 4.0;
            r = 4.0;
            c.process(&mut l, &mut r);
        }
        assert!((l - 2.0).abs() < 0.02, "settled output was {l}");
    }

    /// A sub-threshold signal never triggers gain reduction.
    #[test]
    fn below_threshold_is_unity() {
        let mut c = SimpleCompressor::new(2.0, 50.0, 48_000.0, -6.0, 4.0, 0.0);
        let mut l: Sample = 0.1;
        let mut r: Sample = 0.1;
        for _ in 0..5_000 {
            l = 0.1;
            r = 0.1;
            c.process(&mut l, &mut r);
        }
        assert!((l - 0.1).abs() < 1e-3, "below-threshold output was {l}");
    }

    /// Reset returns the envelope to idle so a sub-threshold sample right after a loud passage
    /// comes out near-unity (no stale-state ducking).
    #[test]
    fn reset_clears_envelope() {
        let mut c = SimpleCompressor::new(2.0, 50.0, 48_000.0, -12.0, 4.0, 0.0);
        let mut l: Sample = 1.0;
        let mut r: Sample = 1.0;
        for _ in 0..2_000 {
            l = 1.0;
            r = 1.0;
            c.process(&mut l, &mut r);
        }
        c.reset_state();
        let (mut l, mut r) = (0.1_f32, 0.1_f32);
        c.process(&mut l, &mut r);
        assert!((l - 0.1).abs() < 1e-3, "post-reset output was {l}");
    }

    /// Makeup gain is applied in addition to (and independent of) the gain reduction: a unity-ratio
    /// compressor with +6 dB makeup passes a 0 dBFS tone at +6 dBFS.
    #[test]
    fn makeup_boosts_independently() {
        let mut c = SimpleCompressor::new(2.0, 50.0, 48_000.0, 0.0, 1.0, 6.0);
        let mut l: Sample = 1.0;
        let mut r: Sample = 1.0;
        for _ in 0..5_000 {
            l = 1.0;
            r = 1.0;
            c.process(&mut l, &mut r);
        }
        let want = 10f64.powf(6.0 / 20.0) as Sample;
        assert!((l - want).abs() < 0.01, "makeup output {l}, want {want}");
    }

    /// Attack is faster than release: a sudden loud input ducks quickly, then a drop back to quiet
    /// recovers more slowly. Pins the time-constant asymmetry (otherwise they're the same detector).
    #[test]
    fn attack_faster_than_release() {
        let mut c = SimpleCompressor::new(0.1, 5_000.0, 48_000.0, -6.0, 4.0, 0.0);
        // Drive loud until the envelope is well settled at the over-threshold level.
        let mut l: Sample = 1.0;
        let mut r: Sample = 1.0;
        for _ in 0..5_000 {
            l = 1.0;
            r = 1.0;
            c.process(&mut l, &mut r);
        }
        // Now go quiet. The release time-constant is 5 s, so after 1 s the envelope has only
        // decayed to ~e^(-0.2) ≈ 82% of its settled value — still substantial attenuation.
        let mut l: Sample = 0.5;
        let mut r: Sample = 0.5;
        for _ in 0..48_000 {
            l = 0.5;
            r = 0.5;
            c.process(&mut l, &mut r);
        }
        // Sub-threshold (0.5 ≈ −6 dB), so no fresh gr — but the release hasn't emptied yet either,
        // so the output is well below 0.5.
        assert!(
            l < 0.4,
            "post-release output {l} should still be attenuated"
        );
    }
}

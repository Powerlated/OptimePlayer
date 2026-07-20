//! 1-band multiband compressor: split the signal at `cutoff_hz` into a low band (untouched) and
//! a high band (compressed), then sum. Only the over-threshold high-band content is dynamically
//! attenuated — the rest of the spectrum passes bit-identical. Structurally after OptimeGBA's
//! `Soundgoodizer` (FL Studio band-split) but with a working compressor in the high path.
//!
//! Split: two cascaded RBJ filters per path at the same `cutoff_hz` with Butterworth Q = 1/√2
//! (≈24 dB/oct, matching the reference's `DbPerOct24 = true`). At unity compressor gain the LP + HP
//! paths sum to unity, so the stage is transparent there.

use crate::dsp::biquad_filter::BiquadFilter;
use crate::dsp::simple_compressor::SimpleCompressor;
use crate::waveform::{Frame, Sample};

/// Butterworth Q (maximally flat passband, no resonant peaking at the knee).
const Q: f64 = core::f64::consts::FRAC_1_SQRT_2;
/// Two cascaded biquad sections per path = 4th order ≈ 24 dB/oct slope.
const SPLIT_ORDER: usize = 4;

/// The runtime parameters a [`HighBandCompressorStage`] runs against — bundled so the per-sample
/// `process` call takes one struct argument instead of six (which trips clippy's
/// `too_many_arguments`). Constructed inline by the controller from its
/// [`HighBandCompressor`](crate::synth_controller::HighBandCompressor) settings struct.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HighBandCompressorParams {
    pub cutoff_hz: f64,
    pub threshold_db: f64,
    pub ratio: f64,
    pub attack_ms: f64,
    pub release_ms: f64,
    pub makeup_db: f64,
}

/// The runtime parameters the stage was last configured for, cached to skip redundant rebuilds on
/// a parameter-unchanged per-sample call. Same shape as [`HighBandCompressorParams`] but private.
#[derive(Debug, Clone, Copy, PartialEq)]
struct StageParams {
    cutoff_hz: f64,
    threshold_db: f64,
    ratio: f64,
    attack_ms: f64,
    release_ms: f64,
    makeup_db: f64,
}

/// One stereo high-band compressor: LPF + HPF split at `cutoff_hz`, [`SimpleCompressor`] on the
/// high path, low path passes the LPF untouched. Each channel has its own filter state but the
/// compressor sidechain is stereo-linked.
///
/// The filters are rebuilt when `cutoff_hz` or the sample rate changes; the compressor's
/// coefficients are rebuilt when any of its params or the rate changes. Both rebuilds are
/// memoised against [`StageParams`], so calling [`Self::process`] every sample with a fixed
/// config does no redundant work.
pub struct HighBandCompressorStage {
    sample_rate: f64,
    params: Option<StageParams>,
    hp_l: BiquadFilter,
    hp_r: BiquadFilter,
    lp_l: BiquadFilter,
    lp_r: BiquadFilter,
    comp: SimpleCompressor,
}

impl HighBandCompressorStage {
    /// Idle stage at `sample_rate`. The first [`Self::process`] call configures the filters and
    /// compressor to the supplied params; subsequent calls rebuild only what changed.
    pub fn new(sample_rate: f64) -> Self {
        // The filters are seeded at a placeholder cutoff; the real cutoff is set on first process.
        const SEED_CUTOFF: f64 = 3000.0;
        Self {
            sample_rate,
            params: None,
            hp_l: BiquadFilter::high_pass(SPLIT_ORDER, sample_rate, SEED_CUTOFF, Q),
            hp_r: BiquadFilter::high_pass(SPLIT_ORDER, sample_rate, SEED_CUTOFF, Q),
            lp_l: BiquadFilter::low_pass(SPLIT_ORDER, sample_rate, SEED_CUTOFF, Q),
            lp_r: BiquadFilter::low_pass(SPLIT_ORDER, sample_rate, SEED_CUTOFF, Q),
            comp: SimpleCompressor::new(2.0, 85.53, sample_rate, -18.0, 2.5, 0.0),
        }
    }

    /// Reconfigures the filters + compressor when the parameters change. No-op if unchanged, so
    /// it's cheap to call once per sample. The filter rebuild on a `cutoff_hz` change is the only
    /// expensive piece, and it's avoided when only compressor params (attack/release/threshold/
    /// ratio/makeup) move.
    fn configure(&mut self, p: HighBandCompressorParams) {
        let next = StageParams {
            cutoff_hz: p.cutoff_hz,
            threshold_db: p.threshold_db,
            ratio: p.ratio,
            attack_ms: p.attack_ms,
            release_ms: p.release_ms,
            makeup_db: p.makeup_db,
        };
        if self.params == Some(next) {
            return;
        }
        let cutoff_changed = self.params.map(|q| q.cutoff_hz) != Some(p.cutoff_hz);
        if cutoff_changed {
            self.hp_l.set_high_pass(self.sample_rate, p.cutoff_hz, Q);
            self.hp_r.set_high_pass(self.sample_rate, p.cutoff_hz, Q);
            self.lp_l.set_low_pass(self.sample_rate, p.cutoff_hz, Q);
            self.lp_r.set_low_pass(self.sample_rate, p.cutoff_hz, Q);
            self.hp_l.reset_state();
            self.hp_r.reset_state();
            self.lp_l.reset_state();
            self.lp_r.reset_state();
        }
        self.comp.set_params(
            p.attack_ms,
            p.release_ms,
            self.sample_rate,
            p.threshold_db,
            p.ratio,
            p.makeup_db,
        );
        self.params = Some(next);
    }

    /// Rebuilds everything for a new output rate (knee stays at a fixed frequency in Hz, matching
    /// the [`crate::synth_controller`] PSG-comp pattern). No-op if the rate is unchanged.
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        if sample_rate == self.sample_rate {
            return;
        }
        self.sample_rate = sample_rate;
        // Drop the cached params so the next `configure` definitely fires; rebuild the filters at
        // the new rate against the previous cutoff (or the seed if never configured).
        let prev_cutoff = self.params.map(|p| p.cutoff_hz).unwrap_or(3000.0);
        self.hp_l = BiquadFilter::high_pass(SPLIT_ORDER, sample_rate, prev_cutoff, Q);
        self.hp_r = BiquadFilter::high_pass(SPLIT_ORDER, sample_rate, prev_cutoff, Q);
        self.lp_l = BiquadFilter::low_pass(SPLIT_ORDER, sample_rate, prev_cutoff, Q);
        self.lp_r = BiquadFilter::low_pass(SPLIT_ORDER, sample_rate, prev_cutoff, Q);
        // The compressor's per-rate coefficients get refreshed by the next `configure` call.
        let prev = self.params;
        self.params = None;
        if let Some(p) = prev {
            self.configure(HighBandCompressorParams {
                cutoff_hz: p.cutoff_hz,
                threshold_db: p.threshold_db,
                ratio: p.ratio,
                attack_ms: p.attack_ms,
                release_ms: p.release_ms,
                makeup_db: p.makeup_db,
            });
        }
    }

    /// Clears all filter + envelope state. Call on the inactive→active edge so a fresh enable
    /// starts from silence rather than whatever the filters were left holding.
    pub fn reset_state(&mut self) {
        self.hp_l.reset_state();
        self.hp_r.reset_state();
        self.lp_l.reset_state();
        self.lp_r.reset_state();
        self.comp.reset_state();
    }

    /// The high-band compressor's most recent smoothed gain reduction in dB (≤ 0, no makeup).
    /// Reads the inner [`SimpleCompressor`]'s detector state, so it reflects the attack/release
    /// envelope rather than a per-sample peak. Stale until the first [`Self::process`] call after
    /// a rate/param change, like the detector itself.
    pub fn last_gr_db(&self) -> f64 {
        self.comp.last_reduction_db()
    }

    /// Processes one stereo sample: splits at `cutoff_hz` → compresses the high band → re-sums.
    /// The low band is touched only by its LPF; summing with the (gain-matched) HPF path restores
    /// unity at unity compressor gain, so the stage is transparent in that case.
    #[inline]
    pub fn process(&mut self, input: Frame, params: HighBandCompressorParams) -> Frame {
        self.configure(params);
        let (l, r) = input;
        let h_l = self.hp_l.transform(l);
        let h_r = self.hp_r.transform(r);
        let low_l = self.lp_l.transform(l);
        let low_r = self.lp_r.transform(r);
        let (mut h_l, mut h_r): (Sample, Sample) = (h_l, h_r);
        self.comp.process(&mut h_l, &mut h_r);
        (low_l + h_l, low_r + h_r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DC passes through the LPF unchanged and is blocked by the HPF; with the compressor at unity
    /// ratio the stage must leave a DC input untouched. Pins the "low band is transparent" half of
    /// the design.
    #[test]
    fn dc_is_transparent_at_unity_ratio() {
        let mut s = HighBandCompressorStage::new(48_000.0);
        let mut out = 0.0;
        for _ in 0..2_000 {
            out = s
                .process(
                    (1.0, 1.0),
                    HighBandCompressorParams {
                        cutoff_hz: 3000.0,
                        threshold_db: -18.0,
                        ratio: 1.0,
                        attack_ms: 2.0,
                        release_ms: 85.53,
                        makeup_db: 0.0,
                    },
                )
                .0;
        }
        assert!((out - 1.0).abs() < 1e-3, "DC output was {out}");
    }

    /// A sub-cutoff sine is in the LPF passband / HPF stopband, so the compressor sees near-zero
    /// sidechain and the output tracks the input — pins the band-split at an audible frequency.
    #[test]
    fn sub_cutoff_sine_passes_unity() {
        let sr = 48_000.0;
        let mut s = HighBandCompressorStage::new(sr);
        let cutoff = 3000.0;
        let tone = 250.0; // well below the split
        let mut peak = 0.0f32;
        let mut recent = 0.0f32;
        for n in 0..20_000 {
            let x = (2.0 * core::f64::consts::PI * tone * n as f64 / sr).sin() as f32 * 0.5;
            let (l, _) = s.process(
                (x, x),
                HighBandCompressorParams {
                    cutoff_hz: cutoff,
                    threshold_db: -18.0,
                    ratio: 4.0,
                    attack_ms: 2.0,
                    release_ms: 85.53,
                    makeup_db: 0.0,
                },
            );
            peak = peak.max(l.abs());
            if n > 15_000 {
                recent = recent.max(l.abs());
            }
        }
        // Below the split, so the compressor never engages meaningfully; settled output ≈ input.
        assert!(recent > 0.45, "sub-cutoff output {recent} too attenuated");
        assert!(peak < 0.6, "sub-cutoff peak {peak} unexpectedly large");
    }

    /// An above-cutoff sine above threshold IS attenuated: the HPF passes it to the compressor,
    /// which ducks it. The LPF blocks it, so the attenuation isn't masked by the low path.
    #[test]
    fn above_cutoff_above_threshold_is_attenuated() {
        let sr = 48_000.0;
        let mut s = HighBandCompressorStage::new(sr);
        let cutoff = 1500.0; // push the split lower so 6 kHz is well into the HPF passband
        let tone = 6_000.0;
        let mut peak_recent = 0.0f32;
        let amp = 0.9_f32; // ≈ −0.92 dBFS, well over the −18 dBFS threshold
        for n in 0..20_000 {
            let x = (2.0 * core::f64::consts::PI * tone * n as f64 / sr).sin() as f32 * amp;
            let (l, _) = s.process(
                (x, x),
                HighBandCompressorParams {
                    cutoff_hz: cutoff,
                    threshold_db: -18.0,
                    ratio: 4.0,
                    attack_ms: 0.5,
                    release_ms: 50.0,
                    makeup_db: 0.0,
                },
            );
            if n > 15_000 {
                peak_recent = peak_recent.max(l.abs());
            }
        }
        // 4:1 ratio: +17 dB over → +4.25 dB out → linear ≈ 0.5 → roughly half amplitude. The
        // band-split is imperfect so the input amplitude doesn't fully reach the HPF, but the
        // attenuation must be substantial (well below the input 0.9).
        assert!(
            peak_recent < 0.5,
            "above-cutoff settled peak {peak_recent} should be substantially attenuated"
        );
    }

    /// `set_sample_rate` rebuilds the filters at the new rate; the cached cutoff is preserved, so
    /// the band split stays at the same absolute Hz.
    #[test]
    fn set_sample_rate_preserves_cutoff() {
        let mut s = HighBandCompressorStage::new(48_000.0);
        // Configure once so a cutoff is cached.
        let _ = s.process(
            (0.0, 0.0),
            HighBandCompressorParams {
                cutoff_hz: 2000.0,
                threshold_db: -18.0,
                ratio: 4.0,
                attack_ms: 2.0,
                release_ms: 85.53,
                makeup_db: 0.0,
            },
        );
        s.set_sample_rate(96_000.0);
        // After the rate change the next process must still see the 2 kHz split. A 100 Hz tone at
        // 96 kHz should be transparent (sub-cutoff), regardless of rate.
        let mut peak = 0.0f32;
        for n in 0..20_000 {
            let x = (2.0 * core::f64::consts::PI * 100.0 * n as f64 / 96_000.0).sin() as f32 * 0.4;
            let (l, _) = s.process(
                (x, x),
                HighBandCompressorParams {
                    cutoff_hz: 2000.0,
                    threshold_db: -18.0,
                    ratio: 4.0,
                    attack_ms: 2.0,
                    release_ms: 85.53,
                    makeup_db: 0.0,
                },
            );
            if n > 15_000 {
                peak = peak.max(l.abs());
            }
        }
        assert!(
            peak > 0.35 && peak < 0.45,
            "post-rate-change sub-cutoff output {peak} drifted from 0.4"
        );
    }
}

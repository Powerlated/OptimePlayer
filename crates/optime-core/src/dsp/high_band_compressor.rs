//! Splits a signal at a crossover and compresses only the high band.

use crate::dsp::biquad_filter::BiquadFilter;
use crate::dsp::block;
use crate::dsp::simple_compressor::SimpleCompressor;
use crate::waveform::{Frame, Sample};

const Q: f64 = core::f64::consts::FRAC_1_SQRT_2;
const SPLIT_ORDER: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HighBandCompressorParams {
    pub cutoff_hz: f64,
    pub threshold_db: f64,
    pub ratio: f64,
    pub attack_ms: f64,
    pub release_ms: f64,
    pub makeup_db: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct StageParams {
    cutoff_hz: f64,
    threshold_db: f64,
    ratio: f64,
    attack_ms: f64,
    release_ms: f64,
    makeup_db: f64,
}

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
    pub fn new(sample_rate: f64) -> Self {
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

    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        if sample_rate == self.sample_rate {
            return;
        }
        self.sample_rate = sample_rate;
        let prev_cutoff = self.params.map(|p| p.cutoff_hz).unwrap_or(3000.0);
        self.hp_l = BiquadFilter::high_pass(SPLIT_ORDER, sample_rate, prev_cutoff, Q);
        self.hp_r = BiquadFilter::high_pass(SPLIT_ORDER, sample_rate, prev_cutoff, Q);
        self.lp_l = BiquadFilter::low_pass(SPLIT_ORDER, sample_rate, prev_cutoff, Q);
        self.lp_r = BiquadFilter::low_pass(SPLIT_ORDER, sample_rate, prev_cutoff, Q);
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

    pub fn reset_state(&mut self) {
        self.hp_l.reset_state();
        self.hp_r.reset_state();
        self.lp_l.reset_state();
        self.lp_r.reset_state();
        self.comp.reset_state();
    }

    pub fn last_gr_db(&self) -> f64 {
        self.comp.last_reduction_db()
    }

    pub fn process_block(
        &mut self,
        l: &mut [Sample],
        r: &mut [Sample],
        params: HighBandCompressorParams,
        high_l: &mut [Sample],
        high_r: &mut [Sample],
    ) {
        self.configure(params);
        let n = block::stereo_len(l, r);
        let (high_l, high_r) = (&mut high_l[..n], &mut high_r[..n]);
        high_l.copy_from_slice(l);
        high_r.copy_from_slice(r);
        self.hp_l.transform_block(high_l);
        self.hp_r.transform_block(high_r);
        self.lp_l.transform_block(l);
        self.lp_r.transform_block(r);
        self.comp.process_block(high_l, high_r);
        for (low, high) in l.iter_mut().zip(high_l.iter()) {
            *low += *high;
        }
        for (low, high) in r.iter_mut().zip(high_r.iter()) {
            *low += *high;
        }
    }

    #[inline]
    pub fn process(&mut self, input: Frame, params: HighBandCompressorParams) -> Frame {
        let (mut l, mut r) = ([input.0], [input.1]);
        let (mut high_l, mut high_r) = ([0.0], [0.0]);
        self.process_block(&mut l, &mut r, params, &mut high_l, &mut high_r);
        (l[0], r[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_block_matches_per_sample() {
        use crate::dsp::block::{MAX_BLOCK, TEST_BLOCK_LENGTHS, test_signal};

        let params = HighBandCompressorParams {
            cutoff_hz: 3000.0,
            threshold_db: -18.0,
            ratio: 4.0,
            attack_ms: 2.0,
            release_ms: 80.0,
            makeup_db: 1.0,
        };
        for n in TEST_BLOCK_LENGTHS {
            let signal = test_signal(4 * n);
            let right: Vec<Sample> = signal.iter().map(|x| -0.4 * x).collect();

            let mut blocked = HighBandCompressorStage::new(48_000.0);
            let (mut high_l, mut high_r) = ([0.0; MAX_BLOCK], [0.0; MAX_BLOCK]);
            let (mut got_l, mut got_r) = (signal.clone(), right.clone());
            for (l, r) in got_l.chunks_mut(n).zip(got_r.chunks_mut(n)) {
                blocked.process_block(l, r, params, &mut high_l, &mut high_r);
            }

            let mut per_sample = HighBandCompressorStage::new(48_000.0);
            let (mut want_l, mut want_r) = (Vec::new(), Vec::new());
            for (&l, &r) in signal.iter().zip(&right) {
                let (l, r) = per_sample.process((l, r), params);
                want_l.push(l);
                want_r.push(r);
            }

            assert_eq!((got_l, got_r), (want_l, want_r), "block length {n}");
        }
    }

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

    #[test]
    fn sub_cutoff_sine_passes_unity() {
        let sr = 48_000.0;
        let mut s = HighBandCompressorStage::new(sr);
        let cutoff = 3000.0;
        let tone = 250.0;
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
        assert!(recent > 0.45, "sub-cutoff output {recent} too attenuated");
        assert!(peak < 0.6, "sub-cutoff peak {peak} unexpectedly large");
    }

    #[test]
    fn above_cutoff_above_threshold_is_attenuated() {
        let sr = 48_000.0;
        let mut s = HighBandCompressorStage::new(sr);
        let cutoff = 1500.0;
        let tone = 6_000.0;
        let mut peak_recent = 0.0f32;
        let amp = 0.9_f32;
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
        assert!(
            peak_recent < 0.5,
            "above-cutoff settled peak {peak_recent} should be substantially attenuated"
        );
    }

    #[test]
    fn set_sample_rate_preserves_cutoff() {
        let mut s = HighBandCompressorStage::new(48_000.0);
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

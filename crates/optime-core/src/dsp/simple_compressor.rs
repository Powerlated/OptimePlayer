use crate::waveform::Sample;

const DC_OFFSET: f64 = 1.0e-25;

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

#[derive(Debug, Clone)]
pub struct SimpleCompressor {
    attack: EnvelopeDetector,
    release: EnvelopeDetector,
    env_db: f64,
    threshold_db: f64,
    ratio: f64,
    makeup_db: f64,
}

impl SimpleCompressor {
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

    pub fn reset_state(&mut self) {
        self.env_db = DC_OFFSET;
    }

    pub fn last_reduction_db(&self) -> f64 {
        let over_db = self.env_db - DC_OFFSET;
        -over_db * (self.ratio - 1.0) / self.ratio
    }

    pub fn process_block(&mut self, l: &mut [Sample], r: &mut [Sample]) -> f64 {
        debug_assert_eq!(l.len(), r.len());
        let (attack, release) = (self.attack, self.release);
        let (threshold_db, makeup_db) = (self.threshold_db, self.makeup_db);
        let slope = (self.ratio - 1.0) / self.ratio;
        let mut env_db = self.env_db;
        let mut gr_db = makeup_db;

        for (l, r) in l.iter_mut().zip(r.iter_mut()) {
            let link = (f64::from(*l)).abs().max(f64::from(*r).abs()) + DC_OFFSET;
            let key_db = 20.0 * link.log10();

            let mut over_db = key_db - threshold_db;
            if over_db < 0.0 {
                over_db = 0.0;
            }
            over_db += DC_OFFSET;

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

    #[test]
    fn process_block_matches_per_sample() {
        use crate::dsp::block::{TEST_BLOCK_LENGTHS, test_signal};

        for n in TEST_BLOCK_LENGTHS {
            let signal: Vec<Sample> = test_signal(4 * n).iter().map(|x| x * 3.0).collect();
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

    #[test]
    fn unity_ratio_is_pure_makeup() {
        let mut c = SimpleCompressor::new(2.0, 50.0, 48_000.0, -6.0, 1.0, 0.0);
        for amp in [0.1_f32, 0.5, 1.0, 2.0] {
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

    #[test]
    fn attack_faster_than_release() {
        let mut c = SimpleCompressor::new(0.1, 5_000.0, 48_000.0, -6.0, 4.0, 0.0);
        let mut l: Sample = 1.0;
        let mut r: Sample = 1.0;
        for _ in 0..5_000 {
            l = 1.0;
            r = 1.0;
            c.process(&mut l, &mut r);
        }
        let mut l: Sample = 0.5;
        let mut r: Sample = 0.5;
        for _ in 0..48_000 {
            l = 0.5;
            r = 0.5;
            c.process(&mut l, &mut r);
        }
        assert!(
            l < 0.4,
            "post-release output {l} should still be attenuated"
        );
    }
}

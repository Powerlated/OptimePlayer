//! A cascaded biquad filter with the usual designs, in scalar and block form.

use core::f64::consts::PI;

use crate::waveform::Sample;

#[derive(Debug, Clone)]
pub struct BiquadFilter {
    a0: f32,
    a1: f32,
    a2: f32,
    a3: f32,
    a4: f32,

    x1: Vec<f32>,
    x2: Vec<f32>,
    y1: Vec<f32>,
    y2: Vec<f32>,
}

impl BiquadFilter {
    pub fn new(order: usize, aa0: f64, aa1: f64, aa2: f64, b0: f64, b1: f64, b2: f64) -> Self {
        assert!(order.is_multiple_of(2), "order not divisible by 2");
        let num_cascade = order / 2;
        let mut filter = Self {
            a0: 0.0,
            a1: 0.0,
            a2: 0.0,
            a3: 0.0,
            a4: 0.0,
            x1: vec![0.0; num_cascade],
            x2: vec![0.0; num_cascade],
            y1: vec![0.0; num_cascade],
            y2: vec![0.0; num_cascade],
        };
        filter.set_coefficients(aa0, aa1, aa2, b0, b1, b2);
        filter
    }

    #[inline]
    pub fn num_cascade(&self) -> usize {
        self.x1.len()
    }

    pub fn reset_state(&mut self) {
        for v in self
            .x1
            .iter_mut()
            .chain(&mut self.x2)
            .chain(&mut self.y1)
            .chain(&mut self.y2)
        {
            *v = 0.0;
        }
    }

    pub fn transform_block(&mut self, block: &mut [Sample]) {
        let (a0, a1, a2, a3, a4) = (self.a0, self.a1, self.a2, self.a3, self.a4);
        for i in 0..self.num_cascade() {
            let (mut x1, mut x2) = (self.x1[i], self.x2[i]);
            let (mut y1, mut y2) = (self.y1[i], self.y2[i]);
            for x in block.iter_mut() {
                let result = a0 * *x + a1 * x1 + a2 * x2 - a3 * y1 - a4 * y2;
                x2 = x1;
                x1 = *x;
                y2 = y1;
                y1 = result;
                *x = result;
            }
            (self.x1[i], self.x2[i]) = (x1, x2);
            (self.y1[i], self.y2[i]) = (y1, y2);
        }
    }

    #[inline]
    pub fn transform(&mut self, in_sample: Sample) -> Sample {
        let mut block = [in_sample];
        self.transform_block(&mut block);
        block[0]
    }

    pub fn set_coefficients(&mut self, aa0: f64, aa1: f64, aa2: f64, b0: f64, b1: f64, b2: f64) {
        self.a0 = (b0 / aa0) as f32;
        self.a1 = (b1 / aa0) as f32;
        self.a2 = (b2 / aa0) as f32;
        self.a3 = (aa1 / aa0) as f32;
        self.a4 = (aa2 / aa0) as f32;
    }

    pub fn set_low_pass(&mut self, sample_rate: f64, cutoff: f64, q: f64) {
        let w0 = 2.0 * PI * cutoff / sample_rate;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q);
        let b1 = 1.0 - cos_w0;
        let b0 = b1 / 2.0;
        let b2 = b0;
        self.set_coefficients(1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha, b0, b1, b2);
    }

    pub fn set_high_pass(&mut self, sample_rate: f64, cutoff: f64, q: f64) {
        let w0 = 2.0 * PI * cutoff / sample_rate;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q);
        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = b0;
        self.set_coefficients(1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha, b0, b1, b2);
    }

    pub fn set_peaking_eq(&mut self, sample_rate: f64, centre: f64, q: f64, db_gain: f64) {
        let w0 = 2.0 * PI * centre / sample_rate;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q);
        let a = 10f64.powf(db_gain / 40.0);
        self.set_coefficients(
            1.0 + alpha / a,
            -2.0 * cos_w0,
            1.0 - alpha / a,
            1.0 + alpha * a,
            -2.0 * cos_w0,
            1.0 - alpha * a,
        );
    }

    pub fn set_high_shelf(&mut self, sample_rate: f64, corner: f64, q: f64, db_gain: f64) {
        let per_section_db = db_gain / self.num_cascade().max(1) as f64;
        let a = 10f64.powf(per_section_db / 40.0);
        let w0 = 2.0 * PI * corner / sample_rate;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q);
        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;
        let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
        let aa0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
        let aa1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
        let aa2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha;
        self.set_coefficients(aa0, aa1, aa2, b0, b1, b2);
    }

    pub fn high_shelf(order: usize, sample_rate: f64, corner: f64, q: f64, db_gain: f64) -> Self {
        let mut f = Self::new(order, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        f.set_high_shelf(sample_rate, corner, q, db_gain);
        f
    }

    pub fn low_pass(order: usize, sample_rate: f64, cutoff: f64, q: f64) -> Self {
        let mut f = Self::new(order, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        f.set_low_pass(sample_rate, cutoff, q);
        f
    }

    pub fn high_pass(order: usize, sample_rate: f64, cutoff: f64, q: f64) -> Self {
        let mut f = Self::new(order, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        f.set_high_pass(sample_rate, cutoff, q);
        f
    }

    pub fn peaking_eq(order: usize, sample_rate: f64, centre: f64, q: f64, db_gain: f64) -> Self {
        let mut f = Self::new(order, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        f.set_peaking_eq(sample_rate, centre, q, db_gain);
        f
    }

    pub fn frequency_response(&self, w_norm: f64) -> (f64, f64) {
        let cos_w = w_norm.cos();
        let sin_w = w_norm.sin();
        let cos_2w = (2.0 * w_norm).cos();
        let sin_2w = (2.0 * w_norm).sin();

        let (a0, a1, a2) = (f64::from(self.a0), f64::from(self.a1), f64::from(self.a2));
        let (a3, a4) = (f64::from(self.a3), f64::from(self.a4));
        let n_re = a0 + a1 * cos_w + a2 * cos_2w;
        let n_im = -(a1 * sin_w + a2 * sin_2w);
        let d_re = 1.0 + a3 * cos_w + a4 * cos_2w;
        let d_im = -(a3 * sin_w + a4 * sin_2w);

        let d_sq = d_re * d_re + d_im * d_im;
        if d_sq < 1e-30 {
            return (f64::INFINITY, 0.0);
        }

        let h_re = (n_re * d_re + n_im * d_im) / d_sq;
        let h_im = (n_im * d_re - n_re * d_im) / d_sq;

        let h_mag = (h_re * h_re + h_im * h_im).sqrt();
        let h_phase = h_im.atan2(h_re);

        let nc = self.num_cascade() as f64;
        (h_mag.powf(nc), h_phase * nc)
    }

    pub fn poles(&self) -> [(f64, f64); 2] {
        quadratic_roots(1.0, f64::from(self.a3), f64::from(self.a4))
    }

    pub fn zeros(&self) -> [(f64, f64); 2] {
        if self.a0.abs() < 1e-15 {
            return [(0.0, 0.0); 2];
        }
        quadratic_roots(f64::from(self.a0), f64::from(self.a1), f64::from(self.a2))
    }
}

fn quadratic_roots(a: f64, b: f64, c: f64) -> [(f64, f64); 2] {
    let p = b / a;
    let q = c / a;
    let disc = p * p - 4.0 * q;
    if disc >= 0.0 {
        let sq = disc.sqrt();
        [((-p + sq) / 2.0, 0.0), ((-p - sq) / 2.0, 0.0)]
    } else {
        let sq = (-disc).sqrt();
        [(-p / 2.0, sq / 2.0), (-p / 2.0, -sq / 2.0)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_block_matches_per_sample() {
        use crate::dsp::block::{TEST_BLOCK_LENGTHS, test_signal};

        for order in [2, 4, 6] {
            for n in TEST_BLOCK_LENGTHS {
                let signal = test_signal(4 * n);
                let make = || BiquadFilter::low_pass(order, 48_000.0, 3_000.0, 0.707);

                let mut blocked = make();
                let mut got = signal.clone();
                for chunk in got.chunks_mut(n) {
                    blocked.transform_block(chunk);
                }

                let mut per_sample = make();
                let want: Vec<_> = signal.iter().map(|&x| per_sample.transform(x)).collect();

                assert_eq!(got, want, "order {order}, block length {n}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "order not divisible by 2")]
    fn rejects_odd_order() {
        BiquadFilter::new(3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    #[test]
    fn cascade_count() {
        let f = BiquadFilter::low_pass(4, 48000.0, 1000.0, 0.707);
        assert_eq!(f.num_cascade(), 2);
    }

    #[test]
    fn low_pass_passes_dc() {
        let mut f = BiquadFilter::low_pass(2, 48000.0, 1000.0, 0.707);
        let mut out = 0.0;
        for _ in 0..2000 {
            out = f.transform(1.0);
        }
        assert!((out - 1.0).abs() < 1e-3, "DC gain was {out}");
    }

    #[test]
    fn high_pass_blocks_dc() {
        let mut f = BiquadFilter::high_pass(2, 48000.0, 1000.0, 0.707);
        let mut out = 0.0;
        for _ in 0..2000 {
            out = f.transform(1.0);
        }
        assert!(out.abs() < 1e-3, "DC leakage was {out}");
    }

    #[test]
    fn reset_clears_state() {
        let mut f = BiquadFilter::low_pass(2, 48000.0, 1000.0, 0.707);
        for _ in 0..100 {
            f.transform(1.0);
        }
        f.reset_state();
        let mut fresh = BiquadFilter::low_pass(2, 48000.0, 1000.0, 0.707);
        assert_eq!(f.transform(1.0), fresh.transform(1.0));
    }

    #[test]
    fn low_pass_dc_frequency_response_is_unity() {
        let f = BiquadFilter::low_pass(4, 48000.0, 1000.0, 0.707);
        let (mag, _) = f.frequency_response(0.0);
        assert!((mag - 1.0).abs() < 1e-5, "DC magnitude = {mag}");
    }

    #[test]
    fn low_pass_poles_inside_unit_circle() {
        let f = BiquadFilter::low_pass(4, 48000.0, 1000.0, 0.707);
        for (re, im) in f.poles() {
            let r = (re * re + im * im).sqrt();
            assert!(r < 1.0, "pole |z| = {r} is outside the unit circle");
        }
    }

    #[test]
    fn high_shelf_boosts_highs_leaves_dc() {
        for order in [2usize, 4, 8] {
            let f = BiquadFilter::high_shelf(order, 48000.0, 4000.0, 0.707, 12.0);
            let (dc, _) = f.frequency_response(0.0);
            let (nyq, _) = f.frequency_response(std::f64::consts::PI);
            assert!((dc - 1.0).abs() < 1e-3, "order {order}: DC gain {dc}");
            let want = 10f64.powf(12.0 / 20.0);
            assert!(
                (nyq - want).abs() < 0.05 * want,
                "order {order}: Nyquist gain {nyq}, want {want}"
            );
        }
    }

    #[test]
    fn high_shelf_cut_attenuates_highs() {
        let f = BiquadFilter::high_shelf(4, 48000.0, 4000.0, 0.707, -12.0);
        let (dc, _) = f.frequency_response(0.0);
        let (nyq, _) = f.frequency_response(std::f64::consts::PI);
        assert!((dc - 1.0).abs() < 1e-3, "DC gain {dc}");
        let want = 10f64.powf(-12.0 / 20.0);
        assert!(
            (nyq - want).abs() < 0.05 * want,
            "Nyquist gain {nyq}, want {want}"
        );
    }

    #[test]
    fn high_pass_nyquist_response_is_unity() {
        let f = BiquadFilter::high_pass(2, 48000.0, 100.0, 0.707);
        let (mag, _) = f.frequency_response(std::f64::consts::PI);
        assert!((mag - 1.0).abs() < 1e-4, "Nyquist magnitude = {mag}");
    }

    #[test]
    fn f32_state_is_stable_at_low_cutoff_crossover() {
        let sr = 48_000.0;
        for (mut f, cutoff) in [
            (
                BiquadFilter::low_pass(4, sr, 120.0, core::f64::consts::FRAC_1_SQRT_2),
                120.0,
            ),
            (BiquadFilter::low_pass(4, sr, 14_534.8, 0.707), 14_534.8),
        ] {
            let tone_hz = cutoff * 0.5;
            let mut peak = 0.0f32;
            let mut last = 0.0f32;
            for n in 0..20_000 {
                let x = (2.0 * std::f64::consts::PI * tone_hz * n as f64 / sr).sin() as f32;
                last = f.transform(x);
                assert!(
                    last.is_finite(),
                    "cutoff {cutoff}: non-finite output at {n}"
                );
                peak = peak.max(last.abs());
            }
            assert!(
                peak < 1.5,
                "cutoff {cutoff}: output peak {peak} suggests instability"
            );
            assert!(
                peak > 0.5,
                "cutoff {cutoff}: output peak {peak} unexpectedly attenuated"
            );
            let _ = last;

            f.reset_state();
            let mut dc = 0.0f32;
            for _ in 0..8_000 {
                dc = f.transform(1.0);
            }
            assert!((dc - 1.0).abs() < 1e-2, "cutoff {cutoff}: DC gain {dc}");
            for (re, im) in f.poles() {
                assert!(
                    re * re + im * im < 1.0,
                    "cutoff {cutoff}: pole outside unit circle"
                );
            }
        }
    }
}

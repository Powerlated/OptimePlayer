//! Cascaded biquad filter, ported from the original `dsp.js` (itself based on NAudio).

use core::f64::consts::PI;

/// A cascade of identical second-order (biquad) filter sections.
///
/// `order` must be even; the cascade contains `order / 2` sections.
#[derive(Debug, Clone)]
pub struct BiquadFilter {
    // Normalized coefficients.
    a0: f64,
    a1: f64,
    a2: f64,
    a3: f64,
    a4: f64,

    // Per-section state.
    x1: Vec<f64>,
    x2: Vec<f64>,
    y1: Vec<f64>,
    y2: Vec<f64>,
}

impl BiquadFilter {
    /// Creates a filter of the given (even) `order` with raw, un-normalized coefficients.
    ///
    /// # Panics
    /// Panics if `order` is odd.
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

    /// Number of cascaded sections.
    #[inline]
    pub fn num_cascade(&self) -> usize {
        self.x1.len()
    }

    /// Clears all filter state.
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

    /// Processes one sample through the whole cascade.
    pub fn transform(&mut self, mut in_sample: f64) -> f64 {
        for i in 0..self.num_cascade() {
            let result = self.a0 * in_sample + self.a1 * self.x1[i] + self.a2 * self.x2[i]
                - self.a3 * self.y1[i]
                - self.a4 * self.y2[i];

            self.x2[i] = self.x1[i];
            self.x1[i] = in_sample;
            self.y2[i] = self.y1[i];
            self.y1[i] = result;

            in_sample = result;
        }
        in_sample
    }

    /// Sets and normalizes the filter coefficients.
    pub fn set_coefficients(&mut self, aa0: f64, aa1: f64, aa2: f64, b0: f64, b1: f64, b2: f64) {
        self.a0 = b0 / aa0;
        self.a1 = b1 / aa0;
        self.a2 = b2 / aa0;
        self.a3 = aa1 / aa0;
        self.a4 = aa2 / aa0;
    }

    /// Configures this filter as a low-pass with the given cutoff and Q.
    pub fn set_low_pass(&mut self, sample_rate: f64, cutoff: f64, q: f64) {
        let w0 = 2.0 * PI * cutoff / sample_rate;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q);
        let b1 = 1.0 - cos_w0;
        let b0 = b1 / 2.0;
        let b2 = b0;
        self.set_coefficients(1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha, b0, b1, b2);
    }

    /// Configures this filter as a high-pass with the given cutoff and Q.
    pub fn set_high_pass(&mut self, sample_rate: f64, cutoff: f64, q: f64) {
        let w0 = 2.0 * PI * cutoff / sample_rate;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q);
        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = b0;
        self.set_coefficients(1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha, b0, b1, b2);
    }

    /// Configures this filter as a peaking EQ with the given centre frequency, Q and gain (dB).
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

    /// Convenience constructor for a low-pass filter.
    pub fn low_pass(order: usize, sample_rate: f64, cutoff: f64, q: f64) -> Self {
        let mut f = Self::new(order, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        f.set_low_pass(sample_rate, cutoff, q);
        f
    }

    /// Convenience constructor for a high-pass filter.
    pub fn high_pass(order: usize, sample_rate: f64, cutoff: f64, q: f64) -> Self {
        let mut f = Self::new(order, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        f.set_high_pass(sample_rate, cutoff, q);
        f
    }

    /// Convenience constructor for a peaking EQ filter.
    pub fn peaking_eq(order: usize, sample_rate: f64, centre: f64, q: f64, db_gain: f64) -> Self {
        let mut f = Self::new(order, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        f.set_peaking_eq(sample_rate, centre, q, db_gain);
        f
    }

    // ─── Analysis ────────────────────────────────────────────────────────────────────────────

    /// Evaluates the frequency response of the full cascade at digital frequency
    /// `w_norm ∈ [0, π]` (0 = DC, π = Nyquist).
    ///
    /// Returns `(magnitude, phase_radians)`.  The cascade raises the single-section response
    /// to the power `num_cascade`.
    pub fn frequency_response(&self, w_norm: f64) -> (f64, f64) {
        let cos_w = w_norm.cos();
        let sin_w = w_norm.sin();
        let cos_2w = (2.0 * w_norm).cos();
        let sin_2w = (2.0 * w_norm).sin();

        // One biquad section: H(z) = (a0 + a1·z⁻¹ + a2·z⁻²) / (1 + a3·z⁻¹ + a4·z⁻²).
        // Evaluate numerator and denominator at z = e^{jw}.
        let n_re = self.a0 + self.a1 * cos_w + self.a2 * cos_2w;
        let n_im = -(self.a1 * sin_w + self.a2 * sin_2w);
        let d_re = 1.0 + self.a3 * cos_w + self.a4 * cos_2w;
        let d_im = -(self.a3 * sin_w + self.a4 * sin_2w);

        let d_sq = d_re * d_re + d_im * d_im;
        if d_sq < 1e-30 {
            return (f64::INFINITY, 0.0);
        }

        // H_section = N / D (complex division).
        let h_re = (n_re * d_re + n_im * d_im) / d_sq;
        let h_im = (n_im * d_re - n_re * d_im) / d_sq;

        let h_mag = (h_re * h_re + h_im * h_im).sqrt();
        let h_phase = h_im.atan2(h_re);

        // Cascade: |H^n| = |H|^n, arg(H^n) = n·arg(H).
        let nc = self.num_cascade() as f64;
        (h_mag.powf(nc), h_phase * nc)
    }

    /// Returns the two poles of one biquad section as `(re, im)` pairs.
    ///
    /// For a cascaded filter each pole has multiplicity `num_cascade`.  Roots of
    /// `z² + a3·z + a4 = 0`.
    pub fn poles(&self) -> [(f64, f64); 2] {
        quadratic_roots(1.0, self.a3, self.a4)
    }

    /// Returns the two zeros of one biquad section as `(re, im)` pairs.
    ///
    /// Roots of `a0·z² + a1·z + a2 = 0`.
    pub fn zeros(&self) -> [(f64, f64); 2] {
        if self.a0.abs() < 1e-15 {
            return [(0.0, 0.0); 2];
        }
        quadratic_roots(self.a0, self.a1, self.a2)
    }
}

/// Computes the two roots of `a·z² + b·z + c = 0` as `(re, im)` pairs.
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
        // A low-pass filter should converge to unity gain on a DC (constant) input.
        let mut f = BiquadFilter::low_pass(2, 48000.0, 1000.0, 0.707);
        let mut out = 0.0;
        for _ in 0..2000 {
            out = f.transform(1.0);
        }
        assert!((out - 1.0).abs() < 1e-3, "DC gain was {out}");
    }

    #[test]
    fn high_pass_blocks_dc() {
        // A high-pass filter should reject DC, settling toward zero.
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
        // First sample after reset matches a fresh filter's first sample.
        let mut fresh = BiquadFilter::low_pass(2, 48000.0, 1000.0, 0.707);
        assert_eq!(f.transform(1.0), fresh.transform(1.0));
    }

    #[test]
    fn low_pass_dc_frequency_response_is_unity() {
        let f = BiquadFilter::low_pass(4, 48000.0, 1000.0, 0.707);
        let (mag, _) = f.frequency_response(0.0);
        assert!((mag - 1.0).abs() < 1e-6, "DC magnitude = {mag}");
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
    fn high_pass_nyquist_response_is_unity() {
        // A high-pass well below Nyquist should pass Nyquist unchanged.
        let f = BiquadFilter::high_pass(2, 48000.0, 100.0, 0.707);
        let (mag, _) = f.frequency_response(std::f64::consts::PI);
        assert!((mag - 1.0).abs() < 1e-4, "Nyquist magnitude = {mag}");
    }
}

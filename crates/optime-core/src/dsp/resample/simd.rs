//! Portable-SIMD (`std::simd`, nightly) impulse gather. Instead of vector-gathering the kernel
//! tables (hardware gathers measured *slower* than the scalar walk), the kernel is evaluated
//! analytically, four taps at a time, with **rotating phasors**: per chunk, each needed sin/cos
//! advances by a constant angle via one complex multiply — pure FMA math, zero table traffic.
//! This is also (slightly) more accurate than the table interpolation.
//!
//! Accumulation runs in four parallel partial sums reduced at the end, so results differ from the
//! scalar build by float rounding — the DC normalization in
//! [`resample_sinc`](super::resample_sinc) stays exact because `out` and `wsum` accumulate with
//! identical operations.

use core::f64::consts::PI;
use std::simd::prelude::*;

use super::kernels::kernel_weight;

const LANES: usize = 4;
type Fv = Simd<f64, LANES>;

/// A vector of sin/cos pairs advanced by a fixed per-chunk angle step via complex rotation.
struct Phasor {
    sin: Fv,
    cos: Fv,
    step_sin: f64,
    step_cos: f64,
}

impl Phasor {
    /// Lanes at angles `rate · (d0 − i)` for `i = 0..LANES`, stepping by `−rate·LANES`/chunk.
    fn new(rate: f64, d0: f64) -> Self {
        let (mut sin, mut cos) = ([0.0; LANES], [0.0; LANES]);
        for i in 0..LANES {
            (sin[i], cos[i]) = f64::sin_cos(rate * (d0 - i as f64));
        }
        let (step_sin, step_cos) = f64::sin_cos(rate * LANES as f64);
        Self {
            sin: Fv::from_array(sin),
            cos: Fv::from_array(cos),
            step_sin,
            step_cos,
        }
    }

    /// Rotates every lane by the chunk step: `θ ← θ − rate·LANES`.
    #[inline]
    fn rotate(&mut self) {
        let (s, c) = (self.sin, self.cos);
        let (ss, sc) = (Fv::splat(self.step_sin), Fv::splat(self.step_cos));
        self.sin = s * sc - c * ss;
        self.cos = c * sc + s * ss;
    }
}

/// Vectorized impulse gather: `w(d) = sinc(2fc·d) · blackman(|d|/P)` evaluated analytically.
/// `sinc` is even and `blackman` is even in `d`, so no run splitting or `abs` ordering is
/// needed — one support mask per chunk. Returns `(Σ src·w, Σ w)`.
pub(crate) fn gather_impulse(src: &[f32], d0: f64, fc: f64, p: f64) -> (f64, f64) {
    // Angle rates per source sample: πτ = a·d for the sinc, and the blackman harmonics.
    let a = PI * 2.0 * fc;
    let b = PI / p;
    let mut ph_sinc = Phasor::new(a, d0); // sin(a·d) / (a·d) = sinc(2fc·d)
    let mut ph_win1 = Phasor::new(b, d0); // cos(π·d/P)
    let mut ph_win2 = Phasor::new(2.0 * b, d0); // cos(2π·d/P)

    let (mut out, mut wsum) = (Fv::splat(0.0), Fv::splat(0.0));
    let mut d = Fv::splat(d0) - Fv::from_array([0.0, 1.0, 2.0, 3.0]);
    for chunk in src.chunks_exact(LANES) {
        // sinc(2fc·d) = sin(a·d)/(a·d), with the removable singularity at d = 0.
        let arg = d * Fv::splat(a);
        let near_zero = arg.abs().simd_lt(Fv::splat(1e-12));
        let sinc = near_zero.select(Fv::splat(1.0), ph_sinc.sin / arg);
        // blackman(|d|/P), 0 outside the ±P support.
        let win = Fv::splat(0.42) + Fv::splat(0.5) * ph_win1.cos + Fv::splat(0.08) * ph_win2.cos;
        let inside = d.abs().simd_lt(Fv::splat(p));
        let w = inside.select(sinc * win, Fv::splat(0.0));

        out += Simd::<f32, LANES>::from_slice(chunk).cast::<f64>() * w;
        wsum += w;
        d -= Fv::splat(LANES as f64);
        ph_sinc.rotate();
        ph_win1.rotate();
        ph_win2.rotate();
    }
    let (mut out, mut wsum) = (out.reduce_sum(), wsum.reduce_sum());

    let done = src.len() - src.chunks_exact(LANES).remainder().len();
    for (j, &s) in src.iter().enumerate().skip(done) {
        let d = d0 - j as f64;
        let w = kernel_weight(d, fc, p);
        out += f64::from(s) * w;
        wsum += w;
    }
    (out, wsum)
}

//! The resampler implementations, one module each, numbered by the order they were written — plus
//! the primitives they are built from. A file under `impl/` holds only what makes *that* one
//! different: its kernel, and its tables if it has any. Anything two implementations would otherwise
//! both need is defined here once — the SIMD lane width, `sinc`, the Blackman window and its
//! cos-folded SIMD form, the `Phasor` rotation that supplies both without a transcendental per tap,
//! and the impulse-mode gather, which no implementation has yet had reason to do differently.
//! Sharing is not negotiable for the sake of keeping implementations independent: they exist to be
//! benchmarked against each other, and a duplicated helper that drifts makes the comparison measure
//! the drift instead of the design.

use core::f32::consts::PI;
use std::simd::prelude::*;

pub mod resample_impl_1;
pub mod resample_impl_2;

pub use resample_impl_1::ResampleImpl1;
pub use resample_impl_2::ResampleImpl2;

const LANES: usize = 4;

type Fv = Simd<f32, LANES>;

fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-7 {
        1.0
    } else {
        let px = PI * x;
        px.sin() / px
    }
}

fn blackman(x: f32) -> f32 {
    if x >= 1.0 {
        return 0.0;
    }
    0.42 + 0.5 * (PI * x).cos() + 0.08 * (2.0 * PI * x).cos()
}

#[inline]
fn blackman_from_cos(c: Fv) -> Fv {
    Fv::splat(0.34) + (Fv::splat(0.5) + Fv::splat(0.16) * c) * c
}

#[inline]
fn kernel_weight(d: f32, fc: f32, p: f32) -> f32 {
    sinc(2.0 * fc * d) * blackman(d.abs() / p)
}

struct Phasor {
    sin: Fv,
    cos: Fv,
    step_sin: f32,
    step_cos: f32,
}

impl Phasor {
    fn new(rate: f32, d0: f32) -> Self {
        let (mut sin, mut cos) = ([0.0; LANES], [0.0; LANES]);
        for i in 0..LANES {
            (sin[i], cos[i]) = f32::sin_cos(rate * (d0 - i as f32));
        }
        let (step_sin, step_cos) = f32::sin_cos(rate * LANES as f32);
        Self {
            sin: Fv::from_array(sin),
            cos: Fv::from_array(cos),
            step_sin,
            step_cos,
        }
    }

    #[inline]
    fn rotate(&mut self) {
        let (s, c) = (self.sin, self.cos);
        let (ss, sc) = (Fv::splat(self.step_sin), Fv::splat(self.step_cos));
        self.sin = s * sc - c * ss;
        self.cos = c * sc + s * ss;
    }
}

fn gather_impulse(src: &[f32], d0: f32, fc: f32, p: f32) -> (f32, f32) {
    let sinc_rate = PI * 2.0 * fc;
    let mut ph_sinc = Phasor::new(sinc_rate, d0);
    let mut ph_win = Phasor::new(PI / p, d0);

    let (mut out, mut wsum) = (Fv::splat(0.0), Fv::splat(0.0));
    let mut d = Fv::splat(d0) - Fv::from_array([0.0, 1.0, 2.0, 3.0]);
    for chunk in src.chunks_exact(LANES) {
        let arg = d * Fv::splat(sinc_rate);
        let near_zero = arg.abs().simd_lt(Fv::splat(1e-7));
        let lobe = near_zero.select(Fv::splat(1.0), ph_sinc.sin / arg);
        let inside = d.abs().simd_lt(Fv::splat(p));
        let w = inside.select(lobe * blackman_from_cos(ph_win.cos), Fv::splat(0.0));

        out += Fv::from_slice(chunk) * w;
        wsum += w;
        d -= Fv::splat(LANES as f32);
        ph_sinc.rotate();
        ph_win.rotate();
    }
    let (mut out, mut wsum) = (out.reduce_sum(), wsum.reduce_sum());

    let done = src.len() - src.chunks_exact(LANES).remainder().len();
    for (j, &s) in src.iter().enumerate().skip(done) {
        let d = d0 - j as f32;
        let w = kernel_weight(d, fc, p);
        out += s * w;
        wsum += w;
    }
    (out, wsum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blackman_folding_matches_the_direct_window() {
        for i in 0..=200 {
            let x = i as f32 / 200.0;
            let folded = blackman_from_cos(Fv::splat((PI * x).cos()))[0];
            assert!(
                (folded - blackman(x)).abs() < 1e-6,
                "blackman({x}): folded={folded}"
            );
        }
    }
}

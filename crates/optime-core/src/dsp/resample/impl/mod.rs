//! The resampler implementations, one module each, plus the primitives they are built from. A file
//! under `impl/` is named for how it computes, not for when it was written — `resample_impl_simd` is
//! the tabulated SIMD kernel, `resample_impl_simd_closed_form` swaps that table for evaluated
//! functions — and holds only what makes *that* one different: its kernel, and its tables if it has
//! any. Anything two of them would otherwise both need is defined here once: the SIMD lane width,
//! `sinc`, the Blackman window and its cos-folded SIMD form, the `Phasor` rotation that supplies
//! both without a transcendental per tap, and the impulse-mode gather, which no implementation has
//! yet had reason to do differently. Sharing is not negotiable for the sake of keeping
//! implementations independent: they exist to be benchmarked against each other, and a duplicated
//! helper that drifts makes the comparison measure the drift instead of the design.

use core::f32::consts::PI;
use std::simd::prelude::*;

pub mod resample_impl_simd;
pub mod resample_impl_simd_closed_form;

pub use resample_impl_simd::ResampleImplSimd;
pub use resample_impl_simd_closed_form::ResampleImplSimdClosedForm;

pub const DEFAULT_LANES: usize = 4;

pub type Fv<const N: usize> = Simd<f32, N>;

const fn lane_offsets<const N: usize>() -> Fv<N> {
    let mut offsets = [0.0f32; N];
    let mut i = 0;
    while i < N {
        offsets[i] = i as f32;
        i += 1;
    }
    Simd::from_array(offsets)
}

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
fn blackman_from_cos<const N: usize>(c: Fv<N>) -> Fv<N> {
    Simd::splat(0.34) + (Simd::splat(0.5) + Simd::splat(0.16) * c) * c
}

#[inline]
fn kernel_weight(d: f32, fc: f32, p: f32) -> f32 {
    sinc(2.0 * fc * d) * blackman(d.abs() / p)
}

struct Phasor<const N: usize> {
    sin: Fv<N>,
    cos: Fv<N>,
    step_sin: f32,
    step_cos: f32,
}

impl<const N: usize> Phasor<N> {
    fn new(rate: f32, d0: f32) -> Self {
        let (mut sin, mut cos) = ([0.0; N], [0.0; N]);
        for i in 0..N {
            (sin[i], cos[i]) = f32::sin_cos(rate * (d0 - i as f32));
        }
        let (step_sin, step_cos) = f32::sin_cos(rate * N as f32);
        Self {
            sin: Simd::from_array(sin),
            cos: Simd::from_array(cos),
            step_sin,
            step_cos,
        }
    }

    #[inline]
    fn rotate(&mut self) {
        let (s, c) = (self.sin, self.cos);
        let (ss, sc) = (Simd::splat(self.step_sin), Simd::splat(self.step_cos));
        self.sin = s * sc - c * ss;
        self.cos = c * sc + s * ss;
    }
}

fn gather_impulse<const N: usize>(src: &[f32], d0: f32, fc: f32, p: f32) -> (f32, f32) {
    let sinc_rate = PI * 2.0 * fc;
    let mut ph_sinc = Phasor::<N>::new(sinc_rate, d0);
    let mut ph_win = Phasor::<N>::new(PI / p, d0);

    let (mut out, mut wsum) = (Fv::<N>::splat(0.0), Fv::<N>::splat(0.0));
    let mut d = Fv::<N>::splat(d0) - lane_offsets::<N>();
    for chunk in src.chunks_exact(N) {
        let arg = d * Simd::splat(sinc_rate);
        let near_zero = arg.abs().simd_lt(Simd::splat(1e-7));
        let lobe = near_zero.select(Simd::splat(1.0), ph_sinc.sin / arg);
        let inside = d.abs().simd_lt(Simd::splat(p));
        let w = inside.select(lobe * blackman_from_cos(ph_win.cos), Simd::splat(0.0));

        out += Fv::<N>::from_slice(chunk) * w;
        wsum += w;
        d -= Simd::splat(N as f32);
        ph_sinc.rotate();
        ph_win.rotate();
    }
    let (mut out, mut wsum) = (out.reduce_sum(), wsum.reduce_sum());

    let done = src.len() - src.chunks_exact(N).remainder().len();
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
    use crate::dsp::resample::Resampler;

    const SOURCE_LEN: usize = 256;
    const SOURCE_PERIODS: usize = 13;
    const OUTPUT_LEN: usize = 4096;

    #[test]
    fn blackman_folding_matches_the_direct_window() {
        for i in 0..=200 {
            let x = i as f32 / 200.0;
            let folded = blackman_from_cos(Fv::<4>::splat((PI * x).cos()))[0];
            assert!(
                (folded - blackman(x)).abs() < 1e-6,
                "blackman({x}): folded={folded}"
            );
        }
    }

    fn looping_sine() -> Vec<f32> {
        (0..SOURCE_LEN)
            .map(|k| {
                let turns = SOURCE_PERIODS as f64 * k as f64 / SOURCE_LEN as f64;
                (std::f64::consts::TAU * turns).sin() as f32
            })
            .collect()
    }

    fn resample_sine<R: Resampler>(half_taps: usize, ratio: f64) -> Vec<f64> {
        let source = looping_sine();
        let tables = R::tables(half_taps);
        let fc = if ratio > 1.0 { 0.5 / ratio as f32 } else { 0.5 };
        (0..OUTPUT_LEN)
            .map(|n| {
                let pos = (half_taps as f64 + n as f64 * ratio) as f32;
                let (lo, hi) = R::tap_window(&tables, pos);
                let window: Vec<f32> = (lo..=hi)
                    .map(|k| source[k.rem_euclid(SOURCE_LEN as i64) as usize])
                    .collect();
                f64::from(R::resample(&tables, &window, pos, fc, false))
            })
            .collect()
    }

    fn spurious_free_snr_db(signal: &[f64], cycles_per_sample: f64) -> f64 {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (n, &x) in signal.iter().enumerate() {
            let phase = std::f64::consts::TAU * cycles_per_sample * n as f64;
            re += x * phase.cos();
            im += x * phase.sin();
        }
        let scale = 2.0 / signal.len() as f64;
        let (re, im) = (re * scale, im * scale);

        let (mut fundamental, mut residual) = (0.0f64, 0.0f64);
        for (n, &x) in signal.iter().enumerate() {
            let phase = std::f64::consts::TAU * cycles_per_sample * n as f64;
            let fitted = re * phase.cos() + im * phase.sin();
            fundamental += fitted * fitted;
            residual += (x - fitted) * (x - fitted);
        }
        10.0 * (fundamental / residual.max(f64::MIN_POSITIVE)).log10()
    }

    fn snr_db<R: Resampler>(half_taps: usize, ratio: f64) -> f64 {
        let out = resample_sine::<R>(half_taps, ratio);
        spurious_free_snr_db(&out, SOURCE_PERIODS as f64 / SOURCE_LEN as f64 * ratio)
    }

    const RATIOS: [f64; 4] = [7.0 / 16.0, 1.0, 43.0 / 32.0, 3.0];
    const CONTRACT_SNR_DB: f64 = 100.0;
    const MIN_HALF_TAPS_FOR_CONTRACT: usize = 16;

    #[test]
    fn every_implementation_resamples_above_100_db_snr() {
        for ratio in RATIOS {
            for half_taps in [MIN_HALF_TAPS_FOR_CONTRACT, 32, 64] {
                for (name, snr) in [
                    ("simd/4", snr_db::<ResampleImplSimd<4>>(half_taps, ratio)),
                    ("simd/8", snr_db::<ResampleImplSimd<8>>(half_taps, ratio)),
                    (
                        "closed/4",
                        snr_db::<ResampleImplSimdClosedForm<4>>(half_taps, ratio),
                    ),
                    (
                        "closed/8",
                        snr_db::<ResampleImplSimdClosedForm<8>>(half_taps, ratio),
                    ),
                ] {
                    assert!(
                        snr > CONTRACT_SNR_DB,
                        "{name}: half_taps={half_taps} ratio={ratio} gave {snr:.1} dB"
                    );
                }
            }
        }
    }

    #[test]
    fn the_contract_needs_at_least_sixteen_half_taps() {
        let thin = MIN_HALF_TAPS_FOR_CONTRACT / 2;
        let worst = RATIOS
            .iter()
            .map(|&r| snr_db::<ResampleImplSimd>(thin, r))
            .fold(f64::INFINITY, f64::min);
        assert!(
            worst < CONTRACT_SNR_DB,
            "{thin} half-taps reached {worst:.1} dB"
        );
    }
}

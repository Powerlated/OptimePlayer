//! The resampler implementations, one module each, plus the primitives they are built from. A file
//! under `impl/` is `resample_impl_<nn>_<technique>.rs` — the number is the order it was written, so
//! the directory reads as the sequence of experiments it is, and the suffix says how it computes.
//! Each holds only what makes *that* one different: its kernel, and its tables if it has
//! any. Anything two of them would otherwise both need is defined here once: the SIMD lane width,
//! `sinc`, the Blackman window and its cos-folded SIMD form, the `Phasor` rotation that supplies
//! both without a transcendental per tap, and the impulse-mode gather, which no implementation has
//! yet had reason to do differently. Sharing is not negotiable for the sake of keeping
//! implementations independent: they exist to be benchmarked against each other, and a duplicated
//! helper that drifts makes the comparison measure the drift instead of the design.

use core::f32::consts::PI;
use std::simd::prelude::*;

pub mod resample_impl_00_simd;
pub mod resample_impl_01_closed_form;

pub use resample_impl_00_simd::ResampleImplSimd;
pub use resample_impl_01_closed_form::ResampleImplSimdClosedForm;

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

#[cfg(test)]
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
fn load_partial<const N: usize>(src: &[f32]) -> Fv<N> {
    if src.len() >= N {
        Fv::<N>::from_slice(src)
    } else {
        let mut lanes = [0.0f32; N];
        lanes[..src.len()].copy_from_slice(src);
        Simd::from_array(lanes)
    }
}

#[inline]
fn occupied_lanes<const N: usize>(len: usize) -> Mask<i32, N> {
    lane_offsets::<N>().simd_lt(Simd::splat(len as f32))
}

fn sin_cos_fast(x: f32) -> (f32, f32) {
    let x = f64::from(x);
    let quadrants = (x * core::f64::consts::FRAC_2_PI).round();
    let r = x - quadrants * core::f64::consts::FRAC_PI_2;
    let z = r * r;
    let sin = r * (1.0 + z * (-1.0 / 6.0 + z * (1.0 / 120.0 + z * (-1.0 / 5040.0))));
    let cos = 1.0 + z * (-0.5 + z * (1.0 / 24.0 + z * (-1.0 / 720.0 + z * (1.0 / 40320.0))));
    let (sin, cos) = match (quadrants as i64).rem_euclid(4) {
        0 => (sin, cos),
        1 => (cos, -sin),
        2 => (-sin, -cos),
        _ => (-cos, sin),
    };
    (sin as f32, cos as f32)
}

struct Phasor<const N: usize> {
    sin: Fv<N>,
    cos: Fv<N>,
    step_sin: f32,
    step_cos: f32,
}

impl<const N: usize> Phasor<N> {
    fn new(rate: f32, d0: f32) -> Self {
        let (step_sin, step_cos) = sin_cos_fast(rate);
        let (mut sin, mut cos) = ([0.0; N], [0.0; N]);
        (sin[0], cos[0]) = sin_cos_fast(rate * d0);
        for i in 1..N {
            sin[i] = sin[i - 1] * step_cos - cos[i - 1] * step_sin;
            cos[i] = cos[i - 1] * step_cos + sin[i - 1] * step_sin;
        }
        let (mut lane_sin, mut lane_cos) = (step_sin, step_cos);
        for _ in 1..N {
            (lane_sin, lane_cos) = (
                lane_sin * step_cos + lane_cos * step_sin,
                lane_cos * step_cos - lane_sin * step_sin,
            );
        }
        Self {
            sin: Simd::from_array(sin),
            cos: Simd::from_array(cos),
            step_sin: lane_sin,
            step_cos: lane_cos,
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
    let mut rest = src;
    while !rest.is_empty() {
        let arg = d * Simd::splat(sinc_rate);
        let near_zero = arg.abs().simd_lt(Simd::splat(1e-7));
        let lobe = near_zero.select(Simd::splat(1.0), ph_sinc.sin / arg);
        let inside = occupied_lanes::<N>(rest.len()) & d.abs().simd_lt(Simd::splat(p));
        let w = inside.select(lobe * blackman_from_cos(ph_win.cos), Simd::splat(0.0));

        out += load_partial::<N>(rest) * w;
        wsum += w;
        d -= Simd::splat(N as f32);
        ph_sinc.rotate();
        ph_win.rotate();
        rest = &rest[rest.len().min(N)..];
    }
    (out.reduce_sum(), wsum.reduce_sum())
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

    #[test]
    fn fast_sin_cos_matches_the_library() {
        let mut worst: f64 = 0.0;
        for i in -400_000..=400_000 {
            let x = i as f32 / 1000.0;
            let (s, c) = sin_cos_fast(x);
            let (want_s, want_c) = f32::sin_cos(x);
            worst = worst
                .max(f64::from((s - want_s).abs()))
                .max(f64::from((c - want_c).abs()));
        }
        assert!(worst < 1e-6, "worst absolute error {worst}");
    }

    fn phasor_lanes<const N: usize>(rate: f32, d0: f32) -> ([f32; N], [f32; N], f32, f32) {
        let p = Phasor::<N>::new(rate, d0);
        (*p.sin.as_array(), *p.cos.as_array(), p.step_sin, p.step_cos)
    }

    #[test]
    fn phasor_seeding_matches_direct_trigonometry() {
        for rate in [0.03f32, PI / 16.0, PI, 2.0 * PI * 0.37, 5.9] {
            for d0 in [0.0f32, 0.5, 7.25, 63.9] {
                let (sin4, cos4, ss4, sc4) = phasor_lanes::<4>(rate, d0);
                let (sin8, cos8, ss8, sc8) = phasor_lanes::<8>(rate, d0);
                for (i, (&s, &c)) in sin8.iter().zip(&cos8).enumerate() {
                    let (want_s, want_c) = f32::sin_cos(rate * (d0 - i as f32));
                    assert!((s - want_s).abs() < 1e-4, "sin lane {i}: {s} vs {want_s}");
                    assert!((c - want_c).abs() < 1e-4, "cos lane {i}: {c} vs {want_c}");
                    if i < 4 {
                        assert!((sin4[i] - want_s).abs() < 1e-4);
                        assert!((cos4[i] - want_c).abs() < 1e-4);
                    }
                }
                for (lanes, ss, sc) in [(4.0, ss4, sc4), (8.0, ss8, sc8)] {
                    let (want_s, want_c) = f32::sin_cos(rate * lanes);
                    assert!((ss - want_s).abs() < 1e-4, "step sin: {ss} vs {want_s}");
                    assert!((sc - want_c).abs() < 1e-4, "step cos: {sc} vs {want_c}");
                }
            }
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

    fn alternating_square() -> Vec<f32> {
        (0..SOURCE_LEN)
            .map(|k| if k % 2 == 0 { 1.0 } else { -1.0 })
            .collect()
    }

    fn additive_square(t: f64, fc: f64) -> f64 {
        let mut sum = 0.0;
        let mut harmonic = 1;
        while harmonic as f64 * 0.5 < fc {
            sum += (std::f64::consts::PI * harmonic as f64 * t).sin() / harmonic as f64;
            harmonic += 2;
        }
        sum * 4.0 / std::f64::consts::PI
    }

    fn step_snr_db<R: Resampler>(half_taps: usize, ratio: f64) -> f64 {
        let source = alternating_square();
        let tables = R::tables(half_taps);
        let fc = 0.5 / ratio as f32;
        let (mut reference, mut residual) = (0.0f64, 0.0f64);
        for n in 0..OUTPUT_LEN {
            let pos = (half_taps as f64 + n as f64 * ratio) as f32;
            let (lo, hi) = R::tap_window(&tables, pos);
            let window: Vec<f32> = (lo..=hi)
                .map(|k| source[k.rem_euclid(SOURCE_LEN as i64) as usize])
                .collect();
            let got = f64::from(R::resample(&tables, &window, pos, fc, true));
            let want = additive_square(f64::from(pos), f64::from(fc));
            reference += want * want;
            residual += (got - want) * (got - want);
        }
        10.0 * (reference / residual.max(f64::MIN_POSITIVE)).log10()
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

    const STEP_RATIOS: [f64; 6] = [0.25, 0.26, 0.17, 0.4, 0.45, 0.7];
    const STEP_SPREAD_RATIOS: [f64; 4] = [0.26, 0.17, 0.45, 0.7];
    const STEP_CONTRACT_SNR_DB: f64 = 35.0;
    const UNTRUNCATED_STEP_SNR_DB: f64 = 85.0;

    #[test]
    fn step_mode_renders_a_band_limited_square_wave() {
        for ratio in STEP_RATIOS {
            for half_taps in [MIN_HALF_TAPS_FOR_CONTRACT, 32, 64] {
                for (name, snr) in [
                    (
                        "simd/4",
                        step_snr_db::<ResampleImplSimd<4>>(half_taps, ratio),
                    ),
                    (
                        "simd/8",
                        step_snr_db::<ResampleImplSimd<8>>(half_taps, ratio),
                    ),
                    (
                        "closed/4",
                        step_snr_db::<ResampleImplSimdClosedForm<4>>(half_taps, ratio),
                    ),
                    (
                        "closed/8",
                        step_snr_db::<ResampleImplSimdClosedForm<8>>(half_taps, ratio),
                    ),
                ] {
                    assert!(
                        snr > STEP_CONTRACT_SNR_DB,
                        "{name}: half_taps={half_taps} ratio={ratio} gave {snr:.1} dB"
                    );
                }
            }
        }
    }

    #[test]
    fn an_untruncated_step_kernel_sharpens_with_more_taps() {
        for ratio in STEP_SPREAD_RATIOS {
            let thin = step_snr_db::<ResampleImplSimdClosedForm>(MIN_HALF_TAPS_FOR_CONTRACT, ratio);
            let wide = step_snr_db::<ResampleImplSimdClosedForm>(64, ratio);
            assert!(wide > thin, "ratio={ratio}: {thin:.1} dB -> {wide:.1} dB");
            assert!(
                wide > UNTRUNCATED_STEP_SNR_DB,
                "ratio={ratio}: 64 half-taps gave {wide:.1} dB"
            );
        }
    }

    #[test]
    fn the_tabulated_step_kernel_degrades_past_its_table_reach() {
        let ratio = 0.26;
        let thin = step_snr_db::<ResampleImplSimd>(MIN_HALF_TAPS_FOR_CONTRACT, ratio);
        let wide = step_snr_db::<ResampleImplSimd>(64, ratio);
        assert!(
            wide < thin,
            "widening past the table reach should cost SNR: {thin:.1} dB -> {wide:.1} dB"
        );
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

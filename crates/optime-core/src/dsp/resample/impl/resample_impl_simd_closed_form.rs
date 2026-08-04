//! The same windowed-sinc contract as `resample_impl_simd`, and the same SIMD lanes, but with the
//! table replaced by evaluated functions — and reached from the other side of the convolution to
//! make that possible. The source is read as a train of scaled Dirac deltas, and the
//! zero-order hold that turns that train into a staircase is folded into the reconstruction kernel
//! instead of into the signal — so step mode convolves the deltas with a *band-limited rect*, one
//! rect per source sample, stretched or squeezed by the cutoff the resample ratio asks for. Impulse
//! mode is that same expression with the rect collapsed to a point, which is the bare windowed sinc
//! every implementation already agrees on, so it defers to the shared `gather_impulse`.
//!
//! The band-limited rect is the difference of the band-limited step at the rect's two edges, and
//! that step is the sine integral. `resample_impl_simd` tables it and reads the table with a SIMD
//! gather; this file evaluates it instead — Taylor series near the origin, Padé approximants of the
//! auxiliary asymptotic series beyond it — so nothing here touches memory outside the tap window.
//! That is the whole point of having both: same output within float tolerance, opposite answers to
//! whether a kernel should cost a table lookup or arithmetic.

use core::f32::consts::{FRAC_PI_2, PI};
use std::simd::prelude::*;

use super::{Fv, LANES, Phasor, blackman, blackman_from_cos, gather_impulse};
use crate::dsp::resample::{MAX_HALF_TAPS, Resampler};
use crate::waveform::Sample;

const SI_TAYLOR_TERMS: usize = 32;
const SI_TAYLOR_LIMIT: f32 = 16.0;

const F_NUM_1: f32 = 214.0 / 3.0;
const F_NUM_2: f32 = 1192.0 / 3.0;
const F_DEN_1: f32 = 220.0 / 3.0;
const F_DEN_2: f32 = 520.0;
const G_NUM_1: f32 = 1026.0 / 11.0;
const G_NUM_2: f32 = 7368.0 / 11.0;
const G_DEN_1: f32 = 1092.0 / 11.0;
const G_DEN_2: f32 = 12600.0 / 11.0;

const SI_TAYLOR: [f64; SI_TAYLOR_TERMS] = si_taylor_coefficients();

const fn si_taylor_coefficients() -> [f64; SI_TAYLOR_TERMS] {
    let mut coefficients = [0.0f64; SI_TAYLOR_TERMS];
    let mut odd_factorial = 1.0f64;
    let mut k = 0;
    while k < SI_TAYLOR_TERMS {
        let odd = (2 * k + 1) as f64;
        if k > 0 {
            odd_factorial *= (2 * k) as f64 * odd;
        }
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        coefficients[k] = sign / (odd * odd_factorial);
        k += 1;
    }
    coefficients
}

pub struct ResampleImplSimdClosedForm;

#[derive(Clone)]
pub struct Tables {
    pub half_taps: usize,
}

impl Resampler for ResampleImplSimdClosedForm {
    type Tables = Tables;

    fn tables(half_taps: usize) -> Tables {
        Tables {
            half_taps: half_taps.clamp(1, MAX_HALF_TAPS),
        }
    }

    #[inline]
    fn half_taps(tables: &Tables) -> usize {
        tables.half_taps
    }

    #[inline]
    fn tap_window(tables: &Tables, pos: f32) -> (i64, i64) {
        let p = tables.half_taps as f32;
        ((pos - p).floor() as i64, (pos + p).ceil() as i64)
    }

    fn resample(tables: &Tables, src: &[f32], pos: f32, fc: f32, step_mode: bool) -> Sample {
        let (k_lo, k_hi) = Self::tap_window(tables, pos);
        debug_assert_eq!(
            src.len() as i64,
            k_hi - k_lo + 1,
            "src must cover the tap window"
        );

        let fc = if step_mode {
            fc.max(1e-6)
        } else {
            fc.clamp(1e-6, 0.5)
        };

        let d0 = pos - k_lo as f32;
        let p = tables.half_taps as f32;
        let (out, wsum) = if step_mode {
            convolve_rects(src, d0, fc, p)
        } else {
            gather_impulse(src, d0, fc, p)
        };
        let wsum = if step_mode { wsum.abs() } else { wsum };

        if wsum > 1e-12 {
            out / wsum
        } else {
            Sample::from(src[(pos.round() as i64 - k_lo) as usize])
        }
    }
}

fn band_limited_step(t: f32) -> f32 {
    let a = t.abs();
    let si = if a <= SI_TAYLOR_LIMIT {
        si_near_origin(a)
    } else {
        let (sin_a, cos_a) = f32::sin_cos(a);
        si_asymptotic(a, sin_a, cos_a)
    };
    let si = if t < 0.0 { -si } else { si };
    si / PI
}

fn si_near_origin(a: f32) -> f32 {
    let a = f64::from(a);
    let z = a * a;
    let mut acc = SI_TAYLOR[SI_TAYLOR_TERMS - 1];
    for &c in SI_TAYLOR[..SI_TAYLOR_TERMS - 1].iter().rev() {
        acc = acc * z + c;
    }
    (a * acc) as f32
}

#[inline]
fn si_asymptotic(a: f32, sin_a: f32, cos_a: f32) -> f32 {
    let y = 1.0 / (a * a);
    let f = (1.0 + y * (F_NUM_1 + y * F_NUM_2)) / (a * (1.0 + y * (F_DEN_1 + y * F_DEN_2)));
    let g = y * (1.0 + y * (G_NUM_1 + y * G_NUM_2)) / (1.0 + y * (G_DEN_1 + y * G_DEN_2));
    FRAC_PI_2 - f * cos_a - g * sin_a
}

#[inline]
fn band_limited_step_simd(t: Fv, sin_t: Fv, cos_t: Fv) -> Fv {
    let one = Fv::splat(1.0);
    let a = t.abs();
    let negative = t.simd_lt(Fv::splat(0.0));

    let y = one / (a * a);
    let f = (one + y * (Fv::splat(F_NUM_1) + y * Fv::splat(F_NUM_2)))
        / (a * (one + y * (Fv::splat(F_DEN_1) + y * Fv::splat(F_DEN_2))));
    let g = y * (one + y * (Fv::splat(G_NUM_1) + y * Fv::splat(G_NUM_2)))
        / (one + y * (Fv::splat(G_DEN_1) + y * Fv::splat(G_DEN_2)));
    let mut si = Fv::splat(FRAC_PI_2) - f * cos_t - g * negative.select(-sin_t, sin_t);

    let near_origin = a.simd_le(Fv::splat(SI_TAYLOR_LIMIT));
    if near_origin.any() {
        for i in 0..LANES {
            if near_origin.test(i) {
                si.as_mut_array()[i] = si_near_origin(a[i]);
            }
        }
    }
    negative.select(-si, si) * Fv::splat(1.0 / PI)
}

fn convolve_rects(src: &[f32], d0: f32, fc: f32, p: f32) -> (f32, f32) {
    let lane_offsets = Fv::from_array([0.0, 1.0, 2.0, 3.0]);
    let edge_rate = 2.0 * PI * fc;
    let mut edges = Phasor::new(edge_rate, d0 - 1.0);
    let mut window = Phasor::new(PI / p, d0 - 0.5);
    let (mut out, mut wsum) = (Fv::splat(0.0), Fv::splat(0.0));
    let mut carry = band_limited_step(edge_rate * d0);
    let mut base = d0 - 1.0;

    for chunk in src.chunks_exact(LANES) {
        let d_lower = Fv::splat(base) - lane_offsets;
        let lower = band_limited_step_simd(d_lower * Fv::splat(edge_rate), edges.sin, edges.cos);
        let mut upper = lower.rotate_elements_right::<1>();
        upper.as_mut_array()[0] = carry;
        carry = lower.as_array()[LANES - 1];

        let mid = Fv::splat(base + 0.5) - lane_offsets;
        let inside = mid.abs().simd_lt(Fv::splat(p));
        let w = inside.select(
            blackman_from_cos(window.cos) * (upper - lower),
            Fv::splat(0.0),
        );

        out += Fv::from_slice(chunk) * w;
        wsum += w;
        base -= LANES as f32;
        edges.rotate();
        window.rotate();
    }
    let (mut out, mut wsum) = (out.reduce_sum(), wsum.reduce_sum());

    let done = src.len() - src.chunks_exact(LANES).remainder().len();
    let mut upper = carry;
    for (j, &s) in src.iter().enumerate().skip(done) {
        let d = d0 - j as f32;
        let lower = band_limited_step(edge_rate * (d - 1.0));
        let w = blackman((d - 0.5).abs() / p) * (upper - lower);
        out += s * w;
        wsum += w;
        upper = lower;
    }
    (out, wsum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::resample::ResampleImplSimd;

    fn simpson(steps: usize, lo: f64, hi: f64, f: impl Fn(f64) -> f64) -> f64 {
        let h = (hi - lo) / steps as f64;
        let mut sum = f(lo) + f(hi);
        for i in 1..steps {
            sum += f(lo + i as f64 * h) * if i % 2 == 0 { 2.0 } else { 4.0 };
        }
        sum * h / 3.0
    }

    fn si_reference(t: f64) -> f64 {
        simpson(200_000, 0.0, t, |u| {
            if u.abs() < 1e-12 { 1.0 } else { u.sin() / u }
        })
    }

    #[test]
    fn band_limited_step_matches_the_sine_integral() {
        let mut worst: f32 = 0.0;
        for i in 0..=100 {
            let t = i as f32 * 1.0;
            let want = (si_reference(f64::from(t)) / std::f64::consts::PI) as f32;
            let got = band_limited_step(t);
            worst = worst.max((got - want).abs());
            assert_eq!(band_limited_step(-t), -got, "must be odd at {t}");
        }
        assert!(worst < 5e-7, "worst absolute error {worst}");
    }

    fn integrated_rect_reference(src: &[f32], d0: f64, fc: f64, p: f64) -> f64 {
        let kernel = |t: f64| {
            let x = 2.0 * fc * t;
            if x.abs() < 1e-12 {
                1.0
            } else {
                (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
            }
        };
        let (mut out, mut wsum) = (0.0, 0.0);
        for (j, &s) in src.iter().enumerate() {
            let d = d0 - j as f64;
            let mid = (d - 0.5).abs() / p;
            if mid >= 1.0 {
                continue;
            }
            let window = 0.42
                + 0.5 * (std::f64::consts::PI * mid).cos()
                + 0.08 * (2.0 * std::f64::consts::PI * mid).cos();
            let w = window * simpson(2048, d - 1.0, d, kernel);
            out += f64::from(s) * w;
            wsum += w;
        }
        out / wsum.abs()
    }

    #[test]
    fn step_mode_convolves_with_numerically_integrated_rects() {
        let data: Vec<f32> = (0..256)
            .map(|k| (0.7 * (k as f32) + 0.3 * (k as f32).sin()).sin())
            .collect();
        let mut worst: f64 = 0.0;
        for half_taps in [8usize, 32] {
            for fc in [0.25f32, 1.4, 2.9] {
                let tables = ResampleImplSimdClosedForm::tables(half_taps);
                for i in 0..8 {
                    let pos = half_taps as f32 + i as f32 * 0.41;
                    let (lo, hi) = ResampleImplSimdClosedForm::tap_window(&tables, pos);
                    let src: Vec<f32> = (lo..=hi)
                        .map(|k| data[k.rem_euclid(data.len() as i64) as usize])
                        .collect();
                    let got = ResampleImplSimdClosedForm::resample(&tables, &src, pos, fc, true);
                    let want = integrated_rect_reference(
                        &src,
                        f64::from(pos - lo as f32),
                        f64::from(fc),
                        half_taps as f64,
                    );
                    worst = worst.max((f64::from(got) - want).abs());
                }
            }
        }
        assert!(worst < 1e-5, "worst absolute error {worst}");
    }

    #[test]
    fn simd_and_scalar_band_limited_steps_agree() {
        for start in [-40.0f32, -8.5, -0.3, 0.0, 3.0, 9.0, 61.7] {
            let t = Fv::from_array([start, start + 1.0, start + 2.0, start + 3.0]);
            let (mut sin, mut cos) = ([0.0f32; LANES], [0.0f32; LANES]);
            for i in 0..LANES {
                (sin[i], cos[i]) = f32::sin_cos(t[i]);
            }
            let got = band_limited_step_simd(t, Fv::from_array(sin), Fv::from_array(cos));
            for i in 0..LANES {
                let want = band_limited_step(t[i]);
                assert!(
                    (got[i] - want).abs() < 2e-6,
                    "t={}: simd {} vs scalar {want}",
                    t[i],
                    got[i]
                );
            }
        }
    }

    fn agreement(half_taps: usize, fc: f32, step_mode: bool, data: &[f32]) -> f32 {
        let one = ResampleImplSimd::tables(half_taps);
        let two = ResampleImplSimdClosedForm::tables(half_taps);
        let mut worst: f32 = 0.0;
        for i in 0..64 {
            let pos = half_taps as f32 + i as f32 * 0.37;
            let (lo, hi) = ResampleImplSimd::tap_window(&one, pos);
            assert_eq!((lo, hi), ResampleImplSimdClosedForm::tap_window(&two, pos));
            let src: Vec<f32> = (lo..=hi)
                .map(|k| data[k.rem_euclid(data.len() as i64) as usize])
                .collect();
            let a = ResampleImplSimd::resample(&one, &src, pos, fc, step_mode);
            let b = ResampleImplSimdClosedForm::resample(&two, &src, pos, fc, step_mode);
            worst = worst.max((a - b).abs());
        }
        worst
    }

    const TABULATED_KERNEL_REACH: f32 = 64.0;

    #[test]
    fn both_implementations_agree_where_the_tabulated_kernel_reaches() {
        let data: Vec<f32> = (0..256)
            .map(|k| (0.7 * (k as f32) + 0.3 * (k as f32).sin()).sin())
            .collect();
        for half_taps in [4usize, 16, 32] {
            for fc in [0.05f32, 0.25, 0.5, 1.4, 2.9] {
                for step_mode in [false, true] {
                    if !step_mode && fc > 0.5 {
                        continue;
                    }
                    if 2.0 * fc * half_taps as f32 > TABULATED_KERNEL_REACH {
                        continue;
                    }
                    let worst = agreement(half_taps, fc, step_mode, &data);
                    assert!(
                        worst < 2e-5,
                        "half_taps={half_taps} fc={fc} step_mode={step_mode}: worst {worst}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_tabulated_implementation_truncates_taps_beyond_its_kernel() {
        let data: Vec<f32> = (0..256)
            .map(|k| (0.7 * (k as f32) + 0.3 * (k as f32).sin()).sin())
            .collect();
        let (half_taps, fc) = (32usize, 2.9f32);
        assert!(2.0 * fc * half_taps as f32 > TABULATED_KERNEL_REACH);
        assert!(agreement(half_taps, fc, true, &data) > 1e-4);
    }

    #[test]
    fn a_square_wave_keeps_its_band_limited_edges() {
        let period = 8usize;
        let data: Vec<f32> = (0..256)
            .map(|k| if (k % period) < period / 2 { 1.0 } else { -1.0 })
            .collect();
        let worst = agreement(32, 0.5 / 3.0, true, &data);
        assert!(worst < 2e-5, "worst {worst}");
    }
}

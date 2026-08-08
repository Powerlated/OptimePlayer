//! The tabulated SIMD resampler, and the default: Blackman-windowed sinc in two modes. Impulse mode
//! is ordinary band-limited interpolation — each source sample weighted by a sinc lobe at its
//! distance from the read position. Step mode treats the source as a zero-order hold instead,
//! weighting each sample by the *difference* of the integrated sinc across the sample it spans, so a
//! square wave's edges come out band-limited rather than ringing; that is what PSG voices and the
//! crunchy output-Nyquist mode want. Both normalise by the summed weights, so an arbitrary cutoff
//! never shifts the gain.
//!
//! The integrated sinc is the only table, and the whole of what this file adds: `OVERSAMPLE` entries
//! per sample of lag out to `TAU_MAX`, built once, Kahan-summed and rescaled so its tail lands
//! exactly on the half it must converge to (a uniform scale, which the weight normalisation cancels).
//! Step mode reads it with a SIMD gather, one interpolated lookup per rect edge, each edge shared
//! with the neighbouring tap. Impulse mode needs no table and is the shared `gather_impulse`.
//!
//! `TAU_MAX` is therefore a real, finite kernel support, not just a table size: past it both of a
//! tap's edges saturate to the same half and the tap's weight is exactly zero. `taps_within_reach`
//! finds where that starts and the gather never visits those taps, which at the cutoffs heavy
//! upsampling asks for is most of a wide tap window.

use core::f32::consts::PI;
use std::simd::prelude::*;
use std::sync::OnceLock;

use super::{
    DEFAULT_LANES, Fv, Phasor, blackman, blackman_from_cos, gather_impulse, lane_offsets, sinc,
};
use crate::dsp::resample::{MAX_HALF_TAPS, Resampler};
use crate::waveform::Sample;

pub struct ResampleImplSimd<const LANES: usize = DEFAULT_LANES>;

#[derive(Clone)]
pub struct Tables {
    pub half_taps: usize,
}

impl<const LANES: usize> Resampler for ResampleImplSimd<LANES> {
    type Tables = Tables;

    fn tables(half_taps: usize) -> Tables {
        let _ = kernels();
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
        let k = kernels();
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
        let sinc_idx_step = 2.0 * fc * OVERSAMPLE as f32;

        let d0 = pos - k_lo as f32;
        let p = tables.half_taps as f32;
        let (out, wsum): (Sample, Sample) = if step_mode {
            let support = kernel_support(k, sinc_idx_step);
            let (first, last) = taps_within_reach::<LANES>(d0, support, src.len());
            gather_step::<LANES>(k, &src[first..last], d0 - first as f32, sinc_idx_step, p)
        } else {
            gather_impulse::<LANES>(src, d0, fc, p)
        };
        let wsum = if step_mode { wsum.abs() } else { wsum };

        if wsum > 1e-12 {
            out / wsum
        } else {
            Sample::from(src[(pos.round() as i64 - k_lo) as usize])
        }
    }
}

const OVERSAMPLE: usize = 512;
const TAU_MAX: usize = 64;

const _: () = assert!(MAX_HALF_TAPS <= TAU_MAX);

struct Kernels {
    sinc_int: Vec<f32>,
}

fn kernels() -> &'static Kernels {
    static K: OnceLock<Kernels> = OnceLock::new();
    K.get_or_init(|| {
        let len = TAU_MAX * OVERSAMPLE;

        let sinc_tab: Vec<f32> = (0..=len)
            .map(|k| sinc(k as f32 / OVERSAMPLE as f32))
            .collect();

        let step = 1.0 / OVERSAMPLE as f32;
        let mut sinc_int = vec![0.0f32; len + 1];
        let mut sum = 0.0f32;
        let mut comp = 0.0f32;
        for k in 1..=len {
            let trap = (sinc_tab[k - 1] + sinc_tab[k]) * 0.5 * step;
            let y = trap - comp;
            let t = sum + y;
            comp = (t - sum) - y;
            sum = t;
            sinc_int[k] = sum;
        }
        let tail = sinc_int[len];
        if tail > 1e-12 {
            let scale = 0.5 / tail;
            for v in &mut sinc_int {
                *v *= scale;
            }
        }

        Kernels { sinc_int }
    })
}

#[inline]
fn lerp(tab: &[f32], idx: f32) -> f32 {
    let i = idx as usize;
    let frac = idx - i as f32;
    let lo = tab[i];
    lo + (tab[i + 1] - lo) * frac
}

#[inline]
fn table_reach(k: &Kernels) -> f32 {
    (k.sinc_int.len() - 1) as f32
}

#[inline]
fn sinc_int_at(k: &Kernels, idx: f32) -> f32 {
    let mag = idx.abs();
    let v = if mag >= table_reach(k) {
        0.5
    } else {
        lerp(&k.sinc_int, mag)
    };
    v.copysign(idx)
}

#[inline]
fn sinc_int_simd<const N: usize>(k: &Kernels, idx: Fv<N>) -> Fv<N> {
    let mag = idx.abs();
    let past_end = mag.simd_ge(Simd::splat(table_reach(k)));
    let mag = past_end.select(Simd::splat(0.0), mag);
    let i = mag.cast::<usize>();
    let frac = mag - i.cast::<f32>();
    let lo = Fv::<N>::gather_or_default(&k.sinc_int, i);
    let hi = Fv::<N>::gather_or_default(&k.sinc_int, i + Simd::splat(1));
    past_end
        .select(Simd::splat(0.5), lo + (hi - lo) * frac)
        .copysign(idx)
}

#[inline]
fn kernel_support(k: &Kernels, sinc_idx_step: f32) -> f32 {
    table_reach(k) / sinc_idx_step
}

fn taps_within_reach<const N: usize>(d0: f32, support: f32, taps: usize) -> (usize, usize) {
    let clamp = |edge: f32| edge.clamp(0.0, taps as f32) as usize;
    let first = clamp(d0 - support - 1.0) / N * N;
    let last = (clamp(d0 + support) + 2).next_multiple_of(N).min(taps);
    (first.min(last), last)
}

fn gather_step<const N: usize>(
    k: &Kernels,
    src: &[f32],
    d0: f32,
    sinc_idx_step: f32,
    p: f32,
) -> (f32, f32) {
    let mut ph_win = Phasor::<N>::new(PI / p, d0 - 0.5);
    let (mut out, mut wsum) = (Fv::<N>::splat(0.0), Fv::<N>::splat(0.0));
    let mut carry = sinc_int_at(k, sinc_idx_step * d0);
    let mut base = d0 - 1.0;

    for chunk in src.chunks_exact(N) {
        let d_lo = Fv::<N>::splat(base) - lane_offsets::<N>();
        let s_lo = sinc_int_simd::<N>(k, d_lo * Simd::splat(sinc_idx_step));
        let mut s_hi = s_lo.rotate_elements_right::<1>();
        s_hi.as_mut_array()[0] = carry;
        carry = s_lo.as_array()[N - 1];

        let d_mid = Fv::<N>::splat(base + 0.5) - lane_offsets::<N>();
        let inside = d_mid.abs().simd_lt(Simd::splat(p));
        let w = inside.select(
            blackman_from_cos(ph_win.cos) * (s_hi - s_lo),
            Simd::splat(0.0),
        );

        out += Fv::<N>::from_slice(chunk) * w;
        wsum += w;
        base -= N as f32;
        ph_win.rotate();
    }
    let (mut out, mut wsum) = (out.reduce_sum(), wsum.reduce_sum());

    let done = src.len() - src.chunks_exact(N).remainder().len();
    let mut si_hi = carry;
    for (j, &s) in src.iter().enumerate().skip(done) {
        let d_hi = d0 - j as f32;
        let si_lo = sinc_int_at(k, sinc_idx_step * (d_hi - 1.0));
        let w = blackman((d_hi - 0.5).abs() / p) * (si_hi - si_lo);
        out += s * w;
        wsum += w;
        si_hi = si_lo;
    }
    (out, wsum)
}

#[cfg(test)]
mod tests {

    use core::f64::consts::PI;

    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn staged(tables: &Tables, pos: f64, f: impl Fn(i64) -> f64) -> Vec<f32> {
        let (k_lo, k_hi) = ResampleImplSimd::<4>::tap_window(tables, pos as f32);
        (k_lo..=k_hi).map(|t| f(t) as f32).collect()
    }

    #[test]
    fn sinc_int_boundary_and_symmetry() {
        let k = kernels();
        let si = |tau: f64| f64::from(sinc_int_at(k, (tau * OVERSAMPLE as f64) as f32));
        assert!(close(si(0.0), 0.0, 1e-12));
        assert!(close(si(TAU_MAX as f64 + 5.0), 0.5, 1e-6));
        assert!(close(si(-(TAU_MAX as f64) - 5.0), -0.5, 1e-6));
        for i in 1..=20 {
            let tau = TAU_MAX as f64 * i as f64 / 20.0;
            assert!(close(si(tau) + si(-tau), 0.0, 1e-12), "S({tau}) not odd");
        }
    }

    #[test]
    fn sinc_int_is_bounded() {
        let k = kernels();
        for i in 0..=2000 {
            let tau = -(TAU_MAX as f64) + 2.0 * TAU_MAX as f64 * i as f64 / 2000.0;
            let s = f64::from(sinc_int_at(k, (tau * OVERSAMPLE as f64) as f32));
            assert!((-0.6..=0.6).contains(&s), "S({tau}) = {s} out of band");
        }
    }

    #[test]
    fn step_gather_matches_a_scalar_oracle() {
        let k = kernels();
        let oracle = |src: &[f32], d0: f32, sinc_idx_step: f32, p: f32| -> (f64, f64) {
            let (mut out, mut wsum) = (0.0f64, 0.0f64);
            for (j, &s) in src.iter().enumerate() {
                let d_hi = d0 - j as f32;
                let rise = sinc_int_at(k, sinc_idx_step * d_hi)
                    - sinc_int_at(k, sinc_idx_step * (d_hi - 1.0));
                let w = f64::from(blackman((d_hi - 0.5).abs() / p) * rise);
                out += f64::from(s) * w;
                wsum += w;
            }
            (out, wsum)
        };

        let mut seed = 0x1234_5678u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5
        };

        for p in [1usize, 3, 16, 32, 64] {
            for n in [2 * p, 2 * p + 1, 2 * p + 2] {
                for fc in [0.5, 0.25, 0.5 / 8.0, 1.5] {
                    for frac in [0.0, 0.13, 0.5, 0.87] {
                        let src: Vec<f32> = (0..n).map(|_| next()).collect();
                        let d0 = p as f32 + frac;
                        let step = 2.0 * fc * OVERSAMPLE as f32;
                        let (got_o, got_w) = gather_step::<4>(k, &src, d0, step, p as f32);
                        let (want_o, want_w) = oracle(&src, d0, step, p as f32);
                        let scale = want_w.abs().max(1e-3);
                        assert!(
                            close(f64::from(got_o), want_o, 1e-5 * scale),
                            "out at p={p} n={n} fc={fc} frac={frac}: {got_o} vs {want_o}"
                        );
                        assert!(
                            close(f64::from(got_w), want_w, 1e-5 * scale),
                            "wsum at p={p} n={n} fc={fc} frac={frac}: {got_w} vs {want_w}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn step_mode_preserves_dc() {
        let tables = ResampleImplSimd::<4>::tables(16);
        for fc in [0.1, 0.25, 0.5, 1.5] {
            for pos in [3.0, 7.35, 20.7] {
                let src = staged(&tables, pos, |_| 1.0);
                let out = f64::from(ResampleImplSimd::<4>::resample(
                    &tables, &src, pos as f32, fc as f32, true,
                ));
                assert!(close(out, 1.0, 1e-9), "DC at fc={fc}, pos={pos}: {out}");
            }
        }
    }

    #[test]
    fn step_mode_is_a_bandlimited_step() {
        let tables = ResampleImplSimd::<4>::tables(32);
        let fc = 0.5 / 4.0;
        let step = |k: i64| if k >= 0 { 1.0_f64 } else { 0.0 };
        let at = |pos: f64| {
            f64::from(ResampleImplSimd::<4>::resample(
                &tables,
                &staged(&tables, pos, step),
                pos as f32,
                fc as f32,
                true,
            ))
        };

        assert!(close(at(0.0), 0.5, 0.02));
        let half_width = tables.half_taps as f64 / (2.0 * fc);
        assert!(close(at(-(half_width + 10.0)), 0.0, 1e-6));
        assert!(close(at(half_width + 10.0), 1.0, 1e-6));
        let mut prev = -1.0;
        for i in 0..=200 {
            let pos = -40.0 + 80.0 * i as f64 / 200.0;
            let v = at(pos);
            assert!(
                v > prev - 0.05,
                "non-monotone at pos={pos}: {v} after {prev}"
            );
            prev = v;
        }
    }

    #[test]
    fn impulse_mode_dc_gain() {
        let tables = ResampleImplSimd::<4>::tables(16);
        let pos = 12.37;
        let src = staged(&tables, pos, |_| 1.0);
        let out = f64::from(ResampleImplSimd::<4>::resample(
            &tables, &src, pos as f32, 0.4, false,
        ));
        assert!(close(out, 1.0, 1e-6), "DC gain = {out}");
    }

    #[test]
    fn impulse_mode_passband_signal_reconstructed() {
        let tables = ResampleImplSimd::<4>::tables(16);
        let fc = 0.45;
        let f0 = 0.05;
        let get = |k: i64| (2.0 * PI * f0 * k as f64).cos();
        for frac in [0.0, 0.25, 0.5, 0.75] {
            let pos = 32.0 + frac;
            let ideal = (2.0 * PI * f0 * pos).cos();
            let out = f64::from(ResampleImplSimd::<4>::resample(
                &tables,
                &staged(&tables, pos, get),
                pos as f32,
                fc as f32,
                false,
            ));
            assert!(
                close(out, ideal, 1e-3),
                "at pos={pos}: reconstructed={out}, ideal={ideal}"
            );
        }
    }

    #[test]
    fn fixed_tap_count_independent_of_ratio() {
        let p = 16usize;
        for fc in [0.5, 0.5 / 4.0, 0.5 / 8.0] {
            let pos = 100.37_f64;
            let k_lo = (pos - p as f64).floor() as i64;
            let k_hi = (pos + p as f64).ceil() as i64;
            let n = (k_lo..=k_hi).count();
            assert!(
                (2 * p..=2 * p + 2).contains(&n),
                "fc={fc}: {n} taps (expected ≈{})",
                2 * p
            );
        }
    }
}

//! Resampler implementation 1: Blackman-windowed sinc, in two modes. Impulse mode is ordinary
//! band-limited interpolation — each source sample weighted by a sinc lobe at its distance from the
//! read position. Step mode treats the source as a zero-order hold instead, weighting each sample by
//! the *difference* of the integrated sinc across the sample it spans, so a square wave's edges come
//! out band-limited rather than ringing; that is what PSG voices and the crunchy output-Nyquist mode
//! want. Both normalise by the summed weights, so an arbitrary cutoff never shifts the gain.
//!
//! The integrated sinc is the only table: `OVERSAMPLE` entries per sample of lag, built once,
//! Kahan-summed and rescaled so its tail lands exactly on the half it must converge to. Impulse mode
//! needs no table at all — its sinc and window both come from a `Phasor`, a complex rotation stepped
//! four lanes at a time, which trades a transcendental per tap for two multiplies and turns the
//! whole gather into SIMD over the tap window with a scalar tail.

use core::f32::consts::PI;
use std::simd::prelude::*;
use std::sync::OnceLock;

use crate::dsp::resample::{MAX_HALF_TAPS, Resampler};
use crate::waveform::Sample;

const OVERSAMPLE: usize = 512;
const TAU_MAX: usize = 64;
const LANES: usize = 4;

const _: () = assert!(MAX_HALF_TAPS <= TAU_MAX);

type Fv = Simd<f32, LANES>;

pub struct ResampleImpl1;

#[derive(Clone)]
pub struct Tables {
    pub half_taps: usize,
}

impl Resampler for ResampleImpl1 {
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
            gather_step(k, src, d0, sinc_idx_step, p)
        } else {
            gather_impulse_simd(src, d0, fc, p)
        };
        let wsum = if step_mode { wsum.abs() } else { wsum };

        if wsum > 1e-12 {
            out / wsum
        } else {
            Sample::from(src[(pos.round() as i64 - k_lo) as usize])
        }
    }
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
fn kernel_weight(d: f32, fc: f32, p: f32) -> f32 {
    sinc(2.0 * fc * d) * blackman(d.abs() / p)
}

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
fn sinc_int_at(k: &Kernels, idx: f32) -> f32 {
    let mag = idx.abs();
    let v = if mag >= (k.sinc_int.len() - 1) as f32 {
        0.5
    } else {
        lerp(&k.sinc_int, mag)
    };
    if idx < 0.0 { -v } else { v }
}

#[inline]
fn sinc_int_simd(k: &Kernels, idx: Fv) -> Fv {
    let mag = idx.abs();
    let past_end = mag.simd_ge(Fv::splat((k.sinc_int.len() - 1) as f32));
    let mag = past_end.select(Fv::splat(0.0), mag);
    let i = mag.cast::<usize>();
    let frac = mag - i.cast::<f32>();
    let lo = Fv::gather_or_default(&k.sinc_int, i);
    let hi = Fv::gather_or_default(&k.sinc_int, i + Simd::splat(1));
    let v = past_end.select(Fv::splat(0.5), lo + (hi - lo) * frac);
    idx.simd_lt(Fv::splat(0.0)).select(-v, v)
}

#[inline]
fn blackman_from_cos(c: Fv) -> Fv {
    Fv::splat(0.34) + (Fv::splat(0.5) + Fv::splat(0.16) * c) * c
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

fn gather_step(k: &Kernels, src: &[f32], d0: f32, sinc_idx_step: f32, p: f32) -> (f32, f32) {
    let lane_offsets = Fv::from_array([0.0, 1.0, 2.0, 3.0]);
    let mut ph_win = Phasor::new(PI / p, d0 - 0.5);
    let (mut out, mut wsum) = (Fv::splat(0.0), Fv::splat(0.0));
    let mut carry = sinc_int_at(k, sinc_idx_step * d0);
    let mut base = d0 - 1.0;

    for chunk in src.chunks_exact(LANES) {
        let d_lo = Fv::splat(base) - lane_offsets;
        let s_lo = sinc_int_simd(k, d_lo * Fv::splat(sinc_idx_step));
        let mut s_hi = s_lo.rotate_elements_right::<1>();
        s_hi.as_mut_array()[0] = carry;
        carry = s_lo.as_array()[LANES - 1];

        let d_mid = Fv::splat(base + 0.5) - lane_offsets;
        let inside = d_mid.abs().simd_lt(Fv::splat(p));
        let w = inside.select(
            blackman_from_cos(ph_win.cos) * (s_hi - s_lo),
            Fv::splat(0.0),
        );

        out += Fv::from_slice(chunk) * w;
        wsum += w;
        base -= LANES as f32;
        ph_win.rotate();
    }
    let (mut out, mut wsum) = (out.reduce_sum(), wsum.reduce_sum());

    let done = src.len() - src.chunks_exact(LANES).remainder().len();
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

fn gather_impulse_simd(src: &[f32], d0: f32, fc: f32, p: f32) -> (f32, f32) {
    let a = PI * 2.0 * fc;
    let b = PI / p;
    let mut ph_sinc = Phasor::new(a, d0);
    let mut ph_win = Phasor::new(b, d0);

    let (mut out, mut wsum) = (Fv::splat(0.0), Fv::splat(0.0));
    let mut d = Fv::splat(d0) - Fv::from_array([0.0, 1.0, 2.0, 3.0]);
    for chunk in src.chunks_exact(LANES) {
        let arg = d * Fv::splat(a);
        let near_zero = arg.abs().simd_lt(Fv::splat(1e-7));
        let sinc = near_zero.select(Fv::splat(1.0), ph_sinc.sin / arg);
        let inside = d.abs().simd_lt(Fv::splat(p));
        let w = inside.select(sinc * blackman_from_cos(ph_win.cos), Fv::splat(0.0));

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

    use core::f64::consts::PI;

    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn staged(tables: &Tables, pos: f64, f: impl Fn(i64) -> f64) -> Vec<f32> {
        let (k_lo, k_hi) = ResampleImpl1::tap_window(tables, pos as f32);
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
    fn blackman_folding_matches_the_direct_window() {
        for i in 0..=200 {
            let x = i as f64 / 200.0;
            let folded = f64::from(blackman_from_cos(Fv::splat((PI * x).cos() as f32))[0]);
            assert!(
                close(folded, f64::from(blackman(x as f32)), 1e-6),
                "blackman({x}): folded={folded}"
            );
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
                        let (got_o, got_w) = gather_step(k, &src, d0, step, p as f32);
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
        let tables = ResampleImpl1::tables(16);
        for fc in [0.1, 0.25, 0.5, 1.5] {
            for pos in [3.0, 7.35, 20.7] {
                let src = staged(&tables, pos, |_| 1.0);
                let out = f64::from(ResampleImpl1::resample(
                    &tables, &src, pos as f32, fc as f32, true,
                ));
                assert!(close(out, 1.0, 1e-9), "DC at fc={fc}, pos={pos}: {out}");
            }
        }
    }

    #[test]
    fn step_mode_is_a_bandlimited_step() {
        let tables = ResampleImpl1::tables(32);
        let fc = 0.5 / 4.0;
        let step = |k: i64| if k >= 0 { 1.0_f64 } else { 0.0 };
        let at = |pos: f64| {
            f64::from(ResampleImpl1::resample(
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
        let tables = ResampleImpl1::tables(16);
        let pos = 12.37;
        let src = staged(&tables, pos, |_| 1.0);
        let out = f64::from(ResampleImpl1::resample(
            &tables, &src, pos as f32, 0.4, false,
        ));
        assert!(close(out, 1.0, 1e-6), "DC gain = {out}");
    }

    #[test]
    fn impulse_mode_passband_signal_reconstructed() {
        let tables = ResampleImpl1::tables(16);
        let fc = 0.45;
        let f0 = 0.05;
        let get = |k: i64| (2.0 * PI * f0 * k as f64).cos();
        for frac in [0.0, 0.25, 0.5, 0.75] {
            let pos = 32.0 + frac;
            let ideal = (2.0 * PI * f0 * pos).cos();
            let out = f64::from(ResampleImpl1::resample(
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

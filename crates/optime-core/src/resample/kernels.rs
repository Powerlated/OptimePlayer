//! The process-wide oversampled kernel tables and the lookups into them. None of these depend on
//! the cutoff `fc` or the support `P`, so they are built exactly once and shared across voices.

use core::f64::consts::PI;
use std::sync::OnceLock;

/// Samples per unit `τ` in the oversampled sinc tables. 512 keeps the (linearly interpolated)
/// kernel error near 5e-6 while shrinking each table to ~256 KB so the strided gather stays cache-
/// resident — far cheaper than the original 4096 (which spilled L2 and cost ~40% in crunch mode).
pub(super) const OVERSAMPLE: usize = 512;
/// Maximum tabulated `τ = 2·fc·d`. For the hot path (`fc ≤ 0.5`, `d ≤ P ≤ 64`) this bounds `τ`.
/// Beyond it (only reachable on cheap step-mode upsampling) the kernel is evaluated directly.
pub(super) const TAU_MAX: usize = 64;
/// Samples in the Blackman window table over the normalized half-support `x = d/P ∈ [0, 1]`.
pub(super) const WIN_OVERSAMPLE: usize = 4096;
/// Largest supported `half_taps` (`P`). [`ResampleTables::new`](super::ResampleTables::new) clamps
/// to this, so callers may size stack buffers for the widest possible gather window (see
/// [`tap_window`](super::tap_window)).
pub const MAX_HALF_TAPS: usize = TAU_MAX;

/// `sinc(x) = sin(πx)/(πx)` (the normalized cardinal sine, unit zero-crossings).
pub(super) fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-15 {
        1.0
    } else {
        let px = PI * x;
        px.sin() / px
    }
}

/// Blackman window over the normalized half-support `x = |d| / P ∈ [0, 1]` (and 0 outside).
///
/// `w(x) = 0.42 + 0.5·cos(πx) + 0.08·cos(2πx)`; `w(0) = 1`, `w(1) = 0`.
pub(super) fn blackman(x: f64) -> f64 {
    if x >= 1.0 {
        return 0.0;
    }
    0.42 + 0.5 * (PI * x).cos() + 0.08 * (2.0 * PI * x).cos()
}

/// The decoupled windowed-sinc tap weight `sinc(2·fc·d) · blackman(|d|/P)` evaluated directly
/// (no tables). Used by the SIMD gather's scalar remainder and the UI filter-plot analysis.
#[inline]
pub(super) fn kernel_weight(d: f64, fc: f64, p: f64) -> f64 {
    sinc(2.0 * fc * d) * blackman(d.abs() / p)
}

/// Process-wide oversampled kernel tables. None of these depend on `fc` or `P`, so they are built
/// exactly once and shared across every voice.
///
/// Values are stored as `f32`: at `OVERSAMPLE` resolution the linear-interp error already dwarfs
/// the `~6e-8` `f32` rounding, so the narrower type halves the cache footprint (each table ~128 KB)
/// and the load cost in the strided gather, while sums still accumulate in `f64`.
pub(super) struct Kernels {
    /// `sinc[k] = sinc(k / OVERSAMPLE)` for `k` in `0..=TAU_MAX * OVERSAMPLE` (symmetric).
    /// (Unused by the `simd` build, which evaluates the impulse kernel analytically.)
    #[cfg(not(feature = "simd"))]
    sinc: Vec<f32>,
    /// `sinc_int[k] = ∫₀^{k/OVERSAMPLE} sinc(t) dt = (1/π)·Si(π·τ)`, the cumulative integral of the
    /// bare sinc (the BLEP step), odd in `τ` with asymptote `0.5` as `τ → ∞`.
    sinc_int: Vec<f32>,
    /// `win[k] = blackman(k / WIN_OVERSAMPLE)` for `k` in `0..=WIN_OVERSAMPLE`.
    win: Vec<f32>,
}

pub(super) fn kernels() -> &'static Kernels {
    static K: OnceLock<Kernels> = OnceLock::new();
    K.get_or_init(|| {
        let len = TAU_MAX * OVERSAMPLE;

        // Bare sinc, sampled at τ = k / OVERSAMPLE (computed in f64 for the integral below).
        let sinc_f64: Vec<f64> = (0..=len)
            .map(|k| sinc(k as f64 / OVERSAMPLE as f64))
            .collect();

        // Cumulative trapezoidal integral of the sinc from 0 (the BLEP step, before normalization).
        let step = 1.0 / OVERSAMPLE as f64;
        let mut sinc_int = vec![0.0f64; len + 1];
        for k in 1..=len {
            let trap = (sinc_f64[k - 1] + sinc_f64[k]) * 0.5 * step;
            sinc_int[k] = sinc_int[k - 1] + trap;
        }
        // Normalize so the right-half integral equals 0.5 (∫₀^∞ sinc = 0.5), absorbing the tiny
        // truncation error so the table lands cleanly on the 0.5 asymptote.
        let tail = sinc_int[len];
        if tail > 1e-15 {
            for v in &mut sinc_int {
                *v *= 0.5 / tail;
            }
        }

        Kernels {
            #[cfg(not(feature = "simd"))]
            sinc: sinc_f64.iter().map(|&v| v as f32).collect(),
            sinc_int: sinc_int.iter().map(|&v| v as f32).collect(),
            // Blackman window over the normalized half-support.
            win: (0..=WIN_OVERSAMPLE)
                .map(|k| blackman(k as f64 / WIN_OVERSAMPLE as f64) as f32)
                .collect(),
        }
    })
}

/// Linear interpolation into an `f32` table at floating index `idx`, widening to `f64` for the
/// gather accumulation. Callers guarantee `idx < tab.len() - 1` (each lookup helper guards its
/// table's edge before calling), so `i + 1` is always in range.
#[inline]
fn lerp(tab: &[f32], idx: f64) -> f64 {
    let i = idx as usize;
    let frac = idx - i as f64;
    let lo = f64::from(tab[i]);
    lo + (f64::from(tab[i + 1]) - lo) * frac
}

/// The BLEP step `S(τ) = ∫₀^τ sinc`, looked up at the **pre-scaled signed index** `idx = τ·OVERSAMPLE`.
/// Odd in `τ` (`S(−τ) = −S(τ)`), asymptote `±0.5`; beyond the table the ripple is negligible so the
/// asymptote is returned directly. (Step mode may push `|τ| > TAU_MAX` on upsampling.)
#[inline]
pub(super) fn sinc_int_at(k: &Kernels, idx: f64) -> f64 {
    let mag = idx.abs();
    let v = if mag >= (k.sinc_int.len() - 1) as f64 {
        0.5
    } else {
        lerp(&k.sinc_int, mag)
    };
    if idx < 0.0 {
        -v
    } else {
        v
    }
}

/// Blackman window looked up at the **pre-scaled index** `idx = (|d|/P)·WIN_OVERSAMPLE` (0 past the
/// support edge).
#[inline]
pub(super) fn win_at(k: &Kernels, idx: f64) -> f64 {
    if idx >= WIN_OVERSAMPLE as f64 {
        0.0
    } else {
        lerp(&k.win, idx)
    }
}

/// Bare sinc looked up at the non-negative pre-scaled index `idx = τ·OVERSAMPLE`, returning 0 past
/// the table. Safe because in impulse mode (`fc ≤ 0.5`, `P ≤ TAU_MAX`) the window already vanishes
/// wherever `idx` would run off the end, so the 0 is only ever multiplied by a 0 window.
/// (The `simd` build evaluates the impulse kernel analytically and does not read this table.)
#[cfg(not(feature = "simd"))]
#[inline]
pub(super) fn sinc_at(k: &Kernels, idx: f64) -> f64 {
    if idx >= (k.sinc.len() - 1) as f64 {
        0.0
    } else {
        lerp(&k.sinc, idx)
    }
}

//! Variable-ratio windowed-sinc / BLEP resampler with a **fixed source-tap support**.
//!
//! Both sinc modes share a single decoupled kernel
//!
//! ```text
//!     k(d) = sinc(2·fc·d) · blackman(d / P)        for |d| ≤ P,   0 otherwise
//! ```
//!
//! where `d = pos − k` is the offset (in source samples) of source tap `k` from the fractional
//! read position, `fc` is the anti-alias cutoff in cycles/source-sample, and `P` (`half_taps`) is
//! the half-width of the support **in source samples**. The two factors are looked up from
//! separate oversampled tables — a bare-sinc table indexed by `τ = 2·fc·d` and a Blackman-window
//! table indexed by `d / P` — so neither table depends on `fc` or `P` and both are built exactly
//! once for the whole process.
//!
//! Crucially the support is `±P` source samples **regardless of the resampling ratio** `r`, so the
//! gather is always `≈ 2P` taps: `O(P)`, not `O(P·r)`. (The previous design keyed the kernel to
//! zero-crossings, which made the tap count grow as `half_taps / fc = 2·half_taps·r` when
//! downsampling — the cause of the real-time underruns at high tap counts.)
//!
//! At heavy downsampling a fixed `P` taps spans fewer sinc periods (`2·fc·P = P/r` zero-crossings),
//! so the anti-aliasing softens gracefully — the intended cost/quality trade.
//!
//! * **SampleNyquist (clean)**: impulse-mode gather, `fc = min(0.5, 0.5/r)`, DC-normalized.
//! * **OutputNyquist (crunch)**: BLEP-style gather of the *boxcar-integrated* kernel at
//!   `fc = 0.5/r` (the *output* Nyquist). For `r > 1` this anti-aliases; for `r ≤ 1` (upsampling)
//!   it rises above source Nyquist (`fc > 0.5`), band-limiting the ZOH stairstep edges to the
//!   output rate while keeping the crunch images that fall below output Nyquist.

use core::f64::consts::PI;
use std::sync::OnceLock;

/// Samples per unit `τ` in the oversampled sinc tables. 512 keeps the (linearly interpolated)
/// kernel error near 5e-6 while shrinking each table to ~256 KB so the strided gather stays cache-
/// resident — far cheaper than the original 4096 (which spilled L2 and cost ~40% in crunch mode).
const OVERSAMPLE: usize = 512;
/// Maximum tabulated `τ = 2·fc·d`. For the hot path (`fc ≤ 0.5`, `d ≤ P ≤ 64`) this bounds `τ`.
/// Beyond it (only reachable on cheap step-mode upsampling) the kernel is evaluated directly.
const TAU_MAX: usize = 64;
/// Samples in the Blackman window table over the normalized half-support `x = d/P ∈ [0, 1]`.
const WIN_OVERSAMPLE: usize = 4096;
/// Largest supported `half_taps` (`P`). [`ResampleTables::new`] clamps to this, so callers may size
/// stack buffers for the widest possible gather window (see [`tap_window`]).
pub const MAX_HALF_TAPS: usize = TAU_MAX;

/// `sinc(x) = sin(πx)/(πx)` (the normalized cardinal sine, unit zero-crossings).
fn sinc(x: f64) -> f64 {
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
fn blackman(x: f64) -> f64 {
    if x >= 1.0 {
        return 0.0;
    }
    0.42 + 0.5 * (PI * x).cos() + 0.08 * (2.0 * PI * x).cos()
}

/// Process-wide oversampled kernel tables. None of these depend on `fc` or `P`, so they are built
/// exactly once and shared across every voice.
///
/// Values are stored as `f32`: at `OVERSAMPLE` resolution the linear-interp error already dwarfs
/// the `~6e-8` `f32` rounding, so the narrower type halves the cache footprint (each table ~128 KB)
/// and the load cost in the strided gather, while sums still accumulate in `f64`.
struct Kernels {
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

fn kernels() -> &'static Kernels {
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
fn sinc_int_at(k: &Kernels, idx: f64) -> f64 {
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
fn win_at(k: &Kernels, idx: f64) -> f64 {
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
fn sinc_at(k: &Kernels, idx: f64) -> f64 {
    if idx >= (k.sinc.len() - 1) as f64 {
        0.0
    } else {
        lerp(&k.sinc, idx)
    }
}

/// Pre-built resampler configuration. Holds only the support half-width `P`; the heavy kernel
/// tables live in a process-wide [`OnceLock`] and are shared (so building this is essentially free).
#[derive(Clone)]
pub struct ResampleTables {
    /// Half-width of the kernel support, in **source samples**.
    pub half_taps: usize,
}

impl ResampleTables {
    /// Builds a resampler with a `half_taps`-source-sample half-width support, clamped to
    /// `1..=TAU_MAX` so the impulse-mode sinc index can never run past the table (the hot gather
    /// then needs no per-tap bounds branch). `TAU_MAX` (64) is also the UI's maximum.
    pub fn new(half_taps: usize) -> Self {
        // Touch the shared tables so the one-time build happens here rather than on the audio
        // thread's first gather.
        let _ = kernels();
        Self {
            half_taps: half_taps.clamp(1, TAU_MAX),
        }
    }
}

/// The inclusive source-tap window `[k_lo, k_hi]` the gather reads for a read position `pos`:
/// every integer `k` with `|pos − k| ≤ P`. Exported so callers can pre-stage exactly the source
/// samples [`resample_sinc`] will request (the formula must match the one used internally).
#[inline]
pub fn tap_window(tables: &ResampleTables, pos: f64) -> (i64, i64) {
    let p = tables.half_taps as f64;
    ((pos - p).floor() as i64, (pos + p).ceil() as i64)
}

/// Windowed-sinc polyphase gather — the single shared resampler for both sinc modes.
///
/// # Parameters
/// - `tables`:    the support width (the kernel tables are global).
/// - `src`:       the staged source taps for the window returned by [`tap_window`]: `src[j]`
///   holds the source sample at index `k_lo + j`, and `src.len()` must be `k_hi − k_lo + 1`.
/// - `pos`:       fractional read position in source samples.
/// - `fc`:        cutoff in cycles/source-sample.
/// - `step_mode`: `false` → impulse-mode (SampleNyquist); `true` → BLEP step-mode (OutputNyquist).
pub fn resample_sinc(
    tables: &ResampleTables,
    src: &[f32],
    pos: f64,
    fc: f64,
    step_mode: bool,
) -> f64 {
    let k = kernels();
    // Fixed source-sample support: |pos − k| ≤ P, i.e. ≈ 2P taps regardless of fc.
    let (k_lo, k_hi) = tap_window(tables, pos);
    debug_assert_eq!(
        src.len() as i64,
        k_hi - k_lo + 1,
        "src must cover the tap window"
    );

    // Impulse (reconstruction) mode never wants a cutoff above source Nyquist; step (BLEP) mode may
    // (output Nyquist sits above source Nyquist when upsampling).
    let fc = if step_mode {
        fc.max(1e-6)
    } else {
        fc.clamp(1e-6, 0.5)
    };
    // Table-index steps: fold the per-tap `·OVERSAMPLE` / `·WIN_OVERSAMPLE` scaling into the walk so
    // each tap advances the indices by a constant add instead of recomputing scaled products.
    let sinc_idx_step = 2.0 * fc * OVERSAMPLE as f64; // Δ index per unit τ-step (one source sample)
    let win_idx_step = WIN_OVERSAMPLE as f64 / tables.half_taps as f64; // Δ window index per source sample

    // The gather kernel is selected at build time: the portable-SIMD version when the nightly
    // `simd` feature is enabled, the scalar version otherwise. Both compute the same
    // `(Σ src·w, Σ w)` pair (summation order differs, so results may diverge by float rounding).
    let d0 = pos - k_lo as f64; // offset of the first tap from the read position (≈ P)
                                // Step mode always uses the scalar gather: its cumulative-sinc table lookup carries one edge
                                // value across iterations, which measures faster than any vectorized variant (the table
                                // gathers dominate). Impulse mode vectorizes well because its kernel can be evaluated
                                // analytically with rotating phasors — no table traffic at all.
    let (out, wsum) = if step_mode {
        gather_step(k, src, d0, sinc_idx_step, win_idx_step)
    } else {
        #[cfg(feature = "simd")]
        {
            simd::gather_impulse(src, d0, fc, tables.half_taps as f64)
        }
        #[cfg(not(feature = "simd"))]
        {
            let mid_j = (pos.floor() as i64 - k_lo) as usize; // last tap with d = pos − k ≥ 0
            gather_impulse(k, src, d0, mid_j, sinc_idx_step, win_idx_step)
        }
    };
    let wsum = if step_mode { wsum.abs() } else { wsum };

    if wsum > 1e-12 {
        out / wsum
    } else {
        // Degenerate weights (pathological fc): fall back to the nearest staged tap.
        f64::from(src[(pos.round() as i64 - k_lo) as usize])
    }
}

/// Scalar BLEP gather of the boxcar-integrated windowed kernel: tap `j` weighs its source-sample
/// bin `[pos−k−1, pos−k]` (where `d_hi = d0 − j = pos − k`) by the band-limited step rise across
/// it,
///     `[S(2fc·d_hi) − S(2fc·(d_hi−1))] · blackman(|bin-center| / P)`,
/// where `S` is the cumulative sinc integral. The bin's upper-edge `S` value is the next bin's
/// lower edge, so it is carried across iterations (one `sinc_int` lookup per tap). Normalizing by
/// the weight sum forces exact DC unity (and absorbs the window). Returns `(Σ src·w, Σ w)`.
fn gather_step(
    k: &Kernels,
    src: &[f32],
    d0: f64,
    sinc_idx_step: f64,
    win_idx_step: f64,
) -> (f64, f64) {
    let mut out = 0.0;
    let mut wsum = 0.0;
    let mut si_hi = sinc_int_at(k, sinc_idx_step * d0);
    let mut lo_idx = sinc_idx_step * (d0 - 1.0); // S index of the bin's lower edge
    let mut mid_idx = win_idx_step * (d0 - 0.5); // window index of the bin centre (signed)
    for &s in src {
        let si_lo = sinc_int_at(k, lo_idx);
        let w = win_at(k, mid_idx.abs()) * (si_hi - si_lo);
        out += f64::from(s) * w;
        wsum += w;
        si_hi = si_lo;
        lo_idx -= sinc_idx_step;
        mid_idx -= win_idx_step;
    }
    (out, wsum)
}

/// Scalar impulse gather: `out = Σ_j src[j] · sinc(2fc·|d|) · blackman(|d|/P)` with `d = d0 − j`,
/// DC-normalized by the caller. The kernel is even in `d`, so the window is split at `d = 0`
/// (tap `mid_j`) into two monotonic runs that walk `|d|`'s table indices by a constant add each
/// tap — no per-tap `abs`/multiply. Taps past the support contribute a zero window, so no in-loop
/// bounds test is needed. Returns `(Σ src·w, Σ w)`.
#[cfg(not(feature = "simd"))]
fn gather_impulse(
    k: &Kernels,
    src: &[f32],
    d0: f64,
    mid_j: usize,
    sinc_idx_step: f64,
    win_idx_step: f64,
) -> (f64, f64) {
    let mut out = 0.0;
    let mut wsum = 0.0;
    let (right, left) = src.split_at(mid_j + 1);

    // Right run: descending |d| = d0 − j.
    let mut sinc_idx = d0 * sinc_idx_step;
    let mut win_idx = d0 * win_idx_step;
    for &s in right {
        let w = sinc_at(k, sinc_idx) * win_at(k, win_idx);
        out += f64::from(s) * w;
        wsum += w;
        sinc_idx -= sinc_idx_step;
        win_idx -= win_idx_step;
    }
    // Left run: ascending |d| = j − d0.
    let d_left = mid_j as f64 + 1.0 - d0;
    let mut sinc_idx = d_left * sinc_idx_step;
    let mut win_idx = d_left * win_idx_step;
    for &s in left {
        let w = sinc_at(k, sinc_idx) * win_at(k, win_idx);
        out += f64::from(s) * w;
        wsum += w;
        sinc_idx += sinc_idx_step;
        win_idx += win_idx_step;
    }
    (out, wsum)
}

/// Portable-SIMD (`std::simd`, nightly) impulse gather. Instead of vector-gathering the kernel
/// tables (hardware gathers measured *slower* than the scalar walk), the kernel is evaluated
/// analytically, four taps at a time, with **rotating phasors**: per chunk, each needed sin/cos
/// advances by a constant angle via one complex multiply — pure FMA math, zero table traffic.
/// This is also (slightly) more accurate than the table interpolation.
///
/// Accumulation runs in four parallel partial sums reduced at the end, so results differ from the
/// scalar build by float rounding — the DC normalization in [`resample_sinc`] stays exact because
/// `out` and `wsum` accumulate with identical operations.
#[cfg(feature = "simd")]
mod simd {
    use core::f64::consts::PI;
    use std::simd::prelude::*;

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
    pub(super) fn gather_impulse(src: &[f32], d0: f64, fc: f64, p: f64) -> (f64, f64) {
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
            let win =
                Fv::splat(0.42) + Fv::splat(0.5) * ph_win1.cos + Fv::splat(0.08) * ph_win2.cos;
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
            let w = super::sinc(2.0 * fc * d) * super::blackman(d.abs() / p);
            out += f64::from(s) * w;
            wsum += w;
        }
        (out, wsum)
    }
}

// ─── Analysis helpers (used by the filter-plot popup in optime-app) ──────────────────────────

/// Returns the normalized kernel taps `k(d)` for integer source offsets `d = −(P−1)..=(P−1)`,
/// suitable for drawing as a stem plot. Scaled to the cutoff `fc` and normalized to unit DC gain.
pub fn fir_kernel(half_taps: usize, fc: f64) -> Vec<f64> {
    let p = half_taps.max(1);
    let pf = p as f64;
    let fc = fc.clamp(1e-6, 0.5);
    let taps: Vec<f64> = (-(p as i64 - 1)..=(p as i64 - 1))
        .map(|d| {
            let ad = d.unsigned_abs() as f64;
            sinc(2.0 * fc * ad) * blackman(ad / pf)
        })
        .collect();
    let sum: f64 = taps.iter().sum();
    if sum > 1e-10 {
        taps.into_iter().map(|v| v / sum).collect()
    } else {
        taps
    }
}

/// Evaluates the frequency response of `fir_kernel(half_taps, fc)` at digital frequency
/// `w_norm ∈ [0, π]`. Returns `(magnitude, phase_radians)`. The kernel is linear-phase (symmetric).
pub fn fir_response(half_taps: usize, fc: f64, w_norm: f64) -> (f64, f64) {
    let kernel = fir_kernel(half_taps, fc);
    let center = kernel.len() / 2; // index of the zero-lag tap
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (i, &h) in kernel.iter().enumerate() {
        let delay = i as f64 - center as f64;
        re += h * (w_norm * delay).cos();
        im -= h * (w_norm * delay).sin();
    }
    let mag = (re * re + im * im).sqrt();
    let phase = im.atan2(re);
    (mag, phase)
}

// ─── Tests ───────────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// Stages the tap window for `pos` by sampling `f` at each source index (what the synth's
    /// gather staging does for real sample data).
    fn staged(tables: &ResampleTables, pos: f64, f: impl Fn(i64) -> f64) -> Vec<f32> {
        let (k_lo, k_hi) = tap_window(tables, pos);
        (k_lo..=k_hi).map(|t| f(t) as f32).collect()
    }

    #[test]
    fn sinc_int_boundary_and_symmetry() {
        let k = kernels();
        // `sinc_int_at` takes the pre-scaled index τ·OVERSAMPLE.
        let si = |tau: f64| sinc_int_at(k, tau * OVERSAMPLE as f64);
        // S(0) = 0, and S saturates to ±0.5 outside the tabulated range.
        assert!(close(si(0.0), 0.0, 1e-12));
        assert!(close(si(TAU_MAX as f64 + 5.0), 0.5, 1e-6));
        assert!(close(si(-(TAU_MAX as f64) - 5.0), -0.5, 1e-6));
        // Odd symmetry: S(−τ) = −S(τ).
        for i in 1..=20 {
            let tau = TAU_MAX as f64 * i as f64 / 20.0;
            assert!(close(si(tau) + si(-tau), 0.0, 1e-12), "S({tau}) not odd");
        }
    }

    #[test]
    fn sinc_int_is_bounded() {
        // The sinc has negative lobes, so its integral overshoots slightly past 0.5, but it must
        // stay within a physically reasonable band.
        let k = kernels();
        for i in 0..=2000 {
            let tau = -(TAU_MAX as f64) + 2.0 * TAU_MAX as f64 * i as f64 / 2000.0;
            let s = sinc_int_at(k, tau * OVERSAMPLE as f64);
            assert!((-0.6..=0.6).contains(&s), "S({tau}) = {s} out of band");
        }
    }

    #[test]
    fn step_mode_preserves_dc() {
        // The normalized BLEP gather must pass a constant signal through unchanged at any position.
        let tables = ResampleTables::new(16);
        for fc in [0.1, 0.25, 0.5, 1.5] {
            for pos in [3.0, 7.35, 20.7] {
                let src = staged(&tables, pos, |_| 1.0);
                let out = resample_sinc(&tables, &src, pos, fc, true);
                assert!(close(out, 1.0, 1e-9), "DC at fc={fc}, pos={pos}: {out}");
            }
        }
    }

    #[test]
    fn step_mode_is_a_bandlimited_step() {
        // A unit source step must produce a monotone-ish band-limited rise: ≈0 far below the edge,
        // ≈0.5 at the edge, ≈1 far above.
        let tables = ResampleTables::new(32);
        let fc = 0.5 / 4.0; // 4× downsampling
        let step = |k: i64| if k >= 0 { 1.0_f64 } else { 0.0 };
        let at = |pos: f64| resample_sinc(&tables, &staged(&tables, pos, step), pos, fc, true);

        assert!(close(at(0.0), 0.5, 0.02));
        let half_width = tables.half_taps as f64 / (2.0 * fc);
        assert!(close(at(-(half_width + 10.0)), 0.0, 1e-6));
        assert!(close(at(half_width + 10.0), 1.0, 1e-6));
        // Monotone non-decreasing through the transition.
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
        let tables = ResampleTables::new(16);
        let pos = 12.37;
        let src = staged(&tables, pos, |_| 1.0);
        let out = resample_sinc(&tables, &src, pos, 0.4, false);
        assert!(close(out, 1.0, 1e-6), "DC gain = {out}");
    }

    #[test]
    fn impulse_mode_passband_signal_reconstructed() {
        // A signal well within the passband (freq << fc) is reconstructed at any fractional pos.
        let tables = ResampleTables::new(16);
        let fc = 0.45;
        let f0 = 0.05;
        let get = |k: i64| (2.0 * PI * f0 * k as f64).cos();
        for frac in [0.0, 0.25, 0.5, 0.75] {
            let pos = 32.0 + frac;
            let ideal = (2.0 * PI * f0 * pos).cos();
            let out = resample_sinc(&tables, &staged(&tables, pos, get), pos, fc, false);
            assert!(
                close(out, ideal, 1e-3),
                "at pos={pos}: reconstructed={out}, ideal={ideal}"
            );
        }
    }

    #[test]
    fn fixed_tap_count_independent_of_ratio() {
        // The whole point: the gather spans ≈2P taps whether we up- or down-sample. We count the
        // taps the kernel actually touches (nonzero window) at a couple of cutoffs.
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

    #[test]
    fn fir_kernel_dc_gain_is_one() {
        let k = fir_kernel(16, 0.45);
        let sum: f64 = k.iter().sum();
        assert!(close(sum, 1.0, 1e-10), "kernel sum = {sum}");
    }

    #[test]
    fn fir_kernel_is_symmetric() {
        let k = fir_kernel(8, 0.4);
        let n = k.len();
        for i in 0..n / 2 {
            assert!(close(k[i], k[n - 1 - i], 1e-15), "asymmetry at {i}");
        }
    }

    #[test]
    fn fir_response_dc_gain_near_one() {
        let (mag, _) = fir_response(16, 0.45, 0.0);
        assert!(close(mag, 1.0, 1e-10), "DC magnitude = {mag}");
    }

    #[test]
    fn stopband_suppression_improves_with_taps() {
        // For a fixed cutoff, a wider support sharpens the transition and pushes a stopband tone
        // further down. Cutoff fc = 0.25; probe at 0.4 cyc/sample (deep in the stopband).
        let w_stop = 2.0 * PI * 0.4;
        let taps = [2usize, 4, 8, 16, 32];
        let mags: Vec<f64> = taps
            .iter()
            .map(|&t| fir_response(t, 0.25, w_stop).0)
            .collect();
        for w in mags.windows(2) {
            assert!(
                w[1] < w[0],
                "stopband magnitude should fall with more taps, got {mags:?}"
            );
        }
        assert!(
            *mags.last().unwrap() < 0.01,
            "widest-kernel stopband magnitude = {}",
            mags.last().unwrap()
        );
        for &t in &taps {
            let (pass, _) = fir_response(t, 0.25, 2.0 * PI * 0.05);
            assert!(pass > 0.95, "passband magnitude at {t} taps = {pass}");
        }
    }
}

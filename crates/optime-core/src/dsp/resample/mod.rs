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
//!
//! The module is split into:
//! - [`kernels`] — the process-wide oversampled tables and their lookups.
//! - [`gather`] / [`simd`] — the scalar and portable-SIMD gather kernels.
//! - [`source`] — the loop-aware source-staging gather ([`gather_sinc`]) that feeds a voice.

mod gather;
mod kernels;
mod source;
#[cfg(feature = "simd")]
mod simd;

#[cfg(test)]
mod tests;

pub use kernels::MAX_HALF_TAPS;
pub use source::{gather_sinc, GatherSource};

use kernels::{kernels, OVERSAMPLE, WIN_OVERSAMPLE};

/// Stack-buffer length covering the widest possible gather window (one full tap window at
/// [`MAX_HALF_TAPS`]). Sized so every staged-window gather reads a plain slice.
pub(crate) const GATHER_BUF_LEN: usize = 2 * MAX_HALF_TAPS + 2;

/// Pre-built resampler configuration. Holds only the support half-width `P`; the heavy kernel
/// tables live in a process-wide [`OnceLock`](std::sync::OnceLock) and are shared (so building
/// this is essentially free).
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
            half_taps: half_taps.clamp(1, MAX_HALF_TAPS),
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
        gather::gather_step(k, src, d0, sinc_idx_step, win_idx_step)
    } else {
        #[cfg(feature = "simd")]
        {
            simd::gather_impulse(src, d0, fc, tables.half_taps as f64)
        }
        #[cfg(not(feature = "simd"))]
        {
            let mid_j = (pos.floor() as i64 - k_lo) as usize; // last tap with d = pos − k ≥ 0
            gather::gather_impulse(k, src, d0, mid_j, sinc_idx_step, win_idx_step)
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

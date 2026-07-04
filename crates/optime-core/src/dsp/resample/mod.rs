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
//! - [`stream`] — the streaming, fixed-ratio [`StreamResampler`] for the mixer-bus → output stage.
//!
//! Resolving an [`InstrumentResampleMode`](crate::waveform::InstrumentResampleMode) into the concrete
//! gather + cutoff ([`effective_gather`] / [`sinc_fc`] / [`mode_half_taps`]) lives here too, shared
//! by the voice gather and the stream resampler.

mod gather;
mod kernels;
#[cfg(feature = "simd")]
mod simd;
mod source;
mod stream;

#[cfg(test)]
mod tests;

pub use kernels::MAX_HALF_TAPS;
pub use source::{gather_sinc, GatherSource};
pub use stream::StreamResampler;

use crate::waveform::{InstrumentResampleMode, Sample};
use kernels::{kernels, OVERSAMPLE, WIN_OVERSAMPLE};

/// The gather a read actually runs after resolving the global [`InstrumentResampleMode`] against
/// the signal kind (a voice's sample, or the mixer bus).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum EffectiveGather {
    Nearest,
    Linear,
    /// Windowed-sinc. `step_mode` selects the BLEP step kernel (output-Nyquist crunch);
    /// `cutoff_hz` is an extra low-pass on top of the mode's natural cutoff.
    Sinc {
        step_mode: bool,
        cutoff_hz: Option<u32>,
    },
}

/// Resolves the global mode for one signal. `is_psg` marks a PSG waveform (square/wave/noise); the
/// mixer bus passes `false` (a finished mix has no PSG/sampled distinction).
///
/// PSG waveforms under the clean (SampleNyquist) mode still take the BLEP step gather: their
/// hard ZOH edges are band-limited to the *output* Nyquist instead of being smoothed down to the
/// (tiny) source Nyquist of an 8-sample loop, preserving their square character alias-free. The
/// crunchy (OutputNyquist) mode additionally applies the user's per-kind cutoff slider.
pub(crate) fn effective_gather(mode: InstrumentResampleMode, is_psg: bool) -> EffectiveGather {
    match mode {
        InstrumentResampleMode::NearestNeighbor => EffectiveGather::Nearest,
        InstrumentResampleMode::Linear => EffectiveGather::Linear,
        InstrumentResampleMode::SincSampleNyquist { .. } => EffectiveGather::Sinc {
            step_mode: is_psg,
            cutoff_hz: None,
        },
        InstrumentResampleMode::SincOutputNyquist {
            psg_cutoff_hz,
            sampler_cutoff_hz,
            ..
        } => EffectiveGather::Sinc {
            step_mode: true,
            cutoff_hz: Some(if is_psg {
                psg_cutoff_hz
            } else {
                sampler_cutoff_hz
            }),
        },
    }
}

/// The half-width of the kernel support a sinc mode needs (`None` for the 1–2 tap / nearest modes).
pub(crate) fn mode_half_taps(mode: InstrumentResampleMode) -> Option<usize> {
    match mode {
        InstrumentResampleMode::SincSampleNyquist { half_taps }
        | InstrumentResampleMode::SincOutputNyquist { half_taps, .. } => Some(half_taps),
        _ => None,
    }
}

/// The sinc gather's low-pass cutoff in cycles/source-sample (source Nyquist = 0.5) for playback
/// speed `r` (source samples per output sample).
///
/// - Impulse mode (clean reconstruction): `min(0.5, 0.5/r)` — removes ZOH images when
///   upsampling, anti-aliases when downsampling.
/// - Step mode (BLEP): `0.5/r`, the *output* Nyquist, so the stairstep edges are band-limited to
///   the output rate. For `r > 1` (downsampling) this is `< 0.5` and anti-aliases; for `r ≤ 1`
///   (upsampling) it is `≥ 0.5`, keeping the crunch images that sit below output Nyquist while
///   still band-limiting the hard edges (no nearest-neighbour jitter).
/// - `cutoff_hz` (the crunchy-mode sliders) lowers either further: an output-domain frequency
///   `f` Hz is `f / (r · sample_rate)` cycles/source-sample.
pub(crate) fn sinc_fc(
    r: f64,
    inv_sample_rate: f64,
    step_mode: bool,
    cutoff_hz: Option<u32>,
) -> f64 {
    let mut fc = if step_mode || r > 1.0 { 0.5 / r } else { 0.5 };
    if let Some(hz) = cutoff_hz {
        fc = fc.min(f64::from(hz) * inv_sample_rate / r);
    }
    fc
}

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
) -> Sample {
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
        (out / wsum) as Sample
    } else {
        // Degenerate weights (pathological fc): fall back to the nearest staged tap.
        Sample::from(src[(pos.round() as i64 - k_lo) as usize])
    }
}

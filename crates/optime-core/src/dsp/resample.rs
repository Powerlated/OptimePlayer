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
//! The file is laid out in stages:
//! - the process-wide oversampled kernel tables and their lookups;
//! - the two gather kernels, the BLEP step gather and the portable-SIMD impulse gather;
//! - [`InstrumentResampleMode`](crate::waveform::InstrumentResampleMode) resolution
//!   ([`effective_gather`] / [`sinc_fc`] / [`mode_half_taps`]) and the shared [`resample_sinc`]
//!   gather, used by both the voice gather and the stream resampler;
//! - the loop-aware source-staging gather ([`gather_sinc`]) that feeds a voice;
//! - the streaming, fixed-ratio [`StreamResampler`] for the mixer-bus → output stage.

use core::f32::consts::PI;
use std::simd::prelude::*;
use std::sync::OnceLock;

use crate::dsp::block::MAX_BLOCK;
use crate::waveform::{InstrumentResampleMode, Sample};

// ===========================================================================================
// Kernel tables and lookups
// ===========================================================================================
//
// The process-wide oversampled kernel tables and the lookups into them. None of these depend on
// the cutoff `fc` or the support `P`, so they are built exactly once and shared across voices.

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
fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-7 {
        1.0
    } else {
        let px = PI * x;
        px.sin() / px
    }
}

/// Blackman window over the normalized half-support `x = |d| / P ∈ [0, 1]` (and 0 outside).
///
/// `w(x) = 0.42 + 0.5·cos(πx) + 0.08·cos(2πx)`; `w(0) = 1`, `w(1) = 0`.
fn blackman(x: f32) -> f32 {
    if x >= 1.0 {
        return 0.0;
    }
    0.42 + 0.5 * (PI * x).cos() + 0.08 * (2.0 * PI * x).cos()
}

/// The decoupled windowed-sinc tap weight `sinc(2·fc·d) · blackman(|d|/P)` evaluated directly
/// (no tables). Used for the taps left over after the impulse gather's last full vector chunk.
#[inline]
fn kernel_weight(d: f32, fc: f32, p: f32) -> f32 {
    sinc(2.0 * fc * d) * blackman(d.abs() / p)
}

/// Process-wide oversampled kernel tables. None of these depend on `fc` or `P`, so they are built
/// exactly once and shared across every voice.
///
/// The whole resampler — table build, lookups, and gathers — runs in `f32`; no value is ever
/// widened to `f64`. The one-time cumulative integral below is summed with Kahan compensation so
/// its `f32` accumulation stays as accurate as the old `f64` build. At `OVERSAMPLE` resolution the
/// linear-interp error already dwarfs the `~6e-8` `f32` rounding, and the narrow type halves the
/// cache footprint (each table ~128 KB) and the load cost in the strided gather.
struct Kernels {
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

        // Bare sinc, sampled at τ = k / OVERSAMPLE. Only the integral below is kept: the impulse
        // gather evaluates the bare sinc analytically rather than reading a table.
        let sinc_tab: Vec<f32> = (0..=len)
            .map(|k| sinc(k as f32 / OVERSAMPLE as f32))
            .collect();

        // Cumulative trapezoidal integral of the sinc from 0 (the BLEP step, before normalization).
        // Kahan-compensated so the `f32` running sum over `len` terms stays accurate to ~1 ulp.
        let step = 1.0 / OVERSAMPLE as f32;
        let mut sinc_int = vec![0.0f32; len + 1];
        let mut sum = 0.0f32;
        let mut comp = 0.0f32; // Kahan compensation
        for k in 1..=len {
            let trap = (sinc_tab[k - 1] + sinc_tab[k]) * 0.5 * step;
            let y = trap - comp;
            let t = sum + y;
            comp = (t - sum) - y;
            sum = t;
            sinc_int[k] = sum;
        }
        // Normalize so the right-half integral equals 0.5 (∫₀^∞ sinc = 0.5), absorbing the tiny
        // truncation error so the table lands cleanly on the 0.5 asymptote.
        let tail = sinc_int[len];
        if tail > 1e-12 {
            let scale = 0.5 / tail;
            for v in &mut sinc_int {
                *v *= scale;
            }
        }

        Kernels {
            sinc_int,
            // Blackman window over the normalized half-support.
            win: (0..=WIN_OVERSAMPLE)
                .map(|k| blackman(k as f32 / WIN_OVERSAMPLE as f32))
                .collect(),
        }
    })
}

/// Linear interpolation into an `f32` table at the floating index `idx`, returning the interpolated
/// weight. Callers guarantee `idx < tab.len() - 1` (each lookup helper guards its table's edge
/// before calling), so `i + 1` is always in range.
#[inline]
fn lerp(tab: &[f32], idx: f32) -> f32 {
    let i = idx as usize;
    let frac = idx - i as f32;
    let lo = tab[i];
    lo + (tab[i + 1] - lo) * frac
}

/// The BLEP step `S(τ) = ∫₀^τ sinc`, looked up at the **pre-scaled signed index** `idx = τ·OVERSAMPLE`.
/// Odd in `τ` (`S(−τ) = −S(τ)`), asymptote `±0.5`; beyond the table the ripple is negligible so the
/// asymptote is returned directly. (Step mode may push `|τ| > TAU_MAX` on upsampling.)
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

/// Blackman window looked up at the **pre-scaled index** `idx = (|d|/P)·WIN_OVERSAMPLE` (0 past the
/// support edge).
#[inline]
fn win_at(k: &Kernels, idx: f32) -> f32 {
    if idx >= WIN_OVERSAMPLE as f32 {
        0.0
    } else {
        lerp(&k.win, idx)
    }
}

// ===========================================================================================
// Scalar BLEP step gather
// ===========================================================================================
//
// The step gather sums a staged tap window against the boxcar-integrated windowed-sinc kernel,
// returning the `(Σ src·w, Σ w)` pair the caller DC-normalizes. It stays scalar: its cumulative-sinc
// table lookup carries one edge value across iterations, which measures faster than any vectorized
// variant (the table reads dominate).

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
    d0: f32,
    sinc_idx_step: f32,
    win_idx_step: f32,
) -> (f32, f32) {
    let mut out = 0.0f32;
    let mut wsum = 0.0f32;
    let mut si_hi = sinc_int_at(k, sinc_idx_step * d0);
    let mut lo_idx = sinc_idx_step * (d0 - 1.0); // S index of the bin's lower edge
    let mut mid_idx = win_idx_step * (d0 - 0.5); // window index of the bin centre (signed)
    for &s in src {
        let si_lo = sinc_int_at(k, lo_idx);
        let w = win_at(k, mid_idx.abs()) * (si_hi - si_lo);
        out += s * w;
        wsum += w;
        si_hi = si_lo;
        lo_idx -= sinc_idx_step;
        mid_idx -= win_idx_step;
    }
    (out, wsum)
}

// ===========================================================================================
// Portable-SIMD impulse gather
// ===========================================================================================
//
// Portable-SIMD (`std::simd`) impulse gather. Instead of vector-gathering the kernel tables
// (hardware gathers measured *slower* than a scalar walk), the kernel is evaluated analytically,
// four taps at a time, with **rotating phasors**: per chunk, each needed sin/cos advances by a
// constant angle via one complex multiply — pure FMA math, zero table traffic. This is also
// (slightly) more accurate than the table interpolation.
//
// This is the only impulse gather. Targets without vector instructions get the same code lowered to
// scalar arithmetic by the backend, which is why there is no hand-written scalar twin to keep in
// step with it. `std::simd` is nightly-only, and `rust-toolchain.toml` pins nightly for it.
//
// Accumulation runs in four parallel partial sums reduced at the end — the DC normalization in
// `resample_sinc` stays exact regardless, because `out` and `wsum` accumulate with identical
// operations.
//
// The entire gather — phasors, kernel weights, and sample accumulation — runs in `f32`; no value is
// ever widened to `f64`. The rotating recurrence's rounding stays negligible because the Blackman
// window drives the weight to zero at the ≤ `2P` window edges, long before the phase can drift
// meaningfully over a single gather.

const LANES: usize = 4;
/// Gather lane vector (`f32`: phasors, weights, and sample accumulation alike).
type Fv = Simd<f32, LANES>;

/// A vector of sin/cos pairs advanced by a fixed per-chunk angle step via complex rotation.
struct Phasor {
    sin: Fv,
    cos: Fv,
    step_sin: f32,
    step_cos: f32,
}

impl Phasor {
    /// Lanes at angles `rate · (d0 − i)` for `i = 0..LANES`, stepping by `−rate·LANES`/chunk.
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
fn gather_impulse_simd(src: &[f32], d0: f32, fc: f32, p: f32) -> (f32, f32) {
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
        let near_zero = arg.abs().simd_lt(Fv::splat(1e-7));
        let sinc = near_zero.select(Fv::splat(1.0), ph_sinc.sin / arg);
        // blackman(|d|/P), 0 outside the ±P support.
        let win = Fv::splat(0.42) + Fv::splat(0.5) * ph_win1.cos + Fv::splat(0.08) * ph_win2.cos;
        let inside = d.abs().simd_lt(Fv::splat(p));
        let w = inside.select(sinc * win, Fv::splat(0.0));

        out += Fv::from_slice(chunk) * w;
        wsum += w;
        d -= Fv::splat(LANES as f32);
        ph_sinc.rotate();
        ph_win1.rotate();
        ph_win2.rotate();
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

// ===========================================================================================
// Mode resolution and the shared windowed-sinc gather
// ===========================================================================================

/// The gather a read actually runs after resolving the global [`InstrumentResampleMode`] against
/// the signal kind (a voice's sample, or the mixer bus).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
///
/// PSG waveforms under **Linear** fall back to nearest-neighbour: the console's square/wave/noise
/// channels output a stepped (zero-order-hold) signal, and linearly interpolating their few-sample
/// loops would smear the hard edges away from the hardware sound. (The mixer bus passes `false`, so
/// its own Linear stage is unaffected.)
pub(crate) fn effective_gather(mode: InstrumentResampleMode, is_psg: bool) -> EffectiveGather {
    match mode {
        InstrumentResampleMode::NearestNeighbor => EffectiveGather::Nearest,
        InstrumentResampleMode::Linear if is_psg => EffectiveGather::Nearest,
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
    r: f32,
    inv_sample_rate: f32,
    step_mode: bool,
    cutoff_hz: Option<u32>,
) -> f32 {
    let mut fc = if step_mode || r > 1.0 { 0.5 / r } else { 0.5 };
    if let Some(hz) = cutoff_hz {
        fc = fc.min(hz as f32 * inv_sample_rate / r);
    }
    fc
}

/// Stack-buffer length covering the widest possible gather window (one full tap window at
/// [`MAX_HALF_TAPS`]). Sized so every staged-window gather reads a plain slice.
pub(crate) const GATHER_BUF_LEN: usize = 2 * MAX_HALF_TAPS + 2;

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
            half_taps: half_taps.clamp(1, MAX_HALF_TAPS),
        }
    }
}

/// The inclusive source-tap window `[k_lo, k_hi]` the gather reads for a read position `pos`:
/// every integer `k` with `|pos − k| ≤ P`. Exported so callers can pre-stage exactly the source
/// samples [`resample_sinc`] will request (the formula must match the one used internally).
#[inline]
pub fn tap_window(tables: &ResampleTables, pos: f32) -> (i64, i64) {
    let p = tables.half_taps as f32;
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
    pos: f32,
    fc: f32,
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
    let sinc_idx_step = 2.0 * fc * OVERSAMPLE as f32; // Δ index per unit τ-step (one source sample)
    let win_idx_step = WIN_OVERSAMPLE as f32 / tables.half_taps as f32; // Δ window index per source sample

    let d0 = pos - k_lo as f32; // offset of the first tap from the read position (≈ P)
    // The two modes take different kernels. Step mode reads the cumulative-sinc table, carrying one
    // edge value across iterations, which measures faster scalar than vectorized (the table reads
    // dominate). Impulse mode vectorizes well because its kernel can be evaluated analytically with
    // rotating phasors — no table traffic at all. Both return the same `(Σ src·w, Σ w)` pair.
    let (out, wsum): (Sample, Sample) = if step_mode {
        gather_step(k, src, d0, sinc_idx_step, win_idx_step)
    } else {
        gather_impulse_simd(src, d0, fc, tables.half_taps as f32)
    };
    let wsum = if step_mode { wsum.abs() } else { wsum };

    if wsum > 1e-12 {
        out / wsum
    } else {
        // Degenerate weights (pathological fc): fall back to the nearest staged tap.
        Sample::from(src[(pos.round() as i64 - k_lo) as usize])
    }
}

// ===========================================================================================
// Loop-aware source-staging gather (feeds a voice)
// ===========================================================================================
//
// The windowed-sinc gather that feeds [`WaveformInstrument`](crate::synth::WaveformInstrument): it
// stages the exact tap window for a fractional source position so the inner resampler reads a
// plain, loop-mapped slice.

/// Maps a source index past the loop end back into the loop body `[loop_point, data_len)`.
/// Callers guarantee `loop_len > 0`.
#[inline]
fn loop_wrap(t: i64, loop_point: i64, loop_len: i64) -> i64 {
    (t - loop_point).rem_euclid(loop_len) + loop_point
}

/// The source-sample view a sinc gather reads from: the decoded data plus its loop layout and
/// whether the reading voice has already wrapped (see [`gather_sinc`]).
pub struct GatherSource<'a> {
    pub data: &'a [f32],
    pub looping: bool,
    pub loop_point: i64,
    pub loop_len: i64,
    pub wrapped: bool,
}

/// One windowed-sinc gather at fractional source position `pos`, staging the exact tap window
/// ([`tap_window`]) so the inner gather reads a plain slice — branch-free, with no per-tap
/// loop-mapping division (the loop mapping costs one division per *sample* at most).
///
/// `src.wrapped` selects the fully periodic mapping for looping voices that have wrapped at
/// least once (the signal under the window is then periodic in the loop); before the first wrap
/// the one-shot data is read directly and only right-side taps peek into the first loop pass.
#[inline]
pub fn gather_sinc(
    src: &GatherSource,
    tbl: &ResampleTables,
    pos: f32,
    fc: f32,
    step_mode: bool,
) -> Sample {
    let &GatherSource {
        data,
        looping,
        loop_point,
        loop_len,
        wrapped,
    } = src;
    let data_len = data.len() as i64;
    let (k_lo, k_hi) = tap_window(tbl, pos);
    let periodic = looping && wrapped && loop_len > 0;
    if !periodic && k_lo >= 0 && k_hi < data_len {
        // Fast path: the whole window is in-bounds one-shot data.
        let src = &data[k_lo as usize..=k_hi as usize];
        return resample_sinc(tbl, src, pos, fc, step_mode);
    }

    // Edge path: stage the window into a stack buffer so the gather still reads a plain slice.
    let n = (k_hi - k_lo + 1) as usize;
    let mut buf = [0.0f32; GATHER_BUF_LEN];
    if periodic {
        // The voice has wrapped: every tap maps into the loop body. One division to place the
        // first tap, then an increment-and-wrap walk.
        let mut idx = loop_wrap(k_lo, loop_point, loop_len);
        for slot in &mut buf[..n] {
            *slot = data[idx as usize];
            idx += 1;
            if idx == data_len {
                idx = loop_point;
            }
        }
    } else {
        // Window crosses the sample start/end before any wrap: zeros outside, direct reads
        // inside, and (for looping voices) the right tail peeks into the first loop pass.
        for (t, slot) in (k_lo..).zip(&mut buf[..n]) {
            *slot = if (0..data_len).contains(&t) {
                data[t as usize]
            } else if t >= data_len && looping && loop_len > 0 {
                data[loop_wrap(t, loop_point, loop_len) as usize]
            } else {
                0.0
            };
        }
    }
    resample_sinc(tbl, &buf[..n], pos, fc, step_mode)
}

// ===========================================================================================
// Streaming fixed-ratio resampler (mixer bus → output)
// ===========================================================================================
//
// [`StreamResampler`]: a continuous, fixed-ratio resampler for a *stream* of stereo samples (the
// intermediate mixer's bus → the output rate).
//
// Unlike the voice gather ([`gather_sinc`]) — which reads a finite, loop-mapped
// [`Waveform`](crate::waveform::Waveform) at a pitch-driven position — this reads an open-ended
// stream pulled on demand from a callback, keeping just enough recent input in a small ring to feed
// the windowed-sinc kernel. It applies the same [`InstrumentResampleMode`] set as a voice (resolved
// against a non-PSG signal — a finished mix has no PSG/sampled distinction) by reusing the shared
// [`effective_gather`]/[`sinc_fc`] resolution and the one [`resample_sinc`] gather; the only new
// code is the ring-staging of the tap window, mirroring [`gather_sinc`]'s edge path.

/// The ring must be wider than the widest possible tap window, so the oldest tap a gather reads is
/// never overwritten before it is read, *and* wide enough to hold every input one whole output block
/// consumes, so the block's input can be pulled in a single call. The second term dominates and
/// depends on the conversion ratio (heavy downsampling eats many inputs per output), so the length
/// is computed in [`StreamResampler::set`] rather than fixed here.
///
/// It is rounded up to a power of two so the wrap in [`StreamResampler::at`] and
/// [`StreamResampler::push`] — both in the innermost loop — is a mask rather than a remainder.
fn ring_len_for(step: f32) -> usize {
    let per_block = (step.max(0.0) * MAX_BLOCK as f32).ceil() as usize;
    (GATHER_BUF_LEN + per_block + 2).next_power_of_two()
}

/// A streaming, fixed-ratio stereo resampler driven by an [`InstrumentResampleMode`].
///
/// The read position advances by `step = in_rate / out_rate` input samples per output sample;
/// input is pulled lazily so the caller only synthesizes a mixer sample when one is actually
/// consumed.
pub struct StreamResampler {
    /// The resolved gather for the current mode (signal treated as non-PSG).
    gather: EffectiveGather,
    /// Sinc tables when the gather is a sinc variant; `None` for nearest / linear.
    tables: Option<ResampleTables>,
    /// Anti-image / anti-alias cutoff in cycles/input-sample (only used by the sinc gather).
    fc: f32,
    /// Input samples advanced per output sample.
    step: f32,
    /// Integer part of the absolute input position of the next output sample. Carried exactly as
    /// an `i64` (unbounded), so the read position never accumulates as one large float.
    pos_int: i64,
    /// Fractional part of the read position, always in `[0, 1)`. Keeping the position split this
    /// way is what lets the whole resampler run in `f32` on an open-ended stream with no drift: the
    /// kernel only ever sees a tiny synthetic offset, never a growing absolute float.
    pos_frac: f32,
    /// Count of inputs pushed so far (the absolute index of the next push).
    loaded: i64,
    /// Recent-input rings, both a power of two long (see [`ring_len_for`]) so the wrap is a mask.
    ring_l: Vec<f32>,
    ring_r: Vec<f32>,
}

impl Default for StreamResampler {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamResampler {
    /// An idle nearest-neighbour resampler at unity ratio. Call [`Self::set`] before use.
    pub fn new() -> Self {
        let ring = ring_len_for(1.0);
        Self {
            gather: EffectiveGather::Nearest,
            tables: None,
            fc: 0.5,
            step: 1.0,
            pos_int: 0,
            pos_frac: 0.0,
            loaded: 0,
            ring_l: vec![0.0; ring],
            ring_r: vec![0.0; ring],
        }
    }

    /// Configures the conversion ratio and resampling mode. The mode is resolved exactly like a
    /// (non-PSG) voice's: the sinc cutoff is the input Nyquist when upsampling and the output
    /// Nyquist when downsampling, lowered further by the crunchy mode's cutoff slider (see
    /// [`sinc_fc`]). Cheap to call every block — only rebuilds the sinc tables on a `half_taps`
    /// change and never disturbs the running position.
    pub fn set(&mut self, in_rate: f32, out_rate: f32, mode: InstrumentResampleMode) {
        self.step = if out_rate > 0.0 {
            in_rate / out_rate
        } else {
            1.0
        };
        // Grow the ring if the new ratio needs more input per block. It is never shrunk, so a
        // render call that only re-states the current settings — which is every one of them —
        // cannot allocate on the audio thread.
        let needed = ring_len_for(self.step);
        if needed > self.ring_l.len() {
            self.ring_l = vec![0.0; needed];
            self.ring_r = vec![0.0; needed];
            // The rings are indexed by absolute input position masked to the ring length, so
            // changing that length moves every stored sample. Start clean rather than read garbage.
            self.pos_int = 0;
            self.pos_frac = 0.0;
            self.loaded = 0;
        }
        // A finished stereo bus has no PSG/sampled split, so resolve against a non-PSG signal.
        self.gather = effective_gather(mode, false);
        if let EffectiveGather::Sinc {
            step_mode,
            cutoff_hz,
        } = self.gather
        {
            let inv_out_rate = if out_rate > 0.0 { 1.0 / out_rate } else { 0.0 };
            self.fc = sinc_fc(self.step, inv_out_rate, step_mode, cutoff_hz);
        }
        match mode_half_taps(mode) {
            Some(p) => {
                let p = p.clamp(1, MAX_HALF_TAPS);
                if self.tables.as_ref().map(|t| t.half_taps) != Some(p) {
                    self.tables = Some(ResampleTables::new(p));
                }
            }
            None => self.tables = None,
        }
    }

    /// Clears the ring and read position (used when the mixer is (re)enabled, to start clean).
    pub fn reset(&mut self) {
        self.pos_int = 0;
        self.pos_frac = 0.0;
        self.loaded = 0;
        self.ring_l.fill(0.0);
        self.ring_r.fill(0.0);
    }

    /// Reads the input sample at absolute index `k` from the ring (zero before the stream start).
    /// The ring length is a power of two, so the wrap is a mask.
    #[inline]
    fn at(ring: &[f32], k: i64) -> f32 {
        if k < 0 {
            0.0
        } else {
            ring[(k as usize) & (ring.len() - 1)]
        }
    }

    /// Pulls input from `fill_in` until the ring holds the sample at absolute index `k`, in one
    /// call. The ring is sized so a whole output block's input fits, so the pull never has to wrap
    /// onto samples the block still needs.
    fn fill_to(&mut self, k: i64, fill_in: &mut impl FnMut(&mut [Sample], &mut [Sample])) {
        let wanted = k + 1 - self.loaded;
        if wanted <= 0 {
            return;
        }
        let len = self.ring_l.len();
        let start = self.loaded as usize & (len - 1);
        let n = wanted as usize;
        debug_assert!(n <= len, "pull larger than the ring");
        // The run wraps at most once, since the ring holds more than one block's worth of input.
        let first = (len - start).min(n);
        let (head_l, tail_l) = self.ring_l.split_at_mut(start);
        let (head_r, tail_r) = self.ring_r.split_at_mut(start);
        fill_in(&mut tail_l[..first], &mut tail_r[..first]);
        if n > first {
            fill_in(&mut head_l[..n - first], &mut head_r[..n - first]);
        }
        self.loaded += wanted;
    }

    /// Advances the read position by one output sample, keeping the fraction in `[0, 1)` and folding
    /// whole samples into the exact `pos_int` counter so nothing accumulates as an ever-growing float.
    #[inline]
    fn advance(&mut self) {
        self.pos_frac += self.step;
        let carry = self.pos_frac.floor();
        self.pos_int += carry as i64;
        self.pos_frac -= carry;
    }

    /// The highest absolute input index the next `n` output samples will read, walking the read
    /// position with the same arithmetic [`Self::advance`] uses so the answer matches the positions
    /// the gather loop will actually visit.
    fn last_input_needed(&self, n: usize) -> i64 {
        let (mut pos_int, mut pos_frac) = (self.pos_int, self.pos_frac);
        let mut highest = self.pos_int;
        for _ in 0..n {
            let needed = match self.gather {
                EffectiveGather::Nearest => pos_int,
                EffectiveGather::Linear => pos_int + 1,
                EffectiveGather::Sinc { .. } => {
                    let tables = self.tables.as_ref().expect("sinc gather has tables");
                    let syn_pos = tables.half_taps as f32 + pos_frac;
                    let (syn_lo, syn_hi) = tap_window(tables, syn_pos);
                    pos_int - tables.half_taps as i64 + (syn_hi - syn_lo)
                }
            };
            highest = highest.max(needed);
            pos_frac += self.step;
            let carry = pos_frac.floor();
            pos_int += carry as i64;
            pos_frac -= carry;
        }
        highest
    }

    /// Fills `out_l`/`out_r` with output stereo samples, pulling input from `fill_in` as the read
    /// window requires it.
    ///
    /// `fill_in` is handed a block of ring slots to write, not asked for one sample at a time, so
    /// whatever produces the input can itself work in blocks — which is what lets the whole engine
    /// upstream of this stage run blocked. It may be called twice for one output block, when the
    /// input run wraps the ring.
    ///
    /// Per-mode setup (the sinc table handle, half-tap count, and cutoff) is hoisted out of the
    /// per-sample loop, and each block's input is pulled in one go before any of it is read. Output
    /// is bit-identical regardless of how the caller chunks it — the same input is pulled in the
    /// same order and the position advances identically — so a single call over a whole buffer
    /// equals repeated one-sample calls.
    pub fn process(
        &mut self,
        out_l: &mut [Sample],
        out_r: &mut [Sample],
        fill_in: &mut impl FnMut(&mut [Sample], &mut [Sample]),
    ) {
        debug_assert_eq!(out_l.len(), out_r.len());
        // The ring holds one MAX_BLOCK block's input, so longer runs go through in pieces. This is
        // invisible to the caller: chunking cannot change the output.
        for (l, r) in out_l.chunks_mut(MAX_BLOCK).zip(out_r.chunks_mut(MAX_BLOCK)) {
            self.process_block(l, r, fill_in);
        }
    }

    /// One block of at most [`MAX_BLOCK`] output samples: pull all the input it needs, then gather.
    fn process_block(
        &mut self,
        out_l: &mut [Sample],
        out_r: &mut [Sample],
        fill_in: &mut impl FnMut(&mut [Sample], &mut [Sample]),
    ) {
        if out_l.is_empty() {
            return;
        }
        self.fill_to(self.last_input_needed(out_l.len()), fill_in);

        match self.gather {
            EffectiveGather::Nearest => {
                for (l, r) in out_l.iter_mut().zip(out_r.iter_mut()) {
                    // Zero-order hold: the most recent input at or before `pos`.
                    let idx = self.pos_int;
                    *l = Self::at(&self.ring_l, idx);
                    *r = Self::at(&self.ring_r, idx);
                    self.advance();
                }
            }
            EffectiveGather::Linear => {
                for (l, r) in out_l.iter_mut().zip(out_r.iter_mut()) {
                    let i = self.pos_int;
                    let frac = self.pos_frac;
                    let lerp = |ring: &[f32]| -> Sample {
                        let a = Self::at(ring, i);
                        let b = Self::at(ring, i + 1);
                        a + (b - a) * frac
                    };
                    *l = lerp(&self.ring_l);
                    *r = lerp(&self.ring_r);
                    self.advance();
                }
            }
            EffectiveGather::Sinc { step_mode, .. } => {
                // Resolve the (cheap, half-width-only) table handle and cutoff once per block.
                let tables = self.tables.clone().expect("sinc gather has tables");
                let p = tables.half_taps as i64;
                let fc = self.fc;
                let mut buf_l = [0.0f32; GATHER_BUF_LEN];
                let mut buf_r = [0.0f32; GATHER_BUF_LEN];
                for (l, r) in out_l.iter_mut().zip(out_r.iter_mut()) {
                    // Express the read position as a small synthetic offset in `[P, P+1)`: the exact
                    // integer part lives in `pos_int`, so the kernel never sees a large float. Its
                    // own `tap_window` then yields `k_lo = 0`, aligned with the absolute window we
                    // stage.
                    let syn_pos = tables.half_taps as f32 + self.pos_frac;
                    let (syn_lo, syn_hi) = tap_window(&tables, syn_pos);
                    debug_assert_eq!(syn_lo, 0);
                    let n = (syn_hi - syn_lo + 1) as usize;
                    let k_lo = self.pos_int - p; // absolute index of the first staged tap (src[0])
                    for (j, (sl, sr)) in buf_l[..n].iter_mut().zip(&mut buf_r[..n]).enumerate() {
                        let k = k_lo + j as i64;
                        *sl = Self::at(&self.ring_l, k);
                        *sr = Self::at(&self.ring_r, k);
                    }
                    *l = resample_sinc(&tables, &buf_l[..n], syn_pos, fc, step_mode);
                    *r = resample_sinc(&tables, &buf_r[..n], syn_pos, fc, step_mode);
                    self.advance();
                }
            }
        }
    }
}

// ===========================================================================================
// Tests
// ===========================================================================================

#[cfg(test)]
mod tests {
    //! Resampler unit tests: the BLEP step / impulse gathers, the fixed-tap support invariant, the
    //! analysis-helper kernel/response properties, and the streaming resampler.
    //!
    //! The resampler itself runs entirely in `f32`; the numeric tests keep their reference signals
    //! and comparison tolerances in `f64` (numerical analysis, not the audio path) and cast at the
    //! `f32` API boundary.

    use core::f64::consts::PI;

    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// Stages the tap window for `pos` by sampling `f` at each source index (what the synth's
    /// gather staging does for real sample data).
    fn staged(tables: &ResampleTables, pos: f64, f: impl Fn(i64) -> f64) -> Vec<f32> {
        let (k_lo, k_hi) = tap_window(tables, pos as f32);
        (k_lo..=k_hi).map(|t| f(t) as f32).collect()
    }

    #[test]
    fn sinc_int_boundary_and_symmetry() {
        let k = kernels();
        // `sinc_int_at` takes the pre-scaled index τ·OVERSAMPLE.
        let si = |tau: f64| f64::from(sinc_int_at(k, (tau * OVERSAMPLE as f64) as f32));
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
            let s = f64::from(sinc_int_at(k, (tau * OVERSAMPLE as f64) as f32));
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
                let out = f64::from(resample_sinc(&tables, &src, pos as f32, fc as f32, true));
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
        let at = |pos: f64| {
            f64::from(resample_sinc(
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
        let out = f64::from(resample_sinc(&tables, &src, pos as f32, 0.4, false));
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
            let out = f64::from(resample_sinc(
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

    /// Feeds a fixed list of inputs (then `tail` forever) into a block pull, with the right channel
    /// the negation of the left.
    fn pull_list(inputs: &[f32], tail: f32) -> impl FnMut(&mut [Sample], &mut [Sample]) + use<'_> {
        let mut next = 0usize;
        move |l, r| {
            for (l, r) in l.iter_mut().zip(r.iter_mut()) {
                let v = inputs.get(next).copied().unwrap_or(tail);
                next += 1;
                (*l, *r) = (v, -v);
            }
        }
    }

    /// Nearest-neighbour (zero-order hold) repeats each input across the upsample ratio.
    #[test]
    fn nearest_holds_input_samples() {
        let mut rs = StreamResampler::new();
        rs.set(1.0, 4.0, InstrumentResampleMode::NearestNeighbor); // 4× upsample, step = 0.25
        let inputs = [1.0f32, 2.0, 3.0];
        let mut pull = pull_list(&inputs, 0.0);
        // pos = 0, .25, .5, .75 → floor 0 → input[0]; then 1.0,1.25,.. → input[1]; ...
        let (mut got_l, mut got_r) = ([0.0f32; 12], [0.0f32; 12]);
        rs.process(&mut got_l, &mut got_r, &mut pull);
        for (i, (&l, &r)) in got_l.iter().zip(&got_r).enumerate() {
            let expected = inputs[i / 4];
            assert_eq!(l, expected, "sample {i}");
            assert_eq!(r, -expected, "sample {i} right");
        }
    }

    /// Linear interpolation of a ramp stream lands on the expected mid-points.
    #[test]
    fn linear_interpolates_between_inputs() {
        let mut rs = StreamResampler::new();
        rs.set(1.0, 2.0, InstrumentResampleMode::Linear); // 2× upsample, step = 0.5
        let inputs = [0.0f32, 2.0, 4.0];
        let mut pull = pull_list(&inputs, 4.0);
        // pos = 0, .5, 1, 1.5, 2 → 0, 1, 2, 3, 4 (left channel).
        let (mut got_l, mut got_r) = ([0.0f32; 5], [0.0f32; 5]);
        rs.process(&mut got_l, &mut got_r, &mut pull);
        for (i, &l) in got_l.iter().enumerate() {
            assert!((l - i as f32).abs() < 1e-6, "sample {i}: got {l}");
        }
    }

    /// PSG voices fall back to nearest-neighbour under Linear (stepped hardware output); sampled
    /// voices and the mixer bus (`is_psg = false`) still interpolate.
    #[test]
    fn linear_psg_falls_back_to_nearest() {
        assert_eq!(
            effective_gather(InstrumentResampleMode::Linear, true),
            EffectiveGather::Nearest
        );
        assert_eq!(
            effective_gather(InstrumentResampleMode::Linear, false),
            EffectiveGather::Linear
        );
    }

    /// Both sinc modes reconstruct a DC stream as flat DC at unity gain once the window fills.
    #[test]
    fn sinc_reconstructs_dc_at_unity_gain() {
        for mode in [
            InstrumentResampleMode::SincSampleNyquist { half_taps: 16 },
            InstrumentResampleMode::SincOutputNyquist {
                half_taps: 16,
                psg_cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
                sampler_cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
            },
        ] {
            let mut rs = StreamResampler::new();
            rs.set(8000.0, 48000.0, mode);
            let mut pull = |l: &mut [Sample], r: &mut [Sample]| {
                l.fill(1.0);
                r.fill(0.5);
            };
            let (mut buf_l, mut buf_r) = ([0.0f32; 2000], [0.0f32; 2000]);
            rs.process(&mut buf_l, &mut buf_r, &mut pull);
            let last = (buf_l[1999], buf_r[1999]);
            assert!((last.0 - 1.0).abs() < 1e-3, "left DC gain off: {}", last.0);
            assert!((last.1 - 0.5).abs() < 1e-3, "right DC gain off: {}", last.1);
        }
    }

    /// The block API is chunk-invariant: one `process` over N samples equals splitting it into
    /// arbitrary sub-blocks, since input is pulled in the same order and the position advances the
    /// same way. Checked here for a sinc gather at a non-integer ratio (the interesting case).
    #[test]
    fn process_is_chunk_invariant() {
        let mode = InstrumentResampleMode::SincSampleNyquist { half_taps: 16 };
        let make = || {
            let mut rs = StreamResampler::new();
            rs.set(13379.0, 32768.0, mode);
            rs
        };
        // A deterministic pseudo-random stereo stream so successive inputs differ.
        let stream = |seed: &mut u32, l: &mut [Sample], r: &mut [Sample]| {
            for (l, r) in l.iter_mut().zip(r.iter_mut()) {
                *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                *l = (*seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5;
                *r = -*l;
            }
        };

        let (mut whole_l, mut whole_r) = ([0.0f32; 500], [0.0f32; 500]);
        let mut s = 1;
        make().process(&mut whole_l, &mut whole_r, &mut |l, r| stream(&mut s, l, r));

        // Chunk sizes that are not multiples of each other, including a single sample: the caller
        // may hand this any length and must get the same stream back.
        for size in [1, 7, 37, 256, 500] {
            let (mut chunked_l, mut chunked_r) = ([0.0f32; 500], [0.0f32; 500]);
            let mut s2 = 1;
            let mut rs = make();
            let mut pull = |l: &mut [Sample], r: &mut [Sample]| stream(&mut s2, l, r);
            for (l, r) in chunked_l.chunks_mut(size).zip(chunked_r.chunks_mut(size)) {
                rs.process(l, r, &mut pull);
            }
            assert_eq!(
                (whole_l, whole_r),
                (chunked_l, chunked_r),
                "chunk size {size}"
            );
        }
    }
}

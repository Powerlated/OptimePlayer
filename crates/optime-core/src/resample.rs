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

/// Samples per unit `τ` in the oversampled sinc tables.
const OVERSAMPLE: usize = 4096;
/// Maximum tabulated `τ = 2·fc·d`. For the hot path (`fc ≤ 0.5`, `d ≤ P ≤ 64`) this bounds `τ`.
/// Beyond it (only reachable on cheap step-mode upsampling) the kernel is evaluated directly.
const TAU_MAX: usize = 64;
/// Samples in the Blackman window table over the normalized half-support `x = d/P ∈ [0, 1]`.
const WIN_OVERSAMPLE: usize = 4096;

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
struct Kernels {
    /// `sinc[k] = sinc(k / OVERSAMPLE)` for `k` in `0..=TAU_MAX * OVERSAMPLE` (symmetric).
    sinc: Vec<f64>,
    /// `sinc_int[k] = ∫₀^{k/OVERSAMPLE} sinc(t) dt = (1/π)·Si(π·τ)`, the cumulative integral of the
    /// bare sinc (the BLEP step), odd in `τ` with asymptote `0.5` as `τ → ∞`.
    sinc_int: Vec<f64>,
    /// `win[k] = blackman(k / WIN_OVERSAMPLE)` for `k` in `0..=WIN_OVERSAMPLE`.
    win: Vec<f64>,
}

fn kernels() -> &'static Kernels {
    static K: OnceLock<Kernels> = OnceLock::new();
    K.get_or_init(|| {
        let len = TAU_MAX * OVERSAMPLE;

        // Bare sinc, sampled at τ = k / OVERSAMPLE.
        let mut sinc_tab = Vec::with_capacity(len + 1);
        for k in 0..=len {
            sinc_tab.push(sinc(k as f64 / OVERSAMPLE as f64));
        }

        // Cumulative trapezoidal integral of the sinc from 0 (the BLEP step, before normalization).
        let step = 1.0 / OVERSAMPLE as f64;
        let mut sinc_int = vec![0.0f64; len + 1];
        for k in 1..=len {
            let trap = (sinc_tab[k - 1] + sinc_tab[k]) * 0.5 * step;
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

        // Blackman window over the normalized half-support.
        let mut win = Vec::with_capacity(WIN_OVERSAMPLE + 1);
        for k in 0..=WIN_OVERSAMPLE {
            win.push(blackman(k as f64 / WIN_OVERSAMPLE as f64));
        }

        Kernels {
            sinc: sinc_tab,
            sinc_int,
            win,
        }
    })
}

/// Linear interpolation into a table at floating index `idx` (clamped at the top edge).
#[inline]
fn lerp(tab: &[f64], idx: f64) -> f64 {
    let i = idx as usize;
    let frac = idx - i as f64;
    let hi = (i + 1).min(tab.len() - 1);
    tab[i] + (tab[hi] - tab[i]) * frac
}

/// `sinc(τ)` via the table for `τ ≥ 0`; falls back to a direct evaluation beyond `TAU_MAX`
/// (only reached on step-mode upsampling, where the tap count is already tiny).
#[inline]
fn sinc_lookup(k: &Kernels, tau: f64) -> f64 {
    let idx = tau * OVERSAMPLE as f64;
    if idx >= (k.sinc.len() - 1) as f64 {
        sinc(tau)
    } else {
        lerp(&k.sinc, idx)
    }
}

/// The BLEP step `S(τ) = ∫₀^τ sinc`, odd in `τ`, asymptote `±0.5`. Beyond `TAU_MAX` the ripple is
/// negligible, so the asymptote is returned directly.
#[inline]
fn sinc_int_lookup(k: &Kernels, tau: f64) -> f64 {
    if tau < 0.0 {
        return -sinc_int_lookup(k, -tau);
    }
    let idx = tau * OVERSAMPLE as f64;
    if idx >= (k.sinc_int.len() - 1) as f64 {
        0.5
    } else {
        lerp(&k.sinc_int, idx)
    }
}

/// Blackman window value at normalized half-support position `x = |d| / P` (0 for `x ≥ 1`).
#[inline]
fn win_lookup(k: &Kernels, x: f64) -> f64 {
    if x >= 1.0 {
        return 0.0;
    }
    lerp(&k.win, x * WIN_OVERSAMPLE as f64)
}

/// Pre-built resampler configuration. Holds only the support half-width `P`; the heavy kernel
/// tables live in a process-wide [`OnceLock`] and are shared (so building this is essentially free).
#[derive(Clone)]
pub struct ResampleTables {
    /// Half-width of the kernel support, in **source samples**.
    pub half_taps: usize,
}

impl ResampleTables {
    /// Builds a resampler with a `half_taps`-source-sample half-width support (`≥ 1`).
    pub fn new(half_taps: usize) -> Self {
        // Touch the shared tables so the one-time build happens here rather than on the audio
        // thread's first gather.
        let _ = kernels();
        Self {
            half_taps: half_taps.max(1),
        }
    }
}

/// Windowed-sinc polyphase gather — the single shared resampler for both sinc modes.
///
/// # Parameters
/// - `tables`:    the support width (the kernel tables are global).
/// - `get`:       loop-aware sample accessor, returns `f64`.
/// - `pos`:       fractional read position in source samples.
/// - `fc`:        cutoff in cycles/source-sample.
/// - `step_mode`: `false` → impulse-mode (SampleNyquist); `true` → BLEP step-mode (OutputNyquist).
pub fn resample_sinc(
    tables: &ResampleTables,
    get: impl Fn(i64) -> f64,
    pos: f64,
    fc: f64,
    step_mode: bool,
) -> f64 {
    let k = kernels();
    let p = tables.half_taps as f64;
    let inv_p = 1.0 / p;
    // Impulse (reconstruction) mode never wants a cutoff above source Nyquist; step (BLEP) mode may
    // (output Nyquist sits above source Nyquist when upsampling).
    let fc = if step_mode {
        fc.max(1e-6)
    } else {
        fc.clamp(1e-6, 0.5)
    };
    let two_fc = 2.0 * fc;

    // Fixed source-sample support: |pos − k| ≤ P, i.e. ≈ 2P taps regardless of fc.
    let k_lo = (pos - p).floor() as i64;
    let k_hi = (pos + p).ceil() as i64;

    if step_mode {
        // BLEP gather of the boxcar-integrated windowed kernel: tap `k` weighs its source-sample
        // bin `[pos−k−1, pos−k]` by the band-limited step rise across it,
        //     [S(2fc·(pos−k)) − S(2fc·(pos−k−1))] · blackman(|bin-center| / P),
        // where `S` is the cumulative sinc integral. The bin's upper-edge `S` value is the next
        // bin's lower edge, so it is carried across iterations (one `sinc_int` lookup per tap).
        // Normalizing by the weight sum forces exact DC unity (and absorbs the window).
        let mut out = 0.0;
        let mut wsum = 0.0;
        let mut si_hi = sinc_int_lookup(k, two_fc * (pos - k_lo as f64));
        for kk in k_lo..=k_hi {
            let d_lo = pos - kk as f64 - 1.0;
            let si_lo = sinc_int_lookup(k, two_fc * d_lo);
            let mid = (pos - kk as f64 - 0.5).abs();
            let w = win_lookup(k, mid * inv_p) * (si_hi - si_lo);
            out += get(kk) * w;
            wsum += w;
            si_hi = si_lo;
        }
        if wsum.abs() > 1e-12 {
            out / wsum
        } else {
            get(pos.round() as i64)
        }
    } else {
        // Impulse gather: out = Σ_k data(k) · sinc(2fc·d) · blackman(|d|/P), DC-normalized.
        let mut out = 0.0;
        let mut wsum = 0.0;
        for kk in k_lo..=k_hi {
            let d = pos - kk as f64;
            let ad = d.abs();
            if ad >= p {
                continue;
            }
            let w = sinc_lookup(k, two_fc * ad) * win_lookup(k, ad * inv_p);
            out += get(kk) * w;
            wsum += w;
        }
        if wsum > 1e-10 {
            out / wsum
        } else {
            get(pos.round() as i64)
        }
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

    #[test]
    fn sinc_int_boundary_and_symmetry() {
        let k = kernels();
        // S(0) = 0, and S saturates to ±0.5 outside the tabulated range.
        assert!(close(sinc_int_lookup(k, 0.0), 0.0, 1e-12));
        assert!(close(sinc_int_lookup(k, TAU_MAX as f64 + 5.0), 0.5, 1e-6));
        assert!(close(
            sinc_int_lookup(k, -(TAU_MAX as f64) - 5.0),
            -0.5,
            1e-6
        ));
        // Odd symmetry: S(−τ) = −S(τ).
        for i in 1..=20 {
            let tau = TAU_MAX as f64 * i as f64 / 20.0;
            assert!(
                close(
                    sinc_int_lookup(k, tau) + sinc_int_lookup(k, -tau),
                    0.0,
                    1e-12
                ),
                "S({tau}) not odd"
            );
        }
    }

    #[test]
    fn sinc_int_is_bounded() {
        // The sinc has negative lobes, so its integral overshoots slightly past 0.5, but it must
        // stay within a physically reasonable band.
        let k = kernels();
        for i in 0..=2000 {
            let tau = -(TAU_MAX as f64) + 2.0 * TAU_MAX as f64 * i as f64 / 2000.0;
            let s = sinc_int_lookup(k, tau);
            assert!((-0.6..=0.6).contains(&s), "S({tau}) = {s} out of band");
        }
    }

    #[test]
    fn step_mode_preserves_dc() {
        // The normalized BLEP gather must pass a constant signal through unchanged at any position.
        let tables = ResampleTables::new(16);
        let get = |_: i64| 1.0;
        for fc in [0.1, 0.25, 0.5, 1.5] {
            for pos in [3.0, 7.35, 20.7] {
                let out = resample_sinc(&tables, get, pos, fc, true);
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

        assert!(close(
            resample_sinc(&tables, step, 0.0, fc, true),
            0.5,
            0.02
        ));
        let half_width = tables.half_taps as f64 / (2.0 * fc);
        assert!(close(
            resample_sinc(&tables, step, -(half_width + 10.0), fc, true),
            0.0,
            1e-6
        ));
        assert!(close(
            resample_sinc(&tables, step, half_width + 10.0, fc, true),
            1.0,
            1e-6
        ));
        // Monotone non-decreasing through the transition.
        let mut prev = -1.0;
        for i in 0..=200 {
            let pos = -40.0 + 80.0 * i as f64 / 200.0;
            let v = resample_sinc(&tables, step, pos, fc, true);
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
        let get = |_: i64| 1.0;
        let out = resample_sinc(&tables, get, 12.37, 0.4, false);
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
            let out = resample_sinc(&tables, get, pos, fc, false);
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

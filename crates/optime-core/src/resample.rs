//! Variable-ratio windowed-sinc / BLEP resampler.
//!
//! Both sinc modes share a single oversampled Blackman-windowed sinc table.  The cutoff is
//! applied at lookup time by scaling the table index, so no per-voice kernel recomputation
//! occurs — each tap costs one table lerp.
//!
//! * **SampleNyquist (clean)**: impulse-mode gather, `fc = min(0.5, 0.5/r)`.
//! * **OutputNyquist (crunch)**: BLEP step-difference gather at `fc = 0.5/r` (the *output*
//!   Nyquist) for every ratio. For `r > 1` this anti-aliases; for `r ≤ 1` (upsampling) it rises
//!   above source Nyquist (`fc > 0.5`), band-limiting the ZOH stairstep edges to the output rate
//!   while keeping the crunch images that fall below output Nyquist.

use core::f64::consts::PI;

/// Resolution of the oversampled table (samples per zero-crossing interval).
const OVERSAMPLE: usize = 4096;

/// Oversampled Blackman-windowed sinc tables shared across all voices.
///
/// Built once per unique `half_taps` value and cached by [`crate::synth::SampleSynthesizer`].
#[derive(Clone)]
pub struct ResampleTables {
    /// Half-width of the kernel in zero-crossings.
    pub half_taps: usize,
    /// `h_table[k]` = `windowed_sinc(k / OVERSAMPLE)` for `k` in `0..=half_taps * OVERSAMPLE`.
    /// The kernel is symmetric: `h(-τ) = h(τ)`.
    h_table: Vec<f64>,
    /// `s_table[k]` = `∫_0^{k/OVERSAMPLE} h(t) dt` (partial integral from zero, trap rule).
    /// Normalized so that `s_table[half_taps * OVERSAMPLE] == 0.5`.
    /// Full BLEP: `S(u) = 0.5 + s_at(u)` for `u ≥ 0`; `S(u) = 0.5 - s_at(-u)` for `u < 0`.
    s_table: Vec<f64>,
}

impl ResampleTables {
    /// Builds the oversampled table for a kernel with `half_taps` zero-crossings.
    pub fn new(half_taps: usize) -> Self {
        let half_taps = half_taps.max(1);
        let len = half_taps * OVERSAMPLE;

        // h_table: windowed sinc at τ = k / OVERSAMPLE for k in 0..=len.
        let mut h_table = Vec::with_capacity(len + 1);
        for k in 0..=len {
            let t = k as f64 / OVERSAMPLE as f64; // τ in [0, half_taps]
            h_table.push(windowed_sinc_unit(t, half_taps));
        }

        // s_table: cumulative trapezoidal integral of h from 0, normalized to end at 0.5.
        let step = 1.0 / OVERSAMPLE as f64;
        let mut s_table = vec![0.0f64; len + 1];
        for k in 1..=len {
            let trap = (h_table[k - 1] + h_table[k]) * 0.5 * step;
            s_table[k] = s_table[k - 1] + trap;
        }
        // Normalize: the full right-half integral should equal 0.5 (by symmetry of sinc).
        let right_half = s_table[len];
        if right_half > 1e-15 {
            for v in &mut s_table {
                *v *= 0.5 / right_half;
            }
        }

        Self {
            half_taps,
            h_table,
            s_table,
        }
    }

    /// Evaluates the windowed-sinc kernel `h(u)` at any real `u` via linear interpolation.
    ///
    /// Zero outside `[-half_taps, half_taps]`; symmetric.
    fn h_at(&self, u: f64) -> f64 {
        let au = u.abs();
        let n = self.half_taps as f64;
        if au >= n {
            return 0.0;
        }
        let idx = au * OVERSAMPLE as f64;
        let i = idx as usize;
        let frac = idx - i as f64;
        let hi = (i + 1).min(self.h_table.len() - 1);
        self.h_table[i] + (self.h_table[hi] - self.h_table[i]) * frac
    }

    /// Evaluates the cumulative integral `S(u)` (the BLEP step function) at any real `u`.
    ///
    /// `S(-∞) = 0`, `S(0) = 0.5`, `S(+∞) = 1`.  Exploits symmetry: `S(-u) = 1 − S(u)`.
    /// The result is clamped to `[0, 1]` to absorb floating-point rounding near the boundary.
    fn s_at(&self, u: f64) -> f64 {
        let n = self.half_taps as f64;
        if u >= n {
            return 1.0;
        }
        if u <= -n {
            return 0.0;
        }
        let raw = if u >= 0.0 {
            let idx = u * OVERSAMPLE as f64;
            let i = idx as usize;
            let frac = idx - i as f64;
            let hi = (i + 1).min(self.s_table.len() - 1);
            0.5 + self.s_table[i] + (self.s_table[hi] - self.s_table[i]) * frac
        } else {
            1.0 - self.s_at(-u)
        };
        raw.clamp(0.0, 1.0)
    }
}

/// The Blackman-windowed sinc at `τ ∈ [0, N]` with unit cutoff (zero-crossings at integers).
///
/// `h(τ) = sinc(τ) · blackman(τ / N)` where `sinc(x) = sin(πx)/(πx)`.
fn windowed_sinc_unit(t: f64, half_taps: usize) -> f64 {
    let n = half_taps as f64;
    if t >= n {
        return 0.0;
    }
    let sinc = if t < 1e-15 {
        1.0
    } else {
        (PI * t).sin() / (PI * t)
    };
    // Blackman window centered at 0, half-width N:
    // w(τ) = 0.42 + 0.5·cos(π·τ/N) + 0.08·cos(2π·τ/N)
    let x = t / n;
    let window = 0.42 + 0.5 * (PI * x).cos() + 0.08 * (2.0 * PI * x).cos();
    sinc * window
}

/// Windowed-sinc polyphase gather — the single shared resampler for both sinc modes.
///
/// # Parameters
/// - `tables`:    pre-built oversampled kernel tables.
/// - `get`:       sample accessor function (loop-aware, returns `f64`).
/// - `pos`:       fractional read position in source samples.
/// - `fc`:        cutoff in cycles/source-sample (≤ 0.5).
/// - `step_mode`: `false` → impulse-mode (SampleNyquist); `true` → BLEP step-mode (OutputNyquist).
pub fn resample_sinc(
    tables: &ResampleTables,
    get: impl Fn(i64) -> f64,
    pos: f64,
    fc: f64,
    step_mode: bool,
) -> f64 {
    // Impulse (reconstruction) mode never wants a cutoff above source Nyquist. Step (BLEP) mode
    // may: when upsampling we band-limit the stairstep edges at the *output* Nyquist, which is
    // above source Nyquist (fc = 0.5/r > 0.5), so only the lower bound is enforced there.
    let fc = if step_mode {
        fc.clamp(1e-6, 1e6)
    } else {
        fc.clamp(1e-6, 0.5)
    };
    // Tap range: |pos − k| ≤ N/(2·fc) in source-sample coordinates.
    let half_width = tables.half_taps as f64 / (2.0 * fc);
    let k_lo = (pos - half_width).floor() as i64;
    let k_hi = (pos + half_width).ceil() as i64;

    if step_mode {
        // BLEP step-difference gather:
        // out = Σ_k data(k) · [S(2fc·(pos−k)) − S(2fc·(pos−k−1))]
        // The differences telescope to exactly 1 — no normalization needed.
        let mut out = 0.0;
        for k in k_lo..=k_hi {
            let u_curr = 2.0 * fc * (pos - k as f64);
            let u_prev = 2.0 * fc * (pos - k as f64 - 1.0);
            let w = tables.s_at(u_curr) - tables.s_at(u_prev);
            out += get(k) * w;
        }
        out
    } else {
        // Impulse gather:
        // out = Σ_k data(k) · 2fc · h(2fc·(pos−k))
        // Normalized by the actual weight sum (handles windowing imprecision).
        let mut out = 0.0;
        let mut wsum = 0.0;
        for k in k_lo..=k_hi {
            let u = 2.0 * fc * (pos - k as f64);
            let w = 2.0 * fc * tables.h_at(u);
            out += get(k) * w;
            wsum += w;
        }
        if wsum > 1e-10 {
            out / wsum
        } else {
            // Degenerate (pos outside sample data entirely).
            get(pos.round() as i64)
        }
    }
}

// ─── Analysis helpers (used by the filter-plot popup in optime-app) ──────────────────────────

/// Returns the normalized kernel taps `h_fc[k]` for integer source offsets
/// `k = −(N−1)..=(N−1)`, suitable for drawing as a stem plot.
///
/// The taps are scaled so the cutoff is at `fc` (cycles/source-sample) and normalized
/// to unit DC gain.
pub fn fir_kernel(half_taps: usize, fc: f64) -> Vec<f64> {
    let n = half_taps.max(1) as i64;
    let fc = fc.clamp(1e-6, 0.5);
    let taps: Vec<f64> = (-(n - 1)..=(n - 1))
        .map(|k| {
            let t = 2.0 * fc * k.abs() as f64; // |τ| in table coordinates
            2.0 * fc * windowed_sinc_unit(t, half_taps)
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
/// `w_norm ∈ [0, π]`.  Returns `(magnitude, phase_radians)`.
///
/// The kernel is linear-phase (symmetric), so the phase is exactly linear in `w_norm`.
pub fn fir_response(half_taps: usize, fc: f64, w_norm: f64) -> (f64, f64) {
    let kernel = fir_kernel(half_taps, fc);
    let center = kernel.len() / 2; // index of the zero-lag tap
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (i, &h) in kernel.iter().enumerate() {
        let delay = i as f64 - center as f64;
        // z^{-delay} = e^{−jw·delay}
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
    fn blep_boundary_and_symmetry() {
        // Boundary conditions: S must approach 0 and 1 outside the kernel support.
        let tables = ResampleTables::new(8);
        let n = tables.half_taps as f64;
        assert!(
            close(tables.s_at(-n - 1.0), 0.0, 1e-10),
            "S(-N-1) should be 0"
        );
        assert!(
            close(tables.s_at(n + 1.0), 1.0, 1e-10),
            "S(N+1) should be 1"
        );
        // Anti-symmetry: S(-u) = 1 - S(u).
        for i in 1..=20 {
            let u = n * i as f64 / 20.0;
            let s_pos = tables.s_at(u);
            let s_neg = tables.s_at(-u);
            assert!(
                close(s_pos + s_neg, 1.0, 1e-10),
                "S({u}) + S(-{u}) = {} (should be 1)",
                s_pos + s_neg
            );
        }
        // S must be bounded in [0, 1] — the windowed sinc has small negative lobes so its
        // integral is not strictly monotone, but it must stay within a physically reasonable
        // range (the Blackman window keeps overshoot < 1%).
        for i in 0..=1000 {
            let u = -n + 2.0 * n * i as f64 / 1000.0;
            let s = tables.s_at(u);
            assert!(
                (-0.01..=1.01).contains(&s),
                "S({u}) = {s} is outside [-0.01, 1.01]"
            );
        }
    }

    #[test]
    fn blep_midpoint_is_half() {
        let tables = ResampleTables::new(8);
        assert!(close(tables.s_at(0.0), 0.5, 1e-6));
    }

    #[test]
    fn step_mode_weights_sum_to_one() {
        // For any position, the BLEP step-difference weights must telescope to 1.
        let tables = ResampleTables::new(8);
        let fc = 0.3;
        let pos = 7.35_f64;
        let half_width = tables.half_taps as f64 / (2.0 * fc);
        let k_lo = (pos - half_width).floor() as i64;
        let k_hi = (pos + half_width).ceil() as i64;
        let weight_sum: f64 = (k_lo..=k_hi)
            .map(|k| {
                tables.s_at(2.0 * fc * (pos - k as f64))
                    - tables.s_at(2.0 * fc * (pos - k as f64 - 1.0))
            })
            .sum();
        assert!(
            close(weight_sum, 1.0, 1e-10),
            "step weights sum = {weight_sum}"
        );
    }

    #[test]
    fn impulse_mode_dc_gain() {
        // A constant source signal should pass through unchanged (DC gain ≈ 1).
        let tables = ResampleTables::new(16);
        let fc = 0.4;
        let get = |_: i64| 1.0;
        let out = resample_sinc(&tables, get, 12.37, fc, false);
        assert!(close(out, 1.0, 1e-6), "DC gain = {out}");
    }

    #[test]
    fn impulse_mode_passband_signal_reconstructed() {
        // A signal well within the passband (freq << fc) should be reconstructed faithfully at
        // any fractional position.  We use a cosine at 0.1 × Nyquist (fc = 0.45 → well inside).
        let tables = ResampleTables::new(16);
        let fc = 0.45;
        let f0 = 0.05; // signal frequency (cycles/source-sample)
        let get = |k: i64| (2.0 * core::f64::consts::PI * f0 * k as f64).cos();
        // Test at a few non-integer positions; the reconstructed value should be close to the
        // ideal continuous-time cos.
        for frac in [0.0, 0.25, 0.5, 0.75] {
            let pos = 32.0 + frac;
            let ideal = (2.0 * core::f64::consts::PI * f0 * pos).cos();
            let out = resample_sinc(&tables, get, pos, fc, false);
            assert!(
                close(out, ideal, 1e-3),
                "at pos={pos}: reconstructed={out}, ideal={ideal}"
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
            assert!(
                close(k[i], k[n - 1 - i], 1e-15),
                "asymmetry at {i}: {} vs {}",
                k[i],
                k[n - 1 - i]
            );
        }
    }

    #[test]
    fn fir_response_dc_gain_near_one() {
        let (mag, _) = fir_response(16, 0.45, 0.0);
        assert!(close(mag, 1.0, 1e-10), "DC magnitude = {mag}");
    }

    #[test]
    fn stopband_suppression_improves_with_taps() {
        // Frequency-domain twin of the synth-level alias test, with no resampling/DFT noise:
        // for a fixed cutoff, a deeper kernel sharpens the transition and pushes a stopband tone
        // further down. Cutoff fc = 0.25 (w_cut = π/2); probe at 0.4 cyc/sample (w = 0.8π), well
        // inside the stopband.
        let w_stop = 2.0 * PI * 0.4;
        let taps = [2usize, 4, 8, 16, 32];
        let mags: Vec<f64> = taps
            .iter()
            .map(|&t| fir_response(t, 0.25, w_stop).0)
            .collect();
        for w in mags.windows(2) {
            assert!(
                w[1] < w[0],
                "stopband magnitude should fall with more taps, got {mags:?} for {taps:?}"
            );
        }
        // The largest kernel should land deep in the stopband (< −40 dB).
        assert!(
            *mags.last().unwrap() < 0.01,
            "32-tap stopband magnitude = {} (expected < 0.01)",
            mags.last().unwrap()
        );
        // Meanwhile the passband stays flat (≈ unity) for every tap count, so the suppression
        // gain isn't coming from a collapsing passband.
        for &t in &taps {
            let (pass, _) = fir_response(t, 0.25, 2.0 * PI * 0.05);
            assert!(pass > 0.95, "passband magnitude at {t} taps = {pass}");
        }
    }

    // ── Crunch-mode (OutputNyquist / step_mode=true) band-limiting tests ──────────────────

    /// The BLEP step-difference gather of a unit source step telescopes exactly to S(2·fc·pos).
    ///
    /// Mathematically:
    ///   Σ_{k≥0} [S(2fc(pos−k)) − S(2fc(pos−k−1))]  =  S(2fc·pos) − S(−∞)  =  S(2fc·pos)
    ///
    /// This is the defining property of the BLEP resampler: source step edges become the
    /// band-limited step function at output Nyquist, not a hard discontinuity.
    #[test]
    fn step_mode_reproduces_bandlimited_step_shape() {
        let tables = ResampleTables::new(32);
        let r = 4.0_f64;
        let fc = 0.5 / r; // = 0.125  (output Nyquist for 4× downsampling)
        let step = |k: i64| if k >= 0 { 1.0_f64 } else { 0.0 };

        // Positions within the kernel support: resampler output must equal S(2fc·pos).
        let positions = [-25.0_f64, -8.3, -1.0, 0.0, 2.7, 9.0, 25.0];
        for pos in positions {
            let out = resample_sinc(&tables, step, pos, fc, true);
            let expected = tables.s_at(2.0 * fc * pos);
            assert!(
                close(out, expected, 1e-9),
                "at pos={pos}: got {out}, expected S(2·fc·pos) = {expected}"
            );
        }

        // Far outside kernel support: must settle to 0 and 1.
        let half_width = tables.half_taps as f64 / (2.0 * fc);
        let far_below = -(half_width + 10.0);
        let far_above = half_width + 10.0;
        assert!(
            close(resample_sinc(&tables, step, far_below, fc, true), 0.0, 1e-6),
            "far below edge should be 0"
        );
        assert!(
            close(resample_sinc(&tables, step, far_above, fc, true), 1.0, 1e-6),
            "far above edge should be 1"
        );
    }

    /// Sinusoidal passband / stopband test: crunch mode acts as a lowpass at output Nyquist `fc`.
    ///
    /// The BLEP step-difference gather is equivalent to first ZOH-ing the source and then
    /// convolving with the windowed-sinc lowpass at `fc`.  A source sinusoid *below* `fc`
    /// must pass through at near-unity amplitude; one *well above* `fc` must be strongly
    /// suppressed.
    ///
    /// Choosing the stopband frequency 2.8× above `fc` (0.35 vs 0.125) places it comfortably
    /// inside the Blackman stopband (−58 dB sidelobes → amplitude < 0.002), so the 0.05
    /// threshold is conservatively loose.
    #[test]
    fn step_mode_sinusoidal_bandlimiting() {
        use core::f64::consts::PI;

        let tables = ResampleTables::new(16);
        let r = 4.0_f64; // 4× downsampling
        let fc = 0.5 / r; // = 0.125 (output Nyquist)

        // Measure peak output amplitude for a sinusoidal source at frequency `f`
        // (cycles/source-sample), advancing `r` source samples per output step.
        // We skip the first 50 output positions so any transient from the kernel
        // boundary has settled, then observe 100 output samples.
        let peak_amp = |f: f64| -> f64 {
            (50..150_usize)
                .map(|n| {
                    let pos = n as f64 * r;
                    let get = |k: i64| (2.0 * PI * f * k as f64).sin();
                    resample_sinc(&tables, get, pos, fc, true).abs()
                })
                .fold(0.0f64, f64::max)
        };

        // Passband (f = 0.02 << fc = 0.125): ZOH sinc factor ≈ 1, lowpass gain ≈ 1.
        // The observed amplitude at positions n·r cycles at output freq 0.08 cycles/step,
        // so 100 samples span 8 complete cycles — the peak is well within the window.
        let amp_pass = peak_amp(0.02);
        assert!(
            amp_pass > 0.7,
            "passband sinusoid (f=0.02, fc=0.125): amplitude = {amp_pass:.4} (expected > 0.7)"
        );

        // Stopband (f = 0.35 >> fc = 0.125): no ZOH image falls below fc, so the
        // Blackman lowpass suppresses all content.  −58 dB → amplitude ≈ 0.002 << 0.05.
        let amp_stop = peak_amp(0.35);
        assert!(
            amp_stop < 0.05,
            "stopband sinusoid (f=0.35 >> output Nyquist {fc}): amplitude = {amp_stop:.4} (expected < 0.05)"
        );
    }
}

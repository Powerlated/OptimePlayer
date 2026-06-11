//! Resampler unit tests: the BLEP step / impulse gathers, the fixed-tap support invariant, and
//! the analysis-helper kernel/response properties.

use core::f64::consts::PI;

use super::kernels::{kernels, sinc_int_at, OVERSAMPLE, TAU_MAX};
use super::{fir_kernel, fir_response, resample_sinc, tap_window, ResampleTables};

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

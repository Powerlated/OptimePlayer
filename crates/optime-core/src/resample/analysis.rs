//! Analysis helpers used by the filter-plot popup in optime-app: the normalized kernel taps and
//! the frequency response of the windowed-sinc low-pass.

use super::kernels::kernel_weight;

/// Returns the normalized kernel taps `k(d)` for integer source offsets `d = −(P−1)..=(P−1)`,
/// suitable for drawing as a stem plot. Scaled to the cutoff `fc` and normalized to unit DC gain.
pub fn fir_kernel(half_taps: usize, fc: f64) -> Vec<f64> {
    let p = half_taps.max(1);
    let pf = p as f64;
    let fc = fc.clamp(1e-6, 0.5);
    let taps: Vec<f64> = (-(p as i64 - 1)..=(p as i64 - 1))
        .map(|d| kernel_weight(d as f64, fc, pf))
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

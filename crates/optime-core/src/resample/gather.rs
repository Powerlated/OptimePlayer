//! The two scalar gather kernels that sum a staged tap window against the windowed-sinc kernel.
//! Each returns the `(Σ src·w, Σ w)` pair the caller DC-normalizes.

#[cfg(not(feature = "simd"))]
use super::kernels::sinc_at;
use super::kernels::{sinc_int_at, win_at, Kernels};

/// Scalar BLEP gather of the boxcar-integrated windowed kernel: tap `j` weighs its source-sample
/// bin `[pos−k−1, pos−k]` (where `d_hi = d0 − j = pos − k`) by the band-limited step rise across
/// it,
///     `[S(2fc·d_hi) − S(2fc·(d_hi−1))] · blackman(|bin-center| / P)`,
/// where `S` is the cumulative sinc integral. The bin's upper-edge `S` value is the next bin's
/// lower edge, so it is carried across iterations (one `sinc_int` lookup per tap). Normalizing by
/// the weight sum forces exact DC unity (and absorbs the window). Returns `(Σ src·w, Σ w)`.
pub(super) fn gather_step(
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
pub(super) fn gather_impulse(
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

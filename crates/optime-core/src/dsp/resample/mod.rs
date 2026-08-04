//! The resampler seam. `Resampler` is what an implementation satisfies: build tables for a tap
//! count, report the window those tables need around a fractional position, and interpolate a
//! contiguous slice covering that window. Everything above it — voices, the mixer bus — names the
//! implementation as a type parameter defaulting to `DefaultResampler`, so the choice is made at
//! compile time and no dyn dispatch or runtime branch reaches the hot path.
//!
//! The provided `gather` is why an implementation writes only a kernel: it owns the awkward part,
//! turning a `GatherSource` (a waveform plus its loop geometry) into that contiguous slice. Inside
//! the waveform with no wrap in play it borrows the slice directly; otherwise it copies the window
//! into a stack buffer, following the loop backwards and forwards so taps that fall off either end
//! read the looped signal rather than silence, and zero-filling only where there is genuinely
//! nothing. `MAX_HALF_TAPS` caps the window so that buffer can be a fixed-size array.

pub mod r#impl;
pub mod mode;
pub mod stream;

pub use r#impl::{ResampleImplSimd, ResampleImplSimdClosedForm};
pub(crate) use mode::{EffectiveGather, effective_gather, mode_half_taps, sinc_fc};
pub use stream::StreamResampler;

use crate::waveform::Sample;

pub type DefaultResampler = ResampleImplSimd;

pub const MAX_HALF_TAPS: usize = 64;
pub(crate) const GATHER_BUF_LEN: usize = 2 * MAX_HALF_TAPS + 2;

pub struct GatherSource<'a> {
    pub data: &'a [f32],
    pub looping: bool,
    pub loop_point: i64,
    pub loop_len: i64,
    pub wrapped: bool,
}

pub trait Resampler {
    type Tables: Clone;

    fn tables(half_taps: usize) -> Self::Tables;

    fn half_taps(tables: &Self::Tables) -> usize;

    fn tap_window(tables: &Self::Tables, pos: f32) -> (i64, i64);

    fn resample(tables: &Self::Tables, src: &[f32], pos: f32, fc: f32, step_mode: bool) -> Sample;

    #[inline]
    fn gather(
        source: &GatherSource,
        tables: &Self::Tables,
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
        } = source;
        let data_len = data.len() as i64;
        let (k_lo, k_hi) = Self::tap_window(tables, pos);
        let periodic = looping && wrapped && loop_len > 0;
        if !periodic && k_lo >= 0 && k_hi < data_len {
            let src = &data[k_lo as usize..=k_hi as usize];
            return Self::resample(tables, src, pos, fc, step_mode);
        }

        let n = (k_hi - k_lo + 1) as usize;
        let mut buf = [0.0f32; GATHER_BUF_LEN];
        if periodic {
            let mut idx = loop_wrap(k_lo, loop_point, loop_len);
            for slot in &mut buf[..n] {
                *slot = data[idx as usize];
                idx += 1;
                if idx == data_len {
                    idx = loop_point;
                }
            }
        } else {
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
        Self::resample(tables, &buf[..n], pos, fc, step_mode)
    }
}

#[inline]
fn loop_wrap(t: i64, loop_point: i64, loop_len: i64) -> i64 {
    (t - loop_point).rem_euclid(loop_len) + loop_point
}

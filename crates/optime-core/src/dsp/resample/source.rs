//! The windowed-sinc gather that feeds [`WaveformInstrument`](crate::synth::WaveformInstrument): it
//! stages the exact tap window for a fractional source position so the inner resampler reads a
//! plain, loop-mapped slice.

use super::{GATHER_BUF_LEN, ResampleTables, resample_sinc, tap_window};
use crate::waveform::Sample;

/// Maps a source index past the loop end back into the loop body `[loop_point, data_len)`.
/// Callers guarantee `loop_len > 0`.
#[inline]
pub(crate) fn loop_wrap(t: i64, loop_point: i64, loop_len: i64) -> i64 {
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
/// (`resample::tap_window`) so the inner gather reads a plain slice — branch-free, with no
/// per-tap loop-mapping division (the loop mapping costs one division per *sample* at most).
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

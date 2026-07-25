//! The block-processing conventions every stage in the signal chain follows.
//!
//! Each stage exposes a `*_block` method that processes a slice of consecutive samples in one call.
//! The block form is the primitive and the single-sample form, where one still exists, is a wrapper
//! that passes a one-element slice — so the two can never disagree. A block of length 1 is always
//! legal and always means exactly one sample, which is what lets a caller that genuinely needs
//! sample-at-a-time control keep it without a second code path.
//!
//! Stereo travels as two separate slices rather than one slice of `(left, right)` pairs. Every
//! per-channel stage in the chain — the biquads, the Haas delay lines, the voice gathers — then
//! reads a dense contiguous run instead of striding past the other channel's samples, and the
//! arithmetic stages can be vectorised across samples. The two channels are interleaved once, at
//! the boundary where audio leaves for the output device or a WAV file.
//!
//! A block never spans a device tick: voice parameters only change on ticks, so a stage may hoist
//! any setup that depends on them out of its inner loop.

use crate::waveform::Sample;

/// Maximum block length, in samples, that a stage must accept in one call. Sized to cover one full
/// sequencer tick at common output rates (≈251 samples at 48 kHz) so a block rarely splits, and
/// small enough that a stage may keep a scratch buffer of this length inline.
///
/// Callers with more work than this split it into successive blocks; every stage's output is
/// independent of how the work is split.
pub const MAX_BLOCK: usize = 256;

/// Checks the invariant a stereo block stage relies on: both channels are the same length, and the
/// block fits the scratch buffers sized by [`MAX_BLOCK`]. Returns the block length.
#[inline]
pub fn stereo_len(l: &[Sample], r: &[Sample]) -> usize {
    debug_assert_eq!(l.len(), r.len(), "stereo block channels differ in length");
    debug_assert!(l.len() <= MAX_BLOCK, "block longer than MAX_BLOCK");
    l.len()
}

/// The block lengths every stage's equivalence test runs: a single sample (the length a caller uses
/// when it needs sample-at-a-time control), a couple of awkward short blocks, and the two longest
/// blocks a stage will ever see.
#[cfg(test)]
pub(crate) const TEST_BLOCK_LENGTHS: [usize; 5] = [1, 2, 3, MAX_BLOCK - 1, MAX_BLOCK];

/// A deterministic pseudo-random signal in roughly `-0.5..0.5`, so a block equivalence test drives
/// its stage with something that actually varies sample to sample. Same generator the resampler's
/// chunk-invariance test uses.
#[cfg(test)]
pub(crate) fn test_signal(len: usize) -> Vec<Sample> {
    let mut seed = 1u32;
    (0..len)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 9) as Sample / (1u32 << 23) as Sample - 0.5
        })
        .collect()
}

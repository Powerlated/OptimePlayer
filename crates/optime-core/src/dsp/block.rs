use crate::waveform::Sample;

pub const MAX_BLOCK: usize = 256;

#[inline]
pub fn stereo_len(l: &[Sample], r: &[Sample]) -> usize {
    debug_assert_eq!(l.len(), r.len(), "stereo block channels differ in length");
    debug_assert!(l.len() <= MAX_BLOCK, "block longer than MAX_BLOCK");
    l.len()
}

#[cfg(test)]
pub(crate) const TEST_BLOCK_LENGTHS: [usize; 5] = [1, 2, 3, MAX_BLOCK - 1, MAX_BLOCK];

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

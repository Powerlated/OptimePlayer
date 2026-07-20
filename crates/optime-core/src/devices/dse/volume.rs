//! DSE voice-volume math — a faithful transcription of the volume path in `DseVoice_PlayNote`,
//! `DseChannel_Init`, and `DseVoice_UpdateParameters` (`lib/DSE/asm/main_02071EB4.s` in
//! `pret/pmd-sky`).
//!
//! Unlike the SSEQ engine (which sums attenuations in the decibel domain), the DSE driver
//! combines its 0..=127 volumes by **integer multiplication**, renormalizing each product through
//! a divide-by-constant, and applies a **square law** to the final voice volume:
//!
//! ```text
//! note_volume   = velocity * program.volume * split.volume / 127^2      (DseVoice_PlayNote)
//! volume_final  = track_volume * expression / 127                       (DseChannel_*)
//! r             = envelope_level * volume_final * note_volume / 8032     (DseVoice_UpdateParameters)
//! voice_volume  = (r * r) >> 9                                           (0..=127, → hardware)
//! ```
//!
//! The two divisors are exactly the compiler magic-number divisions in the ROM: `/16129` (=127²,
//! magic `0x82061029 >> 13`) and `/8032` (magic `0x828CBFBF >> 12`). `8032` is chosen so the
//! triple product maps to 0..=255 before squaring back to 0..=127. We reproduce the ARM
//! signed-divide idiom bit-for-bit so the rounding matches the hardware.

/// The ARM "signed divide by a constant" idiom the DSE driver uses: `smull` to get the high 32
/// bits of `magic * x`, add `x` back, arithmetic-shift right by `shift`, then add the sign bit.
/// Bit-identical to e.g. `smull r1,r2,M,r3; add r2,r3,r2; mov r0,r3,lsr#31; add r2,r0,r2,asr#s`.
#[inline]
fn sdiv_magic(x: i32, magic: u32, shift: u32) -> i32 {
    let high = ((magic as i32 as i64) * i64::from(x)) >> 32;
    let t = i64::from(x) + high;
    (t >> shift) as i32 + ((x as u32 >> 31) as i32)
}

/// Combines two 0..=127 controls the way `DseChannel_Init` builds `volume_final`: `a * b / 127`
/// (via the `*127 / 16129` magic).
#[inline]
fn combine(a: i32, b: i32) -> i32 {
    sdiv_magic(a * b * 127, 0x8206_1029, 13)
}

/// `DseVoice_PlayNote`'s per-note volume: `velocity * program.volume * split.volume / 127^2`
/// (0..=127). Computed once when the note starts.
pub fn note_volume(velocity: u8, program_volume: u8, split_volume: u8) -> u8 {
    let prod = i32::from(velocity) * i32::from(program_volume) * i32::from(split_volume);
    sdiv_magic(prod, 0x8206_1029, 13).clamp(0, 127) as u8
}

/// The channel's combined `volume_final` (0..=127): `track_volume * expression / 127`. The song
/// and synth global volumes default to full (127) and so drop out; `SongVolumeFade` is not yet
/// modelled.
pub fn volume_final(track_volume: u8, expression: u8) -> u8 {
    combine(i32::from(track_volume), i32::from(expression)).clamp(0, 127) as u8
}

/// `DseVoice_UpdateParameters`'s final voice volume as a linear 0..=1 amplitude: the triple
/// product `envelope_level * volume_final * note_volume / 8032`, **squared** and shifted back to
/// 0..=127, then normalized. The square law is what gives DSE its characteristic volume taper.
pub fn voice_amp(envelope_level: i8, volume_final: u8, note_volume: u8) -> f64 {
    let env = i32::from(envelope_level.max(0));
    let prod = env * i32::from(volume_final) * i32::from(note_volume);
    let r = sdiv_magic(prod, 0x828C_BFBF, 12);
    let squared = r * r;
    // The ROM's rounding term `(sq>>8) >> 23` is ~0 for any in-range volume; keep it for fidelity.
    let hw = (squared + ((squared >> 8) >> 23)) >> 9;
    f64::from(hw.clamp(0, 127)) / 127.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference transcription of the ARM magic-divide idiom, computed independently.
    fn ref_sdiv(x: i64, magic: u32, shift: u32) -> i64 {
        let m = i64::from(magic as i32);
        let high = (m * x) >> 32;
        ((x + high) >> shift) + ((x as u64 >> 63) as i64)
    }

    #[test]
    fn note_volume_is_triple_product_over_127_squared() {
        assert_eq!(note_volume(127, 127, 127), 127);
        // Halving any single factor quarters nothing here (linear product, /127^2).
        assert_eq!(note_volume(64, 127, 127), 64);
        assert_eq!(note_volume(127, 64, 127), 64);
        assert_eq!(note_volume(0, 127, 127), 0);
        // Matches the magic division exactly across a grid.
        for &v in &[0u8, 1, 50, 100, 127] {
            for &p in &[64u8, 100, 127] {
                for &s in &[32u8, 127] {
                    let want =
                        ref_sdiv(i64::from(v) * i64::from(p) * i64::from(s), 0x8206_1029, 13);
                    assert_eq!(i64::from(note_volume(v, p, s)), want.clamp(0, 127));
                }
            }
        }
    }

    #[test]
    fn voice_amp_applies_the_square_law() {
        // Full chain peaks at 1.0.
        assert!((voice_amp(127, 127, 127) - 1.0).abs() < 1e-9);
        // Halving one input quarters the amplitude (the square law): full=127 -> ~32/127.
        let full = voice_amp(127, 127, 127);
        let half_env = voice_amp(64, 127, 127);
        assert!(
            (half_env / full - 0.25).abs() < 0.05,
            "halving a factor should ~quarter the amp: {half_env} vs {full}"
        );
        // Monotonic and bounded.
        assert!(voice_amp(0, 127, 127).abs() < 1e-9);
        assert!(voice_amp(127, 0, 127).abs() < 1e-9);
    }

    #[test]
    fn volume_final_combines_track_and_expression() {
        assert_eq!(volume_final(127, 127), 127);
        assert_eq!(volume_final(64, 127), 64);
        assert_eq!(volume_final(127, 64), 64);
        assert_eq!(volume_final(0, 127), 0);
    }
}

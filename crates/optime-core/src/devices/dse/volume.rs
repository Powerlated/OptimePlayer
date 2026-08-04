#[inline]
fn sdiv_magic(x: i32, magic: u32, shift: u32) -> i32 {
    let high = ((magic as i32 as i64) * i64::from(x)) >> 32;
    let t = i64::from(x) + high;
    (t >> shift) as i32 + ((x as u32 >> 31) as i32)
}

#[inline]
fn combine(a: i32, b: i32) -> i32 {
    sdiv_magic(a * b * 127, 0x8206_1029, 13)
}

pub fn note_volume(velocity: u8, program_volume: u8, split_volume: u8) -> u8 {
    let prod = i32::from(velocity) * i32::from(program_volume) * i32::from(split_volume);
    sdiv_magic(prod, 0x8206_1029, 13).clamp(0, 127) as u8
}

pub fn volume_final(track_volume: u8, expression: u8) -> u8 {
    combine(i32::from(track_volume), i32::from(expression)).clamp(0, 127) as u8
}

pub fn voice_amp(envelope_level: i8, volume_final: u8, note_volume: u8) -> f64 {
    let env = i32::from(envelope_level.max(0));
    let prod = env * i32::from(volume_final) * i32::from(note_volume);
    let r = sdiv_magic(prod, 0x828C_BFBF, 12);
    let squared = r * r;
    let hw = (squared + ((squared >> 8) >> 23)) >> 9;
    f64::from(hw.clamp(0, 127)) / 127.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_sdiv(x: i64, magic: u32, shift: u32) -> i64 {
        let m = i64::from(magic as i32);
        let high = (m * x) >> 32;
        ((x + high) >> shift) + ((x as u64 >> 63) as i64)
    }

    #[test]
    fn note_volume_is_triple_product_over_127_squared() {
        assert_eq!(note_volume(127, 127, 127), 127);
        assert_eq!(note_volume(64, 127, 127), 64);
        assert_eq!(note_volume(127, 64, 127), 64);
        assert_eq!(note_volume(0, 127, 127), 0);
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
        assert!((voice_amp(127, 127, 127) - 1.0).abs() < 1e-9);
        let full = voice_amp(127, 127, 127);
        let half_env = voice_amp(64, 127, 127);
        assert!(
            (half_env / full - 0.25).abs() < 0.05,
            "halving a factor should ~quarter the amp: {half_env} vs {full}"
        );
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

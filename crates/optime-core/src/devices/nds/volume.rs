//! Channel-volume math: the decibel-domain attenuation chain pokediamond runs in
//! `SND_ExChannelMain` + `SND_CalcChannelVolume`.

use super::tables::{DECIBEL_SQUARE_TABLE, GET_VOL_TABLE};

/// Decibel-square-table lookup for a 0..127 volume/expression/velocity level (clamped).
#[inline]
pub(super) fn decibel_db(level: i32) -> i32 {
    DECIBEL_SQUARE_TABLE[level.clamp(0, 127) as usize]
}

/// Computes a channel's linear volume the way pokediamond's `SND_ExChannelMain` does:
/// it sums every attenuation contribution in the decibel domain — velocity, the ADSR envelope
/// (`adsr_timer >> 7`), and `extra_db` (track volume + expression + master + volume-LFO) — then
/// runs the combined value through `SND_CalcChannelVolume`.
pub fn calc_channel_volume(velocity: i32, adsr_timer: i32, extra_db: i32) -> f64 {
    const SND_VOL_DB_MIN: i32 = -723;

    let mut vol = decibel_db(velocity);
    vol += adsr_timer >> 7;
    vol += extra_db;
    vol = vol.clamp(SND_VOL_DB_MIN, 0);

    let mut result = f64::from(GET_VOL_TABLE[(vol - SND_VOL_DB_MIN) as usize]);
    if vol < -240 {
        result /= 16.0;
    } else if vol < -120 {
        result /= 4.0;
    } else if vol < -60 {
        result /= 2.0;
    }

    result / 127.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference transcription of pokediamond's channel-volume chain: the `SND_ExChannelMain`
    /// decibel accumulation (`vol = DecibelSquareTable[velocity] + (envAttenuation >> 7) +
    /// userDecay`) followed by `SND_CalcChannelVolume` (`SND_util.c`), with the DS volume divider
    /// (shift 0..3 → ÷1, ÷2, ÷4, ÷16) and our 0..1 normalization.
    fn pokediamond_channel_volume(velocity: i32, env_att: i32, user_decay: i32) -> f64 {
        let mut value =
            DECIBEL_SQUARE_TABLE[velocity.clamp(0, 127) as usize] + (env_att >> 7) + user_decay;
        // SND_CalcChannelVolume clamps the summed attenuation to [SND_VOL_DB_MIN, 0].
        value = value.clamp(-723, 0);
        let mut result = f64::from(GET_VOL_TABLE[(value + 723) as usize]);
        if value < -240 {
            result /= 16.0;
        } else if value < -120 {
            result /= 4.0;
        } else if value < -60 {
            result /= 2.0;
        }
        result / 127.0
    }

    #[test]
    fn calc_channel_volume_matches_pokediamond() {
        // Velocity + envelope + (track volume + expression + master) must all combine in the
        // decibel domain, exactly as `SND_ExChannelMain` accumulates them.
        for &velocity in &[0, 1, 50, 100, 127] {
            for &env_att in &[-92544, -46272, -10000, -1000, -128, 0] {
                for &volume in &[0usize, 32, 64, 100, 127] {
                    for &expression in &[64usize, 100, 127] {
                        for &master in &[100usize, 127] {
                            let user_decay = DECIBEL_SQUARE_TABLE[volume]
                                + DECIBEL_SQUARE_TABLE[expression]
                                + DECIBEL_SQUARE_TABLE[master];
                            let want = pokediamond_channel_volume(velocity, env_att, user_decay);
                            let got = calc_channel_volume(velocity, env_att, user_decay);
                            assert!(
                                (got - want).abs() < 1e-12,
                                "vel={velocity} env={env_att} vol={volume} expr={expression} \
                                 master={master}: got {got}, want {want}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn track_volume_attenuates_in_decibel_domain() {
        // Full velocity + envelope, volume swept down. Track volume must reduce the channel
        // volume through the decibel table — not as a linear `volume/127` multiply (the old,
        // pokediamond-inaccurate behavior).
        let full = calc_channel_volume(127, 0, DECIBEL_SQUARE_TABLE[127] * 3);
        let half = calc_channel_volume(
            127,
            0,
            DECIBEL_SQUARE_TABLE[64] + DECIBEL_SQUARE_TABLE[127] * 2,
        );
        let quiet = calc_channel_volume(
            127,
            0,
            DECIBEL_SQUARE_TABLE[16] + DECIBEL_SQUARE_TABLE[127] * 2,
        );
        assert!(
            half < full && quiet < half,
            "volume must attenuate monotonically"
        );
        // The decibel curve is steeper near the bottom than a linear law: volume 64 sits well
        // below half amplitude (a linear `64/127` would give ≈0.50).
        assert!(
            half < 0.5 * full,
            "dB volume 64 should be quieter than a 0.5 linear multiply"
        );
        // Expression and master fold in identically (a 0 dB / 127 term is a no-op).
        let only_expr = calc_channel_volume(127, 0, DECIBEL_SQUARE_TABLE[64]);
        assert!((only_expr - half).abs() < 1e-12);
    }
}

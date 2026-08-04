//! The DS channel volume calculation, in the decibel domain the driver works in.

use super::tables::{DECIBEL_SQUARE_TABLE, GET_VOL_TABLE};

#[inline]
pub(super) fn decibel_db(level: i32) -> i32 {
    DECIBEL_SQUARE_TABLE[level.clamp(0, 127) as usize]
}

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

    fn pokediamond_channel_volume(velocity: i32, env_att: i32, user_decay: i32) -> f64 {
        let mut value =
            DECIBEL_SQUARE_TABLE[velocity.clamp(0, 127) as usize] + (env_att >> 7) + user_decay;
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
        assert!(
            half < 0.5 * full,
            "dB volume 64 should be quieter than a 0.5 linear multiply"
        );
        let only_expr = calc_channel_volume(127, 0, DECIBEL_SQUARE_TABLE[64]);
        assert!((only_expr - half).abs() < 1e-12);
    }
}

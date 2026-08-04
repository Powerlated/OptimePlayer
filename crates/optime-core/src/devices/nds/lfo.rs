//! The DS channel LFO tick, transcribed from the sound driver.

use super::tables::snd_sin_idx;

pub(super) mod lfo_type {
    pub const PITCH: i32 = 0;
    pub const VOLUME: i32 = 1;
    pub const PAN: i32 = 2;
}

#[derive(Debug)]
pub(super) struct LfoParams {
    pub depth: i32,
    pub delay: i32,
    pub lfo_type: i32,
    pub speed: i32,
    pub range: i32,
}

pub(super) fn lfo_tick(p: &LfoParams, counter: &mut i32, delay_counter: &mut i32) -> i64 {
    let mut value: i64 = if p.depth == 0 || *delay_counter < p.delay {
        0
    } else {
        i64::from(snd_sin_idx(*counter >> 8)) * i64::from(p.depth) * i64::from(p.range)
    };

    if value != 0 {
        match p.lfo_type {
            lfo_type::VOLUME => value *= 60,
            lfo_type::PITCH | lfo_type::PAN => value <<= 6,
            _ => {}
        }
        value >>= 14;
    }

    if *delay_counter < p.delay {
        *delay_counter += 1;
    } else {
        let mut tmp = *counter;
        tmp += p.speed << 6;
        tmp >>= 8;
        while tmp >= 0x80 {
            tmp -= 0x80;
        }
        *counter += p.speed << 6;
        *counter &= 0xFF;
        *counter |= tmp << 8;
    }

    value
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pokediamond_lfo(
        p: &LfoParams,
        counter: &mut i32,
        delay_counter: &mut i32,
        target_scale: bool,
    ) -> i64 {
        let mut value: i64 = if p.depth == 0 || *delay_counter < p.delay {
            0
        } else {
            i64::from(snd_sin_idx((*counter as u32 >> 8) as i32))
                * i64::from(p.depth)
                * i64::from(p.range)
        };
        if target_scale && value != 0 {
            match p.lfo_type {
                lfo_type::VOLUME => value *= 60,
                lfo_type::PITCH | lfo_type::PAN => value <<= 6,
                _ => {}
            }
            value >>= 14;
        }
        if *delay_counter < p.delay {
            *delay_counter += 1;
        } else {
            let mut tmp = *counter;
            tmp += p.speed << 6;
            tmp >>= 8;
            while tmp >= 0x80 {
                tmp -= 0x80;
            }
            *counter += p.speed << 6;
            *counter &= 0xFF;
            *counter |= tmp << 8;
        }
        value
    }

    #[test]
    fn lfo_tick_matches_pokediamond_reference() {
        for &lfo_type in &[lfo_type::VOLUME, lfo_type::PITCH, lfo_type::PAN] {
            for &depth in &[0, 1, 64, 127] {
                for &delay in &[0, 1, 5] {
                    for &speed in &[1, 16, 64] {
                        let p = LfoParams {
                            depth,
                            delay,
                            lfo_type,
                            speed,
                            range: 1,
                        };
                        let (mut c1, mut d1) = (0i32, 0i32);
                        let (mut c2, mut d2) = (0i32, 0i32);
                        for _ in 0..40 {
                            let got = lfo_tick(&p, &mut c1, &mut d1);
                            let want = pokediamond_lfo(&p, &mut c2, &mut d2, true);
                            assert_eq!(got, want, "value mismatch ({p:?})");
                            assert_eq!(c1, c2, "phase counter mismatch");
                            assert_eq!(d1, d2, "delay counter mismatch");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn delayed_lfo_engages_after_delay() {
        let p = LfoParams {
            depth: 127,
            delay: 4,
            lfo_type: lfo_type::PITCH,
            speed: 16,
            range: 1,
        };
        let (mut counter, mut delay_counter) = (0i32, 0i32);
        for tick in 0..p.delay {
            let v = lfo_tick(&p, &mut counter, &mut delay_counter);
            assert_eq!(v, 0, "tick {tick}: value must be 0 during the delay window");
            assert_eq!(
                counter, 0,
                "tick {tick}: phase must stay frozen during the delay"
            );
            assert_eq!(
                delay_counter,
                tick + 1,
                "tick {tick}: delay counter must count up"
            );
        }
        assert_eq!(delay_counter, p.delay);
        lfo_tick(&p, &mut counter, &mut delay_counter);
        assert_ne!(counter, 0, "phase must advance once the delay has elapsed");
        let mut saw_nonzero = false;
        for _ in 0..64 {
            if lfo_tick(&p, &mut counter, &mut delay_counter) != 0 {
                saw_nonzero = true;
                break;
            }
        }
        assert!(saw_nonzero, "delayed LFO never produced a non-zero value");
    }

    #[test]
    fn zero_depth_lfo_is_always_silent() {
        let p = LfoParams {
            depth: 0,
            delay: 0,
            lfo_type: lfo_type::PITCH,
            speed: 16,
            range: 1,
        };
        let (mut counter, mut delay_counter) = (0i32, 0i32);
        for _ in 0..32 {
            assert_eq!(lfo_tick(&p, &mut counter, &mut delay_counter), 0);
        }
    }
}

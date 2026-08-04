use super::envelope::USEC_PER_DRIVER_TICK;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LfoDest {
    Pitch,
    Volume,
    Pan,
}

impl LfoDest {
    fn from_code(code: u8) -> Option<LfoDest> {
        match code {
            1 => Some(LfoDest::Pitch),
            2 => Some(LfoDest::Volume),
            3 => Some(LfoDest::Pan),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LfoConfig {
    pub enabled: u8,
    pub dest: u8,
    pub waveform: u8,
    pub depth: i16,
    pub period: u16,
    pub delay: u16,
    pub fade: u16,
}

#[derive(Debug, Clone)]
pub struct LfoRng {
    state: u32,
}

impl Default for LfoRng {
    fn default() -> Self {
        LfoRng { state: 0x2A2A_2A2A }
    }
}

impl LfoRng {
    fn next(&mut self) -> i32 {
        let mut x = self.state;
        x ^= x << 17;
        x ^= ((x as i32) >> 15) as u32;
        self.state = x;
        (x & 0x7FFF) as i32
    }
}

#[derive(Debug, Clone)]
pub struct Lfo {
    pub dest: LfoDest,
    waveform: u8,
    phase_flags: u8,
    ticks_per_phase_change: u16,
    ticks_until_phase_change: u16,
    current_output: i32,
    amplitude: i32,
    output_delta: i32,
    ticks_until_started: u16,
    envelope_ticks_left: u16,
    envelope_level: i32,
    envelope_delta: i32,
}

const ENV_FULL: i32 = 0x0100_0000;

impl Lfo {
    pub fn build(cfg: &LfoConfig, const_level: i32) -> Option<Lfo> {
        if cfg.enabled == 0 {
            return None;
        }
        let dest = LfoDest::from_code(cfg.dest)?;
        if cfg.period == 0 {
            return None;
        }
        let tick_us = USEC_PER_DRIVER_TICK;
        let to_ticks = |ms: u16| -> i64 { i64::from(ms) * 1000 / tick_us };

        let amplitude = i32::from(cfg.depth) << 10;
        let ticks_per_phase_change = to_ticks(cfg.period).max(1) as u16;
        let output_delta = amplitude / i32::from(ticks_per_phase_change);

        let (ticks_until_started, envelope_ticks_left, envelope_level, envelope_delta) =
            if cfg.enabled == 1 {
                let delay_ticks = to_ticks(cfg.delay) as u16;
                let fade_ticks = to_ticks(cfg.fade) as u16;
                if fade_ticks != 0 {
                    (delay_ticks, fade_ticks, 0, ENV_FULL / i32::from(fade_ticks))
                } else {
                    (delay_ticks, 0, ENV_FULL, 0)
                }
            } else {
                (0, 0, ((const_level << 8) / 127) << 16, 0)
            };

        Some(Lfo {
            dest,
            waveform: cfg.waveform,
            phase_flags: 0,
            ticks_per_phase_change,
            ticks_until_phase_change: 0,
            current_output: 0,
            amplitude,
            output_delta,
            ticks_until_started,
            envelope_ticks_left,
            envelope_level,
            envelope_delta,
        })
    }

    pub fn tick(&mut self, rng: &mut LfoRng) -> i32 {
        if self.ticks_until_started != 0 {
            self.ticks_until_started -= 1;
            return 0;
        }
        let output = self.waveform_step(rng) >> 8;

        if self.envelope_ticks_left != 0 {
            self.envelope_ticks_left -= 1;
            if self.envelope_ticks_left != 0 {
                self.envelope_level = self.envelope_level.wrapping_add(self.envelope_delta);
            } else {
                self.envelope_level = ENV_FULL;
            }
        }

        let env = (self.envelope_level as u32) >> 8;
        output.wrapping_mul(env as i32) >> 16
    }

    fn waveform_step(&mut self, rng: &mut LfoRng) -> i32 {
        match self.waveform {
            0 => self.half_square(),
            1 => self.full_square(),
            2 => self.half_triangle(),
            3 => self.full_triangle(),
            4 => self.saw(),
            5 => self.reverse_saw(),
            6 => self.half_noise(rng),
            7 => self.full_noise(rng),
            _ => 0,
        }
    }

    fn half_square(&mut self) -> i32 {
        if self.ticks_until_phase_change == 0 {
            self.ticks_until_phase_change = self.ticks_per_phase_change;
            self.current_output = if self.current_output != 0 {
                0
            } else {
                self.amplitude
            };
        }
        self.ticks_until_phase_change -= 1;
        self.current_output
    }

    fn full_square(&mut self) -> i32 {
        if self.ticks_until_phase_change == 0 {
            self.ticks_until_phase_change = self.ticks_per_phase_change;
            self.current_output = if self.phase_flags & 2 != 0 {
                -self.amplitude
            } else {
                self.amplitude
            };
            self.phase_flags ^= 2;
        }
        self.ticks_until_phase_change -= 1;
        self.current_output
    }

    fn half_triangle(&mut self) -> i32 {
        if self.ticks_until_phase_change == 0 {
            let flags = self.phase_flags;
            self.ticks_until_phase_change = self.ticks_per_phase_change;
            if flags & 1 != 0 {
                self.output_delta = -self.output_delta;
            }
            self.phase_flags = flags | 1;
        }
        self.ticks_until_phase_change -= 1;
        self.current_output = self.current_output.wrapping_add(self.output_delta);
        self.current_output
    }

    fn full_triangle(&mut self) -> i32 {
        let mut ticks_left = self.ticks_until_phase_change;
        if ticks_left == 0 {
            ticks_left = self.ticks_per_phase_change;
            let flags = self.phase_flags;
            if flags & 1 != 0 {
                self.output_delta = -self.output_delta;
            } else {
                ticks_left /= 2;
                if ticks_left < 1 {
                    ticks_left = 1;
                }
            }
            self.phase_flags = flags | 1;
        }
        self.ticks_until_phase_change = ticks_left - 1;
        self.current_output = self.current_output.wrapping_add(self.output_delta);
        self.current_output
    }

    fn saw(&mut self) -> i32 {
        if self.ticks_until_phase_change == 0 {
            self.ticks_until_phase_change = self.ticks_per_phase_change;
            self.current_output = 0;
        } else {
            self.current_output = self.current_output.wrapping_add(self.output_delta);
        }
        self.ticks_until_phase_change -= 1;
        self.current_output
    }

    fn reverse_saw(&mut self) -> i32 {
        if self.ticks_until_phase_change == 0 {
            self.ticks_until_phase_change = self.ticks_per_phase_change;
            self.current_output = self.amplitude;
        } else {
            self.current_output = self.current_output.wrapping_sub(self.output_delta);
        }
        self.ticks_until_phase_change -= 1;
        self.current_output
    }

    fn half_noise(&mut self, rng: &mut LfoRng) -> i32 {
        if self.ticks_until_phase_change == 0 {
            self.ticks_until_phase_change = self.ticks_per_phase_change;
            self.current_output = (self.amplitude >> 16).wrapping_mul(rng.next());
        }
        self.ticks_until_phase_change -= 1;
        self.current_output
    }

    fn full_noise(&mut self, rng: &mut LfoRng) -> i32 {
        if self.ticks_until_phase_change == 0 {
            self.ticks_until_phase_change = self.ticks_per_phase_change;
            let amplitude = self.amplitude;
            self.current_output = (amplitude >> 15).wrapping_mul(rng.next()) - (amplitude >> 1);
        }
        self.ticks_until_phase_change -= 1;
        self.current_output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vibrato_cfg() -> LfoConfig {
        LfoConfig {
            enabled: 1,
            dest: 1,
            waveform: 3,
            depth: 0x40,
            period: 200,
            delay: 0,
            fade: 0,
        }
    }

    #[test]
    fn disabled_or_zero_period_builds_nothing() {
        assert!(Lfo::build(&LfoConfig::default(), 127).is_none());
        let mut c = vibrato_cfg();
        c.period = 0;
        assert!(Lfo::build(&c, 127).is_none());
    }

    #[test]
    fn build_sets_amplitude_and_phase_ticks() {
        let lfo = Lfo::build(&vibrato_cfg(), 127).unwrap();
        assert_eq!(lfo.dest, LfoDest::Pitch);
        assert_eq!(lfo.amplitude, 0x40 << 10);
        assert_eq!(lfo.ticks_per_phase_change, 20);
        assert_eq!(lfo.envelope_level, ENV_FULL);
    }

    #[test]
    fn triangle_oscillates_around_zero() {
        let mut lfo = Lfo::build(&vibrato_cfg(), 127).unwrap();
        let mut rng = LfoRng::default();
        let mut min = i32::MAX;
        let mut max = i32::MIN;
        for _ in 0..160 {
            let out = lfo.tick(&mut rng);
            min = min.min(out);
            max = max.max(out);
        }
        assert!(max > 0, "should swing positive, got max {max}");
        assert!(min < 0, "should swing negative, got min {min}");
        assert!(max <= (0x40 << 2) + 8, "peak {max} too large");
    }

    #[test]
    fn fade_in_ramps_from_silence_to_full() {
        let mut cfg = vibrato_cfg();
        cfg.fade = 1000;
        cfg.waveform = 1;
        let mut lfo = Lfo::build(&cfg, 127).unwrap();
        let mut rng = LfoRng::default();
        let first = lfo.tick(&mut rng).abs();
        let mut last = first;
        for _ in 0..130 {
            last = lfo.tick(&mut rng).abs();
        }
        assert!(first < last, "fade-in should grow: {first} -> {last}");
    }

    #[test]
    fn start_delay_holds_output_at_zero() {
        let mut cfg = vibrato_cfg();
        cfg.delay = 300;
        cfg.waveform = 1;
        let mut lfo = Lfo::build(&cfg, 127).unwrap();
        let mut rng = LfoRng::default();
        for _ in 0..29 {
            assert_eq!(
                lfo.tick(&mut rng),
                0,
                "must stay silent during the start delay"
            );
        }
    }
}

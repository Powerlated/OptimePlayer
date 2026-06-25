//! DSE channel LFOs — vibrato (pitch), tremolo (volume), and auto-pan — transcribed from the
//! `pret/pmd-sky` `dc_lfo` engine:
//!
//! - [`Lfo::build`] is `SoundLfoBank_Set` (`lib/DSE/asm/dc_lfo_1.s`): amplitude = `depth << 10`,
//!   `ticks_per_phase = max(1, period_ms * 1000 / tick_us)`, `output_delta = amplitude / ticks`,
//!   the fade-in envelope (`delay`/`fade` in ms), and the waveform pick.
//! - [`Lfo::tick`] + the eight waveform steps are `SoundLfoBank_Tick` and the `SoundLfoWave_*`
//!   functions (`lib/DSE/src/dc_lfo_2.c`), reproduced exactly (phase counters, `>> 8` output,
//!   `(output * (envelope_level >> 8)) >> 16` scaling).
//! - The output is applied by `DseVoice_UpdateParameters` (`main_02071EB4.s`): the pitch LFO is
//!   added straight to the 8.8-fixed `note_key`; the volume and pan LFOs are added as `>> 6` to the
//!   0..=127 note-volume / pan index. Routing flags come from `LFO_OUTPUT_VOICE_UPDATE_FLAGS`.
//!
//! In the game each *voice* builds its own bank from the channel's pending config at note-on, so
//! vibrato/tremolo restart their fade-in per note. We mirror that for pitch/volume (per-voice) and
//! run a single auto-pan LFO per track (pan is a track property in the synth layer).

use super::envelope::USEC_PER_DRIVER_TICK;

/// Where an LFO's output is routed (config byte +2). Matches the dest → voice-update-flag map in
/// `LFO_OUTPUT_VOICE_UPDATE_FLAGS` (1 = pitch, 2 = volume, 3 = pan).
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

/// One LFO's pending configuration, written by the SMD `Setup*Lfo` / `SetupLfoEnvelope` /
/// `Use*Lfo` / `SetLfoParameter` opcodes (the `DseTrackEvent_*` handlers). Built into a live
/// [`Lfo`] at note-on by [`Lfo::build`]. All four channel LFO slots live at `channel + 0x74`
/// (0x10 bytes each); the dedicated key-bend/volume/pan opcodes target slots 0/1/2.
#[derive(Clone, Copy, Debug, Default)]
pub struct LfoConfig {
    /// `0` = disabled, `1` = USE_ENVELOPE (fade-in), `3` = CONST_ENVELOPE (config +1).
    pub enabled: u8,
    /// Output destination code (config +2): 1 = pitch, 2 = volume, 3 = pan.
    pub dest: u8,
    /// Waveform index into `LFO_WAVEFORM_CALLBACKS` (config +3).
    pub waveform: u8,
    /// Depth, sign-extended (config +4): the amplitude is `depth << 10`.
    pub depth: i16,
    /// Oscillation period in ms (config +8): `ticks_per_phase = max(1, period*1000/tick_us)`.
    pub period: u16,
    /// Envelope fade-in delay in ms (config +0xA).
    pub delay: u16,
    /// Envelope fade-in duration in ms (config +0xC).
    pub fade: u16,
}

/// A deterministic copy of `DseUtil_GetRandomNumber` (`main_0206A878.s`): a 32-bit xorshift whose
/// low 15 bits feed the noise waveforms. Seeded to a fixed nonzero value so playback is
/// reproducible (the game seeds `DRIVER_WORK+0x34` once at init).
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
    /// `DseUtil_GetRandomNumber`: `x ^= x<<17; x ^= (s32)x >> 15;` then return `x & 0x7FFF`.
    fn next(&mut self) -> i32 {
        let mut x = self.state;
        x ^= x << 17;
        x ^= ((x as i32) >> 15) as u32;
        self.state = x;
        (x & 0x7FFF) as i32
    }
}

/// One live LFO (a `dse_lfo`).
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

/// Fixed-point "full" LFO envelope level (`0x1000000`); `(level >> 8)` is the unity multiplier.
const ENV_FULL: i32 = 0x0100_0000;

impl Lfo {
    /// Builds a live LFO from a channel config slot, or `None` if the slot is disabled or
    /// degenerate (`SoundLfoBank_Set`). `const_level` (0..=127) is the forced level for a
    /// CONST_ENVELOPE LFO; it is irrelevant to the usual fade-in (`USE_ENVELOPE`) LFOs.
    pub fn build(cfg: &LfoConfig, const_level: i32) -> Option<Lfo> {
        if cfg.enabled == 0 {
            return None;
        }
        let dest = LfoDest::from_code(cfg.dest)?;
        // A zero period would leave the amplitude uninitialised in the ROM; real data never does
        // this, so treat it as "no LFO".
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
                // USE_ENVELOPE: fade in from 0 to full over `fade` ms after a `delay` ms wait.
                let delay_ticks = to_ticks(cfg.delay) as u16;
                let fade_ticks = to_ticks(cfg.fade) as u16;
                if fade_ticks != 0 {
                    (delay_ticks, fade_ticks, 0, ENV_FULL / i32::from(fade_ticks))
                } else {
                    (delay_ticks, 0, ENV_FULL, 0)
                }
            } else {
                // CONST_ENVELOPE: a fixed level, no fade-in (SoundLfoBank_SetConstEnvelopes).
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

    /// Advances the LFO one driver tick and returns its signed output contribution (the value
    /// summed into the destination's accumulator). `0` while still inside the start delay
    /// (`SoundLfoBank_Tick`).
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

        // (output * (envelope_level >> 8)) >> 16, matching the ROM's 32-bit truncating multiply.
        let env = (self.envelope_level as u32) >> 8;
        output.wrapping_mul(env as i32) >> 16
    }

    /// One step of the selected waveform, returning `current_output` (the eight `SoundLfoWave_*`
    /// functions). Index 8+ is `Invalid` (silent).
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
        // A full-triangle pitch vibrato: depth 0x40, 200 ms period, no fade-in.
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
        // period 200 ms / 10 ms per tick = 20 ticks per phase.
        assert_eq!(lfo.ticks_per_phase_change, 20);
        // No fade-in: envelope already full.
        assert_eq!(lfo.envelope_level, ENV_FULL);
    }

    #[test]
    fn triangle_oscillates_around_zero() {
        let mut lfo = Lfo::build(&vibrato_cfg(), 127).unwrap();
        let mut rng = LfoRng::default();
        let mut min = i32::MAX;
        let mut max = i32::MIN;
        // Two full periods (4 * 20 ticks) covers a full up/down swing.
        for _ in 0..160 {
            let out = lfo.tick(&mut rng);
            min = min.min(out);
            max = max.max(out);
        }
        // A symmetric vibrato: swings both above and below the centre by a similar amount.
        assert!(max > 0, "should swing positive, got max {max}");
        assert!(min < 0, "should swing negative, got min {min}");
        // Peak magnitude is the amplitude scaled by the tick's `>> 8` (≈ depth << 2).
        assert!(max <= (0x40 << 2) + 8, "peak {max} too large");
    }

    #[test]
    fn fade_in_ramps_from_silence_to_full() {
        let mut cfg = vibrato_cfg();
        cfg.fade = 1000; // 1000 ms / 10 = 100 ticks of fade-in
        cfg.waveform = 1; // full square: constant ±amplitude so the envelope is visible
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
        cfg.delay = 300; // 30 ticks of delay
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

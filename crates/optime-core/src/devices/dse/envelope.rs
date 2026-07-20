//! The DSE volume envelope: a faithful transcription of `dc_envelope.c` from `pret/pmd-sky`.
//!
//! DSE envelopes are piecewise-linear *slides* in a 23-bit fixed-point volume domain
//! (`current_volume`); the output level is `current_volume >> 23` (0..=127). A note moves
//! ATTACK → HOLD → DECAY → SUSTAIN → DONE, and RELEASE → RELEASE_END once the note is released.
//! Each phase's length comes from a 0..=127 parameter indexed into one of two duration tables and
//! scaled to driver ticks by the driver's microseconds-per-tick.

/// Microseconds per sound-driver tick: the driver's ~100 Hz alarm
/// (`Snd_SetupAlarm(0x1474)` in `DseDriver_StartTickTimer`, i.e. `64 * 5236` cycles of the
/// 33.51 MHz clock ≈ 10 ms). Phase durations are `table_value * 1000 / this`.
pub const USEC_PER_DRIVER_TICK: i64 = 10_000;

// Envelope states (`dc_envelope.h`); state 1 (CONST, a forced constant volume) is unused here.
const OFF: u8 = 0;
const DONE: u8 = 2;
const ATTACK: u8 = 3;
const HOLD: u8 = 4;
const DECAY: u8 = 5;
const SUSTAIN: u8 = 6;
const RELEASE: u8 = 7;
const RELEASE_END: u8 = 8;

/// `MUSIC_DURATION_LOOKUP_TABLE_1` (`u16[128]`), used when `slide_time_multiplier != 0`.
#[rustfmt::skip]
const DURATION_TABLE_1: [u16; 128] = [
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31,
    32, 35, 40, 45, 51, 57, 64, 72,
    80, 88, 98, 109, 120, 131, 144, 158,
    172, 188, 204, 222, 240, 260, 281, 303,
    327, 352, 378, 406, 435, 466, 498, 532,
    568, 606, 645, 686, 729, 775, 822, 871,
    923, 977, 1030, 1090, 1150, 1220, 1280, 1350,
    1420, 1570, 1650, 1740, 1820, 1910, 2010, 2100,
    2200, 2310, 2410, 2520, 2640, 2750, 2880, 3000,
    3130, 3260, 3400, 3550, 3690, 3840, 4000, 4160,
    4330, 4500, 4670, 4850, 5040, 5230, 5430, 5630,
    5840, 6050, 6270, 6490, 6720, 6960, 7200, 7450,
    7710, 7970, 8240, 8520, 8800, 9090, 10000, 32767,
];

/// `MUSIC_DURATION_LOOKUP_TABLE_2` (`u32[128]`), used when `slide_time_multiplier == 0`.
#[rustfmt::skip]
const DURATION_TABLE_2: [u32; 128] = [
    0, 4, 7, 10, 15, 21, 28, 36,
    46, 58, 72, 87, 104, 123, 145, 168,
    389, 446, 508, 575, 648, 726, 810, 901,
    997, 1100, 1210, 1326, 1449, 1580, 1717, 1862,
    3023, 3264, 3517, 3782, 4060, 4351, 4655, 4972,
    5302, 5647, 6005, 6378, 6765, 7167, 7584, 8017,
    11286, 11904, 12544, 13205, 13889, 14594, 15323, 16074,
    16848, 17646, 18468, 19315, 20185, 21081, 22002, 22948,
    29900, 31147, 32428, 33742, 35089, 36471, 37887, 39338,
    40824, 42346, 43904, 45499, 47130, 48798, 50503, 52247,
    64834, 67019, 69250, 71528, 73854, 76228, 78651, 81122,
    83643, 86213, 88834, 91506, 94229, 97003, 99829, 102707,
    123245, 126727, 130272, 133879, 137551, 141286, 145086, 148951,
    152882, 156879, 160942, 165072, 169270, 173536, 177870, 182274,
    213424, 218616, 223888, 229241, 234676, 240193, 245793, 251476,
    257242, 263093, 269029, 275050, 281157, 287351, 293632, 2147483647,
];

/// The seven envelope parameters that matter, unpacked from a split's 16-byte
/// `sound_envelope_parameters` block (`use_envelope`@0, `slide_time_multiplier`@1, then the
/// time/level bytes at @0x8..0xF).
#[derive(Debug, Clone, Copy)]
pub struct EnvelopeParams {
    pub use_envelope: bool,
    pub slide_time_multiplier: u8,
    pub attack_begin: u8,
    pub attack_time: u8,
    pub decay_time: u8,
    pub sustain_level: u8,
    pub hold_time: u8,
    pub sustain_time: u8,
    pub release_time: u8,
}

impl EnvelopeParams {
    /// Reads the params from a split's raw 16-byte envelope block.
    pub fn from_block(b: &[u8; 16]) -> EnvelopeParams {
        EnvelopeParams {
            use_envelope: b[0] != 0,
            slide_time_multiplier: b[1],
            attack_begin: b[0x8],
            attack_time: b[0x9],
            decay_time: b[0xA],
            sustain_level: b[0xB],
            hold_time: b[0xC],
            sustain_time: b[0xD],
            release_time: b[0xE],
        }
    }
}

/// One note's running envelope. Build with [`SoundEnvelope::start`], advance once per driver tick
/// with [`SoundEnvelope::tick`] (which returns the 0..=127 level), and begin the tail with
/// [`SoundEnvelope::release`].
#[derive(Debug, Clone)]
pub struct SoundEnvelope {
    params: EnvelopeParams,
    current_volume: i32,
    volume_delta: i32,
    ticks_left: i32,
    state: u8,
    target_volume: u8,
}

impl SoundEnvelope {
    /// Starts the envelope for a new note (`UpdateTrackVolumeEnvelopes`).
    pub fn start(params: EnvelopeParams) -> SoundEnvelope {
        let mut e = SoundEnvelope {
            params,
            current_volume: 0,
            volume_delta: 0,
            ticks_left: 0,
            state: OFF,
            target_volume: 0,
        };
        if params.use_envelope {
            if params.attack_time != 0 {
                e.current_volume = (params.attack_begin as i32) << 23;
                e.state = ATTACK;
                e.set_slide(0x7f, params.attack_time as i32);
            } else {
                e.current_volume = 0x3f80_0000;
                if params.hold_time != 0 {
                    e.set_slide(0x7f, params.hold_time as i32);
                    e.state = HOLD;
                } else if params.decay_time != 0 {
                    e.set_slide(params.sustain_level as i8 as i32, params.decay_time as i32);
                    e.state = DECAY;
                } else {
                    e.set_slide(0, params.sustain_time as i32);
                    e.state = SUSTAIN;
                }
            }
        } else {
            // No envelope: hold full volume until released.
            e.state = OFF;
            e.current_volume = 0x3f80_0000;
        }
        e
    }

    /// Whether this envelope uses the slide model (vs. a constant full-volume note).
    pub fn uses_envelope(&self) -> bool {
        self.params.use_envelope
    }

    /// Whether the note has fully finished its release tail and the voice can be freed.
    pub fn is_finished(&self) -> bool {
        self.state == RELEASE_END
    }

    /// Begins the release phase (`SoundEnvelope_Release`). A no-op for a non-envelope note.
    pub fn release(&mut self) {
        if self.state == OFF {
            return;
        }
        self.set_slide(0, self.params.release_time as i32);
        self.state = RELEASE;
    }

    /// `SoundEnvelope_SetSlide`: schedule a linear slide of `current_volume` to
    /// `target_volume << 23` over the ticks given by `msec_tab_index`.
    fn set_slide(&mut self, target_volume: i32, msec_tab_index: i32) {
        if msec_tab_index == 0x7f {
            self.volume_delta = 0;
            self.ticks_left = 0x7fff_ffff;
            return;
        }
        self.target_volume = target_volume as u8;
        let idx = msec_tab_index.clamp(0, 127) as usize;
        self.ticks_left = if self.params.slide_time_multiplier == 0 {
            (DURATION_TABLE_2[idx] as i64 * 1000 / USEC_PER_DRIVER_TICK) as i32
        } else {
            (self.params.slide_time_multiplier as i64 * DURATION_TABLE_1[idx] as i64 * 1000
                / USEC_PER_DRIVER_TICK) as i32
        };
        self.volume_delta = if self.ticks_left != 0 {
            ((target_volume << 23) - self.current_volume) / self.ticks_left
        } else {
            0
        };
    }

    /// `SoundEnvelope_Tick`: advance one driver tick, returning the 0..=127 output level.
    pub fn tick(&mut self) -> i8 {
        if self.state > DONE {
            if self.ticks_left == 0 {
                self.current_volume = (self.target_volume as i32) << 23;
                self.advance_phase();
            } else {
                let next = (self.current_volume + self.volume_delta).clamp(0, 0x3fff_ffff);
                self.ticks_left -= 1;
                self.current_volume = next;
            }
        }
        (self.current_volume >> 23) as i8
    }

    /// The state-machine transition when a phase's `ticks_left` hits zero. This mirrors the
    /// fall-through `switch` in `SoundEnvelope_Tick` exactly.
    fn advance_phase(&mut self) {
        match self.state {
            ATTACK => {
                if self.params.hold_time != 0 {
                    self.set_slide(0x7f, self.params.hold_time as i32);
                    self.state = HOLD;
                    return;
                }
                self.fall_through_hold();
            }
            HOLD => self.fall_through_hold(),
            DECAY => self.fall_through_decay(),
            SUSTAIN => {
                self.set_slide(0, 0);
                self.state = DONE;
            }
            RELEASE => {
                self.state = RELEASE_END;
                self.current_volume = 0;
                self.ticks_left = 0;
            }
            _ => {}
        }
    }

    /// HOLD-phase fall-through: into DECAY (setting the sustain floor), else SUSTAIN, else DONE.
    fn fall_through_hold(&mut self) {
        if self.params.decay_time != 0 {
            self.set_slide(
                self.params.sustain_level as i8 as i32,
                self.params.decay_time as i32,
            );
            self.state = DECAY;
            return;
        }
        self.current_volume = (self.params.sustain_level as i32) << 23;
        self.fall_through_decay();
    }

    /// DECAY-phase fall-through: into SUSTAIN, else DONE.
    fn fall_through_decay(&mut self) {
        if self.params.sustain_time != 0 {
            self.set_slide(0, self.params.sustain_time as i32);
            self.state = SUSTAIN;
            return;
        }
        self.set_slide(0, 0);
        self.state = DONE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(b: [u8; 16]) -> EnvelopeParams {
        EnvelopeParams::from_block(&b)
    }

    #[test]
    fn from_block_unpacks_fields() {
        // use=1, mult=1, …, attack_begin=0, attack=0, decay=0x3c, sustain=0x69, hold=0,
        // decay2=0x46, release=0x4b (program 12's split in bgm0001).
        let p = params([
            0x01, 0x01, 0x01, 0x03, 0x03, 0xff, 0xff, 0xff, 0x00, 0x00, 0x3c, 0x69, 0x00, 0x46,
            0x4b, 0xff,
        ]);
        assert!(p.use_envelope);
        assert_eq!(p.slide_time_multiplier, 1);
        assert_eq!(p.decay_time, 0x3c);
        assert_eq!(p.sustain_level, 0x69);
        assert_eq!(p.sustain_time, 0x46);
        assert_eq!(p.release_time, 0x4b);
    }

    #[test]
    fn attack_rises_then_decays_to_sustain() {
        // attack_time=40, decay_time=40, sustain_level=64, others 0; mult=1.
        let mut env =
            SoundEnvelope::start(params([1, 1, 0, 0, 0, 0, 0, 0, 0, 40, 40, 64, 0, 0, 60, 0]));
        let first = env.tick();
        // Rises during attack.
        let mut peak = first;
        for _ in 0..2000 {
            peak = peak.max(env.tick());
        }
        assert!(
            peak >= 120,
            "attack should approach full level, peaked at {peak}"
        );
        // After enough ticks, decays toward the sustain level (64).
        let mut last = 127;
        for _ in 0..5000 {
            last = env.tick();
        }
        assert!(
            (60..=70).contains(&last),
            "should settle near sustain level 64, got {last}"
        );
    }

    #[test]
    fn release_falls_to_zero() {
        let mut env =
            SoundEnvelope::start(params([1, 1, 0, 0, 0, 0, 0, 0, 0, 20, 0, 127, 0, 0, 20, 0]));
        for _ in 0..3000 {
            env.tick();
        }
        env.release();
        let mut last = 127;
        for _ in 0..5000 {
            last = env.tick();
        }
        assert_eq!(last, 0, "release should reach silence");
        assert!(env.is_finished());
    }

    #[test]
    fn non_envelope_holds_full_volume() {
        let mut env = SoundEnvelope::start(params([0; 16]));
        assert!(!env.uses_envelope());
        // current_volume = 0x3f800000 -> level 0x7f.
        assert_eq!(env.tick(), 127);
        assert_eq!(env.tick(), 127);
    }
}

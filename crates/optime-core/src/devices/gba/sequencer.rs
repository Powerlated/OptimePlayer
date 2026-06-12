//! The MP2K sequencer: a transcription of `MPlayMain` and the `ply_*` command handlers from
//! `pret/pokeemerald` (`src/m4a_1.s`, `src/m4a.c`).
//!
//! The sequencer owns only *track* state and the command interpreter; it emits [`Mp2kOp`]s for
//! the player (which owns channels and envelopes) to act on. This keeps it usable standalone
//! for look-ahead visualizers.

use std::sync::Arc;

use super::rom::{ptr_to_offset, SongHeader};
use super::tables::CLOCK_TABLE;
use super::voice::{resolve_tone, ResolvedTone, ToneData};
use crate::util::{read_u32, read_u8};

/// Sequencer steps trigger when the tempo accumulator reaches this (tempo 75 ⇒ one step/frame).
pub(crate) const TEMPO_STEP: u16 = 150;

/// A note-on resolved by the sequencer, for the player to allocate a channel for.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NoteOn {
    /// Gate time in sequencer steps; 0 = tie (sounds until `EndTie`).
    pub gate: u8,
    /// The key as played in the track data (pre-keysplit; `EndTie` matches against this).
    pub midi_key: u8,
    pub velocity: u8,
    /// Note priority: song priority + track priority, saturating.
    pub priority: u8,
    pub tone: ResolvedTone,
}

/// One sequencer operation for the player.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Mp2kOp {
    /// A step began for this track: count down its channels' gate timers.
    GateTick { track: usize },
    /// Start a note.
    Note { track: usize, note: NoteOn },
    /// `EOT`: release the tied channel playing `key`.
    EndTie { track: usize, key: u8 },
    /// `FINE`: the track ended — release all its channels.
    TrackEnded { track: usize },
    /// The track took a backward `GOTO` (the song's loop point).
    Looped,
    /// Every track has ended.
    Finished,
}

/// `MPT_FLG`-equivalent change flags, as honest booleans.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ChangeFlags {
    pub volume: bool,
    pub pitch: bool,
}

/// One MP2K track: registers + interpreter state (`struct MusicPlayerTrack`).
#[derive(Debug, Clone)]
pub(crate) struct Mp2kTrack {
    pub exists: bool,
    pub flags: ChangeFlags,
    wait: u8,
    pattern_level: u8,
    rep_n: u8,
    /// Set by `ply_note` (gate scratch shared with the player).
    pub gate_time: u8,
    pub key: u8,
    pub velocity: u8,
    running_status: u8,
    /// Integer-semitone part of the pitch sum (applied to channel keys).
    pub key_m: i8,
    /// Fractional part of the pitch sum, in 1/256ths of a semitone.
    pub pit_m: u8,
    key_shift: i8,
    tune: i8,
    bend: i8,
    bend_range: u8,
    /// Mixer volumes computed by `TrkVolPitSet`.
    pub vol_mr: u8,
    pub vol_ml: u8,
    vol: u8,
    vol_x: u8,
    pan: i8,
    mod_m: i8,
    mod_depth: u8,
    mod_type: u8,
    lfo_speed: u8,
    lfo_speed_c: u8,
    pub lfo_delay: u8,
    pub lfo_delay_c: u8,
    priority: u8,
    pub echo_volume: u8,
    pub echo_length: u8,
    /// The track's current instrument (copied from the voicegroup by `VOICE`).
    pub tone: ToneData,
    /// `XCMD 0x0D` sample start offset (Pokémon cry progress).
    sample_start: u32,
    /// `XCMD 0x0C` wait timer.
    timer: u16,
    cmd_ptr: usize,
    pattern_stack: [usize; 3],
}

impl Mp2kTrack {
    fn new(cmd_ptr: usize) -> Self {
        // Mirrors MPlayMain's MPT_FLG_START init: zeroed state with these defaults.
        Mp2kTrack {
            exists: true,
            flags: ChangeFlags::default(),
            wait: 0,
            pattern_level: 0,
            rep_n: 0,
            gate_time: 0,
            key: 0,
            velocity: 0,
            running_status: 0,
            key_m: 0,
            pit_m: 0,
            key_shift: 0,
            tune: 0,
            bend: 0,
            bend_range: 2,
            vol_mr: 0,
            vol_ml: 0,
            vol: 0,
            vol_x: 64,
            pan: 0,
            mod_m: 0,
            mod_depth: 0,
            mod_type: 0,
            lfo_speed: 22,
            lfo_speed_c: 0,
            lfo_delay: 0,
            lfo_delay_c: 0,
            priority: 0,
            echo_volume: 0,
            echo_length: 0,
            tone: ToneData {
                kind: 1,
                ..ToneData::default()
            },
            sample_start: 0,
            timer: 0,
            cmd_ptr,
            pattern_stack: [0; 3],
        }
    }

    /// `clear_modM`: resets the LFO output and flags a recompute.
    pub fn clear_mod_m(&mut self) {
        self.mod_m = 0;
        self.lfo_speed_c = 0;
        if self.mod_type == 0 {
            self.flags.pitch = true;
        } else {
            self.flags.volume = true;
        }
    }

    /// `TrkVolPitSet`: folds volume×fade×pan(×LFO) into `vol_mr`/`vol_ml`, and
    /// tune+bend+keyshift(×LFO) into `key_m`/`pit_m`. Consumes the change flags' "set" half;
    /// the player still uses the flags to know which channels to refresh.
    pub fn vol_pit_set(&mut self) {
        if self.flags.volume {
            let mut x = (u32::from(self.vol) * u32::from(self.vol_x)) >> 5;
            if self.mod_type == 1 {
                x = (x * (i32::from(self.mod_m) + 128) as u32) >> 7;
            }
            let mut y = 2 * i32::from(self.pan); // pan_x is unused (no external pan control)
            if self.mod_type == 2 {
                y += i32::from(self.mod_m);
            }
            y = y.clamp(-128, 127);
            self.vol_mr = (((y + 128) as u32 * x) >> 8) as u8;
            self.vol_ml = (((127 - y) as u32 * x) >> 8) as u8;
        }

        if self.flags.pitch {
            let bend = i32::from(self.bend) * i32::from(self.bend_range);
            let mut x = (i32::from(self.tune) + bend) * 4 + (i32::from(self.key_shift) << 8);
            if self.mod_type == 0 {
                x += 16 * i32::from(self.mod_m);
            }
            self.key_m = (x >> 8) as i8;
            self.pit_m = x as u8;
        }
    }

    /// The per-step LFO ("mod") tick: a 0..=255 phase folded into a ±64 triangle, scaled by
    /// `mod_depth`. Sets change flags only when the output byte actually changes.
    fn lfo_step(&mut self) {
        if self.lfo_speed == 0 || self.mod_depth == 0 {
            return;
        }
        if self.lfo_delay_c != 0 {
            self.lfo_delay_c -= 1;
            return;
        }
        self.lfo_speed_c = self.lfo_speed_c.wrapping_add(self.lfo_speed);
        let triangle: i32 = if (self.lfo_speed_c.wrapping_sub(0x40) as i8) < 0 {
            i32::from(self.lfo_speed_c as i8)
        } else {
            0x80 - i32::from(self.lfo_speed_c)
        };
        let value = ((i32::from(self.mod_depth) * triangle) >> 6) as i8;
        if value as u8 != self.mod_m as u8 {
            self.mod_m = value;
            if self.mod_type == 0 {
                self.flags.pitch = true;
            } else {
                self.flags.volume = true;
            }
        }
    }
}

/// The MP2K sequencer for one song: all track interpreters plus the tempo clock.
#[derive(Clone)]
pub(crate) struct Mp2kSequencer {
    rom: Arc<[u8]>,
    pub tracks: Vec<Mp2kTrack>,
    /// Offset of the song's voicegroup.
    voicegroup: usize,
    song_priority: u8,
    tempo_i: u16,
    tempo_c: u16,
    mem_acc: [u8; 16],
    /// Sequencer steps executed (the note timeline).
    pub steps: u32,
    finished: bool,
    finish_reported: bool,
}

impl Mp2kSequencer {
    /// Builds a sequencer over `rom` for the song at `header` (mirrors `MPlayStart`).
    pub fn new(rom: Arc<[u8]>, header: &SongHeader) -> Mp2kSequencer {
        let tracks = (0..header.track_count as usize)
            .filter_map(|i| {
                let ptr = read_u32(&rom, header.offset + 8 + i * 4);
                let offset = ptr_to_offset(ptr, rom.len())?;
                Some(Mp2kTrack::new(offset))
            })
            .collect();
        Mp2kSequencer {
            rom,
            tracks,
            voicegroup: header.voicegroup,
            song_priority: header.priority,
            tempo_i: TEMPO_STEP, // MPlayStart: tempoD = tempoI = 150 (75 BPM-pairs)
            tempo_c: 0,
            mem_acc: [0; 16],
            steps: 0,
            finished: false,
            finish_reported: false,
        }
    }

    /// Current sequencer step rate in steps per *frame* — `tempo_i / 150`.
    pub fn steps_per_frame(&self) -> f64 {
        f64::from(self.tempo_i) / f64::from(TEMPO_STEP)
    }

    /// Advances one frame (VBlank): runs `tempo_i / 150` sequencer steps, appending ops.
    pub fn tick_frame(&mut self, ops: &mut Vec<Mp2kOp>) {
        if self.finished {
            if !self.finish_reported {
                self.finish_reported = true;
                ops.push(Mp2kOp::Finished);
            }
            return;
        }
        self.tempo_c += self.tempo_i;
        while self.tempo_c >= TEMPO_STEP {
            self.tempo_c -= TEMPO_STEP;
            self.steps += 1;
            self.step(ops);
            if self.finished {
                break;
            }
        }
    }

    /// One sequencer step over every track (the `tempoC >= 150` body of `MPlayMain`).
    fn step(&mut self, ops: &mut Vec<Mp2kOp>) {
        let mut any_exists = false;
        for t in 0..self.tracks.len() {
            if !self.tracks[t].exists {
                continue;
            }
            ops.push(Mp2kOp::GateTick { track: t });

            // Run commands until the track owes a wait. The guard stops malformed data
            // (e.g. a restless GOTO loop) from hanging the audio thread.
            let mut guard = 0u32;
            while self.tracks[t].wait == 0 {
                self.execute_command(t, ops);
                guard += 1;
                if !self.tracks[t].exists {
                    break;
                }
                if guard > 100_000 {
                    self.fine(t, ops);
                    break;
                }
            }
            if !self.tracks[t].exists {
                continue;
            }
            any_exists = true;
            self.tracks[t].wait -= 1;
            self.tracks[t].lfo_step();
        }

        if !any_exists {
            self.finished = true;
            ops.push(Mp2kOp::Finished);
            self.finish_reported = true;
        }
    }

    #[inline]
    fn read8(&self, offset: usize) -> u8 {
        read_u8(&self.rom, offset)
    }

    /// Reads the 4-byte ROM pointer at the track's PC and jumps there (`ply_goto`).
    /// Emits [`Mp2kOp::Looped`] for backward jumps — MP2K songs loop with a backward `GOTO`.
    fn goto(&mut self, t: usize, ops: &mut Vec<Mp2kOp>) {
        let track = &mut self.tracks[t];
        let target = read_u32(&self.rom, track.cmd_ptr);
        match ptr_to_offset(target, self.rom.len()) {
            Some(offset) => {
                if offset < track.cmd_ptr {
                    ops.push(Mp2kOp::Looped);
                }
                track.cmd_ptr = offset;
            }
            None => self.fine(t, ops),
        }
    }

    /// `ply_fine`: release the track's channels and stop executing it.
    fn fine(&mut self, t: usize, ops: &mut Vec<Mp2kOp>) {
        self.tracks[t].exists = false;
        ops.push(Mp2kOp::TrackEnded { track: t });
    }

    /// Fetches and executes one command (the dispatch loop of `MPlayMain`).
    fn execute_command(&mut self, t: usize, ops: &mut Vec<Mp2kOp>) {
        let mut cmd = self.read8(self.tracks[t].cmd_ptr);
        if cmd < 0x80 {
            cmd = self.tracks[t].running_status;
        } else {
            self.tracks[t].cmd_ptr += 1;
            if cmd >= 0xBD {
                self.tracks[t].running_status = cmd;
            }
        }

        if cmd >= 0xCF {
            self.ply_note(cmd - 0xCF, t, ops);
        } else if cmd > 0xB0 {
            self.control_command(cmd, t, ops);
        } else if cmd >= 0x80 {
            // W00..W96 rest.
            self.tracks[t].wait = CLOCK_TABLE[(cmd - 0x80) as usize];
        } else {
            // Running status was never set and the byte is data: treat as end of track to
            // avoid runaway execution on malformed data.
            self.fine(t, ops);
        }
    }

    /// `ply_note` (the sequencer half): gate length + optional key/velocity/extension bytes,
    /// then instrument resolution. Channel allocation is the player's job.
    fn ply_note(&mut self, n: u8, t: usize, ops: &mut Vec<Mp2kOp>) {
        self.tracks[t].gate_time = CLOCK_TABLE[n as usize];

        let mut byte = self.read8(self.tracks[t].cmd_ptr);
        if byte < 0x80 {
            self.tracks[t].key = byte;
            self.tracks[t].cmd_ptr += 1;
            byte = self.read8(self.tracks[t].cmd_ptr);
            if byte < 0x80 {
                self.tracks[t].velocity = byte;
                self.tracks[t].cmd_ptr += 1;
                byte = self.read8(self.tracks[t].cmd_ptr);
                if byte < 0x80 {
                    self.tracks[t].gate_time = self.tracks[t].gate_time.wrapping_add(byte);
                    self.tracks[t].cmd_ptr += 1;
                }
            }
        }

        let track = &self.tracks[t];
        let Some(tone) = resolve_tone(&self.rom, &track.tone, track.key) else {
            return;
        };
        let priority = track.priority.saturating_add(self.song_priority);
        ops.push(Mp2kOp::Note {
            track: t,
            note: NoteOn {
                gate: track.gate_time,
                midi_key: track.key,
                velocity: track.velocity,
                priority,
                tone,
            },
        });
    }

    /// The `0xB1..=0xCE` control commands (`gMPlayJumpTable`, with the `MPlayExtender` patches).
    fn control_command(&mut self, cmd: u8, t: usize, ops: &mut Vec<Mp2kOp>) {
        match cmd {
            0xB2 => self.goto(t, ops), // GOTO
            0xB3 => {
                // PATT: call a pattern.
                let level = self.tracks[t].pattern_level as usize;
                if level >= 3 {
                    self.fine(t, ops);
                } else {
                    self.tracks[t].pattern_stack[level] = self.tracks[t].cmd_ptr + 4;
                    self.tracks[t].pattern_level += 1;
                    self.goto(t, ops);
                }
            }
            0xB4 => {
                // PEND: return from a pattern.
                if self.tracks[t].pattern_level > 0 {
                    self.tracks[t].pattern_level -= 1;
                    self.tracks[t].cmd_ptr =
                        self.tracks[t].pattern_stack[self.tracks[t].pattern_level as usize];
                }
            }
            0xB5 => {
                // REPT: repeat a section `count` times (0 = forever).
                let count = self.read8(self.tracks[t].cmd_ptr);
                if count == 0 {
                    self.tracks[t].cmd_ptr += 1;
                    self.goto(t, ops);
                } else {
                    self.tracks[t].rep_n = self.tracks[t].rep_n.wrapping_add(1);
                    let rep_n = self.tracks[t].rep_n;
                    self.tracks[t].cmd_ptr += 1;
                    if rep_n < count {
                        self.goto(t, ops);
                    } else {
                        self.tracks[t].rep_n = 0;
                        self.tracks[t].cmd_ptr += 4;
                    }
                }
            }
            0xB9 => self.memacc(t, ops),
            0xBA => {
                self.tracks[t].priority = self.read_arg(t);
            }
            0xBB => {
                // TEMPO: value is half the step rate.
                let v = u16::from(self.read_arg(t));
                self.tempo_i = v * 2; // tempoI = tempoD × tempoU(0x100) >> 8 = tempoD
            }
            0xBC => {
                self.tracks[t].key_shift = self.read_arg(t) as i8;
                self.tracks[t].flags.pitch = true;
            }
            0xBD => {
                // VOICE: copy the program's ToneData from the voicegroup.
                let program = self.read_arg(t) as usize;
                self.tracks[t].tone = ToneData::read(&self.rom, self.voicegroup + program * 12);
            }
            0xBE => {
                self.tracks[t].vol = self.read_arg(t);
                self.tracks[t].flags.volume = true;
            }
            0xBF => {
                self.tracks[t].pan = (self.read_arg(t).wrapping_sub(0x40)) as i8;
                self.tracks[t].flags.volume = true;
            }
            0xC0 => {
                self.tracks[t].bend = (self.read_arg(t).wrapping_sub(0x40)) as i8;
                self.tracks[t].flags.pitch = true;
            }
            0xC1 => {
                self.tracks[t].bend_range = self.read_arg(t);
                self.tracks[t].flags.pitch = true;
            }
            0xC2 => {
                self.tracks[t].lfo_speed = self.read_arg(t);
                if self.tracks[t].lfo_speed == 0 {
                    self.tracks[t].clear_mod_m();
                }
            }
            0xC3 => {
                self.tracks[t].lfo_delay = self.read_arg(t);
            }
            0xC4 => {
                self.tracks[t].mod_depth = self.read_arg(t);
                if self.tracks[t].mod_depth == 0 {
                    self.tracks[t].clear_mod_m();
                }
            }
            0xC5 => {
                let v = self.read_arg(t);
                if self.tracks[t].mod_type != v {
                    self.tracks[t].mod_type = v;
                    self.tracks[t].flags.volume = true;
                    self.tracks[t].flags.pitch = true;
                }
            }
            0xC8 => {
                self.tracks[t].tune = (self.read_arg(t).wrapping_sub(0x40)) as i8;
                self.tracks[t].flags.pitch = true;
            }
            0xCC => {
                // PORT: writes a raw GB sound register — consume the 2 operands, no effect.
                self.tracks[t].cmd_ptr += 2;
            }
            0xCD => self.xcmd(t),
            0xCE => {
                // EOT: end-of-tie, with an optional explicit key byte.
                let byte = self.read8(self.tracks[t].cmd_ptr);
                let key = if byte < 0x80 {
                    self.tracks[t].key = byte;
                    self.tracks[t].cmd_ptr += 1;
                    byte
                } else {
                    self.tracks[t].key
                };
                ops.push(Mp2kOp::EndTie { track: t, key });
            }
            // FINE and every unassigned slot in the jump table (0xB1, 0xB6..0xB8, 0xC6, 0xC7,
            // 0xC9..0xCB) stop the track.
            _ => self.fine(t, ops),
        }
    }

    /// Reads a one-byte command operand at the PC.
    fn read_arg(&mut self, t: usize) -> u8 {
        let v = self.read8(self.tracks[t].cmd_ptr);
        self.tracks[t].cmd_ptr += 1;
        v
    }

    /// `ply_memacc`: byte ops / conditional jumps over a 16-byte scratch area.
    fn memacc(&mut self, t: usize, ops: &mut Vec<Mp2kOp>) {
        let op = self.read_arg(t);
        let addr = (self.read_arg(t) as usize) & 0xF;
        let data = self.read_arg(t);
        let lhs = self.mem_acc[addr];
        let rhs = self.mem_acc[(data as usize) & 0xF];

        let cond = match op {
            0 => return self.mem_acc[addr] = data,
            1 => return self.mem_acc[addr] = lhs.wrapping_add(data),
            2 => return self.mem_acc[addr] = lhs.wrapping_sub(data),
            3 => return self.mem_acc[addr] = rhs,
            4 => return self.mem_acc[addr] = lhs.wrapping_add(rhs),
            5 => return self.mem_acc[addr] = lhs.wrapping_sub(rhs),
            6 => lhs == data,
            7 => lhs != data,
            8 => lhs > data,
            9 => lhs >= data,
            10 => lhs <= data,
            11 => lhs < data,
            12 => lhs == rhs,
            13 => lhs != rhs,
            14 => lhs > rhs,
            15 => lhs >= rhs,
            16 => lhs <= rhs,
            17 => lhs < rhs,
            _ => return,
        };
        if cond {
            self.goto(t, ops);
        } else {
            self.tracks[t].cmd_ptr += 4;
        }
    }

    /// `ply_xcmd`: the extended command set.
    fn xcmd(&mut self, t: usize) {
        let n = self.read_arg(t);
        match n {
            1 => {
                // xWAVE: override the tone's wave pointer.
                let wav = read_u32(&self.rom, self.tracks[t].cmd_ptr);
                self.tracks[t].tone.wav = wav;
                self.tracks[t].cmd_ptr += 4;
            }
            2 => self.tracks[t].tone.kind = self.read_arg(t), // xTYPE
            4 => self.tracks[t].tone.attack = self.read_arg(t), // xATTA
            5 => self.tracks[t].tone.decay = self.read_arg(t), // xDECA
            6 => self.tracks[t].tone.sustain = self.read_arg(t), // xSUST
            7 => self.tracks[t].tone.release = self.read_arg(t), // xRELE
            8 => self.tracks[t].echo_volume = self.read_arg(t), // xIECV
            9 => self.tracks[t].echo_length = self.read_arg(t), // xIECL
            10 => self.tracks[t].tone.length = self.read_arg(t), // xLENG
            11 => self.tracks[t].tone.pan_sweep = self.read_arg(t), // xSWEE
            12 => {
                // xWAIT: hold the track for a 16-bit number of steps.
                let len = u16::from(self.read8(self.tracks[t].cmd_ptr))
                    | (u16::from(self.read8(self.tracks[t].cmd_ptr + 1)) << 8);
                if self.tracks[t].timer < len {
                    self.tracks[t].timer += 1;
                    self.tracks[t].cmd_ptr -= 2; // re-execute `XCMD 0x0C` next step
                    self.tracks[t].wait = 1;
                } else {
                    self.tracks[t].timer = 0;
                    self.tracks[t].cmd_ptr += 2;
                }
            }
            13 => {
                // xCMD_0D: sample start offset (Pokémon cry progress).
                self.tracks[t].sample_start = read_u32(&self.rom, self.tracks[t].cmd_ptr);
                self.tracks[t].cmd_ptr += 4;
            }
            _ => {
                // xXXX and unknown: stop the track (matches ply_xxx → jump table 0 = FINE).
                self.tracks[t].exists = false;
            }
        }
    }
}

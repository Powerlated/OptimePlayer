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
    /// The first track that took a loop `GOTO` — the song's single loop reporter, so one
    /// musical repeat emits exactly one [`Mp2kOp::Looped`] (every track jumps individually,
    /// and usually in different frames).
    loop_track: Option<usize>,
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
            loop_track: None,
            finished: false,
            finish_reported: false,
        }
    }

    /// Current sequencer step rate in steps per *frame* — `tempo_i / 150`.
    pub fn steps_per_frame(&self) -> f64 {
        f64::from(self.tempo_i) / f64::from(TEMPO_STEP)
    }

    /// The current tempo register. MP2K writes `tempoI = bpm`, so this is the musical BPM
    /// (there are 24 sequencer steps per quarter note).
    pub(crate) fn tempo_i(&self) -> u16 {
        self.tempo_i
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
    ///
    /// `loop_point` marks jumps that signify the song repeating: a plain backward `GOTO` (how
    /// MP2K songs loop) emits [`Mp2kOp::Looped`]. Pattern calls (`PATT`), counted repeats, and
    /// `MEMACC` conditionals reuse this jump but are ordinary control flow, not loops.
    fn goto(&mut self, t: usize, ops: &mut Vec<Mp2kOp>, loop_point: bool) {
        let target = read_u32(&self.rom, self.tracks[t].cmd_ptr);
        match ptr_to_offset(target, self.rom.len()) {
            Some(offset) => {
                if loop_point && offset < self.tracks[t].cmd_ptr {
                    // One reporter per song: the first track to loop speaks for all of them.
                    if self.loop_track.is_none() {
                        self.loop_track = Some(t);
                    }
                    if self.loop_track == Some(t) {
                        ops.push(Mp2kOp::Looped);
                    }
                }
                self.tracks[t].cmd_ptr = offset;
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
            0xB2 => self.goto(t, ops, true), // GOTO: the song's loop point
            0xB3 => {
                // PATT: call a pattern.
                let level = self.tracks[t].pattern_level as usize;
                if level >= 3 {
                    self.fine(t, ops);
                } else {
                    self.tracks[t].pattern_stack[level] = self.tracks[t].cmd_ptr + 4;
                    self.tracks[t].pattern_level += 1;
                    self.goto(t, ops, false);
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
                    // REPT 0: repeat forever — this is the song looping.
                    self.tracks[t].cmd_ptr += 1;
                    self.goto(t, ops, true);
                } else {
                    self.tracks[t].rep_n = self.tracks[t].rep_n.wrapping_add(1);
                    let rep_n = self.tracks[t].rep_n;
                    self.tracks[t].cmd_ptr += 1;
                    if rep_n < count {
                        self.goto(t, ops, false);
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
            self.goto(t, ops, false);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a two-track ROM where each track calls a backward pattern (`PATT`/`PEND`) every
    /// iteration and loops with a backward `GOTO`:
    ///
    /// ```text
    /// 0x100: pattern   = VOL 100, PEND
    /// 0x200: track A   = PATT->0x100, W24, GOTO->0x200
    /// 0x300: track B   = PATT->0x100, W24, GOTO->0x300
    /// 0x400: header    = { 2 tracks, voicegroup 0x500, parts A, B }
    /// ```
    fn looping_song() -> (Arc<[u8]>, SongHeader) {
        const PATTERN: usize = 0x100;
        const TRACK_A: usize = 0x200;
        const TRACK_B: usize = 0x300;
        const HEADER: usize = 0x400;

        let mut rom = vec![0u8; 0x600];
        let ptr = |offset: usize| (0x0800_0000 + offset as u32).to_le_bytes();

        rom[PATTERN..PATTERN + 3].copy_from_slice(&[0xBE, 100, 0xB4]); // VOL, PEND

        for track in [TRACK_A, TRACK_B] {
            rom[track] = 0xB3; // PATT
            rom[track + 1..track + 5].copy_from_slice(&ptr(PATTERN));
            rom[track + 5] = 0x98; // W24
            rom[track + 6] = 0xB2; // GOTO (the loop point)
            rom[track + 7..track + 11].copy_from_slice(&ptr(track));
        }

        rom[HEADER] = 2;
        rom[HEADER + 4..HEADER + 8].copy_from_slice(&ptr(0x500));
        rom[HEADER + 8..HEADER + 12].copy_from_slice(&ptr(TRACK_A));
        rom[HEADER + 12..HEADER + 16].copy_from_slice(&ptr(TRACK_B));

        let header = SongHeader {
            offset: HEADER,
            track_count: 2,
            priority: 0,
            voicegroup: 0x500,
        };
        (Arc::from(rom), header)
    }

    /// Pattern calls jump backward every iteration but must not count as song loops; only the
    /// loop `GOTO` does, and only one track reports it (here: once per 24-step iteration).
    #[test]
    fn only_the_loop_goto_reports_a_loop_and_only_once() {
        let (rom, header) = looping_song();
        let mut seq = Mp2kSequencer::new(rom, &header);
        let mut ops = Vec::new();

        let mut loops_seen = Vec::new();
        for frame in 0..100u32 {
            ops.clear();
            seq.tick_frame(&mut ops);
            for op in &ops {
                if matches!(op, Mp2kOp::Looped) {
                    loops_seen.push(frame);
                }
            }
        }
        // Each iteration is W24 (the first frame runs PATT/VOL/PEND/W24, then GOTO fires when
        // the wait expires): loops at frames 24, 48, 72, 96 — and one report per loop, not two.
        assert_eq!(loops_seen, vec![24, 48, 72, 96]);
    }

    /// Direct transcription of `TrkVolPitSet` (`src/m4a.c`) used as the oracle. The unmodeled
    /// external inputs (`volX` fade is modeled; `panX`/`keyShiftX`/`pitX` are not) are zero.
    /// Returns `(vol_mr, vol_ml, key_m, pit_m)`.
    #[allow(clippy::too_many_arguments)]
    fn c_trk_vol_pit_set(
        vol: u8,
        vol_x: u8,
        pan: i8,
        mod_m: i8,
        mod_t: u8,
        bend: i8,
        bend_range: u8,
        tune: i8,
        key_shift: i8,
    ) -> (u8, u8, i8, u8) {
        let mut x = (u32::from(vol) * u32::from(vol_x)) >> 5;
        if mod_t == 1 {
            x = (x * (i32::from(mod_m) + 128) as u32) >> 7;
        }
        let mut y = 2 * i32::from(pan);
        if mod_t == 2 {
            y += i32::from(mod_m);
        }
        y = y.clamp(-128, 127);
        let vol_mr = (((y + 128) as u32 * x) >> 8) as u8;
        let vol_ml = (((127 - y) as u32 * x) >> 8) as u8;

        let bend_full = i32::from(bend) * i32::from(bend_range);
        let mut p = (i32::from(tune) + bend_full) * 4 + (i32::from(key_shift) << 8);
        if mod_t == 0 {
            p += 16 * i32::from(mod_m);
        }
        (vol_mr, vol_ml, (p >> 8) as i8, p as u8)
    }

    #[test]
    fn vol_pit_set_matches_pokeemerald() {
        let (rom, header) = looping_song();
        for vol in [0u8, 1, 90, 127, 255] {
            for vol_x in [0u8, 40, 64] {
                for pan in [-64i8, -1, 0, 63] {
                    for mod_m in [-64i8, 0, 17] {
                        for mod_t in [0u8, 1, 2] {
                            for (bend, bend_range, tune, key_shift) in [
                                (0i8, 2u8, 0i8, 0i8),
                                (-64, 2, 0, 0),
                                (63, 12, -10, -12),
                                (1, 255, 64, 11),
                            ] {
                                let mut seq = Mp2kSequencer::new(rom.clone(), &header);
                                let tr = &mut seq.tracks[0];
                                tr.flags = ChangeFlags {
                                    volume: true,
                                    pitch: true,
                                };
                                tr.vol = vol;
                                tr.vol_x = vol_x;
                                tr.pan = pan;
                                tr.mod_m = mod_m;
                                tr.mod_type = mod_t;
                                tr.bend = bend;
                                tr.bend_range = bend_range;
                                tr.tune = tune;
                                tr.key_shift = key_shift;
                                tr.vol_pit_set();
                                let expect = c_trk_vol_pit_set(
                                    vol, vol_x, pan, mod_m, mod_t, bend, bend_range, tune,
                                    key_shift,
                                );
                                assert_eq!(
                                    (tr.vol_mr, tr.vol_ml, tr.key_m, tr.pit_m),
                                    expect,
                                    "vol={vol} volX={vol_x} pan={pan} modM={mod_m} modT={mod_t} \
                                     bend={bend} range={bend_range} tune={tune} shift={key_shift}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// TrkVolPitSet only recomputes the half whose change flag is set.
    #[test]
    fn vol_pit_set_honors_change_flags() {
        let (rom, header) = looping_song();
        let mut seq = Mp2kSequencer::new(rom, &header);
        let tr = &mut seq.tracks[0];
        tr.vol = 100;
        tr.key_shift = 12;
        tr.flags = ChangeFlags {
            volume: false,
            pitch: true,
        };
        tr.vol_pit_set();
        assert_eq!((tr.vol_mr, tr.vol_ml), (0, 0), "volume untouched");
        assert_eq!(tr.key_m, 12, "pitch recomputed");
    }

    /// The oracle's LFO state, stepped per the `MPlayMain` asm (`src/m4a_1.s`, `_081DD95A`).
    #[derive(Default)]
    struct CLfo {
        speed_c: u8,
        delay_c: u8,
        mod_m: i8,
        flagged_pitch: bool,
        flagged_volume: bool,
    }

    fn c_lfo_step(t: &mut CLfo, speed: u8, depth: u8, mod_t: u8) {
        if speed == 0 || depth == 0 {
            return;
        }
        if t.delay_c != 0 {
            t.delay_c -= 1;
            return;
        }
        t.speed_c = t.speed_c.wrapping_add(speed);
        let triangle: i32 = if (t.speed_c.wrapping_sub(0x40) as i8) < 0 {
            i32::from(t.speed_c as i8)
        } else {
            0x80 - i32::from(t.speed_c)
        };
        let value = (i32::from(depth) * triangle) >> 6;
        if value as u8 != t.mod_m as u8 {
            t.mod_m = value as i8;
            if mod_t == 0 {
                t.flagged_pitch = true;
            } else {
                t.flagged_volume = true;
            }
        }
    }

    #[test]
    fn lfo_step_matches_pokeemerald() {
        let (rom, header) = looping_song();
        for speed in [0u8, 1, 22, 64, 130, 255] {
            for depth in [0u8, 1, 12, 127, 255] {
                for delay in [0u8, 3] {
                    for mod_t in [0u8, 1] {
                        let mut seq = Mp2kSequencer::new(rom.clone(), &header);
                        let tr = &mut seq.tracks[0];
                        tr.lfo_speed = speed;
                        tr.mod_depth = depth;
                        tr.lfo_delay_c = delay;
                        tr.mod_type = mod_t;
                        let mut oracle = CLfo {
                            delay_c: delay,
                            ..CLfo::default()
                        };
                        for step in 0..600u32 {
                            tr.flags = ChangeFlags::default();
                            oracle.flagged_pitch = false;
                            oracle.flagged_volume = false;
                            tr.lfo_step();
                            c_lfo_step(&mut oracle, speed, depth, mod_t);
                            assert_eq!(
                                (tr.mod_m, tr.lfo_speed_c, tr.lfo_delay_c),
                                (oracle.mod_m, oracle.speed_c, oracle.delay_c),
                                "speed={speed} depth={depth} delay={delay} modT={mod_t} step={step}"
                            );
                            assert_eq!(
                                (tr.flags.pitch, tr.flags.volume),
                                (oracle.flagged_pitch, oracle.flagged_volume),
                                "flags: speed={speed} depth={depth} delay={delay} modT={mod_t} step={step}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Builds a one-track song from raw `track` bytecode at 0x200 (voicegroup at 0x100 is
    /// zeroed = a plain DirectSound tone, so notes always resolve).
    fn one_track_song(track: &[u8]) -> (Arc<[u8]>, SongHeader) {
        const VOICEGROUP: usize = 0x100;
        const TRACK: usize = 0x200;
        const HEADER: usize = 0x400;

        let mut rom = vec![0u8; 0x600];
        let ptr = |offset: usize| (0x0800_0000 + offset as u32).to_le_bytes();
        rom[TRACK..TRACK + track.len()].copy_from_slice(track);
        rom[HEADER] = 1;
        rom[HEADER + 4..HEADER + 8].copy_from_slice(&ptr(VOICEGROUP));
        rom[HEADER + 8..HEADER + 12].copy_from_slice(&ptr(TRACK));
        let header = SongHeader {
            offset: HEADER,
            track_count: 1,
            priority: 0,
            voicegroup: VOICEGROUP,
        };
        (Arc::from(rom), header)
    }

    /// Runs the sequencer until it finishes (or 200 frames) and returns every op.
    fn run_song(track: &[u8]) -> Vec<Mp2kOp> {
        let (rom, header) = one_track_song(track);
        let mut seq = Mp2kSequencer::new(rom, &header);
        let mut all = Vec::new();
        for _ in 0..200 {
            let before = all.len();
            seq.tick_frame(&mut all);
            if all[before..]
                .iter()
                .any(|op| matches!(op, Mp2kOp::Finished))
            {
                break;
            }
        }
        all
    }

    fn notes(ops: &[Mp2kOp]) -> Vec<(u8, u8, u8)> {
        ops.iter()
            .filter_map(|op| match op {
                Mp2kOp::Note { note, .. } => Some((note.midi_key, note.velocity, note.gate)),
                _ => None,
            })
            .collect()
    }

    /// Bytes below 0x80 after a note command repeat it (running status), with each of the
    /// optional key / velocity / gate-extension bytes layering onto the previous state.
    #[test]
    fn notes_use_running_status_and_optional_bytes() {
        let ops = run_song(&[
            0xD0, 60, 100,  // N01 key=60 vel=100
            0x81, // W01
            62,   // running status: N01 key=62 (velocity kept)
            0x81, // W01
            63, 90, 2,    // running status: N01 key=63 vel=90 gate 1+2
            0x81, // W01
            0xB1, // FINE
        ]);
        assert_eq!(notes(&ops), vec![(60, 100, 1), (62, 100, 1), (63, 90, 3)]);
    }

    /// `REPT n` runs its section n times in total, then continues — without reporting loops.
    #[test]
    fn rept_repeats_counted_times_without_looping() {
        const TRACK: usize = 0x200;
        let target = (0x0800_0000u32 + TRACK as u32).to_le_bytes();
        let ops = run_song(&[
            0xD0, 60, 100,  // the repeated section: one note
            0x81, // W01
            0xB5, 3, target[0], target[1], target[2], target[3], // REPT 3 -> section start
            0xB1,      // FINE
        ]);
        assert_eq!(notes(&ops).len(), 3, "section runs exactly three times");
        assert!(!ops.iter().any(|op| matches!(op, Mp2kOp::Looped)));
        assert!(ops.iter().any(|op| matches!(op, Mp2kOp::TrackEnded { .. })));
    }

    /// `PATT` calls nest (up to three levels) and `PEND` returns to the caller.
    #[test]
    fn patt_calls_nest_and_return() {
        const TRACK: usize = 0x200;
        // Layout inside the track block: main first, then the outer and inner patterns.
        let ptr = |off: usize| (0x0800_0000u32 + (TRACK + off) as u32).to_le_bytes();
        let outer = ptr(11);
        let inner = ptr(21);
        let outer_call = [0xB3, outer[0], outer[1], outer[2], outer[3]];
        let mut track = Vec::new();
        track.extend_from_slice(&outer_call); // 0x00 main: PATT outer
        track.extend_from_slice(&outer_call); //      and again
        track.push(0xB1); // FINE
        assert_eq!(track.len(), 11);
        track.extend_from_slice(&[0xB3, inner[0], inner[1], inner[2], inner[3]]); // 0x0B outer: PATT inner
        track.extend_from_slice(&[0xD0, 72, 100, 0x81, 0xB4]); //      note 72, W01, PEND
        assert_eq!(track.len(), 21);
        track.extend_from_slice(&[0xD0, 60, 100, 0x81, 0xB4]); // 0x15 inner: note 60, W01, PEND

        let ops = run_song(&track);
        assert_eq!(
            notes(&ops),
            vec![(60, 100, 1), (72, 100, 1), (60, 100, 1), (72, 100, 1)],
            "each outer call plays the inner pattern then its own note"
        );
        assert!(!ops.iter().any(|op| matches!(op, Mp2kOp::Looped)));
    }

    /// `MEMACC` byte ops feed its conditional jumps: a true condition jumps, a false one
    /// skips the 4-byte target and falls through.
    #[test]
    fn memacc_conditionals_jump_or_fall_through() {
        const TRACK: usize = 0x200;
        let ptr = |off: usize| (0x0800_0000u32 + (TRACK + off) as u32).to_le_bytes();
        // 0x00: MEMACC SET [0] = 5
        // 0x04: MEMACC IF_EQ [0] == 7 -> note 99 (false: falls through)
        // 0x0c: MEMACC IF_EQ [0] == 5 -> skip over note 99 (true: jumps to 0x17)
        // 0x14: note 99 (must not play)
        // 0x17: note 60, W01, FINE
        let skip = ptr(0x17);
        let dead = ptr(0x14);
        let ops = run_song(&[
            0xB9, 0, 0, 5, // SET
            0xB9, 6, 0, 7, dead[0], dead[1], dead[2], dead[3], // IF_EQ false
            0xB9, 6, 0, 5, skip[0], skip[1], skip[2], skip[3], // IF_EQ true
            0xD0, 99, 100, // skipped
            0xD0, 60, 100, // 0x17
            0x81, 0xB1,
        ]);
        assert_eq!(notes(&ops), vec![(60, 100, 1)]);
    }

    /// `TEMPO n` sets the step rate to `n*2` tempo units (150 = two steps per frame),
    /// taking effect on the *next* frame — the current frame's budget was already accumulated.
    #[test]
    fn tempo_scales_steps_per_frame() {
        let (rom, header) = one_track_song(&[
            0xBB, 150,  // TEMPO: double speed
            0x98, // W24
            0xB1,
        ]);
        let mut seq = Mp2kSequencer::new(rom, &header);
        let mut ops = Vec::new();
        seq.tick_frame(&mut ops);
        assert_eq!(
            seq.steps, 1,
            "the frame the TEMPO lands in still runs one step"
        );
        seq.tick_frame(&mut ops);
        assert_eq!(seq.steps, 3, "subsequent frames run two steps");
        assert!((seq.steps_per_frame() - 2.0).abs() < 1e-12);
    }

    /// A `TIE` (gate 0) sounds until `EOT` releases it by key.
    #[test]
    fn tie_is_released_by_eot() {
        let ops = run_song(&[
            0xCF, 60, 100,  // TIE key=60
            0x82, // W02
            0xCE, // EOT (implicit key = last played)
            0xB1, // FINE
        ]);
        assert_eq!(notes(&ops), vec![(60, 100, 0)], "a tie has gate 0");
        assert!(
            ops.iter()
                .any(|op| matches!(op, Mp2kOp::EndTie { key: 60, .. })),
            "EOT releases the tied key"
        );
    }
}

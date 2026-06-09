//! The SSEQ bytecode interpreter: 16 tracks reading sequence opcodes and emitting [`Message`]s.

use std::sync::Arc;

use crate::util::{read_u8, CircularBuffer};
use crate::TRACK_COUNT;

/// The kind of a [`Message`] emitted by the sequence to the controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    /// `param0` = MIDI note, `param1` = velocity, `param2` = duration (ticks).
    PlayNote,
    /// `param0` = bank, `param1` = program.
    InstrumentChange,
    /// The track jumped (used for loop detection).
    Jump,
    /// The track ended.
    TrackEnded,
    /// `param0` = volume (0..127).
    VolumeChange,
    /// `param0` = pan (0..128).
    PanChange,
    /// Pitch bend / bend-range changed; the controller reads the track state.
    PitchBend,
}

/// A control message produced by a [`SequenceTrack`] for the controller to act on.
#[derive(Debug, Clone, Copy)]
pub struct Message {
    /// Whether this note originated from live keyboard input rather than the sequence.
    pub from_keyboard: bool,
    /// Which track emitted it.
    pub track_num: usize,
    /// The message kind.
    pub msg_type: MessageType,
    /// First parameter (meaning depends on `msg_type`).
    pub param0: i32,
    /// Second parameter.
    pub param1: i32,
    /// Third parameter.
    pub param2: i32,
    /// Tick the message was generated (filled in by consumers that need it).
    pub timestamp: u32,
}

/// How a command operand is encoded, mirroring pokediamond's `SND_SEQ_VAL_*` and the `0xA0`
/// (random) / `0xA1` (variable) prefixes consumed by `TrackParseValue`.
#[derive(Clone, Copy)]
enum ValType {
    U8,
    U16,
    Vlv,
    Ran,
    Var,
}

/// Per-track interpreter state. Contains no back-reference to its [`Sequence`]; the sequence
/// drives each track by index, which keeps the borrow checker happy and mirrors the hardware's
/// flat array of channels.
#[derive(Debug, Clone)]
pub struct SequenceTrack {
    /// Whether the track is currently executing.
    pub active: bool,
    /// Track tempo (only track 0's BPM drives the sequence clock).
    pub bpm: u32,
    /// Program counter (offset within the SSEQ data region).
    pub pc: u32,
    /// Pan (0..128).
    pub pan: i32,
    /// Mono/poly flag.
    pub mono: bool,
    /// Channel volume (0..127).
    pub volume: i32,
    /// Master volume (0..127).
    pub master_volume: i32,
    /// Track priority.
    pub priority: i32,
    /// Selected program (instrument).
    pub program: usize,
    /// Selected bank.
    pub bank: usize,
    /// LFO waveform type.
    pub lfo_type: i32,
    /// LFO depth.
    pub lfo_depth: i32,
    /// LFO range multiplier.
    pub lfo_range: i32,
    /// LFO speed.
    pub lfo_speed: i32,
    /// LFO delay (in ticks).
    pub lfo_delay: i32,
    /// Raw pitch-bend value.
    pub pitch_bend: i32,
    /// Pitch-bend range in semitones.
    pub pitch_bend_range: i32,
    /// Expression controller.
    pub expression: i32,
    /// Portamento enable.
    pub portamento_enable: i32,
    /// Portamento time.
    pub portamento_time: i32,
    /// Portamento source key (pokediamond `portamentoKey`, default 60).
    pub portamento_key: i32,
    /// Note transposition in semitones (`0xC3`).
    pub transpose: i32,
    /// Sweep-pitch amount (`0xE3`).
    pub sweep_pitch: i32,
    /// "Note wait" flag (pokediamond `flags.noteWait`, default true): when set, a note advances
    /// the track clock by its duration (the DS default), rather than relying on explicit rests.
    pub note_wait: bool,
    /// Whether the track is muted (`0xC8` tie / `0xD7` mute paths).
    pub muted: bool,
    /// Tie flag (`0xC8`).
    pub tie: bool,
    /// Set after a zero-duration note in note-wait mode: the track stalls until its channels
    /// finish (pokediamond `flags.noteFinishWait`).
    pub note_finish_wait: bool,
    /// Conditional-execution flag set by the compare commands (`0xB8`–`0xBD`); gates the next
    /// command after an `0xA2` prefix (pokediamond `flags.cmp`, default true).
    pub cmp: bool,
    /// Remaining ticks to rest before executing again.
    pub resting_for: u32,
    /// Call/return stack.
    pub stack: [u32; 64],
    /// Per-frame loop counters paralleling [`Self::stack`] (`0xD4`/`0xFC`).
    pub loop_count: [u8; 64],
    /// Stack pointer.
    pub sp: usize,
    /// Sequence-overridden ADSR rates (currently informational).
    pub attack_rate: i32,
    /// See [`Self::attack_rate`].
    pub decay_rate: i32,
    /// See [`Self::attack_rate`].
    pub sustain_rate: i32,
    /// See [`Self::attack_rate`].
    pub release_rate: i32,
}

impl Default for SequenceTrack {
    fn default() -> Self {
        Self {
            active: false,
            bpm: 0,
            pc: 0,
            // Defaults mirror pokediamond's `TrackInit`.
            pan: 64, // 0..128 representation; 64 == centre (pokediamond's signed 0)
            mono: false,
            volume: 127,
            master_volume: 127,
            priority: 64,
            program: 0,
            bank: 0,
            lfo_type: 0,
            lfo_depth: 0,
            lfo_range: 1,
            lfo_speed: 16,
            lfo_delay: 0,
            pitch_bend: 0,
            pitch_bend_range: 2,
            expression: 127,
            portamento_enable: 0,
            portamento_time: 0,
            portamento_key: 60,
            transpose: 0,
            sweep_pitch: 0,
            note_wait: true,
            muted: false,
            tie: false,
            note_finish_wait: false,
            cmp: true,
            resting_for: 0,
            stack: [0; 64],
            loop_count: [0; 64],
            sp: 0,
            attack_rate: 0,
            decay_rate: 0,
            sustain_rate: 0,
            release_rate: 0,
        }
    }
}

#[cfg(test)]
pub(crate) static OPCODE_SEEN: [std::sync::atomic::AtomicBool; 256] =
    [const { std::sync::atomic::AtomicBool::new(false) }; 256];

/// A running SSEQ sequence: the data, all 16 tracks, and the outgoing message queue.
pub struct Sequence {
    /// The SSEQ file bytes.
    pub sseq_file: Arc<[u8]>,
    /// Offset of the sequence data region within `sseq_file`.
    pub data_offset: u32,
    /// The 16 interpreter tracks.
    pub tracks: Vec<SequenceTrack>,
    /// Outgoing messages for the controller to consume each tick.
    pub message_buffer: CircularBuffer<Message>,
    /// Total ticks executed.
    pub ticks_elapsed: u32,
    /// Whether the sequence is paused.
    pub paused: bool,
    /// Player-global variables (pokediamond exposes 16; commands `0xB0`–`0xBD`, `0xA1`).
    pub variables: [i16; 16],
    /// LCG state for `SND_CalcRandom` (`0xA0`, `0xB6`).
    random_state: u32,
}

impl Sequence {
    /// Creates a sequence over `sseq_file`, with track 0 primed (active, 120 BPM).
    pub fn new(sseq_file: Arc<[u8]>, data_offset: u32, message_capacity: usize) -> Self {
        let mut tracks = vec![SequenceTrack::default(); TRACK_COUNT];
        tracks[0].active = true;
        tracks[0].bpm = 120;
        Self {
            sseq_file,
            data_offset,
            tracks,
            message_buffer: CircularBuffer::new(message_capacity),
            ticks_elapsed: 0,
            paused: false,
            variables: [0; 16],
            random_state: 0x1234_5678,
        }
    }

    /// `SND_CalcRandom` from `SND_util.c`: a 16-bit value from a 32-bit LCG.
    fn calc_random(&mut self) -> i32 {
        self.random_state = self
            .random_state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        (self.random_state >> 16) as i32
    }

    /// Advances every active track by one tick.
    ///
    /// `track_has_channels[i]` must report whether track `i` still has sounding (or releasing)
    /// channels; it gates pokediamond's `noteFinishWait` (the stall after a zero-duration note in
    /// note-wait mode). Callers without channel state (e.g. the look-ahead visualizer) can pass
    /// all-`false`, which makes such notes advance immediately.
    pub fn tick(&mut self, track_has_channels: &[bool; TRACK_COUNT]) {
        if !self.paused {
            for (i, &has_channels) in track_has_channels.iter().enumerate() {
                if !self.tracks[i].active {
                    continue;
                }
                // pokediamond stalls the track until its channels finish after a 0-length note.
                if self.tracks[i].note_finish_wait {
                    if has_channels {
                        continue;
                    }
                    self.tracks[i].note_finish_wait = false;
                }
                while self.tracks[i].resting_for == 0 && !self.tracks[i].note_finish_wait {
                    self.execute_track(i);
                }
                if self.tracks[i].resting_for > 0 {
                    self.tracks[i].resting_for -= 1;
                }
            }
        }
        self.ticks_elapsed = self.ticks_elapsed.wrapping_add(1);
    }

    /// Starts another track thread at `pc`.
    pub fn start_track(&mut self, num: usize, pc: u32) {
        self.tracks[num].active = true;
        self.tracks[num].pc = pc;
    }

    /// Stops track `num`.
    pub fn end_track(&mut self, num: usize) {
        self.tracks[num].active = false;
    }

    #[inline]
    fn read_pc(&self, idx: usize) -> u8 {
        read_u8(
            &self.sseq_file,
            (self.tracks[idx].pc + self.data_offset) as usize,
        )
    }

    /// Reads `bytes` little-endian bytes at the track's PC, advancing it.
    fn read_pc_inc(&mut self, idx: usize, bytes: u32) -> u32 {
        let mut val: u32 = 0;
        for i in 0..bytes {
            val |= u32::from(self.read_pc(idx)) << (i * 8);
            self.tracks[idx].pc += 1;
        }
        val
    }

    /// Reads a variable-length quantity (7 bits per byte, MSB = continue).
    fn read_variable_length(&mut self, idx: usize) -> u32 {
        let mut num: u32 = 0;
        for _ in 0..4 {
            let val = self.read_pc_inc(idx, 1);
            num <<= 7;
            num |= val & 0x7F;
            if val & 0x80 == 0 {
                break;
            }
        }
        num
    }

    fn push(&mut self, idx: usize, val: u32) {
        let sp = self.tracks[idx].sp;
        if sp < self.tracks[idx].stack.len() {
            self.tracks[idx].stack[sp] = val;
            self.tracks[idx].sp += 1;
        }
    }

    fn pop(&mut self, idx: usize) -> u32 {
        if self.tracks[idx].sp > 0 {
            self.tracks[idx].sp -= 1;
            self.tracks[idx].stack[self.tracks[idx].sp]
        } else {
            0
        }
    }

    fn send_message(
        &mut self,
        idx: usize,
        from_keyboard: bool,
        msg_type: MessageType,
        param0: i32,
        param1: i32,
        param2: i32,
    ) {
        self.message_buffer.insert(Message {
            from_keyboard,
            track_num: idx,
            msg_type,
            param0,
            param1,
            param2,
            timestamp: 0,
        });
    }

    /// Reads one command operand of the given encoding (pokediamond `TrackParseValue`).
    fn parse_value(&mut self, idx: usize, vt: ValType) -> i32 {
        match vt {
            ValType::U8 => self.read_pc_inc(idx, 1) as i32,
            ValType::U16 => self.read_pc_inc(idx, 2) as i32,
            ValType::Vlv => self.read_variable_length(idx) as i32,
            ValType::Var => {
                let var = (self.read_pc_inc(idx, 1) as usize) & 0xF;
                i32::from(self.variables[var])
            }
            ValType::Ran => {
                // lo/hi are signed 16-bit; pick a uniform value in [lo, hi].
                let lo = i32::from(self.read_pc_inc(idx, 2) as u16 as i16);
                let hi = i32::from(self.read_pc_inc(idx, 2) as u16 as i16);
                let ran = self.calc_random();
                let span = hi - lo + 1;
                (ran.wrapping_mul(span) >> 16) + lo
            }
        }
    }

    /// Executes a single command for track `idx`, following pokediamond's `TrackStepTicks`.
    ///
    /// Handles the `0xA2` conditional / `0xA0` random / `0xA1` variable prefixes, the note-on
    /// commands (`< 0x80`) with note-wait timing, and the control-command groups.
    pub fn execute_track(&mut self, idx: usize) {
        let mut opcode = self.read_pc_inc(idx, 1) as u8;
        #[cfg(test)]
        crate::sequence::OPCODE_SEEN[opcode as usize]
            .store(true, std::sync::atomic::Ordering::Relaxed);

        // Prefix bytes (pokediamond order: conditional, then random, then variable).
        let mut run_cmd = true;
        let mut special: Option<ValType> = None;
        if opcode == 0xA2 {
            opcode = self.read_pc_inc(idx, 1) as u8;
            run_cmd = self.tracks[idx].cmp;
        }
        if opcode == 0xA0 {
            opcode = self.read_pc_inc(idx, 1) as u8;
            special = Some(ValType::Ran);
        }
        if opcode == 0xA1 {
            opcode = self.read_pc_inc(idx, 1) as u8;
            special = Some(ValType::Var);
        }

        // Note-on: velocity (u8) + duration (value), key transposed and clamped.
        if opcode < 0x80 {
            let velocity = self.read_pc_inc(idx, 1) as i32;
            let length = self.parse_value(idx, special.unwrap_or(ValType::Vlv));
            let key = (opcode as i32 + self.tracks[idx].transpose).clamp(0, 127);
            if !run_cmd {
                return;
            }
            if !self.tracks[idx].muted {
                self.send_message(idx, false, MessageType::PlayNote, key, velocity, length);
            }
            self.tracks[idx].portamento_key = key;
            if self.tracks[idx].note_wait {
                self.tracks[idx].resting_for = length.max(0) as u32;
                if length == 0 {
                    self.tracks[idx].note_finish_wait = true;
                }
            }
            return;
        }

        match opcode & 0xF0 {
            0x80 => {
                let val = self.parse_value(idx, special.unwrap_or(ValType::Vlv));
                if !run_cmd {
                    return;
                }
                match opcode {
                    0x80 => self.tracks[idx].resting_for = val.max(0) as u32,
                    0x81 => {
                        let program = (val & 0x7F) as usize;
                        let bank = ((val >> 7) & 0x7F) as usize;
                        self.tracks[idx].program = program;
                        self.tracks[idx].bank = bank;
                        self.send_message(
                            idx,
                            false,
                            MessageType::InstrumentChange,
                            bank as i32,
                            program as i32,
                            0,
                        );
                    }
                    _ => {}
                }
            }
            0x90 => match opcode {
                0x93 => {
                    let track_num = self.read_pc_inc(idx, 1) as usize;
                    let track_offs = self.read_pc_inc(idx, 3);
                    if run_cmd && track_num < TRACK_COUNT {
                        self.start_track(track_num, track_offs);
                    }
                }
                0x94 => {
                    let dest = self.read_pc_inc(idx, 3);
                    if run_cmd {
                        self.tracks[idx].pc = dest;
                        self.send_message(idx, false, MessageType::Jump, 0, 0, 0);
                    }
                }
                0x95 => {
                    let dest = self.read_pc_inc(idx, 3);
                    if run_cmd {
                        let ret = self.tracks[idx].pc;
                        self.push(idx, ret);
                        self.tracks[idx].pc = dest;
                    }
                }
                _ => {}
            },
            0xB0 => {
                // Variable op: var index (u8) + value (u16 unless prefixed).
                let var_idx = (self.read_pc_inc(idx, 1) as usize) & 0xF;
                let val = self.parse_value(idx, special.unwrap_or(ValType::U16)) as i16 as i32;
                if run_cmd {
                    self.apply_var_op(idx, opcode, var_idx, val);
                }
            }
            0xC0 | 0xD0 => {
                let raw = self.parse_value(idx, special.unwrap_or(ValType::U8));
                if run_cmd {
                    self.apply_control(idx, opcode, raw);
                }
            }
            0xE0 => {
                let val = self.parse_value(idx, special.unwrap_or(ValType::U16)) as i16 as i32;
                if run_cmd {
                    match opcode {
                        0xE0 => self.tracks[idx].lfo_delay = val,
                        0xE1 => self.tracks[idx].bpm = (val as u32) & 0xFFFF,
                        0xE3 => self.tracks[idx].sweep_pitch = val,
                        _ => {}
                    }
                }
            }
            0xF0 => {
                if !run_cmd {
                    return;
                }
                match opcode {
                    0xFD => self.tracks[idx].pc = self.pop(idx),
                    0xFC => self.loop_end(idx),
                    // In our layout the `0xFE <mask>` track-allocation header sits at the start of
                    // track 0's stream (pokediamond consumes it during player setup, before
                    // stepping), so we read and discard its 2 bytes here to stay PC-aligned.
                    0xFE => {
                        let _ = self.read_pc_inc(idx, 2);
                    }
                    0xFF => {
                        self.end_track(idx);
                        self.send_message(idx, false, MessageType::TrackEnded, 0, 0, 0);
                        // Non-zero so the per-tick `while resting_for == 0` loop stops.
                        self.tracks[idx].resting_for = 1;
                    }
                    _ => {}
                }
            }
            _ => {
                // Unknown opcode; stop the track to avoid runaway execution.
                self.end_track(idx);
                self.tracks[idx].resting_for = 1;
            }
        }
    }

    /// Applies a `0xC0`/`0xD0`-group control command with its already-parsed operand.
    fn apply_control(&mut self, idx: usize, opcode: u8, raw: i32) {
        let u8v = raw & 0xFF;
        let s8v = u8v as u8 as i8 as i32;
        match opcode {
            0xC0 => {
                // Keep our 0..128 pan representation (centre 64); the `127 -> 128` nudge makes a
                // hard-right pan symmetric. (pokediamond stores the signed `raw - 0x40`; our
                // stereo engine is a separate Haas/crossover design, so we keep this mapping.)
                let pan = if u8v == 127 { 128 } else { u8v };
                self.tracks[idx].pan = pan;
                self.send_message(idx, false, MessageType::PanChange, pan, 0, 0);
            }
            0xC1 => {
                self.tracks[idx].volume = u8v;
                self.send_message(idx, false, MessageType::VolumeChange, u8v, 0, 0);
            }
            0xC2 => self.tracks[idx].master_volume = u8v,
            0xC3 => self.tracks[idx].transpose = s8v,
            0xC4 => {
                self.tracks[idx].pitch_bend = s8v;
                self.send_message(idx, false, MessageType::PitchBend, 0, 0, 0);
            }
            0xC5 => {
                self.tracks[idx].pitch_bend_range = u8v;
                self.send_message(idx, false, MessageType::PitchBend, 0, 0, 0);
            }
            0xC6 => self.tracks[idx].priority = u8v,
            0xC7 => self.tracks[idx].note_wait = u8v != 0,
            0xC8 => self.tracks[idx].tie = u8v != 0,
            0xC9 => {
                self.tracks[idx].portamento_key = (s8v + self.tracks[idx].transpose).clamp(0, 127);
                self.tracks[idx].portamento_enable = 1;
            }
            0xCA => self.tracks[idx].lfo_depth = u8v,
            0xCB => self.tracks[idx].lfo_speed = u8v,
            0xCC => self.tracks[idx].lfo_type = u8v,
            0xCD => self.tracks[idx].lfo_range = u8v,
            0xCE => self.tracks[idx].portamento_enable = u8v,
            0xCF => self.tracks[idx].portamento_time = u8v,
            0xD0 => self.tracks[idx].attack_rate = u8v,
            0xD1 => self.tracks[idx].decay_rate = u8v,
            0xD2 => self.tracks[idx].sustain_rate = u8v,
            0xD3 => self.tracks[idx].release_rate = u8v,
            0xD4 => {
                // Loop start: push the current PC and the loop count onto the shared stack.
                let sp = self.tracks[idx].sp;
                if sp < self.tracks[idx].stack.len() {
                    let pc = self.tracks[idx].pc;
                    self.tracks[idx].stack[sp] = pc;
                    self.tracks[idx].loop_count[sp] = u8v as u8;
                    self.tracks[idx].sp += 1;
                }
            }
            0xD5 => self.tracks[idx].expression = u8v,
            // 0xD6 print var, 0xD7 mute: not modelled (operand already consumed).
            _ => {}
        }
    }

    /// Loop-end (`0xFC`): decrement the top loop counter and jump back, or pop when it reaches 0.
    /// A stored count of 0 means an infinite loop. Mirrors pokediamond's `0xFC` handling.
    fn loop_end(&mut self, idx: usize) {
        let sp = self.tracks[idx].sp;
        if sp == 0 {
            return;
        }
        let mut count = self.tracks[idx].loop_count[sp - 1];
        if count != 0 {
            count -= 1;
            if count == 0 {
                self.tracks[idx].sp -= 1;
                return;
            }
        }
        self.tracks[idx].loop_count[sp - 1] = count;
        self.tracks[idx].pc = self.tracks[idx].stack[sp - 1];
    }

    /// Applies a `0xB0`-group variable/arithmetic/compare command (pokediamond `0xB0`–`0xBD`).
    fn apply_var_op(&mut self, idx: usize, opcode: u8, var_idx: usize, par: i32) {
        let v = &mut self.variables[var_idx];
        let cur = i32::from(*v);
        match opcode {
            0xB0 => *v = par as i16,
            0xB1 => *v = cur.wrapping_add(par) as i16,
            0xB2 => *v = cur.wrapping_sub(par) as i16,
            0xB3 => *v = cur.wrapping_mul(par) as i16,
            0xB4 => {
                if par != 0 {
                    *v = (cur / par) as i16;
                }
            }
            0xB5 => {
                *v = if par >= 0 {
                    cur.wrapping_shl(par as u32) as i16
                } else {
                    (cur >> (-par)) as i16
                }
            }
            0xB6 => {
                let (neg, mag) = if par < 0 { (true, -par) } else { (false, par) };
                let mut random = self.calc_random();
                random = random.wrapping_mul(mag + 1) >> 16;
                self.variables[var_idx] = if neg { -random } else { random } as i16;
            }
            0xB8 => self.tracks[idx].cmp = cur == par,
            0xB9 => self.tracks[idx].cmp = cur >= par,
            0xBA => self.tracks[idx].cmp = cur > par,
            0xBB => self.tracks[idx].cmp = cur <= par,
            0xBC => self.tracks[idx].cmp = cur < par,
            0xBD => self.tracks[idx].cmp = cur != par,
            // 0xB7 (unused in pokediamond) and others: no-op.
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(program: &[u8]) -> Sequence {
        Sequence::new(Arc::from(program.to_vec()), 0, 64)
    }

    fn drain(s: &mut Sequence) -> Vec<Message> {
        let mut out = Vec::new();
        while let Some(m) = s.message_buffer.pop() {
            out.push(m);
        }
        out
    }

    #[test]
    fn note_opcode_emits_play_note_with_single_byte_duration() {
        // note 60, velocity 127, duration 0x10. With note-wait on (the pokediamond default), the
        // note's duration advances the track clock, so the tick stops here without reaching the
        // trailing rest.
        let mut s = seq(&[0x3C, 0x7F, 0x10, 0x80, 0x04, 0xFF]);
        s.tick(&[false; TRACK_COUNT]);
        let msgs = drain(&mut s);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].msg_type, MessageType::PlayNote);
        assert_eq!(msgs[0].param0, 60);
        assert_eq!(msgs[0].param1, 127);
        assert_eq!(msgs[0].param2, 0x10);
        // note-wait set resting_for to the duration (0x10); the tick then decrements it once.
        assert_eq!(s.tracks[0].resting_for, 0x10 - 1);
    }

    #[test]
    fn note_wait_off_fires_notes_back_to_back() {
        // 0xC7 0 disables note-wait; the two notes then fire on the same tick (a chord), and only
        // the explicit rest advances the clock. This is the model the original engine assumed.
        let mut s = seq(&[
            0xC7, 0x00, // note-wait off
            0x3C, 0x7F, 0x20, // note 60, dur 0x20
            0x40, 0x7F, 0x20, // note 64, dur 0x20
            0x80, 0x04, // rest 4
            0xFF,
        ]);
        s.tick(&[false; TRACK_COUNT]);
        let msgs = drain(&mut s);
        let notes: Vec<i32> = msgs
            .iter()
            .filter(|m| m.msg_type == MessageType::PlayNote)
            .map(|m| m.param0)
            .collect();
        assert_eq!(
            notes,
            vec![60, 64],
            "both notes should fire on the same tick"
        );
        assert_eq!(s.tracks[0].resting_for, 3); // rest 4 - 1
    }

    #[test]
    fn note_wait_on_advances_by_note_duration() {
        // With note-wait on (default), the first note's duration is the inter-note delay, so the
        // second note has not fired yet after one tick.
        let mut s = seq(&[
            0x3C, 0x7F, 0x08, // note 60, dur 8
            0x40, 0x7F, 0x08, // note 64, dur 8
            0xFF,
        ]);
        s.tick(&[false; TRACK_COUNT]);
        let notes: Vec<i32> = drain(&mut s)
            .iter()
            .filter(|m| m.msg_type == MessageType::PlayNote)
            .map(|m| m.param0)
            .collect();
        assert_eq!(notes, vec![60], "only the first note should have fired");
        assert_eq!(s.tracks[0].resting_for, 7); // dur 8 - 1
    }

    #[test]
    fn transpose_shifts_note_keys() {
        // 0xC3 +12 transposes the following note up an octave; the key is clamped to 0..127.
        let mut s = seq(&[0xC3, 12, 0x3C, 0x7F, 0x04, 0xFF]);
        s.tick(&[false; TRACK_COUNT]);
        let msgs = drain(&mut s);
        assert_eq!(msgs[0].msg_type, MessageType::PlayNote);
        assert_eq!(msgs[0].param0, 72);
    }

    #[test]
    fn loop_start_end_repeats_the_body() {
        // 0xD4 2 (loop twice) ... 0xFC: the pan-setting body runs, looping back once. Drive ticks
        // until the track ends and count how many times the body executed via resting_for resets.
        let mut s = seq(&[
            0xC7, 0x00, // note-wait off so the rest carries timing
            0xD4, 0x02, // loop start, count 2
            0xC0, 70, // pan 70 (body)
            0x80, 0x01, // rest 1
            0xFC, // loop end
            0xFF,
        ]);
        // Tick 1 runs the loop body once (sets pan, rests 1) and pushes the loop frame.
        s.tick(&[false; TRACK_COUNT]);
        assert_eq!(s.tracks[0].pan, 70);
        assert_eq!(s.tracks[0].sp, 1, "loop frame pushed");
        // Tick 2: 0xFC decrements the count (2 -> 1) and jumps back; body runs a 2nd time.
        s.tick(&[false; TRACK_COUNT]);
        assert_eq!(s.tracks[0].sp, 1, "still looping on the second iteration");
        // Tick 3: 0xFC decrements (1 -> 0), pops the frame, and the track reaches 0xFF.
        s.tick(&[false; TRACK_COUNT]);
        assert_eq!(
            s.tracks[0].sp, 0,
            "loop frame popped after the final iteration"
        );
        assert!(!s.tracks[0].active, "track ended after the loop");
    }

    #[test]
    fn conditional_prefix_gates_on_compare_flag() {
        // Set var0 = 5, compare-equal to 3 (false), then a conditional (0xA2) pan command must be
        // skipped; a following unconditional pan must still apply.
        let mut s = seq(&[
            0xB0, 0x00, 0x05, 0x00, // var0 = 5
            0xB8, 0x00, 0x03, 0x00, // cmp: var0 == 3 -> false
            0xA2, 0xC0, 70, // if(cmp) pan 70  -> skipped
            0xC0, 40, // pan 40 (always)
            0x80, 0x02, 0xFF,
        ]);
        s.tick(&[false; TRACK_COUNT]);
        assert!(!s.tracks[0].cmp);
        assert_eq!(
            s.tracks[0].pan, 40,
            "conditional pan must have been skipped"
        );
        assert_eq!(s.variables[0], 5);
    }

    #[test]
    fn variable_length_duration_decodes_multi_byte() {
        // duration 0x81 0x00 -> (1 << 7) | 0 = 128.
        let mut s = seq(&[0x3C, 0x7F, 0x81, 0x00, 0x80, 0x04, 0xFF]);
        s.tick(&[false; TRACK_COUNT]);
        let msgs = drain(&mut s);
        assert_eq!(msgs[0].param2, 128);
    }

    #[test]
    fn control_opcodes_set_state_and_emit_messages() {
        // BPM=0x00F0 (240), volume=100, pan=64, program/bank via 0x81, then rest.
        let mut s = seq(&[
            0xE1, 0xF0, 0x00, // BPM 240
            0xC1, 100, // volume 100
            0xC0, 64, // pan 64
            0x81, 0x05, // bank 0 program 5
            0x80, 0x04, // rest 4
            0xFF,
        ]);
        s.tick(&[false; TRACK_COUNT]);
        let msgs = drain(&mut s);
        assert_eq!(s.tracks[0].bpm, 240);
        assert_eq!(s.tracks[0].volume, 100);
        assert_eq!(s.tracks[0].pan, 64);
        assert_eq!(s.tracks[0].program, 5);

        let types: Vec<MessageType> = msgs.iter().map(|m| m.msg_type).collect();
        assert_eq!(
            types,
            vec![
                MessageType::VolumeChange,
                MessageType::PanChange,
                MessageType::InstrumentChange,
            ]
        );
        assert_eq!(msgs[0].param0, 100);
        assert_eq!(msgs[2].param0, 0); // bank
        assert_eq!(msgs[2].param1, 5); // program
    }

    #[test]
    fn jump_sets_pc_and_emits_jump() {
        // Jump to offset 5 (where a rest lives), avoiding an infinite loop.
        let mut s = seq(&[0x94, 0x05, 0x00, 0x00, 0xFF, 0x80, 0x02, 0xFF]);
        s.tick(&[false; TRACK_COUNT]);
        let msgs = drain(&mut s);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].msg_type, MessageType::Jump);
        // After jumping to 5 and reading the rest opcode + length, pc is at 7.
        assert_eq!(s.tracks[0].pc, 7);
    }

    #[test]
    fn call_and_return_use_the_stack() {
        // Call 6 -> pan opcode + rest; return; then rest. Verifies push/pop of return address.
        let mut s = seq(&[
            0x95, 0x06, 0x00, 0x00, // call 0x000006
            0x80, 0x08, // (return lands here) rest 8
            0xC0, 70,   // pan 70
            0xFD, // return
        ]);
        s.tick(&[false; TRACK_COUNT]);
        let _ = drain(&mut s);
        assert_eq!(s.tracks[0].pan, 70);
        // Returned to offset 4, read rest (resting_for 8 - 1).
        assert_eq!(s.tracks[0].resting_for, 7);
    }

    #[test]
    fn start_track_activates_another_thread() {
        // 0x93: track0 starts track1 at offset 7, then rests; track1 rests 2 then ends.
        let mut s = seq(&[
            0x93, 0x01, 0x07, 0x00, 0x00, 0x80, 0x04, // track0: start track1@7, rest 4
            0x80, 0x02, 0xFF, // track1 @7: rest 2, end
        ]);
        s.tick(&[false; TRACK_COUNT]);
        assert!(s.tracks[1].active);
        // track1 executed its rest (2) once this tick, leaving resting_for = 1.
        assert_eq!(s.tracks[1].resting_for, 1);
    }
}

//! The SSEQ bytecode interpreter: 16 tracks reading sequence opcodes and emitting [`Message`]s.
//!
//! - [`message`] — the [`Message`] / [`MessageType`] the tracks emit.
//! - [`track`] — [`SequenceTrack`], the per-track register/stack state.
//! - [`interpreter`] — the opcode-stepping `impl Sequence` (the bytecode engine).
//! - [`Sequence`] (here) — the data, the 16 tracks, the message queue, and the per-tick driver.

mod interpreter;
mod message;
mod track;

#[cfg(test)]
mod tests;

pub use message::{Message, MessageType};
pub use track::SequenceTrack;

use std::sync::Arc;

use crate::util::{read_u8, CircularBuffer};
use crate::TRACK_COUNT;

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

    fn send_message(&mut self, idx: usize, msg_type: MessageType) {
        self.message_buffer.insert(Message {
            track_num: idx,
            msg_type,
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
}

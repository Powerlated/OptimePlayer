mod interpreter;
mod message;
mod track;

#[cfg(test)]
mod tests;

pub use message::{Message, MessageType};
pub use track::SequenceTrack;

use std::sync::Arc;

use crate::TRACK_COUNT;
use crate::util::{CircularBuffer, read_u8};

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

pub struct Sequence {
    pub sseq_file: Arc<[u8]>,
    pub data_offset: u32,
    pub tracks: Vec<SequenceTrack>,
    pub message_buffer: CircularBuffer<Message>,
    pub ticks_elapsed: u32,
    pub paused: bool,
    pub variables: [i16; 16],
    random_state: u32,
}

impl Sequence {
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

    fn calc_random(&mut self) -> i32 {
        self.random_state = self
            .random_state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        (self.random_state >> 16) as i32
    }

    pub fn tick(&mut self, track_has_channels: &[bool; TRACK_COUNT]) {
        if !self.paused {
            for (i, &has_channels) in track_has_channels.iter().enumerate() {
                if !self.tracks[i].active {
                    continue;
                }
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

    pub fn start_track(&mut self, num: usize, pc: u32) {
        self.tracks[num].active = true;
        self.tracks[num].pc = pc;
    }

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

    fn read_pc_inc(&mut self, idx: usize, bytes: u32) -> u32 {
        let mut val: u32 = 0;
        for i in 0..bytes {
            val |= u32::from(self.read_pc(idx)) << (i * 8);
            self.tracks[idx].pc += 1;
        }
        val
    }

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
                let lo = i32::from(self.read_pc_inc(idx, 2) as u16 as i16);
                let hi = i32::from(self.read_pc_inc(idx, 2) as u16 as i16);
                let ran = self.calc_random();
                let span = hi - lo + 1;
                (ran.wrapping_mul(span) >> 16) + lo
            }
        }
    }
}

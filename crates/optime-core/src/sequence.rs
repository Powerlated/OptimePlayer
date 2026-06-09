//! The SSEQ bytecode interpreter: 16 tracks reading sequence opcodes and emitting [`Message`]s.

use std::sync::Arc;

use crate::util::{bit_test, read_u8, CircularBuffer};
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
    /// Remaining ticks to rest before executing again.
    pub resting_for: u32,
    /// Call/return stack.
    pub stack: [u32; 64],
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
            pan: 64,
            mono: false,
            volume: 0,
            master_volume: 0,
            priority: 0,
            program: 0,
            bank: 0,
            lfo_type: 0,
            lfo_depth: 0,
            lfo_range: 0,
            lfo_speed: 16,
            lfo_delay: 0,
            pitch_bend: 0,
            pitch_bend_range: 0,
            expression: 0,
            portamento_enable: 0,
            portamento_time: 0,
            resting_for: 0,
            stack: [0; 64],
            sp: 0,
            attack_rate: 0,
            decay_rate: 0,
            sustain_rate: 0,
            release_rate: 0,
        }
    }
}

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
        }
    }

    /// Advances every active track by one tick.
    pub fn tick(&mut self) {
        if !self.paused {
            for i in 0..TRACK_COUNT {
                if self.tracks[i].active {
                    while self.tracks[i].resting_for == 0 {
                        self.execute_track(i);
                    }
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

    /// Executes a single opcode for track `idx`.
    pub fn execute_track(&mut self, idx: usize) {
        let opcode = self.read_pc_inc(idx, 1);

        if opcode <= 0x7F {
            // Play note: velocity byte + variable-length duration.
            let velocity = self.read_pc_inc(idx, 1) as i32;
            let duration = self.read_variable_length(idx) as i32;
            self.send_message(
                idx,
                false,
                MessageType::PlayNote,
                opcode as i32,
                velocity,
                duration,
            );
            return;
        }

        match opcode {
            0xFE => {
                // Allocate tracks (not needed for emulation).
                let _alloced = self.read_pc_inc(idx, 2);
            }
            0x93 => {
                // Start new track thread.
                let track_num = self.read_pc_inc(idx, 1) as usize;
                let track_offs = self.read_pc_inc(idx, 3);
                self.start_track(track_num, track_offs);
            }
            0xC7 => {
                let param = self.read_pc_inc(idx, 1);
                self.tracks[idx].mono = bit_test(param, 0);
            }
            0xCE => {
                self.tracks[idx].portamento_enable = self.read_pc_inc(idx, 1) as i32;
            }
            0xCF => {
                self.tracks[idx].portamento_time = self.read_pc_inc(idx, 1) as i32;
            }
            0xE1 => {
                self.tracks[idx].bpm = self.read_pc_inc(idx, 2);
            }
            0xC1 => {
                let volume = self.read_pc_inc(idx, 1) as i32;
                self.tracks[idx].volume = volume;
                self.send_message(idx, false, MessageType::VolumeChange, volume, 0, 0);
            }
            0x81 => {
                let bank_and_program = self.read_variable_length(idx);
                let program = (bank_and_program & 0x7F) as usize;
                let bank = ((bank_and_program >> 7) & 0x7F) as usize;
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
            0xC2 => {
                self.tracks[idx].master_volume = self.read_pc_inc(idx, 1) as i32;
            }
            0xC0 => {
                let mut pan = self.read_pc_inc(idx, 1) as i32;
                if pan == 127 {
                    pan = 128;
                }
                self.tracks[idx].pan = pan;
                self.send_message(idx, false, MessageType::PanChange, pan, 0, 0);
            }
            0xC6 => {
                self.tracks[idx].priority = self.read_pc_inc(idx, 1) as i32;
            }
            0xC5 => {
                self.tracks[idx].pitch_bend_range = self.read_pc_inc(idx, 1) as i32;
                self.send_message(idx, false, MessageType::PitchBend, 0, 0, 0);
            }
            0xCA => {
                self.tracks[idx].lfo_depth = self.read_pc_inc(idx, 1) as i32;
            }
            0xCB => {
                self.tracks[idx].lfo_speed = self.read_pc_inc(idx, 1) as i32;
            }
            0xCC => {
                self.tracks[idx].lfo_type = self.read_pc_inc(idx, 1) as i32;
            }
            0xCD => {
                self.tracks[idx].lfo_range = self.read_pc_inc(idx, 1) as i32;
            }
            0xC4 => {
                self.tracks[idx].pitch_bend = self.read_pc_inc(idx, 1) as i32;
                self.send_message(idx, false, MessageType::PitchBend, 0, 0, 0);
            }
            0x80 => {
                self.tracks[idx].resting_for = self.read_variable_length(idx);
            }
            0x94 => {
                let dest = self.read_pc_inc(idx, 3);
                self.tracks[idx].pc = dest;
                self.send_message(idx, false, MessageType::Jump, 0, 0, 0);
            }
            0x95 => {
                let dest = self.read_pc_inc(idx, 3);
                let ret = self.tracks[idx].pc;
                self.push(idx, ret);
                self.tracks[idx].pc = dest;
            }
            0xFD => {
                self.tracks[idx].pc = self.pop(idx);
            }
            0xB0 => {
                // Arithmetic op (per sseq2mid); skip 3 bytes.
                self.read_pc_inc(idx, 3);
            }
            0xE0 => {
                self.tracks[idx].lfo_delay = self.read_pc_inc(idx, 2) as i32;
            }
            0xD5 => {
                self.tracks[idx].expression = self.read_pc_inc(idx, 1) as i32;
            }
            0xFF => {
                self.end_track(idx);
                self.send_message(idx, false, MessageType::TrackEnded, 0, 0, 0);
                // Non-zero so the per-tick `while resting_for == 0` loop stops.
                self.tracks[idx].resting_for = 1;
            }
            0xD0 => self.tracks[idx].attack_rate = self.read_pc_inc(idx, 1) as i32,
            0xD1 => self.tracks[idx].decay_rate = self.read_pc_inc(idx, 1) as i32,
            0xD2 => self.tracks[idx].sustain_rate = self.read_pc_inc(idx, 1) as i32,
            0xD3 => self.tracks[idx].release_rate = self.read_pc_inc(idx, 1) as i32,
            _ => {
                // Unknown opcode; stop the track to avoid runaway execution.
                self.end_track(idx);
                self.tracks[idx].resting_for = 1;
            }
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
        // note 60, velocity 127, duration 0x10, then rest to stop the tick.
        let mut s = seq(&[0x3C, 0x7F, 0x10, 0x80, 0x04, 0xFF]);
        s.tick();
        let msgs = drain(&mut s);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].msg_type, MessageType::PlayNote);
        assert_eq!(msgs[0].param0, 60);
        assert_eq!(msgs[0].param1, 127);
        assert_eq!(msgs[0].param2, 0x10);
        // The rest consumed leaves resting_for one less than its value.
        assert_eq!(s.tracks[0].resting_for, 3);
    }

    #[test]
    fn variable_length_duration_decodes_multi_byte() {
        // duration 0x81 0x00 -> (1 << 7) | 0 = 128.
        let mut s = seq(&[0x3C, 0x7F, 0x81, 0x00, 0x80, 0x04, 0xFF]);
        s.tick();
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
        s.tick();
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
        s.tick();
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
        s.tick();
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
        s.tick();
        assert!(s.tracks[1].active);
        // track1 executed its rest (2) once this tick, leaving resting_for = 1.
        assert_eq!(s.tracks[1].resting_for, 1);
    }
}

//! The SSEQ bytecode interpreter: one command step per call, plus the control-command
//! (`0xC0`/`0xD0`), variable/arithmetic (`0xB0`), and loop-end handlers.

use super::MessageType;
use super::{Sequence, ValType};
use crate::TRACK_COUNT;

impl Sequence {
    /// Executes a single command for track `idx`, following pokediamond's `TrackStepTicks`.
    ///
    /// Handles the `0xA2` conditional / `0xA0` random / `0xA1` variable prefixes, the note-on
    /// commands (`< 0x80`) with note-wait timing, and the control-command groups.
    pub fn execute_track(&mut self, idx: usize) {
        let mut opcode = self.read_pc_inc(idx, 1) as u8;
        #[cfg(test)]
        super::OPCODE_SEEN[opcode as usize].store(true, std::sync::atomic::Ordering::Relaxed);

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
                self.send_message(
                    idx,
                    MessageType::PlayNote {
                        note: key,
                        velocity,
                        duration: length,
                    },
                );
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
                            MessageType::InstrumentChange {
                                bank: bank as i32,
                                program: program as i32,
                            },
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
                        self.send_message(idx, MessageType::Jump);
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
                        self.send_message(idx, MessageType::TrackEnded);
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
                self.send_message(idx, MessageType::PanChange { pan });
            }
            0xC1 => {
                self.tracks[idx].volume = u8v;
                self.send_message(idx, MessageType::VolumeChange { volume: u8v });
            }
            0xC2 => self.tracks[idx].master_volume = u8v,
            0xC3 => self.tracks[idx].transpose = s8v,
            0xC4 => {
                self.tracks[idx].pitch_bend = s8v;
                self.send_message(idx, MessageType::PitchBend);
            }
            0xC5 => {
                self.tracks[idx].pitch_bend_range = u8v;
                self.send_message(idx, MessageType::PitchBend);
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

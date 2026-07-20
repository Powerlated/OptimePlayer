//! The DSE SMDL track-event bytecode: opcode table and a static disassembler.
//!
//! Every fact in this file is transcribed from the `pret/pmd-sky` decompilation of the
//! Procyon Studios DSE driver — *not* from external wiki tables — so it is exact for
//! Explorers of Sky:
//!
//! - The 0x90–0xFF handler table is `SMD_EVENTS_FUN_TABLE` in
//!   `asm/main_rodata_020A2808.s` (112 entries, opcode `c` → entry `c - 0x90`).
//! - Each event's operand-byte count is how far its handler advances the track pointer
//!   (`u8* DseTrackEvent_*(u8 *ptr_next_byte, …)` returns `ptr_next_byte + n`), read from
//!   `lib/DSE/src/*.c`, `lib/DSE/asm/*.s`, and `asm/main_0206C9BC.s`.
//! - The 0x00–0x7F PlayNote bit layout and the 0x80–0x8F pause lookup come from
//!   `ParseDseEvent` (`asm/main_0206C9BC.s`) and the `_020B0B7C` rodata table.

/// Fixed-duration pause lengths in ticks for opcodes 0x80–0x8F.
///
/// Verbatim from the `_020B0B7C` rodata table in `asm/main_rodata_020A2808.s`; the driver
/// indexes it as `pause_table[opcode - 0x80]`.
pub const PAUSE_TICKS: [u16; 16] = [96, 72, 64, 48, 36, 32, 24, 18, 16, 12, 9, 8, 6, 4, 3, 2];

/// A decoded track event. `Note` and `Pause` are the common cases; everything else is a
/// control opcode carrying its raw operand bytes (named by its decomp handler).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DseEvent {
    /// A `PlayNote` (opcode 0x00–0x7F). `velocity` is the opcode byte.
    Note {
        velocity: u8,
        /// Absolute MIDI key = `octave * 12 + note`, after applying this note's octave delta.
        key: u8,
        /// Octave change this note applied to the track (`((notedata >> 4) & 3) - 2`).
        octave_delta: i8,
        /// `None` reuses the track's previous duration; `Some` is this note's duration in ticks.
        duration: Option<u32>,
    },
    /// A fixed-length rest (opcode 0x80–0x8F), length from [`PAUSE_TICKS`].
    Pause { ticks: u16 },
    /// A control opcode (0x90–0xFF): its name, opcode, and raw operand bytes.
    Control {
        opcode: u8,
        name: &'static str,
        operands: Vec<u8>,
    },
    /// An opcode the driver maps to `DseTrackEvent_Invalid` (a no-op consuming no operands).
    Invalid { opcode: u8 },
}

/// `(name, operand_byte_count)` for control opcodes 0x90–0xFF, indexed by `opcode - 0x90`.
///
/// `None` marks the slots wired to `DseTrackEvent_Invalid`. Operand counts are the pointer
/// advance of each handler in the decomp (see the module docs).
#[rustfmt::skip]
const CONTROL_TABLE: [Option<(&str, u8)>; 112] = [
    /* 0x90 */ Some(("WaitSame", 0)),
    /* 0x91 */ Some(("WaitDelta", 1)),
    /* 0x92 */ Some(("Wait8", 1)),
    /* 0x93 */ Some(("Wait16", 2)),
    /* 0x94 */ Some(("Wait24", 3)),
    /* 0x95 */ Some(("WaitUntilFadeout", 1)),
    /* 0x96 */ None,
    /* 0x97 */ None,
    /* 0x98 */ Some(("EndTrack", 0)),
    /* 0x99 */ Some(("MainLoopBegin", 0)),
    /* 0x9A */ None,
    /* 0x9B */ None,
    /* 0x9C */ Some(("SubLoopBegin", 1)),
    /* 0x9D */ Some(("SubLoopEnd", 0)),
    /* 0x9E */ Some(("SubLoopBreakOnLastIteration", 0)),
    /* 0x9F */ None,
    /* 0xA0 */ Some(("SetOctave", 1)),
    /* 0xA1 */ Some(("OctaveDelta", 1)),
    /* 0xA2 */ None,
    /* 0xA3 */ None,
    /* 0xA4 */ Some(("SetBpm", 1)),
    /* 0xA5 */ Some(("SetBpm2", 1)),
    /* 0xA6 */ None,
    /* 0xA7 */ None,
    /* 0xA8 */ Some(("SetBank", 2)),
    /* 0xA9 */ Some(("SetBankMsb", 1)),
    /* 0xAA */ Some(("SetBankLsb", 1)),
    /* 0xAB */ Some(("Dummy1Byte", 1)),
    /* 0xAC */ Some(("SetInstrument", 1)),
    /* 0xAD */ None,
    /* 0xAE */ None,
    /* 0xAF */ Some(("SongVolumeFade", 3)),
    /* 0xB0 */ Some(("RestoreEnvelopeDefaults", 0)),
    /* 0xB1 */ Some(("SetEnvelopeAttackBegin", 1)),
    /* 0xB2 */ Some(("SetEnvelopeAttackTime", 1)),
    /* 0xB3 */ Some(("SetEnvelopeHoldTime", 1)),
    /* 0xB4 */ Some(("SetEnvelopeDecayTimeAndSustainLevel", 2)),
    /* 0xB5 */ Some(("SetEnvelopeSustainTime", 1)),
    /* 0xB6 */ Some(("SetEnvelopeReleaseTime", 1)),
    /* 0xB7 */ None,
    /* 0xB8 */ None,
    /* 0xB9 */ None,
    /* 0xBA */ None,
    /* 0xBB */ None,
    /* 0xBC */ Some(("SetNoteDurationMultiplier", 1)),
    /* 0xBD */ None,
    /* 0xBE */ Some(("ForceLfoEnvelopeLevel", 1)),
    /* 0xBF */ Some(("SetHoldNotes", 1)),
    /* 0xC0 */ Some(("SetFlagBit1Unknown", 0)),
    /* 0xC1 */ None,
    /* 0xC2 */ None,
    /* 0xC3 */ Some(("SetOptionalVolume", 1)),
    /* 0xC4 */ None,
    /* 0xC5 */ None,
    /* 0xC6 */ None,
    /* 0xC7 */ None,
    /* 0xC8 */ None,
    /* 0xC9 */ None,
    /* 0xCA */ None,
    /* 0xCB */ Some(("Dummy2Bytes", 2)),
    /* 0xCC */ None,
    /* 0xCD */ None,
    /* 0xCE */ None,
    /* 0xCF */ None,
    /* 0xD0 */ Some(("SetTuning", 1)),
    /* 0xD1 */ Some(("TuningDeltaCoarse", 1)),
    /* 0xD2 */ Some(("TuningDeltaFine", 1)),
    /* 0xD3 */ Some(("TuningDeltaFull", 2)),
    /* 0xD4 */ Some(("TuningFade", 3)),
    /* 0xD5 */ Some(("SetNoteRandomRegion", 2)),
    /* 0xD6 */ Some(("SetTuningJitterAmplitude", 2)),
    /* 0xD7 */ Some(("SetKeyBend", 2)),
    /* 0xD8 */ Some(("SetUnknown2", 2)),
    /* 0xD9 */ None,
    /* 0xDA */ None,
    /* 0xDB */ Some(("SetKeyBendRange", 1)),
    /* 0xDC */ Some(("SetupKeyBendLfo", 5)),
    /* 0xDD */ Some(("SetupKeyBendLfoEnvelope", 4)),
    /* 0xDE */ None,
    /* 0xDF */ Some(("UseKeyBendLfo", 1)),
    /* 0xE0 */ Some(("SetVolume", 1)),
    /* 0xE1 */ Some(("VolumeDelta", 1)),
    /* 0xE2 */ Some(("VolumeFade", 3)),
    /* 0xE3 */ Some(("SetExpression", 1)),
    /* 0xE4 */ Some(("SetupVolumeLfo", 5)),
    /* 0xE5 */ Some(("SetupVolumeLfoEnvelope", 4)),
    /* 0xE6 */ None,
    /* 0xE7 */ Some(("UseVolumeLfo", 1)),
    /* 0xE8 */ Some(("SetPan", 1)),
    /* 0xE9 */ Some(("PanDelta", 1)),
    /* 0xEA */ Some(("PanFade", 3)),
    /* 0xEB */ None,
    /* 0xEC */ Some(("SetupPanLfo", 5)),
    /* 0xED */ Some(("SetupPanLfoEnvelope", 4)),
    /* 0xEE */ None,
    /* 0xEF */ Some(("UsePanLfo", 1)),
    /* 0xF0 */ Some(("SetupLfo", 5)),
    /* 0xF1 */ Some(("SetupLfoEnvelope", 4)),
    /* 0xF2 */ Some(("SetLfoParameter", 2)),
    /* 0xF3 */ Some(("UseLfo", 3)),
    /* 0xF4 */ None,
    /* 0xF5 */ None,
    /* 0xF6 */ Some(("Signal", 1)),
    /* 0xF7 */ None,
    /* 0xF8 */ Some(("Dummy2Bytes2", 2)),
    /* 0xF9 */ None,
    /* 0xFA */ None,
    /* 0xFB */ None,
    /* 0xFC */ None,
    /* 0xFD */ None,
    /* 0xFE */ None,
    /* 0xFF */ None,
];

/// Looks up a control opcode (0x90–0xFF): `Some((name, operand_count))`, or `None` if invalid.
pub fn control_info(opcode: u8) -> Option<(&'static str, u8)> {
    if opcode < 0x90 {
        return None;
    }
    CONTROL_TABLE[(opcode - 0x90) as usize]
}

/// Decodes one track's raw event bytes into a flat [`DseEvent`] list.
///
/// `start_octave` seeds the running octave that `PlayNote`/`SetOctave`/`OctaveDelta` mutate, so
/// each note's absolute key is resolved exactly as the driver does in `ParseDseEvent`. Decoding
/// stops after `EndTrack` (0x98) or when the bytes run out.
pub fn decode_track(bytes: &[u8], start_octave: i32) -> Vec<DseEvent> {
    let mut events = Vec::new();
    let mut pos = 0usize;
    let mut octave = start_octave;

    while pos < bytes.len() {
        let opcode = bytes[pos];
        pos += 1;

        if opcode < 0x80 {
            // PlayNote: opcode is velocity; the next byte packs param count, octave mod, note.
            let Some(&notedata) = bytes.get(pos) else {
                break;
            };
            pos += 1;
            let nb_params = (notedata >> 6) & 0x3;
            let octave_delta = (((notedata >> 4) & 0x3) as i8) - 2;
            let note = notedata & 0x0F;
            octave += octave_delta as i32;
            let key = (octave * 12 + note as i32).clamp(0, 127) as u8;

            let duration = if nb_params == 0 {
                None
            } else {
                // Big-endian, `nb_params` bytes (1–3).
                let mut d = 0u32;
                for _ in 0..nb_params {
                    let Some(&b) = bytes.get(pos) else { break };
                    pos += 1;
                    d = (d << 8) | b as u32;
                }
                Some(d)
            };
            events.push(DseEvent::Note {
                velocity: opcode,
                key,
                octave_delta,
                duration,
            });
        } else if opcode < 0x90 {
            events.push(DseEvent::Pause {
                ticks: PAUSE_TICKS[(opcode - 0x80) as usize],
            });
        } else {
            match control_info(opcode) {
                Some((name, n)) => {
                    let end = (pos + n as usize).min(bytes.len());
                    let operands = bytes[pos..end].to_vec();
                    pos = end;
                    // Track the running octave so following notes resolve correctly.
                    match opcode {
                        0xA0 => octave = operands.first().copied().unwrap_or(0) as i32, // SetOctave
                        0xA1 => octave += operands.first().copied().unwrap_or(0) as i8 as i32, // OctaveDelta
                        0x98 => {
                            events.push(DseEvent::Control {
                                opcode,
                                name,
                                operands,
                            });
                            break;
                        }
                        _ => {}
                    }
                    events.push(DseEvent::Control {
                        opcode,
                        name,
                        operands,
                    });
                }
                None => events.push(DseEvent::Invalid { opcode }),
            }
        }
    }
    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pause_table_matches_decomp() {
        // _020B0B7C in asm/main_rodata_020A2808.s
        assert_eq!(PAUSE_TICKS[0], 96); // 0x80 half note
        assert_eq!(PAUSE_TICKS[2], 64); // 0x82 half-note triplet
        assert_eq!(PAUSE_TICKS[15], 2); // 0x8F 64th note
    }

    #[test]
    fn control_table_spot_checks() {
        // Named handlers from SMD_EVENTS_FUN_TABLE with asm-verified operand counts.
        assert_eq!(control_info(0x98), Some(("EndTrack", 0)));
        assert_eq!(control_info(0x99), Some(("MainLoopBegin", 0)));
        assert_eq!(control_info(0xA4), Some(("SetBpm", 1)));
        assert_eq!(control_info(0xA8), Some(("SetBank", 2))); // reads 2 bytes
        assert_eq!(control_info(0xAC), Some(("SetInstrument", 1)));
        assert_eq!(control_info(0xAF), Some(("SongVolumeFade", 3)));
        assert_eq!(control_info(0xE0), Some(("SetVolume", 1)));
        assert_eq!(control_info(0xF0), Some(("SetupLfo", 5)));
        assert_eq!(control_info(0x96), None); // DseTrackEvent_Invalid slot
        assert_eq!(control_info(0x7F), None); // below the control range
    }

    #[test]
    fn decodes_playnote_octave_and_duration() {
        // velocity=0x7F, notedata: nb_params=1 (bits 7-6=01), octavemod=2 (bits 5-4=00 -> -2),
        // note=0 (C). Then one duration byte 0x30 (48 ticks).
        // 0x40 = 0b01_00_0000.
        let ev = decode_track(&[0x7F, 0x40, 0x30], 5);
        assert_eq!(
            ev[0],
            DseEvent::Note {
                velocity: 0x7F,
                key: ((5 - 2) * 12) as u8, // octave dropped by 2 -> 3*12 = 36
                octave_delta: -2,
                duration: Some(0x30),
            }
        );
    }

    #[test]
    fn pause_and_endtrack_terminate() {
        let ev = decode_track(&[0x83, 0x98, 0x81 /* unreached */], 4);
        assert_eq!(ev[0], DseEvent::Pause { ticks: 48 });
        assert!(matches!(ev[1], DseEvent::Control { opcode: 0x98, .. }));
        assert_eq!(ev.len(), 2, "decoding stops at EndTrack");
    }

    #[test]
    fn set_octave_then_note_uses_new_octave() {
        // SetOctave(0xA0) to 6, then a note with octavemod 0 (=-2... use bits to get +0)
        // notedata 0x20 = 0b00_10_0000 -> nb_params=0, octavemod=(2-2)=0, note=0.
        let ev = decode_track(&[0xA0, 0x06, 0x40, 0x20], 0);
        assert!(matches!(ev[0], DseEvent::Control { opcode: 0xA0, .. }));
        assert_eq!(
            ev[1],
            DseEvent::Note {
                velocity: 0x40,
                key: 6 * 12,
                octave_delta: 0,
                duration: None,
            }
        );
    }
}

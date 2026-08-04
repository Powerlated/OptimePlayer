//! Decodes SMDL track bytecode into events, following the opcode and operand table of the game's parser.

pub const PAUSE_TICKS: [u16; 16] = [96, 72, 64, 48, 36, 32, 24, 18, 16, 12, 9, 8, 6, 4, 3, 2];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DseEvent {
    Note {
        velocity: u8,
        key: u8,
        octave_delta: i8,
        duration: Option<u32>,
    },
    Pause {
        ticks: u16,
    },
    Control {
        opcode: u8,
        name: &'static str,
        operands: Vec<u8>,
    },
    Invalid {
        opcode: u8,
    },
}

#[rustfmt::skip]
const CONTROL_TABLE: [Option<(&str, u8)>; 112] = [
     Some(("WaitSame", 0)),
     Some(("WaitDelta", 1)),
     Some(("Wait8", 1)),
     Some(("Wait16", 2)),
     Some(("Wait24", 3)),
     Some(("WaitUntilFadeout", 1)),
     None,
     None,
     Some(("EndTrack", 0)),
     Some(("MainLoopBegin", 0)),
     None,
     None,
     Some(("SubLoopBegin", 1)),
     Some(("SubLoopEnd", 0)),
     Some(("SubLoopBreakOnLastIteration", 0)),
     None,
     Some(("SetOctave", 1)),
     Some(("OctaveDelta", 1)),
     None,
     None,
     Some(("SetBpm", 1)),
     Some(("SetBpm2", 1)),
     None,
     None,
     Some(("SetBank", 2)),
     Some(("SetBankMsb", 1)),
     Some(("SetBankLsb", 1)),
     Some(("Dummy1Byte", 1)),
     Some(("SetInstrument", 1)),
     None,
     None,
     Some(("SongVolumeFade", 3)),
     Some(("RestoreEnvelopeDefaults", 0)),
     Some(("SetEnvelopeAttackBegin", 1)),
     Some(("SetEnvelopeAttackTime", 1)),
     Some(("SetEnvelopeHoldTime", 1)),
     Some(("SetEnvelopeDecayTimeAndSustainLevel", 2)),
     Some(("SetEnvelopeSustainTime", 1)),
     Some(("SetEnvelopeReleaseTime", 1)),
     None,
     None,
     None,
     None,
     None,
     Some(("SetNoteDurationMultiplier", 1)),
     None,
     Some(("ForceLfoEnvelopeLevel", 1)),
     Some(("SetHoldNotes", 1)),
     Some(("SetFlagBit1Unknown", 0)),
     None,
     None,
     Some(("SetOptionalVolume", 1)),
     None,
     None,
     None,
     None,
     None,
     None,
     None,
     Some(("Dummy2Bytes", 2)),
     None,
     None,
     None,
     None,
     Some(("SetTuning", 1)),
     Some(("TuningDeltaCoarse", 1)),
     Some(("TuningDeltaFine", 1)),
     Some(("TuningDeltaFull", 2)),
     Some(("TuningFade", 3)),
     Some(("SetNoteRandomRegion", 2)),
     Some(("SetTuningJitterAmplitude", 2)),
     Some(("SetKeyBend", 2)),
     Some(("SetUnknown2", 2)),
     None,
     None,
     Some(("SetKeyBendRange", 1)),
     Some(("SetupKeyBendLfo", 5)),
     Some(("SetupKeyBendLfoEnvelope", 4)),
     None,
     Some(("UseKeyBendLfo", 1)),
     Some(("SetVolume", 1)),
     Some(("VolumeDelta", 1)),
     Some(("VolumeFade", 3)),
     Some(("SetExpression", 1)),
     Some(("SetupVolumeLfo", 5)),
     Some(("SetupVolumeLfoEnvelope", 4)),
     None,
     Some(("UseVolumeLfo", 1)),
     Some(("SetPan", 1)),
     Some(("PanDelta", 1)),
     Some(("PanFade", 3)),
     None,
     Some(("SetupPanLfo", 5)),
     Some(("SetupPanLfoEnvelope", 4)),
     None,
     Some(("UsePanLfo", 1)),
     Some(("SetupLfo", 5)),
     Some(("SetupLfoEnvelope", 4)),
     Some(("SetLfoParameter", 2)),
     Some(("UseLfo", 3)),
     None,
     None,
     Some(("Signal", 1)),
     None,
     Some(("Dummy2Bytes2", 2)),
     None,
     None,
     None,
     None,
     None,
     None,
     None,
];

pub fn control_info(opcode: u8) -> Option<(&'static str, u8)> {
    if opcode < 0x90 {
        return None;
    }
    CONTROL_TABLE[(opcode - 0x90) as usize]
}

pub fn decode_track(bytes: &[u8], start_octave: i32) -> Vec<DseEvent> {
    let mut events = Vec::new();
    let mut pos = 0usize;
    let mut octave = start_octave;

    while pos < bytes.len() {
        let opcode = bytes[pos];
        pos += 1;

        if opcode < 0x80 {
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
                    match opcode {
                        0xA0 => octave = operands.first().copied().unwrap_or(0) as i32,
                        0xA1 => octave += operands.first().copied().unwrap_or(0) as i8 as i32,
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
        assert_eq!(PAUSE_TICKS[0], 96);
        assert_eq!(PAUSE_TICKS[2], 64);
        assert_eq!(PAUSE_TICKS[15], 2);
    }

    #[test]
    fn control_table_spot_checks() {
        assert_eq!(control_info(0x98), Some(("EndTrack", 0)));
        assert_eq!(control_info(0x99), Some(("MainLoopBegin", 0)));
        assert_eq!(control_info(0xA4), Some(("SetBpm", 1)));
        assert_eq!(control_info(0xA8), Some(("SetBank", 2)));
        assert_eq!(control_info(0xAC), Some(("SetInstrument", 1)));
        assert_eq!(control_info(0xAF), Some(("SongVolumeFade", 3)));
        assert_eq!(control_info(0xE0), Some(("SetVolume", 1)));
        assert_eq!(control_info(0xF0), Some(("SetupLfo", 5)));
        assert_eq!(control_info(0x96), None);
        assert_eq!(control_info(0x7F), None);
    }

    #[test]
    fn decodes_playnote_octave_and_duration() {
        let ev = decode_track(&[0x7F, 0x40, 0x30], 5);
        assert_eq!(
            ev[0],
            DseEvent::Note {
                velocity: 0x7F,
                key: ((5 - 2) * 12) as u8,
                octave_delta: -2,
                duration: Some(0x30),
            }
        );
    }

    #[test]
    fn pause_and_endtrack_terminate() {
        let ev = decode_track(&[0x83, 0x98, 0x81], 4);
        assert_eq!(ev[0], DseEvent::Pause { ticks: 48 });
        assert!(matches!(ev[1], DseEvent::Control { opcode: 0x98, .. }));
        assert_eq!(ev.len(), 2, "decoding stops at EndTrack");
    }

    #[test]
    fn set_octave_then_note_uses_new_octave() {
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

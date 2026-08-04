//! Tests for the SSEQ interpreter, driven by hand-assembled bytecode.

use super::{Message, MessageType, Sequence};
use crate::TRACK_COUNT;
use std::sync::Arc;

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
    let mut s = seq(&[0x3C, 0x7F, 0x10, 0x80, 0x04, 0xFF]);
    s.tick(&[false; TRACK_COUNT]);
    let msgs = drain(&mut s);
    assert_eq!(msgs.len(), 1);
    assert_eq!(
        msgs[0].msg_type,
        MessageType::PlayNote {
            note: 60,
            velocity: 127,
            duration: 0x10,
        }
    );
    assert_eq!(s.tracks[0].resting_for, 0x10 - 1);
}

#[test]
fn note_wait_off_fires_notes_back_to_back() {
    let mut s = seq(&[
        0xC7, 0x00, 0x3C, 0x7F, 0x20, 0x40, 0x7F, 0x20, 0x80, 0x04, 0xFF,
    ]);
    s.tick(&[false; TRACK_COUNT]);
    let msgs = drain(&mut s);
    let notes: Vec<i32> = msgs
        .iter()
        .filter_map(|m| match m.msg_type {
            MessageType::PlayNote { note, .. } => Some(note),
            _ => None,
        })
        .collect();
    assert_eq!(
        notes,
        vec![60, 64],
        "both notes should fire on the same tick"
    );
    assert_eq!(s.tracks[0].resting_for, 3);
}

#[test]
fn note_wait_on_advances_by_note_duration() {
    let mut s = seq(&[0x3C, 0x7F, 0x08, 0x40, 0x7F, 0x08, 0xFF]);
    s.tick(&[false; TRACK_COUNT]);
    let notes: Vec<i32> = drain(&mut s)
        .iter()
        .filter_map(|m| match m.msg_type {
            MessageType::PlayNote { note, .. } => Some(note),
            _ => None,
        })
        .collect();
    assert_eq!(notes, vec![60], "only the first note should have fired");
    assert_eq!(s.tracks[0].resting_for, 7);
}

#[test]
fn transpose_shifts_note_keys() {
    let mut s = seq(&[0xC3, 12, 0x3C, 0x7F, 0x04, 0xFF]);
    s.tick(&[false; TRACK_COUNT]);
    let msgs = drain(&mut s);
    assert!(matches!(
        msgs[0].msg_type,
        MessageType::PlayNote { note: 72, .. }
    ));
}

#[test]
fn loop_start_end_repeats_the_body() {
    let mut s = seq(&[0xC7, 0x00, 0xD4, 0x02, 0xC0, 70, 0x80, 0x01, 0xFC, 0xFF]);
    s.tick(&[false; TRACK_COUNT]);
    assert_eq!(s.tracks[0].pan, 70);
    assert_eq!(s.tracks[0].sp, 1, "loop frame pushed");
    s.tick(&[false; TRACK_COUNT]);
    assert_eq!(s.tracks[0].sp, 1, "still looping on the second iteration");
    s.tick(&[false; TRACK_COUNT]);
    assert_eq!(
        s.tracks[0].sp, 0,
        "loop frame popped after the final iteration"
    );
    assert!(!s.tracks[0].active, "track ended after the loop");
}

#[test]
fn conditional_prefix_gates_on_compare_flag() {
    let mut s = seq(&[
        0xB0, 0x00, 0x05, 0x00, 0xB8, 0x00, 0x03, 0x00, 0xA2, 0xC0, 70, 0xC0, 40, 0x80, 0x02, 0xFF,
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
    let mut s = seq(&[0x3C, 0x7F, 0x81, 0x00, 0x80, 0x04, 0xFF]);
    s.tick(&[false; TRACK_COUNT]);
    let msgs = drain(&mut s);
    assert!(matches!(
        msgs[0].msg_type,
        MessageType::PlayNote { duration: 128, .. }
    ));
}

#[test]
fn control_opcodes_set_state_and_emit_messages() {
    let mut s = seq(&[
        0xE1, 0xF0, 0x00, 0xC1, 100, 0xC0, 64, 0x81, 0x05, 0x80, 0x04, 0xFF,
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
            MessageType::VolumeChange { volume: 100 },
            MessageType::PanChange { pan: 64 },
            MessageType::InstrumentChange {
                bank: 0,
                program: 5,
            },
        ]
    );
}

#[test]
fn jump_sets_pc_and_emits_jump() {
    let mut s = seq(&[0x94, 0x05, 0x00, 0x00, 0xFF, 0x80, 0x02, 0xFF]);
    s.tick(&[false; TRACK_COUNT]);
    let msgs = drain(&mut s);
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0].msg_type, MessageType::Jump);
    assert_eq!(s.tracks[0].pc, 7);
}

#[test]
fn call_and_return_use_the_stack() {
    let mut s = seq(&[0x95, 0x06, 0x00, 0x00, 0x80, 0x08, 0xC0, 70, 0xFD]);
    s.tick(&[false; TRACK_COUNT]);
    let _ = drain(&mut s);
    assert_eq!(s.tracks[0].pan, 70);
    assert_eq!(s.tracks[0].resting_for, 7);
}

#[test]
fn start_track_activates_another_thread() {
    let mut s = seq(&[0x93, 0x01, 0x07, 0x00, 0x00, 0x80, 0x04, 0x80, 0x02, 0xFF]);
    s.tick(&[false; TRACK_COUNT]);
    assert!(s.tracks[1].active);
    assert_eq!(s.tracks[1].resting_for, 1);
}

//! Opcode-interpreter unit tests, driving small hand-assembled SSEQ byte streams.

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

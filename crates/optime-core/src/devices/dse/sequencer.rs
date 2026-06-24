//! [`DseSequencer`]: the SMDL multi-track bytecode interpreter.
//!
//! Runs all of a song's tracks one **sequencer tick** at a time, exactly as the DSE driver's
//! `ParseDseEvent`/`UpdateSequencerTracks` do (transcribed from `pret/pmd-sky`'s
//! `asm/main_0206C9BC.s`): a track with pending wait-ticks just counts down; otherwise it
//! executes events until a pause/wait sets a new wait. Notes are fire-and-forget (the note's
//! own duration drives its release), so the per-track timeline is advanced solely by pauses.
//!
//! The interpreter is headless: it emits a flat [`SeqOp`] stream that the device player turns
//! into voices. It owns no audio or envelope state.

use super::events::{control_info, PAUSE_TICKS};
use super::smdl::Smdl;
use crate::util::{read_u16, read_u8};

/// Default tempo before any `SetBpm` event (the DSE driver's startup tempo).
const DEFAULT_BPM: u8 = 120;
/// Hard cap on events executed per track per tick — guards against a malformed zero-length loop.
const MAX_EVENTS_PER_TICK: u32 = 100_000;

/// One thing the sequencer did on a tick, for the device player to act on.
#[derive(Debug, Clone, PartialEq)]
pub enum SeqOp {
    /// Start a note on `track` (a DSE channel index, 0..=15).
    NoteOn {
        track: usize,
        key: u8,
        velocity: u8,
        /// Auto-release after this many sequencer ticks; `None` reuses the previous duration but
        /// here always resolved to a concrete value, or `0` if never set.
        duration: u32,
    },
    /// Global tempo change (beats/minute).
    Tempo { bpm: u8 },
    /// Track selected a new program/instrument id.
    Program { track: usize, program: u16 },
    /// Track volume (0..=127).
    Volume { track: usize, volume: u8 },
    /// Track expression / secondary volume (0..=127).
    Expression { track: usize, value: u8 },
    /// Track pan (0=left, 64=center, 127=right).
    Pan { track: usize, pan: u8 },
    /// The track jumped back to its main loop point.
    Looped,
    /// A track reached `EndTrack` with no loop point.
    TrackEnded { track: usize },
}

/// A sub-loop stack frame (`0x9C`/`0x9D`/`0x9E`). `count` is iterations remaining; `octave` is
/// restored on each loop-back (the driver saves/restores it).
#[derive(Debug, Clone, Copy)]
struct SubLoop {
    start: usize,
    end: usize,
    count: u8,
    octave: i32,
}

/// Per-track runtime state.
#[derive(Debug, Clone)]
struct TrackState {
    /// DSE channel index this track plays on (0..=15) — the synth track for its notes.
    channel: usize,
    events: Vec<u8>,
    pos: usize,
    active: bool,
    octave: i32,
    program: u16,
    /// Sequencer ticks still to wait before executing more events.
    wait: u32,
    /// Last pause length, for `RepeatLastPause`/`AddToLastPause`.
    last_pause: u32,
    /// Previous note duration, reused by a `PlayNote` with no duration bytes.
    prev_duration: u32,
    /// Main loop point (`MainLoopBegin`), if set.
    loop_start: Option<usize>,
    loop_stack: Vec<SubLoop>,
}

impl TrackState {
    fn new(channel: usize, events: Vec<u8>) -> Self {
        TrackState {
            channel,
            events,
            pos: 0,
            active: true,
            octave: 4,
            program: 0,
            wait: 0,
            last_pause: 0,
            prev_duration: 0,
            loop_start: None,
            loop_stack: Vec::new(),
        }
    }
}

/// The SMDL interpreter: holds every track's state plus the shared tempo and tick counter.
pub struct DseSequencer {
    tracks: Vec<TrackState>,
    /// Ticks per quarter note (from the SMDL `song` chunk).
    pub tpqn: u16,
    /// Current tempo in beats per minute (sequence-global; any `SetBpm` updates it).
    pub bpm: u8,
    /// Sequencer ticks executed so far (the visualizer timeline).
    pub ticks_elapsed: u32,
    /// Set once every track has ended (no main loop point anywhere).
    pub ended: bool,
}

impl DseSequencer {
    /// Builds an interpreter for `smdl`. The control track (track 0) and the per-channel music
    /// tracks all run; notes are emitted on each track's DSE channel index.
    pub fn new(smdl: &Smdl) -> DseSequencer {
        let tracks = smdl
            .tracks
            .iter()
            .map(|t| TrackState::new(t.channel_id as usize, t.events.clone()))
            .collect();
        DseSequencer {
            tracks,
            tpqn: smdl.tpqn.max(1),
            bpm: DEFAULT_BPM,
            ticks_elapsed: 0,
            ended: false,
        }
    }

    /// Advances exactly one sequencer tick, appending what happened to `ops`.
    pub fn seq_tick(&mut self, ops: &mut Vec<SeqOp>) {
        self.ticks_elapsed += 1;
        let start = ops.len();
        let mut any_active = false;
        for i in 0..self.tracks.len() {
            if !self.tracks[i].active {
                continue;
            }
            any_active = true;
            if self.tracks[i].wait > 0 {
                self.tracks[i].wait -= 1;
                continue;
            }
            self.run_track(i, ops);
            // The tick that started a pause counts as its first tick.
            if self.tracks[i].wait > 0 {
                self.tracks[i].wait -= 1;
            }
        }
        // Tempo is sequence-global: track the latest `SetBpm` this tick produced.
        for op in &ops[start..] {
            if let SeqOp::Tempo { bpm } = op {
                self.bpm = *bpm;
            }
        }
        if !any_active {
            self.ended = true;
        }
    }

    /// Executes events on track `i` until a pause/wait is set or the track ends.
    fn run_track(&mut self, i: usize, ops: &mut Vec<SeqOp>) {
        let mut iters = 0u32;
        loop {
            iters += 1;
            if iters > MAX_EVENTS_PER_TICK {
                self.tracks[i].active = false;
                return;
            }
            let tr = &mut self.tracks[i];
            let Some(&op) = tr.events.get(tr.pos) else {
                tr.active = false;
                ops.push(SeqOp::TrackEnded { track: tr.channel });
                return;
            };
            tr.pos += 1;

            if op < 0x80 {
                self.play_note(i, op, ops);
            } else if op < 0x90 {
                let ticks = PAUSE_TICKS[(op - 0x80) as usize] as u32;
                let tr = &mut self.tracks[i];
                tr.last_pause = ticks;
                tr.wait = ticks;
                return;
            } else if self.control(i, op, ops) {
                // `control` returned true => a wait was set (pause-like) => yield this tick.
                return;
            }
        }
    }

    /// A `PlayNote` (opcode 0x00–0x7F): decode note-data + duration, update octave, emit `NoteOn`.
    fn play_note(&mut self, i: usize, velocity: u8, ops: &mut Vec<SeqOp>) {
        let tr = &mut self.tracks[i];
        let Some(&notedata) = tr.events.get(tr.pos) else {
            tr.active = false;
            return;
        };
        tr.pos += 1;
        let nb_params = (notedata >> 6) & 0x3;
        let octave_delta = (((notedata >> 4) & 0x3) as i8) - 2;
        let note = (notedata & 0x0F) as i32;
        tr.octave += octave_delta as i32;
        let key = (tr.octave * 12 + note).clamp(0, 127) as u8;

        if nb_params > 0 {
            let mut d = 0u32;
            for _ in 0..nb_params {
                let Some(&b) = tr.events.get(tr.pos) else {
                    break;
                };
                tr.pos += 1;
                d = (d << 8) | b as u32;
            }
            tr.prev_duration = d;
        }
        let duration = tr.prev_duration;
        let track = tr.channel;
        ops.push(SeqOp::NoteOn {
            track,
            key,
            velocity,
            duration,
        });
    }

    /// Handles a control opcode (0x90–0xFF). Returns `true` if it set a wait (the track should
    /// yield this tick), `false` to keep executing. Unhandled opcodes consume their operands.
    fn control(&mut self, i: usize, op: u8, ops: &mut Vec<SeqOp>) -> bool {
        let tr = &mut self.tracks[i];
        let track = tr.channel;
        // Helper closures can't borrow `tr` mutably and read events; read operands inline.
        match op {
            0x90 => {
                // WaitSame / RepeatLastPause
                tr.wait = tr.last_pause;
                return true;
            }
            0x91 => {
                // WaitDelta / AddToLastPause (signed)
                let delta = tr.events.get(tr.pos).copied().unwrap_or(0) as i8;
                tr.pos += 1;
                tr.last_pause = (tr.last_pause as i64 + delta as i64).max(0) as u32;
                tr.wait = tr.last_pause;
                return true;
            }
            0x92 => {
                let v = read_u8(&tr.events, tr.pos) as u32;
                tr.pos += 1;
                tr.last_pause = v;
                tr.wait = v;
                return true;
            }
            0x93 => {
                let v = read_u16(&tr.events, tr.pos) as u32;
                tr.pos += 2;
                tr.last_pause = v;
                tr.wait = v;
                return true;
            }
            0x94 => {
                let lo = read_u16(&tr.events, tr.pos) as u32;
                let hi = read_u8(&tr.events, tr.pos + 2) as u32;
                tr.pos += 3;
                let v = lo | (hi << 16);
                tr.last_pause = v;
                tr.wait = v;
                return true;
            }
            0x95 => {
                // WaitUntilFadeout: approximate as a fixed wait of its operand.
                let v = read_u8(&tr.events, tr.pos) as u32;
                tr.pos += 1;
                tr.wait = v;
                return true;
            }
            0x98 => {
                // EndTrack: loop back if a main loop point exists, else end.
                if let Some(ls) = tr.loop_start {
                    tr.pos = ls;
                    ops.push(SeqOp::Looped);
                } else {
                    tr.active = false;
                    ops.push(SeqOp::TrackEnded { track });
                }
            }
            0x99 => tr.loop_start = Some(tr.pos), // MainLoopBegin
            0x9C => {
                // SubLoopBegin
                let count = read_u8(&tr.events, tr.pos);
                tr.pos += 1;
                let frame = SubLoop {
                    start: tr.pos,
                    end: tr.pos,
                    count,
                    octave: tr.octave,
                };
                if tr.loop_stack.len() < 4 {
                    tr.loop_stack.push(frame);
                }
            }
            0x9D => {
                // SubLoopEnd
                if let Some(top) = tr.loop_stack.last_mut() {
                    top.count = top.count.wrapping_sub(1);
                    if top.count == 0 {
                        tr.loop_stack.pop();
                    } else {
                        top.end = tr.pos;
                        let (start, octave) = (top.start, top.octave);
                        tr.pos = start;
                        tr.octave = octave;
                    }
                }
            }
            0x9E => {
                // SubLoopBreakOnLastIteration
                if let Some(top) = tr.loop_stack.last().copied() {
                    if top.count == 1 {
                        tr.pos = top.end;
                        tr.loop_stack.pop();
                    }
                }
            }
            0xA0 => {
                tr.octave = read_u8(&tr.events, tr.pos) as i32;
                tr.pos += 1;
            }
            0xA1 => {
                tr.octave += read_u8(&tr.events, tr.pos) as i8 as i32;
                tr.pos += 1;
            }
            0xA4 | 0xA5 => {
                let bpm = read_u8(&tr.events, tr.pos);
                tr.pos += 1;
                ops.push(SeqOp::Tempo { bpm });
            }
            0xAC => {
                let program = read_u8(&tr.events, tr.pos) as u16;
                tr.pos += 1;
                tr.program = program;
                ops.push(SeqOp::Program { track, program });
            }
            0xE0 => {
                let volume = read_u8(&tr.events, tr.pos);
                tr.pos += 1;
                ops.push(SeqOp::Volume { track, volume });
            }
            0xE3 => {
                let value = read_u8(&tr.events, tr.pos);
                tr.pos += 1;
                ops.push(SeqOp::Expression { track, value });
            }
            0xE8 => {
                let pan = read_u8(&tr.events, tr.pos);
                tr.pos += 1;
                ops.push(SeqOp::Pan { track, pan });
            }
            _ => {
                // Any other control opcode: consume its operand bytes and ignore it.
                let n = control_info(op).map(|(_, n)| n as usize).unwrap_or(0);
                tr.pos += n;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::dse::smdl::Smdl;

    /// Wraps raw track-event bytes into a one-track SMDL for testing.
    fn smdl_one_track(events: &[u8]) -> Smdl {
        // Header (0x40) + song (0x40) + trk + eoc.
        let mut d = vec![0u8; 0x40];
        d[0..4].copy_from_slice(b"smdl");
        let mut song = vec![0u8; 0x40];
        song[0..4].copy_from_slice(b"song");
        song[0x12..0x14].copy_from_slice(&48u16.to_le_bytes());
        d.extend_from_slice(&song);
        let mut trk = Vec::new();
        trk.extend_from_slice(b"trk ");
        trk.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);
        let payload_len = 4 + events.len();
        trk.extend_from_slice(&(payload_len as u32).to_le_bytes());
        trk.extend_from_slice(&[1, 0, 0, 0]); // track_id=1, channel_id=0
        trk.extend_from_slice(events);
        while trk.len() % 4 != 0 {
            trk.push(0x98);
        }
        d.extend_from_slice(&trk);
        d.extend_from_slice(b"eoc\0");
        d.extend_from_slice(&[0u8; 12]);
        Smdl::parse(&d).unwrap()
    }

    fn run(events: &[u8], ticks: u32) -> Vec<SeqOp> {
        let smdl = smdl_one_track(events);
        let mut seq = DseSequencer::new(&smdl);
        let mut ops = Vec::new();
        for _ in 0..ticks {
            seq.seq_tick(&mut ops);
        }
        ops
    }

    #[test]
    fn tempo_program_and_note() {
        // SetBpm 150, SetInstrument 5, SetOctave 6, note (C, dur 1 byte = 48), pause quarter.
        // notedata 0x40 = nb_params=1, octavemod=0(-2)... use 0x60 => nb_params=1, octave +0.
        let ops = run(
            &[0xA4, 150, 0xAC, 5, 0xA0, 6, 0x7F, 0x60, 48, 0x83, 0x98],
            1,
        );
        assert!(ops.contains(&SeqOp::Tempo { bpm: 150 }));
        assert!(ops.contains(&SeqOp::Program {
            track: 0,
            program: 5
        }));
        assert!(ops.iter().any(|o| matches!(
            o,
            SeqOp::NoteOn {
                track: 0,
                key: 72,
                velocity: 127,
                duration: 48
            }
        )));
    }

    #[test]
    fn pause_consumes_ticks() {
        // A quarter-note pause (0x83 = 48 ticks) then a note. The note must not fire for 48 ticks.
        let events = [0x83, 0x7F, 0x60, 12, 0x98];
        let after_10 = run(&events, 10);
        assert!(
            !after_10.iter().any(|o| matches!(o, SeqOp::NoteOn { .. })),
            "note fired during the pause"
        );
        let after_50 = run(&events, 50);
        assert!(after_50.iter().any(|o| matches!(o, SeqOp::NoteOn { .. })));
    }

    #[test]
    fn main_loop_repeats() {
        // MainLoopBegin, note, quarter pause, EndTrack -> should loop forever (never TrackEnded).
        let ops = run(&[0x99, 0x7F, 0x60, 12, 0x83, 0x98], 200);
        let notes = ops
            .iter()
            .filter(|o| matches!(o, SeqOp::NoteOn { .. }))
            .count();
        assert!(
            notes >= 3,
            "main loop should replay the note repeatedly, got {notes}"
        );
        assert!(ops.contains(&SeqOp::Looped));
        assert!(!ops.iter().any(|o| matches!(o, SeqOp::TrackEnded { .. })));
    }

    #[test]
    fn sub_loop_repeats_body_n_times() {
        // SubLoopBegin(3), note, quarter pause, SubLoopEnd, EndTrack.
        // Body should play 3 times then end.
        let ops = run(&[0x9C, 3, 0x7F, 0x60, 12, 0x83, 0x9D, 0x98], 500);
        let notes = ops
            .iter()
            .filter(|o| matches!(o, SeqOp::NoteOn { .. }))
            .count();
        assert_eq!(notes, 3, "sub-loop body should play exactly 3x");
        assert!(ops.iter().any(|o| matches!(o, SeqOp::TrackEnded { .. })));
    }
}

//! Note-event tokenization shared by the learned-token backbones
//! ([`crate::m01_event`], [`crate::m02_hier`]).
//!
//! Turns a [`Song`]'s note events into per-note bucketed field indices (pitch,
//! pitch-class, channel, velocity, pan, duration, role) on the 128-frame window
//! grid. How those per-note fields get pooled into a frame token is what
//! distinguishes the generations, so pooling lives in each generation's `batch.rs` —
//! only the per-note representation and the AR targets are shared here.
//!
//! This is the learned counterpart to the hand-engineered per-frame feature grid
//! ([`crate::features`]) — same 128-frame timeline, embeddings instead of fixed
//! scalars, plus the per-note **channel** (the only instrument cue that survives
//! harvesting).

use crate::notes::{Instrument, Song};

/// Channel/track vocabulary (0..15).
pub const N_CHANNELS: usize = 16;
/// Instrument-role vocabulary (`Instrument::COUNT`).
pub const N_ROLES: usize = Instrument::COUNT;
/// Pitch-class count.
pub const N_PC: usize = 12;
/// Velocity / pan / duration bucket counts.
pub const VEL_BUCKETS: usize = 16;
pub const PAN_BUCKETS: usize = 7;
pub const DUR_BUCKETS: usize = 16;

/// Velocity in [0,1] → bucket `[0, VEL_BUCKETS)`.
pub fn vel_bucket(v: f32) -> usize {
    (v.clamp(0.0, 1.0) * (VEL_BUCKETS - 1) as f32).round() as usize
}
/// Signed pan in [-1,1] → bucket `[0, PAN_BUCKETS)`.
pub fn pan_bucket(p: f32) -> usize {
    (((p.clamp(-1.0, 1.0) + 1.0) * 0.5) * (PAN_BUCKETS - 1) as f32).round() as usize
}
/// Duration in frames → log2 bucket `[0, DUR_BUCKETS)` (1→0, 2-3→1, 4-7→2, …).
pub fn dur_bucket(frames: u32) -> usize {
    (frames.max(1) as f32)
        .log2()
        .floor()
        .min((DUR_BUCKETS - 1) as f32) as usize
}

/// One note's field indices (embedding lookups) plus its exact span (used to
/// reconstruct per-frame sounding content).
#[derive(Clone, Copy)]
pub struct NoteToken {
    pub pitch: i64,
    pub pc: i64,
    pub channel: i64,
    pub vel: i64,
    pub pan: i64,
    pub dur: i64,
    pub role: i64,
    pub onset: u32,
    pub dur_frames: u32,
}

/// A tokenized song window: note tokens + per-frame supervised labels.
pub struct EventExample {
    pub n_frames: usize,
    pub tokens: Vec<NoteToken>,
    pub chord_labels: Vec<usize>,
    pub key_label: usize,
}

impl EventExample {
    pub fn from_song(song: &Song) -> EventExample {
        let nf = song.n_frames;
        let mut tokens = Vec::with_capacity(song.notes.len());
        for n in &song.notes {
            let onset = n.start_frame;
            if (onset as usize) >= nf {
                continue;
            }
            let dur_frames = n.end_frame.saturating_sub(n.start_frame).max(1);
            tokens.push(NoteToken {
                pitch: (n.pitch as i64).clamp(0, 127),
                pc: (n.pitch as i64).rem_euclid(12),
                channel: (n.track as i64).clamp(0, (N_CHANNELS - 1) as i64),
                vel: vel_bucket(n.velocity) as i64,
                pan: pan_bucket(n.pan) as i64,
                dur: dur_bucket(dur_frames) as i64,
                role: n.instrument.index() as i64,
                onset,
                dur_frames,
            });
        }
        EventExample {
            n_frames: nf,
            tokens,
            chord_labels: song.chord_labels.clone(),
            key_label: song.key_label,
        }
    }

    /// Per-frame **sounding** content multi-hots: `pc[nf*N_PC]` and
    /// `ch[nf*N_CHANNELS]`, 1.0 where any note of that pitch-class / channel is
    /// sounding in the frame (onset..onset+duration).
    pub fn frame_content(&self) -> (Vec<f32>, Vec<f32>) {
        let nf = self.n_frames;
        let mut pc = vec![0.0f32; nf * N_PC];
        let mut ch = vec![0.0f32; nf * N_CHANNELS];
        for t in &self.tokens {
            let end = ((t.onset + t.dur_frames) as usize).min(nf);
            for f in (t.onset as usize)..end {
                pc[f * N_PC + t.pc as usize] = 1.0;
                ch[f * N_CHANNELS + t.channel as usize] = 1.0;
            }
        }
        (pc, ch)
    }
}

/// Tokenize a slice of songs.
pub fn examples(songs: &[Song]) -> Vec<EventExample> {
    songs.iter().map(EventExample::from_song).collect()
}

/// AR next-frame targets (`pc[batch*nf*N_PC]`, `ch[batch*nf*N_CHANNELS]`): each
/// example's per-frame sounding content, concatenated. Pooling-agnostic — both
/// learned-token generations predict the same thing, so both share this.
pub fn ar_targets(examples: &[EventExample]) -> (Vec<f32>, Vec<f32>) {
    let mut pc = Vec::new();
    let mut ch = Vec::new();
    for ex in examples {
        let (p, c) = ex.frame_content();
        pc.extend_from_slice(&p);
        ch.extend_from_slice(&c);
    }
    (pc, ch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notes::NoteEvent;

    fn song() -> Song {
        Song {
            key_label: 0,
            n_frames: 8,
            notes: vec![
                NoteEvent {
                    start_frame: 0,
                    end_frame: 4,
                    pitch: 60,
                    velocity: 1.0,
                    instrument: Instrument::Bass,
                    track: 5,
                    pan: 0.0,
                },
                NoteEvent {
                    start_frame: 2,
                    end_frame: 3,
                    pitch: 67,
                    velocity: 0.5,
                    instrument: Instrument::Melody,
                    track: 3,
                    pan: 0.0,
                },
            ],
            chord_labels: vec![0; 8],
            is_music: None,
        }
    }

    #[test]
    fn buckets_are_in_range() {
        for &v in &[0.0, 0.5, 1.0, 1.5, -0.1] {
            assert!(vel_bucket(v) < VEL_BUCKETS);
        }
        for &p in &[-1.5, -1.0, 0.0, 0.5, 1.0, 2.0] {
            assert!(pan_bucket(p) < PAN_BUCKETS);
        }
        for f in [1u32, 2, 3, 4, 7, 8, 127, 128, 100_000] {
            assert!(dur_bucket(f) < DUR_BUCKETS);
        }
        // Log2 duration edges.
        assert_eq!(dur_bucket(1), 0);
        assert_eq!(dur_bucket(2), 1);
        assert_eq!(dur_bucket(4), 2);
        assert_eq!(dur_bucket(8), 3);
    }

    #[test]
    fn tokenizes_onsets_and_channel() {
        let ex = EventExample::from_song(&song());
        assert_eq!(ex.tokens.len(), 2);
        // Channel carried through.
        assert_eq!(ex.tokens[0].channel, 5);
        assert_eq!(ex.tokens[1].channel, 3);
        // Sounding content: frame 0..4 has pc 0 (C), frame 2 also has pc 7 (G).
        let (pc, _ch) = ex.frame_content();
        assert_eq!(pc[0], 1.0); // frame 0, pc 0 (C)
        assert_eq!(pc[2 * N_PC + 7], 1.0); // frame 2, pc 7 (G)
        assert_eq!(pc[4 * N_PC], 0.0); // note ended at frame 4 (exclusive)
    }
}

//! Generation 01's pooling: each frame token is a **scatter-add sum** of the notes
//! *onsetting* in that frame.
//!
//! The batch holds per-field index arrays over a flat list of onset tokens plus each
//! token's destination frame row; the model embeds the fields and scatter-adds them
//! into their row ([`super::model`]). Sustain is carried by the per-note `dur` field
//! rather than by repeating a note across frames — contrast [`crate::m02_hier`],
//! which pools the notes *sounding* in each frame.

use crate::notes::Song;
use crate::tokenize::{self, EventExample};

/// CPU-side flattened batch: per-field onset-token index arrays (`n_total` long),
/// each token's destination row `batch*n_frames + onset_frame` for the scatter-add
/// pool, plus flattened supervised labels.
pub struct EventBatchData {
    pub batch: usize,
    pub n_frames: usize,
    pub n_total: usize,
    pub pitch: Vec<i64>,
    pub pc: Vec<i64>,
    pub channel: Vec<i64>,
    pub vel: Vec<i64>,
    pub pan: Vec<i64>,
    pub dur: Vec<i64>,
    pub role: Vec<i64>,
    /// Destination frame-slot row for each token (`b*n_frames + onset`).
    pub target_row: Vec<i64>,
    /// `batch*n_frames` joint chord labels.
    pub chord_labels: Vec<usize>,
    /// `batch` key labels.
    pub key_labels: Vec<usize>,
    /// Retained so the AR pretext can rebuild per-frame sounding content.
    pub examples: Vec<EventExample>,
}

impl EventBatchData {
    /// Build from songs (all must share `n_frames`).
    pub fn build(songs: &[Song]) -> EventBatchData {
        Self::from_examples(tokenize::examples(songs))
    }

    /// Build from already-tokenized examples.
    pub fn from_examples(examples: Vec<EventExample>) -> EventBatchData {
        let batch = examples.len();
        let nf = examples.first().map(|e| e.n_frames).unwrap_or(0);
        let cap: usize = examples.iter().map(|e| e.tokens.len()).sum();
        let mut d = EventBatchData {
            batch,
            n_frames: nf,
            n_total: cap,
            pitch: Vec::with_capacity(cap),
            pc: Vec::with_capacity(cap),
            channel: Vec::with_capacity(cap),
            vel: Vec::with_capacity(cap),
            pan: Vec::with_capacity(cap),
            dur: Vec::with_capacity(cap),
            role: Vec::with_capacity(cap),
            target_row: Vec::with_capacity(cap),
            chord_labels: Vec::with_capacity(batch * nf),
            key_labels: Vec::with_capacity(batch),
            examples: Vec::new(),
        };
        for (bi, ex) in examples.iter().enumerate() {
            for t in &ex.tokens {
                d.pitch.push(t.pitch);
                d.pc.push(t.pc);
                d.channel.push(t.channel);
                d.vel.push(t.vel);
                d.pan.push(t.pan);
                d.dur.push(t.dur);
                d.role.push(t.role);
                d.target_row.push((bi * nf) as i64 + t.onset as i64);
            }
            d.chord_labels.extend_from_slice(&ex.chord_labels);
            d.key_labels.push(ex.key_label);
        }
        d.examples = examples;
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notes::{Instrument, NoteEvent};

    #[test]
    fn onset_tokens_route_to_their_frame_row() {
        let song = Song {
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
        };
        let batch = EventBatchData::build(std::slice::from_ref(&song));
        assert_eq!(batch.n_total, 2);
        assert_eq!(batch.target_row[0], 0); // onset frame 0
        assert_eq!(batch.target_row[1], 2); // onset frame 2
        assert!(batch.channel.contains(&5));
        assert!(batch.channel.contains(&3));
    }
}

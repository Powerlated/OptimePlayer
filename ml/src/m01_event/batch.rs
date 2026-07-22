//! Generation 01's pooling: each frame token is a **scatter-add sum** of the notes
//! *onsetting* in that frame.
//!
//! The batch holds per-field index arrays over a flat list of onset tokens plus each
//! token's destination frame row; the model embeds the fields and scatter-adds them
//! into their row ([`super::model`]). Sustain is carried by the per-note `dur` field
//! rather than by repeating a note across frames — contrast [`crate::m02_hier`],
//! which pools the notes *sounding* in each frame.

use crate::notes::Song;
use crate::theory::NO_CHORD;
use crate::tokenize::{self, EventExample};

/// Per-slot kind codes in [`EventBatchData::slot_kind`].
pub const SLOT_FRAME: i64 = 0;
pub const SLOT_EOS: i64 = 1;
pub const SLOT_PAD: i64 = 2;

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
    /// `batch*n_frames` slot kinds ([`SLOT_FRAME`]/[`SLOT_EOS`]/[`SLOT_PAD`]).
    /// All-`SLOT_FRAME` for the legacy fixed-window layout ([`Self::build`]).
    pub slot_kind: Vec<i64>,
    /// `batch*n_frames` document ids (index of the constituent song a slot belongs
    /// to; its EOS slot included). `-1` for pad. All-`0` for the legacy layout.
    pub doc_id: Vec<i64>,
    /// `batch*n_frames` supervised-label validity (1.0 = this frame's chord label
    /// counts in the loss). Legacy layout: all `1.0`. Generative layout: frame
    /// slots with a trustworthy label only (EOS/pad and unannotated frames are 0).
    pub label_valid: Vec<f32>,
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
            slot_kind: vec![SLOT_FRAME; batch * nf],
            doc_id: vec![0; batch * nf],
            label_valid: vec![1.0; batch * nf],
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

    /// Build a **variable-length / multi-document** batch for the generative
    /// long-context backbone (m03). Each song is laid out per its
    /// [`Song::doc_spans`] (packed sequences come pre-laid-out); a plain song
    /// becomes one document `[0, n_frames)` followed by its EOS slot. All songs
    /// are padded to the batch's longest layout, so `n_frames` here is the padded
    /// slot count — [`Self::slot_kind`] / [`Self::doc_id`] / [`Self::label_valid`]
    /// carry the real structure.
    pub fn build_generative(songs: &[Song]) -> EventBatchData {
        // Per-song layout: spans + total slot count (last EOS inclusive).
        let layouts: Vec<Vec<(u32, u32)>> = songs
            .iter()
            .map(|s| {
                if s.doc_spans.is_empty() {
                    vec![(0u32, s.n_frames as u32)]
                } else {
                    s.doc_spans.clone()
                }
            })
            .collect();
        let padded_nf = songs
            .iter()
            .zip(&layouts)
            .map(|(s, spans)| {
                let layout_end = spans.last().map(|&(_, e)| e + 1).unwrap_or(0) as usize;
                // Packed songs are pre-sized to their full window (pad included).
                if s.doc_spans.is_empty() {
                    layout_end
                } else {
                    s.n_frames.max(layout_end)
                }
            })
            .max()
            .unwrap_or(0);

        // Tokenize against the padded frame count (notes only live inside spans).
        let examples: Vec<EventExample> = songs
            .iter()
            .map(|s| {
                let mut chord_labels = s.chord_labels.clone();
                chord_labels.resize(padded_nf, NO_CHORD);
                let padded = Song {
                    n_frames: padded_nf,
                    chord_labels,
                    ..s.clone()
                };
                EventExample::from_song(&padded)
            })
            .collect();
        let mut d = Self::from_examples(examples);

        // Overwrite the legacy all-frame structure with the real layout.
        for (bi, (song, spans)) in songs.iter().zip(&layouts).enumerate() {
            for f in 0..padded_nf {
                let idx = bi * padded_nf + f;
                let fu = f as u32;
                let in_doc = spans.iter().position(|&(s, e)| fu >= s && fu < e);
                let is_eos = spans.iter().position(|&(_, e)| fu == e);
                let (kind, doc) = match (in_doc, is_eos) {
                    (Some(di), _) => (SLOT_FRAME, di as i64),
                    (None, Some(di)) => (SLOT_EOS, di as i64),
                    (None, None) => (SLOT_PAD, -1),
                };
                d.slot_kind[idx] = kind;
                d.doc_id[idx] = doc;
                d.label_valid[idx] = if kind == SLOT_FRAME {
                    match &song.label_mask {
                        Some(m) => {
                            if m.get(f).copied().unwrap_or(false) {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        None => 1.0,
                    }
                } else {
                    0.0
                };
            }
        }
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
            ..Song::default()
        };
        let batch = EventBatchData::build(std::slice::from_ref(&song));
        assert_eq!(batch.n_total, 2);
        assert_eq!(batch.target_row[0], 0); // onset frame 0
        assert_eq!(batch.target_row[1], 2); // onset frame 2
        assert!(batch.channel.contains(&5));
        assert!(batch.channel.contains(&3));
    }
}

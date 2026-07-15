//! Inference: run a trained model on a note-event stream and produce a predicted
//! global key + a per-frame chord timeline, then merge frames into chord segments
//! suitable for driving a circle-of-fifths / annotated-chord display.

use crate::train::MlDevice;
use burn::module::AutodiffModule;
use burn::prelude::*;

use crate::features::{self, FEATURE_DIM};
use crate::model::KeyChordModel;
use crate::notes::NoteEvent;
use crate::theory::{Chord, Key, N_CHORD_CLASSES, N_KEY_CLASSES};
use crate::train::{Back, Inner};

/// A contiguous run of frames sharing one predicted chord (or silence).
#[derive(Debug, Clone)]
pub struct Segment {
    pub start_frame: usize,
    pub end_frame: usize,
    pub chord: Option<Chord>,
}

/// Model prediction for a whole excerpt.
#[derive(Debug, Clone)]
pub struct Prediction {
    pub key: Key,
    pub key_confidence: f32,
    /// Per-frame chord label.
    pub chord_labels: Vec<usize>,
    pub segments: Vec<Segment>,
}

/// Predict key + chords for a note-event stream of length `n_frames`.
pub fn predict(
    model: &KeyChordModel<Back>,
    notes: &[NoteEvent],
    n_frames: usize,
    device: &MlDevice,
) -> Prediction {
    let grid = features::extract(notes, n_frames);
    let model = model.valid();

    let features = Tensor::<Inner, 3>::from_data(
        TensorData::new(grid.data, [1, n_frames, FEATURE_DIM]),
        device,
    );
    let out = model.forward(features);

    // Key: softmax for a confidence read-out.
    let key_probs = burn::tensor::activation::softmax(out.key_logits, 1);
    let key_probs: Vec<f32> = key_probs.into_data().to_vec().unwrap();
    let (key_idx, key_conf) = argmax_conf(&key_probs, N_KEY_CLASSES);
    let key = Key::from_label(key_idx);

    // Chords: per-frame argmax.
    let chord_pred = out
        .chord_logits
        .reshape([n_frames, N_CHORD_CLASSES])
        .argmax(1)
        .reshape([n_frames]);
    let chord_labels: Vec<usize> = chord_pred
        .into_data()
        .to_vec::<i64>()
        .unwrap()
        .into_iter()
        .map(|x| x as usize)
        .collect();

    let segments = merge_segments(&chord_labels);
    Prediction {
        key,
        key_confidence: key_conf,
        chord_labels,
        segments,
    }
}

fn argmax_conf(probs: &[f32], stride: usize) -> (usize, f32) {
    let row = &probs[0..stride];
    let mut best = 0;
    let mut best_v = row[0];
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    (best, best_v)
}

/// Collapse a per-frame chord-label vector into contiguous segments.
pub fn merge_segments(labels: &[usize]) -> Vec<Segment> {
    let mut out = Vec::new();
    if labels.is_empty() {
        return out;
    }
    let mut start = 0usize;
    for i in 1..=labels.len() {
        if i == labels.len() || labels[i] != labels[start] {
            out.push(Segment {
                start_frame: start,
                end_frame: i,
                chord: Chord::from_label(labels[start]),
            });
            start = i;
        }
    }
    out
}

impl Prediction {
    /// Human-readable timeline, e.g. for logging or a debug overlay.
    pub fn describe(&self) -> String {
        let mut s = format!(
            "key: {} ({:.0}% conf)\n",
            self.key.name(),
            self.key_confidence * 100.0
        );
        for seg in &self.segments {
            let name = match seg.chord {
                Some(c) => c.name(),
                None => "—".to_string(),
            };
            s.push_str(&format!(
                "  frames {:>4}..{:<4}  {}\n",
                seg.start_frame, seg.end_frame, name
            ));
        }
        s
    }
}

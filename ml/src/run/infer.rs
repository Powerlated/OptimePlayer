//! Inference: run a trained model on a note-event stream and produce a predicted
//! global key + a per-frame chord timeline, then merge frames into chord segments
//! suitable for driving a circle-of-fifths / annotated-chord display.
//!
//! Generic over [`Backbone`] — every generation emits the same
//! [`ModelOutput`](crate::backbone::ModelOutput), so decoding it into a
//! [`Prediction`] is written once here rather than once per generation.

use burn::module::AutodiffModule;

use crate::backbone::{Backbone, ModelOutput};
use crate::backend::{Back, Inner, MlDevice};
use crate::notes::{NoteEvent, Song};
use crate::theory::{
    root_quality_to_chord_label, Chord, Key, N_KEY_CLASSES, N_QUALITY_CLASSES, N_ROOT_CLASSES,
};

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

/// Predict key + chords for a note-event stream of length `n_frames`, using any
/// backbone.
pub fn predict<M>(model: &M, notes: &[NoteEvent], n_frames: usize, device: &MlDevice) -> Prediction
where
    M: Backbone<Back> + AutodiffModule<Back>,
    M::InnerModule: Backbone<Inner, Batch = <M as Backbone<Back>>::Batch>,
{
    // A single-window "song" carrying the notes; the labels are placeholders that
    // inference never reads.
    let song = Song {
        key_label: 0,
        n_frames,
        notes: notes.to_vec(),
        chord_labels: vec![0; n_frames],
        is_music: None,
        ..Song::default()
    };
    let data = <M::InnerModule as Backbone<Inner>>::build_batch(std::slice::from_ref(&song));
    let out = model.valid().infer_output(&data, device);
    decode_output(out, n_frames)
}

/// Decode one window's [`ModelOutput`] into a [`Prediction`]. Backbone-agnostic: the
/// factored root + quality argmax and the key read-out are identical for every
/// generation, and match what `shared::eval_counts` scores.
pub fn decode_output(out: ModelOutput<Inner>, n_frames: usize) -> Prediction {
    // Key: softmax for a confidence read-out.
    let key_probs = burn::tensor::activation::softmax(out.key_logits, 1);
    let key_probs: Vec<f32> = key_probs.into_data().to_vec().unwrap();
    let (key_idx, key_conf) = argmax_conf(&key_probs, N_KEY_CLASSES);
    let key = Key::from_label(key_idx);

    // Chords: per-frame factored argmax over the dedicated root + quality heads,
    // recombined into the joint 121-label space. The "none" quality column (index
    // 0) is dropped so a real root always yields a concrete quality; the root head
    // alone decides no-chord. A VARLEN backbone's logits cover `n_frames` real
    // frames plus EOS/pad slots — read the padded length from the tensor and keep
    // the first `n_frames`.
    let padded = out.root_logits.dims()[1];
    let root_pred: Vec<i64> = out
        .root_logits
        .reshape([padded, N_ROOT_CLASSES])
        .argmax(1)
        .reshape([padded])
        .into_data()
        .to_vec()
        .unwrap();
    let quality_pred: Vec<i64> = out
        .quality_logits
        .reshape([padded, N_QUALITY_CLASSES])
        .slice([0..padded, 1..N_QUALITY_CLASSES])
        .argmax(1)
        .reshape([padded])
        .into_data()
        .to_vec()
        .unwrap();
    let chord_labels: Vec<usize> = (0..n_frames.min(padded))
        .map(|i| root_quality_to_chord_label(root_pred[i] as usize, quality_pred[i] as usize + 1))
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

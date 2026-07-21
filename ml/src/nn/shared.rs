//! Backbone-agnostic training pieces that operate on the shared [`ModelOutput`]
//! (factored root + quality + key logits). Both the frame model ([`crate::train`])
//! and the event model ([`crate::event`]) build the same targets, compute the same
//! multi-task loss (+ beat-aware smoothness), and score the same metrics — so those
//! live here once rather than being duplicated per backbone.

use burn::nn::loss::CrossEntropyLoss;
use burn::prelude::*;

use crate::backbone::ModelOutput;
use crate::notes::FRAMES_PER_BEAT;
use crate::theory::{
    chord_label_to_root_quality, root_quality_to_chord_label, NO_CHORD, N_QUALITY_CLASSES,
    N_ROOT_CLASSES,
};

/// The concrete int element returned by an argmax read. burn's `TensorData::to_vec` does not
/// cross-cast int dtypes, so this matches each backend's `IntElem`: `i32` under the WGPU build
/// (WGSL has no first-class i64), `i64` everywhere else (ndarray and the i64-pinned CUDA backend).
#[cfg(feature = "gpu")]
type PredElem = i32;
#[cfg(not(feature = "gpu"))]
type PredElem = i64;

/// Factored per-frame targets `(root_class, quality_class)` from flattened joint
/// chord labels (`batch*seq`). No-chord → class 0 in both (plain CE over all frames).
pub fn root_quality_targets<B: Backend>(
    labels: &[usize],
    device: &B::Device,
) -> (Tensor<B, 1, Int>, Tensor<B, 1, Int>) {
    let n = labels.len();
    let mut root = Vec::with_capacity(n);
    let mut quality = Vec::with_capacity(n);
    for &c in labels {
        let (r, q) = chord_label_to_root_quality(c);
        root.push(r as i64);
        quality.push(q as i64);
    }
    (
        Tensor::<B, 1, Int>::from_data(TensorData::new(root, [n]), device),
        Tensor::<B, 1, Int>::from_data(TensorData::new(quality, [n]), device),
    )
}

/// Key targets from per-window key labels (`batch`).
pub fn key_targets<B: Backend>(labels: &[usize], device: &B::Device) -> Tensor<B, 1, Int> {
    let data: Vec<i64> = labels.iter().map(|&k| k as i64).collect();
    Tensor::<B, 1, Int>::from_data(TensorData::new(data, [labels.len()]), device)
}

/// **Beat-aware** temporal-smoothness penalty on a `[batch, seq, classes]` logit
/// tensor: mean absolute frame-to-frame change of the softmax distribution over
/// **intra-beat** transitions only (a change on a beat boundary is free), so more
/// than one chord change per beat is penalised. Differentiable regulariser.
pub fn beat_aware_tv<B: Backend>(logits: Tensor<B, 3>) -> Tensor<B, 1> {
    let [b, seq, c] = logits.dims();
    let device = logits.device();
    if seq < 2 {
        return Tensor::zeros([1], &device);
    }
    let probs = burn::tensor::activation::softmax(logits, 2);
    let later = probs.clone().slice([0..b, 1..seq, 0..c]);
    let earlier = probs.slice([0..b, 0..seq - 1, 0..c]);
    let diff = (later - earlier).abs();

    let fpb = FRAMES_PER_BEAT as usize;
    let mask_vals: Vec<f32> = (1..seq)
        .map(|j| if j % fpb != 0 { 1.0 } else { 0.0 })
        .collect();
    let kept: f32 = mask_vals.iter().sum();
    let mask = Tensor::<B, 1>::from_data(TensorData::new(mask_vals, [seq - 1]), &device).reshape([
        1,
        seq - 1,
        1,
    ]);
    let denom = (b as f32) * kept.max(1.0) * (c as f32);
    (diff * mask).sum().div_scalar(denom)
}

/// Multi-task supervised loss on the shared output: factored root + quality CE +
/// key CE (weighted) + beat-aware smoothness (weighted). Not yet scaled by `1/k` —
/// the data-parallel caller does that.
#[allow(clippy::too_many_arguments)]
pub fn chord_key_loss<B: Backend>(
    ce: &CrossEntropyLoss<B>,
    out: &ModelOutput<B>,
    root_t: Tensor<B, 1, Int>,
    quality_t: Tensor<B, 1, Int>,
    key_t: Tensor<B, 1, Int>,
    b: usize,
    seq: usize,
    key_weight: f64,
    smoothness_weight: f64,
) -> Tensor<B, 1> {
    let root_logits = out.root_logits.clone().reshape([b * seq, N_ROOT_CLASSES]);
    let quality_logits = out
        .quality_logits
        .clone()
        .reshape([b * seq, N_QUALITY_CLASSES]);
    let chord = ce.forward(root_logits, root_t) + ce.forward(quality_logits, quality_t);
    let key = ce.forward(out.key_logits.clone(), key_t);
    let smooth = beat_aware_tv(out.root_logits.clone()) + beat_aware_tv(out.quality_logits.clone());
    chord + key.mul_scalar(key_weight) + smooth.mul_scalar(smoothness_weight)
}

/// Accumulated validation counts for one batch.
#[derive(Default, Clone, Copy)]
pub struct EvalCounts {
    pub key_correct: usize,
    pub chord_correct: usize,
    pub chord_total: usize,
    /// Within-sequence predicted chord-label transitions (flicker proxy).
    pub chord_changes: usize,
}

/// Score one batch's shared output against ground-truth labels: key accuracy,
/// chord accuracy (non-no-chord frames), and predicted chord transitions per
/// sequence. Reconstructs the joint chord label from root + quality argmax
/// (quality restricted to a real quality), matching inference.
pub fn eval_counts<B: Backend>(
    out: &ModelOutput<B>,
    chord_labels: &[usize],
    key_labels: &[usize],
    b: usize,
    seq: usize,
) -> EvalCounts {
    let mut c = EvalCounts::default();

    let key_pred: Vec<PredElem> = out
        .key_logits
        .clone()
        .argmax(1)
        .reshape([b])
        .into_data()
        .to_vec()
        .unwrap();
    for (i, &k) in key_labels.iter().enumerate() {
        if key_pred[i] as usize == k {
            c.key_correct += 1;
        }
    }

    let n = b * seq;
    let root_pred: Vec<PredElem> = out
        .root_logits
        .clone()
        .reshape([n, N_ROOT_CLASSES])
        .argmax(1)
        .reshape([n])
        .into_data()
        .to_vec()
        .unwrap();
    let quality_pred: Vec<PredElem> = out
        .quality_logits
        .clone()
        .reshape([n, N_QUALITY_CLASSES])
        .slice([0..n, 1..N_QUALITY_CLASSES])
        .argmax(1)
        .reshape([n])
        .into_data()
        .to_vec()
        .unwrap();

    for bi in 0..b {
        let mut prev: Option<usize> = None;
        for fi in 0..seq {
            let idx = bi * seq + fi;
            let pred = root_quality_to_chord_label(
                root_pred[idx] as usize,
                quality_pred[idx] as usize + 1,
            );
            if prev.is_some_and(|p| p != pred) {
                c.chord_changes += 1;
            }
            prev = Some(pred);
            let label = chord_labels[idx];
            if label == NO_CHORD {
                continue;
            }
            c.chord_total += 1;
            if pred == label {
                c.chord_correct += 1;
            }
        }
    }
    c
}

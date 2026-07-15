//! Multi-task training loop: per-frame chord cross-entropy + pooled key
//! cross-entropy, jointly optimised with Adam. A manual loop (rather than Burn's
//! `Learner`) keeps the two-headed objective and its metrics fully explicit.

use burn::backend::Autodiff;
use burn::module::{AutodiffModule, Module};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams};
use burn::prelude::*;
use burn::record::CompactRecorder;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use std::path::Path;

use crate::data::Example;
use crate::features::FEATURE_DIM;
use crate::model::{KeyChordModel, ModelConfig};
use crate::notes::{random_transpose, Song};
use crate::parallel::{default_shards, dp_step};
use crate::theory::{NO_CHORD, N_CHORD_CLASSES, N_KEY_CLASSES};

// Compute backend: the pure-Rust `ndarray` CPU backend. (GPU/threaded-CPU
// backends were evaluated — see git history — but on a small model like this one
// they don't help: intra-op parallelism has nothing to fill and dispatch/launch
// overhead dominates. CPU throughput comes from the data-parallel training loop
// in [`crate::parallel`] instead.) `MlDevice` aliases the device type so the rest
// of the crate never names a concrete backend.
pub type Inner = burn::backend::NdArray<f32>;
pub type MlDevice = burn::backend::ndarray::NdArrayDevice;

/// The autodiff training backend wrapping [`Inner`].
pub type Back = Autodiff<Inner>;

#[derive(Config, Debug)]
pub struct TrainConfig {
    #[config(default = 12)]
    pub epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 3.0e-4)]
    pub lr: f64,
    #[config(default = 0.5)]
    pub key_loss_weight: f64,
    /// On-the-fly random transposition augmentation (key-invariance). The labels
    /// (key + per-frame chords) transpose with the notes.
    #[config(default = true)]
    pub augment: bool,
    #[config(default = 1234)]
    pub seed: u64,
    pub model: ModelConfig,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig::new(ModelConfig::wired())
    }
}

/// Build a `[batch, seq, FEATURE_DIM]` feature tensor for a slice of examples.
fn features_tensor<B: Backend>(batch: &[Example], seq: usize, device: &B::Device) -> Tensor<B, 3> {
    let mut data = Vec::with_capacity(batch.len() * seq * FEATURE_DIM);
    for ex in batch {
        data.extend_from_slice(&ex.features);
    }
    Tensor::<B, 3>::from_data(
        TensorData::new(data, [batch.len(), seq, FEATURE_DIM]),
        device,
    )
}

fn key_targets<B: Backend>(batch: &[Example], device: &B::Device) -> Tensor<B, 1, Int> {
    let data: Vec<i64> = batch.iter().map(|e| e.key_label as i64).collect();
    Tensor::<B, 1, Int>::from_data(TensorData::new(data, [batch.len()]), device)
}

fn chord_targets<B: Backend>(
    batch: &[Example],
    seq: usize,
    device: &B::Device,
) -> Tensor<B, 1, Int> {
    let mut data: Vec<i64> = Vec::with_capacity(batch.len() * seq);
    for ex in batch {
        data.extend(ex.chord_labels.iter().map(|&c| c as i64));
    }
    Tensor::<B, 1, Int>::from_data(TensorData::new(data, [batch.len() * seq]), device)
}

/// Load a pretrained model (weights `prefix` + config `prefix.json`) to warm-start
/// fine-tuning. The self-supervised stage only trains the encoder / input
/// projection / positional embedding (and its reconstruction head); the chord and
/// key heads were never touched by pretraining, so loading the whole record gives
/// a trained trunk with fresh, still-random supervised heads — exactly the warm
/// start we want.
pub fn load_pretrained(prefix: &Path, device: &MlDevice) -> KeyChordModel<Back> {
    let config = ModelConfig::load(prefix.with_extension("json")).expect("load pretrained config");
    let model: KeyChordModel<Back> = config.init(device);
    model
        .load_file(prefix, &CompactRecorder::new(), device)
        .expect("load pretrained weights")
}

/// Train on `train`/`val`, writing `model.mpk` + `model.json` (config) to `out_dir`.
///
/// When `pretrained` is `Some(prefix)`, the encoder is warm-started from a
/// self-supervised checkpoint (see [`load_pretrained`]); when `None`, behavior is
/// identical to a from-scratch supervised run.
pub fn run(
    config: &TrainConfig,
    train: &[Song],
    val: &[Song],
    out_dir: &Path,
    pretrained: Option<&Path>,
) {
    let device = MlDevice::default();
    let mut model: KeyChordModel<Back> = match pretrained {
        Some(prefix) => {
            println!("warm-starting encoder from {}", prefix.display());
            load_pretrained(prefix, &device)
        }
        None => config.model.init(&device),
    };
    let mut optim = AdamConfig::new().init();
    let ce = CrossEntropyLossConfig::new().init(&device);

    let seq = train.first().map(|s| s.n_frames).unwrap_or(0);
    assert!(seq > 0, "empty training set");

    std::fs::create_dir_all(out_dir).expect("create out dir");
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut indices: Vec<usize> = (0..train.len()).collect();
    let n_shards = std::env::var("DP_SHARDS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(default_shards);

    println!(
        "training: {} train / {} val examples, seq_len {seq}, {} chord classes, {} key classes, {n_shards}-way data-parallel",
        train.len(),
        val.len(),
        N_CHORD_CLASSES,
        N_KEY_CLASSES
    );

    for epoch in 1..=config.epochs {
        let epoch_start = std::time::Instant::now();
        indices.shuffle(&mut rng);
        let mut running = 0.0f64;
        let mut n_batches = 0usize;

        for chunk in indices.chunks(config.batch_size) {
            // Data-parallel step: shards differentiate the shared weights in
            // parallel, gradients summed, one Adam step (see `crate::parallel`).
            let k = n_shards.min(chunk.len().max(1));
            let (m, loss) = dp_step(
                model,
                &mut optim,
                config.lr,
                &device,
                chunk,
                k,
                |m, shard| {
                    // Deterministic per-shard, per-epoch transposition augmentation.
                    let mut srng = StdRng::seed_from_u64(
                        config.seed
                            ^ (shard[0] as u64).wrapping_mul(0x9E3779B9)
                            ^ (epoch as u64).wrapping_mul(0x85EBCA77),
                    );
                    let batch: Vec<Example> = shard
                        .iter()
                        .map(|&i| {
                            let shift = if config.augment {
                                random_transpose(&mut srng)
                            } else {
                                0
                            };
                            Example::from_song(&train[i].transpose(shift))
                        })
                        .collect();
                    let features = features_tensor::<Back>(&batch, seq, &device);
                    let out = m.forward(features);
                    let chord_logits = out
                        .chord_logits
                        .reshape([batch.len() * seq, N_CHORD_CLASSES]);
                    let chord_t = chord_targets::<Back>(&batch, seq, &device);
                    let key_t = key_targets::<Back>(&batch, &device);
                    let chord_loss = ce.forward(chord_logits, chord_t);
                    let key_loss = ce.forward(out.key_logits, key_t);
                    // Scale by 1/k so the K shard gradients sum to the batch-mean gradient.
                    let l = (chord_loss + key_loss.mul_scalar(config.key_loss_weight))
                        .mul_scalar(1.0 / k as f64);
                    let grads = GradientsParams::from_grads(l.clone().backward(), m);
                    (grads, l.into_scalar().elem::<f32>() as f64)
                },
            );
            model = m;
            running += loss;
            n_batches += 1;
        }

        let (key_acc, chord_acc) = evaluate(&model, val, config.batch_size, seq, &device);
        println!(
            "epoch {epoch:>3}/{}  loss {:.4}  |  val key acc {:.1}%  chord acc {:.1}%  |  {:.1}s",
            config.epochs,
            running / n_batches.max(1) as f64,
            key_acc * 100.0,
            chord_acc * 100.0,
            epoch_start.elapsed().as_secs_f64()
        );
    }

    // Persist model weights + architecture config.
    let recorder = CompactRecorder::new();
    model
        .clone()
        .save_file(out_dir.join("model"), &recorder)
        .expect("save model weights");
    config
        .model
        .save(out_dir.join("model.json"))
        .expect("save model config");
    println!("saved model to {}", out_dir.display());
}

/// Validation accuracy for both heads (chord accuracy ignores no-chord frames).
pub fn evaluate(
    model: &KeyChordModel<Back>,
    val: &[Song],
    batch_size: usize,
    seq: usize,
    device: &MlDevice,
) -> (f64, f64) {
    if val.is_empty() {
        return (0.0, 0.0);
    }
    let model = model.valid();
    let mut key_correct = 0usize;
    let mut chord_correct = 0usize;
    let mut chord_total = 0usize;

    for chunk in val.chunks(batch_size) {
        // No augmentation at eval.
        let batch: Vec<Example> = chunk.iter().map(Example::from_song).collect();
        let features = features_tensor::<Inner>(&batch, seq, device);
        let out = model.forward(features);

        // Key head.
        let key_pred = out.key_logits.argmax(1).reshape([batch.len()]);
        let key_pred: Vec<i64> = key_pred.into_data().to_vec().unwrap();
        for (i, ex) in batch.iter().enumerate() {
            if key_pred[i] as usize == ex.key_label {
                key_correct += 1;
            }
        }

        // Chord head.
        let chord_pred = out
            .chord_logits
            .reshape([batch.len() * seq, N_CHORD_CLASSES])
            .argmax(1)
            .reshape([batch.len() * seq]);
        let chord_pred: Vec<i64> = chord_pred.into_data().to_vec().unwrap();
        for (bi, ex) in batch.iter().enumerate() {
            for (fi, &label) in ex.chord_labels.iter().enumerate() {
                if label == NO_CHORD {
                    continue;
                }
                chord_total += 1;
                if chord_pred[bi * seq + fi] as usize == label {
                    chord_correct += 1;
                }
            }
        }
    }

    let key_acc = key_correct as f64 / val.len() as f64;
    let chord_acc = chord_correct as f64 / chord_total.max(1) as f64;
    (key_acc, chord_acc)
}

/// Load a trained model + its config from `dir`.
pub fn load_model(dir: &Path, device: &MlDevice) -> KeyChordModel<Back> {
    let config = ModelConfig::load(dir.join("model.json")).expect("load model config");
    let model: KeyChordModel<Back> = config.init(device);
    model
        .load_file(dir.join("model"), &CompactRecorder::new(), device)
        .expect("load model weights")
}

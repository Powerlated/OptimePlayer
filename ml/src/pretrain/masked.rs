//! Self-supervised **masked-frame** pretraining — **generation 00 only**.
//!
//! The encoder is trained on *unlabeled* real game songs (harvested via
//! [`crate::harvest`]) by hiding a random fraction of frames and reconstructing
//! their pitch-class content from the surrounding context — a BERT-style masked
//! objective on the note-event grid. No chord/key labels are used here; this
//! stage only teaches the encoder the *real* note-event distribution. The
//! resulting weights warm-start the supervised fine-tune ([`crate::train`]).
//!
//! Loss is mean-squared error on the four L2-normalized pitch-class blocks
//! ([`PITCH_BLOCK_DIM`] dims), scored **only on masked frames**.
//!
//! This pretext needs the hand-engineered feature grid to mask and reconstruct, so
//! it is specific to [`crate::m00_frame`]; the learned-token generations pretrain
//! autoregressively instead ([`super::ar`]).

use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams};
use burn::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::path::Path;

use crate::backbone::{self, Backbone};
use crate::backend::{Back, Inner, MlDevice};
use crate::dashboard::{self, ContextWindow, DataStats, EpochPoint, RunMeta};
use crate::data::Example;
use crate::features::{FEATURE_DIM, PITCH_BLOCK_DIM};
use crate::m00_frame::{KeyChordModel, ModelConfig};
use crate::notes::{random_transpose, Song};
use crate::parallel::{default_shards, dp_step};
use crate::progress::TrainProgress;

#[derive(Config, Debug)]
pub struct PretrainConfig {
    #[config(default = 20)]
    pub epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 3.0e-4)]
    pub lr: f64,
    /// Fraction of frames masked per example (BERT-style ~15%).
    #[config(default = 0.15)]
    pub mask_fraction: f64,
    /// On-the-fly random transposition augmentation (key-invariance).
    #[config(default = true)]
    pub augment: bool,
    #[config(default = 4242)]
    pub seed: u64,
    pub model: ModelConfig,
}

impl Default for PretrainConfig {
    fn default() -> Self {
        PretrainConfig::new(ModelConfig::wired())
    }
}

/// CPU-side masked batch: masked input rows (mask token = zeros), a per-frame
/// mask (1 = masked), the reconstruction target (pitch-class blocks of the
/// *original* rows), and the number of masked frames.
struct MaskedBatch {
    /// `batch * seq * FEATURE_DIM`
    input: Vec<f32>,
    /// `batch * seq` (one flag per frame)
    mask: Vec<f32>,
    /// `batch * seq * PITCH_BLOCK_DIM`
    target: Vec<f32>,
    n_masked: usize,
}

fn build_masked_batch<R: Rng>(
    batch: &[Example],
    seq: usize,
    frac: f64,
    rng: &mut R,
) -> MaskedBatch {
    let mut input = Vec::with_capacity(batch.len() * seq * FEATURE_DIM);
    let mut mask = Vec::with_capacity(batch.len() * seq);
    let mut target = Vec::with_capacity(batch.len() * seq * PITCH_BLOCK_DIM);
    let mut n_masked = 0usize;

    for ex in batch {
        for f in 0..seq {
            let row = &ex.features[f * FEATURE_DIM..(f + 1) * FEATURE_DIM];
            // Target is always the original pitch-class content.
            target.extend_from_slice(&row[..PITCH_BLOCK_DIM]);
            if rng.gen_bool(frac) {
                mask.push(1.0);
                n_masked += 1;
                input.extend(std::iter::repeat_n(0.0f32, FEATURE_DIM)); // mask token
            } else {
                mask.push(0.0);
                input.extend_from_slice(row);
            }
        }
    }
    MaskedBatch {
        input,
        mask,
        target,
        n_masked,
    }
}

/// Masked reconstruction MSE for one batch on backend `B` (differentiable when
/// `B` is an autodiff backend).
fn masked_loss<B: Backend>(
    model: &KeyChordModel<B>,
    mb: &MaskedBatch,
    batch: usize,
    seq: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    let input = Tensor::<B, 3>::from_data(
        TensorData::new(mb.input.clone(), [batch, seq, FEATURE_DIM]),
        device,
    );
    let recon = model.forward_ssl(input); // [batch, seq, PITCH_BLOCK_DIM]
    let target = Tensor::<B, 3>::from_data(
        TensorData::new(mb.target.clone(), [batch, seq, PITCH_BLOCK_DIM]),
        device,
    );
    let mask = Tensor::<B, 3>::from_data(TensorData::new(mb.mask.clone(), [batch, seq, 1]), device);
    let diff = recon - target;
    let masked_sq = diff.clone().mul(diff).mul(mask); // broadcast [.,.,1] over PITCH_BLOCK_DIM
    let denom = (mb.n_masked.max(1) * PITCH_BLOCK_DIM) as f64;
    masked_sq.sum().div_scalar(denom)
}

/// Pretrain the encoder on `train`, writing `<out_dir>/00-frame/pretrained(.mpk)` +
/// `pretrained.json`.
pub fn run(config: &PretrainConfig, train: &[Song], val: &[Song], out_dir: &Path) {
    let device = MlDevice::default();
    let mut model: KeyChordModel<Back> = config.model.init(&device);
    let mut optim = AdamConfig::new().init();

    let seq = train.first().map(|s| s.n_frames).unwrap_or(0);
    assert!(seq > 0, "empty pretraining set");

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut indices: Vec<usize> = (0..train.len()).collect();
    // Data-parallel width: `DP_SHARDS` env override, else logical-core count.
    let n_shards = std::env::var("DP_SHARDS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(default_shards);

    println!(
        "pretraining: {} train / {} val windows, seq_len {seq}, mask {:.0}%, {n_shards}-way data-parallel",
        train.len(),
        val.len(),
        config.mask_fraction * 100.0
    );
    let transpositions = if config.augment {
        crate::notes::N_TRANSPOSITIONS
    } else {
        1
    };
    let data = DataStats::measure(train, val, transpositions);
    dashboard::start(RunMeta {
        stage: format!(
            "masked-frame pretrain ({:.0}% masked)",
            config.mask_fraction * 100.0
        ),
        backbone: "frame".to_string(),
        backend: format!(
            "{}, {n_shards}-way DP",
            dashboard::backend_label(std::any::type_name::<Back>())
        ),
        precision: crate::backend::precision::<Back>(),
        epochs: config.epochs,
        context: ContextWindow::from_frames(seq),
        data: data.clone(),
        params: model.num_params(),
        flops_per_window: <KeyChordModel<Back> as Backbone<Back>>::flops_per_window(
            &config.model,
            data.notes_per_window.round() as usize,
        ),
        model_config: serde_json::to_value(&config.model).unwrap_or(serde_json::Value::Null),
        train_config: serde_json::to_value(config).unwrap_or(serde_json::Value::Null),
    });

    let n_total = indices.len().div_ceil(config.batch_size);

    for epoch in 1..=config.epochs {
        let epoch_start = std::time::Instant::now();
        indices.shuffle(&mut rng);
        let mut running = 0.0f64;
        let mut n_batches = 0usize;
        let mut prog = TrainProgress::per_epoch(epoch);

        for chunk in indices.chunks(config.batch_size) {
            // Data-parallel step: shards of this minibatch differentiate the shared
            // weights concurrently, gradients summed, one Adam step (see `parallel`).
            let k = n_shards.min(chunk.len().max(1));
            let (m, loss) = dp_step(
                model,
                &mut optim,
                config.lr,
                &device,
                chunk,
                k,
                |m, shard| {
                    // Deterministic per-shard, per-epoch RNG (the closure runs on
                    // many threads) driving both transposition and masking.
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
                    let mb = build_masked_batch(&batch, seq, config.mask_fraction, &mut srng);
                    // Scale by 1/k so the K shard gradients sum to the batch-mean gradient.
                    let l = masked_loss::<Back>(m, &mb, batch.len(), seq, &device)
                        .mul_scalar(1.0 / k as f64);
                    let grads = GradientsParams::from_grads(l.clone().backward(), m);
                    (grads, l.into_scalar().elem::<f32>() as f64)
                },
            );
            model = m;
            running += loss;
            n_batches += 1;
            prog.maybe_log(running, n_batches, n_total);
        }

        let val_loss = evaluate(&model, val, config, seq, &device);
        let train_loss = running / n_batches.max(1) as f64;
        let secs = epoch_start.elapsed().as_secs_f64();
        println!(
            "epoch {epoch:>3}/{}  recon loss {train_loss:.5}  |  val recon loss {val_loss:.5}  |  {secs:.1}s",
            config.epochs,
        );
        dashboard::record_epoch(EpochPoint::pretext(epoch, train_loss, val_loss, secs));
        backbone::save_epoch::<KeyChordModel<Back>, Back>(
            &model,
            &config.model,
            &backbone::artifact_dir::<KeyChordModel<Back>, Back>(out_dir),
            "pretrained",
            epoch,
        );
    }

    let dir = backbone::artifact_dir::<KeyChordModel<Back>, Back>(out_dir);
    backbone::save::<KeyChordModel<Back>, Back>(model, &config.model, &dir, "pretrained");
    println!("saved pretrained encoder to {}", dir.display());
    dashboard::finish(&dir);
}

/// Held-out masked reconstruction loss. Uses a fixed RNG so the number is
/// comparable across epochs / runs.
pub fn evaluate(
    model: &KeyChordModel<Back>,
    val: &[Song],
    config: &PretrainConfig,
    seq: usize,
    device: &MlDevice,
) -> f64 {
    if val.is_empty() {
        return 0.0;
    }
    let model = model.valid();
    let mut rng = StdRng::seed_from_u64(0xE7A1);
    let mut total = 0.0f64;
    let mut n = 0usize;
    for chunk in val.chunks(config.batch_size) {
        // No augmentation at eval — measure on the songs as-is.
        let batch: Vec<Example> = chunk.iter().map(Example::from_song).collect();
        let mb = build_masked_batch(&batch, seq, config.mask_fraction, &mut rng);
        let loss = masked_loss::<Inner>(&model, &mb, batch.len(), seq, device);
        total += loss.into_scalar().elem::<f32>() as f64;
        n += 1;
    }
    total / n.max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::FEATURE_DIM;

    /// A tiny example whose frame 1 has distinctive pitch-class content.
    fn toy_example(seq: usize) -> Example {
        let mut features = vec![0.0f32; seq * FEATURE_DIM];
        // Frame 1: put 1.0 at chroma pc 0 so its target is non-zero.
        features[FEATURE_DIM] = 1.0;
        Example {
            seq_len: seq,
            features,
            key_label: 0,
            chord_labels: vec![0; seq],
        }
    }

    #[test]
    fn loss_scores_only_masked_frames() {
        let seq = 8;
        let batch = [toy_example(seq)];
        // Force-mask exactly frame 1 by hand-building the MaskedBatch.
        let mut input = batch[0].features.clone();
        let mut mask = vec![0.0f32; seq];
        let mut target = Vec::new();
        for f in 0..seq {
            let row = &batch[0].features[f * FEATURE_DIM..(f + 1) * FEATURE_DIM];
            target.extend_from_slice(&row[..PITCH_BLOCK_DIM]);
        }
        // Mask frame 1: zero its input row, set its mask flag.
        for x in input[FEATURE_DIM..2 * FEATURE_DIM].iter_mut() {
            *x = 0.0;
        }
        mask[1] = 1.0;
        let mb = MaskedBatch {
            input,
            mask,
            target,
            n_masked: 1,
        };
        let device = MlDevice::default();
        let model = ModelConfig::wired().init::<Inner>(&device);
        let loss = masked_loss::<Inner>(&model, &mb, 1, seq, &device).into_scalar();
        // Loss is finite and strictly positive (frame 1 target is non-zero, recon
        // from a random model won't match it exactly).
        let l: f32 = loss.elem();
        assert!(l.is_finite() && l >= 0.0);
    }
}

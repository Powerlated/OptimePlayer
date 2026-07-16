//! The supervised fine-tune: per-frame factored chord cross-entropy + pooled key
//! cross-entropy + beat-aware smoothness, optimised with Adam.
//!
//! **One loop for every generation.** It is generic over [`Backbone`], so adding a
//! backbone means implementing that trait — not copying this file. A manual loop
//! (rather than Burn's `Learner`) keeps the multi-headed objective and its metrics
//! explicit.

use burn::config::Config;
use burn::module::AutodiffModule;
use burn::nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig};
use burn::optim::{AdamConfig, GradientsParams};
use burn::prelude::*;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use std::path::Path;

use crate::backbone::{self, Backbone};
use crate::backend::{Back, Inner, MlDevice};
use crate::dashboard::{self, EpochPoint, RunMeta};
use crate::notes::{random_transpose, Song};
use crate::parallel::{default_shards, dp_step};
use crate::progress::TrainProgress;
use crate::shared::{chord_key_loss, eval_counts, key_targets, root_quality_targets, EvalCounts};

/// Hyperparameters shared by every backbone. Deliberately **not** generic over the
/// model config: the three generations always trained with identical values here, and
/// keeping it concrete avoids putting Burn's `Config` derive on a generic struct. The
/// architecture config rides alongside as `&M::Cfg`.
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
    /// Weight of the **beat-aware** temporal-smoothness penalty on the chord
    /// predictions (see [`crate::shared::beat_aware_tv`]): penalises more than one
    /// chord change per beat. `0.0` disables it (the control condition).
    #[config(default = 0.1)]
    pub chord_smoothness_weight: f64,
    /// On-the-fly random transposition augmentation (key-invariance). The labels
    /// (key + per-frame chords) transpose with the notes.
    #[config(default = true)]
    pub augment: bool,
    #[config(default = 1234)]
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig::new()
    }
}

/// Build one batch from `songs[indices]`, applying per-shard transposition
/// augmentation.
fn build_batch<M, B>(songs: &[Song], indices: &[usize], augment: bool, rng: &mut StdRng) -> M::Batch
where
    B: Backend,
    M: Backbone<B>,
{
    let picked: Vec<Song> = indices
        .iter()
        .map(|&i| {
            let shift = if augment { random_transpose(rng) } else { 0 };
            songs[i].transpose(shift)
        })
        .collect();
    M::build_batch(&picked)
}

/// Fine-tune on `train`/`val`, writing `<out_dir>/<M::DIR>/model(.mpk)` +
/// `model.json`.
///
/// When `pretrained` is `Some(prefix)`, the trunk is warm-started from a
/// self-supervised checkpoint; the supervised heads there are still random, which is
/// exactly the warm start we want.
pub fn run<M>(
    config: &TrainConfig,
    model_cfg: &M::Cfg,
    train: &[Song],
    val: &[Song],
    out_dir: &Path,
    pretrained: Option<&Path>,
) where
    M: Backbone<Back> + AutodiffModule<Back>,
    M::InnerModule: Backbone<Inner, Batch = <M as Backbone<Back>>::Batch>,
{
    let device = MlDevice::default();
    let mut model: M = match pretrained {
        Some(prefix) => {
            println!("warm-starting {} trunk from {}", M::NAME, prefix.display());
            backbone::load::<M, Back>(prefix, &device)
        }
        None => M::init(model_cfg, &device),
    };
    let mut optim = AdamConfig::new().init();
    let ce: CrossEntropyLoss<Back> = CrossEntropyLossConfig::new().init(&device);

    let seq = train.first().map(|s| s.n_frames).unwrap_or(0);
    assert!(seq > 0, "empty training set");

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut indices: Vec<usize> = (0..train.len()).collect();
    let n_shards = std::env::var("DP_SHARDS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(default_shards);

    println!(
        "{} fine-tune: {} train / {} val, seq_len {seq}, smoothness {}, {n_shards}-way DP",
        M::NAME,
        train.len(),
        val.len(),
        config.chord_smoothness_weight,
    );
    dashboard::start(RunMeta {
        stage: "supervised fine-tune".to_string(),
        backbone: M::NAME.to_string(),
        backend: format!(
            "{}, {n_shards}-way DP",
            dashboard::backend_label(std::any::type_name::<Back>())
        ),
        epochs: config.epochs,
        batch_size: config.batch_size,
        lr: config.lr,
        train_windows: train.len(),
        val_windows: val.len(),
    });

    let n_total = indices.len().div_ceil(config.batch_size);

    for epoch in 1..=config.epochs {
        let epoch_start = std::time::Instant::now();
        indices.shuffle(&mut rng);
        let mut running = 0.0f64;
        let mut n_batches = 0usize;
        let mut prog = TrainProgress::per_epoch(epoch);

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
                |m: &M, shard| {
                    // Deterministic per-shard, per-epoch augmentation RNG (the
                    // closure runs on many threads).
                    let mut srng = StdRng::seed_from_u64(
                        config.seed
                            ^ (shard[0] as u64).wrapping_mul(0x9E3779B9)
                            ^ (epoch as u64).wrapping_mul(0x85EBCA77),
                    );
                    let data = build_batch::<M, Back>(train, shard, config.augment, &mut srng);
                    let (b, _) = M::dims(&data);
                    let out = m.forward_output(&data, &device);
                    let (root_t, quality_t) =
                        root_quality_targets::<Back>(M::chord_labels(&data), &device);
                    let key_t = key_targets::<Back>(M::key_labels(&data), &device);
                    // Scale by 1/k so the K shard gradients sum to the batch-mean gradient.
                    let l = chord_key_loss(
                        &ce,
                        &out,
                        root_t,
                        quality_t,
                        key_t,
                        b,
                        seq,
                        config.key_loss_weight,
                        config.chord_smoothness_weight,
                    )
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

        let (key_acc, chord_acc, changes) =
            evaluate::<M>(&model, val, config.batch_size, seq, &device);
        let train_loss = running / n_batches.max(1) as f64;
        let secs = epoch_start.elapsed().as_secs_f64();
        println!(
            "epoch {epoch:>3}/{}  loss {train_loss:.4}  |  val key acc {:.1}%  chord acc {:.1}%  changes/seq {changes:.1}  |  {secs:.1}s",
            config.epochs,
            key_acc * 100.0,
            chord_acc * 100.0,
        );
        dashboard::record_epoch(EpochPoint::supervised(
            epoch, train_loss, key_acc, chord_acc, changes, secs,
        ));
    }

    let dir = backbone::artifact_dir::<M, Back>(out_dir);
    backbone::save::<M, Back>(model, model_cfg, &dir, "model");
    println!("saved {} model to {}", M::NAME, dir.display());
    dashboard::finish(&dir);
}

/// Validation metrics `(key_acc, chord_acc, mean_chord_changes_per_sequence)`.
/// Chord accuracy ignores no-chord frames. The last value is the mean number of
/// frame-to-frame predicted-label transitions per `seq`-frame sequence — the flicker
/// proxy the smoothness penalty is meant to reduce.
pub fn evaluate<M>(
    model: &M,
    val: &[Song],
    batch_size: usize,
    seq: usize,
    device: &MlDevice,
) -> (f64, f64, f64)
where
    M: Backbone<Back> + AutodiffModule<Back>,
    M::InnerModule: Backbone<Inner, Batch = <M as Backbone<Back>>::Batch>,
{
    if val.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let model = model.valid();
    let mut acc = EvalCounts::default();
    for chunk in val.chunks(batch_size) {
        // No augmentation at eval.
        let data = <M::InnerModule as Backbone<Inner>>::build_batch(chunk);
        let (b, _) = <M::InnerModule as Backbone<Inner>>::dims(&data);
        let out = model.forward_output(&data, device);
        let c = eval_counts::<Inner>(
            &out,
            <M::InnerModule as Backbone<Inner>>::chord_labels(&data),
            <M::InnerModule as Backbone<Inner>>::key_labels(&data),
            b,
            seq,
        );
        acc.key_correct += c.key_correct;
        acc.chord_correct += c.chord_correct;
        acc.chord_total += c.chord_total;
        acc.chord_changes += c.chord_changes;
    }
    (
        acc.key_correct as f64 / val.len() as f64,
        acc.chord_correct as f64 / acc.chord_total.max(1) as f64,
        acc.chord_changes as f64 / val.len() as f64,
    )
}

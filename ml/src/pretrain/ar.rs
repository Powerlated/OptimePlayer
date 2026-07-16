//! Autoregressive next-frame pretraining, generic over [`ArBackbone`].
//!
//! The causal trunk reads frames left-to-right and, at each frame, predicts the
//! **next** frame's content — which pitch-classes and which channels are sounding
//! (multi-label, binary-cross-entropy-with-logits). A generative pretext on
//! *unlabeled* real songs; the trained trunk warm-starts the supervised fine-tune,
//! where the AR heads are discarded.
//!
//! Two drivers over the same loss:
//! * [`run`] — the CPU path, sharding each batch across cores via [`dp_step`].
//! * [`run_single_device`] — one batch at a time on any autodiff backend. The rayon
//!   data-parallel step is a CPU-only trick to fill cores (the ndarray backend tops
//!   out at ~2); a GPU has nothing to shard, so the `gpu` feature's bin uses this.

use burn::config::Config;
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::path::Path;

use crate::backbone::{self, ArBackbone, ArOutput, Backbone};
use crate::backend::{Back, Inner, MlDevice};
use crate::dashboard::{self, ContextWindow, DataStats, EpochPoint, RunMeta};
use crate::notes::{random_transpose, Song, N_TRANSPOSITIONS};
use crate::parallel::{default_shards, dp_step};
use crate::progress::TrainProgress;
use crate::tokenize::{N_CHANNELS, N_PC};

#[derive(Config, Debug)]
pub struct ArPretrainConfig {
    #[config(default = 20)]
    pub epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 3.0e-4)]
    pub lr: f64,
    #[config(default = true)]
    pub augment: bool,
    #[config(default = 4242)]
    pub seed: u64,
}

impl Default for ArPretrainConfig {
    fn default() -> Self {
        ArPretrainConfig::new()
    }
}

/// Numerically-stable binary-cross-entropy-with-logits, mean over all elements:
/// `max(x,0) - x*t + log(1 + exp(-|x|))`.
fn bce_with_logits<B: Backend>(logits: Tensor<B, 3>, targets: Tensor<B, 3>) -> Tensor<B, 1> {
    let max0 = logits.clone().clamp_min(0.0);
    let xt = logits.clone() * targets;
    let softplus = logits.abs().neg().exp().add_scalar(1.0).log();
    (max0 - xt + softplus).mean()
}

/// AR next-frame loss for one batch: predict frame `f+1`'s sounding pitch-classes +
/// channels from the causal hidden state at frame `f`.
pub fn ar_loss<M, B>(model: &M, data: &M::Batch, device: &B::Device) -> Tensor<B, 1>
where
    B: Backend,
    M: ArBackbone<B>,
{
    let (b, nf) = M::dims(data);
    let ArOutput {
        pc_logits,
        channel_logits,
    } = model.ar_forward(data, device);

    let (pc_t, ch_t) = M::ar_targets(data);
    let pc_target = Tensor::<B, 3>::from_data(TensorData::new(pc_t, [b, nf, N_PC]), device);
    let ch_target = Tensor::<B, 3>::from_data(TensorData::new(ch_t, [b, nf, N_CHANNELS]), device);

    // Shift by one frame: logits at f predict content at f+1.
    let pc_pred = pc_logits.slice([0..b, 0..nf - 1, 0..N_PC]);
    let pc_tgt = pc_target.slice([0..b, 1..nf, 0..N_PC]);
    let ch_pred = channel_logits.slice([0..b, 0..nf - 1, 0..N_CHANNELS]);
    let ch_tgt = ch_target.slice([0..b, 1..nf, 0..N_CHANNELS]);

    bce_with_logits(pc_pred, pc_tgt) + bce_with_logits(ch_pred, ch_tgt)
}

/// Tokenize `songs[indices]` with per-shard transposition augmentation.
fn build_batch<M, B>(songs: &[Song], indices: &[usize], augment: bool, rng: &mut StdRng) -> M::Batch
where
    B: Backend,
    M: ArBackbone<B>,
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

/// Window length of the dataset — every song in a split shares `n_frames`.
fn n_frames(songs: &[Song]) -> usize {
    songs.first().map(|s| s.n_frames).unwrap_or(0)
}

/// Distinct transpositions augmentation reaches; 1 when it is off.
fn transpositions(augment: bool) -> usize {
    if augment {
        N_TRANSPOSITIONS
    } else {
        1
    }
}

/// A `Config` as JSON for the dashboard's hyperparameter table. Serializing the
/// config itself means the table can't drift from the model.
fn config_json<C: burn::config::Config>(cfg: &C) -> serde_json::Value {
    serde_json::to_value(cfg).unwrap_or(serde_json::Value::Null)
}

/// Deterministic per-shard, per-epoch RNG for augmentation.
fn shard_rng(seed: u64, first: usize, epoch: usize) -> StdRng {
    StdRng::seed_from_u64(
        seed ^ (first as u64).wrapping_mul(0x9E3779B9) ^ (epoch as u64).wrapping_mul(0x85EBCA77),
    )
}

/// Pretrain on `train` (CPU, data-parallel), writing
/// `<out_dir>/<M::DIR>/pretrained(.mpk)` + `pretrained.json`.
pub fn run<M>(
    config: &ArPretrainConfig,
    model_cfg: &M::Cfg,
    train: &[Song],
    val: &[Song],
    out_dir: &Path,
) where
    M: ArBackbone<Back> + AutodiffModule<Back>,
    M::InnerModule: ArBackbone<Inner, Batch = <M as Backbone<Back>>::Batch>,
{
    let device = MlDevice::default();
    let mut model: M = M::init(model_cfg, &device);
    let mut optim = AdamConfig::new().init();

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut indices: Vec<usize> = (0..train.len()).collect();
    let n_shards = std::env::var("DP_SHARDS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(default_shards);

    println!(
        "{} AR pretrain: {} train / {} val windows, {n_shards}-way DP",
        M::NAME,
        train.len(),
        val.len()
    );
    let data = DataStats::measure(train, val, transpositions(config.augment));
    dashboard::start(RunMeta {
        stage: "AR pretrain".to_string(),
        backbone: M::NAME.to_string(),
        backend: format!(
            "{}, {n_shards}-way DP",
            dashboard::backend_label(std::any::type_name::<Back>())
        ),
        epochs: config.epochs,
        context: ContextWindow::from_frames(n_frames(train)),
        data,
        params: model.num_params(),
        flops_per_window: M::flops_per_window(model_cfg, data.notes_per_window.round() as usize),
        model_config: config_json(model_cfg),
        train_config: config_json(config),
    });

    let n_total = indices.len().div_ceil(config.batch_size);

    for epoch in 1..=config.epochs {
        let epoch_start = std::time::Instant::now();
        indices.shuffle(&mut rng);
        let mut running = 0.0f64;
        let mut n_batches = 0usize;
        let mut prog = TrainProgress::per_epoch(epoch);

        for chunk in indices.chunks(config.batch_size) {
            let k = n_shards.min(chunk.len().max(1));
            let (m, loss) = dp_step(
                model,
                &mut optim,
                config.lr,
                &device,
                chunk,
                k,
                |m: &M, shard| {
                    let mut srng = shard_rng(config.seed, shard[0], epoch);
                    let data = build_batch::<M, Back>(train, shard, config.augment, &mut srng);
                    // Scale by 1/k so the K shard gradients sum to the batch-mean gradient.
                    let l = ar_loss::<M, Back>(m, &data, &device).mul_scalar(1.0 / k as f64);
                    let grads = GradientsParams::from_grads(l.clone().backward(), m);
                    (grads, l.into_scalar().elem::<f32>() as f64)
                },
            );
            model = m;
            running += loss;
            n_batches += 1;
            prog.maybe_log(running, n_batches, n_total);
        }

        let val_loss = evaluate::<M>(&model, val, config.batch_size, &device);
        let train_loss = running / n_batches.max(1) as f64;
        let secs = epoch_start.elapsed().as_secs_f64();
        println!(
            "epoch {epoch:>3}/{}  AR loss {train_loss:.4}  |  val {val_loss:.4}  |  {secs:.1}s",
            config.epochs,
        );
        dashboard::record_epoch(EpochPoint::pretext(epoch, train_loss, val_loss, secs));
        backbone::save_epoch::<M, Back>(
            &model,
            model_cfg,
            &backbone::artifact_dir::<M, Back>(out_dir),
            "pretrained",
            epoch,
        );
    }

    let dir = backbone::artifact_dir::<M, Back>(out_dir);
    backbone::save::<M, Back>(model, model_cfg, &dir, "pretrained");
    println!("saved {} pretrained trunk to {}", M::NAME, dir.display());
    dashboard::finish(&dir);
}

/// Held-out AR loss (no augmentation), comparable across epochs.
pub fn evaluate<M>(model: &M, val: &[Song], batch_size: usize, device: &MlDevice) -> f64
where
    M: ArBackbone<Back> + AutodiffModule<Back>,
    M::InnerModule: ArBackbone<Inner, Batch = <M as Backbone<Back>>::Batch>,
{
    if val.is_empty() {
        return 0.0;
    }
    let model = model.valid();
    let mut total = 0.0f64;
    let mut n = 0usize;
    for chunk in val.chunks(batch_size) {
        let data = <M::InnerModule as Backbone<Inner>>::build_batch(chunk);
        let loss = ar_loss::<M::InnerModule, Inner>(&model, &data, device);
        total += loss.into_scalar().elem::<f32>() as f64;
        n += 1;
    }
    total / n.max(1) as f64
}

/// Pretrain on a single device with no batch sharding — the driver for non-CPU
/// backends, where `dp_step`'s rayon sharding has nothing to buy.
///
/// Backend-agnostic in, backend-agnostic out: [`CompactRecorder`] output from here
/// loads straight into the ndarray fine-tune.
///
/// [`CompactRecorder`]: burn::record::CompactRecorder
pub fn run_single_device<M, B>(
    config: &ArPretrainConfig,
    model_cfg: &M::Cfg,
    train: &[Song],
    val: &[Song],
    out_dir: &Path,
    device: &B::Device,
) where
    B: burn::tensor::backend::AutodiffBackend,
    M: ArBackbone<B> + AutodiffModule<B>,
    M::InnerModule: ArBackbone<B::InnerBackend, Batch = <M as Backbone<B>>::Batch>,
{
    let mut model: M = M::init(model_cfg, device);
    let mut optim = AdamConfig::new().init();

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut indices: Vec<usize> = (0..train.len()).collect();
    let n_total = indices.len().div_ceil(config.batch_size);

    println!(
        "{} AR pretrain (single-device): {} train / {} val windows, batch {}, lr {}",
        M::NAME,
        train.len(),
        val.len(),
        config.batch_size,
        config.lr
    );
    let data = DataStats::measure(train, val, transpositions(config.augment));
    dashboard::start(RunMeta {
        stage: "AR pretrain".to_string(),
        backbone: M::NAME.to_string(),
        backend: dashboard::backend_label(std::any::type_name::<B>()),
        epochs: config.epochs,
        context: ContextWindow::from_frames(n_frames(train)),
        data,
        params: model.num_params(),
        flops_per_window: M::flops_per_window(model_cfg, data.notes_per_window.round() as usize),
        model_config: config_json(model_cfg),
        train_config: config_json(config),
    });

    for epoch in 1..=config.epochs {
        let epoch_start = std::time::Instant::now();
        indices.shuffle(&mut rng);
        let mut running = 0.0f64;
        let mut n_batches = 0usize;
        let mut prog = TrainProgress::per_epoch(epoch);

        for chunk in indices.chunks(config.batch_size) {
            let mut srng = shard_rng(config.seed, chunk[0], epoch);
            let data = build_batch::<M, B>(train, chunk, config.augment, &mut srng);
            let l = ar_loss::<M, B>(&model, &data, device);
            let grads = GradientsParams::from_grads(l.clone().backward(), &model);
            model = optim.step(config.lr, model, grads);
            running += l.into_scalar().elem::<f32>() as f64;
            n_batches += 1;
            prog.maybe_log(running, n_batches, n_total);
        }

        let val_loss = evaluate_single_device::<M, B>(&model, val, config.batch_size, device);
        let train_loss = running / n_batches.max(1) as f64;
        let secs = epoch_start.elapsed().as_secs_f64();
        println!(
            "epoch {epoch:>3}/{}  AR loss {train_loss:.4}  |  val {val_loss:.4}  |  {secs:.1}s",
            config.epochs,
        );
        dashboard::record_epoch(EpochPoint::pretext(epoch, train_loss, val_loss, secs));
        backbone::save_epoch::<M, B>(
            &model,
            model_cfg,
            &backbone::artifact_dir::<M, B>(out_dir),
            "pretrained",
            epoch,
        );
    }

    let dir = backbone::artifact_dir::<M, B>(out_dir);
    backbone::save::<M, B>(model, model_cfg, &dir, "pretrained");
    println!("saved {} pretrained trunk to {}", M::NAME, dir.display());
    dashboard::finish(&dir);
}

/// Held-out AR loss on the inner (inference) backend — no augmentation, no grad.
fn evaluate_single_device<M, B>(
    model: &M,
    val: &[Song],
    batch_size: usize,
    device: &B::Device,
) -> f64
where
    B: burn::tensor::backend::AutodiffBackend,
    M: ArBackbone<B> + AutodiffModule<B>,
    M::InnerModule: ArBackbone<B::InnerBackend, Batch = <M as Backbone<B>>::Batch>,
{
    if val.is_empty() {
        return 0.0;
    }
    let valid = model.valid();
    let mut total = 0.0f64;
    let mut n = 0usize;
    for chunk in val.chunks(batch_size) {
        let data = <M::InnerModule as Backbone<B::InnerBackend>>::build_batch(chunk);
        let loss = ar_loss::<M::InnerModule, B::InnerBackend>(&valid, &data, device);
        total += loss.into_scalar().elem::<f32>() as f64;
        n += 1;
    }
    total / n.max(1) as f64
}

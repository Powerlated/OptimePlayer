//! Self-supervised pretraining on harvested real game songs, for any backbone.
//!
//! ```sh
//! cargo run --release --bin pretrain -- [epochs] [batch_size] [lr] \
//!     [--backbone frame|event|hier|kda] [--out-dir models]
//! #   reads data/real_train.bin + data/real_val.bin (produced by `harvest`)
//! #   → <out-dir>/<00-frame|01-event|02-hier>/pretrained (+ .json)
//! ```
//!
//! The pretext follows the backbone, because they are genuinely different objectives:
//! `frame` runs **masked-frame reconstruction** (it has a feature grid to mask),
//! while `event`/`hier` run **autoregressive next-frame prediction**.
//!
//! Replaces the old `pretrain` / `event_pretrain` / `hier_pretrain` trio.

use optime_ml::backend::Back;
use optime_ml::cli::{Args, Kind};
use optime_ml::data::load_songs;
use optime_ml::m01_event::{EventModel, EventModelConfig};
use optime_ml::m02_hier::{HierModel, HierModelConfig};
use optime_ml::m03_kda::{KdaModel, KdaModelConfig};
use optime_ml::pack::window_dataset;
use optime_ml::pretrain::ar::{self, ArPretrainConfig};
use optime_ml::pretrain::masked::{self, PretrainConfig};

/// Fixed-window generations (m00–m02) window the whole-song dataset at load time.
const FIXED_SEQ_LEN: usize = 256;

fn main() {
    let args = Args::parse();
    let epochs: usize = args.positional_or(0, 20);
    // A packed m03 batch carries 2048 slots per item and nearly fills an 8 GB
    // GPU at batch 8. Fixed-window generations retain their historical default.
    let default_batch = if args.kind == Kind::Kda { 8 } else { 32 };
    let batch_size: usize = args.positional_or(1, default_batch);
    let lr: f64 = args.positional_or(2, 3.0e-4);

    // Songs are transposed on the fly during training (see the configs' `augment`).
    // The dataset holds whole songs (intro+loop+loop, variable length): the
    // long-context backbone (kda) packs them per epoch; fixed-window backbones
    // slice them into 256-frame windows here.
    let mut train =
        load_songs("data/real_train.bin").expect("load data/real_train.bin (run `harvest` first)");
    let mut val = load_songs("data/real_val.bin").unwrap_or_default();
    println!(
        "loaded {} train / {} val whole real songs",
        train.len(),
        val.len()
    );
    if args.kind != Kind::Kda {
        train = window_dataset(&train, FIXED_SEQ_LEN);
        val = window_dataset(&val, FIXED_SEQ_LEN);
        println!(
            "windowed at {FIXED_SEQ_LEN} frames → {} train / {} val windows",
            train.len(),
            val.len()
        );
    }

    match args.kind {
        // Generation 00: masked-frame reconstruction over the feature grid.
        Kind::Frame => {
            let config = PretrainConfig::default()
                .with_epochs(epochs)
                .with_batch_size(batch_size)
                .with_lr(lr);
            masked::run(&config, &train, &val, &args.out_dir);
        }
        // Learned-token generations: autoregressive next-frame prediction.
        Kind::Event => {
            let config = ar_config(epochs, batch_size, lr);
            ar::run::<EventModel<Back>>(
                &config,
                &EventModelConfig::new(),
                &train,
                &val,
                &args.out_dir,
            );
        }
        Kind::Hier => {
            let config = ar_config(epochs, batch_size, lr);
            ar::run::<HierModel<Back>>(
                &config,
                &HierModelConfig::new(),
                &train,
                &val,
                &args.out_dir,
            );
        }
        Kind::Kda => {
            // Long-context generative backbone: multi-song sequence packing at
            // the model's nominal window.
            let model_cfg = KdaModelConfig::new();
            let config = ar_config(epochs, batch_size, lr).with_pack_to(model_cfg.n_frames);
            ar::run::<KdaModel<Back>>(&config, &model_cfg, &train, &val, &args.out_dir);
        }
    }
}

fn ar_config(epochs: usize, batch_size: usize, lr: f64) -> ArPretrainConfig {
    ArPretrainConfig::default()
        .with_epochs(epochs)
        .with_batch_size(batch_size)
        .with_lr(lr)
}

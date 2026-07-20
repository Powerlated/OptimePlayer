//! Self-supervised pretraining on harvested real game songs, for any backbone.
//!
//! ```sh
//! cargo run --release --bin pretrain -- [epochs] [batch_size] [lr] \
//!     [--backbone frame|event|hier] [--out-dir models]
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
use optime_ml::pretrain::ar::{self, ArPretrainConfig};
use optime_ml::pretrain::masked::{self, PretrainConfig};

fn main() {
    let args = Args::parse();
    let epochs: usize = args.positional_or(0, 20);
    let batch_size: usize = args.positional_or(1, 32);
    let lr: f64 = args.positional_or(2, 3.0e-4);

    // Songs are transposed on the fly during training (see the configs' `augment`).
    let train =
        load_songs("data/real_train.bin").expect("load data/real_train.bin (run `harvest` first)");
    let val = load_songs("data/real_val.bin").unwrap_or_default();
    println!(
        "loaded {} train / {} val real windows",
        train.len(),
        val.len()
    );

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
    }
}

fn ar_config(epochs: usize, batch_size: usize, lr: f64) -> ArPretrainConfig {
    ArPretrainConfig::default()
        .with_epochs(epochs)
        .with_batch_size(batch_size)
        .with_lr(lr)
}

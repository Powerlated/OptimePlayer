//! Train the key/chord transformer on the retained synthetic dataset.
//!
//! Usage:
//!   cargo run --release --bin train -- [epochs] [batch_size] [lr] [--pretrained <prefix>]
//!
//! Reads data/train.bin + data/val.bin (run `generate_data` first), writes the
//! trained model to models/. With `--pretrained models/pretrained`, the encoder
//! is warm-started from a self-supervised checkpoint; without it, the run is a
//! from-scratch synthetic-only baseline (identical to before).

use optime_ml::data::load_songs;
use optime_ml::model::ModelConfig;
use optime_ml::train::{self, TrainConfig};
use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = TrainConfig::new(ModelConfig::wired());

    // Separate the optional `--pretrained <prefix>` flag from the positional args.
    let mut pretrained: Option<PathBuf> = None;
    let mut positional: Vec<String> = Vec::new();
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        if a == "--pretrained" {
            pretrained = Some(PathBuf::from(
                it.next().expect("--pretrained needs a path prefix"),
            ));
        } else {
            positional.push(a.clone());
        }
    }
    if let Some(v) = positional.first() {
        cfg.epochs = v.parse().expect("epochs");
    }
    if let Some(v) = positional.get(1) {
        cfg.batch_size = v.parse().expect("batch_size");
    }
    if let Some(v) = positional.get(2) {
        cfg.lr = v.parse().expect("lr");
    }

    let data_dir = Path::new("data");
    if !data_dir.join("train.bin").exists() {
        eprintln!("data/train.bin not found — run `cargo run --release --bin generate_data` first");
        std::process::exit(1);
    }

    println!("loading dataset ...");
    // Songs are transposed on the fly during training (see `TrainConfig::augment`).
    let train_songs = load_songs(data_dir.join("train.bin")).expect("load train");
    let val_songs = load_songs(data_dir.join("val.bin")).expect("load val");

    let out = PathBuf::from("models");
    train::run(&cfg, &train_songs, &val_songs, &out, pretrained.as_deref());
}

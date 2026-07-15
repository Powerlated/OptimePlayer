//! Self-supervised masked-frame pretraining on harvested real game songs.
//!
//! ```sh
//! cargo run --release --bin pretrain -- [epochs] [batch_size] [lr]
//! #   reads data/real_train.bin + data/real_val.bin (produced by `harvest`)
//! #   → models/pretrained (+ models/pretrained.json)
//! ```

use optime_ml::data::load_songs;
use optime_ml::pretrain::{self, PretrainConfig};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let epochs: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let batch_size: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let lr: Option<f64> = args.get(3).and_then(|s| s.parse().ok());

    // Songs are transposed on the fly during training (see `PretrainConfig::augment`).
    let train =
        load_songs("data/real_train.bin").expect("load data/real_train.bin (run `harvest` first)");
    let val = load_songs("data/real_val.bin").unwrap_or_default();
    println!(
        "loaded {} train / {} val real windows",
        train.len(),
        val.len()
    );

    let mut config = PretrainConfig::default()
        .with_epochs(epochs)
        .with_batch_size(batch_size);
    if let Some(lr) = lr {
        config = config.with_lr(lr);
    }

    pretrain::run(&config, &train, &val, Path::new("models"));
}

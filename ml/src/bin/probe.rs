//! Train the frozen "is-music" linear probe on harvested, weakly-labeled songs.
//!
//! ```sh
//! # First harvest with annotations so real windows carry is-music labels:
//! cargo run --release --features harvest --bin harvest -- ../demos 128 \
//!     --annotate BPEE=../crates/optime-app/src/song_names/pokemon_emerald.json \
//!     --annotate A3UJ=../crates/optime-app/src/song_names/mother_3.json
//! # Then probe on top of a (pre)trained encoder:
//! cargo run --release --bin probe -- [start_prefix]
//! #   start_prefix defaults to models/model, else models/pretrained
//! #   → models/probe(.mpk) + models/probe.json
//! ```

use burn::config::Config;
use burn::module::Module;
use burn::record::CompactRecorder;
use optime_ml::data::load_songs;
use optime_ml::model::{KeyChordModel, ModelConfig};
use optime_ml::probe::{self, build_music_set, ProbeConfig};
use optime_ml::train::Back;
use optime_ml::train::MlDevice;
use std::path::{Path, PathBuf};

fn load(prefix: &Path, device: &MlDevice) -> (KeyChordModel<Back>, ModelConfig) {
    let config = ModelConfig::load(prefix.with_extension("json"))
        .unwrap_or_else(|_| panic!("load {}.json", prefix.display()));
    let model = config
        .init::<Back>(device)
        .load_file(prefix, &CompactRecorder::new(), device)
        .unwrap_or_else(|_| panic!("load {} weights", prefix.display()));
    (model, config)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let device = MlDevice::default();

    // Pick the starting encoder: explicit arg, else fine-tuned model, else pretrained.
    let prefix: PathBuf = match args.get(1) {
        Some(p) => PathBuf::from(p),
        None if Path::new("models/model.json").exists() => PathBuf::from("models/model"),
        None => PathBuf::from("models/pretrained"),
    };
    if !prefix.with_extension("json").exists() {
        eprintln!(
            "{}.json not found — pretrain/train an encoder first",
            prefix.display()
        );
        std::process::exit(1);
    }

    let train_songs =
        load_songs("data/real_train.bin").expect("load data/real_train.bin (run `harvest` first)");
    let val_songs = load_songs("data/real_val.bin").unwrap_or_default();

    let (model, config) = load(&prefix, &device);
    println!("frozen encoder: {}", prefix.display());

    let train = build_music_set(&model, &train_songs, 64, &device);
    let val = build_music_set(&model, &val_songs, 64, &device);
    if train.is_empty() {
        eprintln!(
            "no is-music labels in data/real_train.bin — re-run `harvest` with --annotate CODE=json"
        );
        std::process::exit(1);
    }

    probe::run(
        model,
        &ProbeConfig::default(),
        &config,
        &train,
        &val,
        Path::new("models"),
    );
}

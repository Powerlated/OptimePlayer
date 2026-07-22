//! Train the frozen "is-music" linear probe on harvested, weakly-labeled songs.
//!
//! ```sh
//! # First harvest with annotations so real windows carry is-music labels:
//! cargo run --release --features harvest --bin harvest -- ../demos 128 \
//!     --annotate BPEE=../crates/optime-app/src/song_names/pokemon_emerald.json \
//!     --annotate A3UJ=../crates/optime-app/src/song_names/mother_3.json
//! # Then probe on top of a (pre)trained encoder:
//! cargo run --release --bin probe -- [start_prefix] [--out-dir models]
//! #   start_prefix defaults to <out-dir>/00-frame/model, else .../pretrained
//! #   → <out-dir>/probe(.mpk) + probe.json
//! ```
//!
//! Generation 00 only: the probe reads that backbone's pooled encoder features and
//! its dedicated is-music head, which the learned-token generations don't carry.

use burn::config::Config;
use burn::module::Module;
use burn::record::CompactRecorder;
use optime_ml::backend::{Back, MlDevice};
use optime_ml::cli::{Args, Kind};
use optime_ml::data::load_songs;
use optime_ml::m00_frame::{KeyChordModel, ModelConfig};
use optime_ml::probe::{self, build_music_set, ProbeConfig};
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
    let args = Args::parse();
    let device = MlDevice::default();
    let frame_dir = args.out_dir.join(Kind::Frame.dir());

    // Pick the starting encoder: explicit arg, else fine-tuned model, else pretrained.
    let prefix: PathBuf = match args.positional.first() {
        Some(p) => PathBuf::from(p),
        None if frame_dir.join("model.json").exists() => frame_dir.join("model"),
        None => frame_dir.join("pretrained"),
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
    // Whole-song dataset → 256-frame windows (the probe rides m00's fixed grid).
    let train_songs = optime_ml::pack::window_dataset(&train_songs, 256);
    let val_songs = optime_ml::pack::window_dataset(&val_songs, 256);

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
        &args.out_dir,
    );
}

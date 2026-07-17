//! Train the frozen "is-music" linear probe on harvested, weakly-labeled songs → `<out-dir>/probe`.
//! Generation 00 only (it carries the pooled encoder features + is-music head the learned-token
//! generations don't). `start_prefix` defaults to `<out-dir>/00-frame/model`, else `.../pretrained`.
//! Re-harvest with `--annotate` first so real windows carry is-music labels.

use super::opts::{Backbone, ModelOpts};
use burn::config::Config;
use burn::module::Module;
use burn::record::CompactRecorder;
use clap::Args;
use optime_ml::backend::{Back, MlDevice};
use optime_ml::data::load_songs;
use optime_ml::m00_frame::{KeyChordModel, ModelConfig};
use optime_ml::probe::{self, build_music_set, ProbeConfig};
use std::path::{Path, PathBuf};

#[derive(Args, Debug)]
pub struct ProbeArgs {
    /// Frozen encoder prefix; defaults to the frame model, else its pretrained checkpoint.
    pub start_prefix: Option<PathBuf>,
    #[command(flatten)]
    pub opts: ModelOpts,
}

fn load(prefix: &Path, device: &MlDevice) -> (KeyChordModel<Back>, ModelConfig) {
    let config = ModelConfig::load(prefix.with_extension("json"))
        .unwrap_or_else(|_| panic!("load {}.json", prefix.display()));
    let model = config
        .init::<Back>(device)
        .load_file(prefix, &CompactRecorder::new(), device)
        .unwrap_or_else(|_| panic!("load {} weights", prefix.display()));
    (model, config)
}

pub fn run(args: ProbeArgs) {
    let device = MlDevice::default();
    let frame_dir = args.opts.out_dir.join(Backbone::Frame.dir());

    // Pick the starting encoder: explicit arg, else fine-tuned model, else pretrained.
    let prefix: PathBuf = match args.start_prefix {
        Some(p) => p,
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
        &args.opts.out_dir,
    );
}

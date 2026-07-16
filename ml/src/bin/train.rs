//! Supervised fine-tune on the retained synthetic dataset, for any backbone.
//!
//! ```sh
//! cargo run --release --bin train -- [epochs] [batch_size] [lr] \
//!     [--backbone frame|event|hier] [--out-dir models] [--pretrained <prefix>]
//! #   reads data/train.bin + data/val.bin (run `generate_data` first)
//! #   → <out-dir>/<00-frame|01-event|02-hier>/model (+ .json)
//! ```
//!
//! With `--pretrained <prefix>` the trunk is warm-started from a self-supervised
//! checkpoint; without it the run is a from-scratch synthetic-only baseline.
//! `CHORD_SMOOTH=<w>` overrides the beat-aware smoothness weight (0 = control);
//! `DP_SHARDS=<n>` overrides the data-parallel width.
//!
//! Replaces the old `train` / `event_train` / `hier_train` trio — the loop is generic
//! over the backbone, so this bin only picks the concrete type.

use burn::module::AutodiffModule;
use optime_ml::backbone::Backbone;
use optime_ml::backend::{Back, Inner};
use optime_ml::cli::{Args, Kind};
use optime_ml::data::load_songs;
use optime_ml::m00_frame::FrameModel;
use optime_ml::m01_event::EventModel;
use optime_ml::m02_hier::HierModel;
use optime_ml::notes::Song;
use optime_ml::train::{self, TrainConfig};
use std::path::Path;

fn main() {
    let args = Args::parse();
    let mut cfg = TrainConfig::new();
    cfg.epochs = args.positional_or(0, cfg.epochs);
    cfg.batch_size = args.positional_or(1, cfg.batch_size);
    cfg.lr = args.positional_or(2, cfg.lr);
    // `CHORD_SMOOTH=0` reproduces the no-penalty control.
    if let Ok(w) = std::env::var("CHORD_SMOOTH") {
        cfg.chord_smoothness_weight = w.parse().expect("CHORD_SMOOTH");
    }

    if !Path::new("data/train.bin").exists() {
        eprintln!("data/train.bin not found — run `cargo run --release --bin generate_data` first");
        std::process::exit(1);
    }
    println!("loading dataset ...");
    // Songs are transposed on the fly during training (see `TrainConfig::augment`).
    let train_songs = load_songs("data/train.bin").expect("load train");
    let val_songs = load_songs("data/val.bin").expect("load val");

    match args.kind {
        Kind::Frame => go::<FrameModel<Back>>(&cfg, &train_songs, &val_songs, &args),
        Kind::Event => go::<EventModel<Back>>(&cfg, &train_songs, &val_songs, &args),
        Kind::Hier => go::<HierModel<Back>>(&cfg, &train_songs, &val_songs, &args),
    }
}

fn go<M>(cfg: &TrainConfig, train_songs: &[Song], val_songs: &[Song], args: &Args)
where
    M: Backbone<Back> + AutodiffModule<Back>,
    M::InnerModule: Backbone<Inner, Batch = <M as Backbone<Back>>::Batch>,
{
    train::run::<M>(
        cfg,
        &M::default_cfg(),
        train_songs,
        val_songs,
        &args.out_dir,
        args.pretrained.as_deref(),
        "model",
    );
}

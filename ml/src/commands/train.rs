//! Supervised fine-tune on the retained synthetic dataset → `<out-dir>/<NN>/model`. Reads
//! `data/{train,val}.bin` (run `generate-data` first). `--pretrained <prefix>` warm-starts the
//! trunk; without it the run is a from-scratch synthetic-only baseline. `CHORD_SMOOTH=<w>`
//! overrides the smoothness weight, `DP_SHARDS=<n>` the data-parallel width.

use super::opts::{Backbone, ModelOpts};
use burn::module::AutodiffModule;
use clap::Args;
use optime_ml::backend::{Back, Inner};
use optime_ml::data::load_songs;
use optime_ml::m00_frame::FrameModel;
use optime_ml::m01_event::EventModel;
use optime_ml::m02_hier::HierModel;
use optime_ml::notes::Song;
use optime_ml::train::{self, TrainConfig};
use std::path::Path;

#[derive(Args, Debug)]
pub struct TrainArgs {
    /// Epochs (default from `TrainConfig`).
    pub epochs: Option<usize>,
    /// Batch size.
    pub batch_size: Option<usize>,
    /// Learning rate.
    pub lr: Option<f64>,
    #[command(flatten)]
    pub opts: ModelOpts,
}

pub fn run(args: TrainArgs) {
    let mut cfg = TrainConfig::new();
    if let Some(v) = args.epochs {
        cfg.epochs = v;
    }
    if let Some(v) = args.batch_size {
        cfg.batch_size = v;
    }
    if let Some(v) = args.lr {
        cfg.lr = v;
    }
    // `CHORD_SMOOTH=0` reproduces the no-penalty control.
    if let Ok(w) = std::env::var("CHORD_SMOOTH") {
        cfg.chord_smoothness_weight = w.parse().expect("CHORD_SMOOTH");
    }

    if !Path::new("data/train.bin").exists() {
        eprintln!("data/train.bin not found — run `generate-data` first");
        std::process::exit(1);
    }
    println!("loading dataset ...");
    // Songs are transposed on the fly during training (see `TrainConfig::augment`).
    let train_songs = load_songs("data/train.bin").expect("load train");
    let val_songs = load_songs("data/val.bin").expect("load val");

    match args.opts.backbone {
        Backbone::Frame => go::<FrameModel<Back>>(&cfg, &train_songs, &val_songs, &args.opts),
        Backbone::Event => go::<EventModel<Back>>(&cfg, &train_songs, &val_songs, &args.opts),
        Backbone::Hier => go::<HierModel<Back>>(&cfg, &train_songs, &val_songs, &args.opts),
    }
}

fn go<M>(cfg: &TrainConfig, train_songs: &[Song], val_songs: &[Song], opts: &ModelOpts)
where
    M: optime_ml::backbone::Backbone<Back> + AutodiffModule<Back>,
    M::InnerModule: optime_ml::backbone::Backbone<
        Inner,
        Batch = <M as optime_ml::backbone::Backbone<Back>>::Batch,
    >,
{
    train::run::<M>(
        cfg,
        &M::default_cfg(),
        train_songs,
        val_songs,
        &opts.out_dir,
        opts.pretrained.as_deref(),
        "model",
    );
}

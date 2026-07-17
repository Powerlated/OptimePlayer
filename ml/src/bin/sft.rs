//! Stage 3: supervised fine-tune on **hand-labelled real songs**.
//!
//! ```sh
//! cargo run --release --features harvest --bin sft -- [epochs] [batch] [lr] \
//!     [--backbone frame|event|hier] [--pretrained models/02-hier/model] [--out-dir models] \
//!     [--train-songs 360] [--val-songs 362]
//! #   reads ml/annotations/*.json + ../demos → <dir>/<NN-name>/model_sft
//! ```
//!
//! `--train-songs`/`--val-songs` replace the hash holdout with a hand-picked one, for contrived
//! experiments on a specific pair of songs. They exclude every song they don't name, and a result
//! from them is not the real-music metric — see [`Split::Songs`].
//!
//! The pipeline is SSL-pretrain on real songs → SFT on synthetic theory labels → **this**. Large and
//! noisy first, small and true last, so the human labels get the final word on what the heads mean.
//! Warm-start from the synthetic fine-tune (`--pretrained <dir>/<NN>/model`), not from the SSL
//! checkpoint: the synthetic stage is what teaches the label space at all, and a few hundred real
//! windows can adjust that mapping but cannot invent it.
//!
//! **The held-out songs are never trained on.** `sft` takes [`Split::Train`], `eval_labeled` takes
//! [`Split::Val`], and both derive it from the same deterministic song-level hash — the alternative
//! (train on everything, score on everything) would quietly destroy the only trustworthy number in
//! the project. The hand-picked flags are the one way around that, which is why they check
//! disjointness themselves and shout about it.
//!
//! Does not copy the training loop: it prepares a dataset and hands it to [`train::run`], the one
//! supervised loop every generation shares.

use burn::module::AutodiffModule;
use optime_ml::annotations::{build, Split, DEFAULT_VAL_FRAC};
use optime_ml::backbone::Backbone;
use optime_ml::backend::{Back, Inner};
use optime_ml::cli::{Args, Kind};
use optime_ml::m00_frame::FrameModel;
use optime_ml::m01_event::EventModel;
use optime_ml::m02_hier::HierModel;
use optime_ml::notes::Song;
use optime_ml::train::{self, TrainConfig};
use std::path::{Path, PathBuf};

/// Must match the window every other stage uses; a dataset windowed at one length cannot train a
/// model built for another.
const SEQ_LEN: usize = 256;

/// Artifact stem for the real-label stage, kept distinct from the synthetic `model` it starts from.
const SFT_NAME: &str = "model_sft";

fn main() {
    let args = Args::parse();
    let mut cfg = TrainConfig::new();
    // Real labels are scarce and the trunk is already trained: fewer epochs and a gentler rate than
    // the synthetic stage, so this adjusts the heads rather than overwriting them.
    cfg.epochs = args.positional_or(0, 8);
    cfg.batch_size = args.positional_or(1, 16);
    cfg.lr = args.positional_or(2, 1.0e-4);
    if let Ok(w) = std::env::var("CHORD_SMOOTH") {
        cfg.chord_smoothness_weight = w.parse().expect("CHORD_SMOOTH");
    }

    let ann_dir =
        PathBuf::from(std::env::var("ML_ANNOTATIONS").unwrap_or_else(|_| "annotations".into()));
    let rom_dir = PathBuf::from(std::env::var("ML_ROMS").unwrap_or_else(|_| "../demos".into()));

    // A hand-picked split overrides the deterministic holdout. It exists for contrived experiments
    // ("train on this song, score on that one"); it bypasses `is_val`, so the disjointness that
    // hash guarantees for free has to be checked here instead.
    let (train_split, val_split) = args
        .explicit_split()
        .unwrap_or((Split::Train(DEFAULT_VAL_FRAC), Split::Val(DEFAULT_VAL_FRAC)));
    if let (Some(t), Some(v)) = (&args.train_songs, &args.val_songs) {
        let both: Vec<u32> = t.iter().filter(|id| v.contains(id)).copied().collect();
        if !both.is_empty() {
            eprintln!(
                "--train-songs and --val-songs both name song(s) {both:?}. That trains on the \
                 songs it then scores, which is the one thing the split exists to prevent."
            );
            std::process::exit(1);
        }
    }

    let (train_songs, tstats) =
        match build::songs_from_dir(&ann_dir, &rom_dir, SEQ_LEN, &train_split) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        };
    let (val_songs, _) =
        build::songs_from_dir(&ann_dir, &rom_dir, SEQ_LEN, &val_split).unwrap_or_default();

    match args.explicit_split() {
        Some(_) => {
            println!(
                "hand-labelled: {} train / {} val windows (⚠ CONTRIVED hand-picked split: train \
                 {:?}, val {:?})",
                train_songs.len(),
                val_songs.len(),
                args.train_songs.clone().unwrap_or_default(),
                args.val_songs.clone().unwrap_or_default(),
            );
            println!(
                "⚠ this is not the deterministic holdout. Songs named by neither flag are excluded \
                 entirely, and nothing here is comparable to a default run — do not report it as \
                 the real-music metric."
            );
        }
        None => println!(
            "hand-labelled: {} train / {} val windows (song-level {:.0}% holdout)",
            train_songs.len(),
            val_songs.len(),
            DEFAULT_VAL_FRAC * 100.0
        ),
    }
    println!(
        "dropped: {} with an uncertain quality, {} with no notes; {} song(s) had no key; \
         {} labelled frame(s) left over (no whole window)",
        tstats.dropped_uncertain,
        tstats.dropped_no_notes,
        tstats.songs_missing_key,
        tstats.leftover_frames
    );

    if train_songs.is_empty() {
        eprintln!(
            "\nNothing to fine-tune on yet. A window needs all {SEQ_LEN} frames (= {} bars of 4/4) \
             annotated, and its song needs a key set.\nKeep labelling contiguous bars — partial \
             windows are dropped rather than back-filled with N.C., because an unheard bar is not a \
             rest.",
            SEQ_LEN / 16
        );
        std::process::exit(1);
    }
    if train_songs.len() < 32 {
        println!(
            "\n⚠ only {} training window(s). Expect this to overfit rather than teach; treat any \
             result as provisional until the set is much larger.",
            train_songs.len()
        );
    }

    // Default warm-start: the synthetic fine-tune, which is where the label space comes from.
    let pretrained = args
        .pretrained
        .clone()
        .unwrap_or_else(|| args.out_dir.join(args.kind.dir()).join("model"));
    if !pretrained.with_extension("json").exists() {
        eprintln!(
            "{}.json not found — run `train --backbone {}` first; SFT adjusts that model, it does \
             not replace it",
            pretrained.display(),
            args.kind.name()
        );
        std::process::exit(1);
    }

    match args.kind {
        Kind::Frame => go::<FrameModel<Back>>(&cfg, &train_songs, &val_songs, &args, &pretrained),
        Kind::Event => go::<EventModel<Back>>(&cfg, &train_songs, &val_songs, &args, &pretrained),
        Kind::Hier => go::<HierModel<Back>>(&cfg, &train_songs, &val_songs, &args, &pretrained),
    }
}

fn go<M>(
    cfg: &TrainConfig,
    train_songs: &[Song],
    val_songs: &[Song],
    args: &Args,
    pretrained: &Path,
) where
    M: Backbone<Back> + AutodiffModule<Back>,
    M::InnerModule: Backbone<Inner, Batch = <M as Backbone<Back>>::Batch>,
{
    train::run::<M>(
        cfg,
        &M::default_cfg(),
        train_songs,
        val_songs,
        &args.out_dir,
        Some(pretrained),
        // Never "model": that is what we warm-start from.
        SFT_NAME,
    );
}

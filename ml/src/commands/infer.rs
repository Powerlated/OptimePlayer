//! Run a trained model on a song and print predicted key + chord timeline next to ground truth.
//! With no index, generates a fresh random song; with an index, uses that song from `data/val.bin`.
//! Reads `<out-dir>/<NN>/model`.

use super::opts::{Backbone, ModelOpts};
use burn::module::AutodiffModule;
use clap::Args;
use optime_ml::backbone;
use optime_ml::backend::{Back, Inner, MlDevice};
use optime_ml::data::load_songs;
use optime_ml::infer::{merge_segments, predict};
use optime_ml::m00_frame::FrameModel;
use optime_ml::m01_event::EventModel;
use optime_ml::m02_hier::HierModel;
use optime_ml::notes::{render_song, Song};
use optime_ml::theory::{Chord, Key};
use rand::{rngs::StdRng, SeedableRng};
use std::path::Path;

#[derive(Args, Debug)]
pub struct InferArgs {
    /// Index into `data/val.bin`; omit for a fresh random song.
    pub val_index: Option<usize>,
    #[command(flatten)]
    pub opts: ModelOpts,
}

pub fn run(args: InferArgs) {
    let device = MlDevice::default();
    let backbone = args.opts.backbone;
    let prefix = args.opts.out_dir.join(backbone.dir()).join("model");
    if !prefix.with_extension("json").exists() {
        eprintln!(
            "{}.json not found — run `train --backbone {}` first",
            prefix.display(),
            backbone.name()
        );
        std::process::exit(1);
    }

    let song = if let Some(idx) = args.val_index {
        let songs = load_songs(Path::new("data").join("val.bin")).expect("load val");
        songs.get(idx).cloned().unwrap_or_else(|| {
            eprintln!("index {idx} out of range (0..{})", songs.len());
            std::process::exit(1);
        })
    } else {
        let mut rng = StdRng::from_entropy();
        let key = Key::from_label(rand::Rng::gen_range(&mut rng, 0..24));
        render_song(&mut rng, &key, 128)
    };

    let pred = match backbone {
        Backbone::Frame => predict_song::<FrameModel<Back>>(&prefix, &song, &device),
        Backbone::Event => predict_song::<EventModel<Back>>(&prefix, &song, &device),
        Backbone::Hier => predict_song::<HierModel<Back>>(&prefix, &song, &device),
    };

    println!("=== PREDICTION ({}) ===", backbone.name());
    print!("{}", pred.describe());

    println!("\n=== GROUND TRUTH ===");
    let true_key = Key::from_label(song.key_label);
    println!("key: {}", true_key.name());
    for seg in merge_segments(&song.chord_labels) {
        let name = seg
            .chord
            .map(|c: Chord| c.name())
            .unwrap_or_else(|| "—".to_string());
        println!(
            "  frames {:>4}..{:<4}  {}",
            seg.start_frame, seg.end_frame, name
        );
    }

    // Quick scoring.
    let key_ok = pred.key.label() == song.key_label;
    let chord_ok = pred
        .chord_labels
        .iter()
        .zip(&song.chord_labels)
        .filter(|(_, &t)| t != optime_ml::theory::NO_CHORD)
        .filter(|(p, t)| p == t)
        .count();
    let chord_total = song
        .chord_labels
        .iter()
        .filter(|&&t| t != optime_ml::theory::NO_CHORD)
        .count();
    println!(
        "\nkey correct: {}   frame chord accuracy: {:.1}%",
        key_ok,
        100.0 * chord_ok as f64 / chord_total.max(1) as f64
    );
}

fn predict_song<M>(prefix: &Path, song: &Song, device: &MlDevice) -> optime_ml::infer::Prediction
where
    M: optime_ml::backbone::Backbone<Back> + AutodiffModule<Back>,
    M::InnerModule: optime_ml::backbone::Backbone<
        Inner,
        Batch = <M as optime_ml::backbone::Backbone<Back>>::Batch,
    >,
{
    let model = backbone::load::<M, Back>(prefix, device);
    predict::<M>(&model, &song.notes, song.n_frames, device)
}

//! Run a trained model on a song and print the predicted key + chord timeline
//! alongside the ground truth.
//!
//! ```sh
//! cargo run --release --bin infer -- [val_index] [--backbone frame|event|hier|kda] [--out-dir models]
//! ```
//!
//! With no positional arg, generates a fresh random song. With an index, uses that
//! song from data/val.bin. Reads `<out-dir>/<backbone dir>/model`.

use burn::module::AutodiffModule;
use optime_ml::backbone::{self, Backbone};
use optime_ml::backend::{Back, Inner, MlDevice};
use optime_ml::cli::{Args, Kind};
use optime_ml::data::load_songs;
use optime_ml::infer::{merge_segments, predict};
use optime_ml::m00_frame::FrameModel;
use optime_ml::m01_event::EventModel;
use optime_ml::m02_hier::HierModel;
use optime_ml::m03_kda::KdaModel;
use optime_ml::notes::{render_song, Song};
use optime_ml::theory::{Chord, Key};
use rand::{rngs::StdRng, SeedableRng};
use std::path::Path;

fn main() {
    let device = MlDevice::default();
    let args = Args::parse();
    let prefix = args.out_dir.join(args.kind.dir()).join("model");
    if !prefix.with_extension("json").exists() {
        eprintln!(
            "{}.json not found — run `cargo run --release --bin train -- --backbone {}` first",
            prefix.display(),
            args.kind.name()
        );
        std::process::exit(1);
    }

    let song = if let Some(idx) = args
        .positional
        .first()
        .and_then(|s| s.parse::<usize>().ok())
    {
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

    let pred = match args.kind {
        Kind::Frame => run::<FrameModel<Back>>(&prefix, &song, &device),
        Kind::Event => run::<EventModel<Back>>(&prefix, &song, &device),
        Kind::Hier => run::<HierModel<Back>>(&prefix, &song, &device),
        Kind::Kda => run::<KdaModel<Back>>(&prefix, &song, &device),
    };

    println!("=== PREDICTION ({}) ===", args.kind.name());
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

fn run<M>(prefix: &Path, song: &Song, device: &MlDevice) -> optime_ml::infer::Prediction
where
    M: Backbone<Back> + AutodiffModule<Back>,
    M::InnerModule: Backbone<Inner, Batch = <M as Backbone<Back>>::Batch>,
{
    let model = backbone::load::<M, Back>(prefix, device);
    predict::<M>(&model, &song.notes, song.n_frames, device)
}

//! Run the trained model on a song and print the predicted key + chord timeline
//! alongside the ground truth.
//!
//! Usage:
//!   cargo run --release --bin infer -- [val_index]
//!
//! With no arg, generates a fresh random song. With an index, uses that song
//! from data/val.bin. Reads the model from models/.

use optime_ml::data::load_songs;
use optime_ml::infer::{merge_segments, predict};
use optime_ml::notes::render_song;
use optime_ml::theory::{Chord, Key};
use optime_ml::train::load_model;
use optime_ml::train::MlDevice;
use rand::{rngs::StdRng, SeedableRng};
use std::path::Path;

fn main() {
    let device = MlDevice::default();
    let model_dir = Path::new("models");
    if !model_dir.join("model.json").exists() {
        eprintln!("models/ not found — run `cargo run --release --bin train` first");
        std::process::exit(1);
    }
    let model = load_model(model_dir, &device);

    let args: Vec<String> = std::env::args().collect();
    let song = if let Some(idx) = args.get(1).and_then(|s| s.parse::<usize>().ok()) {
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

    let pred = predict(&model, &song.notes, song.n_frames, &device);

    println!("=== PREDICTION ===");
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

//! Measure the real-music gap: how well models fit real game songs.
//!
//! ```sh
//! cargo run --release --bin eval_real -- [real_val.bin] [model_prefix]
//! ```
//!
//! Real game songs are unlabeled, so this reports two comparable numbers on a
//! held-out set of harvested real windows:
//!
//! 1. **SSL distribution fit** — masked-frame reconstruction loss of a pretrained
//!    encoder (`models/pretrained`) on real windows vs. on the synthetic
//!    validation set (`data/val.bin`). The direct domain-gap number.
//! 2. **Chord agreement %** — a trained model (`models/model` by default) vs. the
//!    training-free chroma-template + Viterbi reference ([`optime_ml::estimate`]).
//!    Run it once on a synthetic-only baseline model and once on the
//!    SSL+fine-tuned model to see the pivot's effect as one number.
//!
//! Missing checkpoints are skipped with a note, so this is safe to run at any
//! stage of the pipeline.

use burn::config::Config;
use burn::module::Module;
use burn::record::CompactRecorder;
use optime_ml::data::load_songs;
use optime_ml::model::{KeyChordModel, ModelConfig};
use optime_ml::pretrain::{self, PretrainConfig};
use optime_ml::theory::NO_CHORD;
use optime_ml::train::Back;
use optime_ml::train::MlDevice;
use optime_ml::{estimate, infer, probe, train};
use std::path::Path;

/// Load a model + its config from a `prefix` (weights `prefix.mpk`, config
/// `prefix.json`).
fn load_model_with_config(prefix: &Path, device: &MlDevice) -> (KeyChordModel<Back>, ModelConfig) {
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
    let real_path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("data/real_val.bin");
    let model_dir = args.get(2).map(String::as_str).unwrap_or("models");
    let device = MlDevice::default();

    let real = load_songs(real_path).unwrap_or_else(|_| {
        eprintln!("could not load {real_path} — run `harvest` first");
        std::process::exit(1);
    });
    println!("loaded {} real val windows from {real_path}\n", real.len());

    // --- 1. SSL distribution fit: pretrained recon loss, real vs. synthetic ---
    let pretrained_prefix = Path::new(model_dir).join("pretrained");
    if pretrained_prefix.with_extension("json").exists() {
        let model = train::load_pretrained(&pretrained_prefix, &device);
        let cfg = PretrainConfig::default();

        let seq = real.first().map(|s| s.n_frames).unwrap_or(0);
        let real_loss = pretrain::evaluate(&model, &real, &cfg, seq, &device);
        println!("SSL recon loss (masked-frame MSE):");
        println!("  real songs      : {real_loss:.5}");

        if let Ok(synth) = load_songs("data/val.bin") {
            let sseq = synth.first().map(|s| s.n_frames).unwrap_or(seq);
            let synth_loss = pretrain::evaluate(&model, &synth, &cfg, sseq, &device);
            println!("  synthetic songs : {synth_loss:.5}");
            println!("  gap (real - synthetic): {:+.5}\n", real_loss - synth_loss);
        } else {
            println!();
        }
    } else {
        println!(
            "(no {}.json — skipping recon-loss report)\n",
            pretrained_prefix.display()
        );
    }

    // --- 2. Chord agreement vs. the training-free reference ---
    let model_json = Path::new(model_dir).join("model.json");
    if model_json.exists() {
        let model = train::load_model(Path::new(model_dir), &device);
        let mut total = 0usize;
        let mut agree = 0usize;
        let mut chord_total = 0usize; // reference frames that carry a real chord
        let mut chord_agree = 0usize;

        for song in &real {
            let reference = estimate::estimate_from_notes(&song.notes, song.n_frames);
            let pred = infer::predict(&model, &song.notes, song.n_frames, &device).chord_labels;
            for (r, p) in reference.iter().zip(pred.iter()) {
                total += 1;
                if r == p {
                    agree += 1;
                }
                if *r != NO_CHORD {
                    chord_total += 1;
                    if r == p {
                        chord_agree += 1;
                    }
                }
            }
        }
        let pct = |a: usize, b: usize| {
            if b == 0 {
                0.0
            } else {
                100.0 * a as f64 / b as f64
            }
        };
        println!(
            "Chord agreement vs. template/Viterbi reference ({}):",
            model_dir
        );
        println!(
            "  all frames       : {:.1}%  ({agree}/{total})",
            pct(agree, total)
        );
        println!(
            "  chord frames only: {:.1}%  ({chord_agree}/{chord_total})",
            pct(chord_agree, chord_total)
        );
    } else {
        println!(
            "(no {} — skipping chord-agreement report)",
            model_json.display()
        );
    }

    // --- 3. Is-music probe accuracy on the weakly-labeled real windows ---
    let probe_prefix = Path::new(model_dir).join("probe");
    if probe_prefix.with_extension("json").exists() {
        let labeled = real.iter().filter(|s| s.is_music.is_some()).count();
        if labeled == 0 {
            println!("\n(real val has no is-music labels — re-harvest with --annotate)");
        } else {
            let (model, _) = load_model_with_config(&probe_prefix, &device);
            let set = probe::build_music_set(&model, &real, 64, &device);
            let acc = probe::accuracy(&model, &set, &device);
            let n_pos = set.labels.iter().filter(|&&l| l == 1).count();
            println!(
                "\nIs-music probe accuracy: {:.1}%  over {} labeled windows ({} music / {} not)",
                acc * 100.0,
                set.len(),
                n_pos,
                set.len() - n_pos
            );
        }
    }
}

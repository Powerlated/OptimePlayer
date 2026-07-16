//! Measure the real-music gap: how well a model fits real game songs.
//!
//! ```sh
//! cargo run --release --bin eval_real -- [real_val.bin] \
//!     [--backbone frame|event|hier] [--out-dir models]
//! ```
//!
//! Real game songs are unlabeled, so this reports comparable proxies on a held-out
//! set of harvested real windows:
//!
//! 1. **SSL distribution fit** — the pretrained trunk's pretext loss on real windows
//!    vs. on the synthetic validation set (`data/val.bin`). The direct domain-gap
//!    number. Generation 00 only: it is the masked-frame MSE, and the AR pretext's
//!    loss is not on the same scale, so mixing them would compare nothing.
//! 2. **Chord agreement %** — the trained model vs. the training-free chroma-template
//!    + Viterbi reference ([`optime_ml::estimate`]).
//! 3. **Is-music probe accuracy** on the weakly-labeled real windows (generation 00).
//!
//! Missing checkpoints are skipped with a note, so this is safe to run at any stage
//! of the pipeline.
//!
//! **This is not accuracy.** Real songs have no ground-truth chords; #2 is a
//! disagreement-between-estimators number against a heuristic reference.
//!
//! Replaces the old `eval_real` / `event_eval_real` / `hier_eval_real` trio.

use burn::module::AutodiffModule;
use optime_ml::backbone::{self, Backbone};
use optime_ml::backend::{Back, Inner, MlDevice};
use optime_ml::cli::{Args, Kind};
use optime_ml::data::load_songs;
use optime_ml::m00_frame::FrameModel;
use optime_ml::m01_event::EventModel;
use optime_ml::m02_hier::HierModel;
use optime_ml::notes::Song;
use optime_ml::pretrain::masked::{self, PretrainConfig};
use optime_ml::theory::NO_CHORD;
use optime_ml::{estimate, infer, probe};
use std::path::Path;

fn main() {
    let args = Args::parse();
    let real_path = args
        .positional
        .first()
        .map(String::as_str)
        .unwrap_or("data/real_val.bin");
    let device = MlDevice::default();
    let dir = args.out_dir.join(args.kind.dir());

    let real = load_songs(real_path).unwrap_or_else(|_| {
        eprintln!("could not load {real_path} — run `harvest` first");
        std::process::exit(1);
    });
    println!(
        "loaded {} real val windows from {real_path}; backbone {} ({})\n",
        real.len(),
        args.kind.name(),
        dir.display()
    );

    // --- 1. SSL distribution fit (generation 00's masked-frame pretext only) ---
    let pretrained_prefix = dir.join("pretrained");
    if args.kind == Kind::Frame && pretrained_prefix.with_extension("json").exists() {
        let model = backbone::load::<FrameModel<Back>, Back>(&pretrained_prefix, &device);
        let cfg = PretrainConfig::default();
        let seq = real.first().map(|s| s.n_frames).unwrap_or(0);
        let real_loss = masked::evaluate(&model, &real, &cfg, seq, &device);
        println!("SSL recon loss (masked-frame MSE):");
        println!("  real songs      : {real_loss:.5}");
        if let Ok(synth) = load_songs("data/val.bin") {
            let sseq = synth.first().map(|s| s.n_frames).unwrap_or(seq);
            let synth_loss = masked::evaluate(&model, &synth, &cfg, sseq, &device);
            println!("  synthetic songs : {synth_loss:.5}");
            println!("  gap (real - synthetic): {:+.5}\n", real_loss - synth_loss);
        } else {
            println!();
        }
    } else if args.kind == Kind::Frame {
        println!(
            "(no {}.json — skipping recon-loss report)\n",
            pretrained_prefix.display()
        );
    }

    // --- 2. Chord agreement vs. the training-free reference ---
    let model_prefix = dir.join("model");
    if model_prefix.with_extension("json").exists() {
        match args.kind {
            Kind::Frame => agreement::<FrameModel<Back>>(&model_prefix, &real, &device),
            Kind::Event => agreement::<EventModel<Back>>(&model_prefix, &real, &device),
            Kind::Hier => agreement::<HierModel<Back>>(&model_prefix, &real, &device),
        }
    } else {
        println!(
            "(no {}.json — skipping chord-agreement report; run `train --backbone {}` first)",
            model_prefix.display(),
            args.kind.name()
        );
    }

    // --- 3. Is-music probe accuracy on the weakly-labeled real windows ---
    let probe_prefix = args.out_dir.join("probe");
    if probe_prefix.with_extension("json").exists() {
        let labeled = real.iter().filter(|s| s.is_music.is_some()).count();
        if labeled == 0 {
            println!("\n(real val has no is-music labels — re-harvest with --annotate)");
        } else {
            let model = backbone::load::<FrameModel<Back>, Back>(&probe_prefix, &device);
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

/// Per-frame chord agreement between a trained backbone and the heuristic reference.
fn agreement<M>(prefix: &Path, real: &[Song], device: &MlDevice)
where
    M: Backbone<Back> + AutodiffModule<Back>,
    M::InnerModule: Backbone<Inner, Batch = <M as Backbone<Back>>::Batch>,
{
    let model = backbone::load::<M, Back>(prefix, device);
    let (mut total, mut agree, mut chord_total, mut chord_agree) = (0usize, 0, 0, 0);
    for song in real {
        let reference = estimate::estimate_from_notes(&song.notes, song.n_frames);
        let pred = infer::predict::<M>(&model, &song.notes, song.n_frames, device).chord_labels;
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
        M::NAME
    );
    println!(
        "  all frames       : {:.1}%  ({agree}/{total})",
        pct(agree, total)
    );
    println!(
        "  chord frames only: {:.1}%  ({chord_agree}/{chord_total})",
        pct(chord_agree, chord_total)
    );
}

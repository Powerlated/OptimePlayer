//! Measure the real-music gap on held-out harvested windows: (1) SSL distribution fit (m00
//! masked-frame MSE, real vs synthetic), (2) chord **agreement** vs the chroma-template + Viterbi
//! reference, (3) is-music probe accuracy (m00). Missing checkpoints are skipped.
//!
//! **NOT accuracy** — real songs have no ground-truth chords; per the ml eval rule the agreement
//! number must never be reported as one.

use super::opts::{Backbone, ModelOpts};
use burn::module::AutodiffModule;
use clap::Args;
use optime_ml::backbone;
use optime_ml::backend::{Back, Inner, MlDevice};
use optime_ml::data::load_songs;
use optime_ml::m00_frame::FrameModel;
use optime_ml::m01_event::EventModel;
use optime_ml::m02_hier::HierModel;
use optime_ml::notes::Song;
use optime_ml::pretrain::masked::{self, PretrainConfig};
use optime_ml::theory::NO_CHORD;
use optime_ml::{estimate, infer, probe};
use std::path::Path;

#[derive(Args, Debug)]
pub struct EvalRealArgs {
    /// Harvested val windows to score (default `data/real_val.bin`).
    pub real_val: Option<String>,
    #[command(flatten)]
    pub opts: ModelOpts,
}

pub fn run(args: EvalRealArgs) {
    let real_path = args.real_val.as_deref().unwrap_or("data/real_val.bin");
    let device = MlDevice::default();
    let backbone = args.opts.backbone;
    let dir = args.opts.out_dir.join(backbone.dir());

    let real = load_songs(real_path).unwrap_or_else(|_| {
        eprintln!("could not load {real_path} — run `harvest` first");
        std::process::exit(1);
    });
    println!(
        "loaded {} real val windows from {real_path}; backbone {} ({})\n",
        real.len(),
        backbone.name(),
        dir.display()
    );

    // --- 1. SSL distribution fit (generation 00's masked-frame pretext only) ---
    let pretrained_prefix = dir.join("pretrained");
    if backbone == Backbone::Frame && pretrained_prefix.with_extension("json").exists() {
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
    } else if backbone == Backbone::Frame {
        println!(
            "(no {}.json — skipping recon-loss report)\n",
            pretrained_prefix.display()
        );
    }

    // --- 2. Chord agreement vs. the training-free reference ---
    let model_prefix = dir.join("model");
    if model_prefix.with_extension("json").exists() {
        match backbone {
            Backbone::Frame => agreement::<FrameModel<Back>>(&model_prefix, &real, &device),
            Backbone::Event => agreement::<EventModel<Back>>(&model_prefix, &real, &device),
            Backbone::Hier => agreement::<HierModel<Back>>(&model_prefix, &real, &device),
        }
    } else {
        println!(
            "(no {}.json — skipping chord-agreement report; run `train --backbone {}` first)",
            model_prefix.display(),
            backbone.name()
        );
    }

    // --- 3. Is-music probe accuracy on the weakly-labeled real windows ---
    let probe_prefix = args.opts.out_dir.join("probe");
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
    M: optime_ml::backbone::Backbone<Back> + AutodiffModule<Back>,
    M::InnerModule: optime_ml::backbone::Backbone<
        Inner,
        Batch = <M as optime_ml::backbone::Backbone<Back>>::Batch,
    >,
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

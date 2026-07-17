//! Score a model against **hand-labelled** real songs — the only trustworthy real-music metric.
//! Defaults to the held-out songs; `--model model_sft` scores the real-label stage, `--val-songs`
//! a contrived hand-picked split, `--all` includes anything `sft` trained on (inspection only).
//!
//! **Root accuracy is primary.** Quality is where agreement collapses (for model and humans alike);
//! spans marked `qualityUncertain` are scored for root only and excluded from the quality figures.
//! This is not `eval-real`: that reports agreement with a heuristic, this compares against a human.

use super::opts::{Backbone, ModelOpts};
use burn::module::AutodiffModule;
use clap::Args;
use optime_core::load_all;
use optime_ml::annotations::{self, Split, DEFAULT_VAL_FRAC};
use optime_ml::backbone;
use optime_ml::backend::{Back, Inner, MlDevice};
use optime_ml::harvest::harvest_song_full;
use optime_ml::infer;
use optime_ml::notes::NoteEvent;
use optime_ml::theory::{chord_label_to_root_quality, NO_CHORD};
use std::path::{Path, PathBuf};

#[derive(Args, Debug)]
pub struct EvalLabeledArgs {
    /// Annotation directory (default `annotations`).
    pub ann_dir: Option<PathBuf>,
    /// ROM/archive directory (default `../demos`).
    pub rom_dir: Option<PathBuf>,
    /// Score every annotated song, including anything `sft` trained on (inspection only).
    #[arg(long)]
    pub all: bool,
    #[command(flatten)]
    pub opts: ModelOpts,
}

/// One window handed to the model: its notes and the truth for each frame.
struct Cut {
    notes: Vec<NoteEvent>,
    truth: Vec<Option<annotations::FrameLabel>>,
}

#[derive(Default)]
struct Score {
    frames: usize,
    root_ok: usize,
    joint_ok: usize,
    /// Quality figures cover only frames whose quality the annotator vouched for.
    quality_frames: usize,
    quality_ok: usize,
    /// Frames both sides call a chord (excludes agreed silence, which is easy).
    chord_frames: usize,
    chord_root_ok: usize,
}

impl Score {
    fn add(&mut self, truth: annotations::FrameLabel, pred: usize) {
        let (tr, tq) = chord_label_to_root_quality(truth.chord);
        let (pr, pq) = chord_label_to_root_quality(pred);
        self.frames += 1;
        if tr == pr {
            self.root_ok += 1;
        }
        if truth.chord != NO_CHORD {
            self.chord_frames += 1;
            if tr == pr {
                self.chord_root_ok += 1;
            }
        }
        if truth.quality_certain {
            self.quality_frames += 1;
            if tq == pq {
                self.quality_ok += 1;
            }
            if truth.chord == pred {
                self.joint_ok += 1;
            }
        }
    }
}

fn pct(a: usize, b: usize) -> String {
    if b == 0 {
        "  n/a".into()
    } else {
        format!("{:5.1}%", 100.0 * a as f64 / b as f64)
    }
}

pub fn run(args: EvalLabeledArgs) {
    let ann_dir = args.ann_dir.clone().unwrap_or_else(|| "annotations".into());
    let rom_dir = args.rom_dir.clone().unwrap_or_else(|| "../demos".into());

    // Default to the held-out songs only. `sft` trains on the complement of exactly this split, so
    // scoring everything would report a number partly measured on the training set.
    let score_all = args.all;
    let hand_picked = args.opts.val_songs.is_some();
    let split = match (&args.opts.val_songs, score_all) {
        // An explicit list wins: the point of `--val-songs` is to score exactly these songs.
        (Some(ids), _) => Split::Songs(ids.clone()),
        (None, true) => Split::All,
        (None, false) => Split::Val(DEFAULT_VAL_FRAC),
    };

    let files = match annotations::load_dir(&ann_dir) {
        Ok(f) if !f.is_empty() => f,
        Ok(_) => {
            eprintln!(
                "no annotation files in {} — label some songs in the app first \
                 (Settings → Data annotation mode)",
                ann_dir.display()
            );
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    // Gather every fully-annotated window. Partially-annotated ones are skipped: a frame nobody has
    // listened to is not evidence either way, and counting it would fake the denominator.
    let seq_len = 256;
    let mut cuts: Vec<Cut> = Vec::new();
    let mut labeled_frames = 0usize;
    for file in &files {
        let path = rom_dir.join(&file.source);
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("⚠ {}: not found (annotation `source`)", path.display());
            continue;
        };
        let archives = load_all(&bytes);
        let Some(data) = archives.first() else {
            eprintln!("⚠ {}: no sound archive", path.display());
            continue;
        };
        for ann in &file.songs {
            if !split.accepts(&file.source, ann.song_id) {
                continue;
            }
            let Some((notes, _)) = harvest_song_full(&**data, ann.song_id) else {
                continue;
            };
            let total = notes.iter().map(|n| n.end_frame).max().unwrap_or(0) as usize;
            let truth = ann.frame_labels(file.steps_per_beat, total);
            labeled_frames += truth.iter().filter(|t| t.is_some()).count();
            // Windows come from the labelled runs, not an absolute grid: a song opens with a pickup,
            // so its labels start a frame or two in and a frame-0 grid would drop every one of them.
            for win in annotations::complete_windows(&truth, seq_len) {
                let (a, b) = (win.start, win.end);
                let win_notes =
                    optime_ml::harvest::clip_notes_to_window(&notes, a as u32, b as u32);
                if win_notes.is_empty() {
                    continue;
                }
                cuts.push(Cut {
                    notes: win_notes,
                    truth: truth[a..b].to_vec(),
                });
            }
        }
    }

    let leftover = labeled_frames.saturating_sub(cuts.len() * seq_len);
    println!(
        "{} annotation file(s), {labeled_frames} labelled frames → {} complete {seq_len}-frame \
         window(s) ({leftover} labelled frame(s) left over) [{}]",
        files.len(),
        cuts.len(),
        if hand_picked {
            "CONTRIVED hand-picked songs"
        } else if score_all {
            "ALL songs — includes anything sft trained on"
        } else {
            "held-out songs only"
        }
    );
    if hand_picked {
        println!(
            "⚠ --val-songs {:?} is a hand-picked split, not the deterministic holdout. Nothing \
             checks that `sft` didn't train on these; not a number to report.",
            args.opts.val_songs.clone().unwrap_or_default()
        );
    } else if score_all {
        println!(
            "⚠ --all scores songs `sft` may have trained on. Fine for inspection; not a number to \
             report."
        );
    }
    if cuts.is_empty() {
        println!(
            "\nNothing to score yet: a window needs all {seq_len} frames (= {} bars of 4/4) \
             annotated.\nKeep labelling — contiguous bars are what turn into eval data.",
            seq_len / 16
        );
        return;
    }

    let device = MlDevice::default();
    // `--model model_sft` scores the real-label stage; the default `model` is the synthetic
    // fine-tune that SFT warm-starts *from*.
    let prefix = args.opts.model_prefix();
    if !prefix.with_extension("json").exists() {
        eprintln!("{}.json not found — train first", prefix.display());
        std::process::exit(1);
    }
    println!("scoring {}", prefix.display());
    let score = match args.opts.backbone {
        Backbone::Frame => {
            run_model::<optime_ml::m00_frame::FrameModel<Back>>(&prefix, &cuts, seq_len, &device)
        }
        Backbone::Event => {
            run_model::<optime_ml::m01_event::EventModel<Back>>(&prefix, &cuts, seq_len, &device)
        }
        Backbone::Hier => {
            run_model::<optime_ml::m02_hier::HierModel<Back>>(&prefix, &cuts, seq_len, &device)
        }
    };

    println!("\nvs. hand labels ({}):", args.opts.backbone.name());
    println!(
        "  root            : {}  ({}/{})   ← primary",
        pct(score.root_ok, score.frames),
        score.root_ok,
        score.frames
    );
    println!(
        "  root, chord frames only : {}  ({}/{})",
        pct(score.chord_root_ok, score.chord_frames),
        score.chord_root_ok,
        score.chord_frames
    );
    println!(
        "  quality (certain only)  : {}  ({}/{})",
        pct(score.quality_ok, score.quality_frames),
        score.quality_ok,
        score.quality_frames
    );
    println!(
        "  joint root+quality      : {}  ({}/{})",
        pct(score.joint_ok, score.quality_frames),
        score.joint_ok,
        score.quality_frames
    );
    let uncertain = score.frames.saturating_sub(score.quality_frames);
    if uncertain > 0 {
        println!(
            "\n{uncertain} frame(s) had an uncertain quality: scored for root, excluded from the \
             quality figures."
        );
    }
    println!(
        "\nThis is real accuracy, not agreement with an estimator — but on {} window(s) it is a \
         small sample. Treat it as a signal, not a verdict, until the set grows.",
        cuts.len()
    );
}

fn run_model<M>(prefix: &Path, cuts: &[Cut], seq_len: usize, device: &MlDevice) -> Score
where
    M: optime_ml::backbone::Backbone<Back> + AutodiffModule<Back>,
    M::InnerModule: optime_ml::backbone::Backbone<
        Inner,
        Batch = <M as optime_ml::backbone::Backbone<Back>>::Batch,
    >,
{
    let model = backbone::load::<M, Back>(prefix, device);
    let mut score = Score::default();
    for cut in cuts {
        let pred = infer::predict::<M>(&model, &cut.notes, seq_len, device).chord_labels;
        for (t, p) in cut.truth.iter().zip(pred.iter()) {
            if let Some(t) = t {
                score.add(*t, *p);
            }
        }
    }
    score
}

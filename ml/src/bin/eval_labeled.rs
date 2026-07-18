//! Score a model against **hand-labelled** real songs — the only trustworthy real-music metric.
//!
//! ```sh
//! cargo run --release --features harvest --bin eval_labeled -- \
//!     [ann_dir] [rom_dir] [--backbone frame|event|hier] [--out-dir models]
//! #   defaults: ann_dir = ml/annotations, rom_dir = ../demos
//! ```
//!
//! This is **not** `eval_real`. That reports agreement with `estimate.rs`, a chroma-template
//! heuristic — not ground truth, and structurally biased toward the chroma-input backbone, so it
//! can neither be called accuracy nor used to rank generations. This bin compares against a human
//! who listened to the bar, so its numbers mean what they say.
//!
//! **Root accuracy is the primary number.** Quality is where agreement collapses — for the model
//! (m02 scored 27% root-only vs 4% joint on the heuristic) and for human annotators alike, since
//! sparse game voicings are genuinely ambiguous above the triad. Spans the annotator marked
//! `qualityUncertain` are therefore scored for root only and excluded from the quality figures
//! rather than being silently counted as wrong.

use burn::module::AutodiffModule;
use optime_core::load_all;
use optime_ml::annotations::{self, Coverage, Split, DEFAULT_VAL_FRAC};
use optime_ml::backbone::{self, Backbone};
use optime_ml::backend::{Back, Inner, MlDevice};
use optime_ml::cli::{Args, Kind};
use optime_ml::harvest::harvest_song_full;
use optime_ml::infer;
use optime_ml::m00_frame::FrameModel;
use optime_ml::m01_event::EventModel;
use optime_ml::m02_hier::HierModel;
use optime_ml::notes::NoteEvent;
use optime_ml::theory::{chord_label_to_root_quality, NO_CHORD};
use std::path::{Path, PathBuf};

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

fn main() {
    let args = Args::parse();
    let ann_dir = PathBuf::from(
        args.positional
            .first()
            .cloned()
            .unwrap_or_else(|| "annotations".into()),
    );
    let rom_dir = PathBuf::from(
        args.positional
            .get(1)
            .cloned()
            .unwrap_or_else(|| "../demos".into()),
    );

    // Default to the held-out songs only. `sft` trains on the complement of exactly this split, so
    // scoring everything would report a number partly measured on the training set.
    let score_all = std::env::args().any(|a| a == "--all");
    let split = if score_all {
        Split::All
    } else {
        Split::Val(DEFAULT_VAL_FRAC)
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
    let mut skipped_partial = 0usize;
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
            for w in 0..(total / seq_len) {
                let (a, b) = (w * seq_len, (w + 1) * seq_len);
                let win = truth[a..b].to_vec();
                if !Coverage::of(&win).is_complete() {
                    skipped_partial += 1;
                    continue;
                }
                let win_notes =
                    optime_ml::harvest::clip_notes_to_window(&notes, a as u32, b as u32);
                if win_notes.is_empty() {
                    continue;
                }
                cuts.push(Cut {
                    notes: win_notes,
                    truth: win,
                });
            }
        }
    }

    println!(
        "{} annotation file(s), {labeled_frames} labelled frames → {} complete {seq_len}-frame \
         window(s) ({skipped_partial} partial skipped) [{}]",
        files.len(),
        cuts.len(),
        if score_all {
            "ALL songs — includes anything sft trained on"
        } else {
            "held-out songs only"
        }
    );
    if score_all {
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
    let prefix = args.out_dir.join(args.kind.dir()).join("model");
    if !prefix.with_extension("json").exists() {
        eprintln!("{}.json not found — train first", prefix.display());
        std::process::exit(1);
    }
    let score = match args.kind {
        Kind::Frame => run::<FrameModel<Back>>(&prefix, &cuts, seq_len, &device),
        Kind::Event => run::<EventModel<Back>>(&prefix, &cuts, seq_len, &device),
        Kind::Hier => run::<HierModel<Back>>(&prefix, &cuts, seq_len, &device),
    };

    println!("\nvs. hand labels ({}):", args.kind.name());
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

fn run<M>(prefix: &Path, cuts: &[Cut], seq_len: usize, device: &MlDevice) -> Score
where
    M: Backbone<Back> + AutodiffModule<Back>,
    M::InnerModule: Backbone<Inner, Batch = <M as Backbone<Back>>::Batch>,
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

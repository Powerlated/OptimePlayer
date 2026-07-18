//! Hand-authored chord labels — the app's annotation dev-tool output (`ml/annotations/*.json`).
//!
//! **This is the only ground truth that exists for real music.** The heuristic in [`crate::estimate`]
//! is a chroma-template matcher, not labels, and scoring against it is structurally biased toward
//! the chroma-input backbone; synthetic validation measures a generator the learned-token backbones
//! can nearly invert. A human listening to a bar is the only source that measures the thing we care
//! about, so these files are the eval set — and, once there are enough of them, the final SFT stage.
//!
//! The JSON is the contract with the app, which deliberately owns none of the mapping into the label
//! space: `theory` stays the single source of that (per the ml conventions), so the app never has to
//! depend on this crate. This module is the *only* place the two meet, and the roundtrip test below
//! is what keeps them honest — if the app's vocabulary and [`theory::Quality`] ever drift apart, it
//! fails here rather than silently mislabelling a training set.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::notes::FRAMES_PER_BEAT;
use crate::theory::{self, root_quality_to_chord_label, NO_CHORD};

/// Chord quality, mirroring the app's `annotation::model::Quality` **in order**.
///
/// The app pins this list 1:1 with [`theory::Quality`] on its side; [`Quality::to_theory`] plus
/// `quality_matches_theory_one_to_one` pins it on ours. Two enums rather than one because the app
/// must not depend on this crate (it would pull in burn), so the JSON — not a shared type — is the
/// seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Quality {
    Major,
    Minor,
    Diminished,
    Augmented,
    Dominant7,
    Major7,
    Minor7,
    HalfDiminished7,
    Sus2,
    Sus4,
}

impl Quality {
    pub const ALL: [Quality; 10] = [
        Quality::Major,
        Quality::Minor,
        Quality::Diminished,
        Quality::Augmented,
        Quality::Dominant7,
        Quality::Major7,
        Quality::Minor7,
        Quality::HalfDiminished7,
        Quality::Sus2,
        Quality::Sus4,
    ];

    pub fn to_theory(self) -> theory::Quality {
        match self {
            Quality::Major => theory::Quality::Major,
            Quality::Minor => theory::Quality::Minor,
            Quality::Diminished => theory::Quality::Diminished,
            Quality::Augmented => theory::Quality::Augmented,
            Quality::Dominant7 => theory::Quality::Dominant7,
            Quality::Major7 => theory::Quality::Major7,
            Quality::Minor7 => theory::Quality::Minor7,
            Quality::HalfDiminished7 => theory::Quality::HalfDiminished7,
            Quality::Sus2 => theory::Quality::Sus2,
            Quality::Sus4 => theory::Quality::Sus4,
        }
    }

    /// The 1-based quality class the factored head predicts (0 is reserved for "none").
    pub fn class(self) -> usize {
        let t = self.to_theory();
        theory::Quality::ALL
            .iter()
            .position(|q| *q == t)
            .expect("quality is in theory::Quality::ALL")
            + 1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chord {
    /// Root pitch class, C = 0.
    pub root: u8,
    pub quality: Quality,
    /// The annotator was confident of the root but not the colour above it. Such spans are scored
    /// for **root only** and are kept out of SFT entirely — training on a stated-but-doubted quality
    /// would teach a guess as if it were fact.
    #[serde(rename = "qualityUncertain", default)]
    pub quality_uncertain: bool,
}

/// One annotated span. `chord: None` is an explicit *no chord*, which is a judgment; a frame no span
/// covers is *unannotated*, which is the absence of one. Conflating the two is the single easiest
/// way to poison this dataset, so they are different shapes here and stay different all the way into
/// [`SongAnnotation::frame_labels`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    #[serde(rename = "startStep")]
    pub start_step: u32,
    #[serde(rename = "endStep")]
    pub end_step: u32,
    pub chord: Option<Chord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Major,
    Minor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Key {
    pub tonic: u8,
    pub mode: Mode,
}

impl Key {
    /// The 24-way key label (`theory::Key::label`).
    pub fn label(&self) -> usize {
        let mode = match self.mode {
            Mode::Major => theory::Mode::Major,
            Mode::Minor => theory::Mode::Minor,
        };
        theory::Key::new(self.tonic, mode).label()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SongAnnotation {
    #[serde(rename = "songId")]
    pub song_id: u32,
    #[serde(rename = "beatsPerBar")]
    pub beats_per_bar: u32,
    #[serde(rename = "gridOffsetSteps")]
    pub grid_offset_steps: i64,
    #[serde(default)]
    pub key: Option<Key>,
    pub spans: Vec<Span>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameAnnotations {
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,
    pub source: String,
    #[serde(rename = "gameCode", default)]
    pub game_code: Option<String>,
    /// The device's sequencer steps per beat, so spans convert to the model's frame grid without
    /// re-running the engine.
    #[serde(rename = "stepsPerBeat")]
    pub steps_per_beat: f64,
    pub songs: Vec<SongAnnotation>,
}

/// Schema version this build understands.
pub const SCHEMA_VERSION: u32 = 1;

/// One frame's ground truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameLabel {
    /// Joint chord label in `theory`'s space ([`NO_CHORD`] for an annotated no-chord).
    pub chord: usize,
    /// Whether the *quality* half is trustworthy. `false` → score/train the root only.
    pub quality_certain: bool,
}

/// Reads a game's annotations. A newer `schemaVersion` is an error rather than a best-effort parse:
/// silently misreading hand labels would corrupt the only ground truth there is.
pub fn load(path: &Path) -> Result<GameAnnotations, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let file: GameAnnotations =
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
    if file.schema_version > SCHEMA_VERSION {
        return Err(format!(
            "{}: schemaVersion {} is newer than this build understands ({SCHEMA_VERSION})",
            path.display(),
            file.schema_version
        ));
    }
    if file.steps_per_beat <= 0.0 {
        return Err(format!("{}: stepsPerBeat must be > 0", path.display()));
    }
    Ok(file)
}

/// Every annotation file in a directory, sorted by name for a deterministic dataset.
pub fn load_dir(dir: &Path) -> Result<Vec<GameAnnotations>, String> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("read {}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    paths.iter().map(|p| load(p)).collect()
}

impl SongAnnotation {
    /// Per-frame ground truth over `n_frames`, `None` where nothing has been annotated.
    ///
    /// `steps_per_beat` comes from the file (the device's own constant), and the model's grid is
    /// [`FRAMES_PER_BEAT`] per beat, so `frame = step × FPB / spb`. Steps are musical time — a tempo
    /// change moves the step *rate*, not this ratio — which is why no tempo map is needed.
    pub fn frame_labels(&self, steps_per_beat: f64, n_frames: usize) -> Vec<Option<FrameLabel>> {
        let mut out = vec![None; n_frames];
        let to_frame =
            |step: u32| ((step as f64 * FRAMES_PER_BEAT as f64) / steps_per_beat).round() as usize;
        for span in &self.spans {
            let (a, b) = (to_frame(span.start_step), to_frame(span.end_step));
            let label = match span.chord {
                Some(c) => FrameLabel {
                    chord: root_quality_to_chord_label(c.root as usize % 12 + 1, c.quality.class()),
                    quality_certain: !c.quality_uncertain,
                },
                // An annotated rest. Its "quality" is genuinely none, not a doubtful guess.
                None => FrameLabel {
                    chord: NO_CHORD,
                    quality_certain: true,
                },
            };
            for slot in out.iter_mut().take(b.min(n_frames)).skip(a.min(n_frames)) {
                *slot = Some(label);
            }
        }
        out
    }

    pub fn key_label(&self) -> Option<usize> {
        self.key.map(|k| k.label())
    }
}

/// Default share of annotated songs held out for evaluation.
pub const DEFAULT_VAL_FRAC: f64 = 0.25;

/// Which half of the hand-labelled set a consumer wants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Split {
    /// Songs to train on.
    Train(f64),
    /// Songs held out to score against.
    Val(f64),
    /// Everything, ignoring the split. Only legitimate for inspection — **never** report a number
    /// from this after training on the same files.
    All,
}

impl Split {
    pub fn accepts(&self, source: &str, song_id: u32) -> bool {
        match self {
            Split::Train(f) => !is_val(source, song_id, *f),
            Split::Val(f) => is_val(source, song_id, *f),
            Split::All => true,
        }
    }
}

/// Whether a song belongs to the **held-out** half, decided from its identity alone.
///
/// `sft` and `eval_labeled` both call this, which is the entire point: hand labels are scarce, and
/// the temptation to train on all of them and then score on all of them is enormous — and would
/// make the only trustworthy number we have meaningless. A song-level split (not window-level) also
/// keeps two windows of the same track from straddling it.
///
/// FNV-1a rather than `DefaultHasher`: the standard hasher's output is explicitly not stable across
/// Rust releases, and a split that silently reshuffles on a toolchain upgrade would leak the eval
/// set into training without anyone noticing.
pub fn is_val(source: &str, song_id: u32, val_frac: f64) -> bool {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in source.as_bytes().iter().chain(song_id.to_le_bytes().iter()) {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    ((h % 10_000) as f64 / 10_000.0) < val_frac
}

/// How much of a frame range is annotated, and whether every quality in it is trusted.
pub struct Coverage {
    pub labeled: usize,
    pub total: usize,
    pub all_quality_certain: bool,
}

impl Coverage {
    pub fn of(labels: &[Option<FrameLabel>]) -> Coverage {
        Coverage {
            labeled: labels.iter().filter(|l| l.is_some()).count(),
            total: labels.len(),
            all_quality_certain: labels.iter().flatten().all(|l| l.quality_certain),
        }
    }

    pub fn is_complete(&self) -> bool {
        self.total > 0 && self.labeled == self.total
    }
}

/// Build labeled [`Song`](crate::notes::Song) windows from annotation files plus the real note
/// streams they describe. Needs the engine, so it lives behind `harvest` like the rest of the
/// real-song path.
#[cfg(feature = "harvest")]
pub mod build {
    use super::*;
    use crate::harvest::harvest_song_full;
    use crate::notes::Song;
    use optime_core::{load_all, SoundData};

    /// Windows dropped while building, and why — the numbers that tell you whether annotating is
    /// actually producing training data or quietly going nowhere.
    #[derive(Debug, Default, Clone, Copy)]
    pub struct BuildStats {
        pub windows_kept: usize,
        pub dropped_incomplete: usize,
        pub dropped_uncertain: usize,
        pub dropped_no_notes: usize,
        pub songs_missing_key: usize,
    }

    /// Turns one game's annotations into fixed-length labeled windows.
    ///
    /// **Only fully-annotated windows are emitted.** A window with any unlabeled frame is dropped
    /// rather than back-filled with `NO_CHORD`: an unheard bar is not a rest, and teaching the model
    /// that it is would be worse than having no data at all. (Partial windows could be salvaged by
    /// masking the loss per frame — deliberately not done yet; it touches the one shared training
    /// loop, and correctness here matters more than squeezing out the last few windows.)
    ///
    /// **Windows containing an uncertain quality are dropped too.** Those labels are real, but only
    /// their root is — they stay in the eval set (scored root-only) rather than teaching the
    /// quality head a guess.
    pub fn songs_from_annotations(
        file: &GameAnnotations,
        data: &dyn SoundData,
        seq_len: usize,
        want: Split,
    ) -> (Vec<Song>, BuildStats) {
        let mut out = Vec::new();
        let mut stats = BuildStats::default();

        for ann in &file.songs {
            if !want.accepts(&file.source, ann.song_id) {
                continue;
            }
            let Some((notes, _spb)) = harvest_song_full(data, ann.song_id) else {
                continue;
            };
            let Some(key_label) = ann.key_label() else {
                // The key head is supervised per window; without a key there is nothing to train it
                // on, and inventing one would be a fabricated label.
                stats.songs_missing_key += 1;
                continue;
            };
            let total_frames = notes.iter().map(|n| n.end_frame).max().unwrap_or(0) as usize;
            let labels = ann.frame_labels(file.steps_per_beat, total_frames);

            for w in 0..(total_frames / seq_len) {
                let (a, b) = (w * seq_len, (w + 1) * seq_len);
                let win = &labels[a..b];
                let cov = Coverage::of(win);
                if !cov.is_complete() {
                    stats.dropped_incomplete += 1;
                    continue;
                }
                if !cov.all_quality_certain {
                    stats.dropped_uncertain += 1;
                    continue;
                }
                let win_notes = crate::harvest::clip_notes_to_window(&notes, a as u32, b as u32);
                if win_notes.is_empty() {
                    stats.dropped_no_notes += 1;
                    continue;
                }
                out.push(Song {
                    key_label,
                    n_frames: seq_len,
                    notes: win_notes,
                    chord_labels: win.iter().map(|l| l.expect("complete").chord).collect(),
                    is_music: Some(true), // hand-annotated implies a real music track
                });
                stats.windows_kept += 1;
            }
        }
        (out, stats)
    }

    /// Every annotation file under `dir`, paired with its archive from `rom_dir` by `source`.
    ///
    /// Missing archives are reported rather than skipped silently: a typo'd `source` would
    /// otherwise look exactly like "no labels yet".
    pub fn songs_from_dir(
        ann_dir: &Path,
        rom_dir: &Path,
        seq_len: usize,
        want: Split,
    ) -> Result<(Vec<Song>, BuildStats), String> {
        let files = load_dir(ann_dir)?;
        let mut all = Vec::new();
        let mut total = BuildStats::default();
        for file in &files {
            let path = rom_dir.join(&file.source);
            let bytes = std::fs::read(&path).map_err(|e| {
                format!(
                    "{}: {e} (annotation `source` must name a file in {})",
                    path.display(),
                    rom_dir.display()
                )
            })?;
            let archives = load_all(&bytes);
            let Some(data) = archives.first() else {
                return Err(format!("{}: no sound archive found", path.display()));
            };
            let (songs, s) = songs_from_annotations(file, &**data, seq_len, want);
            all.extend(songs);
            total.windows_kept += s.windows_kept;
            total.dropped_incomplete += s.dropped_incomplete;
            total.dropped_uncertain += s.dropped_uncertain;
            total.dropped_no_notes += s.dropped_no_notes;
            total.songs_missing_key += s.songs_missing_key;
        }
        Ok((all, total))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::chord_label_to_root_quality;

    /// The app's vocabulary and `theory::Quality` must stay 1:1, in the same order. If they drift,
    /// every hand label silently becomes a different chord — so this fails loudly instead.
    #[test]
    fn quality_matches_theory_one_to_one() {
        assert_eq!(Quality::ALL.len(), theory::Quality::ALL.len());
        for (i, q) in Quality::ALL.iter().enumerate() {
            assert_eq!(q.to_theory(), theory::Quality::ALL[i], "index {i}");
            assert_eq!(q.class(), i + 1, "class is 1-based over ALL");
        }
    }

    #[test]
    fn chords_map_into_the_joint_label_space() {
        for root in 0..12u8 {
            for q in Quality::ALL {
                let label = root_quality_to_chord_label(root as usize + 1, q.class());
                let (r, qq) = chord_label_to_root_quality(label);
                assert_eq!(r, root as usize + 1);
                assert_eq!(qq, q.class());
                assert_ne!(label, NO_CHORD);
            }
        }
    }

    fn song(spans: Vec<Span>) -> SongAnnotation {
        SongAnnotation {
            song_id: 1,
            beats_per_bar: 4,
            grid_offset_steps: 0,
            key: Some(Key {
                tonic: 9,
                mode: Mode::Minor,
            }),
            spans,
        }
    }

    fn chord(root: u8, q: Quality, uncertain: bool) -> Option<Chord> {
        Some(Chord {
            root,
            quality: q,
            quality_uncertain: uncertain,
        })
    }

    #[test]
    fn spans_convert_steps_to_the_frame_grid() {
        // GBA: 24 steps/beat → 4 frames/beat means 6 steps per frame. A 96-step bar = 16 frames.
        let s = song(vec![Span {
            start_step: 0,
            end_step: 96,
            chord: chord(9, Quality::Minor, false),
        }]);
        let labels = s.frame_labels(24.0, 32);
        assert!(labels[..16].iter().all(|l| l.is_some()));
        // Nothing was said about the second bar — that must stay unlabeled, not become "no chord".
        assert!(labels[16..].iter().all(|l| l.is_none()));
        let l = labels[0].unwrap();
        assert_eq!(chord_label_to_root_quality(l.chord), (10, 2)); // A(9)+1, Minor→class 2
        assert!(l.quality_certain);
    }

    #[test]
    fn an_annotated_rest_is_not_an_unannotated_gap() {
        let s = song(vec![Span {
            start_step: 0,
            end_step: 48,
            chord: None,
        }]);
        let labels = s.frame_labels(24.0, 16);
        // Explicit N.C.: a real label, and its quality is genuinely none rather than doubted.
        for l in &labels[..8] {
            let l = l.expect("N.C. is a label");
            assert_eq!(l.chord, NO_CHORD);
            assert!(l.quality_certain);
        }
        // Beyond the span: nobody has listened yet.
        assert!(labels[8..].iter().all(|l| l.is_none()));
    }

    #[test]
    fn uncertain_quality_is_carried_through() {
        let s = song(vec![Span {
            start_step: 0,
            end_step: 24,
            chord: chord(0, Quality::Sus2, true),
        }]);
        let labels = s.frame_labels(24.0, 8);
        assert!(!labels[0].unwrap().quality_certain);
        let cov = Coverage::of(&labels);
        assert!(!cov.all_quality_certain);
        assert_eq!((cov.labeled, cov.total), (4, 8));
        assert!(!cov.is_complete());
    }

    #[test]
    fn coverage_reports_a_fully_labeled_window() {
        let s = song(vec![Span {
            start_step: 0,
            end_step: 48,
            chord: chord(5, Quality::Major, false),
        }]);
        let cov = Coverage::of(&s.frame_labels(24.0, 8));
        assert!(cov.is_complete() && cov.all_quality_certain);
    }

    #[test]
    fn key_maps_to_the_24_way_label() {
        let s = song(vec![]);
        assert_eq!(
            s.key_label(),
            Some(theory::Key::new(9, theory::Mode::Minor).label())
        );
    }

    #[test]
    fn a_newer_schema_is_refused_rather_than_guessed_at() {
        let dir = std::env::temp_dir().join("optime_ml_ann_test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("future.json");
        std::fs::write(
            &p,
            r#"{"schemaVersion":99,"source":"x","stepsPerBeat":24,"songs":[]}"#,
        )
        .unwrap();
        let err = load(&p).unwrap_err();
        assert!(err.contains("newer than this build"), "{err}");
        let _ = std::fs::remove_file(&p);
    }
}

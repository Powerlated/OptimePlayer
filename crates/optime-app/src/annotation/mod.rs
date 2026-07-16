//! Chord annotation: the maintainer dev-tool that turns the piano roll into a labelling platform.
//!
//! Hand-labelled measure-level chords are the only trustworthy signal on real music. The heuristic
//! reference (`ml/src/estimate.rs`) is not ground truth — it is a chroma-template matcher, and
//! scoring against it is structurally biased toward the chroma-input backbone — while synthetic
//! validation has saturated (m02: 99.7%) and no longer discriminates. Labels authored here are the
//! eval set that replaces both, and later a real-data fine-tune stage.
//!
//! Labels are authored **blank-slate**, never pre-filled from a model's predictions: this set judges
//! the model, so it must not be anchored to the model's own guesses.
//!
//! ## Shape
//!
//! * [`Bounce`] / [`BounceJob`] — the whole song rendered to memory. Annotation needs random access
//!   (hear this bar, loop those two, scrub back); the engine has no seek, so the song is rendered
//!   once and every seek becomes an index. Rendered incrementally, never on a thread — web has none.
//! * [`model`] — the data model and its JSON, the contract with `optime-ml`.
//! * [`AnnotationState`] — session state: what's loaded, what's selected, what's unsaved.
//!
//! **Runs on web as well as native.** Labelling needs nothing but the engine; only where the JSON
//! lands differs — the source tree (`ml/annotations/`, the training data of record) on native, a
//! browser download plus an eframe-storage working copy on web. The schema is identical either way,
//! which is the point of the format being the contract rather than the code path.

mod bounce;
pub mod model;

pub use bounce::{Bounce, BounceJob};

use std::collections::HashMap;
use std::sync::Arc;

use crate::piano_roll::Grid;
use model::{GameAnnotations, SongAnnotation, Span};

/// Where hand-authored labels live: `ml/annotations/` next to the code that trains on them.
///
/// They are **training data, not app metadata** — which is why they sit under `ml/` rather than
/// beside [`crate::song_names`]'s cosmetic tables — and they are **git-tracked**: `ml/.gitignore`
/// excludes `/data` and `/models` because those regenerate from a seed or a run. These don't. They
/// are hours of listening and cannot be reproduced by any command.
#[cfg(not(target_arch = "wasm32"))]
pub fn source_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ml/annotations")
}

/// The annotation filename for a source archive: its stem, `.json`. Mirrors
/// [`crate::song_names`]'s derivation so the two dev-tools name a game the same way. Also names the
/// browser download on web.
pub fn filename_for(source: &str) -> String {
    let stem = source.rsplit_once('.').map_or(source, |(s, _)| s);
    let mut out = String::with_capacity(stem.len());
    let mut prev_us = false;
    for ch in stem.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_us = false;
        } else if !prev_us {
            out.push('_');
            prev_us = true;
        }
    }
    format!("{}.json", out.trim_matches('_'))
}

/// Reads a game's annotations. `Ok(None)` when the file doesn't exist yet (a game nobody has
/// annotated is not an error); `Err` only for a file that exists but can't be read or parsed —
/// which must never be silently swallowed, or a parse slip would look like "no labels" and the next
/// save would overwrite hours of work.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_file(path: &std::path::Path) -> Result<Option<GameAnnotations>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_str(&text)
        .map(Some)
        .map_err(|e| format!("parse {}: {e}", path.display()))
}

/// Writes a game's annotations, creating the directory if needed. Pretty-printed: these are
/// committed, and a one-line blob would make every diff unreadable.
#[cfg(not(target_arch = "wasm32"))]
pub fn save_file(path: &std::path::Path, file: &GameAnnotations) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    }
    let json = serde_json::to_string_pretty(file).map_err(|e| format!("encode: {e}"))?;
    std::fs::write(path, json).map_err(|e| format!("write {}: {e}", path.display()))
}

/// Session state for annotation mode.
#[derive(Default)]
pub struct AnnotationState {
    /// Whether the mode is on. Session-only, never persisted — it's a maintainer tool, and leaving
    /// it armed across restarts would be a trap.
    pub enabled: bool,
    /// The current song rendered to memory, shared with the audio thread.
    pub bounce: Option<Arc<Bounce>>,
    /// The song the current [`Self::bounce`] is for, so a song change invalidates it.
    pub bounce_for: Option<u32>,
    /// The render in flight, stepped a slice per frame (see [`BounceJob`]).
    pub job: Option<BounceJob>,
    /// Loaded labels for the current game, keyed by song id.
    pub songs: HashMap<u32, SongAnnotation>,
    /// Source archive the loaded labels belong to; a different one means reload.
    pub loaded_for: Option<String>,
    /// The selected `[start, end)` step range.
    ///
    /// This is the *only* notion of selection: a range can be chosen before any span exists there,
    /// because blank-slate authoring writes nothing until a chord is actually given. "What is
    /// selected" and "what is stored" are different questions, and only the first has an answer
    /// until you pick a chord.
    pub pending_range: Option<(u32, u32)>,
    /// Whether there are edits not yet written to disk.
    pub dirty: bool,
    /// Beat snap instead of the default bar snap.
    pub beat_snap: bool,
    /// Where the chord picker is open, in screen coordinates. `Some` right after a drag or a
    /// right-click: the picker is the primary way to label, so it comes to the region rather than
    /// making the eye travel to a toolbar and back for every bar.
    pub picker_at: Option<(f32, f32)>,
    /// Set on the frame the picker opens, so the very click that opened it can't also dismiss it.
    /// A right-click's press arrives in the same frame the popup first appears, and once the popup
    /// is constrained away from a screen edge it no longer sits under the cursor — so "was the
    /// press inside?" is the wrong question on frame one.
    pub picker_just_opened: bool,
    /// Whether the picker's "quality uncertain" box is ticked; sticks across picks because
    /// ambiguity usually comes in runs (a whole sparse section, not one bar).
    pub picker_uncertain: bool,
    /// Last save/load message for the toolbar.
    pub status: String,
}

impl AnnotationState {
    /// The roll's grid for a song: the device supplies steps-per-beat, the annotation supplies the
    /// meter and pickup offset (defaults until it does).
    pub fn grid_for(&self, song_id: u32, steps_per_beat: f64) -> Grid {
        match self.songs.get(&song_id) {
            Some(a) => Grid {
                steps_per_beat,
                beats_per_bar: a.beats_per_bar,
                offset_steps: a.grid_offset_steps as f64,
            },
            None => Grid {
                steps_per_beat,
                ..Grid::default()
            },
        }
    }

    /// The current song's annotation, created on first edit.
    pub fn song_mut(&mut self, song_id: u32) -> &mut SongAnnotation {
        self.songs
            .entry(song_id)
            .or_insert_with(|| SongAnnotation::new(song_id))
    }

    pub fn song(&self, song_id: u32) -> Option<&SongAnnotation> {
        self.songs.get(&song_id)
    }

    /// Inserts a span, keeping `spans` sorted by start and evicting anything it fully covers.
    /// Overlap is resolved in favour of the new span: the annotator's latest word wins.
    pub fn insert_span(&mut self, song_id: u32, span: Span) {
        let song = self.song_mut(song_id);
        song.spans
            .retain(|s| !(s.start_step >= span.start_step && s.end_step <= span.end_step));
        // Trim neighbours that only partially overlap, so spans stay non-overlapping.
        for s in song.spans.iter_mut() {
            if s.start_step < span.start_step && s.end_step > span.start_step {
                s.end_step = span.start_step;
            }
            if s.start_step < span.end_step && s.end_step > span.end_step {
                s.start_step = span.end_step;
            }
        }
        song.spans.retain(|s| s.end_step > s.start_step);
        song.spans.push(span);
        song.spans.sort_by_key(|s| s.start_step);
        self.dirty = true;
    }

    /// How many bars of a song carry a label, for the progress readout.
    pub fn labeled_bars(&self, song_id: u32, grid: Grid, total_steps: u32) -> (usize, usize) {
        let Some(bar_steps) = grid.bar_steps() else {
            return (0, 0);
        };
        let total = ((total_steps as f64 - grid.offset_steps) / bar_steps)
            .ceil()
            .max(0.0) as usize;
        let Some(song) = self.songs.get(&song_id) else {
            return (0, total);
        };
        let done = (0..total)
            .filter(|i| {
                let start = grid.offset_steps + *i as f64 * bar_steps;
                let mid = (start + bar_steps * 0.5) as u32;
                song.spans
                    .iter()
                    .any(|s| s.start_step <= mid && s.end_step > mid)
            })
            .count();
        (done, total)
    }

    /// Builds the file for the current game from the in-memory songs (sorted for a stable diff —
    /// these are committed, and a re-ordering diff would bury the real edit).
    pub fn to_file(
        &self,
        source: String,
        game_code: Option<String>,
        steps_per_beat: f64,
    ) -> GameAnnotations {
        let mut file = GameAnnotations::new(source, game_code, steps_per_beat);
        file.songs = self.songs.values().cloned().collect();
        file.songs.sort_by_key(|s| s.song_id);
        file
    }
}

#[cfg(test)]
mod tests {
    use super::model::{Chord, Quality};
    use super::*;

    fn span(start: u32, end: u32, root: u8) -> Span {
        Span {
            start_step: start,
            end_step: end,
            chord: Some(Chord {
                root,
                quality: Quality::Major,
                quality_uncertain: false,
            }),
        }
    }

    fn grid() -> Grid {
        Grid {
            steps_per_beat: 24.0,
            beats_per_bar: 4,
            offset_steps: 0.0,
        }
    }

    #[test]
    fn inserting_keeps_spans_sorted_and_non_overlapping() {
        let mut st = AnnotationState::default();
        st.insert_span(1, span(0, 96, 0));
        st.insert_span(1, span(192, 288, 7));
        // Straddles the first span's tail and the gap: the older neighbour is trimmed, not dropped.
        st.insert_span(1, span(48, 192, 5));
        let spans = &st.song(1).unwrap().spans;
        assert_eq!(spans.len(), 3);
        assert_eq!((spans[0].start_step, spans[0].end_step), (0, 48));
        assert_eq!((spans[1].start_step, spans[1].end_step), (48, 192));
        assert_eq!((spans[2].start_step, spans[2].end_step), (192, 288));
        assert!(st.dirty);
    }

    #[test]
    fn a_covering_span_replaces_what_it_covers() {
        let mut st = AnnotationState::default();
        st.insert_span(1, span(0, 96, 0));
        st.insert_span(1, span(96, 192, 2));
        st.insert_span(1, span(0, 192, 5));
        let spans = &st.song(1).unwrap().spans;
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].chord.unwrap().root, 5);
    }

    #[test]
    fn progress_counts_bars_with_a_label() {
        let mut st = AnnotationState::default();
        // 4 bars of 96 steps each.
        assert_eq!(st.labeled_bars(1, grid(), 384), (0, 4));
        st.insert_span(1, span(0, 96, 0));
        st.insert_span(1, span(192, 288, 7));
        assert_eq!(st.labeled_bars(1, grid(), 384), (2, 4));
    }

    #[test]
    fn grid_follows_the_annotation_then_falls_back() {
        let mut st = AnnotationState::default();
        assert_eq!(st.grid_for(1, 48.0).beats_per_bar, 4); // default meter
        assert_eq!(st.grid_for(1, 48.0).steps_per_beat, 48.0); // device's, always
        let s = st.song_mut(1);
        s.beats_per_bar = 3;
        s.grid_offset_steps = 24;
        let g = st.grid_for(1, 48.0);
        assert_eq!(g.beats_per_bar, 3);
        assert_eq!(g.offset_steps, 24.0);
    }
}

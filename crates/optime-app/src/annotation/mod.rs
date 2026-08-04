//! Chord-annotation mode: the annotation files on disk and the UI state holding the current selection, meter, and key.

mod bounce;
pub mod chord_voice;
pub mod model;

pub use bounce::{Bounce, BounceJob};
pub use chord_voice::ChordVoicer;

use std::collections::HashMap;
use std::sync::Arc;

use crate::piano_roll::Grid;
use model::{GameAnnotations, SongAnnotation, Span};

#[cfg(not(target_arch = "wasm32"))]
pub fn source_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ml/annotations")
}

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

#[cfg(not(target_arch = "wasm32"))]
pub fn save_file(path: &std::path::Path, file: &GameAnnotations) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    }
    let json = serde_json::to_string_pretty(file).map_err(|e| format!("encode: {e}"))?;
    std::fs::write(path, json).map_err(|e| format!("write {}: {e}", path.display()))
}

pub struct AnnotationState {
    pub enabled: bool,
    pub bounce: Option<Arc<Bounce>>,
    pub bounce_for: Option<u32>,
    pub job: Option<BounceJob>,
    pub songs: HashMap<u32, SongAnnotation>,
    pub loaded_for: Option<String>,
    pub pending_range: Option<(u32, u32)>,
    pub dirty: bool,
    pub beat_snap: bool,
    pub picker_at: Option<(f32, f32)>,
    pub picker_just_opened: bool,
    pub picker_uncertain: bool,
    pub picker_hovered: Option<model::Chord>,
    pub chord_gain: f32,
    pub chord_inversion: u8,
    pub status: String,
}

impl Default for AnnotationState {
    fn default() -> Self {
        Self {
            enabled: false,
            bounce: None,
            bounce_for: None,
            job: None,
            songs: Default::default(),
            loaded_for: None,
            pending_range: None,
            dirty: false,
            beat_snap: false,
            picker_at: None,
            picker_just_opened: false,
            picker_uncertain: false,
            picker_hovered: None,
            chord_gain: chord_voice::DEFAULT_GAIN,
            chord_inversion: 0,
            status: Default::default(),
        }
    }
}

impl AnnotationState {
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

    pub fn song_mut(&mut self, song_id: u32) -> &mut SongAnnotation {
        self.songs
            .entry(song_id)
            .or_insert_with(|| SongAnnotation::new(song_id))
    }

    pub fn song(&self, song_id: u32) -> Option<&SongAnnotation> {
        self.songs.get(&song_id)
    }

    pub fn insert_span(&mut self, song_id: u32, span: Span) {
        let song = self.song_mut(song_id);
        song.spans
            .retain(|s| !(s.start_step >= span.start_step && s.end_step <= span.end_step));
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
        assert_eq!(st.labeled_bars(1, grid(), 384), (0, 4));
        st.insert_span(1, span(0, 96, 0));
        st.insert_span(1, span(192, 288, 7));
        assert_eq!(st.labeled_bars(1, grid(), 384), (2, 4));
    }

    #[test]
    fn grid_follows_the_annotation_then_falls_back() {
        let mut st = AnnotationState::default();
        assert_eq!(st.grid_for(1, 48.0).beats_per_bar, 4);
        assert_eq!(st.grid_for(1, 48.0).steps_per_beat, 48.0);
        let s = st.song_mut(1);
        s.beats_per_bar = 3;
        s.grid_offset_steps = 24;
        let g = st.grid_for(1, 48.0);
        assert_eq!(g.beats_per_bar, 3);
        assert_eq!(g.offset_steps, 24.0);
    }
}

//! Training-free chord reference: chroma-template matching + Viterbi smoothing.
//!
//! This is the classic MIREX-style chord estimator — no neural net, no labels. It
//! scores each frame's (already L2-normalized) chroma against the pitch-class
//! template of every chord and Viterbi-smooths across time with a switch penalty.
//! Used as a *pseudo-reference* to score trained models against on unlabeled real
//! game songs (see the `eval_real` binary): it is not ground truth, but it gives a
//! single comparable "agreement %" number for the synthetic-only baseline vs. the
//! SSL-pretrained model.

use crate::features::{self, FeatureGrid};
use crate::notes::NoteEvent;
use crate::theory::{Chord, NO_CHORD, N_CHORD_CLASSES};

/// Cost subtracted for changing chord between adjacent frames (in cosine-score
/// units). Higher = fewer, longer chord segments.
const SWITCH_PENALTY: f32 = 0.25;
/// No-chord emission when the frame **is silent** (no active chroma). High so
/// genuine silence reads as no-chord.
const NO_CHORD_SILENT: f32 = 1.0;
/// No-chord emission when the frame **has notes**. Low so no-chord means *silence*,
/// not *ambiguous harmony* — otherwise, on busy real music where the best chord
/// changes every frame, the penalty-free no-chord state would win the whole
/// Viterbi path (each chord switch pays `SWITCH_PENALTY`, no-chord never does).
const NO_CHORD_ACTIVE: f32 = 0.05;

/// L2-normalized 12-dim pitch-class template per chord label (`0` = no-chord,
/// all zeros). Indexable by chord label in `[0, N_CHORD_CLASSES)`.
pub fn chord_templates() -> Vec<[f32; 12]> {
    let mut templates = vec![[0.0f32; 12]; N_CHORD_CLASSES];
    for (label, tmpl) in templates.iter_mut().enumerate().skip(1) {
        if let Some(chord) = Chord::from_label(label) {
            for pc in chord.pitch_classes() {
                tmpl[pc as usize] = 1.0;
            }
            let norm = tmpl.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in tmpl.iter_mut() {
                    *x /= norm;
                }
            }
        }
    }
    templates
}

/// Per-class emission score for one frame: cosine of the (L2-normalized) chroma
/// against each chord template. The no-chord state (slot 0) is gated on activity —
/// it only scores high when the frame is silent, so it represents silence rather
/// than harmonic ambiguity.
fn emission(chroma: &[f32], templates: &[[f32; 12]]) -> Vec<f32> {
    let mut e = vec![0.0f32; templates.len()];
    let active = chroma.iter().map(|x| x.abs()).sum::<f32>() > 1e-6;
    e[NO_CHORD] = if active {
        NO_CHORD_ACTIVE
    } else {
        NO_CHORD_SILENT
    };
    for (label, tmpl) in templates.iter().enumerate().skip(1) {
        e[label] = (0..12).map(|i| chroma[i] * tmpl[i]).sum();
    }
    e
}

/// Estimate a per-frame chord-label timeline for a feature grid via Viterbi over
/// the chroma-template emissions. Uniform switch cost lets each step run in `O(K)`
/// (track the single best previous state) instead of `O(K^2)`.
pub fn estimate_labels(grid: &FeatureGrid) -> Vec<usize> {
    let templates = chord_templates();
    let n = grid.n_frames;
    let k = templates.len();
    if n == 0 {
        return Vec::new();
    }

    let mut score = emission(&grid.row(0)[0..12], &templates);
    let mut back = vec![vec![0usize; k]; n];

    // `f` indexes both the frame's features and its backpointer row.
    #[allow(clippy::needless_range_loop)]
    for f in 1..n {
        let e = emission(&grid.row(f)[0..12], &templates);
        // Best previous state overall (the only candidate a *switch* can come from,
        // since every switch costs the same).
        let (mut best_prev, mut best_prev_v) = (0usize, f32::MIN);
        for (p, &v) in score.iter().enumerate() {
            if v > best_prev_v {
                best_prev_v = v;
                best_prev = p;
            }
        }
        let mut next = vec![f32::MIN; k];
        for s in 0..k {
            let stay = score[s]; // no penalty for staying on chord s
            let switch = best_prev_v - SWITCH_PENALTY;
            let (prev, val) = if stay >= switch {
                (s, stay)
            } else {
                (best_prev, switch)
            };
            next[s] = val + e[s];
            back[f][s] = prev;
        }
        score = next;
    }

    // Backtrack from the best final state.
    let (mut best, mut best_v) = (0usize, f32::MIN);
    for (s, &v) in score.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = s;
        }
    }
    let mut labels = vec![0usize; n];
    labels[n - 1] = best;
    for f in (1..n).rev() {
        labels[f - 1] = back[f][labels[f]];
    }
    labels
}

/// Convenience: estimate straight from a note-event stream.
pub fn estimate_from_notes(notes: &[NoteEvent], n_frames: usize) -> Vec<usize> {
    estimate_labels(&features::extract(notes, n_frames))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notes::{Instrument, NoteEvent};
    use crate::theory::Quality;

    #[test]
    fn recovers_sustained_c_major() {
        // A sustained C major triad (C E G) over 16 frames → C major throughout.
        let notes: Vec<NoteEvent> = [60u8, 64, 67]
            .iter()
            .map(|&p| NoteEvent {
                start_frame: 0,
                end_frame: 16,
                pitch: p,
                velocity: 1.0,
                instrument: Instrument::Harmony,
                pan: 0.0,
            })
            .collect();
        let labels = estimate_from_notes(&notes, 16);
        let c_major = Chord::new(0, Quality::Major).label();
        // Every frame should read C major.
        assert!(
            labels.iter().all(|&l| l == c_major),
            "expected all C major, got {labels:?}"
        );
    }

    #[test]
    fn silence_reads_no_chord() {
        let labels = estimate_from_notes(&[], 8);
        assert!(labels.iter().all(|&l| l == NO_CHORD));
    }
}

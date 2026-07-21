//! Note events -> per-frame feature grid.
//!
//! This is the single bridge between raw synthesizer notes and the model. The
//! same function runs on generated [`Song`]s during training and on a live slice
//! of OptimePlayer `NoteEvent`s during inference, so the model always sees an
//! identical representation.
//!
//! Every feature is derived purely from note pitch + metadata (velocity,
//! instrument role, pan, onset) — no audio rendering, no harmonic guessing.

use crate::notes::{Instrument, NoteEvent, Song};

// Feature block layout within a frame vector.
const CHROMA: usize = 0; // 12: velocity-weighted active pitch classes
const BASS: usize = 12; // 12: lowest sounding pitch class (root cue)
const MELODY: usize = 24; // 12: highest sounding pitch class
const ONSET: usize = 36; // 12: pitch classes attacking this frame
const SCALARS: usize = 48; // scalar block below

const S_POLYPHONY: usize = SCALARS;
const S_TOTAL_VEL: usize = SCALARS + 1;
const S_BASS_MIDI: usize = SCALARS + 2;
const S_MEAN_MIDI: usize = SCALARS + 3;
const S_PITCH_SPREAD: usize = SCALARS + 4;
const S_PAN_MEAN: usize = SCALARS + 5;
const S_PAN_SPREAD: usize = SCALARS + 6;
const S_PERC_ENERGY: usize = SCALARS + 7;
const S_ONSET_FLAG: usize = SCALARS + 8;

/// Length of one frame's feature vector.
pub const FEATURE_DIM: usize = SCALARS + 9;

/// Number of leading feature dims that make up the four L2-normalized pitch-class
/// blocks (chroma + bass + melody + onset). These are the harmonically meaningful
/// part of the vector and serve as the target for self-supervised masked-frame
/// reconstruction pretraining.
pub const PITCH_BLOCK_DIM: usize = SCALARS; // 48

/// Row-major `[n_frames, FEATURE_DIM]` feature matrix.
pub struct FeatureGrid {
    pub n_frames: usize,
    pub data: Vec<f32>,
}

impl FeatureGrid {
    #[inline]
    pub fn row(&self, frame: usize) -> &[f32] {
        &self.data[frame * FEATURE_DIM..(frame + 1) * FEATURE_DIM]
    }
}

/// Build the per-frame feature grid for a note-event stream of length `n_frames`.
pub fn extract(notes: &[NoteEvent], n_frames: usize) -> FeatureGrid {
    let mut data = vec![0.0f32; n_frames * FEATURE_DIM];

    // Per-frame scratch stats we can't accumulate purely additively.
    let mut min_pitch = vec![u8::MAX; n_frames];
    let mut min_vel = vec![0.0f32; n_frames];
    let mut max_pitch = vec![0u8; n_frames];
    let mut max_vel = vec![0.0f32; n_frames];
    let mut sum_midi = vec![0.0f32; n_frames];
    let mut poly = vec![0.0f32; n_frames];
    let mut pan_min = vec![f32::MAX; n_frames];
    let mut pan_max = vec![f32::MIN; n_frames];

    for note in notes {
        let start = note.start_frame as usize;
        let end = (note.end_frame as usize).min(n_frames);
        if start >= n_frames || start >= end {
            continue;
        }
        let pc = (note.pitch as usize) % 12;
        let perc = note.instrument.is_percussion();

        for f in start..end {
            let base = f * FEATURE_DIM;
            if perc {
                data[base + S_PERC_ENERGY] += note.velocity;
                continue;
            }
            data[base + CHROMA + pc] += note.velocity;
            data[base + S_TOTAL_VEL] += note.velocity;
            poly[f] += 1.0;
            sum_midi[f] += note.pitch as f32;
            pan_min[f] = pan_min[f].min(note.pan);
            pan_max[f] = pan_max[f].max(note.pan);
            data[base + S_PAN_MEAN] += note.pan;

            if note.pitch < min_pitch[f] {
                min_pitch[f] = note.pitch;
                min_vel[f] = note.velocity;
            }
            if note.pitch >= max_pitch[f] {
                max_pitch[f] = note.pitch;
                max_vel[f] = note.velocity;
            }
        }

        // Onset contribution (attack frame only).
        if !perc && start < n_frames {
            let base = start * FEATURE_DIM;
            data[base + ONSET + pc] += note.velocity;
            data[base + S_ONSET_FLAG] = 1.0;
        }
    }

    // Finalise per-frame derived features + normalisation.
    for f in 0..n_frames {
        let base = f * FEATURE_DIM;
        let row = &mut data[base..base + FEATURE_DIM];

        // Bass / melody pitch-class cues.
        if min_pitch[f] != u8::MAX {
            row[BASS + (min_pitch[f] as usize % 12)] = min_vel[f];
        }
        if max_vel[f] > 0.0 {
            row[MELODY + (max_pitch[f] as usize % 12)] = max_vel[f];
        }

        // Scalars.
        let p = poly[f];
        row[S_POLYPHONY] = (p / 8.0).min(1.0);
        if p > 0.0 {
            row[S_MEAN_MIDI] = ((sum_midi[f] / p) - 24.0) / 72.0;
            row[S_PAN_MEAN] /= p;
        }
        if min_pitch[f] != u8::MAX {
            row[S_BASS_MIDI] = (min_pitch[f] as f32 - 24.0) / 72.0;
            row[S_PITCH_SPREAD] = ((max_pitch[f] as f32 - min_pitch[f] as f32) / 48.0).min(1.0);
        }
        if pan_max[f] >= pan_min[f] {
            row[S_PAN_SPREAD] = (pan_max[f] - pan_min[f]) / 2.0;
        }
        row[S_PERC_ENERGY] = row[S_PERC_ENERGY].min(4.0) / 4.0;

        // L2-normalise the three pitch-class blocks so absolute loudness doesn't
        // dominate; keeps harmonic *shape* as the signal.
        l2_normalize(&mut row[CHROMA..CHROMA + 12]);
        l2_normalize(&mut row[BASS..BASS + 12]);
        l2_normalize(&mut row[MELODY..MELODY + 12]);
        l2_normalize(&mut row[ONSET..ONSET + 12]);
    }

    FeatureGrid { n_frames, data }
}

/// Convenience: extract straight from a generated [`Song`].
pub fn extract_song(song: &Song) -> FeatureGrid {
    extract(&song.notes, song.n_frames)
}

fn l2_normalize(block: &mut [f32]) {
    let norm: f32 = block.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-6 {
        for x in block.iter_mut() {
            *x /= norm;
        }
    }
}

/// Instrument-role helper kept public for downstream tooling / debugging.
pub fn instrument_index(instrument: Instrument) -> usize {
    instrument.index()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notes::render_song;
    use crate::theory::Key;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn grid_shape_and_finite() {
        let mut rng = StdRng::seed_from_u64(5);
        let song = render_song(&mut rng, &Key::from_label(9), 96);
        let grid = extract_song(&song);
        assert_eq!(grid.data.len(), 96 * FEATURE_DIM);
        assert!(grid.data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn chroma_reflects_a_major_triad() {
        // A single sustained A-major triad (A C# E) over 4 frames.
        let notes = vec![
            NoteEvent {
                start_frame: 0,
                end_frame: 4,
                pitch: 57,
                velocity: 1.0,
                instrument: Instrument::Harmony,
                track: 0,
                pan: 0.0,
            }, // A
            NoteEvent {
                start_frame: 0,
                end_frame: 4,
                pitch: 61,
                velocity: 1.0,
                instrument: Instrument::Harmony,
                track: 0,
                pan: 0.0,
            }, // C#
            NoteEvent {
                start_frame: 0,
                end_frame: 4,
                pitch: 64,
                velocity: 1.0,
                instrument: Instrument::Harmony,
                track: 0,
                pan: 0.0,
            }, // E
        ];
        let grid = extract(&notes, 4);
        let row = grid.row(0);
        // Pitch classes A=9, C#=1, E=4 should carry the energy; others ~0.
        for pc in 0..12 {
            let v = row[CHROMA + pc];
            if pc == 9 || pc == 1 || pc == 4 {
                assert!(v > 0.1, "expected energy at pc {pc}, got {v}");
            } else {
                assert!(v < 1e-3, "unexpected energy at pc {pc}: {v}");
            }
        }
        // Bass cue should point at A (lowest note, pc 9).
        assert!(row[BASS + 9] > 0.1);
    }
}

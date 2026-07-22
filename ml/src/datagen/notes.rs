//! Note events and arrangement. Turns a chord progression into a stream of
//! `NoteEvent`s with full synthesizer-style metadata (pitch, velocity,
//! instrument role, pan), plus the per-frame chord/key ground truth.
//!
//! This mirrors what OptimePlayer's `SynthEvent` stream carries at runtime, so a
//! model trained here consumes the same shape of data live.

use crate::progression;
use crate::theory::{Chord, Key, NO_CHORD};
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Time resolution: frames per beat. A frame is the model's time step.
pub const FRAMES_PER_BEAT: u32 = 4;

/// Instrument role of a voice — part of the note metadata the model sees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Instrument {
    Bass,
    Harmony,
    Arp,
    Melody,
    /// Unpitched percussion. Carries energy/metadata but no harmonic pitch class.
    Percussion,
}

impl Instrument {
    pub fn is_percussion(self) -> bool {
        matches!(self, Instrument::Percussion)
    }
    /// Stable index for optional per-instrument features.
    pub fn index(self) -> usize {
        match self {
            Instrument::Bass => 0,
            Instrument::Harmony => 1,
            Instrument::Arp => 2,
            Instrument::Melody => 3,
            Instrument::Percussion => 4,
        }
    }
    pub const COUNT: usize = 5;
}

/// A single note with synthesizer metadata. `end_frame` is exclusive.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NoteEvent {
    pub start_frame: u32,
    pub end_frame: u32,
    pub pitch: u8,
    pub velocity: f32,
    pub instrument: Instrument,
    /// Sequencer track / channel the note played on (0..15). For real songs this is
    /// the only instrument cue that survives harvesting (the `instrument` role is
    /// unknown → `Harmony`); the event model learns channel→instrument associations
    /// from it. Synthetic songs assign a fixed channel per voice ([`synthetic_channel`]).
    pub track: u8,
    pub pan: f32,
}

/// Fixed synthetic channel per voice role, so generated songs carry the same
/// `track` field shape the model sees on real harvested songs.
fn synthetic_channel(instrument: Instrument) -> u8 {
    match instrument {
        Instrument::Bass => 0,
        Instrument::Harmony => 1,
        Instrument::Arp => 2,
        Instrument::Melody => 3,
        Instrument::Percussion => 9,
    }
}

/// A fully-arranged song: note events + per-frame labels + the global key.
///
/// Implements `Default` (empty song) so construction sites can write only the
/// fields they mean (`..Song::default()`) — the optional metadata tail
/// (`is_music`/`source`/`doc_spans`/`label_mask`/`loop_frame`) defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Song {
    pub key_label: usize,
    pub n_frames: usize,
    pub notes: Vec<NoteEvent>,
    /// One chord label per frame (`theory::NO_CHORD` for silence/rests).
    pub chord_labels: Vec<usize>,
    /// Weak "is this real music" label, when known: `Some(true)` for a curated
    /// game track, `Some(false)` for an unlisted (SFX/jingle) song-table entry,
    /// `None` when unlabeled (synthetic songs, DS/unknown ROMs). Trains the frozen
    /// is-music probe only; ignored by the chord/key/SSL objectives.
    #[serde(default)]
    pub is_music: Option<bool>,
    /// Where the window came from: harvested archive's file stem (e.g.
    /// `pokemon-emerald`), `""` for synthetic. Drives the dashboard's per-ROM data
    /// breakdown. bincode isn't self-describing, so datasets saved before this field
    /// existed don't load — regenerate/re-harvest.
    #[serde(default)]
    pub source: String,
    /// Document layout for **packed** sequences: `(start, end)` frame spans of each
    /// constituent song, in order. The slot at each span's `end` is that document's
    /// **EOS slot**; frames after the last span's EOS are **pad**. Empty = the whole
    /// window is one document with no EOS/pad (the m00–m02 fixed-window layout).
    #[serde(default)]
    pub doc_spans: Vec<(u32, u32)>,
    /// Per-frame "this chord label is a real annotation" mask for partially
    /// hand-labelled songs (`None` = every frame's label is trustworthy). Frames
    /// with `false` are excluded from the supervised loss — an uncovered frame is
    /// *unheard*, not no-chord.
    #[serde(default)]
    pub label_mask: Option<Vec<bool>>,
    /// Frame at which the harvested song first looped (start of loop pass 2 of the
    /// intro+loop+loop export). Diagnostics only — the model gets no loop marker.
    #[serde(default)]
    pub loop_frame: Option<u32>,
}

impl Song {
    /// Transpose the whole song by `shift` semitones: note pitches, the global key,
    /// and every chord label move together (pitch classes rotate; chord *quality*
    /// and *mode*, velocities, timing, pan are unchanged). This is a correct, free
    /// augmentation — the same music in a different key — that teaches the model
    /// key-invariance and multiplies effective dataset size.
    pub fn transpose(&self, shift: i32) -> Song {
        if shift == 0 {
            return self.clone();
        }
        let notes = self
            .notes
            .iter()
            .map(|n| NoteEvent {
                pitch: (n.pitch as i32 + shift).clamp(0, 127) as u8,
                ..*n
            })
            .collect();
        let key = Key::from_label(self.key_label);
        let key_label =
            Key::new(((key.tonic as i32 + shift).rem_euclid(12)) as u8, key.mode).label();
        let chord_labels = self
            .chord_labels
            .iter()
            .map(|&l| match Chord::from_label(l) {
                Some(c) => {
                    Chord::new(((c.root as i32 + shift).rem_euclid(12)) as u8, c.quality).label()
                }
                None => NO_CHORD,
            })
            .collect();
        Song {
            key_label,
            n_frames: self.n_frames,
            notes,
            chord_labels,
            is_music: self.is_music,
            source: self.source.clone(),
            doc_spans: self.doc_spans.clone(),
            label_mask: self.label_mask.clone(),
            loop_frame: self.loop_frame,
        }
    }

    /// Real (non-EOS, non-pad) frame count: sum of `doc_spans` lengths, or all of
    /// `n_frames` when the song is a plain single-document window.
    pub fn real_frames(&self) -> usize {
        if self.doc_spans.is_empty() {
            self.n_frames
        } else {
            self.doc_spans.iter().map(|&(s, e)| (e - s) as usize).sum()
        }
    }
}

/// A random transposition covering all 12 pitch-class rotations while keeping the
/// register shift small (`[-5, +6]` semitones, so notes rarely hit the MIDI range
/// limits). Use per song per epoch for on-the-fly key augmentation.
/// Distinct shifts [`random_transpose`] draws from (-5..=6) — every pitch-class
/// rotation exactly once, so augmentation multiplies the reachable data by this.
pub const N_TRANSPOSITIONS: usize = 12;

pub fn random_transpose<R: Rng>(rng: &mut R) -> i32 {
    let r = rng.gen_range(0..12);
    if r <= 6 {
        r
    } else {
        r - 12
    }
}

#[derive(Clone, Copy)]
enum Style {
    Block,    // sustained pads + walking-ish bass
    Comp,     // rhythmic chord stabs on beats
    Arpeggio, // broken chords
}

/// Put a pitch class into an octave so the resulting MIDI note is closest to
/// `anchor` (used for smooth voice leading / register control).
fn pc_near(pc: u8, anchor: i32) -> u8 {
    let mut note = pc as i32;
    while note < anchor - 6 {
        note += 12;
    }
    while note > anchor + 6 {
        note -= 12;
    }
    note.clamp(0, 127) as u8
}

/// Voice a chord as MIDI notes in root position starting near `base`.
fn voice_chord(chord: &Chord, base: i32, inversion: usize) -> Vec<u8> {
    let ivs = chord.quality.intervals();
    // Voice from the root placed near `base`, then stack the quality's intervals.
    let root_note = pc_near(chord.root, base);
    let mut notes: Vec<u8> = ivs
        .iter()
        .map(|iv| (root_note as i32 + iv).clamp(0, 127) as u8)
        .collect();
    // Apply inversion by lifting the lowest `inversion` tones an octave.
    for note in notes.iter_mut().take(inversion % ivs.len()) {
        *note = (*note as i32 + 12).clamp(0, 127) as u8;
    }
    notes.sort_unstable();
    notes
}

/// Render a random song in `key` spanning exactly `n_frames`.
pub fn render_song<R: Rng>(rng: &mut R, key: &Key, n_frames: usize) -> Song {
    let min_beats = (n_frames as u32).div_ceil(FRAMES_PER_BEAT) + 4;
    let prog = progression::generate(rng, key, min_beats);

    let style = *[Style::Block, Style::Comp, Style::Arpeggio]
        .choose(rng)
        .unwrap();
    let has_melody = rng.gen_bool(0.6);
    let has_perc = rng.gen_bool(0.4);
    let bass_octave = *[36i32, 40, 43].choose(rng).unwrap();
    let harmony_base = *[52i32, 55, 57, 60].choose(rng).unwrap();

    let mut notes: Vec<NoteEvent> = Vec::new();
    let mut chord_labels = vec![NO_CHORD; n_frames];

    let mut frame: u32 = 0;
    let mut melody_note: i32 = harmony_base + 12;

    'outer: for (chord, beats) in &prog {
        let span_frames = beats * FRAMES_PER_BEAT;
        let span_start = frame;
        let span_end = (frame + span_frames).min(n_frames as u32);
        if span_start >= n_frames as u32 {
            break;
        }

        // Label every frame in this span with the current chord.
        for f in span_start..span_end {
            chord_labels[f as usize] = chord.label();
        }

        let inversion = if rng.gen_bool(0.3) {
            rng.gen_range(0..3)
        } else {
            0
        };
        let voiced = voice_chord(chord, harmony_base, inversion);
        let bass_pitch = pc_near(chord.root, bass_octave);

        // --- Bass ---
        add_bass(rng, &mut notes, style, bass_pitch, span_start, span_end);

        // --- Harmony / arp ---
        match style {
            Style::Block => {
                for &p in &voiced {
                    notes.push(NoteEvent {
                        start_frame: span_start,
                        end_frame: span_end,
                        pitch: p,
                        velocity: rng.gen_range(0.45..0.65),
                        instrument: Instrument::Harmony,
                        track: 0,
                        pan: rng.gen_range(-0.35..0.35),
                    });
                }
            }
            Style::Comp => {
                let mut f = span_start;
                while f < span_end {
                    let hit_len = FRAMES_PER_BEAT.min(span_end - f);
                    for &p in &voiced {
                        notes.push(NoteEvent {
                            start_frame: f,
                            end_frame: f + hit_len,
                            pitch: p,
                            velocity: rng.gen_range(0.5..0.7),
                            instrument: Instrument::Harmony,
                            track: 0,
                            pan: rng.gen_range(-0.3..0.3),
                        });
                    }
                    f += FRAMES_PER_BEAT;
                }
            }
            Style::Arpeggio => {
                let step = 1u32.max(FRAMES_PER_BEAT / 2); // eighth notes
                let mut f = span_start;
                let mut i = 0usize;
                while f < span_end {
                    let p = voiced[i % voiced.len()];
                    let len = step.min(span_end - f);
                    notes.push(NoteEvent {
                        start_frame: f,
                        end_frame: f + len,
                        pitch: p,
                        velocity: rng.gen_range(0.4..0.6),
                        instrument: Instrument::Arp,
                        track: 0,
                        pan: rng.gen_range(-0.5..0.5),
                    });
                    f += step;
                    i += 1;
                }
            }
        }

        // --- Melody (chord tones on strong beats, passing scale tones between) ---
        if has_melody {
            add_melody(
                rng,
                &mut notes,
                key,
                chord,
                &mut melody_note,
                span_start,
                span_end,
            );
        }

        // --- Percussion (unpitched; metadata/energy only) ---
        if has_perc {
            let mut f = span_start;
            while f < span_end {
                notes.push(NoteEvent {
                    start_frame: f,
                    end_frame: f + 1,
                    pitch: 36 + (rng.gen_range(0..3) as u8), // kick/snare-ish, ignored as pitch
                    velocity: rng.gen_range(0.5..0.9),
                    instrument: Instrument::Percussion,
                    track: 0,
                    pan: 0.0,
                });
                f += FRAMES_PER_BEAT / 2;
            }
        }

        frame = span_end;
        if frame >= n_frames as u32 {
            break 'outer;
        }
    }

    // Occasionally open or close with a short rest for NO_CHORD examples.
    maybe_insert_rests(rng, &mut notes, &mut chord_labels, n_frames);

    // Assign each voice its fixed synthetic channel (the literals leave `track` at 0).
    for n in notes.iter_mut() {
        n.track = synthetic_channel(n.instrument);
    }

    Song {
        key_label: key.label(),
        n_frames,
        notes,
        chord_labels,
        is_music: None,
        ..Song::default() // synthetic songs are unlabeled for the is-music probe
    }
}

fn add_bass<R: Rng>(
    rng: &mut R,
    notes: &mut Vec<NoteEvent>,
    style: Style,
    bass_pitch: u8,
    start: u32,
    end: u32,
) {
    match style {
        Style::Block => {
            notes.push(NoteEvent {
                start_frame: start,
                end_frame: end,
                pitch: bass_pitch,
                velocity: rng.gen_range(0.6..0.85),
                instrument: Instrument::Bass,
                track: 0,
                pan: 0.0,
            });
        }
        _ => {
            // Bass on each beat.
            let mut f = start;
            while f < end {
                let len = FRAMES_PER_BEAT.min(end - f);
                notes.push(NoteEvent {
                    start_frame: f,
                    end_frame: f + len,
                    pitch: bass_pitch,
                    velocity: rng.gen_range(0.6..0.85),
                    instrument: Instrument::Bass,
                    track: 0,
                    pan: 0.0,
                });
                f += FRAMES_PER_BEAT;
            }
        }
    }
}

fn add_melody<R: Rng>(
    rng: &mut R,
    notes: &mut Vec<NoteEvent>,
    key: &Key,
    chord: &Chord,
    melody_note: &mut i32,
    start: u32,
    end: u32,
) {
    let chord_pcs = chord.pitch_classes();
    let scale = key.scale();
    let mut f = start;
    let mut strong = true;
    while f < end {
        let dur = *[FRAMES_PER_BEAT, FRAMES_PER_BEAT / 2, FRAMES_PER_BEAT * 2]
            .choose(rng)
            .unwrap()
            .min(&(end - f));
        // Strong beats favour chord tones; weak beats allow scale passing tones.
        let target_pc = if strong || rng.gen_bool(0.5) {
            *chord_pcs.choose(rng).unwrap()
        } else {
            let deg = rng.gen_range(0..scale.len());
            key.degree_pc(deg)
        };
        // Move melody toward the target pitch class near current register.
        let cand = pc_near(target_pc, *melody_note);
        *melody_note = (cand as i32).clamp(60, 84);
        if rng.gen_bool(0.85) {
            notes.push(NoteEvent {
                start_frame: f,
                end_frame: f + dur,
                pitch: *melody_note as u8,
                velocity: rng.gen_range(0.55..0.8),
                instrument: Instrument::Melody,
                track: 0,
                pan: rng.gen_range(-0.2..0.4),
            });
        }
        f += dur;
        strong = !strong;
    }
}

fn maybe_insert_rests<R: Rng>(
    rng: &mut R,
    notes: &mut Vec<NoteEvent>,
    chord_labels: &mut [usize],
    n_frames: usize,
) {
    if n_frames < 16 || !rng.gen_bool(0.25) {
        return;
    }
    // Clear a short window (2-4 frames) to create a genuine silence -> NO_CHORD.
    let len = rng.gen_range(2..5).min(n_frames / 4);
    let at_end = rng.gen_bool(0.5);
    let start = if at_end { n_frames - len } else { 0 };
    let range = start as u32..(start + len) as u32;
    notes.retain(|n| n.start_frame >= range.end || n.end_frame <= range.start);
    for f in range.clone() {
        chord_labels[f as usize] = NO_CHORD;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::{Chord, Mode, Quality};
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn transpose_shifts_notes_key_and_chords_together() {
        // C major key, a C-major then G-major chord, one C4 note.
        let song = Song {
            key_label: Key::new(0, Mode::Major).label(),
            n_frames: 2,
            notes: vec![NoteEvent {
                start_frame: 0,
                end_frame: 2,
                pitch: 60, // C4
                velocity: 1.0,
                instrument: Instrument::Harmony,
                track: 0,
                pan: 0.0,
            }],
            chord_labels: vec![
                Chord::new(0, Quality::Major).label(), // C
                Chord::new(7, Quality::Major).label(), // G
            ],
            is_music: Some(true),
            ..Song::default()
        };
        // Up 2 semitones: C→D key, notes +2, chord roots +2 (C→D, G→A), quality kept.
        let t = song.transpose(2);
        assert_eq!(t.key_label, Key::new(2, Mode::Major).label());
        assert_eq!(t.notes[0].pitch, 62); // D4
        assert_eq!(t.chord_labels[0], Chord::new(2, Quality::Major).label()); // D
        assert_eq!(t.chord_labels[1], Chord::new(9, Quality::Major).label()); // A
        assert_eq!(t.is_music, Some(true)); // metadata preserved
                                            // Wrap-around: up 7 from G (7) → D (2).
        let up7 = song.transpose(7);
        assert_eq!(up7.chord_labels[1], Chord::new(2, Quality::Major).label());
        // NO_CHORD stays NO_CHORD.
        let mut silent = song.clone();
        silent.chord_labels[0] = NO_CHORD;
        assert_eq!(silent.transpose(5).chord_labels[0], NO_CHORD);
        // shift 0 is identity.
        assert_eq!(song.transpose(0).notes[0].pitch, 60);
    }

    #[test]
    fn random_transpose_covers_all_rotations_in_range() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..500 {
            let s = random_transpose(&mut rng);
            assert!((-5..=6).contains(&s), "shift {s} out of range");
            seen.insert(s.rem_euclid(12));
        }
        assert_eq!(seen.len(), 12, "should cover all 12 pitch-class rotations");
    }

    #[test]
    fn song_fills_frames_and_labels() {
        let mut rng = StdRng::seed_from_u64(7);
        let key = Key::from_label(3);
        let song = render_song(&mut rng, &key, 128);
        assert_eq!(song.chord_labels.len(), 128);
        assert_eq!(song.n_frames, 128);
        assert!(!song.notes.is_empty());
        // All notes are within the frame range.
        for n in &song.notes {
            assert!(n.end_frame <= 128);
            assert!(n.start_frame < n.end_frame);
        }
        // At least most frames carry a real chord label.
        let voiced = song.chord_labels.iter().filter(|&&l| l != NO_CHORD).count();
        assert!(voiced > 100, "voiced frames = {voiced}");
    }

    #[test]
    fn chord_labels_decode() {
        let mut rng = StdRng::seed_from_u64(11);
        let key = Key::from_label(0);
        let song = render_song(&mut rng, &key, 64);
        for &l in &song.chord_labels {
            if l != NO_CHORD {
                assert!(Chord::from_label(l).is_some());
            }
        }
    }
}

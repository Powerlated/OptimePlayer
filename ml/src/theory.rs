//! Music-theory core: pitch classes, chord qualities, keys, diatonic harmony,
//! and a label vocabulary shared by the data generator and the model heads.
//!
//! Everything here is deterministic and dependency-free so the label space is
//! identical between offline data generation and live inference on OptimePlayer's
//! `SynthEvent` stream.

use serde::{Deserialize, Serialize};

/// Number of pitch classes (C, C#, D, ... B).
pub const N_PITCH_CLASSES: usize = 12;

pub const PITCH_CLASS_NAMES: [&str; N_PITCH_CLASSES] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// Chord qualities the model can emit. Order is stable — it defines the label
/// index (`quality as usize`). Intervals are semitones above the root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Quality {
    Major,           // 0 4 7
    Minor,           // 0 3 7
    Diminished,      // 0 3 6
    Augmented,       // 0 4 8
    Dominant7,       // 0 4 7 10
    Major7,          // 0 4 7 11
    Minor7,          // 0 3 7 10
    HalfDiminished7, // 0 3 6 10  (m7b5)
    Sus2,            // 0 2 7
    Sus4,            // 0 5 7
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

    /// Semitone intervals above the root that make up the chord.
    pub fn intervals(self) -> &'static [i32] {
        match self {
            Quality::Major => &[0, 4, 7],
            Quality::Minor => &[0, 3, 7],
            Quality::Diminished => &[0, 3, 6],
            Quality::Augmented => &[0, 4, 8],
            Quality::Dominant7 => &[0, 4, 7, 10],
            Quality::Major7 => &[0, 4, 7, 11],
            Quality::Minor7 => &[0, 3, 7, 10],
            Quality::HalfDiminished7 => &[0, 3, 6, 10],
            Quality::Sus2 => &[0, 2, 7],
            Quality::Sus4 => &[0, 5, 7],
        }
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Quality::Major => "",
            Quality::Minor => "m",
            Quality::Diminished => "dim",
            Quality::Augmented => "aug",
            Quality::Dominant7 => "7",
            Quality::Major7 => "maj7",
            Quality::Minor7 => "m7",
            Quality::HalfDiminished7 => "m7b5",
            Quality::Sus2 => "sus2",
            Quality::Sus4 => "sus4",
        }
    }
}

/// A concrete chord = root pitch class (0..12) + quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Chord {
    pub root: u8,
    pub quality: Quality,
}

impl Chord {
    pub fn new(root: u8, quality: Quality) -> Self {
        Chord {
            root: root % 12,
            quality,
        }
    }

    /// Absolute pitch classes contained in the chord (0..12).
    pub fn pitch_classes(&self) -> Vec<u8> {
        self.quality
            .intervals()
            .iter()
            .map(|iv| ((self.root as i32 + iv).rem_euclid(12)) as u8)
            .collect()
    }

    pub fn name(&self) -> String {
        format!(
            "{}{}",
            PITCH_CLASS_NAMES[self.root as usize],
            self.quality.symbol()
        )
    }

    /// Roman-numeral analysis of this chord **relative to `key`**, e.g. `V7`, `ii`,
    /// `vii°`, `bVII`. The base numeral comes from the chromatic interval of the
    /// chord root above the key tonic (diatonic degrees get a plain numeral,
    /// chromatic ones a `b`/`#` accidental); case + decoration encode the quality
    /// (upper for major-ish, lower for minor-ish, `°`/`+`/`ø` for dim/aug/m7b5).
    pub fn roman(&self, key: &Key) -> String {
        // Base uppercase numeral per chromatic degree, spelled for the key's mode.
        const MAJOR: [&str; 12] = [
            "I", "bII", "II", "bIII", "III", "IV", "#IV", "V", "bVI", "VI", "bVII", "VII",
        ];
        const MINOR: [&str; 12] = [
            "I", "bII", "II", "III", "#III", "IV", "#IV", "V", "VI", "#VI", "VII", "#VII",
        ];
        let interval = (self.root as i32 - key.tonic as i32).rem_euclid(12) as usize;
        let base = match key.mode {
            Mode::Major => MAJOR[interval],
            Mode::Minor => MINOR[interval],
        };

        // Major-ish qualities keep the uppercase numeral; minor-ish lowercase it.
        let upper = matches!(
            self.quality,
            Quality::Major
                | Quality::Augmented
                | Quality::Dominant7
                | Quality::Major7
                | Quality::Sus2
                | Quality::Sus4
        );
        let numeral = if upper {
            base.to_string()
        } else {
            base.to_lowercase()
        };

        let suffix = match self.quality {
            Quality::Major | Quality::Minor => "",
            Quality::Diminished => "°",
            Quality::Augmented => "+",
            Quality::Dominant7 => "7",
            Quality::Major7 => "maj7",
            Quality::Minor7 => "7",
            Quality::HalfDiminished7 => "ø7",
            Quality::Sus2 => "sus2",
            Quality::Sus4 => "sus4",
        };
        format!("{numeral}{suffix}")
    }

    /// Label index in `[0, N_CHORD_CLASSES)`. Index 0 is reserved for "no chord".
    pub fn label(&self) -> usize {
        let q = self.quality as usize;
        1 + q * N_PITCH_CLASSES + self.root as usize
    }

    /// Inverse of [`Chord::label`]. Returns `None` for the no-chord label 0.
    pub fn from_label(label: usize) -> Option<Chord> {
        if label == 0 {
            return None;
        }
        let idx = label - 1;
        let root = (idx % N_PITCH_CLASSES) as u8;
        let q = idx / N_PITCH_CLASSES;
        Some(Chord {
            root,
            quality: Quality::ALL[q],
        })
    }
}

/// Total chord label count: 1 (no-chord) + 12 roots * 10 qualities.
pub const N_CHORD_CLASSES: usize = 1 + N_PITCH_CLASSES * Quality::ALL.len();
/// Label index for silence / no active harmony.
pub const NO_CHORD: usize = 0;

/// Root-class count for the factored chord head: 1 (none) + 12 pitch classes.
pub const N_ROOT_CLASSES: usize = 1 + N_PITCH_CLASSES;
/// Quality-class count for the factored chord head: 1 (none) + the qualities.
pub const N_QUALITY_CLASSES: usize = 1 + Quality::ALL.len();

/// Split a joint chord label `[0, N_CHORD_CLASSES)` into `(root_class, quality_class)`
/// for the two factored heads. No-chord (0) maps to `(0, 0)`; a real chord to
/// `(root_pc + 1, quality_index + 1)`, so class 0 is reserved for "none" in both
/// factors (letting the heads train with a plain cross-entropy over every frame).
pub fn chord_label_to_root_quality(label: usize) -> (usize, usize) {
    if label == NO_CHORD {
        return (0, 0);
    }
    let idx = label - 1;
    let root_pc = idx % N_PITCH_CLASSES;
    let quality = idx / N_PITCH_CLASSES;
    (root_pc + 1, quality + 1)
}

/// Recombine factored `(root_class, quality_class)` predictions into a joint chord
/// label. A "none" in either factor (class 0) yields [`NO_CHORD`].
pub fn root_quality_to_chord_label(root_class: usize, quality_class: usize) -> usize {
    if root_class == 0 || quality_class == 0 {
        return NO_CHORD;
    }
    1 + (quality_class - 1) * N_PITCH_CLASSES + (root_class - 1)
}

/// Key mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Mode {
    Major,
    Minor,
}

/// A musical key = tonic pitch class + mode. 24 total.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Key {
    pub tonic: u8,
    pub mode: Mode,
}

/// 24 keys.
pub const N_KEY_CLASSES: usize = N_PITCH_CLASSES * 2;

impl Key {
    pub fn new(tonic: u8, mode: Mode) -> Self {
        Key {
            tonic: tonic % 12,
            mode,
        }
    }

    /// Label index `[0,24)`: major keys 0..12, minor keys 12..24.
    pub fn label(&self) -> usize {
        let base = match self.mode {
            Mode::Major => 0,
            Mode::Minor => 12,
        };
        base + self.tonic as usize
    }

    pub fn from_label(label: usize) -> Key {
        if label < 12 {
            Key::new(label as u8, Mode::Major)
        } else {
            Key::new((label - 12) as u8, Mode::Minor)
        }
    }

    pub fn name(&self) -> String {
        let m = match self.mode {
            Mode::Major => "maj",
            Mode::Minor => "min",
        };
        format!("{} {}", PITCH_CLASS_NAMES[self.tonic as usize], m)
    }

    /// Scale degrees (semitone offsets from tonic) for the key.
    pub fn scale(&self) -> &'static [i32] {
        match self.mode {
            // Ionian.
            Mode::Major => &[0, 2, 4, 5, 7, 9, 11],
            // Natural minor (Aeolian); harmonic/melodic colour is added by the
            // progression templates that borrow a major V etc.
            Mode::Minor => &[0, 2, 3, 5, 7, 8, 10],
        }
    }

    /// Absolute pitch class of scale degree `degree` (0-indexed, wraps octave).
    pub fn degree_pc(&self, degree: usize) -> u8 {
        let scale = self.scale();
        let d = degree % scale.len();
        ((self.tonic as i32 + scale[d]).rem_euclid(12)) as u8
    }

    /// The diatonic triad built on scale degree `degree` (0 = tonic).
    pub fn diatonic_triad(&self, degree: usize) -> Chord {
        let root = self.degree_pc(degree);
        let third = self.degree_pc(degree + 2);
        let fifth = self.degree_pc(degree + 4);
        let third_iv = (third as i32 - root as i32).rem_euclid(12);
        let fifth_iv = (fifth as i32 - root as i32).rem_euclid(12);
        let quality = match (third_iv, fifth_iv) {
            (4, 7) => Quality::Major,
            (3, 7) => Quality::Minor,
            (3, 6) => Quality::Diminished,
            (4, 8) => Quality::Augmented,
            // Fallback for exotic modes; treat as major.
            _ => Quality::Major,
        };
        Chord::new(root, quality)
    }

    /// The diatonic seventh chord on `degree`.
    pub fn diatonic_seventh(&self, degree: usize) -> Chord {
        let triad = self.diatonic_triad(degree);
        let root = triad.root;
        let seventh = self.degree_pc(degree + 6);
        let seventh_iv = (seventh as i32 - root as i32).rem_euclid(12);
        let quality = match (triad.quality, seventh_iv) {
            (Quality::Major, 11) => Quality::Major7,
            (Quality::Major, 10) => Quality::Dominant7,
            (Quality::Minor, 10) => Quality::Minor7,
            (Quality::Diminished, 10) => Quality::HalfDiminished7,
            (Quality::Diminished, 9) => Quality::Diminished, // fully diminished ~ dim
            _ => triad.quality,
        };
        Chord::new(root, quality)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chord_label_roundtrip() {
        for q in Quality::ALL {
            for root in 0..12u8 {
                let c = Chord::new(root, q);
                let l = c.label();
                assert!(l < N_CHORD_CLASSES);
                assert_eq!(Chord::from_label(l), Some(c));
            }
        }
        assert_eq!(Chord::from_label(NO_CHORD), None);
    }

    #[test]
    fn roman_numerals_in_c_major() {
        let c = Key::new(0, Mode::Major);
        assert_eq!(Chord::new(0, Quality::Major).roman(&c), "I"); // C
        assert_eq!(Chord::new(2, Quality::Minor).roman(&c), "ii"); // Dm
        assert_eq!(Chord::new(7, Quality::Dominant7).roman(&c), "V7"); // G7
        assert_eq!(Chord::new(9, Quality::Minor).roman(&c), "vi"); // Am
        assert_eq!(Chord::new(11, Quality::Diminished).roman(&c), "vii°"); // B°
        assert_eq!(Chord::new(0, Quality::Major7).roman(&c), "Imaj7"); // Cmaj7
                                                                       // Borrowed bVII (Bb major) — chromatic degree, flat accidental.
        assert_eq!(Chord::new(10, Quality::Major).roman(&c), "bVII");
        // Half-diminished ii in a minor-key context.
        let a_min = Key::new(9, Mode::Minor);
        assert_eq!(
            Chord::new(11, Quality::HalfDiminished7).roman(&a_min),
            "iiø7"
        );
    }

    #[test]
    fn factored_root_quality_roundtrip() {
        // No-chord splits to (none, none) and recombines back.
        assert_eq!(chord_label_to_root_quality(NO_CHORD), (0, 0));
        assert_eq!(root_quality_to_chord_label(0, 0), NO_CHORD);
        // Every real chord label round-trips through the factored representation.
        for q in Quality::ALL {
            for root in 0..12u8 {
                let label = Chord::new(root, q).label();
                let (rc, qc) = chord_label_to_root_quality(label);
                assert!((1..=N_PITCH_CLASSES).contains(&rc));
                assert!((1..N_QUALITY_CLASSES).contains(&qc));
                assert_eq!(root_quality_to_chord_label(rc, qc), label);
            }
        }
    }

    #[test]
    fn key_label_roundtrip() {
        for label in 0..N_KEY_CLASSES {
            assert_eq!(Key::from_label(label).label(), label);
        }
    }

    #[test]
    fn c_major_diatonic_triads() {
        let k = Key::new(0, Mode::Major);
        // I ii iii IV V vi vii°
        assert_eq!(k.diatonic_triad(0), Chord::new(0, Quality::Major)); // C
        assert_eq!(k.diatonic_triad(1), Chord::new(2, Quality::Minor)); // Dm
        assert_eq!(k.diatonic_triad(2), Chord::new(4, Quality::Minor)); // Em
        assert_eq!(k.diatonic_triad(3), Chord::new(5, Quality::Major)); // F
        assert_eq!(k.diatonic_triad(4), Chord::new(7, Quality::Major)); // G
        assert_eq!(k.diatonic_triad(5), Chord::new(9, Quality::Minor)); // Am
        assert_eq!(k.diatonic_triad(6), Chord::new(11, Quality::Diminished)); // B°
    }

    #[test]
    fn c_major_diatonic_sevenths() {
        let k = Key::new(0, Mode::Major);
        assert_eq!(k.diatonic_seventh(0), Chord::new(0, Quality::Major7)); // Cmaj7
        assert_eq!(k.diatonic_seventh(1), Chord::new(2, Quality::Minor7)); // Dm7
        assert_eq!(k.diatonic_seventh(4), Chord::new(7, Quality::Dominant7)); // G7
        assert_eq!(
            k.diatonic_seventh(6),
            Chord::new(11, Quality::HalfDiminished7)
        ); // Bm7b5
    }
}

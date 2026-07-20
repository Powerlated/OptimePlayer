//! Chord-progression generation. Produces a sequence of `(Chord, beats)` spans
//! in a given key, either from a library of real-world progressions (pop, jazz,
//! blues, Pachelbel, Andalusian, ...) or from a functional-harmony Markov walk.
//!
//! Secondary dominants and borrowed chords are injected so the model sees
//! chromatic harmony that still belongs to a single global key.

use crate::theory::{Chord, Key, Mode, Quality};
use rand::seq::SliceRandom;
use rand::Rng;

/// How to build a concrete chord from the current key.
#[derive(Debug, Clone, Copy)]
enum Deg {
    /// Diatonic triad on scale degree (0 = tonic).
    Triad(usize),
    /// Diatonic seventh chord on scale degree.
    Seventh(usize),
    /// Secondary dominant (V7) resolving to the chord on `degree`.
    SecondaryDom(usize),
    /// Explicit chord: semitone offset above tonic + quality (borrowed/chromatic).
    Fixed(i32, Quality),
}

impl Deg {
    fn resolve(self, key: &Key) -> Chord {
        match self {
            Deg::Triad(d) => key.diatonic_triad(d),
            Deg::Seventh(d) => key.diatonic_seventh(d),
            Deg::SecondaryDom(d) => {
                let target = key.degree_pc(d);
                Chord::new((target + 7) % 12, Quality::Dominant7)
            }
            Deg::Fixed(off, q) => Chord::new(((key.tonic as i32 + off).rem_euclid(12)) as u8, q),
        }
    }
}

struct Template {
    /// Which modes this template is musically valid for.
    modes: &'static [Mode],
    /// The chords, each with a duration weight in beats.
    steps: &'static [(Deg, u32)],
}

use Deg::*;
use Mode::{Major, Minor};
use Quality::{Dominant7, HalfDiminished7};

/// Curated progression templates. Beats are relative; the renderer scales them.
static TEMPLATES: &[Template] = &[
    // I–V–vi–IV (axis of awesome / pop).
    Template {
        modes: &[Major],
        steps: &[(Triad(0), 4), (Triad(4), 4), (Triad(5), 4), (Triad(3), 4)],
    },
    // I–vi–IV–V (50s doo-wop).
    Template {
        modes: &[Major],
        steps: &[(Triad(0), 4), (Triad(5), 4), (Triad(3), 4), (Triad(4), 4)],
    },
    // I–IV–V–I.
    Template {
        modes: &[Major],
        steps: &[(Triad(0), 4), (Triad(3), 4), (Triad(4), 4), (Triad(0), 4)],
    },
    // vi–ii–V–I (circle of fifths, jazzy sevenths).
    Template {
        modes: &[Major],
        steps: &[
            (Seventh(5), 4),
            (Seventh(1), 4),
            (Seventh(4), 4),
            (Seventh(0), 4),
        ],
    },
    // ii–V–I turnaround.
    Template {
        modes: &[Major],
        steps: &[(Seventh(1), 4), (Seventh(4), 4), (Seventh(0), 8)],
    },
    // Pachelbel's canon: I–V–vi–iii–IV–I–IV–V.
    Template {
        modes: &[Major],
        steps: &[
            (Triad(0), 2),
            (Triad(4), 2),
            (Triad(5), 2),
            (Triad(2), 2),
            (Triad(3), 2),
            (Triad(0), 2),
            (Triad(3), 2),
            (Triad(4), 2),
        ],
    },
    // 12-bar blues (dominant 7ths throughout).
    Template {
        modes: &[Major],
        steps: &[
            (Fixed(0, Dominant7), 4),
            (Fixed(5, Dominant7), 4),
            (Fixed(0, Dominant7), 4),
            (Fixed(0, Dominant7), 4),
            (Fixed(5, Dominant7), 4),
            (Fixed(5, Dominant7), 4),
            (Fixed(0, Dominant7), 4),
            (Fixed(0, Dominant7), 4),
            (Fixed(7, Dominant7), 4),
            (Fixed(5, Dominant7), 4),
            (Fixed(0, Dominant7), 4),
            (Fixed(7, Dominant7), 4),
        ],
    },
    // Secondary dominant: I – V7/V – V – I.
    Template {
        modes: &[Major],
        steps: &[
            (Triad(0), 4),
            (SecondaryDom(4), 4),
            (Triad(4), 4),
            (Triad(0), 4),
        ],
    },
    // I – vi – ii – V (rhythm changes A).
    Template {
        modes: &[Major],
        steps: &[
            (Seventh(0), 4),
            (Seventh(5), 4),
            (Seventh(1), 4),
            (Seventh(4), 4),
        ],
    },
    // ---- Minor ----
    // i–VI–III–VII (epic minor pop).
    Template {
        modes: &[Minor],
        steps: &[(Triad(0), 4), (Triad(5), 4), (Triad(2), 4), (Triad(6), 4)],
    },
    // i–iv–v–i.
    Template {
        modes: &[Minor],
        steps: &[(Triad(0), 4), (Triad(3), 4), (Triad(4), 4), (Triad(0), 4)],
    },
    // Andalusian cadence: i–VII–VI–V(major, harmonic-minor dominant).
    Template {
        modes: &[Minor],
        steps: &[
            (Triad(0), 4),
            (Triad(6), 4),
            (Triad(5), 4),
            (Fixed(7, Quality::Major), 4),
        ],
    },
    // Minor ii–V–i: iiø7 – V7 – i.
    Template {
        modes: &[Minor],
        steps: &[
            (Fixed(2, HalfDiminished7), 4),
            (Fixed(7, Dominant7), 4),
            (Triad(0), 8),
        ],
    },
    // i–VI–III–VII with a harmonic-minor V7 turnaround.
    Template {
        modes: &[Minor],
        steps: &[
            (Triad(0), 4),
            (Triad(5), 4),
            (Fixed(7, Dominant7), 4),
            (Triad(0), 4),
        ],
    },
];

/// Functional category of a diatonic degree, used by the Markov walk.
fn function_of(degree: usize) -> usize {
    // 0 = tonic, 1 = predominant, 2 = dominant.
    match degree {
        0 | 5 => 0, // I, vi
        1 | 3 => 1, // ii, IV
        4 | 6 => 2, // V, vii°
        2 => 0,     // iii ~ tonic prolongation
        _ => 0,
    }
}

/// Transition weights between diatonic degrees, biased toward T→PD→D→T motion.
fn markov_next<R: Rng>(rng: &mut R, current: usize) -> usize {
    let func = function_of(current);
    // Preferred next function.
    let target_func = match func {
        0 => 1, // tonic -> predominant
        1 => 2, // predominant -> dominant
        _ => 0, // dominant -> tonic
    };
    let mut candidates: Vec<usize> = (0..7).collect();
    candidates.shuffle(rng);
    // 75% follow functional flow, else free move (colour).
    if rng.gen_bool(0.75) {
        if let Some(&d) = candidates
            .iter()
            .find(|&&d| function_of(d) == target_func && d != current)
        {
            return d;
        }
    }
    *candidates.iter().find(|&&d| d != current).unwrap_or(&0)
}

/// Generate a chord progression of at least `min_beats` total length.
pub fn generate<R: Rng>(rng: &mut R, key: &Key, min_beats: u32) -> Vec<(Chord, u32)> {
    let use_template = rng.gen_bool(0.6);
    let mut out: Vec<(Chord, u32)> = Vec::new();

    if use_template {
        let valid: Vec<&Template> = TEMPLATES
            .iter()
            .filter(|t| t.modes.contains(&key.mode))
            .collect();
        let tmpl = valid.choose(rng).copied().unwrap_or(&TEMPLATES[0]);
        while total_beats(&out) < min_beats {
            for &(deg, beats) in tmpl.steps {
                let mut chord = deg.resolve(key);
                // Occasionally upgrade a plain triad to its diatonic seventh.
                chord = maybe_add_seventh(rng, key, chord, deg);
                out.push((chord, beats));
            }
        }
    } else {
        // Functional-harmony Markov walk over diatonic degrees.
        let mut degree = 0usize; // start on tonic
        let beats_per_chord = *[2u32, 4, 4].choose(rng).unwrap();
        while total_beats(&out) < min_beats {
            let seventh = rng.gen_bool(0.35);
            let mut chord = if seventh {
                key.diatonic_seventh(degree)
            } else {
                key.diatonic_triad(degree)
            };
            // Occasionally precede a chord by its secondary dominant.
            if rng.gen_bool(0.12) && degree != 0 {
                let sd = Deg::SecondaryDom(degree).resolve(key);
                out.push((sd, beats_per_chord));
            }
            chord = maybe_borrow(rng, key, chord);
            out.push((chord, beats_per_chord));
            degree = markov_next(rng, degree);
        }
    }
    out
}

fn maybe_add_seventh<R: Rng>(rng: &mut R, key: &Key, chord: Chord, deg: Deg) -> Chord {
    if let Deg::Triad(d) = deg {
        if rng.gen_bool(0.25) {
            return key.diatonic_seventh(d);
        }
    }
    let _ = chord;
    chord
}

/// Occasionally swap in a borrowed (modal-interchange) chord that still points
/// at the same tonic — e.g. a Picardy-ish major tonic or a bVII.
fn maybe_borrow<R: Rng>(rng: &mut R, key: &Key, chord: Chord) -> Chord {
    if rng.gen_bool(0.06) {
        match key.mode {
            Mode::Major => Chord::new((key.tonic + 10) % 12, Quality::Major), // bVII
            Mode::Minor => Chord::new(key.tonic, Quality::Major),             // Picardy third
        }
    } else {
        chord
    }
}

fn total_beats(spans: &[(Chord, u32)]) -> u32 {
    spans.iter().map(|(_, b)| *b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn generates_enough_beats() {
        let mut rng = StdRng::seed_from_u64(1);
        for label in 0..24usize {
            let key = Key::from_label(label);
            let prog = generate(&mut rng, &key, 32);
            assert!(total_beats(&prog) >= 32);
            assert!(!prog.is_empty());
        }
    }

    #[test]
    fn template_chords_are_reasonable() {
        let mut rng = StdRng::seed_from_u64(42);
        let key = Key::new(0, Mode::Major);
        // Force a few generations; every chord must have a valid label.
        for _ in 0..50 {
            for (chord, _) in generate(&mut rng, &key, 16) {
                assert!(chord.label() < crate::theory::N_CHORD_CLASSES);
            }
        }
    }
}

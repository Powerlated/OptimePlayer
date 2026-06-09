//! MIDI-note-to-frequency conversion, supporting equal temperament and Pythagorean "pure"
//! tuning relative to a configurable tonic.

/// Selects how MIDI notes are mapped to frequencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TuningSystem {
    /// Standard 12-tone equal temperament (A4 = 440 Hz).
    #[default]
    Equal,
    /// Pythagorean just intonation with the given tonic, where `tonic` is a semitone offset
    /// from A (0 = A, 1 = A#, ... 11 = G#).
    Pure { tonic: i32 },
}

/// Pythagorean tuning ratios within an octave, starting from C.
const PYTHAGOREAN_RATIOS: [f64; 12] = [
    1.0,           // C
    256.0 / 243.0, // C#
    9.0 / 8.0,     // D
    32.0 / 27.0,   // D#
    81.0 / 64.0,   // E
    4.0 / 3.0,     // F
    729.0 / 512.0, // F#
    3.0 / 2.0,     // G
    128.0 / 81.0,  // G#
    27.0 / 16.0,   // A
    16.0 / 9.0,    // A#
    243.0 / 128.0, // B
];

/// Converts a (possibly fractional) MIDI note number to a frequency in Hz under `tuning`.
pub fn midi_note_to_hz(note: f64, tuning: TuningSystem) -> f64 {
    match tuning {
        TuningSystem::Equal => 440.0 * 2f64.powf((note - 69.0) / 12.0),
        TuningSystem::Pure { tonic } => {
            let tonic = tonic as f64;
            let round_error = note - note.round();
            let note = note.round();

            let note_rel_root = note - 69.0 - tonic;
            let octave = (note_rel_root / 12.0).floor();
            // Euclidean modulo to keep the index in 0..12 for negative inputs.
            let note_in_octave = note_rel_root.rem_euclid(12.0) as usize;
            let root_note_hz = 440.0 * 2f64.powf(((tonic + round_error) / 12.0) + octave);

            root_note_hz * PYTHAGOREAN_RATIOS[note_in_octave]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn equal_temperament_reference_pitches() {
        // A4 = 440 Hz, A5 = 880 Hz, A3 = 220 Hz.
        assert!(close(midi_note_to_hz(69.0, TuningSystem::Equal), 440.0));
        assert!(close(midi_note_to_hz(81.0, TuningSystem::Equal), 880.0));
        assert!(close(midi_note_to_hz(57.0, TuningSystem::Equal), 220.0));
        // Middle C.
        assert!(close(
            midi_note_to_hz(60.0, TuningSystem::Equal),
            261.625_565_300_598_6
        ));
    }

    #[test]
    fn pure_tuning_tonic_is_anchored_to_equal() {
        // With tonic A (0), the tonic note A4 still lands on 440 Hz.
        let hz = midi_note_to_hz(69.0, TuningSystem::Pure { tonic: 0 });
        assert!(close(hz, 440.0), "got {hz}");
    }

    #[test]
    fn pure_tuning_uses_just_fifth() {
        // A perfect fifth above the tonic should be exactly 3/2.
        let tonic = midi_note_to_hz(69.0, TuningSystem::Pure { tonic: 0 });
        let fifth = midi_note_to_hz(76.0, TuningSystem::Pure { tonic: 0 });
        assert!(close(fifth / tonic, 3.0 / 2.0), "ratio {}", fifth / tonic);
    }
}

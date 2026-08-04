//! Note-to-frequency conversion for the supported tuning systems.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TuningSystem {
    #[default]
    Equal,
    Pure {
        tonic: i32,
    },
}

const PYTHAGOREAN_RATIOS: [f64; 12] = [
    1.0,
    256.0 / 243.0,
    9.0 / 8.0,
    32.0 / 27.0,
    81.0 / 64.0,
    4.0 / 3.0,
    729.0 / 512.0,
    3.0 / 2.0,
    128.0 / 81.0,
    27.0 / 16.0,
    16.0 / 9.0,
    243.0 / 128.0,
];

pub fn midi_note_to_hz(note: f64, tuning: TuningSystem) -> f64 {
    match tuning {
        TuningSystem::Equal => 440.0 * 2f64.powf((note - 69.0) / 12.0),
        TuningSystem::Pure { tonic } => {
            let tonic = tonic as f64;
            let round_error = note - note.round();
            let note = note.round();

            let note_rel_root = note - 69.0 - tonic;
            let octave = (note_rel_root / 12.0).floor();
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
        assert!(close(midi_note_to_hz(69.0, TuningSystem::Equal), 440.0));
        assert!(close(midi_note_to_hz(81.0, TuningSystem::Equal), 880.0));
        assert!(close(midi_note_to_hz(57.0, TuningSystem::Equal), 220.0));
        assert!(close(
            midi_note_to_hz(60.0, TuningSystem::Equal),
            261.625_565_300_598_6
        ));
    }

    #[test]
    fn pure_tuning_tonic_is_anchored_to_equal() {
        let hz = midi_note_to_hz(69.0, TuningSystem::Pure { tonic: 0 });
        assert!(close(hz, 440.0), "got {hz}");
    }

    #[test]
    fn pure_tuning_uses_just_fifth() {
        let tonic = midi_note_to_hz(69.0, TuningSystem::Pure { tonic: 0 });
        let fifth = midi_note_to_hz(76.0, TuningSystem::Pure { tonic: 0 });
        assert!(close(fifth / tonic, 3.0 / 2.0), "ratio {}", fifth / tonic);
    }
}

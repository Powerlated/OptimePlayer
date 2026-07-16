//! Chord playback: hear the chord you labelled, over the song you labelled it on.
//!
//! Reading a chord symbol tells you what you *wrote*; hearing it against the track tells you whether
//! it is *right*. That is the whole loop this tool exists to close, so the audition uses a real
//! instrument rather than a sine: a sine agrees with everything.
//!
//! The instrument is lifted from the ROM itself — no sample ships with the app. `SynthEvent::
//! NoteStarted` already carries `waveform: Arc<Waveform>` and the voice's pitch, so [`capture`] runs
//! a song headlessly and takes the first sampled note it sees. That works for any device and needs
//! no GBA voicegroup internals; the engine has already done the decoding.

use std::sync::Arc;

use optime_core::synth_controller::messages::TickFeedback;
use optime_core::{
    PerDeviceSettings, SoundData, SynthEvent, VoicePitch, Waveform, WaveformSynthesizer,
};

use super::model::{Chord, Quality};
use crate::audio::ENGINE_SAMPLE_RATE_HZ;

/// Pokémon Emerald song 435 ("Trainers' School") opens on a clean piano — a good neutral voice for
/// auditioning harmony. Falls back to the song being annotated when this id isn't present (another
/// game, or a ROM without it), so chord playback works everywhere rather than only on Emerald.
pub const PREFERRED_SONG: u32 = 435;

/// Voices in the pool: four chord tones plus headroom to let a previous chord ring while the next
/// one strikes, instead of cutting it dead.
const VOICES: usize = 8;

/// Root position starting at middle C — the obvious default. High enough to sit above a bass line,
/// low enough not to shriek over a lead.
const ROOT_BASE_KEY: u8 = 60;

/// Chord level relative to the song, ~−12 dB. Present enough to judge against, quiet enough that it
/// doesn't drown the very thing you're checking it against.
pub const DEFAULT_GAIN: f32 = 0.25;

/// Ticks to run while hunting for the first sampled note before giving up.
const CAPTURE_TICK_LIMIT: usize = 20_000;

/// A voice lifted from a ROM: its decoded waveform and the pitch it sounded at.
#[derive(Clone)]
pub struct Instrument {
    waveform: Arc<Waveform>,
    /// The pitch the captured note played at, and the key it played — together these calibrate
    /// every other key.
    pitch: VoicePitch,
    key: u8,
}

/// Runs `song_id` headlessly and captures the first **sampled** note's voice.
///
/// PSG squares are skipped: they are the console's beeper channels, useless as a reference voice.
/// Returns `None` if the song won't start or has no sampled note.
pub fn capture(data: &dyn SoundData, song_id: u32) -> Option<Instrument> {
    let mut player = data.make_player(song_id)?;
    let config = PerDeviceSettings::neutral();
    let mut feedback = TickFeedback::default();
    let mut events: Vec<SynthEvent> = Vec::new();

    for _ in 0..CAPTURE_TICK_LIMIT {
        events.clear();
        player.tick(&mut feedback, &config, &mut events);
        feedback.ended_voices.clear();
        for ev in events.drain(..) {
            if let SynthEvent::NoteStarted {
                waveform,
                pitch,
                key,
                ..
            } = ev
                && !waveform.is_psg_square
                && !waveform.data.is_empty()
            {
                return Some(Instrument {
                    waveform,
                    pitch,
                    key,
                });
            }
        }
    }
    None
}

impl Instrument {
    /// The pitch that sounds MIDI `note` on this voice.
    ///
    /// The two `VoicePitch` forms need opposite treatment. `Midi` is already key-relative, so the
    /// note is simply substituted. `DataRateHz` (how the GBA speaks) is an absolute data rate for
    /// *one* key, so it has to be scaled by the interval — an octave up is twice the rate.
    fn pitch_for(&self, note: u8) -> VoicePitch {
        match self.pitch {
            VoicePitch::Midi {
                sample_pitch_hz, ..
            } => VoicePitch::Midi {
                note: note as f64,
                sample_pitch_hz,
            },
            VoicePitch::DataRateHz(hz) => {
                let semitones = note as f64 - self.key as f64;
                VoicePitch::DataRateHz(hz * 2.0_f64.powf(semitones / 12.0))
            }
        }
    }
}

impl Quality {
    /// Semitones above the root. Matches `optime_ml::theory::Quality`'s intervals — the label space
    /// and what you hear must describe the same chord, or the audition is checking the wrong thing.
    pub fn intervals(&self) -> &'static [u8] {
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
}

/// The MIDI keys of a chord, root position from middle C.
pub fn chord_keys(chord: &Chord) -> Vec<u8> {
    let root = ROOT_BASE_KEY + (chord.root % 12);
    chord
        .quality
        .intervals()
        .iter()
        .map(|i| root.saturating_add(*i))
        .collect()
}

/// Plays annotated chords through a captured instrument, mixed over the bounce.
pub struct ChordVoicer {
    instrument: Instrument,
    synth: WaveformSynthesizer,
    /// Neutral, not the user's device settings: this is a reference voice, and colouring it with
    /// PSG crunch or a stereo expander would make it lie about the pitch you are checking.
    config: PerDeviceSettings,
    /// What is currently sounding, so an unchanged chord isn't retriggered every frame.
    current: Option<Chord>,
    pub gain: f32,
}

impl ChordVoicer {
    pub fn new(instrument: Instrument) -> ChordVoicer {
        ChordVoicer {
            instrument,
            synth: WaveformSynthesizer::new(ENGINE_SAMPLE_RATE_HZ, VOICES),
            config: PerDeviceSettings::neutral(),
            current: None,
            gain: DEFAULT_GAIN,
        }
    }

    /// Sets the sounding chord, striking it only when it actually changes. `None` lets the current
    /// chord ring out (an annotated rest shouldn't cut the tail mid-decay).
    pub fn set_chord(&mut self, chord: Option<Chord>) {
        // Compare on pitch alone: re-striking because `qualityUncertain` was ticked would be a
        // click for no musical reason.
        let same = match (self.current, chord) {
            (Some(a), Some(b)) => a.root % 12 == b.root % 12 && a.quality == b.quality,
            (None, None) => true,
            _ => false,
        };
        if same {
            return;
        }
        self.current = chord;
        if let Some(c) = chord {
            self.strike(c);
        }
    }

    /// Strikes `chord` now, whether or not it is already sounding — the audition a right-click asks
    /// for, which must sound even when you click the chord that is already playing.
    pub fn strike(&mut self, chord: Chord) {
        self.current = Some(chord);
        for key in chord_keys(&chord) {
            let pitch = self.instrument.pitch_for(key);
            self.synth
                .play(self.instrument.waveform.clone(), pitch, 1.0, &self.config);
        }
    }

    /// Forgets what is sounding, so the next chord strikes even if it matches. Used on a seek: after
    /// jumping, the chord under the playhead should sound again.
    pub fn reset(&mut self) {
        self.current = None;
    }

    /// One stereo frame of chord audio, already at [`DEFAULT_GAIN`].
    #[inline]
    pub fn next_frame(&mut self) -> (f32, f32) {
        self.synth.next_sample(&self.config);
        (self.synth.val_l * self.gain, self.synth.val_r * self.gain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotation::model::Quality;

    fn chord(root: u8, quality: Quality) -> Chord {
        Chord {
            root,
            quality,
            quality_uncertain: false,
        }
    }

    #[test]
    fn chords_voice_in_root_position_from_middle_c() {
        // C major → C4 E4 G4.
        assert_eq!(chord_keys(&chord(0, Quality::Major)), vec![60, 64, 67]);
        // A minor → A4 C5 E5: the root moves up from middle C, it does not wrap below it.
        assert_eq!(chord_keys(&chord(9, Quality::Minor)), vec![69, 72, 76]);
        // Sevenths add a fourth tone.
        assert_eq!(
            chord_keys(&chord(7, Quality::Dominant7)),
            vec![67, 71, 74, 77]
        );
    }

    /// Every quality must voice a distinct pitch set — two qualities that sound identical would make
    /// the audition useless for telling them apart, which is exactly what it is for.
    #[test]
    fn every_quality_sounds_different() {
        let mut seen = Vec::new();
        for q in Quality::ALL {
            let keys = chord_keys(&chord(0, q));
            assert!(
                !seen.contains(&keys),
                "{q:?} voices the same pitches as another quality"
            );
            assert_eq!(keys[0], ROOT_BASE_KEY, "{q:?} must be root position");
            seen.push(keys);
        }
    }

    #[test]
    fn data_rate_pitch_scales_by_the_interval() {
        let instrument = Instrument {
            waveform: Arc::new(Waveform::new(vec![0.0; 4], 440.0, 8000.0, false, 0)),
            pitch: VoicePitch::DataRateHz(1000.0),
            key: 60,
        };
        // An octave up is exactly double the data rate.
        match instrument.pitch_for(72) {
            VoicePitch::DataRateHz(hz) => assert!((hz - 2000.0).abs() < 1e-6, "got {hz}"),
            other => panic!("expected DataRateHz, got {other:?}"),
        }
        // An octave down is half.
        match instrument.pitch_for(48) {
            VoicePitch::DataRateHz(hz) => assert!((hz - 500.0).abs() < 1e-6, "got {hz}"),
            other => panic!("expected DataRateHz, got {other:?}"),
        }
        // The captured key itself is unchanged.
        match instrument.pitch_for(60) {
            VoicePitch::DataRateHz(hz) => assert!((hz - 1000.0).abs() < 1e-6, "got {hz}"),
            other => panic!("expected DataRateHz, got {other:?}"),
        }
    }

    /// The real thing: Emerald song 435 must actually yield a sampled voice, and repitching it must
    /// produce a distinct rate per chord tone. Pins the one assumption this feature rests on — that
    /// `NoteStarted` carries a usable waveform — against the real ROM rather than a mock.
    #[test]
    fn captures_a_real_voice_from_emerald() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../demos/pokemon-emerald.gbaaudio");
        let bytes = std::fs::read(path).expect("demo file should exist");
        let archives = optime_core::load_all(&bytes);
        let data = archives.first().expect("an archive");
        let instrument = capture(&**data, PREFERRED_SONG).expect("song 435 should yield a voice");
        assert!(
            !instrument.waveform.data.is_empty(),
            "voice must have sample data"
        );
        assert!(
            !instrument.waveform.is_psg_square,
            "must not be a PSG beeper"
        );

        // Each tone of a C major triad must land on its own rate — if they collapsed, the audition
        // would play a unison and agree with anything.
        let rates: Vec<String> = chord_keys(&chord(0, Quality::Major))
            .iter()
            .map(|k| format!("{:?}", instrument.pitch_for(*k)))
            .collect();
        assert_eq!(
            rates.iter().collect::<std::collections::HashSet<_>>().len(),
            3,
            "three tones, three pitches: {rates:?}"
        );
    }

    #[test]
    fn midi_pitch_substitutes_the_note() {
        let instrument = Instrument {
            waveform: Arc::new(Waveform::new(vec![0.0; 4], 440.0, 8000.0, false, 0)),
            pitch: VoicePitch::Midi {
                note: 60.0,
                sample_pitch_hz: 440.0,
            },
            key: 60,
        };
        match instrument.pitch_for(67) {
            VoicePitch::Midi {
                note,
                sample_pitch_hz,
            } => {
                assert_eq!(note, 67.0);
                // The sample's own recorded pitch is a property of the data, not of the key.
                assert_eq!(sample_pitch_hz, 440.0);
            }
            other => panic!("expected Midi, got {other:?}"),
        }
    }
}

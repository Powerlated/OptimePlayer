use std::sync::Arc;

use optime_core::biquad_filter::BiquadFilter;
use optime_core::{PerDeviceSettings, Sample, VoicePitch, Waveform, WaveformSynthesizer};

use super::model::{Chord, Quality};
use crate::audio::ENGINE_SAMPLE_RATE_HZ;

const VOICES: usize = 12;

const ROOT_BASE_KEY: u8 = 60;

const AUDITION_SECONDS: f64 = 1.5;

pub const DEFAULT_GAIN: f32 = 0.25;

const PIANO_FREQ: f64 = 261.6256;
const PIANO_ADSR: [u8; 4] = [255, 250, 0, 221];
const PIANO_TICK_SECONDS: f64 = 1.0 / 59.7275;

#[derive(Clone)]
pub struct Instrument {
    waveform: Arc<Waveform>,
    pitch: VoicePitch,
    key: u8,
    held: Vec<Sample>,
    release: Vec<Sample>,
    tick_seconds: f64,
}

impl Instrument {
    fn held_level(&self, pos: f64) -> Sample {
        interpolate(&self.held, pos).unwrap_or_else(|| self.held.last().copied().unwrap_or(0.0))
    }

    fn release_level(&self, pos: f64) -> Sample {
        interpolate(&self.release, pos).unwrap_or(0.0)
    }

    fn release_ticks(&self) -> usize {
        self.release.len()
    }

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

fn interpolate(curve: &[Sample], pos: f64) -> Option<Sample> {
    if pos < 0.0 {
        return curve.first().copied();
    }
    let i = pos.floor() as usize;
    if i + 1 >= curve.len() {
        return None;
    }
    let (a, b) = (curve[i], curve[i + 1]);
    Some(a + (b - a) * (pos - i as f64) as Sample)
}

impl Quality {
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

fn voiced_keys(chord: &Chord, inversion: u8) -> Vec<u8> {
    let root = ROOT_BASE_KEY + (chord.root % 12);
    let intervals = chord.quality.intervals();
    let lift = (inversion as usize).min(intervals.len().saturating_sub(1));
    intervals
        .iter()
        .enumerate()
        .map(|(i, &iv)| root.saturating_add(iv + if i < lift { 12 } else { 0 }))
        .collect()
}

pub fn embedded_piano() -> Instrument {
    const PIANO_WAV: &[u8] = include_bytes!("../chord_data/piano_c4.wav");
    let (sample_rate, data) = find_data_chunk(PIANO_WAV).unwrap_or((13_379.0, &[][..]));
    let samples: Vec<f32> = data
        .iter()
        .map(|&b| ((b as i32 - 128) as f32) / 128.0)
        .collect();
    let waveform = Waveform::new(samples, PIANO_FREQ, sample_rate, true, 0);
    let (held, release) = piano_envelope(PIANO_ADSR);
    Instrument {
        waveform: Arc::new(waveform),
        pitch: VoicePitch::Midi {
            note: 60.0,
            sample_pitch_hz: PIANO_FREQ,
        },
        key: 60,
        held,
        release,
        tick_seconds: PIANO_TICK_SECONDS,
    }
}

fn find_data_chunk(bytes: &[u8]) -> Option<(f64, &[u8])> {
    if bytes.len() < 12 || &bytes[0..4] != "RIFF".as_bytes() || &bytes[8..12] != "WAVE".as_bytes() {
        return None;
    }
    let mut sample_rate = 13_379.0;
    let mut data: &[u8] = &[];
    let mut o = 12usize;
    while o + 8 <= bytes.len() {
        let id = &bytes[o..o + 4];
        let sz =
            u32::from_le_bytes([bytes[o + 4], bytes[o + 5], bytes[o + 6], bytes[o + 7]]) as usize;
        let body = o + 8;
        if id == "fmt ".as_bytes() && body + 12 <= bytes.len() {
            sample_rate = u32::from_le_bytes([
                bytes[body + 8],
                bytes[body + 9],
                bytes[body + 10],
                bytes[body + 11],
            ]) as f64;
        } else if id == "data".as_bytes() && body + sz <= bytes.len() {
            data = &bytes[body..body + sz];
        }
        o = body + sz + (sz & 1);
    }
    Some((sample_rate, data))
}

fn piano_envelope(adsr: [u8; 4]) -> (Vec<Sample>, Vec<Sample>) {
    let [attack, decay, sustain, release] = adsr;
    let peak: u32 = 0xFF;

    let mut held: Vec<Sample> = Vec::new();
    let mut env: u32 = attack as u32;
    if env >= peak {
        env = peak;
        held.push(1.0);
    } else {
        held.push(env as Sample / peak as Sample);
        while env < peak {
            env += attack as u32;
            if env >= peak {
                env = peak;
            }
            held.push(env as Sample / peak as Sample);
        }
    }
    let sustain = sustain as u32;
    loop {
        env = (env * decay as u32) >> 8;
        held.push(env as Sample / peak as Sample);
        if env <= sustain {
            break;
        }
    }

    let mut rel: Vec<Sample> = Vec::new();
    let mut env = peak;
    loop {
        rel.push(env as Sample / peak as Sample);
        env = (env * release as u32) >> 8;
        if env == 0 {
            rel.push(0.0);
            break;
        }
    }

    (held, rel)
}

enum Phase {
    Held,
    Released { from: Sample },
}

struct Ringing {
    index: usize,
    pos: f64,
    phase: Phase,
}

pub struct ChordVoicer {
    instrument: Instrument,
    synth: WaveformSynthesizer,
    config: PerDeviceSettings,
    current: Option<(u32, Chord)>,
    ringing: Vec<Ringing>,
    audition_frames: Option<usize>,
    env_step: f64,
    pub gain: f32,
    inversion: u8,
    shelf_l: BiquadFilter,
    shelf_r: BiquadFilter,
}

impl ChordVoicer {
    pub fn new(instrument: Instrument) -> ChordVoicer {
        let env_step = 1.0 / (instrument.tick_seconds * ENGINE_SAMPLE_RATE_HZ);
        let hs = PerDeviceSettings::enhanced_gba().shelf;
        let make_shelf = || {
            BiquadFilter::high_shelf(
                hs.order,
                ENGINE_SAMPLE_RATE_HZ,
                hs.cutoff_hz,
                hs.q,
                hs.gain_db,
            )
        };
        ChordVoicer {
            instrument,
            synth: WaveformSynthesizer::new(ENGINE_SAMPLE_RATE_HZ, VOICES),
            config: PerDeviceSettings::neutral(),
            current: None,
            ringing: Vec::new(),
            audition_frames: None,
            env_step,
            gain: DEFAULT_GAIN,
            inversion: 0,
            shelf_l: make_shelf(),
            shelf_r: make_shelf(),
        }
    }

    pub fn set_gain(&mut self, gain: f32) {
        self.gain = gain;
    }

    pub fn set_inversion(&mut self, inversion: u8) {
        if self.inversion == inversion {
            return;
        }
        self.inversion = inversion;
        if let Some((_, c)) = self.current {
            self.release_all();
            self.play_tones(c);
        }
    }

    pub fn set_chord(&mut self, section: Option<(u32, Chord)>) {
        let same = match (self.current, section) {
            (Some((sa, a)), Some((sb, b))) => {
                sa == sb && a.root % 12 == b.root % 12 && a.quality == b.quality
            }
            (None, None) => true,
            _ => false,
        };
        if same {
            self.audition_frames = None;
            return;
        }
        self.current = section;
        self.audition_frames = None;
        self.release_all();
        if let Some((_, c)) = section {
            self.play_tones(c);
        }
    }

    pub fn strike(&mut self, section: (u32, Chord)) {
        self.current = Some(section);
        self.audition_frames = Some((AUDITION_SECONDS * ENGINE_SAMPLE_RATE_HZ) as usize);
        self.release_all();
        self.play_tones(section.1);
    }

    fn play_tones(&mut self, chord: Chord) {
        let mut keys = voiced_keys(&chord, self.inversion);
        keys.push(ROOT_BASE_KEY + (chord.root % 12) - 12);
        for key in keys {
            let pitch = self.instrument.pitch_for(key);
            let index = self.synth.play(
                self.instrument.waveform.clone(),
                pitch,
                self.instrument.held_level(0.0),
                &self.config,
            );
            self.ringing.retain(|r| r.index != index);
            self.ringing.push(Ringing {
                index,
                pos: 0.0,
                phase: Phase::Held,
            });
        }
    }

    fn release_all(&mut self) {
        for r in &mut self.ringing {
            if let Phase::Held = r.phase {
                let from = self.instrument.held_level(r.pos);
                r.phase = Phase::Released { from };
                r.pos = 0.0;
            }
        }
    }

    pub fn reset(&mut self) {
        self.current = None;
    }

    pub fn stop_following(&mut self) {
        if self.audition_frames.is_some() {
            return;
        }
        self.current = None;
        self.release_all();
    }

    #[inline]
    pub fn next_frame(&mut self) -> (f32, f32) {
        self.advance_envelopes();
        self.synth.next_sample(&self.config);
        let l = self.shelf_l.transform(self.synth.val_l * self.gain);
        let r = self.shelf_r.transform(self.synth.val_r * self.gain);
        (l, r)
    }

    fn advance_envelopes(&mut self) {
        if let Some(left) = &mut self.audition_frames {
            *left = left.saturating_sub(1);
            if *left == 0 {
                self.audition_frames = None;
                self.current = None;
                self.release_all();
            }
        }
        let mut i = 0;
        while i < self.ringing.len() {
            let r = &mut self.ringing[i];
            r.pos += self.env_step;
            let (index, pos) = (r.index, r.pos);
            let level = match r.phase {
                Phase::Held => self.instrument.held_level(pos),
                Phase::Released { from } => {
                    if pos >= self.instrument.release_ticks() as f64 {
                        self.ringing.swap_remove(i);
                        self.synth.cut_instrument(index);
                        continue;
                    }
                    from * self.instrument.release_level(pos)
                }
            };
            self.synth.instr_mut(index).volume = level;
            i += 1;
        }
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

    fn test_instrument() -> Instrument {
        Instrument {
            waveform: Arc::new(Waveform::new(vec![0.5; 64], 440.0, 8000.0, true, 0)),
            pitch: VoicePitch::DataRateHz(8000.0),
            key: 60,
            held: vec![0.0, 1.0, 0.5],
            release: vec![1.0, 0.75, 0.5, 0.25, 0.0],
            tick_seconds: 0.01,
        }
    }

    #[test]
    fn chords_voice_in_root_position_from_middle_c() {
        assert_eq!(voiced_keys(&chord(0, Quality::Major), 0), vec![60, 64, 67]);
        assert_eq!(voiced_keys(&chord(9, Quality::Minor), 0), vec![69, 72, 76]);
        assert_eq!(
            voiced_keys(&chord(7, Quality::Dominant7), 0),
            vec![67, 71, 74, 77]
        );
    }

    #[test]
    fn inversions_lift_the_lowest_tones() {
        assert_eq!(voiced_keys(&chord(0, Quality::Major), 1), vec![72, 64, 67]);
        assert_eq!(voiced_keys(&chord(0, Quality::Major), 2), vec![72, 76, 67]);
        assert_eq!(voiced_keys(&chord(0, Quality::Major), 3), vec![72, 76, 67]);
        assert_eq!(
            voiced_keys(&chord(0, Quality::Dominant7), 1),
            vec![72, 64, 67, 70]
        );
        assert_eq!(
            voiced_keys(&chord(0, Quality::Dominant7), 3),
            vec![72, 76, 79, 70]
        );
        assert_eq!(
            voiced_keys(&chord(0, Quality::Dominant7), 4),
            vec![72, 76, 79, 70]
        );
    }

    #[test]
    fn every_quality_sounds_different() {
        let mut seen = Vec::new();
        for q in Quality::ALL {
            let keys = voiced_keys(&chord(0, q), 0);
            assert!(
                !seen.contains(&keys),
                "{q:?} voices the same pitches as another quality"
            );
            assert_eq!(keys[0], ROOT_BASE_KEY, "{q:?} must be root position");
            seen.push(keys);
        }
    }

    #[test]
    fn a_held_chord_sustains_indefinitely() {
        let mut v = ChordVoicer::new(test_instrument());
        v.set_chord(Some((0, chord(0, Quality::Major))));

        let mut loudest: f32 = 0.0;
        for _ in 0..(2.0 * ENGINE_SAMPLE_RATE_HZ) as usize {
            let (l, r) = v.next_frame();
            loudest = loudest.max(l.abs()).max(r.abs());
        }
        assert!(loudest > 0.0, "a held chord must still sound after 2 s");
        assert_eq!(v.ringing.len(), 4, "four tones (triad + bass) still held");
        for r in &v.ringing {
            assert!(
                matches!(r.phase, Phase::Held),
                "an unchanged section must not release"
            );
        }
    }

    #[test]
    fn a_released_chord_dies_and_frees_its_voice() {
        let mut v = ChordVoicer::new(test_instrument());
        v.set_chord(Some((0, chord(0, Quality::Major))));
        for _ in 0..(0.5 * ENGINE_SAMPLE_RATE_HZ) as usize {
            v.next_frame();
        }
        v.set_chord(None);
        assert!(
            v.ringing
                .iter()
                .all(|r| matches!(r.phase, Phase::Released { .. })),
            "N.C. must release the chord"
        );

        for _ in 0..(0.2 * ENGINE_SAMPLE_RATE_HZ) as usize {
            v.next_frame();
        }
        assert!(v.ringing.is_empty(), "released voices must be freed");
        for _ in 0..1000 {
            let (l, r) = v.next_frame();
            assert!(
                l.abs() < 1e-9 && r.abs() < 1e-9,
                "a looping voice was left ringing ({l}, {r})"
            );
        }
    }

    #[test]
    fn sections_strike_once_each() {
        let mut v = ChordVoicer::new(test_instrument());
        let c = chord(0, Quality::Major);

        v.set_chord(Some((0, c)));
        assert_eq!(
            v.ringing.len(),
            4,
            "C major strikes four tones (triad + bass)"
        );
        for _ in 0..10 {
            v.set_chord(Some((0, c)));
        }
        assert_eq!(v.ringing.len(), 4, "one section, one strike");
        assert!(
            v.ringing.iter().all(|r| matches!(r.phase, Phase::Held)),
            "the same section must keep holding"
        );

        v.set_chord(Some((96, c)));
        assert_eq!(v.ringing.len(), 8, "a new section restrikes the chord");
        assert_eq!(
            v.ringing
                .iter()
                .filter(|r| matches!(r.phase, Phase::Held))
                .count(),
            4,
            "exactly the new tones are held; the old ones are releasing"
        );
    }

    #[test]
    fn release_starts_from_the_current_level() {
        let mut v = ChordVoicer::new(test_instrument());
        v.set_chord(Some((0, chord(0, Quality::Major))));
        for _ in 0..(0.5 * ENGINE_SAMPLE_RATE_HZ) as usize {
            v.next_frame();
        }
        v.set_chord(None);
        for r in &v.ringing {
            match r.phase {
                Phase::Released { from } => assert!(
                    (from - 0.5).abs() < 1e-6,
                    "released from {from}, expected the 0.5 sustain"
                ),
                Phase::Held => panic!("should be releasing"),
            }
        }
    }

    #[test]
    fn stopping_releases_the_section_chord() {
        let mut v = ChordVoicer::new(test_instrument());
        v.set_chord(Some((0, chord(0, Quality::Major))));
        for _ in 0..(0.1 * ENGINE_SAMPLE_RATE_HZ) as usize {
            v.next_frame();
        }
        v.stop_following();
        for _ in 0..(0.2 * ENGINE_SAMPLE_RATE_HZ) as usize {
            v.next_frame();
        }
        assert!(v.ringing.is_empty(), "a paused chord must not drone");
        v.set_chord(Some((0, chord(0, Quality::Major))));
        assert_eq!(v.ringing.len(), 4);
    }

    #[test]
    fn a_hand_struck_chord_outlives_a_stopped_playhead_then_expires() {
        let mut v = ChordVoicer::new(test_instrument());
        v.strike((0, chord(0, Quality::Major)));
        for _ in 0..(0.5 * ENGINE_SAMPLE_RATE_HZ) as usize {
            v.stop_following();
            v.next_frame();
        }
        assert_eq!(v.ringing.len(), 4, "the audition must still be ringing");

        for _ in 0..((AUDITION_SECONDS + 0.5) * ENGINE_SAMPLE_RATE_HZ) as usize {
            v.stop_following();
            v.next_frame();
        }
        assert!(v.ringing.is_empty(), "a hand-struck chord must end itself");
    }

    #[test]
    fn the_playhead_adopts_a_hand_struck_chord() {
        let mut v = ChordVoicer::new(test_instrument());
        v.strike((0, chord(0, Quality::Major)));
        v.next_frame();
        v.set_chord(Some((0, chord(0, Quality::Major))));
        assert_eq!(
            v.ringing.len(),
            4,
            "must not restrike what is already sounding"
        );
        assert!(v.audition_frames.is_none(), "the section owns it now");
        for _ in 0..((AUDITION_SECONDS + 0.5) * ENGINE_SAMPLE_RATE_HZ) as usize {
            v.set_chord(Some((0, chord(0, Quality::Major))));
            v.next_frame();
        }
        assert_eq!(v.ringing.len(), 4, "a section chord holds for the section");
    }

    #[test]
    fn data_rate_pitch_scales_by_the_interval() {
        let instrument = test_instrument();
        match instrument.pitch_for(72) {
            VoicePitch::DataRateHz(hz) => assert!((hz - 16000.0).abs() < 1e-6, "got {hz}"),
            other => panic!("expected DataRateHz, got {other:?}"),
        }
        match instrument.pitch_for(48) {
            VoicePitch::DataRateHz(hz) => assert!((hz - 4000.0).abs() < 1e-6, "got {hz}"),
            other => panic!("expected DataRateHz, got {other:?}"),
        }
        match instrument.pitch_for(60) {
            VoicePitch::DataRateHz(hz) => assert!((hz - 8000.0).abs() < 1e-6, "got {hz}"),
            other => panic!("expected DataRateHz, got {other:?}"),
        }
    }

    #[test]
    fn midi_pitch_substitutes_the_note() {
        let instrument = Instrument {
            pitch: VoicePitch::Midi {
                note: 60.0,
                sample_pitch_hz: 440.0,
            },
            ..test_instrument()
        };
        match instrument.pitch_for(67) {
            VoicePitch::Midi {
                note,
                sample_pitch_hz,
            } => {
                assert_eq!(note, 67.0);
                assert_eq!(sample_pitch_hz, 440.0);
            }
            other => panic!("expected Midi, got {other:?}"),
        }
    }

    #[test]
    fn a_struck_chord_contains_its_three_pitches() {
        let mut v = ChordVoicer::new(embedded_piano());
        v.gain = 1.0;
        v.strike((0, chord(0, Quality::Major)));
        let n = (0.4 * ENGINE_SAMPLE_RATE_HZ) as usize;
        let buf: Vec<f64> = (0..n)
            .map(|_| {
                let (l, r) = v.next_frame();
                (l as f64 + r as f64) * 0.5
            })
            .collect();
        let mag = |f: f64| -> f64 {
            let (mut re, mut im) = (0.0, 0.0);
            for (i, x) in buf.iter().enumerate() {
                let t = std::f64::consts::TAU * f * i as f64 / ENGINE_SAMPLE_RATE_HZ;
                re += x * t.cos();
                im += x * t.sin();
            }
            (re * re + im * im).sqrt() / n as f64
        };
        let (c4, e4, g4) = (mag(261.63), mag(329.63), mag(392.00));
        let d4 = mag(293.66);
        for (name, m) in [("C4", c4), ("E4", e4), ("G4", g4)] {
            assert!(
                m > d4 * 4.0,
                "{name} ({m:.5}) must stand clear of the unplayed D4 ({d4:.5})"
            );
        }
    }
}

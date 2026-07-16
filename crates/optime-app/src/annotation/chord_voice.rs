//! Chord playback: hear the chord you labelled, over the song you labelled it on.
//!
//! Reading a chord symbol tells you what you *wrote*; hearing it against the track tells you whether
//! it is *right*. That is the whole loop this tool exists to close, so the audition uses a real
//! instrument rather than a sine: a sine agrees with everything.
//!
//! Everything about the instrument is lifted from the ROM — no sample and no envelope ships with the
//! app — and all of it comes off the standard `SynthEvent` stream, so this module knows nothing about
//! any console's voice format:
//!
//! - **The waveform and pitch**: `NoteStarted` carries `waveform: Arc<Waveform>` and the voice's
//!   pitch, which together calibrate every other key.
//! - **The ADSR**: devices re-evaluate envelopes every tick and report the result as `VoiceVolume`,
//!   so following one voice records its curve. `NoteReleased` says where the game let go, which is
//!   what splits that curve into the [`Instrument::held`] part (attack → decay → sustain) and the
//!   [`Instrument::release`] tail. A chord then holds its sustain for as long as the annotated
//!   section lasts and releases when the section changes — the shape of a real held chord.
//!
//! The envelope is load-bearing, not decoration: these ROM waveforms **loop** (Emerald's voices loop
//! mid-sample), and nothing here sends a note-off to the synth — a chord audition is not a sequence.
//! The captured release is the only thing that ever ends a note.

use std::sync::Arc;

use optime_core::synth_controller::messages::{TickFeedback, VoiceId};
use optime_core::{
    PerDeviceSettings, Sample, SoundData, SynthEvent, VoicePitch, Waveform, WaveformSynthesizer,
};

use super::model::{Chord, Quality};
use crate::audio::ENGINE_SAMPLE_RATE_HZ;

/// Pokémon Emerald song 435 ("Trainers' School") is a quiet, sparse track whose voices survey well.
/// Falls back to the song being annotated when this id isn't present (another game, or a ROM without
/// it), so chord playback works everywhere rather than only on Emerald.
pub const PREFERRED_SONG: u32 = 435;

/// Voices in the pool: two full four-note chords, so a chord's release tail always has somewhere to
/// ring while the next one sustains, instead of being stolen and cut.
const VOICES: usize = 12;

/// Root position starting at middle C — the obvious default. High enough to sit above a bass line,
/// low enough not to shriek over a lead.
const ROOT_BASE_KEY: u8 = 60;

/// How long a hand-struck chord (a right-click audition) holds its sustain before releasing itself.
/// The playhead can't end it — it isn't rolling, that is often the whole point — so it needs its own
/// clock. Roughly how long you'd hold a key to check a chord.
const AUDITION_SECONDS: f64 = 1.5;

/// Chord level relative to the song, ~−12 dB. Present enough to judge against, quiet enough that it
/// doesn't drown the very thing you're checking it against.
pub const DEFAULT_GAIN: f32 = 0.25;

/// How much of the song to survey for candidate voices. Long enough for every instrument to have
/// played (a game track states its material in the first phrase), short enough to stay instant.
const SURVEY_SECONDS: f64 = 20.0;

/// Longest single note to record. Guards against adopting a drone as the reference envelope.
const MAX_ENVELOPE_SECONDS: f64 = 8.0;

/// Release to synthesise for a voice the game cut without ever releasing. Stepping a looping
/// sample's gain straight to zero clicks.
const TAIL_SECONDS: f64 = 0.05;

/// A voice lifted from a ROM: its waveform, the pitch it sounded at, and the game's own ADSR split
/// at the note-off.
#[derive(Clone)]
pub struct Instrument {
    waveform: Arc<Waveform>,
    /// The pitch the captured note played at, and the key it played — together these calibrate
    /// every other key.
    pitch: VoicePitch,
    key: u8,
    /// The envelope while the note was **held**: attack, decay, and on to the sustain level, one
    /// entry per device tick, peak-normalised. Its last entry is the sustain level, which a chord
    /// holds for as long as its section lasts. Peak-normalised because the level we captured at is
    /// an accident of that note's velocity and track volume; the audition's level is
    /// [`DEFAULT_GAIN`]'s business.
    held: Vec<Sample>,
    /// The **release** tail the game played after the note-off, stored as a multiplier falling from
    /// 1.0 to 0. Relative rather than absolute so it can be applied from whatever level a chord is
    /// sitting at when it is released. Always ends at exactly 0 — this is what stops the voice.
    release: Vec<Sample>,
    /// Seconds per envelope entry (the device's tick period).
    tick_seconds: f64,
}

/// A note being followed during the survey, before it is judged against the others.
struct Candidate {
    track: usize,
    key: u8,
    waveform: Arc<Waveform>,
    pitch: VoicePitch,
    held: Vec<Sample>,
    release: Vec<Sample>,
    released: bool,
}

/// Surveys `song_id` headlessly and lifts the most **sustaining** sampled voice it plays.
///
/// Longest-ringing wins because a chord audition wants a voice that holds: the first note a song
/// happens to play is usually its bass, and three pitched-up bass thumps do not read as a chord.
/// PSG squares are skipped — they are the console's beeper channels, useless as a reference.
///
/// Returns `None` if the song won't start or never plays an audible sampled note.
pub fn capture(data: &dyn SoundData, song_id: u32) -> Option<Instrument> {
    let mut player = data.make_player(song_id)?;
    let config = PerDeviceSettings::neutral();
    let mut feedback = TickFeedback::default();
    let mut events: Vec<SynthEvent> = Vec::new();
    let tick_seconds = 1.0 / player.tick_rate();
    let max_ticks = (MAX_ENVELOPE_SECONDS / tick_seconds).ceil() as usize;
    let survey_ticks = (SURVEY_SECONDS / tick_seconds).ceil() as usize;

    // Notes in flight, keyed by voice. A Vec, not a HashMap: at most a handful sound at once, and
    // iteration order has to be deterministic — it decides ties between equally sustaining voices.
    let mut live: Vec<(VoiceId, Candidate)> = Vec::new();
    let mut done: Vec<Candidate> = Vec::new();

    for _ in 0..survey_ticks {
        events.clear();
        player.tick(&mut feedback, &config, &mut events);
        // Nothing is reported as ended: no synth is rendering these voices, so the device must go
        // on believing its notes are still sounding, or it would cut the envelopes short.
        feedback.ended_voices.clear();
        for ev in events.drain(..) {
            match ev {
                SynthEvent::NoteStarted {
                    track,
                    voice,
                    waveform,
                    pitch,
                    key,
                    volume,
                    ..
                } => {
                    if waveform.is_psg_square || waveform.data.is_empty() {
                        continue;
                    }
                    live.push((
                        voice,
                        Candidate {
                            track,
                            key,
                            waveform,
                            pitch,
                            held: vec![volume as Sample],
                            release: Vec::new(),
                            released: false,
                        },
                    ));
                }
                SynthEvent::VoiceVolume { voice, volume, .. } => {
                    if let Some((_, c)) = live.iter_mut().find(|(v, _)| *v == voice) {
                        if c.released {
                            c.release.push(volume as Sample);
                        } else {
                            c.held.push(volume as Sample);
                        }
                    }
                }
                // Where the game let go: everything after this is release tail. `NoteReleased`
                // addresses a key on a track rather than a voice, which is all we need — we know
                // both for every note we are following.
                SynthEvent::NoteReleased { track, key } => {
                    for (_, c) in live.iter_mut() {
                        if c.track == track && c.key == key {
                            c.released = true;
                        }
                    }
                }
                SynthEvent::VoiceStopped { voice, .. } => {
                    if let Some(i) = live.iter().position(|(v, _)| *v == voice) {
                        done.push(live.remove(i).1);
                    }
                }
                _ => {}
            }
        }
        // Retire anything that has outstayed the cap rather than letting a drone become the
        // reference envelope.
        let mut i = 0;
        while i < live.len() {
            if live[i].1.held.len() + live[i].1.release.len() >= max_ticks {
                done.push(live.remove(i).1);
            } else {
                i += 1;
            }
        }
    }
    // Notes still sounding when the survey ended are candidates too — a pad may simply be long.
    done.extend(live.into_iter().map(|(_, c)| c));

    // Longest total envelope wins; ties break on track then key so the pick is reproducible.
    done.sort_by_key(|c| {
        (
            std::cmp::Reverse(c.held.len() + c.release.len()),
            c.track,
            c.key,
        )
    });
    done.into_iter().find_map(|c| c.finish(tick_seconds))
}

impl Candidate {
    /// Turns a surveyed note into a playable [`Instrument`], or `None` if it never made a sound.
    fn finish(self, tick_seconds: f64) -> Option<Instrument> {
        let Candidate {
            key,
            waveform,
            pitch,
            mut held,
            release,
            ..
        } = self;
        let peak = held.iter().cloned().fold(0.0, Sample::max);
        if peak <= 0.0 {
            return None;
        }
        for v in &mut held {
            *v /= peak;
        }

        // The release is stored relative to the level the note was released *at*, so it can be
        // applied from wherever a held chord happens to be sitting.
        let from = release.first().copied().unwrap_or(0.0);
        let mut release: Vec<Sample> = if from > 0.0 {
            release.iter().map(|v| v / from).collect()
        } else {
            // The game cut this note instead of releasing it (or released it already silent): it
            // has no tail to lift, so fall back to a short linear one.
            Vec::new()
        };
        if release.last().copied().unwrap_or(1.0) > 0.0 {
            let start = release.last().copied().unwrap_or(1.0);
            let steps = (TAIL_SECONDS / tick_seconds).ceil().max(1.0) as usize;
            for i in 1..=steps {
                release.push(start * (1.0 - i as Sample / steps as Sample).max(0.0));
            }
        }
        Some(Instrument {
            waveform,
            pitch,
            key,
            held,
            release,
            tick_seconds,
        })
    }
}

impl Instrument {
    /// The held level `pos` entries in, interpolated between ticks and **clamped to the sustain
    /// level** past the end of the curve: attack and decay play out, then the chord sits on sustain
    /// for as long as the section holds it.
    fn held_level(&self, pos: f64) -> Sample {
        interpolate(&self.held, pos).unwrap_or_else(|| self.held.last().copied().unwrap_or(0.0))
    }

    /// The release multiplier `pos` entries in. Past the end = 0, which is how a voice learns it is
    /// finished.
    fn release_level(&self, pos: f64) -> Sample {
        interpolate(&self.release, pos).unwrap_or(0.0)
    }

    /// Entries in the release curve — how long a released chord takes to die.
    fn release_ticks(&self) -> usize {
        self.release.len()
    }

    /// Seconds of attack+decay before the sustain level is reached.
    #[cfg(test)]
    fn held_seconds(&self) -> f64 {
        self.held.len() as f64 * self.tick_seconds
    }

    /// The level a held chord settles on.
    #[cfg(test)]
    fn sustain_level(&self) -> Sample {
        self.held.last().copied().unwrap_or(0.0)
    }

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

/// Linearly interpolates `curve` at `pos` entries in. `None` once `pos` runs past the last entry —
/// the callers disagree about what that means, so neither gets a default here.
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

/// Where a sounding chord tone is in the captured ADSR.
enum Phase {
    /// Playing the held curve: attack, decay, then sitting on the sustain level indefinitely.
    Held,
    /// Playing the release tail, scaled by the level the tone was released at.
    Released { from: Sample },
}

/// A chord tone still sounding: which pool voice it took, how far it is (in envelope entries, so a
/// fraction of a device tick) through its current phase, and which phase that is.
struct Ringing {
    index: usize,
    pos: f64,
    phase: Phase,
}

/// Plays annotated chords through a captured instrument, mixed over the bounce.
pub struct ChordVoicer {
    instrument: Instrument,
    synth: WaveformSynthesizer,
    /// Neutral, not the user's device settings: this is a reference voice, and colouring it with
    /// PSG crunch or a stereo expander would make it lie about the pitch you are checking.
    config: PerDeviceSettings,
    /// The annotated span currently sounding, keyed by its start step, so a chord strikes once per
    /// section instead of every frame — and a repeat of the same chord in the next section still
    /// strikes, because that is a new section.
    current: Option<(u32, Chord)>,
    /// Tones still playing out their envelope, sustaining or releasing.
    ringing: Vec<Ringing>,
    /// Frames left before a hand-struck chord releases itself; `None` when the sounding chord
    /// belongs to the playhead, which holds it for as long as the section lasts.
    audition_frames: Option<usize>,
    /// Envelope entries advanced per output frame.
    env_step: f64,
    pub gain: f32,
}

impl ChordVoicer {
    pub fn new(instrument: Instrument) -> ChordVoicer {
        let env_step = 1.0 / (instrument.tick_seconds * ENGINE_SAMPLE_RATE_HZ);
        ChordVoicer {
            instrument,
            synth: WaveformSynthesizer::new(ENGINE_SAMPLE_RATE_HZ, VOICES),
            config: PerDeviceSettings::neutral(),
            current: None,
            ringing: Vec::new(),
            audition_frames: None,
            env_step,
            gain: DEFAULT_GAIN,
        }
    }

    /// Sets the sounding section — `(span start step, chord)` — striking it only when the section
    /// under the playhead actually changes, and releasing the previous chord as it does.
    ///
    /// `None` means N.C. or unlabelled: the chord releases and nothing replaces it. An annotated
    /// rest is a statement that no chord is sounding, so it must be audible as one.
    pub fn set_chord(&mut self, section: Option<(u32, Chord)>) {
        // Compare on the span and the chord's pitch: re-striking because `qualityUncertain` was
        // ticked would be a click for no musical reason.
        let same = match (self.current, section) {
            (Some((sa, a)), Some((sb, b))) => {
                sa == sb && a.root % 12 == b.root % 12 && a.quality == b.quality
            }
            (None, None) => true,
            _ => false,
        };
        if same {
            // The playhead has caught up with a chord that was struck by hand: it owns it now, so
            // the chord holds for the section rather than expiring on the audition clock.
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

    /// Strikes `section`'s chord now, whether or not it is already sounding — the audition a
    /// right-click asks for, which must sound even when you click the chord already playing, and
    /// must sound when the transport is stopped.
    ///
    /// It rings on the audition clock rather than indefinitely: nothing else would end it, since a
    /// stopped playhead never leaves the section. If the playhead is rolling and reaches this same
    /// section, [`Self::set_chord`] promotes it to a normal section hold.
    pub fn strike(&mut self, section: (u32, Chord)) {
        self.current = Some(section);
        self.audition_frames = Some((AUDITION_SECONDS * ENGINE_SAMPLE_RATE_HZ) as usize);
        self.release_all();
        self.play_tones(section.1);
    }

    /// Starts every tone of `chord` at the head of the held curve.
    fn play_tones(&mut self, chord: Chord) {
        for key in chord_keys(&chord) {
            let pitch = self.instrument.pitch_for(key);
            let index = self.synth.play(
                self.instrument.waveform.clone(),
                pitch,
                self.instrument.held_level(0.0),
                &self.config,
            );
            // `play` hands out voices round-robin, so this may be one still ringing from an earlier
            // chord. The new note owns it now; the old envelope must not keep writing to it.
            self.ringing.retain(|r| r.index != index);
            self.ringing.push(Ringing {
                index,
                pos: 0.0,
                phase: Phase::Held,
            });
        }
    }

    /// Lets go of every sounding tone: each starts the captured release tail from wherever its
    /// level currently is. Already-releasing tones are left alone — releasing twice would restart
    /// the tail and make the chord swell back up.
    fn release_all(&mut self) {
        for r in &mut self.ringing {
            if let Phase::Held = r.phase {
                let from = self.instrument.held_level(r.pos);
                r.phase = Phase::Released { from };
                r.pos = 0.0;
            }
        }
    }

    /// Forgets what is sounding, so the next chord strikes even if it matches. Used on a seek: after
    /// jumping, the chord under the playhead should sound again.
    pub fn reset(&mut self) {
        self.current = None;
    }

    /// The transport stopped: let go of the chord the playhead was holding, and forget it so
    /// resuming strikes it afresh.
    ///
    /// Without this a section chord would sustain forever the moment you hit pause — it is held
    /// until the *section* changes, and a stopped playhead never changes section. A hand-struck
    /// audition is left alone: it has its own clock, and right-clicking while stopped is exactly
    /// when you most want to hear one.
    pub fn stop_following(&mut self) {
        if self.audition_frames.is_some() {
            return;
        }
        self.current = None;
        self.release_all();
    }

    /// One stereo frame of chord audio, already at [`DEFAULT_GAIN`].
    #[inline]
    pub fn next_frame(&mut self) -> (f32, f32) {
        self.advance_envelopes();
        self.synth.next_sample(&self.config);
        (self.synth.val_l * self.gain, self.synth.val_r * self.gain)
    }

    /// Walks every ringing tone one frame along the captured ADSR, cutting it once its release has
    /// run out. Held tones never cut: they sit on the sustain level until the section changes. This
    /// is the only thing that ever stops a voice — the waveforms loop.
    fn advance_envelopes(&mut self) {
        // A hand-struck chord lets go of itself when its clock runs out.
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

    /// An instrument whose ADSR is known exactly, so envelope behaviour can be asserted rather than
    /// inferred from a ROM. Attacks over 2 ticks, decays to a 0.5 sustain, releases over 4.
    fn test_instrument() -> Instrument {
        Instrument {
            waveform: Arc::new(Waveform::new(
                vec![0.5; 64],
                440.0,
                8000.0,
                /* looping */ true,
                0,
            )),
            pitch: VoicePitch::DataRateHz(8000.0),
            key: 60,
            held: vec![0.0, 1.0, 0.5],
            release: vec![1.0, 0.75, 0.5, 0.25, 0.0],
            tick_seconds: 0.01,
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

    /// The point of splitting the ADSR at the note-off: a held chord sits on the sustain level for
    /// as long as its section lasts, rather than replaying one captured note's length and quitting.
    #[test]
    fn a_held_chord_sustains_indefinitely() {
        let mut v = ChordVoicer::new(test_instrument());
        v.set_chord(Some((0, chord(0, Quality::Major))));

        // Well past the 3-tick (30 ms) held curve, the chord is still sounding, on sustain.
        let mut loudest: f32 = 0.0;
        for _ in 0..(2.0 * ENGINE_SAMPLE_RATE_HZ) as usize {
            let (l, r) = v.next_frame();
            loudest = loudest.max(l.abs()).max(r.abs());
        }
        assert!(loudest > 0.0, "a held chord must still sound after 2 s");
        assert_eq!(v.ringing.len(), 3, "three tones still held");
        for r in &v.ringing {
            assert!(
                matches!(r.phase, Phase::Held),
                "an unchanged section must not release"
            );
        }
    }

    /// The other half of the same contract: once the section changes (or ends), the captured release
    /// runs and the voice is actually freed. These waveforms loop, so nothing else would stop them.
    #[test]
    fn a_released_chord_dies_and_frees_its_voice() {
        let mut v = ChordVoicer::new(test_instrument());
        v.set_chord(Some((0, chord(0, Quality::Major))));
        for _ in 0..(0.5 * ENGINE_SAMPLE_RATE_HZ) as usize {
            v.next_frame();
        }
        // N.C.: an annotated rest releases the chord and puts nothing in its place.
        v.set_chord(None);
        assert!(
            v.ringing
                .iter()
                .all(|r| matches!(r.phase, Phase::Released { .. })),
            "N.C. must release the chord"
        );

        // The 4-tick (40 ms) release runs out, the tones are cut, and silence is permanent.
        for _ in 0..(0.2 * ENGINE_SAMPLE_RATE_HZ) as usize {
            v.next_frame();
        }
        assert!(v.ringing.is_empty(), "released voices must be freed");
        for _ in 0..1000 {
            assert_eq!(
                v.next_frame(),
                (0.0, 0.0),
                "a looping voice was left ringing"
            );
        }
    }

    /// Following the playhead must strike once per *section*, not once per frame — and a repeat of
    /// the same chord in the next section is a new section, so it strikes again.
    #[test]
    fn sections_strike_once_each() {
        let mut v = ChordVoicer::new(test_instrument());
        let c = chord(0, Quality::Major);

        v.set_chord(Some((0, c)));
        assert_eq!(v.ringing.len(), 3, "C major strikes three tones");
        for _ in 0..10 {
            v.set_chord(Some((0, c)));
        }
        assert_eq!(v.ringing.len(), 3, "one section, one strike");
        assert!(
            v.ringing.iter().all(|r| matches!(r.phase, Phase::Held)),
            "the same section must keep holding"
        );

        // The next bar, labelled the same chord, is a new section: three fresh tones strike while
        // the old three release.
        v.set_chord(Some((96, c)));
        assert_eq!(v.ringing.len(), 6, "a new section restrikes the chord");
        assert_eq!(
            v.ringing
                .iter()
                .filter(|r| matches!(r.phase, Phase::Held))
                .count(),
            3,
            "exactly the new tones are held; the old ones are releasing"
        );
    }

    /// A release must start from wherever the chord actually was, not from full level — otherwise
    /// letting go of a decayed chord makes it jump back up.
    #[test]
    fn release_starts_from_the_current_level() {
        let mut v = ChordVoicer::new(test_instrument());
        v.set_chord(Some((0, chord(0, Quality::Major))));
        // Run past the held curve so every tone is sitting on the 0.5 sustain.
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

    /// Stopping the transport must let go of a section chord — it is held until the section
    /// changes, and a stopped playhead never changes section, so nothing else would ever end it.
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
        // And resuming re-strikes it, rather than treating the section as already sounding.
        v.set_chord(Some((0, chord(0, Quality::Major))));
        assert_eq!(v.ringing.len(), 3);
    }

    /// A right-clicked chord must survive the stopped playhead's release (auditioning while stopped
    /// is the main way this tool is used) and must then end on its own, since nothing else can.
    #[test]
    fn a_hand_struck_chord_outlives_a_stopped_playhead_then_expires() {
        let mut v = ChordVoicer::new(test_instrument());
        v.strike((0, chord(0, Quality::Major)));
        // The follower runs every UI frame while stopped; it must not touch the audition.
        for _ in 0..(0.5 * ENGINE_SAMPLE_RATE_HZ) as usize {
            v.stop_following();
            v.next_frame();
        }
        assert_eq!(v.ringing.len(), 3, "the audition must still be ringing");

        // Past the audition clock it releases itself and falls silent for good.
        for _ in 0..((AUDITION_SECONDS + 0.5) * ENGINE_SAMPLE_RATE_HZ) as usize {
            v.stop_following();
            v.next_frame();
        }
        assert!(v.ringing.is_empty(), "a hand-struck chord must end itself");
    }

    /// The playhead reaching a chord that was struck by hand adopts it, rather than restriking it
    /// (a flam) or letting the audition clock cut it off mid-section.
    #[test]
    fn the_playhead_adopts_a_hand_struck_chord() {
        let mut v = ChordVoicer::new(test_instrument());
        v.strike((0, chord(0, Quality::Major)));
        v.next_frame();
        v.set_chord(Some((0, chord(0, Quality::Major))));
        assert_eq!(
            v.ringing.len(),
            3,
            "must not restrike what is already sounding"
        );
        assert!(v.audition_frames.is_none(), "the section owns it now");
        // Which means it holds well past where the audition clock would have expired.
        for _ in 0..((AUDITION_SECONDS + 0.5) * ENGINE_SAMPLE_RATE_HZ) as usize {
            v.set_chord(Some((0, chord(0, Quality::Major))));
            v.next_frame();
        }
        assert_eq!(v.ringing.len(), 3, "a section chord holds for the section");
    }

    #[test]
    fn data_rate_pitch_scales_by_the_interval() {
        let instrument = test_instrument();
        // An octave up is exactly double the data rate.
        match instrument.pitch_for(72) {
            VoicePitch::DataRateHz(hz) => assert!((hz - 16000.0).abs() < 1e-6, "got {hz}"),
            other => panic!("expected DataRateHz, got {other:?}"),
        }
        // An octave down is half.
        match instrument.pitch_for(48) {
            VoicePitch::DataRateHz(hz) => assert!((hz - 4000.0).abs() < 1e-6, "got {hz}"),
            other => panic!("expected DataRateHz, got {other:?}"),
        }
        // The captured key itself is unchanged.
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
                // The sample's own recorded pitch is a property of the data, not of the key.
                assert_eq!(sample_pitch_hz, 440.0);
            }
            other => panic!("expected Midi, got {other:?}"),
        }
    }

    /// The real thing: Emerald must actually yield a sampled voice with a usable ADSR. Pins the
    /// assumptions this feature rests on — that `NoteStarted` carries a waveform and that
    /// `VoiceVolume`/`NoteReleased` describe a real envelope — against the ROM rather than a mock.
    #[test]
    fn captures_a_real_voice_from_emerald() {
        let instrument = emerald_voice();
        assert!(
            !instrument.waveform.data.is_empty(),
            "voice must have sample data"
        );
        assert!(
            !instrument.waveform.is_psg_square,
            "must not be a PSG beeper"
        );

        // The release is what ends the note: this sample loops, so without a tail that reaches
        // silence the chord would drone under the song forever.
        assert!(instrument.waveform.looping, "premise: the ROM voice loops");
        assert_eq!(
            instrument.release.last().copied(),
            Some(0.0),
            "the release must end in silence"
        );
        assert!(
            instrument.held_seconds() <= MAX_ENVELOPE_SECONDS,
            "held curve ran {}s",
            instrument.held_seconds()
        );

        // The split must land on a real sustain: a chord holds this level for the whole of its
        // section, so a sustain of ~0 would mean `NoteReleased` was found in the wrong place and
        // every chord silently died at the end of its decay.
        assert!(
            instrument.sustain_level() > 0.05,
            "sustain {} is inaudible — the ADSR split is wrong",
            instrument.sustain_level()
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

    fn emerald_voice() -> Instrument {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../demos/pokemon-emerald.gbaaudio");
        let bytes = std::fs::read(path).expect("demo file should exist");
        let archives = optime_core::load_all(&bytes);
        let data = archives.first().expect("an archive");
        capture(&**data, PREFERRED_SONG).expect("song 435 should yield a voice")
    }

    /// The whole point, measured: a struck C major must actually contain C, E and G. Guards the
    /// repitching end-to-end on the real ROM voice — a chord that is merely *audible* could still be
    /// a unison, a detuned mess, or one note.
    #[test]
    fn a_struck_chord_contains_its_three_pitches() {
        let mut v = ChordVoicer::new(emerald_voice());
        v.gain = 1.0;
        v.strike((0, chord(0, Quality::Major)));
        let n = (0.4 * ENGINE_SAMPLE_RATE_HZ) as usize;
        let buf: Vec<f64> = (0..n)
            .map(|_| {
                let (l, r) = v.next_frame();
                (l as f64 + r as f64) * 0.5
            })
            .collect();
        // Goertzel-style magnitude at one frequency.
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
        // A pitch the chord does not contain, as the noise floor to beat.
        let d4 = mag(293.66);
        for (name, m) in [("C4", c4), ("E4", e4), ("G4", g4)] {
            assert!(
                m > d4 * 8.0,
                "{name} ({m:.5}) must stand well clear of the unplayed D4 ({d4:.5})"
            );
        }
    }
}

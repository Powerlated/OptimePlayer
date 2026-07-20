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

use optime_core::biquad_filter::BiquadFilter;
use optime_core::{PerDeviceSettings, Sample, VoicePitch, Waveform, WaveformSynthesizer};

use super::model::{Chord, Quality};
use crate::audio::ENGINE_SAMPLE_RATE_HZ;

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

/// The embedded piano sample's reference pitch — middle C (MIDI 60). The sample is the SC-88pro
/// piano lifted from pokeemerald's source tree, shipped in `chord_data/piano_c4.wav` and
/// `include_bytes!`'d in so the audition always has a known-good reference instrument regardless of
/// the loaded ROM (the ROM-capture path would grab a bass thump on quiet tracks).
const PIANO_FREQ: f64 = 261.6256;
/// The piano's ADSR, lifted verbatim from pokeemerald's voicegroup
/// (`sound/voicegroups/keysplits/piano.inc`: `voice_directsound 60, 0, ..., 255, 250, 0, 221` — the
/// macro param order is `base_midi_key, pan, sample, attack, decay, sustain, release`, per
/// `asm/macros/music_voice.inc`). Sustain is 0: a percussive envelope that decays to silence.
const PIANO_ADSR: [u8; 4] = [255, 250, 0, 221];
/// One envelope entry per GBA VBlank (`GBA_CLOCK_RATE / CYCLES_PER_FRAME` ≈ 59.73 Hz), the rate the
/// m4a DirectSound envelope steps at — so the curve matches how the real hardware plays it.
const PIANO_TICK_SECONDS: f64 = 1.0 / 59.7275;

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

/// The MIDI keys of `chord` voiced in `inversion`, root position starting at middle C. The lowest
/// `inversion` tones are lifted an octave, so the (0-indexed) `inversion`-th scale degree becomes
/// the bass — 1 = first inversion, 2 = second, and so on. `inversion` is clamped to one below the
/// tone count: lifting *every* tone would only shift the whole chord an octave, not change the bass.
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

/// The embedded middle-C piano sample as a playable [`Instrument`], voiced with the piano's **real
/// ADSR** from pokeemerald's voicegroup ([`PIANO_ADSR`]), stepped through the m4a DirectSound
/// envelope algorithm (see [`piano_envelope`]).
///
/// The waveform **loops end-to-end** (`looping=true`): the tones pitch the sample at different
/// rates, so each voice's loop period differs and the per-sample attack drifts into an arpeggio —
/// which is the intended colour here. The real ADSR (sustain 0) still makes the *envelope* decay to
/// silence over a few seconds, so a struck chord rings out as a decaying arpeggio rather than
/// droning forever; a section that outlasts the decay simply goes quiet.
pub fn embedded_piano() -> Instrument {
    const PIANO_WAV: &[u8] = include_bytes!("../chord_data/piano_c4.wav");
    let (sample_rate, data) = find_data_chunk(PIANO_WAV).unwrap_or((13_379.0, &[][..]));
    // GBA DirectSound stores 8-bit PCM *unsigned*, centred at 128 — not the signed 8-bit the stock
    // `decode_pcm8` assumes, so decode it here rather than reaching for the shared decoder.
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

/// Walks the RIFF chunks, returning the `fmt ` sample rate and the `data` payload slice. The stock
/// `decode_wav` assumes a fixed 44-byte header this GBA-flavoured file (with `smpl`/`agbp`/`agbl`
/// chunks) doesn't have, so this finds the chunks by tag. Returns `None` if it isn't a WAVE file.
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
        o = body + sz + (sz & 1); // chunks are word-aligned (padded to even)
    }
    Some((sample_rate, data))
}

/// The held + release curves for `adsr`, transcribed from the m4a DirectSound envelope stepper
/// (`direct_sound_env` in optime-core's gba device) with pseudo-echo off (`echo_volume = 0`). The
/// envelope is one entry per GBA VBlank ([`PIANO_TICK_SECONDS`]), stepped exactly as the hardware
/// would: attack adds, decay and release multiply (`env = env * rate >> 8`).
///
/// Returns `(held, release)`:
/// - **held** runs attack → decay down to `sustain`, peak-normalised to `1.0`; its last entry is the
///   sustain level (0.0 for the piano → a held section settles to silence, i.e. percussive).
/// - **release** is a relative `1.0 → 0.0` multiplier (the release rate applied from peak), applied
///   from whatever level a chord sits at when let go.
fn piano_envelope(adsr: [u8; 4]) -> (Vec<Sample>, Vec<Sample>) {
    let [attack, decay, sustain, release] = adsr;
    let peak: u32 = 0xFF;

    // Held. SF_START sets env = attack (peak, if attack >= 0xFF — drops straight into decay); then
    // the ATTACK phase adds `attack` per VBlank until peak; then DECAY multiplies.
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
    // Decay: env = env * decay >> 8 each VBlank, until it reaches sustain (the piano's is 0).
    let sustain = sustain as u32;
    loop {
        env = (env * decay as u32) >> 8;
        held.push(env as Sample / peak as Sample);
        if env <= sustain {
            break;
        }
    }

    // Release: env = env * release >> 8 each VBlank from peak until 0 — relative 1.0 → 0.0.
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
    /// Which inversion chords voice in. Clamped to `intervals.len() - 1` at voicing time, so a value
    /// picked on a seventh still applies to a triad without putting every tone an octave up.
    inversion: u8,
    /// Master high-shelf on the audition output, mirroring the "Enhanced" GBA preset's shelf so the
    /// reference piano is judged through the same top-end the user hears the song through.
    shelf_l: BiquadFilter,
    shelf_r: BiquadFilter,
}

impl ChordVoicer {
    pub fn new(instrument: Instrument) -> ChordVoicer {
        let env_step = 1.0 / (instrument.tick_seconds * ENGINE_SAMPLE_RATE_HZ);
        // The audition shelf mirrors the "Enhanced" GBA preset exactly, so a label is judged through
        // the same top-end attenuation the user hears the song through.
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

    /// Sets the audition gain. Takes effect on the next output frame — `next_frame` reads `gain`
    /// every sample, so there is nothing to re-strike.
    pub fn set_gain(&mut self, gain: f32) {
        self.gain = gain;
    }

    /// Sets the voicing inversion, re-voicing whatever chord is currently sounding so the change is
    /// audible at once rather than only on the next strike. A no-op (apart from storing) if nothing
    /// is sounding, in which case the value applies to the next strike.
    pub fn set_inversion(&mut self, inversion: u8) {
        if self.inversion == inversion {
            return;
        }
        self.inversion = inversion;
        if let Some((_, c)) = self.current {
            // Copy the chord out of the borrow before mutating through `self`.
            self.release_all();
            self.play_tones(c);
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

    /// Starts every tone of `chord` at the head of the held curve: the triad (in the chosen
    /// inversion) plus the root doubled an octave below as a bass note. The bass is always the root
    /// regardless of inversion, so the chord's function reads clearly against the song.
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

    /// One stereo frame of chord audio, at [`Self::gain`] and through the audition high-shelf.
    #[inline]
    pub fn next_frame(&mut self) -> (f32, f32) {
        self.advance_envelopes();
        self.synth.next_sample(&self.config);
        let l = self.shelf_l.transform(self.synth.val_l * self.gain);
        let r = self.shelf_r.transform(self.synth.val_r * self.gain);
        (l, r)
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
        assert_eq!(voiced_keys(&chord(0, Quality::Major), 0), vec![60, 64, 67]);
        // A minor → A4 C5 E5: the root moves up from middle C, it does not wrap below it.
        assert_eq!(voiced_keys(&chord(9, Quality::Minor), 0), vec![69, 72, 76]);
        // Sevenths add a fourth tone.
        assert_eq!(
            voiced_keys(&chord(7, Quality::Dominant7), 0),
            vec![67, 71, 74, 77]
        );
    }

    #[test]
    fn inversions_lift_the_lowest_tones() {
        // C major [C4 E4 G4] = [60, 64, 67].
        // 1st inversion: the root is lifted an octave → E G C5.
        assert_eq!(voiced_keys(&chord(0, Quality::Major), 1), vec![72, 64, 67]);
        // 2nd inversion: root and third lifted → G C5 E5.
        assert_eq!(voiced_keys(&chord(0, Quality::Major), 2), vec![72, 76, 67]);
        // 3rd on a triad clamps to 2nd — lifting all three tones would only shift the octave, not
        // change the bass, so it is not a distinct voicing.
        assert_eq!(voiced_keys(&chord(0, Quality::Major), 3), vec![72, 76, 67]);
        // C7 [C4 E4 G4 Bb4] = [60, 64, 67, 70]. 1st inversion: C up → E G Bb C5.
        assert_eq!(
            voiced_keys(&chord(0, Quality::Dominant7), 1),
            vec![72, 64, 67, 70]
        );
        // 3rd inversion on a seventh: three lowest lifted, the seventh becomes the bass.
        assert_eq!(
            voiced_keys(&chord(0, Quality::Dominant7), 3),
            vec![72, 76, 79, 70]
        );
        // 4th on a seventh clamps to 3rd.
        assert_eq!(
            voiced_keys(&chord(0, Quality::Dominant7), 4),
            vec![72, 76, 79, 70]
        );
    }

    /// Every quality must voice a distinct pitch set — two qualities that sound identical would make
    /// the audition useless for telling them apart, which is exactly what it is for.
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
        assert_eq!(v.ringing.len(), 4, "four tones (triad + bass) still held");
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

        // The 4-tick (40 ms) release runs out, the tones are cut, and silence is permanent. The
        // high-shelf is a recursive filter, so a fully-muted input can leave a denormal residual —
        // compare against a floor rather than exact zero.
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

    /// Following the playhead must strike once per *section*, not once per frame — and a repeat of
    /// the same chord in the next section is a new section, so it strikes again.
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

        // The next bar, labelled the same chord, is a new section: four fresh tones strike while
        // the old four release.
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
        assert_eq!(v.ringing.len(), 4);
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
        assert_eq!(v.ringing.len(), 4, "the audition must still be ringing");

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
            4,
            "must not restrike what is already sounding"
        );
        assert!(v.audition_frames.is_none(), "the section owns it now");
        // Which means it holds well past where the audition clock would have expired.
        for _ in 0..((AUDITION_SECONDS + 0.5) * ENGINE_SAMPLE_RATE_HZ) as usize {
            v.set_chord(Some((0, chord(0, Quality::Major))));
            v.next_frame();
        }
        assert_eq!(v.ringing.len(), 4, "a section chord holds for the section");
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

    /// The whole point, measured: a struck C major must actually contain C, E and G (and, below
    /// them, the root bass). Guards the repitching end-to-end on the embedded piano — a chord that
    /// is merely *audible* could still be a unison, a detuned mess, or one note.
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
        // The embedded piano is an 8-bit sample pitched up to E4/G4, then put through the audition
        // shelf, so its upper tones don't tower over the floor the way a clean ROM voice would — a
        // 4× clear margin still proves the tone is there rather than bleed.
        for (name, m) in [("C4", c4), ("E4", e4), ("G4", g4)] {
            assert!(
                m > d4 * 4.0,
                "{name} ({m:.5}) must stand clear of the unplayed D4 ({d4:.5})"
            );
        }
    }
}

//! [`DsePlayer`]: the DSE device player. Runs the [`DseSequencer`], resolves notes to samples
//! through the program/split tables, drives one [`SoundEnvelope`] per voice, and emits
//! standardized [`SynthEvent`]s.
//!
//! Pitch and volume are transcribed faithfully from the driver's voice-update code: the absolute
//! playback rate comes from [`super::pitch::note_key_to_hz`] (the `DseVoice_UpdateParameters`
//! pitch tables) and the per-voice amplitude from [`super::volume`] (the same routine's square
//! law). Not yet modelled: channel pitch-bend / per-axis LFOs, `SongVolumeFade`, and per-note
//! pan (the split's pan field) — none of which change which notes or rhythm play.

use std::collections::HashMap;
use std::sync::Arc;

use super::envelope::{EnvelopeParams, SoundEnvelope, USEC_PER_DRIVER_TICK};
use super::lfo::{Lfo, LfoConfig, LfoDest, LfoRng};
use super::pitch::note_key_to_hz;
use super::sequencer::{DseSequencer, SeqOp};
use super::swdl::Swdl;
use super::{volume, SampleInfo, Smdl};
use crate::devices::{SynthEvent, TickFeedback, VoiceId, VoicePitch};
use crate::sample::Sample;
use crate::PerDeviceSettings;
use crate::TRACK_COUNT;

/// DSE driver clock cycles between ticks: `64 * 5236` cycles of the 33.51 MHz clock (the
/// `DseDriver_StartTickTimer` alarm), ≈ 100 Hz.
pub const DSE_CYCLES_PER_TICK: u64 = 64 * 5236;

/// Master gain applied to every DSE voice. Each voice's faithfully-modelled volume reaches unity
/// at full (`volume.rs`), and the controller sums all 16 tracks with no limiter — so dense
/// Explorers mixes would clip. On hardware that headroom comes from the NDS mixer + the driver's
/// song/global master volume (`volume_final = g1*g2/127`); we fold it into one constant. Measured
/// across a spread of Explorers BGM, the busiest songs peak near full scale at unity, so this
/// keeps them comfortably below clipping while matching the level of the DS/GBA backends.
const DSE_MASTER_GAIN: f64 = 0.8;

/// A linearly-interpolated control value (the DSE `dse_fade`): holds `value << 8` for sub-integer
/// ramp precision. `SetX` snaps it; `XFade` ramps it to a target over N driver ticks; `XDelta`
/// nudges it. Read the integer value with [`Ramp::value`].
#[derive(Debug, Clone, Copy)]
struct Ramp {
    current: i32,
    target: i32,
    delta: i32,
    ticks: i32,
}

impl Ramp {
    fn new(value: i32) -> Ramp {
        Ramp {
            current: value << 8,
            target: value << 8,
            delta: 0,
            ticks: 0,
        }
    }

    /// The current integer value.
    fn value(&self) -> i32 {
        self.current >> 8
    }

    /// Jumps immediately to `value`, cancelling any fade.
    fn set(&mut self, value: i32) {
        self.current = value << 8;
        self.target = self.current;
        self.ticks = 0;
    }

    /// Begins a linear fade to `target` over `ticks` driver ticks (`ticks <= 0` jumps).
    fn fade_to(&mut self, target: i32, ticks: i32) {
        self.target = target << 8;
        if ticks <= 0 {
            self.current = self.target;
            self.ticks = 0;
        } else {
            self.delta = (self.target - self.current) / ticks;
            self.ticks = ticks;
        }
    }

    /// Advances one driver tick. Returns `true` if the value changed.
    fn tick(&mut self) -> bool {
        if self.ticks <= 0 {
            return false;
        }
        self.ticks -= 1;
        if self.ticks == 0 {
            self.current = self.target;
        } else {
            self.current += self.delta;
        }
        true
    }
}

/// Per-track state the player tracks for note resolution, volume, pan, and pitch bend.
#[derive(Debug, Clone)]
struct DseTrack {
    program: u16,
    /// Channel volume (0..=127), `SetVolume`/`VolumeFade`/`VolumeDelta`.
    volume: Ramp,
    expression: u8,
    /// Pan (0=left, 64=center, 127=right), `SetPan`/`PanFade`/`PanDelta`.
    pan: Ramp,
    /// Base tuning in 8.8 fixed-point semitones (`SetTuning` + `TuningDelta*`).
    tuning_8_8: i32,
    /// The separate `TuningFade` contribution, also 8.8 semitones.
    tuning_fade: Ramp,
    /// Pitch-wheel bend value (`SetKeyBend`, signed; `±8192` ≈ ±`range` semitones).
    key_bend: i32,
    /// Bend range in semitones (`SetKeyBendRange`); `0` falls back to the split's `bend_sensitivity`.
    key_bend_range: u8,
    /// The current note's split `bend_sensitivity`, captured at note-on for that fallback.
    split_bend_sensitivity: u8,
    /// The channel's four pending LFO config slots (`channel + 0x74`, 0x10 bytes each). The
    /// dedicated key-bend/volume/pan opcodes target slots 0/1/2; the generic opcodes any slot.
    lfo_slots: [LfoConfig; 4],
    /// The current generic-LFO slot index (`channel + 0x61`), set by `UseLfo`/`SetLfoParameter`.
    lfo_index: usize,
    /// The live auto-pan LFO (a pan-routed slot), ticked at the track level since pan is a track
    /// property in the synth layer. Pitch/volume LFOs are per-voice instead (see [`DseVoice`]).
    pan_lfo: Option<Lfo>,
    /// The pan index (0..=127) last emitted as a `TrackPan`, to suppress duplicate events.
    last_pan: i32,
}

impl DseTrack {
    /// Net pitch bend in semitones — tuning (`SetTuning`/`TuningDelta`/`TuningFade`) plus the
    /// pitch-wheel key bend — applied as a track detune. The key-bend offset matches
    /// `DseChannel_SetKeyBend`: `range * value / 8192` semitones (`range * (value<<8) / 8192` in
    /// the 8.8 note_key units the ROM uses).
    fn bend_semitones(&self) -> f64 {
        let tuning = f64::from(self.tuning_8_8 + self.tuning_fade.value()) / 256.0;
        let range = if self.key_bend_range != 0 {
            self.key_bend_range
        } else {
            self.split_bend_sensitivity
        };
        let key_bend = f64::from(i32::from(range) * self.key_bend) / 8192.0;
        tuning + key_bend
    }
}

impl Default for DseTrack {
    fn default() -> Self {
        DseTrack {
            program: 0,
            volume: Ramp::new(127),
            expression: 127,
            pan: Ramp::new(64),
            tuning_8_8: 0,
            tuning_fade: Ramp::new(0),
            key_bend: 0,
            key_bend_range: 0,
            split_bend_sensitivity: 2,
            lfo_slots: [LfoConfig::default(); 4],
            lfo_index: 0,
            pan_lfo: None,
            last_pan: -1,
        }
    }
}

/// One sounding voice.
struct DseVoice {
    id: VoiceId,
    track: usize,
    key: u8,
    /// `DseVoice_PlayNote`'s per-note volume (`velocity * program * split / 127^2`, 0..=127).
    note_volume: u8,
    env: SoundEnvelope,
    /// Sequencer tick at which the note auto-releases.
    release_tick: u32,
    released: bool,
    /// This note's own pitch (vibrato) and volume (tremolo) LFOs, built from the track's config
    /// at note-on so each note restarts its fade-in — exactly as the hardware builds the voice's
    /// LFO bank in `DseVoice_PlayNote`.
    lfos: Vec<Lfo>,
    /// The last pitch-LFO detune (semitones) emitted as a `VoiceDetune`, to suppress duplicates.
    last_detune: f64,
}

/// The DSE device player.
pub struct DsePlayer {
    seq: DseSequencer,
    /// Shared main bank (holds the `pcmd` sample data).
    main_bank: Arc<Swdl>,
    /// Per-song bank (programs/splits; its WAVI is ignored — see [`Self::sample`]).
    song_bank: Arc<Swdl>,
    /// Decoded samples, keyed by a split's `wave_index` (resolved against the main bank's WAVI).
    sample_cache: HashMap<i16, Option<Arc<Sample>>>,
    tracks: [DseTrack; TRACK_COUNT],
    voices: Vec<DseVoice>,
    accum_us: i64,
    next_voice: VoiceId,
    /// Shared LFO noise RNG (`DseUtil_GetRandomNumber`).
    lfo_rng: LfoRng,
}

impl DsePlayer {
    /// Builds a player for `smdl` using `song_bank`'s programs and `main_bank`'s samples.
    pub fn new(smdl: &Smdl, song_bank: Arc<Swdl>, main_bank: Arc<Swdl>) -> DsePlayer {
        DsePlayer {
            seq: DseSequencer::new(smdl),
            main_bank,
            song_bank,
            sample_cache: HashMap::new(),
            tracks: std::array::from_fn(|_| DseTrack::default()),
            voices: Vec::new(),
            accum_us: 0,
            next_voice: 0,
            lfo_rng: LfoRng::default(),
        }
    }

    /// Sequencer ticks executed (the visualizer timeline).
    pub fn steps_elapsed(&self) -> u32 {
        self.seq.ticks_elapsed
    }

    /// Current sequencer step rate in Hz: `bpm * tpqn / 60`.
    pub fn step_rate(&self) -> f64 {
        f64::from(self.seq.bpm.max(1)) * f64::from(self.seq.tpqn) / 60.0
    }

    /// Microseconds per sequencer tick at the current tempo.
    fn seq_tick_us(&self) -> i64 {
        let denom = i64::from(self.seq.bpm.max(1)) * i64::from(self.seq.tpqn);
        (60_000_000 / denom).max(1)
    }

    /// Advances one driver tick: updates envelopes, then runs any sequencer ticks now due.
    pub fn tick(
        &mut self,
        feedback: &mut TickFeedback,
        _config: &PerDeviceSettings,
        events: &mut Vec<SynthEvent>,
    ) {
        self.update_voices(feedback, events);
        feedback.ended_voices.clear();

        self.accum_us += USEC_PER_DRIVER_TICK;
        let mut threshold = self.seq_tick_us();
        let mut ops = Vec::new();
        while self.accum_us >= threshold {
            self.accum_us -= threshold;
            ops.clear();
            self.seq.seq_tick(&mut ops);
            for op in ops.drain(..) {
                self.handle_op(op, events);
            }
            threshold = self.seq_tick_us();
        }
    }

    /// Advances every voice's envelope one driver tick, beginning releases whose note duration
    /// has elapsed and freeing voices that have finished or been stolen.
    fn update_voices(&mut self, feedback: &TickFeedback, events: &mut Vec<SynthEvent>) {
        let now = self.seq.ticks_elapsed;
        // The LFO noise RNG is shared; lift it out so both the per-track and per-voice loops can
        // tick LFOs without aliasing `self`.
        let mut rng = std::mem::take(&mut self.lfo_rng);

        // Step the per-track fades one driver tick and the auto-pan LFO. A tuning change re-emits
        // its track detune; the pan event combines the base pan (fade) with the LFO offset.
        for track in 0..self.tracks.len() {
            let t = &mut self.tracks[track];
            t.volume.tick();
            let pan_faded = t.pan.tick();
            if t.tuning_fade.tick() {
                events.push(SynthEvent::TrackDetune {
                    track,
                    semitones: t.bend_semitones(),
                });
            }
            // Auto-pan: add the pan LFO's `>> 6` offset to the 0..=127 pan index, exactly as
            // `DseVoice_UpdateParameters` does for the pan path.
            let pan_mod = match &mut t.pan_lfo {
                Some(lfo) => lfo.tick(&mut rng),
                None => 0,
            };
            if t.pan_lfo.is_some() || pan_faded {
                let pan_idx = (t.pan.value() + (pan_mod >> 6)).clamp(0, 127);
                if pan_idx != t.last_pan {
                    t.last_pan = pan_idx;
                    events.push(SynthEvent::TrackPan {
                        track,
                        pan: f64::from(pan_idx) / 127.0,
                    });
                }
            }
        }

        let mut remove = Vec::new();
        for idx in 0..self.voices.len() {
            if feedback.is_ended(self.voices[idx].track, self.voices[idx].id) {
                remove.push(idx);
                continue;
            }

            let v = &mut self.voices[idx];

            // Auto-release once the note's scheduled duration elapses.
            if !v.released && now >= v.release_tick {
                v.released = true;
                v.env.release();
                events.push(SynthEvent::NoteReleased {
                    track: v.track,
                    key: v.key,
                });
                // A note that never used the slide envelope has no release tail: cut it now.
                if !v.env.uses_envelope() {
                    events.push(SynthEvent::VoiceStopped {
                        track: v.track,
                        voice: v.id,
                    });
                    remove.push(idx);
                    continue;
                }
            }

            let level = v.env.tick();
            if v.env.is_finished() {
                events.push(SynthEvent::VoiceStopped {
                    track: v.track,
                    voice: v.id,
                });
                remove.push(idx);
                continue;
            }

            // Tick this note's vibrato/tremolo LFOs: pitch adds to the 8.8 note_key (→ a per-voice
            // detune), volume adds as `>> 6` to the 0..=127 note volume.
            let (mut pitch_mod, mut vol_mod) = (0i32, 0i32);
            for lfo in &mut v.lfos {
                let out = lfo.tick(&mut rng);
                match lfo.dest {
                    LfoDest::Pitch => pitch_mod += out,
                    LfoDest::Volume => vol_mod += out,
                    LfoDest::Pan => {} // pan LFOs are ticked at the track level
                }
            }
            let detune = f64::from(pitch_mod) / 256.0;
            if detune != v.last_detune {
                v.last_detune = detune;
                events.push(SynthEvent::VoiceDetune {
                    track: v.track,
                    voice: v.id,
                    semitones: detune,
                });
            }
            let note_volume = (i32::from(v.note_volume) + (vol_mod >> 6)).clamp(0, 127) as u8;

            let (track, voice_id) = (v.track, v.id);
            let t = &self.tracks[track];
            let vol_final = volume::volume_final(t.volume.value() as u8, t.expression);
            let volume = volume::voice_amp(level, vol_final, note_volume) * DSE_MASTER_GAIN;
            events.push(SynthEvent::VoiceVolume {
                track,
                voice: voice_id,
                volume,
            });
        }
        // Remove from the back so earlier indices stay valid.
        for &idx in remove.iter().rev() {
            self.voices.remove(idx);
        }
        self.lfo_rng = rng;
    }

    /// Applies one sequencer op.
    fn handle_op(&mut self, op: SeqOp, events: &mut Vec<SynthEvent>) {
        match op {
            SeqOp::NoteOn {
                track,
                key,
                velocity,
                duration,
            } => self.start_note(track, key, velocity, duration, events),
            SeqOp::Program { track, program } => {
                if let Some(t) = self.tracks.get_mut(track) {
                    t.program = program;
                }
            }
            SeqOp::Volume { track, volume } => {
                if let Some(t) = self.tracks.get_mut(track) {
                    t.volume.set(i32::from(volume));
                }
            }
            SeqOp::Expression { track, value } => {
                if let Some(t) = self.tracks.get_mut(track) {
                    t.expression = value;
                }
            }
            SeqOp::Pan { track, pan } => {
                if let Some(t) = self.tracks.get_mut(track) {
                    t.pan.set(i32::from(pan));
                    t.last_pan = i32::from(pan);
                    events.push(SynthEvent::TrackPan {
                        track,
                        pan: f64::from(pan) / 127.0,
                    });
                }
            }
            SeqOp::Control {
                track,
                opcode,
                operands,
            } => self.handle_control(track, opcode, &operands, events),
            SeqOp::Tempo { .. } => {} // tempo lives in the sequencer; affects scheduling only
            SeqOp::Looped => events.push(SynthEvent::Looped),
            SeqOp::TrackEnded { .. } => {
                if self.seq.ended {
                    events.push(SynthEvent::Ended);
                }
            }
        }
    }

    /// Handles a musical control opcode (tuning/bend, volume/pan fades & deltas). LFO setup is
    /// handled separately. Mirrors the `DseTrackEvent_*` handlers in `lib/DSE/asm/main_02071EB4.s`.
    fn handle_control(
        &mut self,
        track: usize,
        opcode: u8,
        ops: &[u8],
        events: &mut Vec<SynthEvent>,
    ) {
        let Some(t) = self.tracks.get_mut(track) else {
            return;
        };
        // Operand readers (signed/unsigned bytes, little-endian u16/duration).
        let s8 = |i: usize| ops.get(i).copied().unwrap_or(0) as i8 as i32;
        let u8r = |i: usize| i32::from(ops.get(i).copied().unwrap_or(0));
        let dur = |a: usize, b: usize| u8r(a) | (u8r(b) << 8); // u16 duration in driver ticks
        match opcode {
            0xD0 => t.tuning_8_8 = s8(0) << 8, // SetTuning (whole semitones, 8.8)
            0xD1 => t.tuning_8_8 += s8(0) << 8, // TuningDeltaCoarse
            0xD2 => t.tuning_8_8 += s8(0) << 2, // TuningDeltaFine (1/64 semitone steps)
            0xD3 => t.tuning_8_8 += (u8r(0) | (u8r(1) << 8)) as i16 as i32, // TuningDeltaFull
            0xD4 => t.tuning_fade.fade_to(s8(2) << 8, dur(0, 1)), // TuningFade -> op2 semitones
            0xD7 => t.key_bend = ((u8r(0) << 8) | u8r(1)) as i16 as i32, // SetKeyBend (BE signed)
            0xDB => t.key_bend_range = u8r(0) as u8, // SetKeyBendRange
            0xE1 => t.volume.set((t.volume.value() + s8(0)).clamp(0, 127)), // VolumeDelta
            0xE2 => t.volume.fade_to(s8(2).clamp(0, 127), dur(0, 1)), // VolumeFade
            0xE9 => t.pan.set((t.pan.value() + s8(0)).clamp(0, 127)), // PanDelta
            0xEA => t.pan.fade_to(u8r(2).clamp(0, 127), dur(0, 1)), // PanFade
            _ => {} // LFO setup/use opcodes are handled in `handle_lfo_control`
        }
        // A tuning or key-bend change re-detunes the whole track immediately; pan delta repositions it.
        if matches!(opcode, 0xD0..=0xD3 | 0xD7 | 0xDB) {
            events.push(SynthEvent::TrackDetune {
                track,
                semitones: t.bend_semitones(),
            });
        }
        if opcode == 0xE9 {
            t.last_pan = t.pan.value();
            events.push(SynthEvent::TrackPan {
                track,
                pan: f64::from(t.pan.value()) / 127.0,
            });
        }
        self.handle_lfo_control(track, opcode, ops, events);
    }

    /// Handles the LFO setup/use opcodes — vibrato (key-bend, slot 0), tremolo (volume, slot 1),
    /// auto-pan (slot 2), and the generic `SetupLfo`/`SetLfoParameter`/`UseLfo` family — by writing
    /// the channel's pending LFO config slots, mirroring the `DseTrackEvent_*Lfo*` handlers in
    /// `lib/DSE/asm/main_02071EB4.s`. The live LFOs are built per-note in [`Self::start_note`]
    /// (pitch/volume) or rebuilt here at the track level (pan).
    fn handle_lfo_control(
        &mut self,
        track: usize,
        opcode: u8,
        ops: &[u8],
        _events: &mut Vec<SynthEvent>,
    ) {
        let Some(t) = self.tracks.get_mut(track) else {
            return;
        };
        let u8r = |i: usize| ops.get(i).copied().unwrap_or(0);
        let idx = t.lfo_index.min(3);
        match opcode {
            // Dedicated key-bend (pitch), volume, pan LFOs: set up slots 0/1/2 with a hardwired dest.
            0xDC => lfo_setup_params(&mut t.lfo_slots[0], ops, Some(1)),
            0xE4 => lfo_setup_params(&mut t.lfo_slots[1], ops, Some(2)),
            0xEC => lfo_setup_params(&mut t.lfo_slots[2], ops, Some(3)),
            0xDD => lfo_setup_envelope(&mut t.lfo_slots[0], ops),
            0xE5 => lfo_setup_envelope(&mut t.lfo_slots[1], ops),
            0xED => lfo_setup_envelope(&mut t.lfo_slots[2], ops),
            0xDF => lfo_use(&mut t.lfo_slots[0], u8r(0), 1),
            0xE7 => lfo_use(&mut t.lfo_slots[1], u8r(0), 2),
            0xEF => lfo_use(&mut t.lfo_slots[2], u8r(0), 3),
            // Generic LFO ops act on the current slot (`channel + 0x61`); dest comes from `UseLfo`.
            0xF0 => lfo_setup_params(&mut t.lfo_slots[idx], ops, None),
            0xF1 => lfo_setup_envelope(&mut t.lfo_slots[idx], ops),
            0xF2 => lfo_set_parameter(t, u8r(0), u8r(1)),
            0xF3 => {
                let slot = usize::from(u8r(0)).min(3);
                t.lfo_index = slot;
                lfo_use(&mut t.lfo_slots[slot], u8r(1), u8r(2));
            }
            _ => return, // not an LFO opcode
        }
        // A pan-routed slot may have changed: rebuild the track-level auto-pan LFO. (Pitch/volume
        // LFOs are rebuilt per-note instead, so their opcodes don't disturb a running pan LFO.)
        if matches!(opcode, 0xEC | 0xED | 0xEF | 0xF0 | 0xF1 | 0xF2 | 0xF3) {
            t.pan_lfo = t
                .lfo_slots
                .iter()
                .filter_map(|c| Lfo::build(c, 127))
                .find(|l| l.dest == LfoDest::Pan);
        }
    }

    /// Resolves a note to a sample via the program/split tables and starts a voice.
    fn start_note(
        &mut self,
        track: usize,
        key: u8,
        velocity: u8,
        duration: u32,
        events: &mut Vec<SynthEvent>,
    ) {
        let program_id = self.tracks.get(track).map(|t| t.program).unwrap_or(0);
        let Some(program) = self.song_bank.program(program_id) else {
            return;
        };
        let Some(split) = program.resolve_split(key) else {
            return;
        };
        let wave_index = split.wave_index;
        let bend_sensitivity = split.bend_sensitivity;
        // Per-note volume: velocity * program.volume * split.volume / 127^2 (DseVoice_PlayNote).
        let note_volume = volume::note_volume(velocity, program.volume, split.volume);
        let env_params = EnvelopeParams::from_block(&split.envelope);

        // DSE pitch: the voice's `note_key` (8.8 fixed-point semitones) is built from the split
        // alone (`DseVoice_PlayNote`), then converted to an absolute PCM playback rate in Hz by
        // the driver's pitch tables (`DseVoice_UpdateParameters`). The WAVI `sample_rate` plays no
        // part — the timer drives playback at this rate regardless — so a `DataRateHz` voice
        // reproduces the hardware exactly. Channel tuning bends apply as a `TrackDetune`; the
        // pitch (key-bend) LFO is the per-voice vibrato built below.
        let note_key =
            i32::from(split.key_base) + (i32::from(split.note_delta) << 8) + (i32::from(key) << 8);

        let Some(sample) = self.sample(wave_index) else {
            return;
        };

        let env = SoundEnvelope::start(env_params);
        let voice = self.next_voice;
        self.next_voice += 1;

        let pitch = VoicePitch::DataRateHz(note_key_to_hz(note_key));

        // Build this note's vibrato/tremolo LFOs from the track's pitch/volume config slots. Each
        // note gets its own bank so the fade-in restarts per note, exactly as `DseVoice_PlayNote`.
        let lfos: Vec<Lfo> = self.tracks[track]
            .lfo_slots
            .iter()
            .filter_map(|cfg| Lfo::build(cfg, 127))
            .filter(|lfo| matches!(lfo.dest, LfoDest::Pitch | LfoDest::Volume))
            .collect();

        let v = DseVoice {
            id: voice,
            track,
            key,
            note_volume,
            env,
            release_tick: self.seq.ticks_elapsed + duration,
            released: false,
            lfos,
            last_detune: 0.0,
        };
        // Record this split's default bend range so an active pitch-wheel bend resolves with it,
        // re-detuning the track (and thus this new voice) to the correct bent pitch.
        self.tracks[track].split_bend_sensitivity = bend_sensitivity;
        if self.tracks[track].key_bend != 0 {
            let semitones = self.tracks[track].bend_semitones();
            events.push(SynthEvent::TrackDetune { track, semitones });
        }

        // Peek the starting level without consuming the voice's first tick.
        let t = &self.tracks[track];
        let vol_final = volume::volume_final(t.volume.value() as u8, t.expression);
        let initial =
            volume::voice_amp(v.env.clone().tick(), vol_final, note_volume) * DSE_MASTER_GAIN;

        events.push(SynthEvent::NoteStarted {
            track,
            voice,
            key,
            sample,
            pitch,
            volume: initial,
            duration_ticks: Some(duration),
        });
        self.voices.push(v);
    }

    /// Returns the decoded sample for a split's `wave_index`, decoding + caching on first use.
    ///
    /// Sample data is resolved through the **main bank's** WAVI, not the per-song bank's. The
    /// per-song `bgm####.swd` carries its own sparse WAVI listing the slots the song uses, but
    /// with *local* `pcm_offset`s (as if its samples were packed from zero) — they do not index
    /// the shared `bgm.swd` `pcmd`. Only the main bank's global WAVI has the correct offsets
    /// (root key / rate / loop are identical in both), so we look the slot up there.
    fn sample(&mut self, wave_index: i16) -> Option<Arc<Sample>> {
        if let Some(cached) = self.sample_cache.get(&wave_index) {
            return cached.clone();
        }
        let decoded = self
            .main_bank
            .sample_for_wave(wave_index)
            .and_then(|info: &SampleInfo| self.main_bank.decode_sample(info, &self.main_bank.pcmd))
            .map(Arc::new);
        self.sample_cache.insert(wave_index, decoded.clone());
        decoded
    }
}

/// Reads a little-endian `u16` from `ops[a], ops[b]` (0 past the end).
fn lfo_u16(ops: &[u8], a: usize, b: usize) -> u16 {
    u16::from(ops.get(a).copied().unwrap_or(0)) | (u16::from(ops.get(b).copied().unwrap_or(0)) << 8)
}

/// `Setup*Lfo` (5 operands): depth (op0,1), period in ms (op2,3), waveform (op4); clears the
/// fade-in envelope. The dedicated key-bend/volume/pan opcodes pass `dest = Some(1|2|3)` to also
/// enable + route the slot; the generic `SetupLfo` passes `None` (its dest comes from `UseLfo`).
fn lfo_setup_params(slot: &mut LfoConfig, ops: &[u8], dest: Option<u8>) {
    slot.depth = lfo_u16(ops, 0, 1) as i16;
    slot.period = lfo_u16(ops, 2, 3);
    slot.waveform = ops.get(4).copied().unwrap_or(0);
    slot.delay = 0;
    slot.fade = 0;
    if let Some(d) = dest {
        slot.enabled = 1;
        slot.dest = d;
    }
}

/// `Setup*LfoEnvelope` (4 operands): the fade-in delay (op0,1) and duration (op2,3), in ms.
fn lfo_setup_envelope(slot: &mut LfoConfig, ops: &[u8]) {
    slot.delay = lfo_u16(ops, 0, 1);
    slot.fade = lfo_u16(ops, 2, 3);
}

/// `Use*Lfo`: enable (`op == 2` ⇒ on) and route the slot to `dest`, or disable it (`op == 0`).
fn lfo_use(slot: &mut LfoConfig, op: u8, dest: u8) {
    slot.enabled = if op == 2 { 1 } else { op };
    slot.dest = if slot.enabled != 0 { dest } else { 0 };
}

/// `SetLfoParameter` (`DseTrackEvent_SetLfoParameter`): tweak one field of the current slot,
/// indexed by `param`, with the ROM's per-field scalings (depth scaled by the slot's dest).
fn lfo_set_parameter(t: &mut DseTrack, param: u8, value: u8) {
    if param == 1 {
        t.lfo_index = usize::from(value).min(3);
        return;
    }
    let slot = &mut t.lfo_slots[t.lfo_index.min(3)];
    let v = i32::from(value);
    match param {
        2 => slot.enabled = value,
        3 => slot.dest = value,
        4 => slot.waveform = value,
        5 => {
            let mult = match slot.dest {
                1 => 10,
                2 => -20,
                4 => 10,
                _ => 20, // dest 0/3 and anything else
            };
            slot.depth = (v * mult) as i16;
        }
        6 => slot.period = (v * 5) as u16,
        7 => slot.delay = (v * 20) as u16,
        8 => slot.delay = (slot.delay & 0xff00) | u16::from(value),
        9 => slot.delay = (slot.delay & 0x00ff) | (u16::from(value) << 8),
        10 => slot.fade = (v * 20) as u16,
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_volume_lfo_builds_a_volume_lfo() {
        // SetupVolumeLfo operands: depth 0x0040, period 200 ms, waveform 3 (full triangle), dest 2.
        let mut slot = LfoConfig::default();
        lfo_setup_params(&mut slot, &[0x40, 0x00, 0xC8, 0x00, 3], Some(2));
        assert_eq!(
            (slot.enabled, slot.dest, slot.depth, slot.period),
            (1, 2, 0x40, 200)
        );
        assert_eq!(Lfo::build(&slot, 127).unwrap().dest, LfoDest::Volume);
    }

    #[test]
    fn use_lfo_toggles_enable_and_dest() {
        let mut slot = LfoConfig {
            enabled: 1,
            dest: 1,
            period: 100,
            depth: 10,
            ..Default::default()
        };
        lfo_use(&mut slot, 0, 1); // op 0 → disable
        assert_eq!((slot.enabled, slot.dest), (0, 0));
        lfo_use(&mut slot, 2, 2); // op 2 → enable, route to volume
        assert_eq!((slot.enabled, slot.dest), (1, 2));
    }

    #[test]
    fn key_bend_matches_dse_channel_setkeybend() {
        // range * value / 8192 semitones (DseChannel_SetKeyBend).
        let mut t = DseTrack {
            key_bend_range: 12,
            key_bend: 8192,
            ..DseTrack::default()
        };
        assert!((t.bend_semitones() - 12.0).abs() < 1e-9); // full octave
        t.key_bend = -4096;
        assert!((t.bend_semitones() + 6.0).abs() < 1e-9);
        // Range 0 falls back to the split's bend_sensitivity captured at note-on.
        t.key_bend_range = 0;
        t.split_bend_sensitivity = 2;
        t.key_bend = 8192;
        assert!((t.bend_semitones() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn set_lfo_parameter_selects_slot_and_scales_fields() {
        let mut t = DseTrack::default();
        lfo_set_parameter(&mut t, 1, 2); // select slot 2
        assert_eq!(t.lfo_index, 2);
        lfo_set_parameter(&mut t, 6, 40); // period = value * 5
        assert_eq!(t.lfo_slots[2].period, 200);
        lfo_set_parameter(&mut t, 3, 1); // dest = pitch
        lfo_set_parameter(&mut t, 5, 4); // depth scaled ×10 for a pitch LFO
        assert_eq!(t.lfo_slots[2].depth, 40);
    }
}

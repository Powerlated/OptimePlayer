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
#[derive(Debug, Clone, Copy)]
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
}

impl DseTrack {
    /// Net pitch bend in semitones (base tuning + tuning fade), applied as a track detune.
    fn bend_semitones(&self) -> f64 {
        f64::from(self.tuning_8_8 + self.tuning_fade.value()) / 256.0
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
}

impl DsePlayer {
    /// Builds a player for `smdl` using `song_bank`'s programs and `main_bank`'s samples.
    pub fn new(smdl: &Smdl, song_bank: Arc<Swdl>, main_bank: Arc<Swdl>) -> DsePlayer {
        DsePlayer {
            seq: DseSequencer::new(smdl),
            main_bank,
            song_bank,
            sample_cache: HashMap::new(),
            tracks: [DseTrack::default(); TRACK_COUNT],
            voices: Vec::new(),
            accum_us: 0,
            next_voice: 0,
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

        // Step the per-track fades one driver tick; a pan/tuning change re-emits its track event,
        // while a volume change is picked up by the per-voice `VoiceVolume` below.
        for track in 0..self.tracks.len() {
            let t = &mut self.tracks[track];
            t.volume.tick();
            if t.pan.tick() {
                events.push(SynthEvent::TrackPan {
                    track,
                    pan: f64::from(t.pan.value()) / 127.0,
                });
            }
            if t.tuning_fade.tick() {
                events.push(SynthEvent::TrackDetune {
                    track,
                    semitones: t.bend_semitones(),
                });
            }
        }

        let tracks = self.tracks; // Copy snapshot, so the per-voice `&mut` below can't alias it.
        let mut remove = Vec::new();
        for idx in 0..self.voices.len() {
            let v = &mut self.voices[idx];

            if feedback.is_ended(v.track, v.id) {
                remove.push(idx);
                continue;
            }

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

            let t = tracks[v.track];
            let vol_final = volume::volume_final(t.volume.value() as u8, t.expression);
            let volume = volume::voice_amp(level, vol_final, v.note_volume) * DSE_MASTER_GAIN;
            events.push(SynthEvent::VoiceVolume {
                track: v.track,
                voice: v.id,
                volume,
            });
        }
        // Remove from the back so earlier indices stay valid.
        for &idx in remove.iter().rev() {
            self.voices.remove(idx);
        }
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
    fn handle_control(&mut self, track: usize, opcode: u8, ops: &[u8], events: &mut Vec<SynthEvent>) {
        let Some(t) = self.tracks.get_mut(track) else {
            return;
        };
        // Operand readers (signed/unsigned bytes, little-endian u16/duration).
        let s8 = |i: usize| ops.get(i).copied().unwrap_or(0) as i8 as i32;
        let u8r = |i: usize| i32::from(ops.get(i).copied().unwrap_or(0));
        let dur = |a: usize, b: usize| u8r(a) | (u8r(b) << 8); // u16 duration in driver ticks
        match opcode {
            0xD0 => t.tuning_8_8 = s8(0) << 8,        // SetTuning (whole semitones, 8.8)
            0xD1 => t.tuning_8_8 += s8(0) << 8,       // TuningDeltaCoarse
            0xD2 => t.tuning_8_8 += s8(0) << 2,       // TuningDeltaFine (1/64 semitone steps)
            0xD3 => t.tuning_8_8 += (u8r(0) | (u8r(1) << 8)) as i16 as i32, // TuningDeltaFull
            0xD4 => t.tuning_fade.fade_to(s8(2) << 8, dur(0, 1)), // TuningFade -> op2 semitones
            0xE1 => t.volume.set((t.volume.value() + s8(0)).clamp(0, 127)), // VolumeDelta
            0xE2 => t.volume.fade_to(s8(2).clamp(0, 127), dur(0, 1)),       // VolumeFade
            0xE9 => t.pan.set((t.pan.value() + s8(0)).clamp(0, 127)),       // PanDelta
            0xEA => t.pan.fade_to(u8r(2).clamp(0, 127), dur(0, 1)),         // PanFade
            _ => {} // LFO setup/use opcodes are handled in `handle_lfo_control`
        }
        // A tuning change re-detunes the whole track immediately; pan delta repositions it.
        if matches!(opcode, 0xD0..=0xD3) {
            events.push(SynthEvent::TrackDetune {
                track,
                semitones: t.bend_semitones(),
            });
        }
        if opcode == 0xE9 {
            events.push(SynthEvent::TrackPan {
                track,
                pan: f64::from(t.pan.value()) / 127.0,
            });
        }
        self.handle_lfo_control(track, opcode, ops, events);
    }

    /// Handles the LFO setup/use opcodes (vibrato / tremolo / auto-pan). Implemented below.
    fn handle_lfo_control(
        &mut self,
        _track: usize,
        _opcode: u8,
        _ops: &[u8],
        _events: &mut Vec<SynthEvent>,
    ) {
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
        // Per-note volume: velocity * program.volume * split.volume / 127^2 (DseVoice_PlayNote).
        let note_volume = volume::note_volume(velocity, program.volume, split.volume);
        let env_params = EnvelopeParams::from_block(&split.envelope);

        // DSE pitch: the voice's `note_key` (8.8 fixed-point semitones) is built from the split
        // alone (`DseVoice_PlayNote`), then converted to an absolute PCM playback rate in Hz by
        // the driver's pitch tables (`DseVoice_UpdateParameters`). The WAVI `sample_rate` plays no
        // part — the timer drives playback at this rate regardless — so a `DataRateHz` voice
        // reproduces the hardware exactly. (Channel bend / pitch LFO are not yet applied.)
        let note_key =
            i32::from(split.key_base) + (i32::from(split.note_delta) << 8) + (i32::from(key) << 8);

        let Some(sample) = self.sample(wave_index) else {
            return;
        };

        let env = SoundEnvelope::start(env_params);
        let voice = self.next_voice;
        self.next_voice += 1;

        let pitch = VoicePitch::DataRateHz(note_key_to_hz(note_key));

        let v = DseVoice {
            id: voice,
            track,
            key,
            note_volume,
            env,
            release_tick: self.seq.ticks_elapsed + duration,
            released: false,
        };
        // Peek the starting level without consuming the voice's first tick.
        let t = self.tracks[track];
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

//! [`GbaPlayer`]: the GBA device player. Drives the MP2K sequencer one VBlank frame per tick,
//! owns the DirectSound/CGB channel state and envelopes (transcribed from `SoundMainRAM` /
//! `CgbSound` in `pret/pokeemerald`), and emits standardized [`SynthEvent`]s.

use std::collections::HashMap;
use std::sync::Arc;

use super::rom::GbaRom;
use super::sequencer::{Mp2kOp, Mp2kSequencer, NoteOn};
use super::tables::{midi_key_to_cgb_freq, midi_key_to_freq};
use super::voice::{CgbKind, ToneKind, WaveData};
use crate::devices::{SynthEvent, TickFeedback, VoiceId, VoicePitch};
use crate::sample::Sample;
use crate::synth_controller::SynthConfig;

/// `MAX_DIRECTSOUND_CHANNELS` — we run the full hardware-struct count rather than the
/// game-configured `maxChans` (usually 5), so dense songs don't drop notes.
const MAX_DIRECTSOUND_CHANNELS: usize = 12;

/// `SOUND_MODE_MASVOL` value every Pokémon game passes to `m4aSoundMode`.
const MASTER_VOLUME: u32 = 12;

/// The software mixer rate (`SOUND_MODE_FREQ_13379`) — the playback rate of fixed-frequency
/// voices.
const ENGINE_RATE: f64 = 13379.0;

/// Linear gain of a full-scale CGB channel relative to the DirectSound scale.
const CGB_GAIN: f64 = 1.0;

/// DirectSound channel envelope phase (the `SOUND_CHANNEL_SF_ENV` states + pseudo-echo).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnvPhase {
    Attack,
    Decay,
    Sustain,
    Release,
    /// `SOUND_CHANNEL_SF_IEC`: the fixed-volume echo tail after release.
    PseudoEcho,
}

/// State shared by both channel kinds.
#[derive(Debug, Clone)]
struct ChannelCommon {
    voice: VoiceId,
    track: usize,
    /// Cleared when the track ends (`FINE`): the channel finishes its release unattended.
    linked: bool,
    /// Resolved key the channel sounds (post key-split/rhythm).
    key: u8,
    /// Key as played in the track data (`EOT` matches it; drives the visualizer).
    midi_key: u8,
    velocity: u8,
    priority: u8,
    /// Remaining gate time in sequencer steps; 0 = tie (no automatic release).
    gate: u8,
    /// `SOUND_CHANNEL_SF_STOP`: released.
    stop: bool,
    rhythm_pan: Option<i8>,
    /// `ChnVolSetAsm` outputs (velocity × pan × track volume).
    right_vol: u8,
    left_vol: u8,
    /// attack/decay/sustain/release.
    adsr: [u8; 4],
    echo_volume: u8,
    echo_length: u8,
}

/// One software-mixed PCM channel (`struct SoundChannel`).
#[derive(Debug, Clone)]
struct DirectSoundChannel {
    common: ChannelCommon,
    phase: EnvPhase,
    /// Envelope volume 0..=255.
    env: u8,
    /// WaveData `freq` field, for pitch recomputation.
    wav_freq: u32,
    fixed: bool,
}

/// One CGB legacy channel (`struct CgbChannel`).
#[derive(Debug, Clone)]
struct CgbChannel {
    common: ChannelCommon,
    kind: CgbKind,
    phase: EnvPhase,
    /// Whether the start-frame initialization has run (`SOUND_CHANNEL_SF_START`).
    started: bool,
    /// Envelope volume 0..=15.
    env: u8,
    env_goal: u8,
    sustain_goal: u8,
    env_counter: u8,
}

/// The GBA device player. One [`tick`](Self::tick) is one VBlank frame (≈59.73 Hz).
pub struct GbaPlayer {
    rom: Arc<[u8]>,
    seq: Mp2kSequencer,
    ds_channels: Vec<Option<DirectSoundChannel>>,
    cgb_channels: [Option<CgbChannel>; 4],
    /// DirectSound sample cache: wave address → decoded sample (`None` caches failures).
    sample_cache: HashMap<u32, Option<Arc<Sample>>>,
    /// CGB programmable-wave cache.
    wave_cache: HashMap<u32, Arc<Sample>>,
    square_samples: [Arc<Sample>; 4],
    noise_samples: [Arc<Sample>; 2],
    ops: Vec<Mp2kOp>,
    next_voice: VoiceId,
    /// `SoundInfo::c15`: every 15th frame the CGB envelope steps twice.
    c15: u8,
}

impl GbaPlayer {
    /// Binds song `song_id` from `rom`. Returns `None` for empty or invalid songs.
    pub fn new(rom: &GbaRom, song_id: u32) -> Option<GbaPlayer> {
        let header = rom.song_header(song_id)?;
        let seq = Mp2kSequencer::new(rom.data.clone(), &header);
        Some(GbaPlayer {
            rom: rom.data.clone(),
            seq,
            ds_channels: vec![None; MAX_DIRECTSOUND_CHANNELS],
            cgb_channels: [None, None, None, None],
            sample_cache: HashMap::new(),
            wave_cache: HashMap::new(),
            square_samples: build_square_samples(),
            noise_samples: build_noise_samples(),
            ops: Vec::new(),
            next_voice: 0,
            c15: 0,
        })
    }

    /// Sequencer steps executed (the visualizer timeline).
    pub fn steps_elapsed(&self) -> u32 {
        self.seq.steps
    }

    /// Sequencer steps per second at the current tempo.
    pub fn step_rate(&self) -> f64 {
        let frame_rate = super::GBA_CLOCK_RATE as f64 / super::CYCLES_PER_FRAME as f64;
        frame_rate * self.seq.steps_per_frame()
    }

    /// One VBlank frame: sequencer steps → channel ops → track volume/pitch refresh →
    /// envelope frame (mirroring the hardware's MPlayMain-then-SoundMain order).
    pub fn tick(
        &mut self,
        feedback: &mut TickFeedback,
        _config: &SynthConfig,
        events: &mut Vec<SynthEvent>,
    ) {
        // Channels whose voices the synthesizer stopped (one-shot samples that ran out).
        for &(track, voice) in &feedback.ended_voices {
            for chan in self.ds_channels.iter_mut() {
                if chan
                    .as_ref()
                    .is_some_and(|c| c.common.voice == voice && c.common.track == track)
                {
                    *chan = None;
                }
            }
            for chan in self.cgb_channels.iter_mut() {
                if chan
                    .as_ref()
                    .is_some_and(|c| c.common.voice == voice && c.common.track == track)
                {
                    *chan = None;
                }
            }
        }
        feedback.ended_voices.clear();

        let mut ops = std::mem::take(&mut self.ops);
        ops.clear();
        self.seq.tick_frame(&mut ops);
        for op in ops.drain(..) {
            self.apply_op(op, events);
        }
        self.ops = ops;

        self.refresh_changed_tracks(events);
        self.envelope_frame(events);
    }

    /// Applies one sequencer op to the channel state.
    fn apply_op(&mut self, op: Mp2kOp, events: &mut Vec<SynthEvent>) {
        match op {
            Mp2kOp::GateTick { track } => self.gate_tick(track, events),
            Mp2kOp::Note { track, note } => self.start_note(track, note, events),
            Mp2kOp::EndTie { track, key } => self.end_tie(track, key, events),
            Mp2kOp::TrackEnded { track } => self.release_track(track, events),
            Mp2kOp::Looped => events.push(SynthEvent::Looped),
            Mp2kOp::Finished => events.push(SynthEvent::Ended),
        }
    }

    /// Counts down gate timers on `track`'s channels; expired gates begin the release.
    fn gate_tick(&mut self, track: usize, events: &mut Vec<SynthEvent>) {
        let mut release = |common: &mut ChannelCommon| {
            if common.linked && common.track == track && !common.stop && common.gate != 0 {
                common.gate -= 1;
                if common.gate == 0 {
                    common.stop = true;
                    events.push(SynthEvent::NoteReleased {
                        track,
                        key: common.midi_key,
                        keyboard: false,
                    });
                }
            }
        };
        for chan in self.ds_channels.iter_mut().flatten() {
            release(&mut chan.common);
        }
        for chan in self.cgb_channels.iter_mut().flatten() {
            release(&mut chan.common);
        }
    }

    /// `EOT`: releases the first still-held channel of `track` playing `key`.
    fn end_tie(&mut self, track: usize, key: u8, events: &mut Vec<SynthEvent>) {
        let matching =
            |c: &ChannelCommon| c.linked && c.track == track && !c.stop && c.midi_key == key;
        for chan in self.ds_channels.iter_mut().flatten() {
            if matching(&chan.common) {
                chan.common.stop = true;
                events.push(SynthEvent::NoteReleased {
                    track,
                    key,
                    keyboard: false,
                });
                return;
            }
        }
        for chan in self.cgb_channels.iter_mut().flatten() {
            if matching(&chan.common) {
                chan.common.stop = true;
                events.push(SynthEvent::NoteReleased {
                    track,
                    key,
                    keyboard: false,
                });
                return;
            }
        }
    }

    /// `FINE`: every channel of the track is released and orphaned.
    fn release_track(&mut self, track: usize, events: &mut Vec<SynthEvent>) {
        let mut orphan = |common: &mut ChannelCommon| {
            if common.linked && common.track == track {
                common.linked = false;
                if !common.stop {
                    common.stop = true;
                    events.push(SynthEvent::NoteReleased {
                        track,
                        key: common.midi_key,
                        keyboard: false,
                    });
                }
            }
        };
        for chan in self.ds_channels.iter_mut().flatten() {
            orphan(&mut chan.common);
        }
        for chan in self.cgb_channels.iter_mut().flatten() {
            orphan(&mut chan.common);
        }
    }

    /// `ply_note`'s channel-allocation half.
    fn start_note(&mut self, track: usize, note: NoteOn, events: &mut Vec<SynthEvent>) {
        let lfo_delay = self.seq.tracks[track].lfo_delay;
        let echo_volume = self.seq.tracks[track].echo_volume;
        let echo_length = self.seq.tracks[track].echo_length;

        let sample = match note.tone.kind {
            ToneKind::DirectSound { wav_addr, .. } => self.direct_sound_sample(wav_addr),
            ToneKind::Cgb(kind) => self.cgb_sample(kind),
        };
        let Some(sample) = sample else {
            return;
        };

        let voice = self.next_voice;
        self.next_voice += 1;

        // Allocate the hardware channel (may steal; may fail → the note is dropped).
        let common = |key: u8, right_vol: u8, left_vol: u8| ChannelCommon {
            voice,
            track,
            linked: true,
            key,
            midi_key: note.midi_key,
            velocity: note.velocity,
            priority: note.priority,
            gate: note.gate,
            stop: false,
            rhythm_pan: note.tone.rhythm_pan,
            right_vol,
            left_vol,
            adsr: note.tone.adsr,
            echo_volume,
            echo_length,
        };

        match note.tone.kind {
            ToneKind::DirectSound { wav_addr, fixed } => {
                let Some(slot) = self.alloc_direct_sound(note.priority, track) else {
                    return;
                };
                if let Some(old) = self.ds_channels[slot].take() {
                    events.push(SynthEvent::VoiceStopped {
                        track: old.common.track,
                        voice: old.common.voice,
                    });
                }
                self.note_track_setup(track, lfo_delay);
                let tr = &self.seq.tracks[track];
                let (right_vol, left_vol) =
                    chn_vol_set(note.velocity, note.tone.rhythm_pan, tr.vol_mr, tr.vol_ml);
                let key2 = add_key(note.tone.key, tr.key_m);
                let wav_freq = WaveData::read(&self.rom, wav_addr).map_or(0, |w| w.freq);
                let rate = if fixed {
                    ENGINE_RATE
                } else {
                    f64::from(midi_key_to_freq(wav_freq, key2, tr.pit_m))
                };
                self.ds_channels[slot] = Some(DirectSoundChannel {
                    common: common(note.tone.key, right_vol, left_vol),
                    phase: EnvPhase::Attack,
                    env: 0,
                    wav_freq,
                    fixed,
                });
                events.push(SynthEvent::NoteStarted {
                    track,
                    voice,
                    key: note.midi_key,
                    keyboard: false,
                    sample,
                    pitch: VoicePitch::DataRateHz(rate),
                    volume: 0.0,
                    duration_ticks: (note.gate > 0).then_some(u32::from(note.gate)),
                });
            }
            ToneKind::Cgb(kind) => {
                let slot = (kind.channel_num() - 1) as usize;
                if let Some(existing) = &self.cgb_channels[slot] {
                    // Priority rules: a sounding, unreleased, higher-priority channel wins;
                    // ties go to the earlier track.
                    if !existing.common.stop
                        && (existing.common.priority > note.priority
                            || (existing.common.priority == note.priority
                                && existing.common.track < track))
                    {
                        return;
                    }
                }
                if let Some(old) = self.cgb_channels[slot].take() {
                    events.push(SynthEvent::VoiceStopped {
                        track: old.common.track,
                        voice: old.common.voice,
                    });
                }
                self.note_track_setup(track, lfo_delay);
                let tr = &self.seq.tracks[track];
                let (right_vol, left_vol) =
                    chn_vol_set(note.velocity, note.tone.rhythm_pan, tr.vol_mr, tr.vol_ml);
                let key2 = add_key(note.tone.key, tr.key_m);
                let freq = midi_key_to_cgb_freq(kind.channel_num(), key2, tr.pit_m);
                self.cgb_channels[slot] = Some(CgbChannel {
                    common: common(note.tone.key, right_vol, left_vol),
                    kind,
                    phase: EnvPhase::Attack,
                    started: false,
                    env: 0,
                    env_goal: 0,
                    sustain_goal: 0,
                    env_counter: 0,
                });
                events.push(SynthEvent::NoteStarted {
                    track,
                    voice,
                    key: note.midi_key,
                    keyboard: false,
                    sample,
                    pitch: VoicePitch::DataRateHz(cgb_data_rate(kind, freq)),
                    volume: 0.0,
                    duration_ticks: (note.gate > 0).then_some(u32::from(note.gate)),
                });
            }
        }
    }

    /// Track-state side effects of a successful note-on (`ply_note` after allocation).
    fn note_track_setup(&mut self, track: usize, lfo_delay: u8) {
        let tr = &mut self.seq.tracks[track];
        tr.lfo_delay_c = lfo_delay;
        if lfo_delay != 0 {
            tr.clear_mod_m();
        }
        tr.vol_pit_set();
        tr.flags = Default::default();
    }

    /// The DirectSound channel allocator from `ply_note`: first free channel, else steal the
    /// lowest-priority one (releasing channels preferred; ties go to the latest track).
    fn alloc_direct_sound(&mut self, priority: u8, track: usize) -> Option<usize> {
        let mut best: Option<usize> = None;
        let mut best_priority = priority;
        let mut best_track = track;
        let mut found_releasing = false;

        for i in 0..self.ds_channels.len() {
            let Some(chan) = &self.ds_channels[i] else {
                return Some(i); // free channel
            };
            if chan.common.stop {
                if !found_releasing {
                    found_releasing = true;
                    best_priority = chan.common.priority;
                    best_track = chan.common.track;
                    best = Some(i);
                    continue;
                }
            } else if found_releasing {
                continue; // releasing channels are always preferred over held ones
            }
            if chan.common.priority < best_priority {
                best_priority = chan.common.priority;
                best_track = chan.common.track;
                best = Some(i);
            } else if chan.common.priority == best_priority && chan.common.track >= best_track {
                best_track = chan.common.track;
                best = Some(i);
            }
        }
        best
    }

    /// The end-of-frame `MPT_FLG_VOLCHG`/`PITCHG` pass: recompute track mixers and refresh
    /// every linked channel's volume scaling and pitch.
    fn refresh_changed_tracks(&mut self, events: &mut Vec<SynthEvent>) {
        for t in 0..self.seq.tracks.len() {
            let flags = self.seq.tracks[t].flags;
            if !flags.volume && !flags.pitch {
                continue;
            }
            self.seq.tracks[t].vol_pit_set();
            self.seq.tracks[t].flags = Default::default();
            let tr = &self.seq.tracks[t];

            if flags.volume {
                // The track's stereo position comes from the mixer-volume pair.
                let (mr, ml) = (f64::from(tr.vol_mr), f64::from(tr.vol_ml));
                let pan = if mr + ml > 0.0 { mr / (mr + ml) } else { 0.5 };
                events.push(SynthEvent::TrackPan { track: t, pan });
            }

            for chan in self.ds_channels.iter_mut().flatten() {
                if !chan.common.linked || chan.common.track != t {
                    continue;
                }
                if flags.volume {
                    let (r, l) = chn_vol_set(
                        chan.common.velocity,
                        chan.common.rhythm_pan,
                        tr.vol_mr,
                        tr.vol_ml,
                    );
                    chan.common.right_vol = r;
                    chan.common.left_vol = l;
                }
                if flags.pitch && !chan.fixed {
                    let key2 = add_key(chan.common.key, tr.key_m);
                    let rate = midi_key_to_freq(chan.wav_freq, key2, tr.pit_m);
                    events.push(SynthEvent::VoicePitch {
                        track: t,
                        voice: chan.common.voice,
                        pitch: VoicePitch::DataRateHz(f64::from(rate)),
                    });
                }
            }
            for chan in self.cgb_channels.iter_mut().flatten() {
                if !chan.common.linked || chan.common.track != t {
                    continue;
                }
                if flags.volume {
                    let (r, l) = chn_vol_set(
                        chan.common.velocity,
                        chan.common.rhythm_pan,
                        tr.vol_mr,
                        tr.vol_ml,
                    );
                    chan.common.right_vol = r;
                    chan.common.left_vol = l;
                }
                if flags.pitch {
                    let key2 = add_key(chan.common.key, tr.key_m);
                    let freq = midi_key_to_cgb_freq(chan.kind.channel_num(), key2, tr.pit_m);
                    events.push(SynthEvent::VoicePitch {
                        track: t,
                        voice: chan.common.voice,
                        pitch: VoicePitch::DataRateHz(cgb_data_rate(chan.kind, freq)),
                    });
                }
            }
        }
    }

    /// One envelope frame over every channel (`SoundMainRAM` + `CgbSound`).
    fn envelope_frame(&mut self, events: &mut Vec<SynthEvent>) {
        // CGB envelopes step twice every 15th frame to track the hardware 64 Hz rate.
        if self.c15 != 0 {
            self.c15 -= 1;
        } else {
            self.c15 = 14;
        }
        let double_step = self.c15 == 0;

        for chan in self.ds_channels.iter_mut() {
            let Some(c) = chan else { continue };
            if direct_sound_env_frame(c) {
                events.push(SynthEvent::VoiceStopped {
                    track: c.common.track,
                    voice: c.common.voice,
                });
                *chan = None;
                continue;
            }
            let c = chan.as_ref().expect("still present");
            // Master-volume scaling, then the per-side velocity/pan/track volumes
            // (`SoundMainRAM`'s `env · (masterVolume + 1) >> 4` then `· side >> 8`).
            let uvol = (u32::from(c.env) * (MASTER_VOLUME + 1)) >> 4;
            let env_r = (u32::from(c.common.right_vol) * uvol) >> 8;
            let env_l = (u32::from(c.common.left_vol) * uvol) >> 8;
            events.push(SynthEvent::VoiceVolume {
                track: c.common.track,
                voice: c.common.voice,
                volume: (env_l + env_r) as f64 / 512.0,
            });
        }

        for chan in self.cgb_channels.iter_mut() {
            let Some(c) = chan else { continue };
            let mut died = cgb_env_frame(c);
            // The pseudo-echo tail counts in frames, not envelope steps — never doubled.
            if !died && double_step && c.phase != EnvPhase::PseudoEcho {
                died = cgb_env_frame(c);
            }
            if died {
                events.push(SynthEvent::VoiceStopped {
                    track: c.common.track,
                    voice: c.common.voice,
                });
                *chan = None;
                continue;
            }
            let c = chan.as_ref().expect("still present");
            events.push(SynthEvent::VoiceVolume {
                track: c.common.track,
                voice: c.common.voice,
                volume: f64::from(c.env.min(15)) / 15.0 * CGB_GAIN,
            });
        }
    }

    /// Decodes (and caches) the DirectSound wave at `wav_addr`.
    fn direct_sound_sample(&mut self, wav_addr: u32) -> Option<Arc<Sample>> {
        if let Some(cached) = self.sample_cache.get(&wav_addr) {
            return cached.clone();
        }
        let sample = WaveData::read(&self.rom, wav_addr).map(|wav| {
            let raw = &self.rom[wav.data_offset..wav.data_offset + wav.size as usize];
            let data = crate::sample::decode_pcm8(raw);
            let mut sample = Sample::new(
                data,
                440.0,
                f64::from(wav.freq) / 1024.0,
                wav.looping,
                i64::from(wav.loop_start),
            );
            sample.sample_length = wav.size as usize;
            Arc::new(sample)
        });
        self.sample_cache.insert(wav_addr, sample.clone());
        sample
    }

    /// Fetches the waveform for a CGB voice.
    fn cgb_sample(&mut self, kind: CgbKind) -> Option<Arc<Sample>> {
        match kind {
            CgbKind::Square1 { duty, .. } | CgbKind::Square2 { duty } => {
                Some(self.square_samples[(duty & 3) as usize].clone())
            }
            CgbKind::Noise { period7 } => Some(self.noise_samples[usize::from(period7)].clone()),
            CgbKind::Wave { wave_addr } => {
                if let Some(cached) = self.wave_cache.get(&wave_addr) {
                    return Some(cached.clone());
                }
                let offset = super::rom::ptr_to_offset(wave_addr, self.rom.len())?;
                let bytes = self.rom.get(offset..offset + 16)?;
                let mut data = Vec::with_capacity(32);
                for &b in bytes {
                    for nibble in [b >> 4, b & 0xF] {
                        data.push((f32::from(nibble) - 7.5) / 7.5 * 0.5);
                    }
                }
                let mut sample = Sample::new(data, 440.0, 1.0, true, 0);
                sample.is_psg_square = true;
                let sample = Arc::new(sample);
                self.wave_cache.insert(wave_addr, sample.clone());
                Some(sample)
            }
        }
    }
}

/// `chan.key + track.key_m`, clamped at zero (`ply_note` / `MPlayMain`).
fn add_key(key: u8, key_m: i8) -> u8 {
    (i32::from(key) + i32::from(key_m)).max(0) as u8
}

/// `ChnVolSetAsm`: per-channel right/left volumes from velocity, rhythm pan, and the track
/// mixer volumes.
fn chn_vol_set(velocity: u8, rhythm_pan: Option<i8>, vol_mr: u8, vol_ml: u8) -> (u8, u8) {
    let pan = i32::from(rhythm_pan.unwrap_or(0));
    let right = ((128 + pan) * i32::from(velocity) * i32::from(vol_mr)) >> 14;
    let left = ((127 - pan) * i32::from(velocity) * i32::from(vol_ml)) >> 14;
    (right.min(255) as u8, left.min(255) as u8)
}

/// The data rate (sample values per second) of a CGB voice given its frequency-register value.
fn cgb_data_rate(kind: CgbKind, reg: u32) -> f64 {
    match kind {
        // Tone frequency 131072/(2048−x) Hz × 8 samples per duty period.
        CgbKind::Square1 { .. } | CgbKind::Square2 { .. } => {
            8.0 * 131072.0 / (2048.0 - reg.min(2047) as f64)
        }
        // 32-sample wave at 2097152/(2048−x) samples per second.
        CgbKind::Wave { .. } => 2097152.0 / (2048.0 - reg.min(2047) as f64),
        // NR43: clock = 524288 / r / 2^(s+1), where r=0 means r=0.5.
        CgbKind::Noise { .. } => {
            let r = (reg & 7) as f64;
            let s = (reg >> 4) & 0xF;
            let divisor = if r == 0.0 { 0.5 } else { r };
            524288.0 / divisor / f64::from(1u32 << (s + 1))
        }
    }
}

/// One `SoundMainRAM` envelope frame for a DirectSound channel. Returns `true` if the channel
/// shut off.
fn direct_sound_env_frame(c: &mut DirectSoundChannel) -> bool {
    let mut env = u32::from(c.env);
    match c.phase {
        EnvPhase::PseudoEcho => {
            c.common.echo_length = c.common.echo_length.wrapping_sub(1);
            if c.common.echo_length == 0 || c.common.echo_length > 0x80 {
                return true;
            }
        }
        _ if c.common.stop => {
            env = (env * u32::from(c.common.adsr[3])) >> 8;
            if env <= u32::from(c.common.echo_volume) {
                if c.common.echo_volume == 0 {
                    return true;
                }
                env = u32::from(c.common.echo_volume);
                c.phase = EnvPhase::PseudoEcho;
            }
        }
        EnvPhase::Decay => {
            env = (env * u32::from(c.common.adsr[1])) >> 8;
            let sustain = u32::from(c.common.adsr[2]);
            if env <= sustain {
                if sustain == 0 {
                    if c.common.echo_volume == 0 {
                        return true;
                    }
                    env = u32::from(c.common.echo_volume);
                    c.phase = EnvPhase::PseudoEcho;
                } else {
                    env = sustain;
                    c.phase = EnvPhase::Sustain;
                }
            }
        }
        EnvPhase::Attack => {
            env += u32::from(c.common.adsr[0]);
            if env >= 255 {
                env = 255;
                c.phase = EnvPhase::Decay;
            }
        }
        EnvPhase::Sustain | EnvPhase::Release => {}
    }
    c.env = env as u8;
    false
}

/// One `CgbSound` envelope step for a CGB channel. Returns `true` if the channel shut off.
fn cgb_env_frame(c: &mut CgbChannel) -> bool {
    if !c.started {
        // SOUND_CHANNEL_SF_START: initialize the envelope.
        c.started = true;
        cgb_mod_vol(c);
        c.env_counter = c.common.adsr[0];
        if c.common.adsr[0] != 0 {
            c.env = 0;
        } else {
            return cgb_decay_start(c);
        }
    } else if c.phase == EnvPhase::PseudoEcho {
        c.common.echo_length = c.common.echo_length.wrapping_sub(1);
        if c.common.echo_length == 0 || c.common.echo_length > 0x80 {
            return true;
        }
        return false;
    } else if c.common.stop && c.phase != EnvPhase::Release {
        c.phase = EnvPhase::Release;
        c.env_counter = c.common.adsr[3];
        if c.common.adsr[3] == 0 {
            return cgb_echo_start(c);
        }
    } else if c.env_counter == 0 {
        cgb_mod_vol(c);
        match c.phase {
            EnvPhase::Release => {
                c.env = c.env.wrapping_sub(1);
                if c.env == 0 || c.env > 0x80 {
                    return cgb_echo_start(c);
                }
                c.env_counter = c.common.adsr[3];
            }
            EnvPhase::Sustain => {
                c.env = c.sustain_goal;
                c.env_counter = 7;
            }
            EnvPhase::Decay => {
                c.env = c.env.wrapping_sub(1);
                if c.env <= c.sustain_goal || c.env > 0x80 {
                    if c.common.adsr[2] == 0 {
                        return cgb_echo_start(c);
                    }
                    c.phase = EnvPhase::Sustain;
                    c.env = c.sustain_goal;
                    c.env_counter = 7;
                } else {
                    c.env_counter = c.common.adsr[1];
                }
            }
            EnvPhase::Attack | EnvPhase::PseudoEcho => {
                c.env += 1;
                if c.env >= c.env_goal {
                    return cgb_decay_start(c);
                }
                c.env_counter = c.common.adsr[0];
            }
        }
    }
    c.env_counter = c.env_counter.wrapping_sub(1);
    false
}

/// `envelope_decay_start`. Returns `true` if the channel shut off.
fn cgb_decay_start(c: &mut CgbChannel) -> bool {
    c.phase = EnvPhase::Decay;
    c.env_counter = c.common.adsr[1];
    c.env = c.env_goal;
    if c.common.adsr[1] == 0 {
        // No decay: jump straight to sustain (or the echo tail at zero sustain).
        if c.common.adsr[2] == 0 {
            return cgb_echo_start(c);
        }
        c.phase = EnvPhase::Sustain;
        c.env = c.sustain_goal;
        c.env_counter = 7;
    }
    false
}

/// `envelope_pseudoecho_start`. Returns `true` if the channel shut off.
fn cgb_echo_start(c: &mut CgbChannel) -> bool {
    c.env = (((u32::from(c.env_goal) * u32::from(c.common.echo_volume)) + 0xFF) >> 8) as u8;
    if c.env == 0 || c.common.echo_length == 0 {
        return true;
    }
    c.phase = EnvPhase::PseudoEcho;
    false
}

/// `CgbModVol`: the envelope ceiling from the channel's left/right volumes.
fn cgb_mod_vol(c: &mut CgbChannel) {
    let sum = (u32::from(c.common.left_vol) + u32::from(c.common.right_vol)) / 16;
    c.env_goal = sum.min(15) as u8;
    c.sustain_goal = (((u32::from(c.env_goal) * u32::from(c.common.adsr[2])) + 15) >> 4) as u8;
}

/// The four GB square duty cycles as 8-sample loops.
fn build_square_samples() -> [Arc<Sample>; 4] {
    const DUTIES: [[f32; 8]; 4] = [
        [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5], // 12.5%
        [0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5],  // 25%
        [0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5],    // 50%
        [-0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5],      // 75%
    ];
    DUTIES.map(|duty| {
        let mut s = Sample::new(duty.to_vec(), 1.0, 1.0, true, 0);
        s.is_psg_square = true;
        Arc::new(s)
    })
}

/// The 15-bit and 7-bit LFSR noise sequences as looping ±0.5 sample data.
fn build_noise_samples() -> [Arc<Sample>; 2] {
    let generate = |seven_bit: bool| {
        let len = if seven_bit { 127 } else { 32767 };
        let mut lfsr: u16 = if seven_bit { 0x7F } else { 0x7FFF };
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            let bit = (lfsr ^ (lfsr >> 1)) & 1;
            lfsr >>= 1;
            if seven_bit {
                lfsr = (lfsr & !(1 << 6)) | (bit << 6);
            } else {
                lfsr |= bit << 14;
            }
            data.push(if lfsr & 1 != 0 { -0.5 } else { 0.5 });
        }
        let mut s = Sample::new(data, 1.0, 1.0, true, 0);
        s.is_psg_square = true;
        Arc::new(s)
    };
    [generate(false), generate(true)]
}

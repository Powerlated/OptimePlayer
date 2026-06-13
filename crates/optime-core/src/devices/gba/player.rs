//! [`GbaPlayer`]: the GBA device player. Drives the MP2K sequencer one VBlank frame per tick,
//! owns the DirectSound/CGB channel state and envelopes (transcribed from `SoundMainRAM` /
//! `CgbSound` in `pret/pokeemerald`), and emits standardized [`SynthEvent`]s.

use std::collections::HashMap;
use std::sync::Arc;

use super::rom::GbaRom;
use super::sequencer::{Mp2kOp, Mp2kSequencer, NoteOn};
use super::tables::{midi_key_to_cgb_freq, midi_key_to_freq};
use super::voice::{CgbKind, ToneKind, WaveData};
use super::ENGINE_RATE;
use crate::devices::{SynthEvent, TickFeedback, VoiceId, VoicePitch};
use crate::sample::Sample;
use crate::synth_controller::SynthConfig;

/// `MAX_DIRECTSOUND_CHANNELS` — we run the full hardware-struct count rather than the
/// game-configured `maxChans` (usually 5), so dense songs don't drop notes.
const MAX_DIRECTSOUND_CHANNELS: usize = 12;

/// `SOUND_MODE_MASVOL` value every Pokémon game passes to `m4aSoundMode`.
const MASTER_VOLUME: u32 = 12;

/// Linear gain of a full-scale CGB channel relative to the DirectSound scale.
///
/// On hardware the two paths meet at the 10-bit DAC: a full-volume DirectSound stream spans
/// roughly ±0x100 DAC units, while a single PSG channel at envelope 15 with the usual NR50
/// master peaks near ±0x20 — about 1/8 of the DS scale. Our square/wave/noise sample data is
/// ±0.5, so 0.25 here lands a full-scale CGB channel at ±0.125 of a full-scale DS voice.
const CGB_GAIN: f64 = 0.25;

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
    /// Whether the start-frame initialization has run (`SOUND_CHANNEL_SF_START`).
    started: bool,
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
                let Some(slot) = alloc_direct_sound(&self.ds_channels, note.priority, track) else {
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
                    started: false,
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

/// The DirectSound channel allocator from `ply_note` (`src/m4a_1.s`, `_081DDBEC`): the first
/// free channel wins outright; otherwise steal the lowest-priority channel, preferring
/// releasing ones (once a releasing channel is seen, held channels are no longer candidates),
/// with priority ties going to the latest track. A new note that loses every comparison is
/// dropped (`None`).
fn alloc_direct_sound(
    channels: &[Option<DirectSoundChannel>],
    priority: u8,
    track: usize,
) -> Option<usize> {
    let mut best: Option<usize> = None;
    let mut best_priority = priority;
    let mut best_track = track;
    let mut found_releasing = false;

    for (i, chan) in channels.iter().enumerate() {
        let Some(chan) = chan else {
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
    if !c.started {
        // SOUND_CHANNEL_SF_START: a channel released before its first frame never sounds.
        c.started = true;
        if c.common.stop {
            return true;
        }
        // Otherwise the start frame falls straight into the attack step below.
    }
    let mut env = u32::from(c.env);
    match c.phase {
        EnvPhase::PseudoEcho => {
            // `subs; bhi`: the channel survives only while the pre-decrement length exceeds 1.
            let length = c.common.echo_length;
            c.common.echo_length = length.wrapping_sub(1);
            if length <= 1 {
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
        // SOUND_CHANNEL_SF_START: a channel released before its first frame never sounds;
        // otherwise initialize the envelope.
        c.started = true;
        if c.common.stop {
            return true;
        }
        cgb_mod_vol(c);
        c.env_counter = c.common.adsr[0];
        if c.common.adsr[0] != 0 {
            c.env = 0;
        } else if let Some(died) = cgb_decay_start(c) {
            return died;
        }
    } else if c.phase == EnvPhase::PseudoEcho {
        // C: `if ((s8)(chan->pseudoEchoLength) <= 0)` after the decrement.
        c.common.echo_length = c.common.echo_length.wrapping_sub(1);
        if c.common.echo_length as i8 <= 0 {
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
                if c.env as i8 <= 0 {
                    return cgb_echo_start(c);
                }
                c.env_counter = c.common.adsr[3];
            }
            EnvPhase::Sustain => {
                c.env = c.sustain_goal;
                c.env_counter = 7;
            }
            EnvPhase::Decay => {
                // C compares both sides as s8.
                c.env = c.env.wrapping_sub(1);
                if c.env as i8 <= c.sustain_goal as i8 {
                    if let Some(died) = cgb_sustain_start(c) {
                        return died;
                    }
                } else {
                    c.env_counter = c.common.adsr[1];
                }
            }
            EnvPhase::Attack | EnvPhase::PseudoEcho => {
                c.env += 1;
                if c.env >= c.env_goal {
                    if let Some(died) = cgb_decay_start(c) {
                        return died;
                    }
                } else {
                    c.env_counter = c.common.adsr[0];
                }
            }
        }
    }
    c.env_counter = c.env_counter.wrapping_sub(1);
    false
}

/// `envelope_decay_start`: `Some(died)` ends the frame right here (the C code jumped to
/// `envelope_complete` / `oscillator_off`); `None` falls through to the counter decrement
/// (`envelope_step_complete`), which matters for the every-15th-frame double step.
fn cgb_decay_start(c: &mut CgbChannel) -> Option<bool> {
    c.phase = EnvPhase::Decay;
    c.env_counter = c.common.adsr[1];
    if c.env_counter != 0 {
        c.env = c.env_goal;
        None
    } else {
        cgb_sustain_start(c)
    }
}

/// `envelope_sustain_start`: same exit convention as [`cgb_decay_start`].
fn cgb_sustain_start(c: &mut CgbChannel) -> Option<bool> {
    if c.common.adsr[2] == 0 {
        Some(cgb_echo_start(c))
    } else {
        c.phase = EnvPhase::Sustain;
        c.env = c.sustain_goal;
        c.env_counter = 7;
        None
    }
}

/// `envelope_pseudoecho_start`. Returns `true` if the channel shut off. (A zero echo length
/// is caught on the *next* frame's IEC pass, exactly as in C.)
fn cgb_echo_start(c: &mut CgbChannel) -> bool {
    c.env = (((u32::from(c.env_goal) * u32::from(c.common.echo_volume)) + 0xFF) >> 8) as u8;
    if c.env == 0 {
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

#[cfg(test)]
mod tests {
    use super::*;

    // `SOUND_CHANNEL_SF_*` from `pret/pokeemerald` `include/gba/m4a_internal.h`.
    const SF_START: u8 = 0x80;
    const SF_STOP: u8 = 0x40;
    const SF_IEC: u8 = 0x04;
    const SF_ENV: u8 = 0x03;
    const SF_ENV_DECAY: u8 = 0x02;
    const SF_ENV_ATTACK: u8 = 0x03;

    /// Direct transcription of `ChnVolSetAsm` (`src/m4a_1.s`) used as the oracle.
    fn c_chn_vol_set_asm(velocity: u8, rhythm_pan: i8, vol_mr: u8, vol_ml: u8) -> (u8, u8) {
        let mut right =
            ((0x80 + i32::from(rhythm_pan)) * i32::from(velocity) * i32::from(vol_mr)) >> 14;
        if right > 0xFF {
            right = 0xFF;
        }
        let mut left =
            ((0x7F - i32::from(rhythm_pan)) * i32::from(velocity) * i32::from(vol_ml)) >> 14;
        if left > 0xFF {
            left = 0xFF;
        }
        (right as u8, left as u8)
    }

    /// The pan-law agreement between this device core and the SynthController.
    ///
    /// The hardware computes *per-side* channel envelopes (`env_l`, `env_r`). The controller's
    /// stereo stage is linear — `l = volume·(1−pan)`, `r = volume·pan` — so the device emits
    /// `volume = (env_l+env_r)/512` per voice and `pan = mr/(mr+ml)` per track
    /// (`refresh_changed_tracks`/`envelope_frame`). For the non-rhythm voices that share the
    /// track's mixer volumes, that composition must land on the hardware's own per-side values
    /// `env_l/512`, `env_r/512` (up to the integer quantization and the engine's inherent
    /// 127-left/128-right asymmetry). Rhythm voices with a fixed per-voice pan keep their pan
    /// inside the per-voice volume instead — the track pan is the agreed approximation there.
    #[test]
    fn pan_law_composition_matches_per_side_envelopes() {
        for velocity in [1u8, 64, 100, 127] {
            for vol_ml in [1u8, 30, 90, 178, 255] {
                for vol_mr in [1u8, 30, 90, 178, 255] {
                    for env in [40u8, 128, 255] {
                        // The device's per-voice emission (envelope_frame).
                        let (right_vol, left_vol) = chn_vol_set(velocity, None, vol_mr, vol_ml);
                        let uvol = (u32::from(env) * (MASTER_VOLUME + 1)) >> 4;
                        let env_r = (u32::from(right_vol) * uvol) >> 8;
                        let env_l = (u32::from(left_vol) * uvol) >> 8;
                        let volume = (env_l + env_r) as f64 / 512.0;
                        // The device's per-track pan (refresh_changed_tracks).
                        let (mr, ml) = (f64::from(vol_mr), f64::from(vol_ml));
                        let pan = mr / (mr + ml);
                        // The controller's linear stereo stage (SampleSynthesizer::apply_stereo).
                        let l = volume * (1.0 - pan);
                        let r = volume * pan;

                        let want_l = env_l as f64 / 512.0;
                        let want_r = env_r as f64 / 512.0;
                        let tol = 0.01;
                        assert!(
                            (l - want_l).abs() < tol && (r - want_r).abs() < tol,
                            "vel={velocity} ml={vol_ml} mr={vol_mr} env={env}: \
                             got ({l:.4}, {r:.4}), hardware ({want_l:.4}, {want_r:.4})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn chn_vol_set_matches_pokeemerald() {
        for velocity in [0u8, 1, 64, 100, 127, 255] {
            for pan in [-128i8, -64, -2, 0, 2, 63, 126] {
                for vol_mr in [0u8, 1, 90, 178, 255] {
                    for vol_ml in [0u8, 1, 90, 178, 255] {
                        let expect = c_chn_vol_set_asm(velocity, pan, vol_mr, vol_ml);
                        let got = chn_vol_set(velocity, Some(pan), vol_mr, vol_ml);
                        assert_eq!(
                            got, expect,
                            "velocity={velocity} pan={pan} volMR={vol_mr} volML={vol_ml}"
                        );
                    }
                }
            }
        }
        // A non-rhythm voice plays with rhythmPan 0.
        assert_eq!(
            chn_vol_set(100, None, 200, 200),
            c_chn_vol_set_asm(100, 0, 200, 200)
        );
    }

    /// A DirectSound channel in its note-on state (everything irrelevant zeroed).
    fn make_ds_channel(adsr: [u8; 4], echo_volume: u8, echo_length: u8) -> DirectSoundChannel {
        DirectSoundChannel {
            common: ChannelCommon {
                voice: 0,
                track: 0,
                linked: true,
                key: 60,
                midi_key: 60,
                velocity: 100,
                priority: 0,
                gate: 0,
                stop: false,
                rhythm_pan: None,
                right_vol: 0,
                left_vol: 0,
                adsr,
                echo_volume,
                echo_length,
            },
            phase: EnvPhase::Attack,
            started: false,
            env: 0,
            wav_freq: 0,
            fixed: false,
        }
    }

    /// The oracle's DirectSound channel: raw `statusFlags` state, as the hardware keeps it.
    struct CDsChannel {
        status: u8,
        env: u8,
        adsr: [u8; 4],
        echo_volume: u8,
        echo_length: u8,
    }

    /// Direct transcription of the `SoundMainRAM` envelope section (`src/m4a_1.s`,
    /// `_081DCF6A`..`_081DD006`). Returns `false` once the channel turns off.
    fn c_sound_main_ram_env(c: &mut CDsChannel) -> bool {
        let [attack, decay, sustain, release] = c.adsr;
        if c.status & SF_START != 0 {
            if c.status & SF_STOP != 0 {
                c.status = 0;
                return false;
            }
            // Start: status = ENV_ATTACK, env = 0, then fall into the attack step.
            c.status = SF_ENV_ATTACK;
            let env = u32::from(attack); // 0 + attack
            if env >= 0xFF {
                c.env = 0xFF;
                c.status -= 1;
            } else {
                c.env = env as u8;
            }
            return true;
        }
        let mut env = u32::from(c.env);
        if c.status & SF_IEC != 0 {
            // `subs r0, 1; strb; bhi` — survives only while the pre-decrement length > 1.
            let orig = c.echo_length;
            c.echo_length = orig.wrapping_sub(1);
            if orig <= 1 {
                c.status = 0;
                return false;
            }
        } else if c.status & SF_STOP != 0 {
            env = (env * u32::from(release)) >> 8;
            if env <= u32::from(c.echo_volume) {
                if c.echo_volume == 0 {
                    c.status = 0;
                    return false;
                }
                env = u32::from(c.echo_volume);
                c.status |= SF_IEC;
            }
        } else if c.status & SF_ENV == SF_ENV_DECAY {
            env = (env * u32::from(decay)) >> 8;
            if env <= u32::from(sustain) {
                env = u32::from(sustain);
                if sustain == 0 {
                    if c.echo_volume == 0 {
                        c.status = 0;
                        return false;
                    }
                    env = u32::from(c.echo_volume);
                    c.status |= SF_IEC;
                } else {
                    c.status -= 1;
                }
            }
        } else if c.status & SF_ENV == SF_ENV_ATTACK {
            env += u32::from(attack);
            if env >= 0xFF {
                env = 0xFF;
                c.status -= 1;
            }
        }
        c.env = env as u8;
        true
    }

    #[test]
    fn direct_sound_envelope_matches_pokeemerald() {
        for attack in [0u8, 1, 9, 80, 255] {
            for decay in [0u8, 128, 235, 255] {
                for sustain in [0u8, 77, 255] {
                    for release in [0u8, 128, 245] {
                        for (echo_volume, echo_length) in [
                            (0u8, 0u8),
                            (40, 0),
                            (40, 1),
                            (40, 5),
                            (40, 0x81),
                            (40, 0xC0),
                        ] {
                            for release_frame in [0u32, 1, 3, 25] {
                                ds_envelope_scenario(
                                    [attack, decay, sustain, release],
                                    echo_volume,
                                    echo_length,
                                    release_frame,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Runs one note: started, released at `release_frame`, followed for 600 frames.
    fn ds_envelope_scenario(adsr: [u8; 4], echo_volume: u8, echo_length: u8, release_frame: u32) {
        let label = format!(
            "adsr={adsr:?} echo=({echo_volume},{echo_length}) release_frame={release_frame}"
        );
        let mut ours = make_ds_channel(adsr, echo_volume, echo_length);
        let mut oracle = CDsChannel {
            status: SF_START,
            env: 0,
            adsr,
            echo_volume,
            echo_length,
        };
        for frame in 0..600u32 {
            if frame == release_frame {
                ours.common.stop = true;
                oracle.status |= SF_STOP;
            }
            let ours_alive = !direct_sound_env_frame(&mut ours);
            let oracle_alive = c_sound_main_ram_env(&mut oracle);
            assert_eq!(
                ours_alive, oracle_alive,
                "{label}: aliveness at frame {frame}"
            );
            if !ours_alive {
                return;
            }
            assert_eq!(ours.env, oracle.env, "{label}: env at frame {frame}");
        }
    }

    /// A CGB channel in its note-on state.
    fn make_cgb_channel(
        adsr: [u8; 4],
        echo_volume: u8,
        echo_length: u8,
        left_vol: u8,
        right_vol: u8,
    ) -> CgbChannel {
        CgbChannel {
            common: ChannelCommon {
                voice: 0,
                track: 0,
                linked: true,
                key: 60,
                midi_key: 60,
                velocity: 100,
                priority: 0,
                gate: 0,
                stop: false,
                rhythm_pan: None,
                right_vol,
                left_vol,
                adsr,
                echo_volume,
                echo_length,
            },
            kind: CgbKind::Square2 { duty: 2 },
            phase: EnvPhase::Attack,
            started: false,
            env: 0,
            env_goal: 0,
            sustain_goal: 0,
            env_counter: 0,
        }
    }

    /// The oracle's CGB channel: raw `statusFlags` state.
    struct CCgbChannel {
        status: u8,
        env: u8,
        env_counter: u8,
        env_goal: u8,
        sustain_goal: u8,
        adsr: [u8; 4],
        echo_volume: u8,
        echo_length: u8,
        left_vol: u8,
        right_vol: u8,
    }

    /// `CgbModVol` (`src/m4a.c`), stereo branch (we don't model the GB pan mask).
    fn c_cgb_mod_vol(c: &mut CCgbChannel) {
        let mut goal = (u32::from(c.left_vol) + u32::from(c.right_vol)) / 16;
        if goal > 15 {
            goal = 15;
        }
        c.env_goal = goal as u8;
        c.sustain_goal = ((goal * u32::from(c.adsr[2]) + 15) >> 4) as u8;
    }

    /// `envelope_pseudoecho_start`: `Some(alive)` ends the frame (envelope_complete /
    /// oscillator_off); the caller must return it.
    fn c_pseudoecho_start(c: &mut CCgbChannel) -> Option<bool> {
        c.env = ((u32::from(c.env_goal) * u32::from(c.echo_volume) + 0xFF) >> 8) as u8;
        if c.env != 0 {
            c.status |= SF_IEC;
            Some(true)
        } else {
            c.status = 0;
            Some(false)
        }
    }

    /// `envelope_sustain_start`. `None` falls through to envelope_step_complete.
    fn c_sustain_start(c: &mut CCgbChannel) -> Option<bool> {
        if c.adsr[2] == 0 {
            c.status &= !SF_ENV;
            c_pseudoecho_start(c)
        } else {
            c.status -= 1;
            c.env = c.sustain_goal;
            c.env_counter = 7;
            None
        }
    }

    /// `envelope_decay_start`. `None` falls through to envelope_step_complete.
    fn c_decay_start(c: &mut CCgbChannel) -> Option<bool> {
        c.status -= 1;
        c.env_counter = c.adsr[1];
        if c.env_counter != 0 {
            c.env = c.env_goal;
            None
        } else {
            c_sustain_start(c)
        }
    }

    /// Direct transcription of the `CgbSound` envelope section (`src/m4a.c`) for one frame of
    /// one channel. `c15_zero` is `soundInfo->c15 == 0` (the double-step frame). Returns
    /// `false` once the channel turns off.
    fn c_cgb_sound_env(c: &mut CCgbChannel, c15_zero: bool) -> bool {
        let mut prev_c15: i32 = if c15_zero { 0 } else { 1 };

        // The pre-step branches; `step` says whether control reached envelope_step_repeat
        // (true) or envelope_step_complete (false).
        let step;
        if c.status & SF_START != 0 {
            if c.status & SF_STOP != 0 {
                c.status = 0;
                return false;
            }
            c.status = SF_ENV_ATTACK;
            c_cgb_mod_vol(c);
            c.env_counter = c.adsr[0];
            if c.adsr[0] as i8 != 0 {
                c.env = 0;
                step = false;
            } else {
                match c_decay_start(c) {
                    Some(alive) => return alive,
                    None => step = false,
                }
            }
        } else if c.status & SF_IEC != 0 {
            c.echo_length = c.echo_length.wrapping_sub(1);
            if c.echo_length as i8 <= 0 {
                c.status = 0;
                return false;
            }
            return true; // envelope_complete: no step, no double-step
        } else if c.status & SF_STOP != 0 && c.status & SF_ENV != 0 {
            c.status &= !SF_ENV;
            c.env_counter = c.adsr[3];
            if c.adsr[3] as i8 != 0 {
                step = false;
            } else {
                match c_pseudoecho_start(c) {
                    Some(alive) => return alive,
                    None => unreachable!(),
                }
            }
        } else {
            step = true; // straight into envelope_step_repeat
        }

        let mut do_step = step;
        loop {
            if do_step && c.env_counter == 0 {
                c_cgb_mod_vol(c);
                let exit = match c.status & SF_ENV {
                    0 => {
                        // RELEASE
                        c.env = c.env.wrapping_sub(1);
                        if c.env as i8 <= 0 {
                            c_pseudoecho_start(c)
                        } else {
                            c.env_counter = c.adsr[3];
                            None
                        }
                    }
                    1 => {
                        // SUSTAIN
                        c.env = c.sustain_goal;
                        c.env_counter = 7;
                        None
                    }
                    2 => {
                        // DECAY (both sides compared as s8)
                        c.env = c.env.wrapping_sub(1);
                        if c.env as i8 <= c.sustain_goal as i8 {
                            c_sustain_start(c)
                        } else {
                            c.env_counter = c.adsr[1];
                            None
                        }
                    }
                    _ => {
                        // ATTACK
                        c.env = c.env.wrapping_add(1);
                        if c.env >= c.env_goal {
                            c_decay_start(c)
                        } else {
                            c.env_counter = c.adsr[0];
                            None
                        }
                    }
                };
                if let Some(alive) = exit {
                    return alive;
                }
            }
            // envelope_step_complete:
            c.env_counter = c.env_counter.wrapping_sub(1);
            if prev_c15 == 0 {
                prev_c15 -= 1;
                do_step = true;
                continue;
            }
            return true;
        }
    }

    #[test]
    fn cgb_envelope_matches_pokeemerald() {
        for attack in [0u8, 1, 3] {
            for decay in [0u8, 1, 5] {
                for sustain in [0u8, 8, 15] {
                    for release in [0u8, 1, 4] {
                        for (echo_volume, echo_length) in
                            [(0u8, 0u8), (200, 0), (200, 3), (200, 0x81)]
                        {
                            for (left, right) in [(0u8, 0u8), (60, 80), (255, 255)] {
                                for release_frame in [0u32, 1, 7, 40] {
                                    cgb_envelope_scenario(
                                        [attack, decay, sustain, release],
                                        echo_volume,
                                        echo_length,
                                        left,
                                        right,
                                        release_frame,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Runs one CGB note through both machines, including the every-15th-frame double step.
    fn cgb_envelope_scenario(
        adsr: [u8; 4],
        echo_volume: u8,
        echo_length: u8,
        left_vol: u8,
        right_vol: u8,
        release_frame: u32,
    ) {
        let label = format!(
            "adsr={adsr:?} echo=({echo_volume},{echo_length}) vol=({left_vol},{right_vol}) \
             release_frame={release_frame}"
        );
        let mut ours = make_cgb_channel(adsr, echo_volume, echo_length, left_vol, right_vol);
        let mut oracle = CCgbChannel {
            status: SF_START,
            env: 0,
            env_counter: 0,
            env_goal: 0,
            sustain_goal: 0,
            adsr,
            echo_volume,
            echo_length,
            left_vol,
            right_vol,
        };
        // `SoundInfo::c15` starts at 0 and is updated at the top of every CgbSound call.
        let mut c15 = 0u8;
        for frame in 0..400u32 {
            if frame == release_frame {
                ours.common.stop = true;
                oracle.status |= SF_STOP;
            }
            if c15 != 0 {
                c15 -= 1;
            } else {
                c15 = 14;
            }
            let double_step = c15 == 0;

            // Ours, as `GbaPlayer::envelope_frame` drives it.
            let mut died = cgb_env_frame(&mut ours);
            if !died && double_step && ours.phase != EnvPhase::PseudoEcho {
                died = cgb_env_frame(&mut ours);
            }
            let ours_alive = !died;

            let oracle_alive = c_cgb_sound_env(&mut oracle, double_step);
            assert_eq!(
                ours_alive, oracle_alive,
                "{label}: aliveness at frame {frame}"
            );
            if !ours_alive {
                return;
            }
            assert_eq!(ours.env, oracle.env, "{label}: env at frame {frame}");
            assert_eq!(
                ours.env_counter, oracle.env_counter,
                "{label}: counter at frame {frame}"
            );
        }
    }

    #[test]
    fn cgb_mod_vol_matches_pokeemerald() {
        for left in [0u8, 1, 60, 127, 255] {
            for right in [0u8, 1, 80, 127, 255] {
                for sustain in [0u8, 1, 8, 15] {
                    let mut ours = make_cgb_channel([0, 0, sustain, 0], 0, 0, left, right);
                    cgb_mod_vol(&mut ours);
                    let mut oracle = CCgbChannel {
                        status: 0,
                        env: 0,
                        env_counter: 0,
                        env_goal: 0,
                        sustain_goal: 0,
                        adsr: [0, 0, sustain, 0],
                        echo_volume: 0,
                        echo_length: 0,
                        left_vol: left,
                        right_vol: right,
                    };
                    c_cgb_mod_vol(&mut oracle);
                    assert_eq!(
                        (ours.env_goal, ours.sustain_goal),
                        (oracle.env_goal, oracle.sustain_goal),
                        "left={left} right={right} sustain={sustain}"
                    );
                }
            }
        }
    }
}

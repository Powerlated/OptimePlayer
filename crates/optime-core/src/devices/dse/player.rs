use std::collections::HashMap;
use std::sync::Arc;

use super::envelope::{EnvelopeParams, SoundEnvelope, USEC_PER_DRIVER_TICK};
use super::lfo::{Lfo, LfoConfig, LfoDest, LfoRng};
use super::pitch::note_key_to_hz;
use super::sequencer::{DseSequencer, SeqOp};
use super::swdl::Swdl;
use super::{Smdl, WaveformInfo, volume};
use crate::PerDeviceSettings;
use crate::TRACK_COUNT;
use crate::devices::{SynthEvent, TickFeedback, VoiceId, VoicePitch};
use crate::waveform::Waveform;

pub const DSE_CYCLES_PER_TICK: u64 = 64 * 5236;

const DSE_MASTER_GAIN: f64 = 0.8;

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

    fn value(&self) -> i32 {
        self.current >> 8
    }

    fn set(&mut self, value: i32) {
        self.current = value << 8;
        self.target = self.current;
        self.ticks = 0;
    }

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

#[derive(Debug, Clone)]
struct DseTrack {
    program: u16,
    volume: Ramp,
    expression: u8,
    pan: Ramp,
    tuning_8_8: i32,
    tuning_fade: Ramp,
    key_bend: i32,
    key_bend_range: u8,
    split_bend_sensitivity: u8,
    lfo_slots: [LfoConfig; 4],
    lfo_index: usize,
    pan_lfo: Option<Lfo>,
    last_pan: i32,
}

impl DseTrack {
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

struct DseVoice {
    id: VoiceId,
    track: usize,
    key: u8,
    note_volume: u8,
    env: SoundEnvelope,
    release_tick: u32,
    released: bool,
    lfos: Vec<Lfo>,
    last_detune: f64,
}

pub struct DsePlayer {
    seq: DseSequencer,
    main_bank: Arc<Swdl>,
    song_bank: Arc<Swdl>,
    waveform_cache: HashMap<i16, Option<Arc<Waveform>>>,
    tracks: [DseTrack; TRACK_COUNT],
    voices: Vec<DseVoice>,
    accum_us: i64,
    next_voice: VoiceId,
    lfo_rng: LfoRng,
}

impl DsePlayer {
    pub fn new(smdl: &Smdl, song_bank: Arc<Swdl>, main_bank: Arc<Swdl>) -> DsePlayer {
        DsePlayer {
            seq: DseSequencer::new(smdl),
            main_bank,
            song_bank,
            waveform_cache: HashMap::new(),
            tracks: std::array::from_fn(|_| DseTrack::default()),
            voices: Vec::new(),
            accum_us: 0,
            next_voice: 0,
            lfo_rng: LfoRng::default(),
        }
    }

    fn seq_tick_us(&self) -> i64 {
        let denom = i64::from(self.seq.bpm.max(1)) * i64::from(self.seq.tpqn);
        (60_000_000 / denom).max(1)
    }

    fn tick_impl(&mut self, feedback: &mut TickFeedback, events: &mut Vec<SynthEvent>) {
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

    fn update_voices(&mut self, feedback: &TickFeedback, events: &mut Vec<SynthEvent>) {
        let now = self.seq.ticks_elapsed;
        let mut rng = std::mem::take(&mut self.lfo_rng);

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
            let pan_mod = match &mut t.pan_lfo {
                Some(lfo) => lfo.tick(&mut rng),
                None => 0,
            };
            if t.pan_lfo.is_some() || pan_faded {
                let pan_idx = (t.pan.value() + (pan_mod >> 6)).clamp(0, 127);
                if pan_idx != t.last_pan {
                    t.last_pan = pan_idx;
                    let p = f64::from(pan_idx) / 127.0;
                    events.push(SynthEvent::TrackPan {
                        track,
                        pan_vol_l: 1.0 - p,
                        pan_vol_r: p,
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

            if !v.released && now >= v.release_tick {
                v.released = true;
                v.env.release();
                events.push(SynthEvent::NoteReleased {
                    track: v.track,
                    key: v.key,
                });
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

            let (mut pitch_mod, mut vol_mod) = (0i32, 0i32);
            for lfo in &mut v.lfos {
                let out = lfo.tick(&mut rng);
                match lfo.dest {
                    LfoDest::Pitch => pitch_mod += out,
                    LfoDest::Volume => vol_mod += out,
                    LfoDest::Pan => {}
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
        for &idx in remove.iter().rev() {
            self.voices.remove(idx);
        }
        self.lfo_rng = rng;
    }

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
                    let p = f64::from(pan) / 127.0;
                    events.push(SynthEvent::TrackPan {
                        track,
                        pan_vol_l: 1.0 - p,
                        pan_vol_r: p,
                    });
                }
            }
            SeqOp::Control {
                track,
                opcode,
                operands,
            } => self.handle_control(track, opcode, &operands, events),
            SeqOp::Tempo { .. } => {}
            SeqOp::Looped => events.push(SynthEvent::Looped),
            SeqOp::TrackEnded { .. } => {
                if self.seq.ended {
                    events.push(SynthEvent::Ended);
                }
            }
        }
    }

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
        let s8 = |i: usize| ops.get(i).copied().unwrap_or(0) as i8 as i32;
        let u8r = |i: usize| i32::from(ops.get(i).copied().unwrap_or(0));
        let dur = |a: usize, b: usize| u8r(a) | (u8r(b) << 8);
        match opcode {
            0xD0 => t.tuning_8_8 = s8(0) << 8,
            0xD1 => t.tuning_8_8 += s8(0) << 8,
            0xD2 => t.tuning_8_8 += s8(0) << 2,
            0xD3 => t.tuning_8_8 += (u8r(0) | (u8r(1) << 8)) as i16 as i32,
            0xD4 => t.tuning_fade.fade_to(s8(2) << 8, dur(0, 1)),
            0xD7 => t.key_bend = ((u8r(0) << 8) | u8r(1)) as i16 as i32,
            0xDB => t.key_bend_range = u8r(0) as u8,
            0xE1 => t.volume.set((t.volume.value() + s8(0)).clamp(0, 127)),
            0xE2 => t.volume.fade_to(s8(2).clamp(0, 127), dur(0, 1)),
            0xE9 => t.pan.set((t.pan.value() + s8(0)).clamp(0, 127)),
            0xEA => t.pan.fade_to(u8r(2).clamp(0, 127), dur(0, 1)),
            _ => {}
        }
        if matches!(opcode, 0xD0..=0xD3 | 0xD7 | 0xDB) {
            events.push(SynthEvent::TrackDetune {
                track,
                semitones: t.bend_semitones(),
            });
        }
        if opcode == 0xE9 {
            t.last_pan = t.pan.value();
            let p = f64::from(t.pan.value()) / 127.0;
            events.push(SynthEvent::TrackPan {
                track,
                pan_vol_l: 1.0 - p,
                pan_vol_r: p,
            });
        }
        self.handle_lfo_control(track, opcode, ops, events);
    }

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
            0xDC => lfo_setup_params(&mut t.lfo_slots[0], ops, Some(1)),
            0xE4 => lfo_setup_params(&mut t.lfo_slots[1], ops, Some(2)),
            0xEC => lfo_setup_params(&mut t.lfo_slots[2], ops, Some(3)),
            0xDD => lfo_setup_envelope(&mut t.lfo_slots[0], ops),
            0xE5 => lfo_setup_envelope(&mut t.lfo_slots[1], ops),
            0xED => lfo_setup_envelope(&mut t.lfo_slots[2], ops),
            0xDF => lfo_use(&mut t.lfo_slots[0], u8r(0), 1),
            0xE7 => lfo_use(&mut t.lfo_slots[1], u8r(0), 2),
            0xEF => lfo_use(&mut t.lfo_slots[2], u8r(0), 3),
            0xF0 => lfo_setup_params(&mut t.lfo_slots[idx], ops, None),
            0xF1 => lfo_setup_envelope(&mut t.lfo_slots[idx], ops),
            0xF2 => lfo_set_parameter(t, u8r(0), u8r(1)),
            0xF3 => {
                let slot = usize::from(u8r(0)).min(3);
                t.lfo_index = slot;
                lfo_use(&mut t.lfo_slots[slot], u8r(1), u8r(2));
            }
            _ => return,
        }
        if matches!(opcode, 0xEC | 0xED | 0xEF | 0xF0 | 0xF1 | 0xF2 | 0xF3) {
            t.pan_lfo = t
                .lfo_slots
                .iter()
                .filter_map(|c| Lfo::build(c, 127))
                .find(|l| l.dest == LfoDest::Pan);
        }
    }

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
        let note_volume = volume::note_volume(velocity, program.volume, split.volume);
        let env_params = EnvelopeParams::from_block(&split.envelope);

        let note_key =
            i32::from(split.key_base) + (i32::from(split.note_delta) << 8) + (i32::from(key) << 8);

        let Some(waveform) = self.waveform(wave_index) else {
            return;
        };

        let env = SoundEnvelope::start(env_params);
        let voice = self.next_voice;
        self.next_voice += 1;

        let pitch = VoicePitch::DataRateHz(note_key_to_hz(note_key));

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
        self.tracks[track].split_bend_sensitivity = bend_sensitivity;
        if self.tracks[track].key_bend != 0 {
            let semitones = self.tracks[track].bend_semitones();
            events.push(SynthEvent::TrackDetune { track, semitones });
        }

        let t = &self.tracks[track];
        let vol_final = volume::volume_final(t.volume.value() as u8, t.expression);
        let initial =
            volume::voice_amp(v.env.clone().tick(), vol_final, note_volume) * DSE_MASTER_GAIN;

        events.push(SynthEvent::NoteStarted {
            track,
            voice,
            key,
            waveform,
            pitch,
            volume: initial,
            duration_ticks: Some(duration),
        });
        self.voices.push(v);
    }

    fn waveform(&mut self, wave_index: i16) -> Option<Arc<Waveform>> {
        if let Some(cached) = self.waveform_cache.get(&wave_index) {
            return cached.clone();
        }
        let decoded = self
            .main_bank
            .waveform_for_wave(wave_index)
            .and_then(|info: &WaveformInfo| {
                self.main_bank.decode_waveform(info, &self.main_bank.pcmd)
            })
            .map(Arc::new);
        self.waveform_cache.insert(wave_index, decoded.clone());
        decoded
    }
}

fn lfo_u16(ops: &[u8], a: usize, b: usize) -> u16 {
    u16::from(ops.get(a).copied().unwrap_or(0)) | (u16::from(ops.get(b).copied().unwrap_or(0)) << 8)
}

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

fn lfo_setup_envelope(slot: &mut LfoConfig, ops: &[u8]) {
    slot.delay = lfo_u16(ops, 0, 1);
    slot.fade = lfo_u16(ops, 2, 3);
}

fn lfo_use(slot: &mut LfoConfig, op: u8, dest: u8) {
    slot.enabled = if op == 2 { 1 } else { op };
    slot.dest = if slot.enabled != 0 { dest } else { 0 };
}

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
                _ => 20,
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

impl crate::devices::DevicePlayer for DsePlayer {
    fn clock_rate(&self) -> f64 {
        crate::DS_CLOCK_RATE as f64
    }

    fn cycles_per_tick(&self) -> f64 {
        DSE_CYCLES_PER_TICK as f64
    }

    fn steps_elapsed(&self) -> u32 {
        self.seq.ticks_elapsed
    }

    fn step_rate(&self) -> f64 {
        f64::from(self.seq.bpm.max(1)) * f64::from(self.seq.tpqn) / 60.0
    }

    fn steps_per_beat(&self) -> f64 {
        48.0
    }

    fn tick(
        &mut self,
        feedback: &mut TickFeedback,
        _config: &PerDeviceSettings,
        events: &mut Vec<SynthEvent>,
    ) {
        self.tick_impl(feedback, events);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_volume_lfo_builds_a_volume_lfo() {
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
        lfo_use(&mut slot, 0, 1);
        assert_eq!((slot.enabled, slot.dest), (0, 0));
        lfo_use(&mut slot, 2, 2);
        assert_eq!((slot.enabled, slot.dest), (1, 2));
    }

    #[test]
    fn key_bend_matches_dse_channel_setkeybend() {
        let mut t = DseTrack {
            key_bend_range: 12,
            key_bend: 8192,
            ..DseTrack::default()
        };
        assert!((t.bend_semitones() - 12.0).abs() < 1e-9);
        t.key_bend = -4096;
        assert!((t.bend_semitones() + 6.0).abs() < 1e-9);
        t.key_bend_range = 0;
        t.split_bend_sensitivity = 2;
        t.key_bend = 8192;
        assert!((t.bend_semitones() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn set_lfo_parameter_selects_slot_and_scales_fields() {
        let mut t = DseTrack::default();
        lfo_set_parameter(&mut t, 1, 2);
        assert_eq!(t.lfo_index, 2);
        lfo_set_parameter(&mut t, 6, 40);
        assert_eq!(t.lfo_slots[2].period, 200);
        lfo_set_parameter(&mut t, 3, 1);
        lfo_set_parameter(&mut t, 5, 4);
        assert_eq!(t.lfo_slots[2].depth, 40);
    }
}

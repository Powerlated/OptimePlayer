//! [`GbaVoices`]: the synth-side glue between MP2K's hardware channel structs and the
//! [`SynthController`](crate::SynthController)'s voices.
//!
//! This has no reference-source origin — it is the bookkeeping Optime needs because it renders the
//! channels in software instead of feeding real DirectSound FIFOs and CGB registers: which voice a
//! hardware slot currently drives, the decoded/generated waveform behind it, and the `SynthEvent`s
//! that report the slot's state each frame.
//!
//! It is deliberately separate from the engine (`m4a`/`m4a_1`) and from whatever *drives* the
//! engine. Both drivers share it: [`GbaPlayer`](super::GbaPlayer) runs the song bytecode, while
//! [`ParamPlayer`](super::param_player::ParamPlayer) is driven by VST3 parameters and DAW notes.

use std::collections::HashMap;
use std::sync::Arc;

use crate::devices::{SynthEvent, TickFeedback, VoiceId, VoicePitch};
use crate::waveform::{Waveform, decode_pcm8};

use super::m4a::{
    self, CgbChannel, MusicPlayerTrack, SoundInfo, TONEDATA_TYPE_CGB, TONEDATA_TYPE_FIX,
};
use super::rom::ptr_to_offset;
use super::{CGB_FULL_SCALE_GAIN, DS_MIXER_FULL_SCALE, ENGINE_RATE, MAX_DS_CHANNELS};

/// The synth-side identity of a hardware channel slot: which voice it drives and what we last told
/// the [`SynthController`](crate::SynthController) about it.
#[derive(Debug, Clone, Copy)]
pub(super) struct SlotVoice {
    voice: VoiceId,
    track: usize,
    midi_key: u8,
    last_volume: f64,
    last_rate: f64,
    released: bool,
}

/// Per-slot voice bookkeeping, waveform caches, and the generated PSG sample data.
pub(super) struct GbaVoices {
    rom: Arc<[u8]>,
    /// Per-hardware-slot voice bookkeeping (DirectSound then the four CGB channels).
    ds_slots: [Option<SlotVoice>; MAX_DS_CHANNELS],
    cgb_slots: [Option<SlotVoice>; 4],
    next_voice: VoiceId,
    /// Last `(volMR, volML)` emitted as `TrackPan`, per track, so pan changes emit exactly once.
    last_pan: Vec<Option<(u8, u8)>>,
    /// DirectSound sample cache: (wave address, DC-removed?) → decoded waveform (`None` = failure).
    waveform_cache: HashMap<(u32, bool), Option<Arc<Waveform>>>,
    /// CGB programmable-wave cache.
    wave_cache: HashMap<u32, Arc<Waveform>>,
    square_waveforms: [Arc<Waveform>; 4],
    noise_waveforms: [Arc<Waveform>; 2],
    /// Whether to subtract each DirectSound sample's DC offset (refreshed from config each tick).
    remove_dc: bool,
}

impl GbaVoices {
    pub(super) fn new(rom: Arc<[u8]>, track_count: usize) -> GbaVoices {
        GbaVoices {
            rom,
            ds_slots: [None; MAX_DS_CHANNELS],
            cgb_slots: [None; 4],
            next_voice: 0,
            last_pan: vec![None; track_count],
            waveform_cache: HashMap::new(),
            wave_cache: HashMap::new(),
            square_waveforms: build_square_waveforms(),
            noise_waveforms: build_noise_waveforms(),
            remove_dc: false,
        }
    }

    pub(super) fn set_remove_dc(&mut self, remove_dc: bool) {
        self.remove_dc = remove_dc;
    }

    /// Reaps voices the synthesizer stopped on its own (one-shot samples that ran out): detaches the
    /// hardware channel so the engine stops driving it.
    pub(super) fn reap_ended(&mut self, si: &mut SoundInfo, feedback: &mut TickFeedback) {
        for &(track, voice) in &feedback.ended_voices {
            for i in 0..MAX_DS_CHANNELS {
                if self.ds_slots[i].is_some_and(|s| s.voice == voice && s.track == track) {
                    self.ds_slots[i] = None;
                    si.chans[i].statusFlags = 0;
                    si.chans[i].track = None;
                }
            }
            for i in 0..4 {
                if self.cgb_slots[i].is_some_and(|s| s.voice == voice && s.track == track) {
                    self.cgb_slots[i] = None;
                    si.cgbChans[i].statusFlags = 0;
                    si.cgbChans[i].track = None;
                }
            }
        }
        feedback.ended_voices.clear();
    }

    /// Emits `TrackPan` for any track whose mixer volumes changed this frame (the normalized
    /// left/right split; the per-voice volume carries the absolute level separately).
    pub(super) fn emit_track_pans(
        &mut self,
        tracks: &[MusicPlayerTrack],
        events: &mut Vec<SynthEvent>,
    ) {
        for (t, track) in tracks.iter().enumerate() {
            let (mr, ml) = (track.volMR, track.volML);
            if self.last_pan[t] == Some((mr, ml)) {
                continue;
            }
            self.last_pan[t] = Some((mr, ml));
            let (mrf, mlf) = (f64::from(mr), f64::from(ml));
            let (pan_vol_l, pan_vol_r) = if mrf + mlf > 0.0 {
                (mlf / (mrf + mlf), mrf / (mrf + mlf))
            } else {
                (0.5, 0.5)
            };
            events.push(SynthEvent::TrackPan {
                track: t,
                pan_vol_l,
                pan_vol_r,
            });
        }
    }

    /// Starts a voice for every channel `ply_note` just allocated (`SOUND_CHANNEL_SF_START` still
    /// set, before `SoundMain` clears it). Decodes/generates the waveform; a channel whose sample
    /// can't be produced (compressed/invalid) is silenced.
    pub(super) fn start_new_notes(&mut self, si: &mut SoundInfo, events: &mut Vec<SynthEvent>) {
        for i in 0..MAX_DS_CHANNELS {
            if si.chans[i].statusFlags & m4a::SOUND_CHANNEL_SF_START == 0 {
                continue;
            }
            let c = si.chans[i];
            if let Some(old) = self.ds_slots[i].take() {
                events.push(SynthEvent::VoiceStopped {
                    track: old.track,
                    voice: old.voice,
                });
            }
            let Some(waveform) = self.direct_sound_waveform(c.wav) else {
                si.chans[i].statusFlags = 0;
                si.chans[i].track = None;
                continue;
            };
            let rate = ds_rate(c.type_, c.frequency);
            let track = c.track.unwrap_or(0);
            let voice = self.next_voice;
            self.next_voice += 1;
            events.push(SynthEvent::NoteStarted {
                track,
                voice,
                key: c.midiKey,
                velocity: c.velocity,
                waveform,
                pitch: VoicePitch::DataRateHz(rate),
                volume: 0.0,
                duration_ticks: (c.gateTime > 0).then_some(u32::from(c.gateTime)),
            });
            self.ds_slots[i] = Some(SlotVoice {
                voice,
                track,
                midi_key: c.midiKey,
                last_volume: 0.0,
                last_rate: rate,
                released: false,
            });
        }

        for i in 0..4 {
            if si.cgbChans[i].statusFlags & m4a::SOUND_CHANNEL_SF_START == 0 {
                continue;
            }
            let c = si.cgbChans[i];
            if let Some(old) = self.cgb_slots[i].take() {
                events.push(SynthEvent::VoiceStopped {
                    track: old.track,
                    voice: old.voice,
                });
            }
            let Some(waveform) = self.cgb_waveform(&c) else {
                si.cgbChans[i].statusFlags = 0;
                si.cgbChans[i].track = None;
                continue;
            };
            let rate = cgb_data_rate(c.type_ & TONEDATA_TYPE_CGB, c.frequency);
            let track = c.track.unwrap_or(0);
            let voice = self.next_voice;
            self.next_voice += 1;
            events.push(SynthEvent::NoteStarted {
                track,
                voice,
                key: c.midiKey,
                velocity: c.velocity,
                waveform,
                pitch: VoicePitch::DataRateHz(rate),
                volume: 0.0,
                duration_ticks: (c.gateTime > 0).then_some(u32::from(c.gateTime)),
            });
            self.cgb_slots[i] = Some(SlotVoice {
                voice,
                track,
                midi_key: c.midiKey,
                last_volume: 0.0,
                last_rate: rate,
                released: false,
            });
        }
    }

    /// After `SoundMain`: emit per-voice volume/pitch changes, note releases, and stops for
    /// channels the envelope shut off.
    pub(super) fn emit_updates(&mut self, si: &SoundInfo, events: &mut Vec<SynthEvent>) {
        for i in 0..MAX_DS_CHANNELS {
            let Some(mut slot) = self.ds_slots[i] else {
                continue;
            };
            let c = si.chans[i];
            if c.statusFlags == 0 {
                events.push(SynthEvent::VoiceStopped {
                    track: slot.track,
                    voice: slot.voice,
                });
                self.ds_slots[i] = None;
                continue;
            }
            let volume = (f64::from(c.envelopeVolumeLeft) + f64::from(c.envelopeVolumeRight))
                / DS_MIXER_FULL_SCALE;
            let rate = ds_rate(c.type_, c.frequency);
            let released = c.statusFlags & m4a::SOUND_CHANNEL_SF_STOP != 0;
            update_voice(events, &mut slot, volume, rate, released);
            self.ds_slots[i] = Some(slot);
        }

        for i in 0..4 {
            let Some(mut slot) = self.cgb_slots[i] else {
                continue;
            };
            let c = si.cgbChans[i];
            if c.statusFlags == 0 {
                events.push(SynthEvent::VoiceStopped {
                    track: slot.track,
                    voice: slot.voice,
                });
                self.cgb_slots[i] = None;
                continue;
            }
            let volume = f64::from(c.envelopeVolume.min(15)) / 15.0 * CGB_FULL_SCALE_GAIN;
            let rate = cgb_data_rate(c.type_ & TONEDATA_TYPE_CGB, c.frequency);
            let released = c.statusFlags & m4a::SOUND_CHANNEL_SF_STOP != 0;
            update_voice(events, &mut slot, volume, rate, released);
            self.cgb_slots[i] = Some(slot);
        }
    }

    /// Decodes (and caches) the DirectSound wave at `wav_addr`.
    fn direct_sound_waveform(&mut self, wav_addr: u32) -> Option<Arc<Waveform>> {
        let key = (wav_addr, self.remove_dc);
        if let Some(cached) = self.waveform_cache.get(&key) {
            return cached.clone();
        }
        let remove_dc = self.remove_dc;
        let waveform = m4a::WaveData::read(&self.rom, wav_addr).map(|wav| {
            let raw = &self.rom[wav.data..wav.data + wav.size as usize];
            let mut data = decode_pcm8(raw);
            // Real DirectSound output is AC-coupled; opt-in DC removal stops the voice thumping by
            // its offset when the envelope opens/closes.
            if remove_dc {
                dc_center(&mut data);
            }
            let mut waveform = Waveform::new(
                data,
                440.0,
                f64::from(wav.freq) / 1024.0,
                wav.looping(),
                i64::from(wav.loopStart),
            );
            waveform.sample_length = wav.size as usize;
            Arc::new(waveform)
        });
        self.waveform_cache.insert(key, waveform.clone());
        waveform
    }

    /// Fetches/generates the waveform for a CGB voice (from its `type_` + `wavePointer`).
    fn cgb_waveform(&mut self, c: &CgbChannel) -> Option<Arc<Waveform>> {
        match c.type_ & TONEDATA_TYPE_CGB {
            // Square 1 / 2: duty select in the low bits of the tone's wave field.
            1 | 2 => Some(self.square_waveforms[(c.wavePointer & 3) as usize].clone()),
            // Programmable wave: 32 packed 4-bit samples in ROM.
            3 => {
                if let Some(cached) = self.wave_cache.get(&c.wavePointer) {
                    return Some(cached.clone());
                }
                let offset = ptr_to_offset(c.wavePointer, self.rom.len())?;
                let bytes = self.rom.get(offset..offset + 16)?;
                let mut data = Vec::with_capacity(32);
                for &b in bytes {
                    for nibble in [b >> 4, b & 0xF] {
                        data.push((f32::from(nibble) - 7.5) / 7.5 * 0.5);
                    }
                }
                dc_center(&mut data);
                let mut waveform = Waveform::new(data, 440.0, 1.0, true, 0);
                waveform.is_psg_square = true;
                let waveform = Arc::new(waveform);
                self.wave_cache.insert(c.wavePointer, waveform.clone());
                Some(waveform)
            }
            // Noise: short (7-bit) sequence selected by the low bit.
            _ => Some(self.noise_waveforms[usize::from(c.wavePointer & 1 != 0)].clone()),
        }
    }
}

/// Emits `VoiceVolume`/`VoicePitch` when they changed and `NoteReleased` when the channel entered
/// its release (`SF_STOP`), updating `slot` bookkeeping.
fn update_voice(
    events: &mut Vec<SynthEvent>,
    slot: &mut SlotVoice,
    volume: f64,
    rate: f64,
    released: bool,
) {
    if volume != slot.last_volume {
        slot.last_volume = volume;
        events.push(SynthEvent::VoiceVolume {
            track: slot.track,
            voice: slot.voice,
            volume,
        });
    }
    if rate != slot.last_rate {
        slot.last_rate = rate;
        events.push(SynthEvent::VoicePitch {
            track: slot.track,
            voice: slot.voice,
            pitch: VoicePitch::DataRateHz(rate),
        });
    }
    // The gate expired / track ended: the note is no longer held (release tail keeps sounding).
    if released && !slot.released {
        slot.released = true;
        events.push(SynthEvent::NoteReleased {
            track: slot.track,
            key: slot.midi_key,
        });
    }
}

/// The DirectSound playback rate (data-samples/second): fixed-rate voices play at the mixer rate,
/// the rest at the `MidiKeyToFreq` result the engine stored.
fn ds_rate(type_: u8, frequency: u32) -> f64 {
    if type_ & TONEDATA_TYPE_FIX != 0 {
        ENGINE_RATE
    } else {
        f64::from(frequency)
    }
}

/// The data rate (sample values per second) of a CGB voice given its frequency-register value.
fn cgb_data_rate(ch: u8, reg: u32) -> f64 {
    match ch {
        // Tone frequency 131072/(2048−x) Hz × 8 samples per duty period.
        1 | 2 => 8.0 * 131072.0 / (2048.0 - reg.min(2047) as f64),
        // 32-sample wave at 2097152/(2048−x) samples per second.
        3 => 2097152.0 / (2048.0 - reg.min(2047) as f64),
        // NR43: clock = 524288 / r / 2^(s+1), where r=0 means r=0.5.
        _ => {
            let r = (reg & 7) as f64;
            let s = (reg >> 4) & 0xF;
            let divisor = if r == 0.0 { 0.5 } else { r };
            524288.0 / divisor / f64::from(1u32 << (s + 1))
        }
    }
}

/// Removes a waveform's DC offset (real GB/GBA audio is AC-coupled).
pub(super) fn dc_center(data: &mut [f32]) {
    if data.is_empty() {
        return;
    }
    let mean = data.iter().map(|&v| f64::from(v)).sum::<f64>() / data.len() as f64;
    for v in data.iter_mut() {
        *v -= mean as f32;
    }
}

/// The four GB square duty cycles as 8-sample loops.
fn build_square_waveforms() -> [Arc<Waveform>; 4] {
    const DUTIES: [[f32; 8]; 4] = [
        [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5], // 12.5%
        [0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5],  // 25%
        [0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5],    // 50%
        [-0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5],      // 75%
    ];
    DUTIES.map(|duty| {
        let mut data = duty.to_vec();
        dc_center(&mut data);
        let mut s = Waveform::new(data, 1.0, 1.0, true, 0);
        s.is_psg_square = true;
        Arc::new(s)
    })
}

/// The 15-bit and 7-bit LFSR noise sequences as looping ±0.5 sample data.
fn build_noise_waveforms() -> [Arc<Waveform>; 2] {
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
        dc_center(&mut data);
        let mut s = Waveform::new(data, 1.0, 1.0, true, 0);
        s.is_psg_square = true;
        Arc::new(s)
    };
    [generate(false), generate(true)]
}

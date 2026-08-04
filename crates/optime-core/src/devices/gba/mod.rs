//! The GBA backend: drives the MP2K player once per frame and turns its channel state into synth events.

mod extract;
pub mod m4a;
mod m4a_1;
pub mod m4a_tables;
pub mod rom;

pub use extract::extract_audio;
pub use extract::waveform_dc_stats;
pub use rom::GbaRom;

use std::collections::HashMap;
use std::sync::Arc;

use crate::PerDeviceSettings;
use crate::devices::{SynthEvent, TickFeedback, VoiceId, VoicePitch};
use crate::util::read_u32;
use crate::waveform::{Waveform, decode_pcm8};
use m4a::{
    CgbChannel, ID_NUMBER, MusicPlayerInfo, MusicPlayerTrack, SongHeader, SoundInfo,
    TONEDATA_TYPE_CGB, TONEDATA_TYPE_FIX,
};
use rom::ptr_to_offset;

pub const GBA_CLOCK_RATE: u64 = 16_777_216;

pub const CYCLES_PER_FRAME: u64 = 280_896;

pub const ENGINE_RATE: f64 = 13379.0;

const MAX_DS_CHANNELS: usize = m4a::MAX_DIRECTSOUND_CHANNELS;

const MASTER_VOLUME: u8 = 12;

const DS_MIXER_FULL_SCALE: f64 = 256.0;

const CGB_FULL_SCALE_GAIN: f64 = 0.5;

#[derive(Debug, Clone, Copy)]
struct SlotVoice {
    voice: VoiceId,
    track: usize,
    midi_key: u8,
    last_volume: f64,
    last_rate: f64,
    released: bool,
}

pub struct GbaPlayer {
    rom: Arc<[u8]>,
    mp: MusicPlayerInfo,
    si: SoundInfo,
    ds_slots: [Option<SlotVoice>; MAX_DS_CHANNELS],
    cgb_slots: [Option<SlotVoice>; 4],
    next_voice: VoiceId,
    last_pan: Vec<Option<(u8, u8)>>,
    waveform_cache: HashMap<(u32, bool), Option<Arc<Waveform>>>,
    wave_cache: HashMap<u32, Arc<Waveform>>,
    square_waveforms: [Arc<Waveform>; 4],
    noise_waveforms: [Arc<Waveform>; 2],
    remove_dc: bool,
    last_reverb: Option<u8>,
    finish_reported: bool,
}

impl GbaPlayer {
    pub fn new(rom: &GbaRom, song_id: u32) -> Option<GbaPlayer> {
        let header = rom.song_header(song_id)?;
        let data = rom.data.clone();

        let mut song = SongHeader {
            trackCount: header.track_count,
            priority: header.priority,
            reverb: header.reverb,
            tone: header.voicegroup as u32,
            ..SongHeader::default()
        };
        for i in 0..header.track_count as usize {
            let ptr = read_u32(&data, header.offset + 8 + i * 4);
            song.part[i] = ptr_to_offset(ptr, data.len())? as u32;
        }

        let mut mp = MusicPlayerInfo {
            ident: ID_NUMBER,
            trackCount: header.track_count,
            tracks: vec![MusicPlayerTrack::default(); header.track_count as usize],
            ..MusicPlayerInfo::default()
        };
        m4a::MPlayStart(&mut mp, song);

        let mut si = SoundInfo {
            maxChans: MAX_DS_CHANNELS as u8,
            masterVolume: MASTER_VOLUME,
            ..SoundInfo::default()
        };
        if let Some(reverb) = m4a::reverb_from_song_header(header.reverb) {
            si.reverb = reverb;
        }

        Some(GbaPlayer {
            rom: data,
            mp,
            si,
            ds_slots: [None; MAX_DS_CHANNELS],
            cgb_slots: [None; 4],
            next_voice: 0,
            last_pan: vec![None; header.track_count as usize],
            waveform_cache: HashMap::new(),
            wave_cache: HashMap::new(),
            square_waveforms: build_square_waveforms(),
            noise_waveforms: build_noise_waveforms(),
            remove_dc: false,
            last_reverb: None,
            finish_reported: false,
        })
    }

    fn steps_per_frame(&self) -> f64 {
        f64::from(self.mp.tempoI) / 150.0
    }

    fn tick_impl(
        &mut self,
        feedback: &mut TickFeedback,
        config: &PerDeviceSettings,
        events: &mut Vec<SynthEvent>,
    ) {
        self.remove_dc = config.remove_sample_dc_offset;

        if self.last_reverb != Some(self.si.reverb) {
            self.last_reverb = Some(self.si.reverb);
            events.push(SynthEvent::ReverbAmount {
                amount: self.si.reverb,
            });
        }

        for &(track, voice) in &feedback.ended_voices {
            for i in 0..MAX_DS_CHANNELS {
                if self.ds_slots[i].is_some_and(|s| s.voice == voice && s.track == track) {
                    self.ds_slots[i] = None;
                    self.si.chans[i].statusFlags = 0;
                    self.si.chans[i].track = None;
                }
            }
            for i in 0..4 {
                if self.cgb_slots[i].is_some_and(|s| s.voice == voice && s.track == track) {
                    self.cgb_slots[i] = None;
                    self.si.cgbChans[i].statusFlags = 0;
                    self.si.cgbChans[i].track = None;
                }
            }
        }
        feedback.ended_voices.clear();

        let result = m4a_1::MPlayMain(&mut self.mp, &mut self.si, &self.rom);
        self.emit_track_pans(events);
        self.start_new_notes(events);
        m4a_1::SoundMain(&mut self.si);
        self.emit_updates(events);

        if result.looped {
            events.push(SynthEvent::Looped);
        }
        if self.mp.status & 0x8000_0000 != 0 && !self.finish_reported {
            self.finish_reported = true;
            events.push(SynthEvent::Ended);
        }
    }

    fn emit_track_pans(&mut self, events: &mut Vec<SynthEvent>) {
        for t in 0..self.mp.tracks.len() {
            let (mr, ml) = (self.mp.tracks[t].volMR, self.mp.tracks[t].volML);
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

    fn start_new_notes(&mut self, events: &mut Vec<SynthEvent>) {
        for i in 0..MAX_DS_CHANNELS {
            if self.si.chans[i].statusFlags & m4a::SOUND_CHANNEL_SF_START == 0 {
                continue;
            }
            let c = self.si.chans[i];
            if let Some(old) = self.ds_slots[i].take() {
                events.push(SynthEvent::VoiceStopped {
                    track: old.track,
                    voice: old.voice,
                });
            }
            let Some(waveform) = self.direct_sound_waveform(c.wav) else {
                self.si.chans[i].statusFlags = 0;
                self.si.chans[i].track = None;
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
            if self.si.cgbChans[i].statusFlags & m4a::SOUND_CHANNEL_SF_START == 0 {
                continue;
            }
            let c = self.si.cgbChans[i];
            if let Some(old) = self.cgb_slots[i].take() {
                events.push(SynthEvent::VoiceStopped {
                    track: old.track,
                    voice: old.voice,
                });
            }
            let Some(waveform) = self.cgb_waveform(&c) else {
                self.si.cgbChans[i].statusFlags = 0;
                self.si.cgbChans[i].track = None;
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

    fn emit_updates(&mut self, events: &mut Vec<SynthEvent>) {
        for i in 0..MAX_DS_CHANNELS {
            let Some(mut slot) = self.ds_slots[i] else {
                continue;
            };
            let c = self.si.chans[i];
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
            let c = self.si.cgbChans[i];
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

    fn direct_sound_waveform(&mut self, wav_addr: u32) -> Option<Arc<Waveform>> {
        let key = (wav_addr, self.remove_dc);
        if let Some(cached) = self.waveform_cache.get(&key) {
            return cached.clone();
        }
        let remove_dc = self.remove_dc;
        let waveform = m4a::WaveData::read(&self.rom, wav_addr).map(|wav| {
            let raw = &self.rom[wav.data..wav.data + wav.size as usize];
            let mut data = decode_pcm8(raw);
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

    fn cgb_waveform(&mut self, c: &CgbChannel) -> Option<Arc<Waveform>> {
        match c.type_ & TONEDATA_TYPE_CGB {
            1 | 2 => Some(self.square_waveforms[(c.wavePointer & 3) as usize].clone()),
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
            _ => Some(self.noise_waveforms[usize::from(c.wavePointer & 1 != 0)].clone()),
        }
    }
}

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
    if released && !slot.released {
        slot.released = true;
        events.push(SynthEvent::NoteReleased {
            track: slot.track,
            key: slot.midi_key,
        });
    }
}

fn ds_rate(type_: u8, frequency: u32) -> f64 {
    if type_ & TONEDATA_TYPE_FIX != 0 {
        ENGINE_RATE
    } else {
        f64::from(frequency)
    }
}

fn cgb_data_rate(ch: u8, reg: u32) -> f64 {
    match ch {
        1 | 2 => 8.0 * 131072.0 / (2048.0 - reg.min(2047) as f64),
        3 => 2097152.0 / (2048.0 - reg.min(2047) as f64),
        _ => {
            let r = (reg & 7) as f64;
            let s = (reg >> 4) & 0xF;
            let divisor = if r == 0.0 { 0.5 } else { r };
            524288.0 / divisor / f64::from(1u32 << (s + 1))
        }
    }
}

fn dc_center(data: &mut [f32]) {
    if data.is_empty() {
        return;
    }
    let mean = data.iter().map(|&v| f64::from(v)).sum::<f64>() / data.len() as f64;
    for v in data.iter_mut() {
        *v -= mean as f32;
    }
}

fn build_square_waveforms() -> [Arc<Waveform>; 4] {
    const DUTIES: [[f32; 8]; 4] = [
        [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5],
        [0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5],
        [0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5],
    ];
    DUTIES.map(|duty| {
        let mut data = duty.to_vec();
        dc_center(&mut data);
        let mut s = Waveform::new(data, 1.0, 1.0, true, 0);
        s.is_psg_square = true;
        Arc::new(s)
    })
}

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

impl crate::devices::DevicePlayer for GbaPlayer {
    fn clock_rate(&self) -> f64 {
        GBA_CLOCK_RATE as f64
    }

    fn cycles_per_tick(&self) -> f64 {
        CYCLES_PER_FRAME as f64
    }

    fn steps_elapsed(&self) -> u32 {
        self.mp.clock
    }

    fn step_rate(&self) -> f64 {
        let frame_rate = GBA_CLOCK_RATE as f64 / CYCLES_PER_FRAME as f64;
        frame_rate * self.steps_per_frame()
    }

    fn steps_per_beat(&self) -> f64 {
        24.0
    }

    fn tick(
        &mut self,
        feedback: &mut TickFeedback,
        config: &PerDeviceSettings,
        events: &mut Vec<SynthEvent>,
    ) {
        self.tick_impl(feedback, config, events);
    }
}

impl crate::devices::SoundData for GbaRom {
    fn song_ids(&self) -> Vec<u32> {
        (0..self.song_count() as u32)
            .filter(|&id| self.song_header(id).is_some())
            .collect()
    }

    fn make_player(&self, id: u32) -> Option<Box<dyn crate::devices::DevicePlayer>> {
        Some(Box::new(GbaPlayer::new(self, id)?))
    }

    fn waveform_dc_stats(&self, id: u32) -> Vec<crate::devices::WaveformDcStat> {
        waveform_dc_stats(self, id)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

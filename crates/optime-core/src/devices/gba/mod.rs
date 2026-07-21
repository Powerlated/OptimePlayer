//! The Game Boy Advance sound device: GBA ROMs running the MP2K ("Sappy" / `m4a`) engine,
//! emulated from the `pret/pokeemerald` decompilation.
//!
//! Data flow within this folder:
//!
//! ```text
//! .gba bytes        ─► rom::GbaRom (song table + headers)          — the archive
//! GbaRom + song id  ─► GbaPlayer (this file — the synth glue)
//! GbaPlayer::tick   ─► m4a_1::MPlayMain / m4a_1::SoundMain          — the faithful engine,
//!                      driving m4a.rs's MusicPlayerInfo / SoundInfo / SoundChannel / CgbChannel
//!                   ─► read the resolved channel state → SynthEvent stream → SynthController
//! ```
//!
//! The engine ([`m4a`] = the `m4a.c` port, [`m4a_1`] = the `m4a_1.s` port, [`m4a_tables`] = the
//! `m4a_tables.c` port) is a faithful transliteration that only mutates hardware state. Everything
//! in *this* file is the minimal synth glue with no reference-source origin: driving the engine per
//! VBlank, turning the resulting `SoundChannel`/`CgbChannel` state into `SynthEvent`s, decoding
//! DirectSound PCM, and generating the PSG (square / noise / programmable-wave) sample data the
//! hardware would produce in silicon.

mod extract;
/// Faithful transliteration of `pret/pokeemerald`'s `src/m4a.c` (structs + the C sound routines).
pub mod m4a;
/// Faithful transliteration of `src/m4a_1.s` (the hand-written ARM engine: `MPlayMain`, `ply_*`,
/// `ChnVolSetAsm`, `SoundMain`/`SoundMainRAM`, `ply_lfos`, `TrackStop`), driving [`m4a`]'s structs.
mod m4a_1;
/// Faithful transliteration of `src/m4a_tables.c` (the `MidiKeyTo*` LUTs + helpers).
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

/// GBA CPU clock, in Hz.
pub const GBA_CLOCK_RATE: u64 = 16_777_216;

/// CPU cycles per LCD refresh — the MP2K engine runs once per VBlank (≈59.7275 Hz).
pub const CYCLES_PER_FRAME: u64 = 280_896;

/// The software mixer rate (`SOUND_MODE_FREQ_13379`) — the playback rate of fixed-frequency
/// (`TONEDATA_TYPE_FIX`) voices and the rate every DirectSound voice is mixed at on hardware.
pub const ENGINE_RATE: f64 = 13379.0;

/// We run the full hardware channel-struct count rather than the game-configured `maxChans`
/// (usually 5–8) so dense songs don't drop notes.
const MAX_DS_CHANNELS: usize = m4a::MAX_DIRECTSOUND_CHANNELS;

/// `SOUND_MODE_MASVOL` value every Pokémon game passes to `m4aSoundMode`.
const MASTER_VOLUME: u8 = 12;

const DS_MIXER_FULL_SCALE: f64 = 128.0;

const CGB_FULL_SCALE_GAIN: f64 = 1.0;

/// The synth-side identity of a hardware channel slot: which voice it drives and what we last told
/// the [`SynthController`](crate::SynthController) about it.
#[derive(Debug, Clone, Copy)]
struct SlotVoice {
    voice: VoiceId,
    track: usize,
    midi_key: u8,
    last_volume: f64,
    last_rate: f64,
    released: bool,
}

/// The GBA device player: the minimal glue driving the faithful engine and emitting `SynthEvent`s.
pub struct GbaPlayer {
    rom: Arc<[u8]>,
    mp: MusicPlayerInfo,
    si: SoundInfo,
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
    /// Last MP2K reverb amount emitted as `ReverbAmount`, so it is sent once (and again if it ever
    /// changes). `None` until the first tick emits the song's amount.
    last_reverb: Option<u8>,
    finish_reported: bool,
}

impl GbaPlayer {
    /// Binds song `song_id` from `rom` (mirrors `MPlayStart`). Returns `None` for empty/invalid
    /// songs.
    pub fn new(rom: &GbaRom, song_id: u32) -> Option<GbaPlayer> {
        let header = rom.song_header(song_id)?;
        let data = rom.data.clone();

        // Build the m4a `SongHeader` the engine wants, translating ROM-space pointers to offsets
        // (Optime reads a byte slice, so `cmdPtr` and the voicegroup are offsets, not addresses).
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
        // `MPlayStart` runs `m4aSoundMode(songHeader->reverb)` when the SET bit is present; its only
        // Optime-visible effect is storing the reverb amount, applied here since `si` is built after.
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

        // Announce the song's MP2K reverb amount once (and again only if it ever changes). The
        // controller applies it as a mono feedback delay on the sampled bus.
        if self.last_reverb != Some(self.si.reverb) {
            self.last_reverb = Some(self.si.reverb);
            events.push(SynthEvent::ReverbAmount {
                amount: self.si.reverb,
            });
        }

        // Reap voices the synthesizer stopped on its own (one-shot samples that ran out): detach the
        // hardware channel so the engine stops driving it.
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

        // One VBlank: advance the tracks, then start any freshly-allocated notes (while their
        // `SF_START` flag is still set), then step the envelopes, then emit the per-voice updates.
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

    /// Emits `TrackPan` for any track whose mixer volumes changed this frame (the normalized
    /// left/right split; the per-voice volume carries the absolute level separately).
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

    /// Starts a voice for every channel `ply_note` just allocated (`SOUND_CHANNEL_SF_START` still
    /// set, before `SoundMain` clears it). Decodes/generates the waveform; a channel whose sample
    /// can't be produced (compressed/invalid) is silenced.
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

    /// After `SoundMain`: emit per-voice volume/pitch changes, note releases, and stops for
    /// channels the envelope shut off.
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
fn dc_center(data: &mut [f32]) {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The per-side output byte the `m4a_1.s` DirectSound mixer (`SoundMainRAM`, inner loop
    /// `_081DD0B0`) writes for one channel into a freshly cleared PCM buffer: `(envVolSide · s) >> 8`
    /// (`mul` then `bic …, 0xFF0000` then the rotate-accumulate keep only the high byte of the
    /// 16-bit product). The buffer is signed 8-bit, so this is the channel's contribution in units of
    /// full scale = ±128.
    fn mixer_output_byte(env_vol_side: u8, sample: i8) -> i32 {
        (i32::from(env_vol_side) * i32::from(sample)) >> 8
    }

    /// The DirectSound voice level emitted by [`GbaPlayer::emit_updates`] carries the *summed* L+R
    /// gain (the panner splits it back, preserving the sum). Pinned here against the reference
    /// mixer: for any `(envVolLeft, envVolRight)` and sample, the synth's summed output
    /// `sample_norm · volume` equals the mixer's summed normalized output (both sides), to within the
    /// mixer's per-side `>> 8` truncation.
    #[test]
    fn ds_voice_level_matches_pokeemerald() {
        for env_l in [0u8, 1, 32, 100, 180, 206, 254] {
            for env_r in [0u8, 1, 32, 100, 180, 206, 254] {
                // The exact expression from `emit_updates`.
                let volume = (f64::from(env_l) + f64::from(env_r)) / DS_MIXER_FULL_SCALE;
                for sample in [-128i8, -100, -1, 0, 1, 64, 127] {
                    let sample_norm = f64::from(sample) / 128.0;
                    let synth_sum = sample_norm * volume;
                    // Reference: both channels' bytes, normalized to full scale (±128).
                    let reference_sum = (f64::from(mixer_output_byte(env_l, sample))
                        + f64::from(mixer_output_byte(env_r, sample)))
                        / 128.0;
                    // Each side's `>> 8` floors away up to 1 PCM8 LSB (= 1/128 of full scale); two
                    // sides ⇒ tolerance just over 2/128.
                    let tol = 2.0 / 128.0 + 1e-9;
                    assert!(
                        (synth_sum - reference_sum).abs() <= tol,
                        "env_l={env_l} env_r={env_r} sample={sample}: \
                         synth={synth_sum} reference={reference_sum}"
                    );
                }
            }
        }
    }

    /// A single max-volume DirectSound channel (both sides near the 8-bit ceiling) reaches essentially
    /// full scale on each output side — the invariant the mixer's `(envVolSide · s) >> 8` into a ±128
    /// buffer enforces, and the reason the emitted `volume` divides by exactly 256.
    #[test]
    fn full_ds_channel_spans_full_scale() {
        // Max side volume with the game's master (12): rightVolume 255 · uvol((12+1)·255>>4=207)>>8.
        let env = ((255u32 * ((13 * 255) >> 4)) >> 8) as u8; // == 206
        let volume = (f64::from(env) + f64::from(env)) / DS_MIXER_FULL_SCALE;
        // Full-scale sample ±1.0, centred (pan split 0.5/0.5) ⇒ per-side peak = 0.5 · volume.
        let per_side_peak = 0.5 * volume;
        assert!(
            (0.75..=1.0).contains(&per_side_peak),
            "per-side peak {per_side_peak} should approach full scale"
        );
    }

    /// A full-scale PSG (CGB) channel spans a quarter of output full scale: ±0.5 sample data times
    /// the `env=15` volume scalar (`0.5`) gives a summed peak of `0.25`.
    #[test]
    fn full_psg_channel_spans_quarter_scale() {
        let volume = 15.0 / 15.0 * CGB_FULL_SCALE_GAIN;
        let psg_peak = 0.5; // PSG sample data is ±0.5.
        assert!((psg_peak * volume - 0.25).abs() < 1e-9);
    }
}

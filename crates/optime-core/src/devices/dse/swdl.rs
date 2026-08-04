//! Parses the SWDL bank container and decodes the waveforms it holds.

use crate::util::{read_u16, read_u32};
use crate::waveform::{Waveform, decode_adpcm, decode_pcm8, decode_pcm16};

const HEADER_LEN: usize = 0x50;
const CHUNK_HEADER_LEN: usize = 0x10;
const SAMPLE_INFO_LEN: usize = 0x40;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    Pcm8,
    Pcm16,
    ImaAdpcm,
    Psg,
    Noise,
    Unknown(u16),
}

impl SampleFormat {
    fn from_code(code: u16) -> Self {
        match code >> 8 {
            0x00 => SampleFormat::Pcm8,
            0x01 => SampleFormat::Pcm16,
            0x02 => SampleFormat::ImaAdpcm,
            0x03 => SampleFormat::Psg,
            0x04 => SampleFormat::Noise,
            other => SampleFormat::Unknown(other),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WaveformInfo {
    pub id: u16,
    pub fine_tune: i8,
    pub coarse_tune: i8,
    pub root_key: u8,
    pub key_transpose: i8,
    pub volume: u8,
    pub pan: u8,
    pub format: SampleFormat,
    pub looping: bool,
    pub sample_rate: u32,
    pub pcm_offset: u32,
    pub loop_start_words: u32,
    pub loop_length_words: u32,
    pub envelope: [u8; 16],
}

#[derive(Debug, Clone)]
pub struct Split {
    pub min_note: u8,
    pub max_note: u8,
    pub bend_sensitivity: u8,
    pub wave_index: i16,
    pub key_base: i16,
    pub note_delta: i8,
    pub volume: u8,
    pub pan: u8,
    pub keygroup: u8,
    pub envelope: [u8; 16],
}

#[derive(Debug, Clone)]
pub struct Program {
    pub id: u16,
    pub volume: u8,
    pub pan: u8,
    pub splits: Vec<Split>,
}

impl Program {
    pub fn resolve_split(&self, key: u8) -> Option<&Split> {
        self.splits
            .iter()
            .find(|s| key >= s.min_note && key <= s.max_note)
            .or_else(|| self.splits.first())
    }
}

#[derive(Debug, Clone)]
pub struct Swdl {
    pub name: String,
    pub version: u16,
    pub waveforms: Vec<WaveformInfo>,
    pub wavi_by_slot: Vec<Option<WaveformInfo>>,
    pub pcmd: Vec<u8>,
    pub programs: Vec<Program>,
}

fn chunk_header(data: &[u8], pos: usize) -> Option<([u8; 4], usize, usize)> {
    let label: [u8; 4] = data.get(pos..pos + 4)?.try_into().ok()?;
    let len = read_u32(data, pos + 0x0C) as usize;
    Some((label, pos + CHUNK_HEADER_LEN, len))
}

fn walk_chunks(data: &[u8], start: usize, mut f: impl FnMut(&[u8; 4], &[u8])) {
    let mut pos = start;
    while pos + CHUNK_HEADER_LEN <= data.len() {
        let Some((label, payload_start, len)) = chunk_header(data, pos) else {
            break;
        };
        if &label == b"eod\0" {
            break;
        }
        let end = (payload_start + len).min(data.len());
        f(&label, &data[payload_start..end]);
        pos = (end + 0xF) & !0xF;
    }
}

impl Swdl {
    pub fn parse(data: &[u8]) -> Option<Swdl> {
        if data.len() < HEADER_LEN || &data[0..4] != b"swdl" {
            return None;
        }
        let version = read_u16(data, 0x0C);
        let name = read_name(data, 0x20, 16);
        let nb_wavi_slots = read_u16(data, 0x46) as usize;

        let nb_prgi_slots = read_u16(data, 0x48) as usize;

        let mut wavi_by_slot = Vec::new();
        let mut pcmd = Vec::new();
        let mut programs = Vec::new();

        walk_chunks(data, HEADER_LEN, |label, payload| match label {
            b"wavi" => wavi_by_slot = parse_wavi(payload, nb_wavi_slots),
            b"pcmd" => pcmd = payload.to_vec(),
            b"prgi" => programs = parse_prgi(payload, nb_prgi_slots),
            _ => {}
        });

        let waveforms = wavi_by_slot.iter().flatten().cloned().collect();

        Some(Swdl {
            name,
            version,
            waveforms,
            wavi_by_slot,
            pcmd,
            programs,
        })
    }

    pub fn program(&self, id: u16) -> Option<&Program> {
        self.programs.iter().find(|p| p.id == id)
    }

    pub fn waveform_for_wave(&self, wave_index: i16) -> Option<&WaveformInfo> {
        usize::try_from(wave_index)
            .ok()
            .and_then(|i| self.wavi_by_slot.get(i))
            .and_then(|s| s.as_ref())
    }

    pub fn decode_waveform(&self, info: &WaveformInfo, main_pcmd: &[u8]) -> Option<Waveform> {
        let pcmd = if self.pcmd.is_empty() {
            main_pcmd
        } else {
            &self.pcmd
        };
        let start = info.pcm_offset as usize;
        let byte_len = (info.loop_start_words as usize + info.loop_length_words as usize) * 4;
        let raw = pcmd.get(start..(start + byte_len).min(pcmd.len()))?;

        let data = match info.format {
            SampleFormat::Pcm8 => decode_pcm8(raw),
            SampleFormat::Pcm16 => decode_pcm16(raw),
            SampleFormat::ImaAdpcm => decode_adpcm(raw),
            SampleFormat::Psg | SampleFormat::Noise | SampleFormat::Unknown(_) => return None,
        };
        if data.is_empty() {
            return None;
        }

        let loop_point = match info.format {
            SampleFormat::Pcm16 => info.loop_start_words as i64 * 2,
            SampleFormat::Pcm8 => info.loop_start_words as i64 * 4,
            SampleFormat::ImaAdpcm => (info.loop_start_words as i64 - 1).max(0) * 8,
            _ => 0,
        };

        let mut waveform = Waveform::new(
            data,
            info.sample_rate as f64,
            info.sample_rate as f64,
            info.looping,
            loop_point,
        );
        waveform.sample_length = waveform.data.len();
        Some(waveform)
    }
}

fn parse_wavi(payload: &[u8], nb_slots: usize) -> Vec<Option<WaveformInfo>> {
    let mut out = Vec::with_capacity(nb_slots);
    for slot in 0..nb_slots {
        let ptr = read_u16(payload, slot * 2) as usize;
        if ptr == 0 || ptr + SAMPLE_INFO_LEN > payload.len() {
            out.push(None);
        } else {
            out.push(Some(parse_sample_info(
                &payload[ptr..ptr + SAMPLE_INFO_LEN],
            )));
        }
    }
    out
}

fn parse_prgi(payload: &[u8], nb_slots: usize) -> Vec<Program> {
    const SPLIT_BASE: usize = 0x60;
    const SPLIT_LEN: usize = 0x30;
    let mut out = Vec::new();
    for slot in 0..nb_slots {
        let ptr = read_u16(payload, slot * 2) as usize;
        if ptr == 0 || ptr + SPLIT_BASE > payload.len() {
            continue;
        }
        let p = &payload[ptr..];
        let nsplits = read_u16(p, 0x02) as usize;
        let mut splits = Vec::with_capacity(nsplits);
        for s in 0..nsplits {
            let so = SPLIT_BASE + s * SPLIT_LEN;
            let Some(e) = p.get(so..so + SPLIT_LEN) else {
                break;
            };
            let mut envelope = [0u8; 16];
            envelope.copy_from_slice(&e[0x20..0x30]);
            splits.push(Split {
                min_note: e[0x04],
                max_note: e[0x05],
                bend_sensitivity: e[0x02],
                wave_index: read_u16(e, 0x12) as i16,
                key_base: read_u16(e, 0x14) as i16,
                note_delta: e[0x17] as i8,
                volume: e[0x18],
                pan: e[0x19],
                keygroup: e[0x1A],
                envelope,
            });
        }
        out.push(Program {
            id: read_u16(p, 0x00),
            volume: p[0x04],
            pan: p[0x05],
            splits,
        });
    }
    out
}

fn parse_sample_info(e: &[u8]) -> WaveformInfo {
    let mut envelope = [0u8; 16];
    envelope.copy_from_slice(&e[0x30..0x40]);
    WaveformInfo {
        id: read_u16(e, 0x02),
        fine_tune: e[0x04] as i8,
        coarse_tune: e[0x05] as i8,
        root_key: e[0x06],
        key_transpose: e[0x07] as i8,
        volume: e[0x08],
        pan: e[0x09],
        format: SampleFormat::from_code(read_u16(e, 0x12)),
        looping: e[0x15] != 0,
        sample_rate: read_u32(e, 0x20),
        pcm_offset: read_u32(e, 0x24),
        loop_start_words: read_u32(e, 0x28),
        loop_length_words: read_u32(e, 0x2C),
        envelope,
    }
}

fn read_name(data: &[u8], offset: usize, max: usize) -> String {
    let mut s = String::new();
    for i in 0..max {
        let b = data.get(offset + i).copied().unwrap_or(0);
        if b == 0 || b == 0xAA {
            break;
        }
        s.push(b as char);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_format_codes() {
        assert_eq!(SampleFormat::from_code(0x0000), SampleFormat::Pcm8);
        assert_eq!(SampleFormat::from_code(0x0100), SampleFormat::Pcm16);
        assert_eq!(SampleFormat::from_code(0x0200), SampleFormat::ImaAdpcm);
        assert_eq!(SampleFormat::from_code(0x0300), SampleFormat::Psg);
    }

    #[test]
    fn parses_synthetic_waveform_info() {
        let mut e = [0u8; SAMPLE_INFO_LEN];
        e[0x02] = 1;
        e[0x04] = 42;
        e[0x05] = (-7i8) as u8;
        e[0x06] = 60;
        e[0x08] = 0x7F;
        e[0x09] = 0x40;
        e[0x12] = 0x00;
        e[0x13] = 0x01;
        e[0x15] = 1;
        e[0x20..0x24].copy_from_slice(&22050u32.to_le_bytes());
        e[0x28..0x2C].copy_from_slice(&1u32.to_le_bytes());
        e[0x2C..0x30].copy_from_slice(&10u32.to_le_bytes());

        let info = parse_sample_info(&e);
        assert_eq!(info.id, 1);
        assert_eq!(info.coarse_tune, -7);
        assert_eq!(info.root_key, 60);
        assert_eq!(info.fine_tune, 42);
        assert_eq!(info.format, SampleFormat::Pcm16);
        assert!(info.looping);
        assert_eq!(info.sample_rate, 22050);
        assert_eq!(info.loop_start_words, 1);
        assert_eq!(info.loop_length_words, 10);
    }
}

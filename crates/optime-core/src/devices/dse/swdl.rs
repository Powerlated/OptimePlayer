//! The DSE SWDL sample/instrument bank container.
//!
//! Layout transcribed from the real Explorers of Sky banks (`files/SOUND/BGM/*.swd` in
//! `pret/pmd-sky`) and cross-checked against the in-memory structs in `lib/DSE/include/dse.h`
//! (`sound_envelope_parameters`, `dse_instrument_split`). A bank is an 0x50-byte header
//! followed by 0x10-aligned chunks (`wavi`, `prgi`, `kgrp`, `pcmd`, `eod\0`), each a 16-byte
//! `ChunkHeader` (4-byte label, … `u32` length at +0x0C) then `length` bytes of payload.
//!
//! The game splits banks: `bgm.swd` is the **main bank** carrying the `pcmd` sample data and
//! the global `wavi` table; each `bgm####.swd` is a **per-song bank** carrying only `prgi`
//! programs + `kgrp` keygroups whose splits reference the main bank's samples by index.

use crate::sample::{decode_adpcm, decode_pcm16, decode_pcm8, Sample};
use crate::util::{read_u16, read_u32};

const HEADER_LEN: usize = 0x50;
const CHUNK_HEADER_LEN: usize = 0x10;
const SAMPLE_INFO_LEN: usize = 0x40;

/// How a [`SampleInfo`]'s PCM is encoded (the `smplfmt` field at +0x12 in a WAVI entry).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    Pcm8,
    Pcm16,
    ImaAdpcm,
    Psg,
    Unknown(u16),
}

impl SampleFormat {
    fn from_code(code: u16) -> Self {
        match code {
            0x0000 | 0x0100 => SampleFormat::Pcm8,
            0x0200 => SampleFormat::Pcm16,
            0x0300 => SampleFormat::ImaAdpcm,
            0x0001 => SampleFormat::Psg,
            other => SampleFormat::Unknown(other),
        }
    }
}

/// One WAVI sample-info entry (64 bytes). Offsets are relative to the entry start; derived from
/// `bgm.swd` and confirmed against `dse.h`'s `sound_envelope_parameters` (the envelope at +0x30).
#[derive(Debug, Clone)]
pub struct SampleInfo {
    pub id: u16,
    /// Fine tune (cents, +0x04) and coarse tune (semitones, +0x05). DSE's classic ctune is -7.
    pub fine_tune: i8,
    pub coarse_tune: i8,
    /// Root key the sample is tuned to (+0x06), usually 60 (middle C).
    pub root_key: u8,
    pub key_transpose: i8,
    pub volume: u8,
    pub pan: u8,
    pub format: SampleFormat,
    pub looping: bool,
    /// Recording rate in Hz (+0x20).
    pub sample_rate: u32,
    /// Byte offset of this sample's PCM within the main bank's `pcmd` payload (+0x24).
    pub pcm_offset: u32,
    /// Loop start (+0x28) and length (+0x2C), in 32-bit words (×4 bytes); total sample data
    /// length is `(loop_start + loop_length) * 4` bytes.
    pub loop_start_words: u32,
    pub loop_length_words: u32,
    /// The 16-byte ADSR envelope block at +0x30 (`sound_envelope_parameters`): bytes
    /// `[atkvol, atk, decay, sustain, hold, decay2, release]` live at sub-offsets 0x8..0xF.
    pub envelope: [u8; 16],
}

/// A parsed SWDL bank.
#[derive(Debug, Clone)]
pub struct Swdl {
    /// Internal file name from the header (+0x20, `\0`-terminated, 0xAA-padded).
    pub name: String,
    pub version: u16,
    /// WAVI sample-info entries that have a non-empty pointer-table slot.
    pub samples: Vec<SampleInfo>,
    /// The `pcmd` payload (sample data), if this bank carries one (main bank only).
    pub pcmd: Vec<u8>,
    /// Whether a `prgi` program chunk is present (per-song banks only).
    pub has_programs: bool,
}

/// A 16-byte chunk header: returns `(label, payload_start, payload_len)`.
fn chunk_header(data: &[u8], pos: usize) -> Option<([u8; 4], usize, usize)> {
    let label: [u8; 4] = data.get(pos..pos + 4)?.try_into().ok()?;
    let len = read_u32(data, pos + 0x0C) as usize;
    Some((label, pos + CHUNK_HEADER_LEN, len))
}

/// Walks chunks from `start`, calling `f(label, payload)` for each until `eod\0` or EOF.
fn walk_chunks(data: &[u8], start: usize, mut f: impl FnMut(&[u8; 4], &[u8])) {
    let mut pos = start;
    while pos + CHUNK_HEADER_LEN <= data.len() {
        let Some((label, payload_start, len)) = chunk_header(data, pos) else { break };
        if &label == b"eod\0" {
            break;
        }
        let end = (payload_start + len).min(data.len());
        f(&label, &data[payload_start..end]);
        // Chunks are padded up to the next 16-byte boundary.
        pos = (end + 0xF) & !0xF;
    }
}

impl Swdl {
    /// Parses a single SWDL bank from `data` (which must start at the `swdl` magic).
    pub fn parse(data: &[u8]) -> Option<Swdl> {
        if data.len() < HEADER_LEN || &data[0..4] != b"swdl" {
            return None;
        }
        let version = read_u16(data, 0x0C);
        let name = read_name(data, 0x20, 16);
        // `nbwavislots` at +0x46: the WAVI pointer table has this many u16 entries.
        let nb_wavi_slots = read_u16(data, 0x46) as usize;

        let mut samples = Vec::new();
        let mut pcmd = Vec::new();
        let mut has_programs = false;

        walk_chunks(data, HEADER_LEN, |label, payload| match label {
            b"wavi" => samples = parse_wavi(payload, nb_wavi_slots),
            b"pcmd" => pcmd = payload.to_vec(),
            b"prgi" => has_programs = true,
            _ => {}
        });

        Some(Swdl {
            name,
            version,
            samples,
            pcmd,
            has_programs,
        })
    }

    /// Decodes one sample to a playable [`Sample`], reading PCM from this bank's `pcmd` (for the
    /// main bank) or from `main_pcmd` (for a per-song bank referencing the main bank).
    pub fn decode_sample(&self, info: &SampleInfo, main_pcmd: &[u8]) -> Option<Sample> {
        let pcmd = if self.pcmd.is_empty() { main_pcmd } else { &self.pcmd };
        let start = info.pcm_offset as usize;
        let byte_len = (info.loop_start_words as usize + info.loop_length_words as usize) * 4;
        let raw = pcmd.get(start..(start + byte_len).min(pcmd.len()))?;

        let data = match info.format {
            SampleFormat::Pcm8 => decode_pcm8(raw),
            SampleFormat::Pcm16 => decode_pcm16(raw),
            SampleFormat::ImaAdpcm => decode_adpcm(raw),
            SampleFormat::Psg | SampleFormat::Unknown(_) => return None,
        };
        if data.is_empty() {
            return None;
        }

        // Loop point in samples: words → samples depends on the format's bytes-per-sample.
        let samples_per_word = match info.format {
            SampleFormat::Pcm16 => 2,  // 4 bytes / 2 bytes-per-sample
            SampleFormat::Pcm8 => 4,   // 4 bytes / 1
            SampleFormat::ImaAdpcm => 8, // 4 bytes / 0.5 (2 nibble-samples per byte)
            _ => 1,
        };
        let loop_point = info.loop_start_words as i64 * samples_per_word;

        let mut sample = Sample::new(
            data,
            info.sample_rate as f64,
            info.sample_rate as f64,
            info.looping,
            loop_point,
        );
        sample.sample_length = sample.data.len();
        Some(sample)
    }
}

/// Parses the WAVI chunk: a `nb_slots`-entry u16 pointer table (offsets relative to the chunk
/// payload start) followed by 64-byte [`SampleInfo`] entries. Zero pointers are empty slots.
fn parse_wavi(payload: &[u8], nb_slots: usize) -> Vec<SampleInfo> {
    let mut out = Vec::new();
    for slot in 0..nb_slots {
        let ptr = read_u16(payload, slot * 2) as usize;
        if ptr == 0 || ptr + SAMPLE_INFO_LEN > payload.len() {
            continue;
        }
        out.push(parse_sample_info(&payload[ptr..ptr + SAMPLE_INFO_LEN]));
    }
    out
}

fn parse_sample_info(e: &[u8]) -> SampleInfo {
    let mut envelope = [0u8; 16];
    envelope.copy_from_slice(&e[0x30..0x40]);
    SampleInfo {
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

/// Reads a `\0`-terminated, fixed-width name, stopping at the first `\0` or 0xAA pad byte.
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
        assert_eq!(SampleFormat::from_code(0x0200), SampleFormat::Pcm16);
        assert_eq!(SampleFormat::from_code(0x0300), SampleFormat::ImaAdpcm);
        assert_eq!(SampleFormat::from_code(0x0001), SampleFormat::Psg);
    }

    #[test]
    fn parses_synthetic_sample_info() {
        // A minimal 64-byte WAVI entry: id=1, ctune=-7, rootkey=60, format=PCM16, rate=22050.
        let mut e = [0u8; SAMPLE_INFO_LEN];
        e[0x02] = 1;
        e[0x04] = 42; // fine tune
        e[0x05] = (-7i8) as u8; // coarse tune
        e[0x06] = 60; // root key
        e[0x08] = 0x7F; // volume
        e[0x09] = 0x40; // pan
        e[0x12] = 0x00;
        e[0x13] = 0x02; // smplfmt = 0x0200 (PCM16, little-endian)
        e[0x15] = 1; // looping
        e[0x20..0x24].copy_from_slice(&22050u32.to_le_bytes());
        e[0x28..0x2C].copy_from_slice(&1u32.to_le_bytes()); // loop start words
        e[0x2C..0x30].copy_from_slice(&10u32.to_le_bytes()); // loop length words

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

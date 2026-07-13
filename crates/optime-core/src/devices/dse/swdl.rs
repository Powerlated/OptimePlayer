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

use crate::util::{read_u16, read_u32};
use crate::waveform::{Waveform, decode_adpcm, decode_pcm8, decode_pcm16};

const HEADER_LEN: usize = 0x50;
const CHUNK_HEADER_LEN: usize = 0x10;
const SAMPLE_INFO_LEN: usize = 0x40;

/// How a [`WaveformInfo`]'s PCM is encoded.
///
/// The WAVI `smplfmt` u16 at +0x12 carries the value in its **high byte** (the DSE driver only
/// reads `dse_sample.sample_format` = WAVI byte +0x13, then hands it straight to the NDS
/// `Snd_SetupChannelPcm`/`Psg`/`Noise`): `0`=PCM8, `1`=PCM16, `2`=IMA-ADPCM, `3`=PSG square,
/// `4`=noise. So `smplfmt` `0x0100` is PCM16 and `0x0200` is **ADPCM** — not the other way round.
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

/// One WAVI sample-info entry (64 bytes). Offsets are relative to the entry start; derived from
/// `bgm.swd` and confirmed against `dse.h`'s `sound_envelope_parameters` (the envelope at +0x30).
#[derive(Debug, Clone)]
pub struct WaveformInfo {
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

/// One PRGI **split**: a key range mapping to a sample, with its own tuning + envelope. A
/// program is a list of splits (like a sampler keygroup). Offsets within the 0x30-byte on-disk
/// entry match `dse.h`'s `dse_instrument_split`.
#[derive(Debug, Clone)]
pub struct Split {
    pub min_note: u8,
    pub max_note: u8,
    /// Default pitch-bend range in semitones (+0x02, `bend_sensitivity`). Used by `SetKeyBend`
    /// when the channel's `SetKeyBendRange` is unset.
    pub bend_sensitivity: u8,
    /// Index into this bank's WAVI table of the sample to play (+0x12).
    pub wave_index: i16,
    /// Root key the sample is tuned to for this split (+0x14).
    pub key_base: i16,
    /// Per-split key transpose in semitones (+0x17).
    pub note_delta: i8,
    pub volume: u8,
    pub pan: u8,
    pub keygroup: u8,
    /// 16-byte `sound_envelope_parameters` block (+0x20). Bytes 0x8..0xF are
    /// `[atkvol, attack, decay, sustain, hold, decay2, release, unk]`.
    pub envelope: [u8; 16],
}

/// One PRGI **program** (an "instrument"): a volume/pan and a set of [`Split`]s.
#[derive(Debug, Clone)]
pub struct Program {
    pub id: u16,
    pub volume: u8,
    pub pan: u8,
    pub splits: Vec<Split>,
}

impl Program {
    /// Resolves which split plays `key`, or the first split as a fallback. Mirrors the driver's
    /// `DseSwd_GetNextSplitInRange` (first split whose `[min_note, max_note]` contains `key`).
    pub fn resolve_split(&self, key: u8) -> Option<&Split> {
        self.splits
            .iter()
            .find(|s| key >= s.min_note && key <= s.max_note)
            .or_else(|| self.splits.first())
    }
}

/// A parsed SWDL bank.
#[derive(Debug, Clone)]
pub struct Swdl {
    /// Internal file name from the header (+0x20, `\0`-terminated, 0xAA-padded).
    pub name: String,
    pub version: u16,
    /// WAVI sample-info entries that have a non-empty pointer-table slot.
    pub waveforms: Vec<WaveformInfo>,
    /// WAVI entries indexed by their original pointer-table slot (`wave_index` in a [`Split`]
    /// refers to this), or `None` for an empty slot.
    pub wavi_by_slot: Vec<Option<WaveformInfo>>,
    /// The `pcmd` payload (sample data), if this bank carries one (main bank only).
    pub pcmd: Vec<u8>,
    /// PRGI programs (per-song banks only), in file order.
    pub programs: Vec<Program>,
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
        let Some((label, payload_start, len)) = chunk_header(data, pos) else {
            break;
        };
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

        // `nbprgislots` at +0x48: the PRGI pointer table length.
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

    /// Looks up a PRGI program by its id (what a track's `SetInstrument` selects).
    pub fn program(&self, id: u16) -> Option<&Program> {
        self.programs.iter().find(|p| p.id == id)
    }

    /// The WAVI sample-info for a [`Split`]'s `wave_index`, if that slot is populated.
    pub fn waveform_for_wave(&self, wave_index: i16) -> Option<&WaveformInfo> {
        usize::try_from(wave_index)
            .ok()
            .and_then(|i| self.wavi_by_slot.get(i))
            .and_then(|s| s.as_ref())
    }

    /// Decodes one waveform to a playable [`Waveform`], reading PCM from this bank's `pcmd` (for the
    /// main bank) or from `main_pcmd` (for a per-song bank referencing the main bank).
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
            // PSG square/noise voices are generated by the hardware, not sampled — skip for now.
            SampleFormat::Psg | SampleFormat::Noise | SampleFormat::Unknown(_) => return None,
        };
        if data.is_empty() {
            return None;
        }

        // Loop point in samples. `loop_start_words` counts 32-bit words from the start of this
        // sample's data; converting to a decoded-sample index depends on the format's density.
        // IMA-ADPCM carries a 4-byte (one-word) predictor preamble that decodes to no output, so
        // its loop word is offset by that one word (`(w - 1) * 8` samples).
        let loop_point = match info.format {
            SampleFormat::Pcm16 => info.loop_start_words as i64 * 2, // 4 bytes / 2 per sample
            SampleFormat::Pcm8 => info.loop_start_words as i64 * 4,  // 4 bytes / 1
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

/// Parses the WAVI chunk: a `nb_slots`-entry u16 pointer table (offsets relative to the chunk
/// payload start) followed by 64-byte [`WaveformInfo`] entries. Returns one slot-indexed entry per
/// pointer (`None` for a zero/empty slot) so a [`Split`]'s `wave_index` can index it directly.
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

/// Parses the PRGI chunk: a `nb_slots`-entry u16 pointer table followed by program entries. Each
/// program is a header (`id`@0, `nsplits`@2, `vol`@4, `pan`@5, four 16-byte LFO entries @0x10)
/// then `nsplits` 0x30-byte [`Split`]s starting at +0x60.
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
        // The format lives in the high byte of `smplfmt` (the DSE driver reads WAVI +0x13):
        // 0x0100 = PCM16, 0x0200 = ADPCM, 0x0300 = PSG.
        assert_eq!(SampleFormat::from_code(0x0000), SampleFormat::Pcm8);
        assert_eq!(SampleFormat::from_code(0x0100), SampleFormat::Pcm16);
        assert_eq!(SampleFormat::from_code(0x0200), SampleFormat::ImaAdpcm);
        assert_eq!(SampleFormat::from_code(0x0300), SampleFormat::Psg);
    }

    #[test]
    fn parses_synthetic_waveform_info() {
        // A minimal 64-byte WAVI entry: id=1, ctune=-7, rootkey=60, format=PCM16, rate=22050.
        let mut e = [0u8; SAMPLE_INFO_LEN];
        e[0x02] = 1;
        e[0x04] = 42; // fine tune
        e[0x05] = (-7i8) as u8; // coarse tune
        e[0x06] = 60; // root key
        e[0x08] = 0x7F; // volume
        e[0x09] = 0x40; // pan
        e[0x12] = 0x00;
        e[0x13] = 0x01; // smplfmt = 0x0100 -> high byte 0x01 = PCM16
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

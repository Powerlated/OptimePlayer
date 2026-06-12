//! Voicegroup (ToneData) parsing and instrument resolution, mirroring `ply_note`'s key-split /
//! rhythm handling in `pret/pokeemerald`.

use crate::util::{read_u32, read_u8};

use super::rom::ptr_to_offset;

/// `TONEDATA_TYPE_*` bits.
const TYPE_CGB: u8 = 0x07;
const TYPE_FIX: u8 = 0x08;
/// Compressed / reversed DirectSound flags (unsupported here).
const TYPE_CMP_REV: u8 = 0x30;
const TYPE_SPL: u8 = 0x40;
const TYPE_RHY: u8 = 0x80;

/// One raw 12-byte `ToneData` record.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ToneData {
    pub kind: u8,
    pub key: u8,
    pub length: u8,
    pub pan_sweep: u8,
    /// Wave pointer / sub-voicegroup pointer / CGB duty or wave-pattern selector.
    pub wav: u32,
    pub attack: u8,
    pub decay: u8,
    pub sustain: u8,
    pub release: u8,
}

impl ToneData {
    /// Reads the record at `offset`.
    pub fn read(rom: &[u8], offset: usize) -> ToneData {
        ToneData {
            kind: read_u8(rom, offset),
            key: read_u8(rom, offset + 1),
            length: read_u8(rom, offset + 2),
            pan_sweep: read_u8(rom, offset + 3),
            wav: read_u32(rom, offset + 4),
            attack: read_u8(rom, offset + 8),
            decay: read_u8(rom, offset + 9),
            sustain: read_u8(rom, offset + 10),
            release: read_u8(rom, offset + 11),
        }
    }
}

/// Which CGB ("GameBoy compatible") channel a tone drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CgbKind {
    /// NR1x: square with sweep. `duty` 0..=3, `sweep` register value.
    Square1 { duty: u8, sweep: u8 },
    /// NR2x: plain square.
    Square2 { duty: u8 },
    /// NR3x: 32×4-bit programmable wave at `wave_addr`.
    Wave { wave_addr: u32 },
    /// NR4x: LFSR noise; `period7` selects the short 7-bit sequence.
    Noise { period7: bool },
}

impl CgbKind {
    /// The hardware channel number (1..=4) — selects `MidiKeyToCgbFreq` behavior.
    pub fn channel_num(self) -> u8 {
        match self {
            CgbKind::Square1 { .. } => 1,
            CgbKind::Square2 { .. } => 2,
            CgbKind::Wave { .. } => 3,
            CgbKind::Noise { .. } => 4,
        }
    }
}

/// The sound source of a resolved tone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToneKind {
    /// A PCM sample mixed in software. `fixed` plays at the mixer rate (no repitching).
    DirectSound { wav_addr: u32, fixed: bool },
    /// One of the four GB legacy channels.
    Cgb(CgbKind),
}

/// A fully resolved instrument for one note: key-splits and rhythm sub-voicegroups applied.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResolvedTone {
    pub kind: ToneKind,
    /// The key the channel actually sounds (rhythm voices replace the played key).
    pub key: u8,
    /// Rhythm-voice fixed pan (−64..=63 doubled to −128..=126), if the voice carries one.
    pub rhythm_pan: Option<i8>,
    /// attack/decay/sustain/release envelope bytes.
    pub adsr: [u8; 4],
}

/// Resolves `tone` (a track's current `ToneData`) for `key`, following `ply_note`:
/// key-split voices index a sub-voicegroup through their key table, rhythm voices index it by
/// the key itself (taking the sub-voice's own key and optional fixed pan). Returns `None` for
/// malformed or unsupported (compressed) voices — the note simply doesn't sound.
pub(crate) fn resolve_tone(rom: &[u8], tone: &ToneData, key: u8) -> Option<ResolvedTone> {
    let (tone, key, rhythm_pan) = if tone.kind & (TYPE_RHY | TYPE_SPL) != 0 {
        let index = if tone.kind & TYPE_SPL != 0 {
            // The key-split table pointer lives in the ADSR word of the parent record.
            let table = u32::from_le_bytes([tone.attack, tone.decay, tone.sustain, tone.release]);
            read_u8(rom, ptr_to_offset(table, rom.len())? + key as usize)
        } else {
            key
        };
        let group = ptr_to_offset(tone.wav, rom.len())?;
        let sub = ToneData::read(rom, group + index as usize * 12);
        if sub.kind & (TYPE_RHY | TYPE_SPL) != 0 {
            return None;
        }
        let (key, pan) = if tone.kind & TYPE_RHY != 0 {
            let pan = (sub.pan_sweep & 0x80 != 0)
                .then(|| (sub.pan_sweep.wrapping_sub(0xC0) as i8).wrapping_mul(2));
            (sub.key, pan)
        } else {
            (key, None)
        };
        (sub, key, pan)
    } else {
        (*tone, key, None)
    };

    let kind = match tone.kind & TYPE_CGB {
        0 => {
            if tone.kind & TYPE_CMP_REV != 0 {
                return None; // compressed / reversed PCM unsupported
            }
            ToneKind::DirectSound {
                wav_addr: tone.wav,
                fixed: tone.kind & TYPE_FIX != 0,
            }
        }
        1 => ToneKind::Cgb(CgbKind::Square1 {
            duty: (tone.wav & 3) as u8,
            // ply_note: pan_sweep is the sweep register unless bit 7 set or bits 4–6 clear.
            sweep: if tone.pan_sweep & 0x80 == 0 && tone.pan_sweep & 0x70 != 0 {
                tone.pan_sweep
            } else {
                8
            },
        }),
        2 => ToneKind::Cgb(CgbKind::Square2 {
            duty: (tone.wav & 3) as u8,
        }),
        3 => ToneKind::Cgb(CgbKind::Wave {
            wave_addr: tone.wav,
        }),
        4 => ToneKind::Cgb(CgbKind::Noise {
            period7: tone.wav & 1 != 0,
        }),
        _ => return None,
    };

    Some(ResolvedTone {
        kind,
        key,
        rhythm_pan,
        adsr: [tone.attack, tone.decay, tone.sustain, tone.release],
    })
}

/// A parsed DirectSound `WaveData` header (the PCM data follows at `offset + 16`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct WaveData {
    /// Frequency field: sample rate << 10.
    pub freq: u32,
    pub loop_start: u32,
    pub size: u32,
    pub looping: bool,
    /// Offset of the s8 PCM data in the ROM.
    pub data_offset: usize,
}

impl WaveData {
    /// Reads and validates the header at ROM address `wav_addr`.
    pub fn read(rom: &[u8], wav_addr: u32) -> Option<WaveData> {
        let offset = ptr_to_offset(wav_addr, rom.len())?;
        let freq = read_u32(rom, offset + 4);
        let loop_start = read_u32(rom, offset + 8);
        let size = read_u32(rom, offset + 12);
        let data_offset = offset + 16;
        if size as usize > rom.len().saturating_sub(data_offset) || loop_start > size {
            return None;
        }
        Some(WaveData {
            freq,
            loop_start,
            size,
            looping: read_u8(rom, offset + 3) & 0x40 != 0,
            data_offset,
        })
    }
}

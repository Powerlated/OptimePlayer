//! SDAT container parsing: SYMB symbol tables, INFO records, the FAT, and SBNK bank decoding.

use std::collections::HashMap;
use std::sync::Arc;

use super::bank::{
    InstrumentBank, InstrumentRecord, InstrumentRegion, InstrumentType, calc_decay_coeff,
    get_effective_attack, get_sustain_level,
};
use crate::util::{read_u8, read_u16, read_u32, search_for_sequence};

/// INFO record for a sequence (SSEQ).
#[derive(Debug, Clone, Default)]
pub struct SseqInfo {
    /// FAT file id of the SSEQ data.
    pub file_id: u16,
    /// Instrument bank (SBNK) id this sequence uses.
    pub bank: u16,
    /// Sequence master volume.
    pub volume: u8,
    /// Channel pressure range (purpose unclear in the original).
    pub cpr: u8,
    /// Polyphonic pressure range (purpose unclear in the original).
    pub ppr: u8,
    /// Play setting (purpose unclear in the original).
    pub ply: u8,
}

/// INFO record for a sequence archive (SSAR).
#[derive(Debug, Clone, Default)]
pub struct SsarInfo {
    /// FAT file id.
    pub file_id: u16,
}

/// INFO record for an instrument bank (SBNK), referencing up to four sample archives.
#[derive(Debug, Clone, Default)]
pub struct BankInfo {
    /// FAT file id of the bank.
    pub file_id: u16,
    /// Linked SWAR archive ids.
    pub swar_id: [u16; 4],
}

/// INFO record for a sample/wave archive (SWAR).
#[derive(Debug, Clone, Default)]
pub struct SwarInfo {
    /// FAT file id.
    pub file_id: u16,
}

/// A parsed SDAT sound archive.
pub struct Sdat {
    /// The raw SDAT container bytes.
    pub data: Arc<[u8]>,
    /// Ids of sequences that have INFO records, in order.
    pub sseq_list: Vec<u32>,
    /// SSEQ INFO records, indexed by id.
    pub sseq_infos: Vec<Option<SseqInfo>>,
    /// Sequence name → id.
    pub sseq_name_to_id: HashMap<String, u32>,
    /// Sequence id → name.
    pub sseq_id_to_name: HashMap<u32, String>,
    /// Bank name → id.
    pub sbnk_name_to_id: HashMap<String, u32>,
    /// Bank id → name.
    pub sbnk_id_to_name: HashMap<u32, String>,
    /// SSAR INFO records, indexed by id.
    pub ssar_infos: Vec<Option<SsarInfo>>,
    /// SBNK INFO records, indexed by id.
    pub sbnk_infos: Vec<Option<BankInfo>>,
    /// SWAR INFO records, indexed by id.
    pub swar_infos: Vec<Option<SwarInfo>>,
    /// Decoded instrument banks, indexed by bank id.
    pub instrument_banks: Vec<Option<InstrumentBank>>,
    /// File Allocation Table: file id → (offset, length) into `data`.
    pub fat: HashMap<u32, (usize, usize)>,
}

/// SDAT container magic: "SDAT", byte order 0xFEFF, version 0x0100.
const SDAT_MAGIC: [u8; 8] = [0x53, 0x44, 0x41, 0x54, 0xFF, 0xFE, 0x00, 0x01];

impl Sdat {
    /// Returns the bytes of FAT file `id`, if present.
    pub fn file(&self, id: u16) -> Option<&[u8]> {
        let (off, len) = self.fat.get(&u32::from(id)).copied()?;
        self.data.get(off..off + len)
    }

    /// Scans `rom` for every SDAT container and parses each one.
    pub fn load_all(rom: &[u8]) -> Vec<Sdat> {
        search_for_sequence(rom, &SDAT_MAGIC)
            .into_iter()
            .filter_map(|offset| Sdat::parse(&rom[offset..]))
            .collect()
    }

    /// Parses a single SDAT container starting at the beginning of `view`.
    ///
    /// Returns `None` if the header looks malformed.
    pub fn parse(view: &[u8]) -> Option<Sdat> {
        let header_size = read_u16(view, 0xC);
        if header_size > 256 {
            return None;
        }

        let symb_offs = read_u32(view, 0x10) as usize;
        let info_offs = read_u32(view, 0x18) as usize;
        let fat_offs = read_u32(view, 0x20) as usize;

        // Copy the container so the Sdat owns its bytes independent of the ROM buffer.
        let data: Arc<[u8]> = Arc::from(view.to_vec());

        let mut sdat = Sdat {
            data: data.clone(),
            sseq_list: Vec::new(),
            sseq_infos: Vec::new(),
            sseq_name_to_id: HashMap::new(),
            sseq_id_to_name: HashMap::new(),
            sbnk_name_to_id: HashMap::new(),
            sbnk_id_to_name: HashMap::new(),
            ssar_infos: Vec::new(),
            sbnk_infos: Vec::new(),
            swar_infos: Vec::new(),
            instrument_banks: Vec::new(),
            fat: HashMap::new(),
        };

        let d = &data[..];

        // --- SYMB: SSEQ names ---
        if symb_offs != 0 {
            let symb = symb_offs;
            let sseq_list_offs = read_u32(d, symb + 0x8) as usize;
            if symb + sseq_list_offs < d.len() {
                let n = read_u32(d, symb + sseq_list_offs) as usize;
                for i in 0..n {
                    let name_offs = read_u32(d, symb + sseq_list_offs + 4 + i * 4) as usize;
                    if name_offs != 0 {
                        let name = read_c_string(d, symb + name_offs);
                        sdat.sseq_name_to_id.insert(name.clone(), i as u32);
                        sdat.sseq_id_to_name.insert(i as u32, name);
                    }
                }
            }

            // --- SYMB: BANK names ---
            let bank_list_offs = read_u32(d, symb + 0x10) as usize;
            let bn = read_u32(d, symb + bank_list_offs) as usize;
            for i in 0..bn {
                let name_offs = read_u32(d, symb + bank_list_offs + 4 + i * 4) as usize;
                if name_offs != 0 {
                    let name = read_c_string(d, symb + name_offs);
                    sdat.sbnk_name_to_id.insert(name.clone(), i as u32);
                    sdat.sbnk_id_to_name.insert(i as u32, name);
                }
            }
        }

        // --- INFO: SSEQ ---
        let info = info_offs;
        let sseq_info_offs = read_u32(d, info + 0x8) as usize;
        let sseq_n = read_u32(d, info + sseq_info_offs) as usize;
        sdat.sseq_infos.resize(sseq_n, None);
        for i in 0..sseq_n {
            let entry = read_u32(d, info + sseq_info_offs + 4 + i * 4) as usize;
            if entry != 0 {
                sdat.sseq_infos[i] = Some(SseqInfo {
                    file_id: read_u16(d, info + entry),
                    bank: read_u16(d, info + entry + 4),
                    volume: read_u8(d, info + entry + 6),
                    cpr: read_u8(d, info + entry + 7),
                    ppr: read_u8(d, info + entry + 8),
                    ply: read_u8(d, info + entry + 9),
                });
                sdat.sseq_list.push(i as u32);
            }
        }

        // --- INFO: SSAR ---
        let ssar_info_offs = read_u32(d, info + 0xC) as usize;
        let ssar_n = read_u32(d, info + ssar_info_offs) as usize;
        sdat.ssar_infos.resize(ssar_n, None);
        for i in 0..ssar_n {
            let entry = read_u32(d, info + ssar_info_offs + 4 + i * 4) as usize;
            if entry != 0 {
                sdat.ssar_infos[i] = Some(SsarInfo {
                    file_id: read_u16(d, info + entry),
                });
            }
        }

        // --- INFO: BANK ---
        let bank_info_offs = read_u32(d, info + 0x10) as usize;
        let bank_n = read_u32(d, info + bank_info_offs) as usize;
        sdat.sbnk_infos.resize(bank_n, None);
        for i in 0..bank_n {
            let entry = read_u32(d, info + bank_info_offs + 4 + i * 4) as usize;
            if entry != 0 {
                sdat.sbnk_infos[i] = Some(BankInfo {
                    file_id: read_u16(d, info + entry),
                    swar_id: [
                        read_u16(d, info + entry + 0x4),
                        read_u16(d, info + entry + 0x6),
                        read_u16(d, info + entry + 0x8),
                        read_u16(d, info + entry + 0xA),
                    ],
                });
            }
        }

        // --- INFO: SWAR ---
        let swar_info_offs = read_u32(d, info + 0x14) as usize;
        let swar_n = read_u32(d, info + swar_info_offs) as usize;
        sdat.swar_infos.resize(swar_n, None);
        for i in 0..swar_n {
            let entry = read_u32(d, info + swar_info_offs + 4 + i * 4) as usize;
            if entry != 0 {
                sdat.swar_infos[i] = Some(SwarInfo {
                    file_id: read_u16(d, info + entry),
                });
            }
        }

        // --- FAT ---
        let fat = fat_offs;
        let num_files = read_u32(d, fat + 8) as usize;
        for i in 0..num_files {
            let entry = fat + 0xC + i * 0x10;
            let file_offs = read_u32(d, entry) as usize;
            let file_size = read_u32(d, entry + 4) as usize;
            sdat.fat.insert(i as u32, (file_offs, file_size));
        }

        // --- Decode instrument banks (SBNK) ---
        sdat.instrument_banks.resize(sdat.sbnk_infos.len(), None);
        for i in 0..sdat.sbnk_infos.len() {
            let Some(bank_info) = sdat.sbnk_infos[i].clone() else {
                continue;
            };
            let Some((off, len)) = sdat.fat.get(&u32::from(bank_info.file_id)).copied() else {
                continue;
            };
            let bank_file = &d[off..off + len];
            sdat.instrument_banks[i] = Some(decode_bank(bank_file));
        }

        Some(sdat)
    }
}

/// Reads a NUL-terminated ASCII string from `data` starting at `offset`.
fn read_c_string(data: &[u8], offset: usize) -> String {
    let mut s = String::new();
    let mut i = offset;
    loop {
        let c = read_u8(data, i);
        if c == 0 {
            break;
        }
        s.push(c as char);
        i += 1;
    }
    s
}

/// Decodes one SBNK bank file into an [`InstrumentBank`].
fn decode_bank(bank_file: &[u8]) -> InstrumentBank {
    let num_instruments = read_u32(bank_file, 0x38) as usize;
    let mut bank = InstrumentBank {
        instruments: Vec::with_capacity(num_instruments),
    };

    // Reads the 12-byte region payload at `record_offset + offset` into a region.
    let read_region = |record_offset: usize, offset: usize| -> InstrumentRegion {
        let base = record_offset + offset;
        let attack = read_u8(bank_file, base + 0x5);
        let decay = read_u8(bank_file, base + 0x6);
        let sustain = read_u8(bank_file, base + 0x7);
        let release = read_u8(bank_file, base + 0x8);
        InstrumentRegion {
            swav_info_id: read_u16(bank_file, base),
            swar_info_id: read_u16(bank_file, base + 0x2),
            note_number: read_u8(bank_file, base + 0x4),
            attack,
            attack_coefficient: get_effective_attack(i32::from(attack)),
            decay,
            decay_coefficient: calc_decay_coeff(i32::from(decay)),
            sustain,
            sustain_level: get_sustain_level(i32::from(sustain)),
            release,
            release_coefficient: calc_decay_coeff(i32::from(release)),
            pan: read_u8(bank_file, base + 0x9),
        }
    };

    for j in 0..num_instruments {
        let f_record = read_u8(bank_file, 0x3C + j * 4);
        let record_offset = read_u16(bank_file, 0x3C + j * 4 + 1) as usize;

        let mut record = InstrumentRecord {
            f_record,
            ..Default::default()
        };

        match InstrumentType::from_u8(f_record) {
            Some(InstrumentType::Empty) => {}
            Some(InstrumentType::SingleSample)
            | Some(InstrumentType::PsgPulse)
            | Some(InstrumentType::PsgNoise) => {
                record.regions.push(read_region(record_offset, 0));
            }
            Some(InstrumentType::Drumset) => {
                let lower = read_u8(bank_file, record_offset);
                let upper = read_u8(bank_file, record_offset + 1);
                record.lower_note = lower;
                record.upper_note = upper;
                let count = i32::from(upper) - i32::from(lower) + 1;
                for k in 0..count.max(0) as usize {
                    record.regions.push(read_region(record_offset, 4 + k * 12));
                }
            }
            Some(InstrumentType::MultiSample) => {
                let mut count = 0;
                for k in 0..8 {
                    let end = read_u8(bank_file, record_offset + k);
                    record.region_end[k] = end;
                    if end == 0 {
                        count = k;
                        break;
                    } else if end == 0x7F {
                        count = k + 1;
                        break;
                    }
                }
                for k in 0..count {
                    record.regions.push(read_region(record_offset, 10 + k * 12));
                }
            }
            None => {}
        }

        bank.instruments.push(record);
    }

    bank
}

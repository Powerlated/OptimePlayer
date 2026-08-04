//! The DS instrument bank: its regions and records, plus the ADSR coefficient helpers they need.

use super::tables::{ATTACK_COEFF_TABLE, DECIBEL_SQUARE_TABLE};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstrumentType {
    Empty,
    SingleSample,
    PsgPulse,
    PsgNoise,
    Drumset,
    MultiSample,
}

impl InstrumentType {
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0x0 => InstrumentType::Empty,
            0x1 => InstrumentType::SingleSample,
            0x2 => InstrumentType::PsgPulse,
            0x3 => InstrumentType::PsgNoise,
            0x10 => InstrumentType::Drumset,
            0x11 => InstrumentType::MultiSample,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct InstrumentRegion {
    pub swav_info_id: u16,
    pub swar_info_id: u16,
    pub note_number: u8,
    pub attack: u8,
    pub attack_coefficient: i32,
    pub decay: u8,
    pub decay_coefficient: i32,
    pub sustain: u8,
    pub sustain_level: i32,
    pub release: u8,
    pub release_coefficient: i32,
    pub pan: u8,
}

#[derive(Debug, Clone, Default)]
pub struct InstrumentRecord {
    pub f_record: u8,
    pub lower_note: u8,
    pub upper_note: u8,
    pub region_end: [u8; 8],
    pub regions: Vec<InstrumentRegion>,
}

impl InstrumentRecord {
    pub fn instrument_type(&self) -> Option<InstrumentType> {
        InstrumentType::from_u8(self.f_record)
    }

    pub fn resolve_entry_index(&self, note: u8) -> usize {
        match self.instrument_type() {
            Some(InstrumentType::SingleSample)
            | Some(InstrumentType::PsgPulse)
            | Some(InstrumentType::PsgNoise) => 0,
            Some(InstrumentType::Drumset) => (note.saturating_sub(self.lower_note)) as usize,
            Some(InstrumentType::MultiSample) => {
                for i in 0..8 {
                    if note <= self.region_end[i] {
                        return i;
                    }
                }
                7
            }
            _ => 0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct InstrumentBank {
    pub instruments: Vec<InstrumentRecord>,
}

pub fn calc_decay_coeff(vol: i32) -> i32 {
    if vol == 127 {
        0xFFFF
    } else if vol == 126 {
        0x3C00
    } else if vol < 50 {
        (vol * 2 + 1) & 0xFFFF
    } else {
        (0x1E00 / (126 - vol)) & 0xFFFF
    }
}

pub fn get_effective_attack(attack: i32) -> i32 {
    if attack < 109 {
        255 - attack
    } else {
        ATTACK_COEFF_TABLE[(127 - attack) as usize]
    }
}

pub fn get_sustain_level(sustain: i32) -> i32 {
    DECIBEL_SQUARE_TABLE[sustain as usize] << 7
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_coeff_boundaries() {
        assert_eq!(calc_decay_coeff(127), 0xFFFF);
        assert_eq!(calc_decay_coeff(126), 0x3C00);
        assert_eq!(calc_decay_coeff(0), 1);
        assert_eq!(calc_decay_coeff(49), 99);
        assert_eq!(calc_decay_coeff(50), 0x1E00 / 76);
        assert_eq!(calc_decay_coeff(100), 0x1E00 / 26);
    }

    #[test]
    fn effective_attack_uses_table_above_109() {
        assert_eq!(get_effective_attack(0), 255);
        assert_eq!(get_effective_attack(108), 147);
        assert_eq!(get_effective_attack(109), ATTACK_COEFF_TABLE[18]);
        assert_eq!(get_effective_attack(127), ATTACK_COEFF_TABLE[0]);
    }

    #[test]
    fn sustain_level_shifts_decibel_table() {
        assert_eq!(get_sustain_level(127), DECIBEL_SQUARE_TABLE[127] << 7);
        assert_eq!(get_sustain_level(0), DECIBEL_SQUARE_TABLE[0] << 7);
    }

    #[test]
    fn resolve_entry_index_multisample() {
        let mut rec = InstrumentRecord {
            f_record: 0x11,
            ..Default::default()
        };
        rec.region_end = [60, 72, 0x7F, 0, 0, 0, 0, 0];
        assert_eq!(rec.resolve_entry_index(48), 0);
        assert_eq!(rec.resolve_entry_index(60), 0);
        assert_eq!(rec.resolve_entry_index(61), 1);
        assert_eq!(rec.resolve_entry_index(100), 2);
    }

    #[test]
    fn resolve_entry_index_drumset_offsets_by_lower_note() {
        let rec = InstrumentRecord {
            f_record: 0x10,
            lower_note: 35,
            upper_note: 40,
            ..Default::default()
        };
        assert_eq!(rec.resolve_entry_index(35), 0);
        assert_eq!(rec.resolve_entry_index(38), 3);
    }
}

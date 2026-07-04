//! Instrument banks (SBNK): per-program records, sample regions, and ADSR coefficient math.

use super::tables::{ATTACK_COEFF_TABLE, DECIBEL_SQUARE_TABLE};

/// The kind of an instrument record (its `fRecord` byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstrumentType {
    /// Empty / unused slot.
    Empty,
    /// A single PCM/ADPCM sample.
    SingleSample,
    /// PSG square-wave pulse.
    PsgPulse,
    /// PSG noise.
    PsgNoise,
    /// A drumset: one region per note in `[lower_note, upper_note]`.
    Drumset,
    /// A multi-region instrument split by note range.
    MultiSample,
}

impl InstrumentType {
    /// Maps a raw `fRecord` byte to its type, if recognized.
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

/// One playable region within an instrument: which sample, its base note, and ADSR settings.
#[derive(Debug, Clone, Default)]
pub struct InstrumentRegion {
    /// Sample (SWAV) index within the linked archive.
    pub swav_info_id: u16,
    /// Index into the bank's four linked sample archives (SWAR).
    pub swar_info_id: u16,
    /// MIDI note the sample is tuned to.
    pub note_number: u8,
    /// Raw attack value (0..127).
    pub attack: u8,
    /// Precomputed attack coefficient.
    pub attack_coefficient: i32,
    /// Raw decay value.
    pub decay: u8,
    /// Precomputed decay coefficient.
    pub decay_coefficient: i32,
    /// Raw sustain value.
    pub sustain: u8,
    /// Precomputed sustain level.
    pub sustain_level: i32,
    /// Raw release value.
    pub release: u8,
    /// Precomputed release coefficient.
    pub release_coefficient: i32,
    /// Pan (0..127).
    pub pan: u8,
}

/// A single instrument (program) within a bank.
#[derive(Debug, Clone, Default)]
pub struct InstrumentRecord {
    /// Raw `fRecord` type byte.
    pub f_record: u8,
    /// Lowest note (drumsets).
    pub lower_note: u8,
    /// Highest note (drumsets).
    pub upper_note: u8,
    /// Region upper-note boundaries (multi-sample instruments).
    pub region_end: [u8; 8],
    /// Playable regions.
    pub regions: Vec<InstrumentRegion>,
}

impl InstrumentRecord {
    /// The typed kind of this record.
    pub fn instrument_type(&self) -> Option<InstrumentType> {
        InstrumentType::from_u8(self.f_record)
    }

    /// Resolves which region index plays `note`, mirroring the original `resolveEntryIndex`.
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

/// A decoded instrument bank: up to 128 programs.
#[derive(Debug, Clone, Default)]
pub struct InstrumentBank {
    /// Instrument records, indexed by program number.
    pub instruments: Vec<InstrumentRecord>,
}

/// Computes the decay/release coefficient for a volume value.
///
/// Thanks to ipatix and `pret/pokediamond`.
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

/// Remaps a raw attack value to its effective coefficient.
///
/// Thanks to ipatix and `pret/pokediamond`.
pub fn get_effective_attack(attack: i32) -> i32 {
    if attack < 109 {
        255 - attack
    } else {
        ATTACK_COEFF_TABLE[(127 - attack) as usize]
    }
}

/// Computes the sustain level from a raw sustain value.
///
/// Thanks to ipatix and `pret/pokediamond`.
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
        // vol < 50 branch: vol * 2 + 1.
        assert_eq!(calc_decay_coeff(0), 1);
        assert_eq!(calc_decay_coeff(49), 99);
        // vol >= 50 branch: 0x1E00 / (126 - vol).
        assert_eq!(calc_decay_coeff(50), 0x1E00 / 76);
        assert_eq!(calc_decay_coeff(100), 0x1E00 / 26);
    }

    #[test]
    fn effective_attack_uses_table_above_109() {
        // attack < 109: 255 - attack.
        assert_eq!(get_effective_attack(0), 255);
        assert_eq!(get_effective_attack(108), 147);
        // attack >= 109: ATTACK_COEFF_TABLE[127 - attack].
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
            f_record: 0x11, // MultiSample
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
            f_record: 0x10, // Drumset
            lower_note: 35,
            upper_note: 40,
            ..Default::default()
        };
        assert_eq!(rec.resolve_entry_index(35), 0);
        assert_eq!(rec.resolve_entry_index(38), 3);
    }
}

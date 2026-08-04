pub mod envelope;
pub mod events;
pub mod lfo;
pub mod pitch;
pub mod player;
pub mod sequencer;
pub mod smdl;
pub mod swdl;
pub mod volume;

pub use envelope::{EnvelopeParams, SoundEnvelope};
pub use events::{DseEvent, PAUSE_TICKS, control_info, decode_track};
pub use lfo::{Lfo, LfoConfig, LfoDest};
pub use pitch::note_key_to_hz;
pub use player::{DSE_CYCLES_PER_TICK, DsePlayer};
pub use sequencer::{DseSequencer, SeqOp};
pub use smdl::{Smdl, Track};
pub use swdl::{Program, SampleFormat, Split, Swdl, WaveformInfo};

use std::sync::Arc;

use crate::util::{read_u32, search_for_sequence};

pub fn find_smdl_offsets(data: &[u8]) -> Vec<usize> {
    search_for_sequence(data, b"smdl")
}

pub fn find_swdl_offsets(data: &[u8]) -> Vec<usize> {
    search_for_sequence(data, b"swdl")
}

struct DseSong {
    smdl: Smdl,
    bank: Arc<Swdl>,
    name: String,
}

pub struct DseSoundData {
    main_bank: Arc<Swdl>,
    songs: Vec<DseSong>,
}

impl DseSoundData {
    pub fn load_all(bytes: &[u8]) -> Option<DseSoundData> {
        let swdl_offsets = find_swdl_offsets(bytes);
        let smdl_offsets = find_smdl_offsets(bytes);
        if swdl_offsets.is_empty() || smdl_offsets.is_empty() {
            return None;
        }

        let blob = |off: usize| -> &[u8] {
            let len = read_u32(bytes, off + 0x08) as usize;
            &bytes[off..(off + len).min(bytes.len())]
        };

        let banks: Vec<Swdl> = swdl_offsets
            .iter()
            .filter_map(|&o| Swdl::parse(blob(o)))
            .collect();
        let main_bank = Arc::new(banks.iter().find(|b| !b.pcmd.is_empty())?.clone());

        let song_banks: Vec<Arc<Swdl>> = banks
            .into_iter()
            .filter(|b| b.pcmd.is_empty() && !b.programs.is_empty())
            .map(Arc::new)
            .collect();

        let songs: Vec<DseSong> = smdl_offsets
            .iter()
            .filter_map(|&o| Smdl::parse(blob(o)))
            .zip(song_banks)
            .map(|(smdl, bank)| {
                let name = if !bank.name.is_empty() {
                    bank.name.clone()
                } else {
                    smdl.name.clone()
                };
                DseSong { smdl, bank, name }
            })
            .collect();

        if songs.is_empty() {
            return None;
        }
        Some(DseSoundData { main_bank, songs })
    }

    pub fn song_ids(&self) -> Vec<u32> {
        (0..self.songs.len() as u32).collect()
    }

    pub fn song_name(&self, id: u32) -> Option<String> {
        self.songs.get(id as usize).map(|s| s.name.clone())
    }

    fn make_dse_player(&self, id: u32) -> Option<DsePlayer> {
        let song = self.songs.get(id as usize)?;
        Some(DsePlayer::new(
            &song.smdl,
            song.bank.clone(),
            self.main_bank.clone(),
        ))
    }
}

impl crate::devices::SoundData for DseSoundData {
    fn song_ids(&self) -> Vec<u32> {
        (0..self.songs.len() as u32).collect()
    }

    fn song_name(&self, id: u32) -> Option<String> {
        self.songs.get(id as usize).map(|s| s.name.clone())
    }

    fn make_player(&self, id: u32) -> Option<Box<dyn crate::devices::DevicePlayer>> {
        Some(Box::new(self.make_dse_player(id)?))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

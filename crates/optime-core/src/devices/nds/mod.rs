pub mod bank;
mod lfo;
mod player;
pub mod sdat;
pub mod sequence;
pub mod tables;
mod volume;

pub use bank::{InstrumentBank, InstrumentRecord, InstrumentType};
pub use player::NdsPlayer;
pub use sdat::{BankInfo, Sdat, SseqInfo, SwarInfo};
pub use sequence::{Message, MessageType, Sequence, SequenceTrack};
pub use volume::calc_channel_volume;

impl crate::devices::SoundData for Sdat {
    fn song_ids(&self) -> Vec<u32> {
        self.sseq_list.clone()
    }

    fn song_name(&self, id: u32) -> Option<String> {
        self.sseq_id_to_name.get(&id).cloned()
    }

    fn make_player(&self, id: u32) -> Option<Box<dyn crate::devices::DevicePlayer>> {
        Some(Box::new(NdsPlayer::new(self, id)?))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

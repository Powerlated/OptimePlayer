//! The Nintendo DS sound device: SDAT archives, the SSEQ sequencer, and the DS note/ADSR/LFO
//! hardware model (ported from `pret/pokediamond`).
//!
//! Data flow within this folder:
//!
//! ```text
//! .nds/.sdat bytes ─► sdat::Sdat (SYMB/INFO/FAT, banks)        — the archive
//! Sdat + song id   ─► player::NdsPlayer                        — decoded samples + sequencer
//! NdsPlayer::tick  ─► sequence::Sequence (SSEQ bytecode) ─► Message
//!                  ─► note lifecycle (ADSR via volume.rs, LFO via lfo.rs)
//!                  ─► standardized SynthEvent stream            — into the SynthController
//! ```

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

//! The Game Boy Advance sound device: GBA ROMs running the MP2K ("Sappy" / `m4a`) engine,
//! emulated from the `pret/pokeemerald` decompilation.
//!
//! Data flow within this folder:
//!
//! ```text
//! .gba bytes        ─► rom::GbaRom (song table + headers)        — the archive
//! GbaRom + song id  ─► player::GbaPlayer
//! GbaPlayer::tick   ─► sequencer::Mp2kSequencer (track bytecode) ─► Mp2kOp
//!                   ─► channel allocation + envelopes (player.rs, voice.rs, tables.rs)
//!                   ─► standardized SynthEvent stream             — into the SynthController
//! ```

mod extract;
mod player;
pub mod rom;
pub(crate) mod sequencer;
pub mod tables;
mod voice;

pub use extract::extract_audio;
pub use extract::sample_dc_stats;
pub use player::GbaPlayer;
pub use rom::GbaRom;

/// GBA CPU clock, in Hz.
pub const GBA_CLOCK_RATE: u64 = 16_777_216;

/// CPU cycles per LCD refresh — the MP2K engine runs once per VBlank (≈59.7275 Hz).
pub const CYCLES_PER_FRAME: u64 = 280_896;

/// The software mixer rate (`SOUND_MODE_FREQ_13379`) — the playback rate of fixed-frequency
/// voices and the rate every DirectSound voice is mixed at on hardware.
pub const ENGINE_RATE: f64 = 13379.0;

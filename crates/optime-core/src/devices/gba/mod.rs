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
pub use extract::waveform_dc_stats;
pub use player::GbaPlayer;
pub use rom::GbaRom;

/// GBA CPU clock, in Hz.
pub const GBA_CLOCK_RATE: u64 = 16_777_216;

/// CPU cycles per LCD refresh — the MP2K engine runs once per VBlank (≈59.7275 Hz).
pub const CYCLES_PER_FRAME: u64 = 280_896;

/// The software mixer rate (`SOUND_MODE_FREQ_13379`) — the playback rate of fixed-frequency
/// voices and the rate every DirectSound voice is mixed at on hardware.
pub const ENGINE_RATE: f64 = 13379.0;

impl crate::devices::SoundData for GbaRom {
    fn song_ids(&self) -> Vec<u32> {
        (0..self.song_count() as u32)
            .filter(|&id| self.song_header(id).is_some())
            .collect()
    }

    fn make_player(&self, id: u32) -> Option<Box<dyn crate::devices::DevicePlayer>> {
        Some(Box::new(GbaPlayer::new(self, id)?))
    }

    fn waveform_dc_stats(&self, id: u32) -> Vec<crate::devices::WaveformDcStat> {
        waveform_dc_stats(self, id)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

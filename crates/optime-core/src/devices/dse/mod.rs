//! Procyon Studios **DSE** ("Digital Sound Elements") sound engine — the format used by
//! *Pokémon Mystery Dungeon: Explorers of Sky* (and Time/Darkness).
//!
//! Unlike the standard NDS SDAT/SSEQ engine in [`super::nds`], DSE music is split into
//! two container types:
//!
//! - [`smdl::Smdl`] — a music **sequence** (`.smd`): a MIDI-like bytecode of notes, pauses, and
//!   control events, decoded by [`events`].
//! - [`swdl::Swdl`] — a sample/instrument **bank** (`.swd`). The game ships one shared *main
//!   bank* (`bgm.swd`) holding all PCM sample data, plus a *per-song bank* (`bgm####.swd`) of
//!   programs/keygroups whose splits reference the main bank's samples by index.
//!
//! On top of the parsers this module is a full playback backend, wired into
//! [`super::SoundData`]/[`super::DevicePlayer`]: [`sequencer`] interprets the SMDL bytecode,
//! [`envelope`] runs the volume slide, [`pitch`] and [`volume`] reproduce the driver's voice
//! update (note→frequency tables and the square-law volume), and [`player`] ties them together
//! into the standardized [`SynthEvent`](super::SynthEvent) stream. The decode path is also
//! exercised standalone by the `dump_dse` example.
//!
//! Every offset/table here is transcribed from the `pret/pmd-sky` decompilation (the `lib/DSE`
//! engine sources and the real `files/SOUND/BGM` banks) — see the per-module docs.

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

/// Finds the byte offset of every SMDL (`smdl`) sequence in `data` (e.g. a whole ROM or a
/// concatenation of `.smd` files).
pub fn find_smdl_offsets(data: &[u8]) -> Vec<usize> {
    search_for_sequence(data, b"smdl")
}

/// Finds the byte offset of every SWDL (`swdl`) bank in `data`.
pub fn find_swdl_offsets(data: &[u8]) -> Vec<usize> {
    search_for_sequence(data, b"swdl")
}

/// One playable DSE song: its sequence, its per-song bank, and a display name.
struct DseSong {
    smdl: Smdl,
    bank: Arc<Swdl>,
    name: String,
}

/// A loaded DSE sound archive: the shared main bank plus every paired song.
///
/// Built by scanning a PMD ROM (or a concatenation of the loose `Data/Sound/BGM` files) for
/// `swdl`/`smdl` magics. The single bank carrying `pcmd` sample data is the main bank; the rest
/// are per-song banks, paired with the sequences in file order.
pub struct DseSoundData {
    main_bank: Arc<Swdl>,
    songs: Vec<DseSong>,
}

impl DseSoundData {
    /// Parses every DSE song found in `bytes`, or `None` if it isn't a DSE archive.
    pub fn load_all(bytes: &[u8]) -> Option<DseSoundData> {
        let swdl_offsets = find_swdl_offsets(bytes);
        let smdl_offsets = find_smdl_offsets(bytes);
        if swdl_offsets.is_empty() || smdl_offsets.is_empty() {
            return None;
        }

        // Slice each blob to its declared length (header +0x08) before parsing.
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

    /// Ids of the playable songs (their indices).
    pub fn song_ids(&self) -> Vec<u32> {
        (0..self.songs.len() as u32).collect()
    }

    /// The embedded display name for a song id (DSE banks carry real names).
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

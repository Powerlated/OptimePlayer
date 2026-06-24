//! Procyon Studios **DSE** ("Digital Sound Elements") sound engine — the format used by
//! *Pokémon Mystery Dungeon: Explorers of Sky* (and Time/Darkness).
//!
//! Unlike the standard NDS SDAT/SSEQ engine in [`super::nintendo_ds`], DSE music is split into
//! two container types:
//!
//! - [`smdl::Smdl`] — a music **sequence** (`.smd`): a MIDI-like bytecode of notes, pauses, and
//!   control events, decoded by [`events`].
//! - [`swdl::Swdl`] — a sample/instrument **bank** (`.swd`). The game ships one shared *main
//!   bank* (`bgm.swd`) holding all PCM sample data, plus a *per-song bank* (`bgm####.swd`) of
//!   programs/keygroups whose splits reference the main bank's samples by index.
//!
//! This module currently provides parsing + sample decoding (the decode path), which is
//! verified end-to-end by the `dump_dse` example. A full playback backend (an SMDL sequencer +
//! the DSE envelope/LFO synth wired into [`super::SoundData`]/[`super::DevicePlayer`]) builds on
//! top of these parsers.
//!
//! Every offset/table here is transcribed from the `pret/pmd-sky` decompilation (the `lib/DSE`
//! engine sources and the real `files/SOUND/BGM` banks) — see the per-module docs.

pub mod events;
pub mod smdl;
pub mod swdl;

pub use events::{control_info, decode_track, DseEvent, PAUSE_TICKS};
pub use smdl::{Smdl, Track};
pub use swdl::{SampleFormat, SampleInfo, Swdl};

use crate::util::search_for_sequence;

/// Finds the byte offset of every SMDL (`smdl`) sequence in `data` (e.g. a whole ROM or a
/// concatenation of `.smd` files).
pub fn find_smdl_offsets(data: &[u8]) -> Vec<usize> {
    search_for_sequence(data, b"smdl")
}

/// Finds the byte offset of every SWDL (`swdl`) bank in `data`.
pub fn find_swdl_offsets(data: &[u8]) -> Vec<usize> {
    search_for_sequence(data, b"swdl")
}

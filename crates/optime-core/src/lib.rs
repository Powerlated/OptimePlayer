#![cfg_attr(feature = "simd", feature(portable_simd))]
//! Platform-independent emulation of retro console sound systems.
//!
//! The crate is organized around a single data flow:
//!
//! ```text
//! ROM / archive bytes ─► devices::SoundData          (per-console parsing: songs, instruments)
//! SoundData + song id ─► devices::DevicePlayer       (per-console sequencer + envelope model)
//! DevicePlayer::tick  ─► devices::SynthEvent stream  (the standardized message set)
//! SynthEvent stream   ─► SynthController             (voice pools, master clock, mixing)
//! ```
//!
//! Each console lives in its own folder under [`devices`] (`nintendo_ds`, `gba`); the
//! synthesis layer ([`synth_controller`], [`synth`], [`dsp`] incl. `dsp::resample`) is shared and
//! knows nothing about any console's formats.
//!
//! The engine is deliberately free of any I/O or platform dependencies: feed it bytes, pull
//! samples. The browser/audio/UI concerns live in the `optime-app` crate.

pub mod devices;
mod dsp;
pub mod sample;
pub mod synth;
pub mod synth_controller;
pub mod tuning;
pub mod util;

pub use devices::nintendo_ds::{
    calc_channel_volume, BankInfo, InstrumentBank, InstrumentRecord, InstrumentType, Message,
    MessageType, Sdat, Sequence, SequenceTrack, SseqInfo, SwarInfo,
};
pub use devices::{DevicePlayer, SoundData, SynthEvent, VoiceId, VoicePitch};
pub use dsp::biquad_filter;
pub use dsp::resample::ResampleTables;
pub use sample::{
    decode_adpcm, decode_pcm16, decode_pcm8, decode_wav, InstrumentResampleMode, Sample,
};
pub use synth::{DelayLine, SampleInstrument, SampleSynthesizer, CROSSOVER_Q};
pub use synth_controller::{
    DelaySmoothing, FsVisController, HighShelf, PopSmoothing, SynthConfig, SynthController, VisNote,
};
pub use tuning::{midi_note_to_hz, TuningSystem};

/// Number of sequence tracks the synthesis layer exposes (both consoles fit in 16).
pub const TRACK_COUNT: usize = 16;

/// DS system clock, in Hz. The DS sequence timer is driven from this.
pub const DS_CLOCK_RATE: u64 = 33_513_982;

/// Number of DS clock cycles between sequence ticks (`64 * 2728`).
pub const CYCLES_PER_TICK: u64 = 64 * 2728;

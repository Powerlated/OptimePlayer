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
//! Each console lives in its own folder under [`devices`] (`nds`, `gba`); the
//! synthesis layer ([`synth_controller`], [`synth`], [`dsp`] incl. `dsp::resample`) is shared and
//! knows nothing about any console's formats.
//!
//! The engine is deliberately free of any I/O or platform dependencies: feed it bytes, pull
//! samples. The browser/audio/UI concerns live in the `optime-app` crate.

pub mod device_settings;
pub mod devices;
pub mod dsp;
pub mod synth;
pub mod synth_controller;
pub mod tuning;
pub mod util;
pub mod waveform;

pub use device_settings::{
    InstrumentResampleChoice, InstrumentResampleSettings, MixerResampleSettings, PerDeviceSettings,
    PopSmoothingEdge,
};
pub use devices::nds::{
    BankInfo, InstrumentBank, InstrumentRecord, InstrumentType, Message, MessageType, Sdat,
    Sequence, SequenceTrack, SseqInfo, SwarInfo, calc_channel_volume,
};
pub use devices::{
    DevicePlayer, SoundData, SynthEvent, VoiceId, VoicePitch, WaveformDcStat, load_all,
};
pub use dsp::biquad_filter;
pub use dsp::resample::{ResampleTables, StreamResampler};
pub use synth::{CROSSOVER_Q, DelayLine, WaveformInstrument, WaveformSynthesizer};
pub use synth_controller::{
    DelaySmoothing, FsVisController, HighShelf, LoopAndTransitionOptions, PlaybackEvent,
    PopSmoothing, SongOverview, SynthController, VisNote,
};
pub use tuning::{TuningSystem, midi_note_to_hz};
pub use waveform::{
    Frame, InstrumentResampleMode, Sample, Waveform, decode_adpcm, decode_pcm8, decode_pcm16,
    decode_wav,
};

/// Number of sequence tracks the synthesis layer exposes (both consoles fit in 16).
pub const TRACK_COUNT: usize = 16;

/// Size in bytes of one [`Sample`] (the signal-path amplitude type). Lets tools report whether the
/// engine was built with `Sample = f64` (8) or `Sample = f32` (4) without knowing the alias.
pub const SAMPLE_SIZE_BYTES: usize = core::mem::size_of::<Sample>();

/// DS system clock, in Hz. The DS sequence timer is driven from this.
pub const DS_CLOCK_RATE: u64 = 33_513_982;

/// Number of DS clock cycles between sequence ticks (`64 * 2728`).
pub const CYCLES_PER_TICK: u64 = 64 * 2728;

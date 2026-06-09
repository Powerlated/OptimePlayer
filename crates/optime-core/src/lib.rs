//! Platform-independent emulation of the Nintendo DS sound system.
//!
//! This is an idiomatic Rust port of the original `OptimePlayer.js` engine. It parses SDAT
//! sound archives out of `.nds` ROMs / standalone `.sdat` files, interprets SSEQ sequence
//! bytecode, and software-synthesizes it into stereo audio.
//!
//! The engine is deliberately free of any I/O or platform dependencies: feed it bytes, pull
//! samples. The browser/audio/UI concerns live in the `optime-app` crate.
//!
//! ## Pipeline
//!
//! 1. [`Sdat::load_all`] scans a buffer for SDAT containers and parses each one.
//! 2. [`Controller::new`] binds an SSEQ + its instrument bank + sample archives at a sample rate.
//! 3. [`Controller::next_sample`] (or [`Controller::fill`]) pulls stereo samples, advancing the
//!    DS master clock internally — the single place the hardware tick math lives.

pub mod bank;
pub mod controller;
pub mod dsp;
pub mod resample;
pub mod sample;
pub mod sdat;
pub mod sequence;
pub mod synth;
pub mod tables;
pub mod tuning;
pub mod util;

pub use bank::{InstrumentBank, InstrumentRecord, InstrumentType};
pub use controller::{Controller, FsVisController, PitchBendEvent, SynthConfig};
pub use dsp::BiquadFilter;
pub use resample::{fir_kernel, fir_response, ResampleTables};
pub use sample::{decode_adpcm, decode_pcm16, decode_pcm8, decode_wav, ResampleMode, Sample};
pub use sdat::{BankInfo, Sdat, SseqInfo, SwarInfo};
pub use sequence::{Message, MessageType, Sequence, SequenceTrack};
pub use synth::{DelayLine, SampleInstrument, SampleSynthesizer, CROSSOVER_Q};
pub use tuning::{midi_note_to_hz, TuningSystem};

/// Number of sequence tracks the DS sound system exposes.
pub const TRACK_COUNT: usize = 16;

/// DS system clock, in Hz. The sequence timer is driven from this.
pub const DS_CLOCK_RATE: u64 = 33_513_982;

/// Number of DS clock cycles between sequence ticks (`64 * 2728`).
pub const CYCLES_PER_TICK: u64 = 64 * 2728;

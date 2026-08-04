#![feature(portable_simd)]

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

pub const TRACK_COUNT: usize = 16;

pub const SAMPLE_SIZE_BYTES: usize = core::mem::size_of::<Sample>();

pub const DS_CLOCK_RATE: u64 = 33_513_982;

pub const CYCLES_PER_TICK: u64 = 64 * 2728;

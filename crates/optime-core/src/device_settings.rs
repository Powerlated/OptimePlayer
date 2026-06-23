//! Per-console synth/audio settings bundle.
//!
//! These are the user-tunable knobs that belong to a single console (the DS and GBA each keep
//! their own copy). They live in the core — rather than the app — so they can be threaded straight
//! into the synthesis layer. Defaults are intentionally *not* defined here; the app owns the
//! out-of-the-box values.

use crate::synth_controller::HighShelf;

/// Which resampling algorithm a stage uses. Mirrors the variants of
/// [`InstrumentResampleMode`](crate::InstrumentResampleMode) at the settings level.
#[derive(Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum InstrumentResampleChoice {
    Nearest,
    Linear,
    SincOutputNyquist,
    SincSampleNyquist,
}

impl InstrumentResampleChoice {
    pub fn text(&self) -> &'static str {
        match self {
            InstrumentResampleChoice::Nearest => "Nearest neighbour",
            InstrumentResampleChoice::Linear => "Linear",
            InstrumentResampleChoice::SincOutputNyquist => "Sinc – output Nyquist (crunch)",
            InstrumentResampleChoice::SincSampleNyquist => "Sinc – sample Nyquist (clean)",
        }
    }
}

/// Per-device resampling settings — each console keeps its own, so e.g. the DS can play
/// Crunchy sinc while the GBA plays Clean sinc.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct InstrumentResampleSettings {
    /// Resampling choice enum
    pub choice: InstrumentResampleChoice,
    /// Total source-tap count for the sinc/reconstruction kernel.
    pub sinc_taps: usize,
    /// Crunchy-mode low-pass cutoff (Hz) for PSG voices.
    pub psg_cutoff_hz: u32,
    /// Crunchy-mode low-pass cutoff (Hz) for DirectSound/sampled voices.
    pub sampler_cutoff_hz: u32,
    /// Smooth out PSG on/off pops (a gain slew) instead of preserving the clicks. Applies in
    /// every resampling mode.
    pub smooth_psg_pops: bool,
    /// Smooth out sampled (DirectSound/SWAR) voice pops/clicks. Applies in every resampling mode.
    pub smooth_sample_pops: bool,
}

/// Mixer-to-output resampling settings. Reuses the same algorithm choice as the per-instrument
/// stage ([`InstrumentResampleChoice`]); the bus is a finished mix (no PSG/sampled split), so the
/// crunch mode carries a single `cutoff_hz` rather than the per-kind PSG/sampler cutoffs.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct MixerResampleSettings {
    /// Resampling choice enum (shared with the instrument stage).
    pub choice: InstrumentResampleChoice,
    /// Total source-tap count for the sinc/reconstruction kernel.
    pub sinc_taps: usize,
    /// Crunchy-mode low-pass cutoff (Hz) for the bus.
    pub cutoff_hz: u32,
}

/// The synth/audio settings that belong to a single console — the DS and GBA each keep their own
/// copy, so e.g. one can run crunchy resampling and a high-shelf cut while the other stays clean.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct PerDeviceSettings {
    pub stereo_separation: bool,
    pub force_stereo_separation: bool,
    pub bass_mono: bool,
    pub bass_mono_freq: f32,
    pub tuning_choice: usize,
    pub pure_tonic: i32,
    pub instrument_resample: InstrumentResampleSettings,
    pub mixer_resample: MixerResampleSettings,
    /// Per-device master high-shelf EQ applied to the final mix.
    pub shelf: HighShelf,
    /// Stereo-expander delay-change handling: 0 = immediate, 1 = hold during notes.
    pub delay_smoothing_choice: usize,
    pub mixer_sample_rate: u32,
    /// Route sampled (non-PSG) voices through the intermediate mixer (then upsample to output).
    pub use_mixer: bool,
    pub psg_crunch_compensation: bool,
}

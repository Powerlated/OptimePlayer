//! Per-console synth/audio settings bundle.
//!
//! These are the user-tunable knobs that belong to a single console (the DS and GBA each keep
//! their own copy). They live in the core — rather than the app — so they can be threaded straight
//! into the synthesis layer. Defaults are intentionally *not* defined here; the app owns the
//! out-of-the-box values.

use crate::TRACK_COUNT;
use crate::synth_controller::{DelaySmoothing, HighShelf, PopSmoothing};
use crate::tuning::TuningSystem;
use crate::waveform::InstrumentResampleMode;

/// Which resampling algorithm a stage uses. Mirrors the variants of
/// [`InstrumentResampleMode`](crate::InstrumentResampleMode) at the settings level.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
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
    /// How long (milliseconds) the PSG/sample de-click gain ramp takes. `#[serde(default)]` so old
    /// saved settings (which predate the field) load as the previous fixed 2 ms.
    #[serde(default = "default_pop_slew_ms")]
    pub pop_slew_ms: f32,
}

/// The previous fixed de-click ramp time (2 ms); the serde fallback for `pop_slew_ms` so old saves
/// keep their original behavior.
fn default_pop_slew_ms() -> f32 {
    2.0
}

/// The GBA's native mixer bit depth (8-bit); the serde fallback for `bitcrush_bits` so old saves
/// load with the hardware value.
fn default_bitcrush_bits() -> u32 {
    8
}

impl InstrumentResampleSettings {
    /// Resolve the choice + per-kind cutoffs into the concrete [`InstrumentResampleMode`] the
    /// synthesis layer consumes. `half_taps` is half the (even) source-tap count, at least 1.
    pub fn mode(&self) -> InstrumentResampleMode {
        resolve_mode(
            &self.choice,
            self.sinc_taps,
            self.psg_cutoff_hz,
            self.sampler_cutoff_hz,
        )
    }
}

/// Mixer-to-output resampling settings. Reuses the same algorithm choice as the per-instrument
/// stage ([`InstrumentResampleChoice`]); the bus is a finished mix (no PSG/sampled split), so the
/// crunch mode carries a single `cutoff_hz` rather than the per-kind PSG/sampler cutoffs.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct MixerResampleSettings {
    /// Resampling choice enum (shared with the instrument stage).
    pub choice: InstrumentResampleChoice,
    /// Total source-tap count for the sinc/reconstruction kernel.
    pub sinc_taps: usize,
    /// Crunchy-mode low-pass cutoff (Hz) for the bus.
    pub cutoff_hz: u32,
}

impl MixerResampleSettings {
    /// Resolve into the concrete [`InstrumentResampleMode`]. The bus is a finished (non-PSG) mix,
    /// so the single `cutoff_hz` feeds both per-kind cutoff slots.
    pub fn mode(&self) -> InstrumentResampleMode {
        resolve_mode(&self.choice, self.sinc_taps, self.cutoff_hz, self.cutoff_hz)
    }
}

/// Shared choice → [`InstrumentResampleMode`] resolution for both resampling stages.
fn resolve_mode(
    choice: &InstrumentResampleChoice,
    sinc_taps: usize,
    psg_cutoff_hz: u32,
    sampler_cutoff_hz: u32,
) -> InstrumentResampleMode {
    let half_taps = (sinc_taps / 2).max(1);
    match choice {
        InstrumentResampleChoice::Nearest => InstrumentResampleMode::NearestNeighbor,
        InstrumentResampleChoice::Linear => InstrumentResampleMode::Linear,
        InstrumentResampleChoice::SincSampleNyquist => {
            InstrumentResampleMode::SincSampleNyquist { half_taps }
        }
        InstrumentResampleChoice::SincOutputNyquist => InstrumentResampleMode::SincOutputNyquist {
            half_taps,
            psg_cutoff_hz,
            sampler_cutoff_hz,
        },
    }
}

/// Every track enabled — the default for the runtime-only [`PerDeviceSettings::track_enables`]
/// (so old persisted data, which never stored it, deserializes to "all on").
fn all_tracks_enabled() -> [bool; TRACK_COUNT] {
    [true; TRACK_COUNT]
}

/// The synth/audio settings that belong to a single console — the DS and GBA each keep their own
/// copy, so e.g. one can run crunchy resampling and a high-shelf cut while the other stays clean.
///
/// This is also the runtime config the synthesis layer consumes directly: the resolver methods
/// ([`Self::resample`], [`Self::tuning`], …) turn the settings-level choices into the concrete
/// values the engine wants, so there is no separate "resolved config" struct.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct PerDeviceSettings {
    pub stereo_separation: bool,
    pub force_stereo_separation: bool,
    /// Slew the left/right pan gains over a few milliseconds on a pan change instead of stepping
    /// them, so an abrupt pan jump doesn't click. `#[serde(default)]` so old saved settings (which
    /// predate the field) load as `false`.
    #[serde(default)]
    pub smooth_pan: bool,
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
    /// Quantize the intermediate mixer bus to [`Self::bitcrush_bits`]-bit signed, truncating and
    /// wrapping on overflow exactly like the GBA m4a software mixer's 8-bit DirectSound buffer
    /// (`m4a_1.s` `SoundMainRAM`). Only engages when `use_mixer` is on. `#[serde(default)]` so old
    /// saves load as `false`.
    #[serde(default)]
    pub bitcrush_mixer: bool,
    /// The signed bit depth the mixer bus is crushed to when [`Self::bitcrush_mixer`] is on (8 =
    /// the GBA's native 8-bit). `#[serde(default)]` so old saves load as the 8-bit hardware value.
    #[serde(default = "default_bitcrush_bits")]
    pub bitcrush_bits: u32,
    pub psg_crunch_compensation: bool,
    /// Apply the MP2K (GBA) reverb: a mono-summing feedback delay on the sampled bus, using the
    /// song's `soundInfo.reverb` amount. Only engages when `use_mixer` is on (the sampled bus is
    /// isolated there). `#[serde(default)]` so old saves load as `false`.
    #[serde(default)]
    pub mp2k_reverb: bool,
    /// Subtract each decoded sample's DC offset (GBA DirectSound only) to match the console's
    /// AC-coupled output. Off by default — it changes the raw sample data, so it's opt-in.
    /// `#[serde(default)]` so old saves load as `false`.
    #[serde(default)]
    pub remove_sample_dc_offset: bool,
    /// Which of the 16 tracks are mixed into the output. Runtime UI state (the app injects the
    /// live piano-roll mutes each frame), **not** persisted — deserializing old data falls back to
    /// "all enabled".
    #[serde(skip, default = "all_tracks_enabled")]
    pub track_enables: [bool; TRACK_COUNT],
}

impl PerDeviceSettings {
    /// The per-voice sample interpolation / anti-aliasing mode.
    pub fn resample(&self) -> InstrumentResampleMode {
        self.instrument_resample.mode()
    }

    /// How the intermediate mixer bus is brought up to the output rate. (The field
    /// `mixer_resample` is the settings struct; this resolves it to the concrete mode.)
    pub fn mixer_resample_mode(&self) -> InstrumentResampleMode {
        self.mixer_resample.mode()
    }

    /// The active tuning system (`tuning_choice == 0` is equal temperament; otherwise Pythagorean
    /// pure tuning anchored at `pure_tonic`).
    pub fn tuning(&self) -> TuningSystem {
        if self.tuning_choice == 0 {
            TuningSystem::Equal
        } else {
            TuningSystem::Pure {
                tonic: self.pure_tonic,
            }
        }
    }

    /// Per-voice-kind de-click gain slew. Orthogonal to the resampling mode, so it applies in every
    /// mode.
    pub fn pop_smoothing(&self) -> PopSmoothing {
        PopSmoothing {
            psg: self.instrument_resample.smooth_psg_pops,
            sampled: self.instrument_resample.smooth_sample_pops,
            slew_seconds: f64::from(self.instrument_resample.pop_slew_ms.max(0.0)) / 1000.0,
        }
    }

    /// How the stereo expander handles delay-line length changes.
    pub fn delay_smoothing(&self) -> DelaySmoothing {
        match self.delay_smoothing_choice {
            1 => DelaySmoothing::HoldDuringNotes,
            _ => DelaySmoothing::None,
        }
    }

    /// An engine-neutral baseline — every effect off, nearest-neighbour resampling, equal tuning,
    /// all tracks enabled. This is *not* the user-facing default (the app owns those in
    /// `Persisted::default`); it exists so the core's tests and examples have a config to start
    /// from and override (`..PerDeviceSettings::neutral()`).
    pub fn neutral() -> Self {
        Self {
            stereo_separation: false,
            force_stereo_separation: false,
            smooth_pan: false,
            bass_mono: false,
            bass_mono_freq: 200.0,
            tuning_choice: 0,
            pure_tonic: 0,
            instrument_resample: InstrumentResampleSettings {
                choice: InstrumentResampleChoice::Nearest,
                sinc_taps: 32,
                psg_cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
                sampler_cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
                smooth_psg_pops: false,
                smooth_sample_pops: false,
                pop_slew_ms: 2.0,
            },
            // Clean reconstruction is the sane default for upsampling a finished bus.
            mixer_resample: MixerResampleSettings {
                choice: InstrumentResampleChoice::SincSampleNyquist,
                sinc_taps: 32,
                cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
            },
            shelf: HighShelf::default(),
            delay_smoothing_choice: 0,
            mixer_sample_rate: 48_000,
            use_mixer: false,
            bitcrush_mixer: false,
            bitcrush_bits: 8,
            psg_crunch_compensation: false,
            mp2k_reverb: false,
            remove_sample_dc_offset: false,
            track_enables: all_tracks_enabled(),
        }
    }

    /// The high-quality Nintendo DS preset: the app's out-of-the-box DS settings (mixer on at
    /// 32768 Hz, crunchy output-Nyquist sinc voices, clean sinc mixer upsample, stereo separation
    /// and the high-shelf de-harsher). Shared by `Persisted::default` (the app) and offline tools.
    pub fn high_quality_nintendo_ds() -> Self {
        Self {
            stereo_separation: true,
            force_stereo_separation: false,
            smooth_pan: true,
            delay_smoothing_choice: 1,
            bass_mono: true,
            bass_mono_freq: 200.0,
            tuning_choice: 0,
            pure_tonic: 0,
            instrument_resample: InstrumentResampleSettings {
                choice: InstrumentResampleChoice::SincOutputNyquist,
                sinc_taps: 32,
                psg_cutoff_hz: 15_000,
                sampler_cutoff_hz: 15_000,
                smooth_psg_pops: false,
                smooth_sample_pops: false,
                pop_slew_ms: 2.0,
            },
            use_mixer: true,
            mixer_sample_rate: 32768,
            bitcrush_mixer: false,
            bitcrush_bits: 8,
            psg_crunch_compensation: true,
            mp2k_reverb: false,
            remove_sample_dc_offset: false,
            mixer_resample: MixerResampleSettings {
                choice: InstrumentResampleChoice::SincSampleNyquist,
                sinc_taps: 32,
                cutoff_hz: 15_000,
            },
            shelf: HighShelf {
                enabled: true,
                order: 2,
                q: 0.5,
                cutoff_hz: 12700.0,
                gain_db: -10.0,
            },
            track_enables: all_tracks_enabled(),
        }
    }

    /// The "Enhanced" Game Boy Advance preset: the app's out-of-the-box GBA settings (mixer on at
    /// the GBA-native 13379 Hz with crunchy output-Nyquist mixer upsample, clean sinc voices with
    /// pop-smoothing, stereo separation and a deeper high-shelf). The polished counterpart to
    /// [`Self::original_gba`]. Shared by `Persisted::default` (the app) and offline tools.
    pub fn enhanced_gba() -> Self {
        Self {
            stereo_separation: true,
            force_stereo_separation: false,
            smooth_pan: true,
            delay_smoothing_choice: 1,
            bass_mono: true,
            bass_mono_freq: 200.0,
            tuning_choice: 0,
            pure_tonic: 0,
            instrument_resample: InstrumentResampleSettings {
                choice: InstrumentResampleChoice::SincSampleNyquist,
                sinc_taps: 32,
                psg_cutoff_hz: 15_000,
                sampler_cutoff_hz: 15_000,
                smooth_psg_pops: false,
                smooth_sample_pops: false,
                pop_slew_ms: 2.0,
            },
            use_mixer: true,
            mixer_sample_rate: 13379,
            bitcrush_mixer: false,
            bitcrush_bits: 8,
            psg_crunch_compensation: true,
            mp2k_reverb: true,
            remove_sample_dc_offset: false,
            mixer_resample: MixerResampleSettings {
                choice: InstrumentResampleChoice::SincOutputNyquist,
                sinc_taps: 32,
                cutoff_hz: 13379,
            },
            shelf: HighShelf {
                enabled: true,
                order: 6,
                q: 0.707,
                cutoff_hz: 14000.0,
                gain_db: -24.0,
            },
            track_enables: all_tracks_enabled(),
        }
    }

    /// The "Original" Game Boy Advance preset: the raw m4a signal chain with none of the enhancement
    /// DSP. No stereo widening or smoothing, instrument→mixer linear interpolation, the intermediate
    /// mixer at the GBA-native 13379 Hz crushed to 8-bit (as `m4a_1.s` renders DirectSound), a
    /// nearest-neighbour mixer→output upsample, and no high-shelf EQ. The MP2K reverb pre-pass stays
    /// on — it is part of the original engine, applied at the amount each song requests.
    pub fn original_gba() -> Self {
        Self {
            stereo_separation: false,
            force_stereo_separation: false,
            smooth_pan: false,
            delay_smoothing_choice: 0,
            bass_mono: false,
            bass_mono_freq: 200.0,
            tuning_choice: 0,
            pure_tonic: 0,
            instrument_resample: InstrumentResampleSettings {
                choice: InstrumentResampleChoice::Linear,
                sinc_taps: 32,
                psg_cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
                sampler_cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
                smooth_psg_pops: false,
                smooth_sample_pops: false,
                pop_slew_ms: 2.0,
            },
            use_mixer: true,
            mixer_sample_rate: 13379,
            bitcrush_mixer: true,
            bitcrush_bits: 8,
            psg_crunch_compensation: false,
            mp2k_reverb: true,
            remove_sample_dc_offset: false,
            mixer_resample: MixerResampleSettings {
                choice: InstrumentResampleChoice::Nearest,
                sinc_taps: 32,
                cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
            },
            shelf: HighShelf {
                enabled: false,
                ..HighShelf::default()
            },
            track_enables: all_tracks_enabled(),
        }
    }
}

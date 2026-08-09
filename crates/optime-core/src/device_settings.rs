//! The per-console settings: what the app stores, and how each stored choice resolves to a concrete engine value.

use crate::TRACK_COUNT;
use crate::dsp::slewer::Direction;
use crate::synth_controller::{
    DelaySmoothing, Exciter, HighBandCompressor, HighShelf, PopSmoothing,
};
use crate::tuning::TuningSystem;
use crate::waveform::InstrumentResampleMode;

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

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq, Default)]
pub enum PopSmoothingEdge {
    Attack,
    Release,
    #[default]
    Both,
}

impl PopSmoothingEdge {
    pub fn text(self) -> &'static str {
        match self {
            PopSmoothingEdge::Attack => "Attack",
            PopSmoothingEdge::Release => "Release",
            PopSmoothingEdge::Both => "Both",
        }
    }

    pub fn direction(self) -> Direction {
        match self {
            PopSmoothingEdge::Attack => Direction::UpOnly,
            PopSmoothingEdge::Release => Direction::DownOnly,
            PopSmoothingEdge::Both => Direction::UpAndDown,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct InstrumentResampleSettings {
    pub choice: InstrumentResampleChoice,
    pub sinc_taps: usize,
    pub psg_cutoff_hz: u32,
    pub sampler_cutoff_hz: u32,
    pub smooth_psg_pops: bool,
    pub smooth_sample_pops: bool,
    #[serde(default = "default_pop_slew_ms")]
    pub pop_slew_ms: f32,
    #[serde(default)]
    pub pop_smooth_edge: PopSmoothingEdge,
}

fn default_pop_slew_ms() -> f32 {
    2.0
}

fn default_bitcrush_bits() -> u32 {
    8
}

impl InstrumentResampleSettings {
    pub fn mode(&self) -> InstrumentResampleMode {
        resolve_mode(
            &self.choice,
            self.sinc_taps,
            self.psg_cutoff_hz,
            self.sampler_cutoff_hz,
        )
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct MixerResampleSettings {
    pub choice: InstrumentResampleChoice,
    pub sinc_taps: usize,
    pub cutoff_hz: u32,
}

impl MixerResampleSettings {
    pub fn mode(&self) -> InstrumentResampleMode {
        resolve_mode(&self.choice, self.sinc_taps, self.cutoff_hz, self.cutoff_hz)
    }
}

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

fn all_tracks_enabled() -> [bool; TRACK_COUNT] {
    [true; TRACK_COUNT]
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct PerDeviceSettings {
    pub stereo_separation: bool,
    pub force_stereo_separation: bool,
    #[serde(default)]
    pub smooth_pan: bool,
    pub bass_mono: bool,
    pub bass_mono_freq: f32,
    pub tuning_choice: usize,
    pub pure_tonic: i32,
    pub instrument_resample: InstrumentResampleSettings,
    pub mixer_resample: MixerResampleSettings,
    pub shelf: HighShelf,
    #[serde(default)]
    pub high_band_compress: HighBandCompressor,
    #[serde(default)]
    pub exciter: Exciter,
    pub delay_smoothing_choice: usize,
    pub mixer_sample_rate: u32,
    pub use_mixer: bool,
    #[serde(default)]
    pub bitcrush_mixer: bool,
    #[serde(default = "default_bitcrush_bits")]
    pub bitcrush_bits: u32,
    pub psg_crunch_compensation: bool,
    #[serde(default)]
    pub mp2k_reverb: bool,
    #[serde(default)]
    pub remove_sample_dc_offset: bool,
    #[serde(skip, default = "all_tracks_enabled")]
    pub track_enables: [bool; TRACK_COUNT],
}

impl PerDeviceSettings {
    pub fn resample(&self) -> InstrumentResampleMode {
        self.instrument_resample.mode()
    }

    pub fn mixer_resample_mode(&self) -> InstrumentResampleMode {
        self.mixer_resample.mode()
    }

    pub fn tuning(&self) -> TuningSystem {
        if self.tuning_choice == 0 {
            TuningSystem::Equal
        } else {
            TuningSystem::Pure {
                tonic: self.pure_tonic,
            }
        }
    }

    pub fn pop_smoothing(&self) -> PopSmoothing {
        PopSmoothing {
            psg: self.instrument_resample.smooth_psg_pops,
            sampled: self.instrument_resample.smooth_sample_pops,
            slew_seconds: f64::from(self.instrument_resample.pop_slew_ms.max(0.0)) / 1000.0,
            direction: self.instrument_resample.pop_smooth_edge.direction(),
        }
    }

    pub fn delay_smoothing(&self) -> DelaySmoothing {
        match self.delay_smoothing_choice {
            1 => DelaySmoothing::HoldDuringNotes,
            _ => DelaySmoothing::None,
        }
    }

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
                pop_smooth_edge: PopSmoothingEdge::Both,
            },
            mixer_resample: MixerResampleSettings {
                choice: InstrumentResampleChoice::SincSampleNyquist,
                sinc_taps: 32,
                cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
            },
            shelf: HighShelf::default(),
            high_band_compress: HighBandCompressor::default(),
            exciter: Exciter::default(),
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
                pop_smooth_edge: PopSmoothingEdge::Both,
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
            high_band_compress: HighBandCompressor::default(),
            exciter: Exciter::default(),
            track_enables: all_tracks_enabled(),
        }
    }

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
                smooth_sample_pops: true,
                pop_slew_ms: 10.0,
                pop_smooth_edge: PopSmoothingEdge::Release,
            },
            use_mixer: true,
            mixer_sample_rate: 13379,
            bitcrush_mixer: false,
            bitcrush_bits: 8,
            psg_crunch_compensation: false,
            mp2k_reverb: true,
            remove_sample_dc_offset: false,
            mixer_resample: MixerResampleSettings {
                choice: InstrumentResampleChoice::SincOutputNyquist,
                sinc_taps: 32,
                cutoff_hz: 13379,
            },
            shelf: HighShelf {
                enabled: true,
                gain_db: -1.0,
                cutoff_hz: 6000.0,
                q: 0.5,
                order: 2,
            },
            high_band_compress: HighBandCompressor {
                enabled_psg: true,
                enabled_sampled: true,
                ..HighBandCompressor::default()
            },
            exciter: Exciter::default(),
            track_enables: all_tracks_enabled(),
        }
    }

    pub fn enhanced_plus_gba() -> Self {
        Self {
            mixer_sample_rate: 48_000,
            mixer_resample: MixerResampleSettings {
                choice: InstrumentResampleChoice::SincSampleNyquist,
                sinc_taps: 32,
                cutoff_hz: 48_000,
            },
            psg_crunch_compensation: false,
            exciter: Exciter {
                enabled: true,
                crossover_hz: 1351.1,
                drive: 22.147,
                amount: 1.240,
            },
            high_band_compress: HighBandCompressor {
                enabled_psg: true,
                enabled_sampled: true,
                cutoff_hz: 15880.2,
                threshold_db: -59.97,
                ratio: 6.027,
                attack_ms: 5.009,
                release_ms: 17.514,
                makeup_db: -0.274,
            },
            shelf: HighShelf {
                enabled: true,
                order: 2,
                q: 0.404,
                cutoff_hz: 10509.2,
                gain_db: -6.054,
            },
            ..Self::enhanced_gba()
        }
    }

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
                pop_smooth_edge: PopSmoothingEdge::Both,
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
            high_band_compress: HighBandCompressor::default(),
            exciter: Exciter::default(),
            track_enables: all_tracks_enabled(),
        }
    }
}

//! The option types the signal chain reads: pop smoothing, delay smoothing, shelf, and compressor settings.

use crate::dsp::high_band_compressor::HighBandCompressorParams;
use crate::dsp::slewer::Direction;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DelaySmoothing {
    #[default]
    None,
    HoldDuringNotes,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HighShelf {
    pub enabled: bool,
    pub order: usize,
    pub q: f64,
    pub cutoff_hz: f64,
    pub gain_db: f64,
}

impl Default for HighShelf {
    fn default() -> Self {
        Self {
            enabled: false,
            order: 2,
            q: 0.707,
            cutoff_hz: 4000.0,
            gain_db: 0.0,
        }
    }
}

impl HighShelf {
    pub fn is_active(&self) -> bool {
        self.enabled && self.gain_db != 0.0 && self.order >= 2
    }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HighBandCompressor {
    #[serde(default)]
    pub enabled_psg: bool,
    #[serde(default)]
    pub enabled_sampled: bool,
    #[serde(default = "HighBandCompressor::default_cutoff_hz")]
    pub cutoff_hz: f64,
    #[serde(default = "HighBandCompressor::default_threshold_db")]
    pub threshold_db: f64,
    #[serde(default = "HighBandCompressor::default_ratio")]
    pub ratio: f64,
    #[serde(default = "HighBandCompressor::default_attack_ms")]
    pub attack_ms: f64,
    #[serde(default = "HighBandCompressor::default_release_ms")]
    pub release_ms: f64,
    #[serde(default = "HighBandCompressor::default_makeup_db")]
    pub makeup_db: f64,
}

impl Default for HighBandCompressor {
    fn default() -> Self {
        Self {
            enabled_psg: false,
            enabled_sampled: false,
            cutoff_hz: Self::default_cutoff_hz(),
            threshold_db: Self::default_threshold_db(),
            ratio: Self::default_ratio(),
            attack_ms: Self::default_attack_ms(),
            release_ms: Self::default_release_ms(),
            makeup_db: Self::default_makeup_db(),
        }
    }
}

impl HighBandCompressor {
    fn default_cutoff_hz() -> f64 {
        12700.0
    }
    fn default_threshold_db() -> f64 {
        -100.0
    }
    fn default_ratio() -> f64 {
        2.0
    }
    fn default_attack_ms() -> f64 {
        2.0
    }
    fn default_release_ms() -> f64 {
        85.53
    }
    fn default_makeup_db() -> f64 {
        0.0
    }

    pub fn is_active_psg(&self) -> bool {
        self.enabled_psg && self.ratio > 1.0
    }

    pub fn is_active_sampled(&self) -> bool {
        self.enabled_sampled && self.ratio > 1.0
    }

    pub fn params(&self) -> HighBandCompressorParams {
        HighBandCompressorParams {
            cutoff_hz: self.cutoff_hz,
            threshold_db: self.threshold_db,
            ratio: self.ratio,
            attack_ms: self.attack_ms,
            release_ms: self.release_ms,
            makeup_db: self.makeup_db,
        }
    }
}

pub const DEFAULT_POP_SLEW_SECONDS: f64 = 0.002;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PopSmoothing {
    pub psg: bool,
    pub sampled: bool,
    pub slew_seconds: f64,
    pub direction: Direction,
}

impl Default for PopSmoothing {
    fn default() -> Self {
        Self {
            psg: false,
            sampled: false,
            slew_seconds: DEFAULT_POP_SLEW_SECONDS,
            direction: Direction::UpAndDown,
        }
    }
}

impl PopSmoothing {
    #[inline]
    pub fn enabled_for(self, is_psg: bool) -> bool {
        if is_psg { self.psg } else { self.sampled }
    }
}

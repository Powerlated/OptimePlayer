//! The standalone synthesis-option types ([`DelaySmoothing`], [`HighShelf`], [`PopSmoothing`])
//! that make up a [`PerDeviceSettings`](crate::PerDeviceSettings) — the struct the synthesis layer
//! consumes directly.

use crate::dsp::slewer::Direction;

/// How the stereo expander's Haas delay lines react to a pan change while audio is flowing
/// through them (a delay-length jump shifts the signal and clicks).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DelaySmoothing {
    /// Apply the new delay lengths immediately, clicks and all.
    #[default]
    None,
    /// Defer the delay-length change until the track has no notes playing, so it can never
    /// land in the middle of one.
    HoldDuringNotes,
}

/// A master high-shelf EQ applied to the final mixed output (one per device, chosen by the
/// caller). All four RBJ parameters are user-adjustable; `enabled` off (or a 0 dB gain) is a
/// transparent bypass.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HighShelf {
    /// Whether the shelf is applied at all.
    pub enabled: bool,
    /// Filter order (even); the cascade has `order / 2` biquad sections — higher = steeper.
    pub order: usize,
    /// Resonance at the corner.
    pub q: f64,
    /// Corner frequency in Hz.
    pub cutoff_hz: f64,
    /// Shelf gain in dB (negative attenuates the highs, positive boosts them).
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
    /// Whether the shelf actually changes the signal (enabled and not a 0 dB / silent pass).
    pub fn is_active(&self) -> bool {
        self.enabled && self.gain_db != 0.0 && self.order >= 2
    }
}

/// A 1-band multiband compressor that dynamically tames the high band above `cutoff_hz`.
///
/// Structurally after OptimeGBA's `Soundgoodizer` (a FL Studio-style band-split) but with a working
/// compressor in the high path. The signal is split into a low band (untouched) and a high band
/// (compressed above `threshold_db` at `ratio`:1, with attack/release time-constants and a makeup
/// gain); the two are then summed. Only the over-threshold high-band content is touched — the rest
/// of the spectrum is bit-identical.
///
/// The two `enabled_*` flags select which bus the stage runs on, and only engage when the
/// intermediate mixer is on (the PSG and sampled buses are isolated there). Each bus owns its own
/// stage, so the two run independently — a sampled-bus peak doesn't duck PSG highs.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HighBandCompressor {
    /// Run the stage on the PSG bus (PSG voices on the output set when the mixer is engaged).
    /// `#[serde(default)]` so old saves load as the field defaults below.
    #[serde(default)]
    pub enabled_psg: bool,
    /// Run the stage on the sampled (DirectSound / SWAR) bus (the mixer set).
    #[serde(default)]
    pub enabled_sampled: bool,
    /// LPF/HPF split frequency in Hz.
    #[serde(default = "HighBandCompressor::default_cutoff_hz")]
    pub cutoff_hz: f64,
    /// Sidechain threshold in dBFS (0 = full scale). Over-threshold signal is compressed.
    #[serde(default = "HighBandCompressor::default_threshold_db")]
    pub threshold_db: f64,
    /// Compression ratio (`1.0` = no compression, `∞` = hard limit at the threshold).
    #[serde(default = "HighBandCompressor::default_ratio")]
    pub ratio: f64,
    /// Attack time in milliseconds (response to a rising sidechain).
    #[serde(default = "HighBandCompressor::default_attack_ms")]
    pub attack_ms: f64,
    /// Release time in milliseconds (response to a falling sidechain).
    #[serde(default = "HighBandCompressor::default_release_ms")]
    pub release_ms: f64,
    /// Makeup gain in dB applied after the gain reduction.
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
        13500.0
    }
    fn default_threshold_db() -> f64 {
        -80.0
    }
    fn default_ratio() -> f64 {
        4.0
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

    /// Whether the PSG stage should run: the PSG flag is on, the mixer is engaged (the PSG bus is
    /// isolated there), and there's an actual ratio to apply.
    pub fn is_active_psg(&self) -> bool {
        self.enabled_psg && self.ratio > 1.0
    }

    /// Whether the sampled stage should run: the sampled flag is on, the mixer is engaged, and
    /// there's an actual ratio to apply.
    pub fn is_active_sampled(&self) -> bool {
        self.enabled_sampled && self.ratio > 1.0
    }
}

/// The default de-click slew time in seconds — slow enough to kill the click, short enough to be
/// inaudible as an envelope. Used when no explicit time is configured.
pub const DEFAULT_POP_SLEW_SECONDS: f64 = 0.002;

/// Per-voice-kind de-click gain slew: slew a voice's gain over a few ms on note start/stop instead
/// of stepping it, turning abrupt on/off transitions into click-free ramps. Selected per kind so
/// PSG (square/wave/noise) and sampled (DirectSound/SWAR) voices can be smoothed independently —
/// the hard PSG edges are part of the chiptune character, while sampled de-clicking is usually
/// wanted. [`slew_seconds`](Self::slew_seconds) sets how long the ramp takes (shared by both kinds).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PopSmoothing {
    /// Slew PSG (square/wave/noise) voices.
    pub psg: bool,
    /// Slew sampled (DirectSound/SWAR) voices.
    pub sampled: bool,
    /// Seconds the de-click ramp takes to cross the full gain range. `0` makes it instant.
    pub slew_seconds: f64,
    /// Which gain moves get the ramp: [`Direction::UpOnly`] smooths only the attack (a note
    /// turning on or getting louder), [`Direction::DownOnly`] only the release (a note fading or
    /// being cut), [`Direction::UpAndDown`] both. The unramped direction jumps as the hardware
    /// does.
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
    /// Whether a voice of the given kind should slew its gain.
    #[inline]
    pub fn enabled_for(self, is_psg: bool) -> bool {
        if is_psg { self.psg } else { self.sampled }
    }
}

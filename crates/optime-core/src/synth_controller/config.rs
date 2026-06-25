//! The standalone synthesis-option types ([`DelaySmoothing`], [`HighShelf`], [`PopSmoothing`])
//! that make up a [`PerDeviceSettings`](crate::PerDeviceSettings) — the struct the synthesis layer
//! consumes directly.

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
    pub sample: bool,
    /// Seconds the de-click ramp takes to cross the full gain range. `0` makes it instant.
    pub slew_seconds: f64,
}

impl Default for PopSmoothing {
    fn default() -> Self {
        Self {
            psg: false,
            sample: false,
            slew_seconds: DEFAULT_POP_SLEW_SECONDS,
        }
    }
}

impl PopSmoothing {
    /// Whether a voice of the given kind should slew its gain.
    #[inline]
    pub fn enabled_for(self, is_psg: bool) -> bool {
        if is_psg {
            self.psg
        } else {
            self.sample
        }
    }
}

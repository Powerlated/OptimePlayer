//! [`SynthConfig`] — the runtime-tunable synthesis options threaded through every render call.

use crate::sample::InstrumentResampleMode;
use crate::tuning::TuningSystem;
use crate::TRACK_COUNT;

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
#[derive(Debug, Clone, Copy, PartialEq)]
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

/// Per-voice-kind de-click gain slew: slew a voice's gain over ~2 ms on note start/stop instead of
/// stepping it, turning abrupt on/off transitions into click-free ramps. Selected per kind so PSG
/// (square/wave/noise) and sampled (DirectSound/SWAR) voices can be smoothed independently — the
/// hard PSG edges are part of the chiptune character, while sampled de-clicking is usually wanted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PopSmoothing {
    /// Slew PSG (square/wave/noise) voices.
    pub psg: bool,
    /// Slew sampled (DirectSound/SWAR) voices.
    pub sample: bool,
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

/// Runtime-tunable synthesis options (replaces the original engine's global flags).
#[derive(Debug, Clone)]
pub struct SynthConfig {
    /// Apply the Haas-effect stereo widening delay lines.
    pub stereo_separation: bool,
    /// Force minimum stereo separation on barely-panned channels.
    pub force_stereo_separation: bool,
    /// Keep low frequencies centered ("bass mono"): only content above
    /// [`Self::bass_mono_freq`] is widened by the stereo separation, while the bass stays glued
    /// to the center.
    pub bass_mono: bool,
    /// Crossover cutoff (Hz) below which the signal is kept mono when [`Self::bass_mono`] is set.
    pub bass_mono_freq: f64,
    /// The active tuning system.
    pub tuning: TuningSystem,
    /// Which of the 16 tracks are mixed into the output.
    pub track_enables: [bool; TRACK_COUNT],
    /// Sample interpolation / anti-aliasing mode.
    pub resample: InstrumentResampleMode,
    /// Per-voice-kind de-click: slew a voice's gain over ~2 ms on note start/stop instead of
    /// stepping it. Off preserves the hard-edged hardware behaviour.
    pub pop_smoothing: PopSmoothing,
    /// How the stereo expander handles delay-line length changes.
    pub delay_smoothing: DelaySmoothing,
    /// Master high-shelf EQ on the final mixed output (per-device; chosen by the caller).
    pub high_shelf: HighShelf,
    /// Route the sampled (non-PSG) voices through an intermediate mixer running at
    /// [`Self::mixer_sample_rate`], then resample that stereo bus up to the output rate. PSG
    /// (square/wave/noise) voices bypass the mixer and render straight at the output rate. Off
    /// renders every voice at the output rate (the direct path).
    pub use_mixer: bool,
    /// Sample rate (Hz) of the intermediate mixer bus, when [`Self::use_mixer`] is set.
    pub mixer_sample_rate: f64,
    /// How the intermediate mixer bus is brought up to the output rate (the same algorithm set as
    /// the per-voice [`Self::resample`], resolved against the bus as a non-PSG signal).
    pub mixer_resample: InstrumentResampleMode,
}

impl Default for SynthConfig {
    fn default() -> Self {
        Self {
            stereo_separation: false,
            force_stereo_separation: false,
            bass_mono: false,
            bass_mono_freq: 200.0,
            tuning: TuningSystem::Equal,
            track_enables: [true; TRACK_COUNT],
            resample: InstrumentResampleMode::NearestNeighbor,
            pop_smoothing: PopSmoothing::default(),
            delay_smoothing: DelaySmoothing::None,
            high_shelf: HighShelf::default(),
            use_mixer: false,
            mixer_sample_rate: 48_000.0,
            // Clean reconstruction is the sane default for upsampling a finished bus.
            mixer_resample: InstrumentResampleMode::SincSampleNyquist { half_taps: 16 },
        }
    }
}

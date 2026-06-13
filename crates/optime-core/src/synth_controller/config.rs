//! [`SynthConfig`] — the runtime-tunable synthesis options threaded through every render call.

use crate::sample::ResampleMode;
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
    pub resample: ResampleMode,
    /// Smooth out the pops and clicks of PSG channels turning abruptly on and off (a ~2 ms
    /// gain slew). Off preserves the hard-edged hardware behaviour.
    pub smooth_psg_pops: bool,
    /// How the stereo expander handles delay-line length changes.
    pub delay_smoothing: DelaySmoothing,
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
            resample: ResampleMode::NearestNeighbor,
            smooth_psg_pops: false,
            delay_smoothing: DelaySmoothing::None,
        }
    }
}

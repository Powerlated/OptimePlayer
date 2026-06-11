//! Sample playback: per-voice [`SampleInstrument`]s, a polyphonic [`SampleSynthesizer`] per
//! track, and the stereo-separation [`DelayLine`].
//!
//! The pieces live in this module's children:
//! - [`gather`] — the windowed-sinc tap-staging gather (resampler front-end).
//! - [`instrument`] — [`SampleInstrument`], a single pitch-shifted voice.
//! - [`synthesizer`] — [`SampleSynthesizer`], the per-track polyphonic voice pool + stereo stage.
//! - [`delay`] — the Haas-effect [`DelayLine`].

mod delay;
mod gather;
mod instrument;
mod synthesizer;

pub use delay::DelayLine;
pub use instrument::SampleInstrument;
pub use synthesizer::SampleSynthesizer;

/// Q for the bass-mono crossover low-pass (Butterworth). `pub` so the app can reconstruct the
/// filters for the analysis popup without duplicating the constant.
pub const CROSSOVER_Q: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Maximum block length (in output samples) accepted by [`SampleSynthesizer::render_block`] and
/// [`Controller::fill`]'s internal blocking. Sized to cover one full sequencer tick at common
/// output rates (≈251 samples at 48 kHz) so a block rarely splits.
///
/// [`Controller::fill`]: crate::controller::Controller::fill
pub const MAX_BLOCK: usize = 256;

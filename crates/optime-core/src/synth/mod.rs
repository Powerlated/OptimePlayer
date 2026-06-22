//! Sample playback: per-voice [`SampleInstrument`]s, a polyphonic [`SampleSynthesizer`] per
//! track, and the stereo-separation [`DelayLine`].
//!
//! The pieces live in this module's children:
//! - [`instrument`] — [`SampleInstrument`], a single pitch-shifted voice.
//! - [`synthesizer`] — [`SampleSynthesizer`], the per-track polyphonic voice pool + stereo stage.
//! - [`delay`] — the Haas-effect [`DelayLine`].
//!
//! The resampling front-end ([`gather_sinc`](crate::resample::gather_sinc)) and the Authentic
//! hardware-chain state ([`AuthenticState`](crate::resample::AuthenticState)) live in
//! [`crate::resample`].

mod delay;
mod instrument;
mod synthesizer;

pub use delay::DelayLine;
pub use instrument::SampleInstrument;
pub use synthesizer::SampleSynthesizer;

/// Q for the bass-mono crossover low-pass (Butterworth). `pub` so the app can reconstruct the
/// filters for the analysis popup without duplicating the constant.
pub const CROSSOVER_Q: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Maximum block length (in output samples) accepted by [`SampleSynthesizer::render_block`] and
/// [`SynthController::fill`]'s internal blocking. Sized to cover one full sequencer tick at common
/// output rates (≈251 samples at 48 kHz) so a block rarely splits.
///
/// [`SynthController::fill`]: crate::synth_controller::SynthController::fill
pub const MAX_BLOCK: usize = 256;

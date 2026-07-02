//! Sample playback: per-voice [`WaveformInstrument`]s, a polyphonic [`WaveformSynthesizer`] per
//! track, and the stereo-separation [`DelayLine`].
//!
//! The pieces live in this module's children:
//! - [`instrument`] — [`WaveformInstrument`], a single pitch-shifted voice.
//! - [`synthesizer`] — [`WaveformSynthesizer`], the per-track polyphonic voice pool + stereo stage.
//! - [`delay_line`] — the Haas-effect [`DelayLine`].
//!
//! The resampling front-end ([`gather_sinc`](crate::dsp::resample::gather_sinc)) lives in
//! [`crate::dsp::resample`].

mod delay_line;
mod instrument;
mod synthesizer;

pub use delay_line::DelayLine;
pub use instrument::WaveformInstrument;
pub use synthesizer::WaveformSynthesizer;

/// Q for the bass-mono crossover low-pass (Butterworth). `pub` so the app can reconstruct the
/// filters for the analysis popup without duplicating the constant.
pub const CROSSOVER_Q: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Maximum block length (in output samples) accepted by [`WaveformSynthesizer::render_block`] and
/// [`SynthController::fill`]'s internal blocking. Sized to cover one full sequencer tick at common
/// output rates (≈251 samples at 48 kHz) so a block rarely splits.
///
/// [`SynthController::fill`]: crate::synth_controller::SynthController::fill
pub const MAX_BLOCK: usize = 256;

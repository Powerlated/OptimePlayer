mod delay_line;
mod instrument;
mod synthesizer;

pub use delay_line::DelayLine;
pub use instrument::WaveformInstrument;
pub use synthesizer::WaveformSynthesizer;

pub const CROSSOVER_Q: f64 = std::f64::consts::FRAC_1_SQRT_2;

pub use crate::dsp::block::MAX_BLOCK;

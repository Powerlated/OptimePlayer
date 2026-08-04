//! Resolves a user-facing resample mode into the gather and cutoff a voice or a stream should actually use.

use crate::waveform::InstrumentResampleMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EffectiveGather {
    Nearest,
    Linear,
    Sinc {
        step_mode: bool,
        cutoff_hz: Option<u32>,
    },
}

pub(crate) fn effective_gather(mode: InstrumentResampleMode, is_psg: bool) -> EffectiveGather {
    match mode {
        InstrumentResampleMode::NearestNeighbor => EffectiveGather::Nearest,
        InstrumentResampleMode::Linear if is_psg => EffectiveGather::Nearest,
        InstrumentResampleMode::Linear => EffectiveGather::Linear,
        InstrumentResampleMode::SincSampleNyquist { .. } => EffectiveGather::Sinc {
            step_mode: is_psg,
            cutoff_hz: None,
        },
        InstrumentResampleMode::SincOutputNyquist {
            psg_cutoff_hz,
            sampler_cutoff_hz,
            ..
        } => EffectiveGather::Sinc {
            step_mode: true,
            cutoff_hz: Some(if is_psg {
                psg_cutoff_hz
            } else {
                sampler_cutoff_hz
            }),
        },
    }
}

pub(crate) fn mode_half_taps(mode: InstrumentResampleMode) -> Option<usize> {
    match mode {
        InstrumentResampleMode::SincSampleNyquist { half_taps }
        | InstrumentResampleMode::SincOutputNyquist { half_taps, .. } => Some(half_taps),
        _ => None,
    }
}

pub(crate) fn sinc_fc(
    r: f32,
    inv_sample_rate: f32,
    step_mode: bool,
    cutoff_hz: Option<u32>,
) -> f32 {
    let mut fc = if step_mode || r > 1.0 { 0.5 / r } else { 0.5 };
    if let Some(hz) = cutoff_hz {
        fc = fc.min(hz as f32 * inv_sample_rate / r);
    }
    fc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_psg_falls_back_to_nearest() {
        assert_eq!(
            effective_gather(InstrumentResampleMode::Linear, true),
            EffectiveGather::Nearest
        );
        assert_eq!(
            effective_gather(InstrumentResampleMode::Linear, false),
            EffectiveGather::Linear
        );
    }

    #[test]
    fn output_nyquist_always_steps_and_carries_the_matching_cutoff() {
        let mode = InstrumentResampleMode::SincOutputNyquist {
            half_taps: 16,
            psg_cutoff_hz: 12_000,
            sampler_cutoff_hz: 15_000,
        };
        assert_eq!(
            effective_gather(mode, true),
            EffectiveGather::Sinc {
                step_mode: true,
                cutoff_hz: Some(12_000)
            }
        );
        assert_eq!(
            effective_gather(mode, false),
            EffectiveGather::Sinc {
                step_mode: true,
                cutoff_hz: Some(15_000)
            }
        );
    }
}

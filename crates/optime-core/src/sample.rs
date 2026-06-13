//! Decoded audio samples and the PCM8/PCM16/IMA-ADPCM/WAV decoders.

use crate::devices::nintendo_ds::tables::{ADPCM_INDEX_TABLE, ADPCM_STEP_TABLE};
use crate::util::{read_u16, read_u32, read_u8};

/// How a sample is interpolated during pitch-shifting / resampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleMode {
    /// No interpolation; pick the nearest source sample (the DS hardware's own behaviour).
    NearestNeighbor,
    /// Linear interpolation between adjacent source samples.
    Linear,
    /// Windowed-sinc low-pass at the **source** Nyquist rate: always removes ZOH imaging
    /// artifacts when upsampling and aliasing when downsampling.  Clean but smooth.
    SincSampleNyquist {
        /// Half-width of the Blackman-windowed sinc kernel in zero-crossings (≥ 1).
        half_taps: usize,
    },
    /// Windowed-sinc (BLEP-style) low-pass at the **output** Nyquist rate: preserves the
    /// characteristic crunchy ZOH colouring below output Nyquist while removing aliasing above
    /// it.  On upsampled voices it band-limits the ZOH stairstep edges to the output rate
    /// (instead of point-sampling hard edges), suppressing nearest-neighbour jitter/aliasing.
    ///
    /// The two cutoffs further darken the crunch independently for the PSG waveform channels
    /// and the sampled (DirectSound/SWAR) channels: the gather's low-pass runs at
    /// `min(output Nyquist, cutoff)`.
    SincOutputNyquist {
        /// Half-width of the Blackman-windowed sinc kernel in zero-crossings (≥ 1).
        half_taps: usize,
        /// Low-pass cutoff (Hz) for PSG waveform voices ([`Sample::is_psg_square`]).
        psg_cutoff_hz: u32,
        /// Low-pass cutoff (Hz) for sampled (DirectSound) voices.
        sampler_cutoff_hz: u32,
    },
    /// Reproduce the device's fixed-rate hardware output chain
    /// ([`HardwareChain`](crate::devices::HardwareChain)): sampled voices are
    /// linear-interpolated to the software-mixer rate (GBA: 13379 Hz), nearest-neighbour held at
    /// the DAC rate (32768 Hz), and properly rate-converted to the output rate; PSG voices are
    /// nearest-neighbour sampled straight at the DAC rate. Indistinguishable from hardware
    /// output by construction.
    Authentic {
        /// Half-width of the final-stage reconstruction kernel in DAC samples (≥ 1).
        half_taps: usize,
        /// Extra low-pass cutoff (Hz) applied by the final reconstruction stage
        /// ([`CUTOFF_OFF_HZ`](Self::CUTOFF_OFF_HZ) = no extra filtering).
        cutoff_hz: u32,
    },
    /// Like [`Authentic`](Self::Authentic), but the sampled-voice chain is reconstructed instead
    /// of point-held: the source is band-limited-sinc resampled up to the software-mixer rate
    /// (GBA: 13379 Hz) and a band-limited zero-order hold takes that to the DAC rate (32768 Hz),
    /// before the same final rate conversion to the output. PSG voices are unchanged
    /// (nearest-neighbour at the DAC rate).
    CrunchyAuthentic {
        /// Half-width of every kernel in the chain (≥ 1).
        half_taps: usize,
        /// Extra low-pass cutoff (Hz) applied by the final reconstruction stage.
        cutoff_hz: u32,
    },
}

impl ResampleMode {
    /// A cutoff high enough to never bite below any practical output Nyquist — the "no extra
    /// filtering" slider position (and the slider's maximum). This is a *transparent* sentinel,
    /// not a default: the out-of-the-box cutoff the user actually starts on is a lower,
    /// listenable value defined in the app's `default_settings`.
    pub const CUTOFF_OFF_HZ: u32 = 24_000;
}

/// A decoded waveform plus the metadata needed to play it back at the correct pitch.
#[derive(Debug, Clone)]
pub struct Sample {
    /// Normalized sample data in roughly `[-1.0, 1.0]`.
    pub data: Vec<f32>,
    /// The frequency (Hz) this waveform represents when played at `sample_rate`.
    pub frequency: f64,
    /// The rate (Hz) the waveform was recorded at.
    pub sample_rate: f64,
    /// Whether playback loops back to `loop_point` after reaching the end.
    pub looping: bool,
    /// Sample index to loop back to. Signed because the ADPCM loop-point formula can yield a
    /// small negative value that the original engine relies on.
    pub loop_point: i64,
    /// Set for the eight PSG square-wave waveforms so the resampler can special-case them.
    pub is_psg_square: bool,
    /// Original SWAR sample length in samples (informational).
    pub sample_length: usize,
}

impl Sample {
    /// Builds a sample from decoded data and playback metadata.
    pub fn new(
        data: Vec<f32>,
        frequency: f64,
        sample_rate: f64,
        looping: bool,
        loop_point: i64,
    ) -> Self {
        Self {
            data,
            frequency,
            sample_rate,
            looping,
            loop_point,
            is_psg_square: false,
            sample_length: 0,
        }
    }
}

/// Decodes signed 8-bit PCM into normalized `f32` samples.
pub fn decode_pcm8(data: &[u8]) -> Vec<f32> {
    data.iter().map(|&b| (b as i8 as f32) / 128.0).collect()
}

/// Decodes signed little-endian 16-bit PCM into normalized `f32` samples.
pub fn decode_pcm16(data: &[u8]) -> Vec<f32> {
    let count = data.len() / 2;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let raw = read_u16(data, i * 2) as i16;
        out.push(f32::from(raw) / 32768.0);
    }
    out
}

/// Decodes Nintendo IMA-ADPCM into normalized `f32` samples.
///
/// This faithfully reproduces the original engine's decoder, including its non-sign-extended
/// initial predictor (`header & 0xFFFF`) so that golden-parity output matches exactly.
pub fn decode_adpcm(data: &[u8]) -> Vec<f32> {
    if data.len() < 4 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity((data.len() - 4) * 2);

    let header = read_u32(data, 0);
    let mut current_value: i32 = (header & 0xFFFF) as i32;
    let mut adpcm_index: i32 = ((header >> 16) as i32).clamp(0, 88);

    for i in 4..data.len() {
        let byte = read_u8(data, i);
        for j in 0..2 {
            let nibble = ((byte >> (j * 4)) & 0xF) as i32;

            let step = ADPCM_STEP_TABLE[adpcm_index as usize];
            let mut diff = step >> 3;
            if nibble & 1 != 0 {
                diff += step >> 2;
            }
            if nibble & 2 != 0 {
                diff += step >> 1;
            }
            if nibble & 4 != 0 {
                diff += step;
            }

            if nibble & 8 == 8 {
                current_value = (current_value - diff).max(-0x7FFF);
            } else {
                current_value = (current_value + diff).min(0x7FFF);
            }
            adpcm_index = (adpcm_index + ADPCM_INDEX_TABLE[(nibble & 7) as usize]).clamp(0, 88);

            out.push(current_value as f32 / 32768.0);
        }
    }

    out
}

/// Decodes a (8- or 16-bit PCM) RIFF/WAV file into a [`Sample`] at `sample_frequency`.
///
/// Returns `None` for unsupported bit depths.
pub fn decode_wav(data: &[u8], sample_frequency: f64) -> Option<Sample> {
    let num_channels = read_u16(data, 22) as usize;
    let sample_rate = read_u32(data, 24) as f64;
    let bits_per_sample = read_u16(data, 34) as usize;

    if bits_per_sample != 8 && bits_per_sample != 16 {
        return None;
    }

    let subchunk2_size = read_u32(data, 40) as usize;
    let stride = bits_per_sample / 8 * num_channels.max(1);
    let mut sample_data = Vec::new();

    let mut i = 44;
    while i < 44 + subchunk2_size {
        match bits_per_sample {
            8 => sample_data.push(read_u8(data, i) as f32 / 255.0),
            16 => sample_data.push((read_u16(data, i) as i16) as f32 / 32767.0),
            _ => unreachable!(),
        }
        i += stride;
    }

    Some(Sample::new(
        sample_data,
        sample_frequency,
        sample_rate,
        false,
        0,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn pcm8_is_signed_and_normalized() {
        // 0x80 = -128 -> -1.0, 0x00 -> 0.0, 0x7F = 127 -> 127/128.
        let out = decode_pcm8(&[0x80, 0x00, 0x7F, 0xFF]);
        assert!(close(out[0], -1.0));
        assert!(close(out[1], 0.0));
        assert!(close(out[2], 127.0 / 128.0));
        assert!(close(out[3], -1.0 / 128.0)); // 0xFF = -1
    }

    #[test]
    fn pcm16_is_signed_and_normalized() {
        // 0x0000 -> 0, 0x8000 -> -1.0, 0x7FFF -> ~0.9999.
        let out = decode_pcm16(&[0x00, 0x00, 0x00, 0x80, 0xFF, 0x7F]);
        assert_eq!(out.len(), 3);
        assert!(close(out[0], 0.0));
        assert!(close(out[1], -1.0));
        assert!(close(out[2], 32767.0 / 32768.0));
    }

    #[test]
    fn adpcm_first_nibbles_step_correctly() {
        // Header: predictor 0, index 0. Step table[0] = 7.
        // Byte 0x04 -> low nibble 4 (diff = step = 7), high nibble 0.
        let out = decode_adpcm(&[0x00, 0x00, 0x00, 0x00, 0x04]);
        assert_eq!(out.len(), 2);
        assert!(close(out[0], 7.0 / 32768.0), "got {}", out[0]);
        // After nibble 4, index += 2 -> step table[2] = 9; nibble 0 adds 9>>3 = 1 -> 8.
        assert!(close(out[1], 8.0 / 32768.0), "got {}", out[1]);
    }

    #[test]
    fn adpcm_empty_on_short_input() {
        assert!(decode_adpcm(&[0, 0, 0]).is_empty());
    }
}

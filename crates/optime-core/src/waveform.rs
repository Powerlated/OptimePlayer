use crate::devices::nds::tables::{ADPCM_INDEX_TABLE, ADPCM_STEP_TABLE};
use crate::util::{read_u8, read_u16, read_u32};

pub type Sample = f32;

pub type Frame = (Sample, Sample);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstrumentResampleMode {
    NearestNeighbor,
    Linear,
    SincSampleNyquist {
        half_taps: usize,
    },
    SincOutputNyquist {
        half_taps: usize,
        psg_cutoff_hz: u32,
        sampler_cutoff_hz: u32,
    },
}

impl InstrumentResampleMode {
    pub const CUTOFF_OFF_HZ: u32 = 24_000;
}

#[derive(Debug, Clone)]
pub struct Waveform {
    pub data: Vec<f32>,
    pub frequency: f64,
    pub sample_rate: f64,
    pub looping: bool,
    pub loop_point: i64,
    pub is_psg_square: bool,
    pub sample_length: usize,
}

impl Waveform {
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

pub fn decode_pcm8(data: &[u8]) -> Vec<f32> {
    data.iter().map(|&b| (b as i8 as f32) / 128.0).collect()
}

pub fn decode_pcm16(data: &[u8]) -> Vec<f32> {
    let count = data.len() / 2;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let raw = read_u16(data, i * 2) as i16;
        out.push(f32::from(raw) / 32768.0);
    }
    out
}

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

pub fn decode_wav(data: &[u8], sample_frequency: f64) -> Option<Waveform> {
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

    Some(Waveform::new(
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
        let out = decode_pcm8(&[0x80, 0x00, 0x7F, 0xFF]);
        assert!(close(out[0], -1.0));
        assert!(close(out[1], 0.0));
        assert!(close(out[2], 127.0 / 128.0));
        assert!(close(out[3], -1.0 / 128.0));
    }

    #[test]
    fn pcm16_is_signed_and_normalized() {
        let out = decode_pcm16(&[0x00, 0x00, 0x00, 0x80, 0xFF, 0x7F]);
        assert_eq!(out.len(), 3);
        assert!(close(out[0], 0.0));
        assert!(close(out[1], -1.0));
        assert!(close(out[2], 32767.0 / 32768.0));
    }

    #[test]
    fn adpcm_first_nibbles_step_correctly() {
        let out = decode_adpcm(&[0x00, 0x00, 0x00, 0x00, 0x04]);
        assert_eq!(out.len(), 2);
        assert!(close(out[0], 7.0 / 32768.0), "got {}", out[0]);
        assert!(close(out[1], 8.0 / 32768.0), "got {}", out[1]);
    }

    #[test]
    fn adpcm_empty_on_short_input() {
        assert!(decode_adpcm(&[0, 0, 0]).is_empty());
    }
}

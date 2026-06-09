//! Minimal 16-bit PCM stereo WAV (RIFF) encoder, used for the in-app "export to WAV" feature.
//! No external dependency — mirrors the legacy `WavEncoder`.

/// Encodes interleaved stereo `f32` samples (range roughly -1..1) into a 16-bit PCM WAV file.
pub fn encode_stereo_i16(samples: &[(f32, f32)], sample_rate: u32) -> Vec<u8> {
    let channels: u16 = 2;
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * u32::from(channels) * u32::from(bits_per_sample) / 8;
    let block_align = channels * bits_per_sample / 8;
    let data_len = (samples.len() * 2 * 2) as u32; // frames * channels * 2 bytes

    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");

    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());

    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &(l, r) in samples {
        out.extend_from_slice(&to_i16(l).to_le_bytes());
        out.extend_from_slice(&to_i16(r).to_le_bytes());
    }
    out
}

fn to_i16(v: f32) -> i16 {
    (v.clamp(-1.0, 1.0) * 32767.0) as i16
}

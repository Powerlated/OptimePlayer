//! Renders every playable song from a loaded sound archive (DS SDAT, DSE, or GBA / `.gbaaudio`
//! image) to a mono 16-bit WAV, so the clips can be audio-fingerprinted against a reference
//! recording (see `tools/match_songs.py`). Also writes a `manifest.json` mapping each clip back to
//! its native listing index and song id, so the matcher can resolve which `songId` an audio match
//! belongs to (the curated table's `sparseIndex` comes from the reference playlist order, not this
//! native index).
//!
//! The input must already be decompressed (the `tools/` orchestrator gunzips `*.gbaaudio.gz`
//! first), so this example needs no extra dependencies.
//!
//! Usage: `cargo run -p optime-core --example render_songs -- <archive> <out_dir> [seconds=40]`

use std::path::Path;
use std::process::ExitCode;

use optime_core::{load_all, PerDeviceSettings, SynthController};

const SAMPLE_RATE: u32 = 32_768;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let (Some(in_path), Some(out_dir)) = (args.next(), args.next()) else {
        eprintln!("Usage: render_songs <archive> <out_dir> [seconds=40]");
        return ExitCode::FAILURE;
    };
    let seconds: f64 = args
        .next()
        .map(|s| s.parse().unwrap_or(40.0))
        .unwrap_or(40.0);

    let bytes = match std::fs::read(&in_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to read '{in_path}': {e}");
            return ExitCode::FAILURE;
        }
    };
    let Some(data) = load_all(&bytes).into_iter().next() else {
        eprintln!("No songs found in '{in_path}' (not an SDAT, DSE, or GBA image).");
        return ExitCode::FAILURE;
    };
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        eprintln!("Failed to create '{out_dir}': {e}");
        return ExitCode::FAILURE;
    }

    let song_ids = data.song_ids();
    println!(
        "Loaded {} songs; rendering {seconds:.0}s each.",
        song_ids.len()
    );

    let config = PerDeviceSettings::neutral();
    let frames = (seconds * SAMPLE_RATE as f64) as usize;
    // manifest.json entries, hand-formatted (keeps optime-core dependency-free).
    let mut manifest = String::from("[\n");

    for (index, &id) in song_ids.iter().enumerate() {
        let wav_name = format!("{index:04}_{id}.wav");
        let Some(mut controller) = SynthController::new(SAMPLE_RATE as f64, &*data, id) else {
            eprintln!("song {id}: failed to start, skipping");
            continue;
        };
        // Render `frames` stereo frames in device-buffer-sized chunks, downmixing to mono.
        let mut mono = Vec::with_capacity(frames);
        let mut buf = vec![0.0f32; 2 * 512];
        while mono.len() < frames {
            let n = 512.min(frames - mono.len());
            let chunk = &mut buf[..2 * n];
            controller.fill(chunk, &config);
            for frame in chunk.chunks_exact(2) {
                mono.push(0.5 * (frame[0] + frame[1]));
            }
        }
        let wav = encode_mono_i16(&mono, SAMPLE_RATE);
        let path = Path::new(&out_dir).join(&wav_name);
        if let Err(e) = std::fs::write(&path, wav) {
            eprintln!("Failed to write '{}': {e}", path.display());
            return ExitCode::FAILURE;
        }
        if index > 0 {
            manifest.push_str(",\n");
        }
        manifest.push_str(&format!(
            "  {{ \"index\": {index}, \"songId\": {id}, \"wav\": \"{wav_name}\" }}"
        ));
        if index % 50 == 0 {
            println!("  rendered {index}/{} (song {id})", song_ids.len());
        }
    }
    manifest.push_str("\n]\n");

    let manifest_path = Path::new(&out_dir).join("manifest.json");
    if let Err(e) = std::fs::write(&manifest_path, manifest) {
        eprintln!("Failed to write '{}': {e}", manifest_path.display());
        return ExitCode::FAILURE;
    }
    println!(
        "Wrote {} WAVs + {} to {out_dir}",
        song_ids.len(),
        manifest_path.display()
    );
    ExitCode::SUCCESS
}

/// Encodes mono f32 samples (roughly -1..1) as a 16-bit PCM WAV.
fn encode_mono_i16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = samples.len() * 2;
    let mut out = Vec::with_capacity(44 + data_len);
    let byte_rate = sample_rate * 2; // mono, 2 bytes/sample
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

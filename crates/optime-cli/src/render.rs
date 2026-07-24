//! Renders every playable song from a loaded sound archive (DS SDAT, DSE, or GBA / `.gbaaudio`
//! image) to a mono 16-bit WAV, so the clips can be audio-fingerprinted against a reference
//! recording (see the `match-ost` subcommand, which does that matching in-process). Also writes a
//! `manifest.json` mapping each clip back to its native listing index and song id, so an external
//! matcher can resolve which `songId` an audio match belongs to (the curated table's `sparseIndex`
//! comes from the reference playlist order, not this native index).
//!
//! The input must already be decompressed — gunzip any `*.gbaaudio.gz` first.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;
use optime_core::{PerDeviceSettings, SynthController, load_all};

use crate::wav::encode_mono_i16;

const SAMPLE_RATE: u32 = 32_768;

#[derive(Parser)]
#[command(about = "Render every playable song in an archive to a mono WAV, plus a manifest.")]
pub struct Args {
    /// Decompressed sound archive (DS SDAT, DSE, or GBA `.gbaaudio`).
    archive: PathBuf,
    /// Directory to write the WAVs and `manifest.json` into; created if missing.
    out_dir: PathBuf,
    /// Seconds of audio to render per song.
    #[arg(default_value_t = 40.0)]
    seconds: f64,
}

pub fn run(args: Args) -> ExitCode {
    let bytes = match std::fs::read(&args.archive) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to read '{}': {e}", args.archive.display());
            return ExitCode::FAILURE;
        }
    };
    let Some(data) = load_all(&bytes).into_iter().next() else {
        eprintln!(
            "No songs found in '{}' (not an SDAT, DSE, or GBA image).",
            args.archive.display()
        );
        return ExitCode::FAILURE;
    };
    if let Err(e) = std::fs::create_dir_all(&args.out_dir) {
        eprintln!("Failed to create '{}': {e}", args.out_dir.display());
        return ExitCode::FAILURE;
    }

    let song_ids = data.song_ids();
    println!(
        "Loaded {} songs; rendering {:.0}s each.",
        song_ids.len(),
        args.seconds
    );

    let config = PerDeviceSettings::neutral();
    let frames = (args.seconds * f64::from(SAMPLE_RATE)) as usize;
    // manifest.json entries, hand-formatted (the shape is fixed and trivially small).
    let mut manifest = String::from("[\n");

    for (index, &id) in song_ids.iter().enumerate() {
        let wav_name = format!("{index:04}_{id}.wav");
        let Some(mut controller) = SynthController::new(f64::from(SAMPLE_RATE), &*data, id) else {
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
        let path = Path::new(&args.out_dir).join(&wav_name);
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

    let manifest_path = Path::new(&args.out_dir).join("manifest.json");
    if let Err(e) = std::fs::write(&manifest_path, manifest) {
        eprintln!("Failed to write '{}': {e}", manifest_path.display());
        return ExitCode::FAILURE;
    }
    println!(
        "Wrote {} WAVs + {} to {}",
        song_ids.len(),
        manifest_path.display(),
        args.out_dir.display()
    );
    ExitCode::SUCCESS
}

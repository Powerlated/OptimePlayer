//! Extracts the MP2K audio data from a GBA ROM into an audio-only image (see
//! `devices/gba/extract.rs`), verifies the result loads with the same song list and renders a
//! few songs bit-identically, and writes it next to the input as `<stem>-audio.gba`.
//!
//! Usage: `cargo run -p optime-core --example extract_mp2k -- <ROM path>`

use std::path::Path;
use std::process::ExitCode;

use optime_core::{SoundData, SynthConfig, SynthController};

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("Usage: extract_mp2k <GBA ROM path>");
        return ExitCode::FAILURE;
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to read '{path}': {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut archives = SoundData::load_all(&bytes);
    let Some(data @ SoundData::Gba(_)) = archives.pop() else {
        eprintln!("No MP2K song table found in '{path}'.");
        return ExitCode::FAILURE;
    };
    let SoundData::Gba(gba) = &data else {
        unreachable!()
    };

    let song_ids = data.song_ids();
    println!(
        "Loaded {} bytes; song table at 0x{:X}; {} playable songs.",
        bytes.len(),
        gba.song_table,
        song_ids.len()
    );

    let extracted = gba.extract_audio();
    let kept = extracted.iter().filter(|&&b| b != 0).count();
    println!(
        "Extracted image: {} bytes ({} non-zero, {:.1}% of the ROM).",
        extracted.len(),
        kept,
        100.0 * kept as f64 / bytes.len() as f64
    );

    // Verify: same song list, and a sample of songs renders bit-identically.
    let Some(stripped) = SoundData::load_all(&extracted).pop() else {
        eprintln!("BUG: the extracted image no longer parses.");
        return ExitCode::FAILURE;
    };
    if stripped.song_ids() != song_ids {
        eprintln!(
            "BUG: song list changed ({} vs {} songs).",
            stripped.song_ids().len(),
            song_ids.len()
        );
        return ExitCode::FAILURE;
    }
    let sr = 32768.0;
    let config = SynthConfig::default();
    let step = (song_ids.len() / 8).max(1);
    for &id in song_ids.iter().step_by(step) {
        let (Some(mut a), Some(mut b)) = (
            SynthController::new(sr, &data, id),
            SynthController::new(sr, &stripped, id),
        ) else {
            eprintln!("BUG: song {id} no longer starts.");
            return ExitCode::FAILURE;
        };
        let mut buf_a = vec![0.0f32; 2 * (5.0 * sr) as usize];
        let mut buf_b = vec![0.0f32; buf_a.len()];
        a.fill(&mut buf_a, &config);
        b.fill(&mut buf_b, &config);
        if buf_a != buf_b {
            eprintln!("BUG: song {id} renders differently from the extract.");
            return ExitCode::FAILURE;
        }
        println!("song {id}: 5 s render bit-identical OK");
    }

    let stem = Path::new(&path).with_extension("");
    let out_path = format!("{}-audio.gba", stem.display());
    if let Err(e) = std::fs::write(&out_path, &extracted) {
        eprintln!("Failed to write '{out_path}': {e}");
        return ExitCode::FAILURE;
    }
    println!("Wrote {out_path}");
    ExitCode::SUCCESS
}

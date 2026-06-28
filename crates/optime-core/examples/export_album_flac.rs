//! Renders an archive's songs, in the order given by a curated song-name JSON, into a single
//! stereo FLAC, loudness-normalized album-wide to -16 LUFS (EBU R128 / ITU-R BS.1770).
//!
//! Each track is rendered with the engine's high-quality preset for its console
//! ([`PerDeviceSettings::high_quality_gba`] / [`PerDeviceSettings::high_quality_nintendo_ds`]),
//! playing two loops then a 10-second fade — the same policy as the app's WAV export.
//!
//! Rendering is parallelized across CPUs into per-track temp PCM files; two cheap sequential passes
//! then (1) measure the whole album's integrated loudness and (2) apply the single gain that lands
//! it at -16 LUFS while encoding the FLAC, so memory stays bounded regardless of album length.
//!
//! The input archive must already be decompressed (gunzip any `*.gbaaudio.gz` first). The JSON is
//! the curated `[{ "songId", "title" }]` table (its array order is the album order).
//!
//! Usage: `cargo run -p optime-core --example export_album_flac -- <archive> <names.json> <out.flac> [limit]`

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};

use ebur128::{EbuR128, Mode};
use flac_codec::encode::{FlacSampleWriter, Options};
use optime_core::{PerDeviceSettings, SoundData, SynthController};
use serde_json::Value;

const SR: u32 = 32_768; // matches the app's EXPORT_SAMPLE_RATE
const GAP_SECS: f32 = 0.8; // silence between tracks
const TARGET_LUFS: f64 = -16.0;

/// Renders one song to stereo frames: two loops then a 10-second fade, capped at 480s. Mirrors the
/// app's `render_to_samples` (including its 0.5 headroom gain).
fn render_song(data: &SoundData, song_id: u32, config: &PerDeviceSettings) -> Vec<(f32, f32)> {
    const FADEOUT_LENGTH: f64 = 10.0;
    const LOOP_COUNT: u32 = 2;
    let sr = f64::from(SR);
    let Some(mut controller) = SynthController::new(sr, data, song_id) else {
        return Vec::new();
    };
    let max_samples = (sr * 480.0) as u64;
    const CHUNK_FRAMES: usize = 512;
    let mut buf = vec![0.0f32; 2 * CHUNK_FRAMES];

    let mut out = Vec::new();
    let mut sample: u64 = 0;
    let mut loop_count = 0u32;
    let mut fading_out = false;
    let mut fadeout_start_sample = 0u64;

    'render: while sample < max_samples {
        let n = CHUNK_FRAMES.min((max_samples - sample) as usize);
        let chunk = &mut buf[..2 * n];
        controller.fill(chunk, config);

        if controller.jumps > 0 {
            controller.jumps = 0;
            loop_count += 1;
            if loop_count == LOOP_COUNT {
                controller.fading_start = true;
            }
        }
        if controller.fading_start {
            controller.fading_start = false;
            fading_out = true;
            fadeout_start_sample = sample + (sr * 2.0) as u64;
        }

        for frame in chunk.chunks_exact(2) {
            let mut mul = 1.0f32;
            if fading_out && sample >= fadeout_start_sample {
                let t = (sample - fadeout_start_sample) as f64 / sr;
                mul = (1.0 - t / FADEOUT_LENGTH) as f32;
                if mul <= 0.0 {
                    break 'render;
                }
            }
            out.push((frame[0] * 0.5 * mul, frame[1] * 0.5 * mul));
            sample += 1;
        }
    }
    out
}

/// `dir/trk_{i:05}.pcm` — the per-track interleaved-i16 scratch file for album position `i`.
fn track_path(dir: &Path, i: usize) -> PathBuf {
    dir.join(format!("trk_{i:05}.pcm"))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [archive, json_path, out_path] = match args.as_slice() {
        [a, j, o, ..] => [a, j, o],
        _ => {
            eprintln!(
                "Usage: export_album_flac <archive> <names.json> <out.flac> [limit]"
            );
            return ExitCode::FAILURE;
        }
    };
    let limit: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);

    let bytes = match fs::read(archive) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to read '{archive}': {e}");
            return ExitCode::FAILURE;
        }
    };
    let Some(data) = SoundData::load_all(&bytes).into_iter().next() else {
        eprintln!("No songs found in '{archive}' (not an SDAT, DSE, or GBA image).");
        return ExitCode::FAILURE;
    };

    // High-quality preset for the archive's console.
    let is_gba = data.gba_game_code().is_some();
    let config = if is_gba {
        PerDeviceSettings::high_quality_gba()
    } else {
        PerDeviceSettings::high_quality_nintendo_ds()
    };
    eprintln!(
        "Console: {} (using high-quality {} preset)",
        if is_gba { "GBA" } else { "DS/other" },
        if is_gba { "GBA" } else { "DS" }
    );

    // Album order = the JSON array order, restricted to songs actually present in the archive.
    let json: Value = match fs::read_to_string(json_path).map(|s| serde_json::from_str(&s)) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            eprintln!("Failed to parse '{json_path}': {e}");
            return ExitCode::FAILURE;
        }
        Err(e) => {
            eprintln!("Failed to read '{json_path}': {e}");
            return ExitCode::FAILURE;
        }
    };
    let playable: HashSet<u32> = data.song_ids().into_iter().collect();
    let album: Vec<(u32, String)> = json
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .filter_map(|e| {
            let id = u32::try_from(e["songId"].as_u64()?).ok()?;
            playable
                .contains(&id)
                .then(|| (id, e["title"].as_str().unwrap_or("").to_string()))
        })
        .take(limit)
        .collect();
    if album.is_empty() {
        eprintln!("No playable songs from '{json_path}' are present in the archive.");
        return ExitCode::FAILURE;
    }

    let tmp_dir = std::env::temp_dir().join(format!("album_export_{}", std::process::id()));
    if let Err(e) = fs::create_dir_all(&tmp_dir) {
        eprintln!("Failed to create temp dir '{}': {e}", tmp_dir.display());
        return ExitCode::FAILURE;
    }

    // --- Parallel render: each album position -> its own interleaved-i16 PCM file. ---
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(album.len());
    eprintln!("Rendering {} tracks on {threads} threads...", album.len());
    let cursor = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let i = cursor.fetch_add(1, Ordering::Relaxed);
                let Some((id, title)) = album.get(i) else { break };
                let frames = render_song(&data, *id, &config);
                let mut bytes = Vec::with_capacity(frames.len() * 4);
                for &(l, r) in &frames {
                    for s in [l, r] {
                        bytes.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
                    }
                }
                fs::write(track_path(&tmp_dir, i), &bytes).expect("write track pcm");
                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                eprintln!(
                    "  [{d:3}/{}] songId {id:<5} {:6.1}s  \"{title}\"",
                    album.len(),
                    frames.len() as f32 / SR as f32
                );
            });
        }
    });

    let gap_frames = (GAP_SECS * SR as f32) as usize;
    let gap = vec![0i16; gap_frames * 2];

    // --- Pass 1: measure album-wide integrated loudness. ---
    let mut meter = EbuR128::new(2, SR, Mode::I).expect("ebur128 init");
    let mut total_frames: u64 = 0;
    for i in 0..album.len() {
        if i > 0 {
            meter.add_frames_i16(&gap).expect("meter gap");
            total_frames += gap_frames as u64;
        }
        let raw = fs::read(track_path(&tmp_dir, i)).expect("read track pcm");
        let samples = bytes_to_i16(&raw);
        meter.add_frames_i16(&samples).expect("meter frames");
        total_frames += (samples.len() / 2) as u64;
    }
    let loudness = meter.loudness_global().expect("integrated loudness");
    let gain_db = TARGET_LUFS - loudness;
    let gain = (10f64).powf(gain_db / 20.0) as f32;
    eprintln!(
        "\nAlbum: {} tracks, {:.1} min.\nIntegrated loudness {loudness:.2} LUFS -> gain {gain_db:+.2} dB to reach {TARGET_LUFS:.0} LUFS",
        album.len(),
        total_frames as f32 / SR as f32 / 60.0
    );

    // --- Pass 2: apply the gain and encode the FLAC. ---
    let _ = fs::remove_file(out_path); // FlacSampleWriter::create refuses to overwrite
    let mut writer = match FlacSampleWriter::create(out_path, Options::default(), SR, 16, 2, None) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Failed to create '{out_path}': {e}");
            return ExitCode::FAILURE;
        }
    };
    let gap_i32 = vec![0i32; gap_frames * 2];
    let mut peak = 0f32;
    let mut clipped: u64 = 0;
    for i in 0..album.len() {
        if i > 0 {
            writer.write(&gap_i32).expect("write gap");
        }
        let raw = fs::read(track_path(&tmp_dir, i)).expect("read track pcm");
        let out: Vec<i32> = bytes_to_i16(&raw)
            .iter()
            .map(|&s| {
                let g = (f32::from(s) / 32767.0) * gain;
                peak = peak.max(g.abs());
                if g.abs() > 1.0 {
                    clipped += 1;
                }
                (g.clamp(-1.0, 1.0) * 32767.0).round() as i32
            })
            .collect();
        writer.write(&out).expect("write samples");
    }
    writer.finalize().expect("finalize flac");
    let _ = fs::remove_dir_all(&tmp_dir);

    let size = fs::metadata(out_path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "Post-gain peak {peak:.3} ({:+.2} dBFS); {clipped} samples hard-limited.\nWrote {out_path} ({:.1} MB).",
        20.0 * peak.max(1e-9).log10(),
        size as f64 / 1.0e6
    );
    ExitCode::SUCCESS
}

/// Reinterprets a little-endian byte buffer as interleaved i16 samples.
fn bytes_to_i16(raw: &[u8]) -> Vec<i16> {
    raw.chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect()
}

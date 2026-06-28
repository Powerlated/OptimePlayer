//! Renders an archive's songs, in the order given by a curated song-name JSON, into a single
//! stereo FLAC, loudness-normalized album-wide to -16 LUFS (EBU R128 / ITU-R BS.1770).
//!
//! Each track is rendered with the engine's high-quality preset for its console
//! ([`PerDeviceSettings::high_quality_gba`] / [`PerDeviceSettings::high_quality_nintendo_ds`]),
//! playing two loops then a 10-second fade — the same policy as the app's WAV export. Leading and
//! trailing near-silence is trimmed from every track and a fixed `--max-silence` gap is inserted
//! between songs, so the dead air between tracks is capped at that length.
//!
//! Rendering is parallelized across CPUs into per-track temp PCM files (with one live progress bar
//! per worker); two cheap sequential passes then (1) measure the whole album's integrated loudness
//! and (2) apply the single gain that lands it at -16 LUFS while encoding the FLAC, so memory stays
//! bounded regardless of album length.
//!
//! The input archive must already be decompressed (gunzip any `*.gbaaudio.gz` first). The JSON is
//! the curated `[{ "songId", "title" }]` table (its array order is the album order).
//!
//! Usage: `cargo run -p optime-core --example export_album_flac -- <archive> <names.json> <out.flac> [--max-silence S] [--limit N]`

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};

use clap::Parser;
use ebur128::{EbuR128, Mode};
use flac_codec::encode::{FlacSampleWriter, Options};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use optime_core::{PerDeviceSettings, SoundData, SynthController};
use rayon::prelude::*;
use serde_json::Value;

const SR: u32 = 32_768; // matches the app's EXPORT_SAMPLE_RATE
const TARGET_LUFS: f64 = -16.0;
/// A frame counts as silence when both channels are within this magnitude (~-66 dBFS).
const SILENCE_I16: i16 = 16;

#[derive(Parser)]
#[command(about = "Render an archive's songs into one -16 LUFS stereo FLAC, in song-name JSON order.")]
struct Args {
    /// Decompressed sound archive (DS SDAT, DSE, or GBA `.gbaaudio`).
    archive: PathBuf,
    /// Curated `[{ "songId", "title" }]` JSON; its array order is the album order.
    names_json: PathBuf,
    /// Output FLAC path.
    out: PathBuf,
    /// Trim each track's leading/trailing silence and insert at most this many seconds of silence
    /// between songs.
    #[arg(long, default_value_t = 0.8)]
    max_silence: f32,
    /// Only export the first N songs (for quick tests).
    #[arg(long)]
    limit: Option<usize>,
}

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

/// Reinterprets a little-endian byte buffer as interleaved i16 samples.
fn bytes_to_i16(raw: &[u8]) -> Vec<i16> {
    raw.chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect()
}

/// The interleaved-i16 sub-slice of one track with leading and trailing near-silent frames removed.
fn trim_silence(samples: &[i16]) -> &[i16] {
    let frames = samples.len() / 2;
    let silent = |f: usize| samples[2 * f].abs() <= SILENCE_I16 && samples[2 * f + 1].abs() <= SILENCE_I16;
    let first = (0..frames).find(|&f| !silent(f));
    let Some(first) = first else { return &[] };
    let last = (0..frames).rev().find(|&f| !silent(f)).unwrap();
    &samples[2 * first..2 * (last + 1)]
}

fn main() -> ExitCode {
    let args = Args::parse();

    let bytes = match fs::read(&args.archive) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to read '{}': {e}", args.archive.display());
            return ExitCode::FAILURE;
        }
    };
    let Some(data) = SoundData::load_all(&bytes).into_iter().next() else {
        eprintln!("No songs found in '{}' (not an SDAT, DSE, or GBA image).", args.archive.display());
        return ExitCode::FAILURE;
    };

    // High-quality preset for the archive's console.
    let is_gba = data.gba_game_code().is_some();
    let config = if is_gba {
        PerDeviceSettings::high_quality_gba()
    } else {
        PerDeviceSettings::high_quality_nintendo_ds()
    };

    // Album order = the JSON array order, restricted to songs actually present in the archive.
    let json: Value = match fs::read_to_string(&args.names_json).map(|s| serde_json::from_str(&s)) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            eprintln!("Failed to parse '{}': {e}", args.names_json.display());
            return ExitCode::FAILURE;
        }
        Err(e) => {
            eprintln!("Failed to read '{}': {e}", args.names_json.display());
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
        .take(args.limit.unwrap_or(usize::MAX))
        .collect();
    if album.is_empty() {
        eprintln!("No playable songs from '{}' are present in the archive.", args.names_json.display());
        return ExitCode::FAILURE;
    }

    eprintln!(
        "Console: {} (high-quality {} preset) — {} tracks, {:.1}s max silence between songs.",
        if is_gba { "GBA" } else { "DS/other" },
        if is_gba { "GBA" } else { "DS" },
        album.len(),
        args.max_silence
    );

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

    let mp = MultiProgress::new();
    let overall = mp.add(ProgressBar::new(album.len() as u64));
    overall.set_style(
        ProgressStyle::with_template("  render [{bar:32.cyan/blue}] {pos}/{len} ({elapsed})")
            .unwrap()
            .progress_chars("=>-"),
    );
    let worker_style = ProgressStyle::with_template("    {spinner:.green} {wide_msg}").unwrap();
    let worker_bars: Vec<ProgressBar> = (0..threads)
        .map(|_| {
            let pb = mp.add(ProgressBar::new_spinner());
            pb.set_style(worker_style.clone());
            pb.enable_steady_tick(std::time::Duration::from_millis(100));
            pb
        })
        .collect();

    let cursor = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for pb in &worker_bars {
            let (overall, cursor, album, data, config, tmp_dir) =
                (&overall, &cursor, &album, &data, &config, &tmp_dir);
            scope.spawn(move || {
                loop {
                    let i = cursor.fetch_add(1, Ordering::Relaxed);
                    let Some((id, title)) = album.get(i) else { break };
                    pb.set_message(format!("songId {id:<5} \"{title}\""));
                    let frames = render_song(data, *id, config);
                    let mut bytes = Vec::with_capacity(frames.len() * 4);
                    for &(l, r) in &frames {
                        for s in [l, r] {
                            bytes.extend_from_slice(
                                &((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes(),
                            );
                        }
                    }
                    fs::write(track_path(tmp_dir, i), &bytes).expect("write track pcm");
                    overall.inc(1);
                }
                pb.finish_and_clear();
            });
        }
    });
    overall.finish();

    let gap_frames = (args.max_silence.max(0.0) * SR as f32) as usize;

    // --- Pass 1: measure each track's loudness in parallel (one EbuR128 state per track), then
    // combine. Integrated loudness gates out the inter-song silence, so per-track measurement of the
    // trimmed cores matches measuring the whole concatenated album. ---
    let measure = mp.add(ProgressBar::new(album.len() as u64));
    measure.set_style(
        ProgressStyle::with_template("  measure [{bar:32.yellow/blue}] {pos}/{len}")
            .unwrap()
            .progress_chars("=>-"),
    );
    let states: Vec<(EbuR128, u64)> = (0..album.len())
        .into_par_iter()
        .filter_map(|i| {
            let raw = fs::read(track_path(&tmp_dir, i)).expect("read track pcm");
            let samples = bytes_to_i16(&raw);
            let core = trim_silence(&samples);
            measure.inc(1);
            if core.is_empty() {
                return None;
            }
            let mut state = EbuR128::new(2, SR, Mode::I).expect("ebur128 init");
            state.add_frames_i16(core).expect("meter frames");
            Some((state, (core.len() / 2) as u64))
        })
        .collect();
    measure.finish();

    let nonempty = states.len() as u64;
    let total_frames =
        states.iter().map(|(_, n)| n).sum::<u64>() + nonempty.saturating_sub(1) * gap_frames as u64;
    let loudness =
        EbuR128::loudness_global_multiple(states.iter().map(|(s, _)| s)).expect("integrated loudness");
    let gain_db = TARGET_LUFS - loudness;
    let gain = (10f64).powf(gain_db / 20.0) as f32;

    // --- Pass 2: apply the gain and encode the FLAC. The FLAC bitstream is serial, but flac-codec's
    // `rayon` feature compresses the channels in parallel internally, so the per-track feed loop
    // stays sequential while the heavy compression is multithreaded. ---
    let _ = fs::remove_file(&args.out); // FlacSampleWriter::create refuses to overwrite
    let mut writer = match FlacSampleWriter::create(&args.out, Options::default(), SR, 16, 2, None) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Failed to create '{}': {e}", args.out.display());
            return ExitCode::FAILURE;
        }
    };
    let encode = mp.add(ProgressBar::new(album.len() as u64));
    encode.set_style(
        ProgressStyle::with_template("  encode  [{bar:32.magenta/blue}] {pos}/{len}")
            .unwrap()
            .progress_chars("=>-"),
    );
    let gap_i32 = vec![0i32; gap_frames * 2];
    let mut peak = 0f32;
    let mut clipped: u64 = 0;
    let mut wrote_any = false;
    for i in 0..album.len() {
        let raw = fs::read(track_path(&tmp_dir, i)).expect("read track pcm");
        let samples = bytes_to_i16(&raw);
        let core = trim_silence(&samples);
        if core.is_empty() {
            encode.inc(1);
            continue;
        }
        if wrote_any {
            writer.write(&gap_i32).expect("write gap");
        }
        let out: Vec<i32> = core
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
        wrote_any = true;
        encode.inc(1);
    }
    writer.finalize().expect("finalize flac");
    encode.finish();
    let _ = fs::remove_dir_all(&tmp_dir);

    let size = fs::metadata(&args.out).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "\nAlbum: {} tracks, {:.1} min.\nIntegrated loudness {loudness:.2} LUFS -> gain {gain_db:+.2} dB to reach {TARGET_LUFS:.0} LUFS.\nPost-gain peak {peak:.3} ({:+.2} dBFS); {clipped} samples hard-limited.\nWrote {} ({:.1} MB).",
        album.len(),
        total_frames as f32 / SR as f32 / 60.0,
        20.0 * peak.max(1e-9).log10(),
        args.out.display(),
        size as f64 / 1.0e6
    );
    ExitCode::SUCCESS
}

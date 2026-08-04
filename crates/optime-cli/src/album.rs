//! Renders a curated soundtrack into one loudness-normalised FLAC album, and the shared render machinery the benchmark reuses.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, Ordering};

use clap::Args as ClapArgs;
use ebur128::{EbuR128, Mode};
use flac_codec::encode::{FlacSampleWriter, Options};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use optime_core::devices::gba::GbaRom;
use optime_core::{
    LoopAndTransitionOptions, PerDeviceSettings, PlaybackEvent, SoundData, SynthController,
    load_all,
};
use rayon::prelude::*;
use serde_json::Value;

pub const SR: u32 = 32_768;
const TARGET_LUFS: f64 = -16.0;
const SILENCE_I16: i16 = 16;

#[derive(ClapArgs)]
#[command(
    about = "Render an archive's songs into one -16 LUFS stereo FLAC, in song-name JSON order."
)]
pub struct Args {
    archive: PathBuf,
    names_json: PathBuf,
    out: PathBuf,
    #[arg(long, default_value_t = 0.8)]
    max_silence: f32,
    #[arg(long)]
    limit: Option<usize>,
    #[arg(long, num_args = 0..=1, default_missing_value = "100%", value_name = "PCT")]
    benchmark: Option<String>,
}

fn parse_percent(s: &str) -> Result<f64, String> {
    let t = s.trim().trim_end_matches('%').trim();
    let raw: f64 = t.parse().map_err(|_| format!("invalid percentage '{s}'"))?;
    let frac = if s.contains('%') || raw > 1.0 {
        raw / 100.0
    } else {
        raw
    };
    if frac > 0.0 && frac <= 1.0 {
        Ok(frac)
    } else {
        Err(format!("percentage out of range (0, 100]: '{s}'"))
    }
}

fn spread_indices(total: usize, frac: f64) -> Vec<usize> {
    let count = ((total as f64 * frac).round() as usize).clamp(1, total.max(1));
    (0..count)
        .map(|k| (((k as f64 + 0.5) * total as f64 / count as f64) as usize).min(total - 1))
        .collect()
}

fn run_benchmark(
    data: &dyn SoundData,
    album: &[(u32, String)],
    config: &PerDeviceSettings,
    frac: f64,
    is_gba: bool,
) -> ExitCode {
    use std::time::Instant;

    let idxs = spread_indices(album.len(), frac);
    let sample_bits = optime_core::SAMPLE_SIZE_BYTES * 8;
    eprintln!(
        "Benchmark: Sample = f{sample_bits} ({} B) — {}/{} tracks ({:.0}% of album), {} console, high-quality preset, {BENCH_THREADS} threads.",
        optime_core::SAMPLE_SIZE_BYTES,
        idxs.len(),
        album.len(),
        frac * 100.0,
        if is_gba { "GBA" } else { "DS/other" },
    );

    const BENCH_THREADS: usize = 4;

    let render_pass = || -> (u64, std::time::Duration) {
        let frames = AtomicU64::new(0);
        let t0 = Instant::now();
        parallel_render(data, album, config, &idxs, BENCH_THREADS, |_, _, f, _| {
            frames.fetch_add(f.len() as u64, Ordering::Relaxed);
        });
        (frames.load(Ordering::Relaxed), t0.elapsed())
    };

    let _ = render_pass();
    const PASSES: usize = 3;
    let mut runs: Vec<(u64, f64)> = Vec::with_capacity(PASSES);
    for p in 0..PASSES {
        let (frames, dt) = render_pass();
        let wall = dt.as_secs_f64();
        let audio_s = frames as f64 / f64::from(SR);
        eprintln!(
            "  pass {}/{PASSES}: {frames} frames ({audio_s:.1}s audio) in {wall:.3}s  →  {:.1}× realtime, {:.2} Msamp/s",
            p + 1,
            audio_s / wall,
            frames as f64 * 2.0 / wall / 1.0e6,
        );
        runs.push((frames, wall));
    }
    runs.sort_by(|a, b| a.1.total_cmp(&b.1));
    let (frames, wall) = runs[runs.len() / 2];
    let audio_s = frames as f64 / f64::from(SR);
    eprintln!(
        "\nMEDIAN  Sample=f{sample_bits}  tracks={}  frames={frames}  audio={audio_s:.1}s  wall={wall:.3}s  realtime={:.1}x  throughput={:.2}Msamp/s",
        idxs.len(),
        audio_s / wall,
        frames as f64 * 2.0 / wall / 1.0e6,
    );
    ExitCode::SUCCESS
}

pub fn high_quality_preset(data: &dyn SoundData) -> (PerDeviceSettings, bool) {
    let is_gba = data.as_any().downcast_ref::<GbaRom>().is_some();
    let config = if is_gba {
        PerDeviceSettings::enhanced_gba()
    } else {
        PerDeviceSettings::high_quality_nintendo_ds()
    };
    (config, is_gba)
}

pub fn album_order(
    data: &dyn SoundData,
    names_json: &Path,
    limit: Option<usize>,
) -> Result<Vec<(u32, String)>, String> {
    let text = fs::read_to_string(names_json)
        .map_err(|e| format!("Failed to read '{}': {e}", names_json.display()))?;
    let json: Value = serde_json::from_str(&text)
        .map_err(|e| format!("Failed to parse '{}': {e}", names_json.display()))?;
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
        .take(limit.unwrap_or(usize::MAX))
        .collect();
    if album.is_empty() {
        return Err(format!(
            "No playable songs from '{}' are present in the archive.",
            names_json.display()
        ));
    }
    Ok(album)
}

pub fn render_song(
    data: &dyn SoundData,
    song_id: u32,
    config: &PerDeviceSettings,
) -> Vec<(f32, f32)> {
    let sr = f64::from(SR);
    let Some(mut controller) = SynthController::new(sr, data, song_id) else {
        return Vec::new();
    };
    controller.set_loop_and_transition(LoopAndTransitionOptions::export());
    let max_samples = (sr * 480.0) as u64;
    const CHUNK_FRAMES: usize = 512;
    let mut buf = vec![0.0f32; 2 * CHUNK_FRAMES];

    let mut out = Vec::new();
    let mut sample: u64 = 0;
    while sample < max_samples {
        let n = CHUNK_FRAMES.min((max_samples - sample) as usize);
        let chunk = &mut buf[..2 * n];
        controller.fill(chunk, config);

        for frame in chunk.chunks_exact(2) {
            out.push((frame[0] * 0.5, frame[1] * 0.5));
            sample += 1;
        }
        if controller
            .take_messages()
            .any(|m| m == PlaybackEvent::Finished)
        {
            break;
        }
    }
    out
}

fn parallel_render<F>(
    data: &dyn SoundData,
    album: &[(u32, String)],
    config: &PerDeviceSettings,
    idxs: &[usize],
    threads: usize,
    on_track: F,
) where
    F: Fn(usize, &(u32, String), &[(f32, f32)], usize) + Sync,
{
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("build render thread pool");
    pool.install(|| {
        idxs.par_iter().for_each(|&i| {
            let frames = render_song(data, album[i].0, config);
            let worker = rayon::current_thread_index().unwrap_or(0);
            on_track(i, &album[i], &frames, worker);
        });
    });
}

fn track_path(dir: &Path, i: usize) -> PathBuf {
    dir.join(format!("trk_{i:05}.pcm"))
}

fn bytes_to_i16(raw: &[u8]) -> Vec<i16> {
    raw.chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect()
}

fn trim_silence(samples: &[i16]) -> &[i16] {
    let frames = samples.len() / 2;
    let silent =
        |f: usize| samples[2 * f].abs() <= SILENCE_I16 && samples[2 * f + 1].abs() <= SILENCE_I16;
    let first = (0..frames).find(|&f| !silent(f));
    let Some(first) = first else { return &[] };
    let last = (0..frames).rev().find(|&f| !silent(f)).unwrap();
    &samples[2 * first..2 * (last + 1)]
}

pub fn run(args: Args) -> ExitCode {
    let bytes = match fs::read(&args.archive) {
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

    let (config, is_gba) = high_quality_preset(&*data);

    let album = match album_order(&*data, &args.names_json, args.limit) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    if let Some(pct) = &args.benchmark {
        let frac = match parse_percent(pct) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        };
        return run_benchmark(&*data, &album, &config, frac, is_gba);
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

    let idxs: Vec<usize> = (0..album.len()).collect();
    parallel_render(
        &*data,
        &album,
        &config,
        &idxs,
        threads,
        |i, (id, title), frames, worker| {
            worker_bars[worker].set_message(format!("songId {id:<5} \"{title}\""));
            let mut bytes = Vec::with_capacity(frames.len() * 4);
            for &(l, r) in frames {
                for s in [l, r] {
                    bytes.extend_from_slice(
                        &((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes(),
                    );
                }
            }
            fs::write(track_path(&tmp_dir, i), &bytes).expect("write track pcm");
            overall.inc(1);
        },
    );
    for pb in &worker_bars {
        pb.finish_and_clear();
    }
    overall.finish();

    let gap_frames = (args.max_silence.max(0.0) * SR as f32) as usize;

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
    let loudness = EbuR128::loudness_global_multiple(states.iter().map(|(s, _)| s))
        .expect("integrated loudness");
    let gain_db = TARGET_LUFS - loudness;
    let gain = (10f64).powf(gain_db / 20.0) as f32;

    let _ = fs::remove_file(&args.out);
    let mut writer = match FlacSampleWriter::create(&args.out, Options::default(), SR, 16, 2, None)
    {
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

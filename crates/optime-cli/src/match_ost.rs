//! Matches an archive's songs against a folder of reference recordings, so a game's numbered song
//! ids can be paired with the track listing of its official soundtrack.
//!
//! A GBA/DS song table carries no titles, and a soundtrack release carries no song ids, so the only
//! thing the two have in common is the audio itself. This renders every song in the archive, decodes
//! every reference recording, reduces both to the same chroma feature, and reports which songs each
//! reference track sounds like. The report is for a human to read: short jingles and near-duplicate
//! cues are exactly where an automatic match goes wrong, so nothing here writes a song-name table.
//!
//! Reference tracks are visited in sorted relative-path order, which for a normal soundtrack rip
//! (`Disc 1/01 - ….mp3`) is album order.

use std::collections::HashMap;
use std::f32::consts::PI;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use clap::Args as ClapArgs;
use indicatif::{ProgressBar, ProgressStyle};
use optime_core::devices::gba::GbaRom;
use optime_core::{PerDeviceSettings, SoundData, SynthController, load_all};
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex32};
use serde_json::{Value, json};

/// Analysis frames per second. Both sides are framed at this rate regardless of their native
/// sample rates, so no resampling is needed anywhere in the pipeline.
const FRAME_RATE: f64 = 10.0;
/// Nominal analysis window length. The real window is the nearest power of two in samples, so it
/// varies a little with the source rate; chroma is insensitive to that.
const WINDOW_SECONDS: f64 = 0.2;
/// Chroma ignores everything outside this band: below it is bass rumble the pitch mapping resolves
/// badly, above it is mostly percussion and aliasing.
const MIN_HZ: f32 = 55.0;
const MAX_HZ: f32 = 5000.0;
/// The furthest the two recordings may be shifted against each other when scoring, in frames. A
/// reference rip's lead-in and our render's start rarely differ by more than a couple of seconds.
const MAX_LAG_FRAMES: usize = 40;
/// A comparison needs at least this many overlapping frames to mean anything. Kept low because a
/// game's short cues (a level-up flourish, a jingle for receiving an item) are only a second or two
/// long, and refusing to score them at all is less useful than scoring them and flagging the result.
const MIN_OVERLAP_FRAMES: usize = 8;
/// Overlap (in frames) a comparison needs before its score is trusted at face value. A mean cosine
/// over a handful of frames is mostly noise and will happily reach 0.9 between unrelated music, so
/// shorter comparisons are scaled down in proportion. Without this a sub-second fragment elsewhere
/// in the ROM outranks the real, correctly-sized match.
const CONFIDENT_OVERLAP_FRAMES: usize = 30;
/// Frames quieter than this (relative to the loudest frame) are treated as silence and trimmed off
/// each end before scoring.
const SILENCE_FLOOR: f32 = 0.02;

/// The rate the archive's songs are rendered at.
const RENDER_RATE: u32 = 32_768;

#[derive(ClapArgs)]
#[command(
    about = "Match an archive's song ids against a folder of reference soundtrack recordings."
)]
pub struct Args {
    /// Decompressed sound archive (DS SDAT, DSE, or GBA `.gbaaudio`).
    archive: PathBuf,
    /// Folder of reference recordings, searched recursively. Sorted relative path = album order.
    reference_dir: PathBuf,
    /// Curated `[{ "songId", "title" }]` JSON, so each candidate is shown with the title it
    /// already has. Purely informational.
    #[arg(long)]
    names: Option<PathBuf>,
    /// Also write the full report here as JSON.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Seconds of each recording to compare.
    #[arg(long, default_value_t = 30.0)]
    seconds: f64,
    /// Candidates to list per reference track.
    #[arg(long, default_value_t = 5)]
    top: usize,
    /// Scores below this are called out as weak matches.
    #[arg(long, default_value_t = 0.5)]
    min_score: f32,
}

/// A recording reduced to what the matcher compares: one L2-normalised, mean-removed chroma vector
/// per analysis frame, silence already trimmed from both ends.
struct Chroma {
    frames: Vec<[f32; 12]>,
}

impl Chroma {
    /// Reduces mono `samples` at `rate` Hz to chroma frames.
    fn analyze(samples: &[f32], rate: f64) -> Self {
        let hop = (rate / FRAME_RATE).round().max(1.0) as usize;
        // The nearest power of two to the nominal window, so the FFT stays cheap.
        let n = {
            let want = (WINDOW_SECONDS * rate).max(64.0);
            1usize << (want.log2().round() as u32)
        };
        if samples.len() < n {
            return Self { frames: Vec::new() };
        }

        // Hann window, and the pitch class each FFT bin lands in (or none, outside the band).
        let window: Vec<f32> = (0..n)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos())
            .collect();
        let bin_pc: Vec<Option<usize>> = (0..n / 2)
            .map(|k| {
                let hz = k as f32 * rate as f32 / n as f32;
                (MIN_HZ..=MAX_HZ).contains(&hz).then(|| {
                    // Semitones above A440, wrapped into a pitch class.
                    let semitone = 12.0 * (hz / 440.0).log2();
                    (semitone.round() as i32).rem_euclid(12) as usize
                })
            })
            .collect();

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n);
        let mut scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
        let mut buf = vec![Complex32::default(); n];

        let mut frames = Vec::new();
        let mut energies = Vec::new();
        let mut start = 0;
        while start + n <= samples.len() {
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = Complex32::new(samples[start + i] * window[i], 0.0);
            }
            fft.process_with_scratch(&mut buf, &mut scratch);

            let mut pc = [0.0f32; 12];
            for (k, class) in bin_pc.iter().enumerate() {
                if let Some(c) = class {
                    pc[*c] += buf[k].norm();
                }
            }
            energies.push(pc.iter().sum::<f32>());
            frames.push(pc);
            start += hop;
        }

        // Trim the silent head and tail, judged against the loudest frame in the recording.
        let peak = energies.iter().copied().fold(0.0f32, f32::max);
        let threshold = peak * SILENCE_FLOOR;
        let first = energies.iter().position(|&e| e > threshold);
        let Some(first) = first else {
            return Self { frames: Vec::new() };
        };
        let last = energies.iter().rposition(|&e| e > threshold).unwrap();
        frames.drain(last + 1..);
        frames.drain(..first);

        // Per frame: normalise away loudness, then remove the mean across the twelve classes. The
        // mean is the frame's overall tonal density, which is similar for all tonal music; what
        // discriminates one song from another is how the energy *deviates* from it.
        for f in &mut frames {
            let mean = f.iter().sum::<f32>() / 12.0;
            for v in f.iter_mut() {
                *v -= mean;
            }
            let norm = f.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 1e-9 {
                for v in f.iter_mut() {
                    *v /= norm;
                }
            }
        }
        Self { frames }
    }

    /// The best per-frame cosine similarity between the two chroma sequences, over every alignment
    /// within [`MAX_LAG_FRAMES`], scaled down when the compared span is too short to trust (see
    /// [`CONFIDENT_OVERLAP_FRAMES`]). `-1.0` when they cannot be compared at all.
    fn similarity(&self, other: &Chroma) -> f32 {
        let (a, b) = (&self.frames, &other.frames);
        if a.len() < MIN_OVERLAP_FRAMES || b.len() < MIN_OVERLAP_FRAMES {
            return -1.0;
        }
        let mut best = -1.0f32;
        // `lag` shifts `b` right relative to `a`; the negative half shifts it left.
        for lag in -(MAX_LAG_FRAMES as isize)..=(MAX_LAG_FRAMES as isize) {
            let (a_start, b_start) = if lag >= 0 {
                (lag as usize, 0)
            } else {
                (0, (-lag) as usize)
            };
            let overlap = (a.len().saturating_sub(a_start)).min(b.len().saturating_sub(b_start));
            if overlap < MIN_OVERLAP_FRAMES {
                continue;
            }
            let mut sum = 0.0f32;
            for i in 0..overlap {
                let (fa, fb) = (&a[a_start + i], &b[b_start + i]);
                sum += (0..12).map(|c| fa[c] * fb[c]).sum::<f32>();
            }
            let confidence = (overlap as f32 / CONFIDENT_OVERLAP_FRAMES as f32).min(1.0);
            best = best.max(sum / overlap as f32 * confidence);
        }
        best
    }
}

/// Every file under `dir` that looks like audio, in sorted relative-path order — which for a
/// standard soundtrack rip (`Disc 1/01 - ….mp3`) is the album's own track order.
fn reference_files(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    const AUDIO_EXTENSIONS: &[&str] = &["mp3", "flac", "wav", "m4a", "aac", "ogg"];
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d)? {
            let path = entry?.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .extension()
                .and_then(|e| e.to_str())
                .map(str::to_ascii_lowercase)
                .is_some_and(|e| AUDIO_EXTENSIONS.contains(&e.as_str()))
            {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Decodes an audio file to mono samples, returning them with the file's sample rate.
fn decode_reference(path: &Path) -> Result<(Vec<f32>, f64), String> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| e.to_string())?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| "no default track".to_string())?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| e.to_string())?;

    let mut mono = Vec::new();
    let mut rate = 0f64;
    let mut interleaved: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            // Both of these are how symphonia reports a clean end of stream.
            Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(Error::ResetRequired) => break,
            Err(e) => return Err(e.to_string()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            // A damaged packet is worth skipping rather than failing the whole file.
            Err(Error::DecodeError(_)) => continue,
            Err(e) => return Err(e.to_string()),
        };
        let spec = *decoded.spec();
        rate = f64::from(spec.rate);
        let channels = spec.channels.count().max(1);
        let buf =
            interleaved.get_or_insert_with(|| SampleBuffer::new(decoded.capacity() as u64, spec));
        buf.copy_interleaved_ref(decoded);
        for frame in buf.samples().chunks_exact(channels) {
            mono.push(frame.iter().sum::<f32>() / channels as f32);
        }
    }
    if mono.is_empty() || rate <= 0.0 {
        return Err("decoded no audio".to_string());
    }
    Ok((mono, rate))
}

/// Renders `seconds` of one song from the archive, downmixed to mono.
fn render_song(
    data: &dyn SoundData,
    song_id: u32,
    config: &PerDeviceSettings,
    frames: usize,
) -> Vec<f32> {
    let Some(mut controller) = SynthController::new(f64::from(RENDER_RATE), data, song_id) else {
        return Vec::new();
    };
    let mut mono = Vec::with_capacity(frames);
    let mut buf = vec![0.0f32; 2 * 512];
    while mono.len() < frames {
        let n = 512.min(frames - mono.len());
        let chunk = &mut buf[..2 * n];
        controller.fill(chunk, config);
        for frame in chunk.chunks_exact(2) {
            mono.push(0.5 * (frame[0] + frame[1]));
        }
    }
    mono
}

/// `songId -> curated title`, from a `[{ "songId", "title" }]` table. Missing file = empty map.
fn curated_titles(path: Option<&PathBuf>) -> HashMap<u32, String> {
    let Some(path) = path else {
        return HashMap::new();
    };
    let Ok(text) = std::fs::read_to_string(path) else {
        eprintln!("Warning: could not read '{}'", path.display());
        return HashMap::new();
    };
    let Ok(json) = serde_json::from_str::<Value>(&text) else {
        eprintln!("Warning: could not parse '{}'", path.display());
        return HashMap::new();
    };
    json.as_array()
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .filter_map(|e| {
            let id = u32::try_from(e["songId"].as_u64()?).ok()?;
            Some((id, e["title"].as_str().unwrap_or("").to_string()))
        })
        .collect()
}

/// One reference track's ranked candidates.
struct Match {
    /// Path relative to the reference directory, which is also the album track order key.
    label: String,
    /// `(song id, score)`, best first.
    candidates: Vec<(u32, f32)>,
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
    let data: Arc<dyn SoundData> = Arc::from(data);

    let refs = match reference_files(&args.reference_dir) {
        Ok(r) if !r.is_empty() => r,
        Ok(_) => {
            eprintln!("No audio files under '{}'.", args.reference_dir.display());
            return ExitCode::FAILURE;
        }
        Err(e) => {
            eprintln!("Failed to read '{}': {e}", args.reference_dir.display());
            return ExitCode::FAILURE;
        }
    };

    let titles = curated_titles(args.names.as_ref());
    let song_ids = data.song_ids();
    // Match on the raw engine output: the enhancement DSP colours the sound in ways the reference
    // recording never had, and chroma only cares about which pitches are sounding.
    let config = PerDeviceSettings::neutral();
    let render_frames = (args.seconds * f64::from(RENDER_RATE)) as usize;
    eprintln!(
        "Archive: {} songs ({}). Reference: {} recordings under '{}'. Comparing {:.0}s of each.",
        song_ids.len(),
        if data.as_any().downcast_ref::<GbaRom>().is_some() {
            "GBA"
        } else {
            "DS/other"
        },
        refs.len(),
        args.reference_dir.display(),
        args.seconds,
    );

    let bar_style =
        ProgressStyle::with_template("  {msg} [{bar:32.cyan/blue}] {pos}/{len} ({elapsed})")
            .unwrap()
            .progress_chars("=>-");

    // --- Analyze both sides. ---
    let rom_bar = ProgressBar::new(song_ids.len() as u64);
    rom_bar.set_style(bar_style.clone());
    rom_bar.set_message("render ");
    let rom_chroma: Vec<(u32, Chroma)> = song_ids
        .par_iter()
        .map(|&id| {
            let mono = render_song(&*data, id, &config, render_frames);
            rom_bar.inc(1);
            (id, Chroma::analyze(&mono, f64::from(RENDER_RATE)))
        })
        .collect();
    rom_bar.finish_and_clear();

    let ref_bar = ProgressBar::new(refs.len() as u64);
    ref_bar.set_style(bar_style);
    ref_bar.set_message("decode ");
    let ref_chroma: Vec<(String, Option<Chroma>)> = refs
        .par_iter()
        .map(|path| {
            let label = path
                .strip_prefix(&args.reference_dir)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            let result = decode_reference(path).map(|(mono, rate)| {
                // Only the compared span is needed, and decoding stops mattering past it.
                let take = (args.seconds * rate) as usize;
                Chroma::analyze(&mono[..take.min(mono.len())], rate)
            });
            ref_bar.inc(1);
            match result {
                Ok(c) => (label, Some(c)),
                Err(e) => {
                    eprintln!("  {label}: {e}");
                    (label, None)
                }
            }
        })
        .collect();
    ref_bar.finish_and_clear();

    // --- Score every reference track against every song. ---
    let matches: Vec<Match> = ref_chroma
        .par_iter()
        .map(|(label, chroma)| {
            let mut candidates: Vec<(u32, f32)> = match chroma {
                Some(c) => rom_chroma
                    .iter()
                    .map(|(id, rc)| (*id, c.similarity(rc)))
                    .collect(),
                None => Vec::new(),
            };
            candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
            candidates.truncate(args.top.max(1));
            Match {
                label: label.clone(),
                candidates,
            }
        })
        .collect();

    // A song claimed as the best match by more than one reference track means at least one of them
    // is wrong, so the report flags every track involved.
    let mut claims: HashMap<u32, usize> = HashMap::new();
    for m in &matches {
        if let Some(&(id, _)) = m.candidates.first() {
            *claims.entry(id).or_default() += 1;
        }
    }

    // --- Report. ---
    let mut weak = 0usize;
    let mut contested = 0usize;
    for m in &matches {
        let Some(&(best_id, best_score)) = m.candidates.first() else {
            println!("\n{}\n    (no candidates - could not analyze)", m.label);
            continue;
        };
        let mut flags = Vec::new();
        if best_score < args.min_score {
            flags.push("WEAK");
            weak += 1;
        }
        if claims.get(&best_id).copied().unwrap_or(0) > 1 {
            flags.push("CONTESTED");
            contested += 1;
        }
        let flag = if flags.is_empty() {
            String::new()
        } else {
            format!("   << {}", flags.join(" + "))
        };
        println!("\n{}{flag}", m.label);
        for (rank, &(id, score)) in m.candidates.iter().enumerate() {
            let title = titles.get(&id).map(String::as_str).unwrap_or("");
            let marker = if rank == 0 { '*' } else { ' ' };
            println!("  {marker} {score:>6.3}  songId {id:<5} {title}");
        }
    }
    eprintln!(
        "\n{} reference tracks matched; {weak} weak (< {:.2}), {contested} contested. \
         Check those by ear before trusting them.",
        matches.len(),
        args.min_score,
    );

    if let Some(out) = &args.out {
        let report = Value::Array(
            matches
                .iter()
                .map(|m| {
                    json!({
                        "track": m.label,
                        "candidates": m.candidates.iter().map(|&(id, score)| json!({
                            "songId": id,
                            "score": score,
                            "curatedTitle": titles.get(&id),
                        })).collect::<Vec<_>>(),
                    })
                })
                .collect(),
        );
        let text = serde_json::to_string_pretty(&report).expect("serialize report");
        if let Err(e) = std::fs::write(out, text) {
            eprintln!("Failed to write '{}': {e}", out.display());
            return ExitCode::FAILURE;
        }
        eprintln!("Wrote {}", out.display());
    }
    ExitCode::SUCCESS
}

//! Renders a whole soundtrack on the calling thread, for a sampling profiler (AMD uProf, Intel
//! VTune). Defaults to the Pokémon Emerald demo archive and its curated song table, so the
//! executable can be launched with no arguments at all.
//!
//! The workload is exactly the album exporter's: [`album::render_song`] per song with the console's
//! high-quality preset, in curated album order. The difference is the surrounding machinery, all of
//! which is absent here — no rayon pool, no temp files, no FLAC encoder, no loudness pass, no
//! progress bars. One thread runs one song at a time, so every sample the profiler collects lands
//! in the engine, and the call tree is a single stack rather than N worker stacks that have to be
//! merged before a hot leaf is visible.
//!
//! `export-album --benchmark` remains the throughput number to quote; it is deliberately
//! multithreaded and reports a median of timed passes. This tool reports its wall time too, but
//! that figure is a sanity check on the run the profiler just observed, not a benchmark result.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;
use optime_core::load_all;

use crate::album::{SR, album_order, high_quality_preset, render_song};

/// Defaults baked in at compile time, relative to this crate's directory, so the executable renders
/// the intended soundtrack from any working directory. Profilers commonly launch a target from
/// their own project directory rather than the shell's.
const DEFAULT_ARCHIVE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../demos/pokemon-emerald.gbaaudio"
);
const DEFAULT_NAMES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../crates/optime-app/src/song_names/pokemon_emerald.json"
);

/// Frame stride for the checksum. The rendered frames are otherwise dropped unread, and summing
/// every one of them would put a memory-bandwidth pass over ~100 MB per song into the profile
/// alongside the render being measured. A prime stride touches every song's whole buffer for a
/// fraction of that cost.
const CHECKSUM_STRIDE: usize = 1021;

#[derive(Parser)]
#[command(
    name = "profile-emerald",
    about = "Render a whole soundtrack single-threaded, for AMD uProf / Intel VTune.",
    version
)]
pub struct Args {
    /// Decompressed sound archive (DS SDAT, DSE, or GBA `.gbaaudio`).
    #[arg(default_value = DEFAULT_ARCHIVE)]
    pub archive: PathBuf,
    /// Curated `[{ "songId", "title" }]` JSON; its array order is the render order.
    #[arg(default_value = DEFAULT_NAMES)]
    pub names_json: PathBuf,
    /// Only render the first N songs.
    #[arg(long)]
    pub limit: Option<usize>,
    /// Render the whole soundtrack this many times, to lengthen the profiler's collection window.
    #[arg(long, default_value_t = 1)]
    pub repeat: usize,
    /// Don't print a line per song (leaves stderr silent until the summary).
    #[arg(long)]
    pub quiet: bool,
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
    let (config, is_gba) = high_quality_preset(&*data);
    let album = match album_order(&*data, &args.names_json, args.limit) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let sample_bits = optime_core::SAMPLE_SIZE_BYTES * 8;
    eprintln!(
        "profile-emerald: {} tracks from '{}', {} high-quality preset, Sample = f{sample_bits}, 1 thread, {} pass(es).",
        album.len(),
        args.archive.display(),
        if is_gba { "GBA" } else { "DS/other" },
        args.repeat.max(1),
    );

    for pass in 1..=args.repeat.max(1) {
        let t0 = Instant::now();
        let mut frames: u64 = 0;
        let mut checksum = 0.0f64;
        for (i, (id, title)) in album.iter().enumerate() {
            let song_start = Instant::now();
            let rendered = render_song(&*data, *id, &config);
            frames += rendered.len() as u64;
            // Reading the render back keeps it observably used, so no amount of inlining can
            // discard the work the profiler is here to measure.
            for &(l, r) in rendered.iter().step_by(CHECKSUM_STRIDE) {
                checksum += f64::from(l.abs() + r.abs());
            }
            if !args.quiet {
                eprintln!(
                    "  [{}/{}] pass {pass} songId {id:<5} {:6.1}s audio in {:6.2}s  \"{title}\"",
                    i + 1,
                    album.len(),
                    rendered.len() as f64 / f64::from(SR),
                    song_start.elapsed().as_secs_f64(),
                );
            }
        }
        let wall = t0.elapsed().as_secs_f64();
        let audio_s = frames as f64 / f64::from(SR);
        eprintln!(
            "pass {pass}: {frames} frames ({audio_s:.1}s audio) in {wall:.3}s  →  {:.1}x realtime, {:.2} Msamp/s, checksum {checksum:.3}",
            audio_s / wall,
            frames as f64 * 2.0 / wall / 1.0e6,
        );
    }
    ExitCode::SUCCESS
}

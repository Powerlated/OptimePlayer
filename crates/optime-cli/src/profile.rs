use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;
use optime_core::load_all;

use crate::album::{SR, album_order, high_quality_preset, render_song};

const DEFAULT_ARCHIVE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../demos/pokemon-emerald.gbaaudio"
);
const DEFAULT_NAMES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../crates/optime-app/src/song_names/pokemon_emerald.json"
);

const CHECKSUM_STRIDE: usize = 1021;

#[derive(Parser)]
#[command(
    name = "profile-emerald",
    about = "Render a soundtrack single-threaded, for AMD uProf / Intel VTune.",
    version
)]
pub struct Args {
    #[arg(default_value = DEFAULT_ARCHIVE)]
    pub archive: PathBuf,
    #[arg(default_value = DEFAULT_NAMES)]
    pub names_json: PathBuf,
    #[arg(long, default_value = "10")]
    pub limit: Option<usize>,
    #[arg(long, default_value_t = 1)]
    pub repeat: usize,
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

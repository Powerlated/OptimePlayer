//! Builds a reference `timbre::Target` from a folder of finished recordings — the corpus that
//! stands in for what the rendered music is being asked to sound like.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Args as ClapArgs;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

use crate::timbre::{BAND_COUNT, Profile, Target};

const AUDIO_EXTENSIONS: [&str; 6] = ["ogg", "flac", "mp3", "wav", "m4a", "opus"];

#[derive(ClapArgs)]
#[command(
    about = "Reduce a folder of recordings to one reference timbre profile (spectrum + dynamics)."
)]
pub struct Args {
    reference_dir: PathBuf,
    out: PathBuf,
    #[arg(long, default_value_t = 90.0)]
    seconds: f64,
    #[arg(long)]
    limit: Option<usize>,
}

fn audio_files(dir: &std::path::Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| AUDIO_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
                .unwrap_or(false)
        })
        .collect();
    files.sort();
    files
}

pub fn run(args: Args) -> ExitCode {
    let mut files = audio_files(&args.reference_dir);
    if files.is_empty() {
        eprintln!(
            "No audio files found in '{}'.",
            args.reference_dir.display()
        );
        return ExitCode::FAILURE;
    }
    if let Some(limit) = args.limit {
        files.truncate(limit);
    }

    let bar = ProgressBar::new(files.len() as u64);
    bar.set_style(
        ProgressStyle::with_template("{bar:40} {pos}/{len} {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );

    let results: Vec<(PathBuf, Result<Profile, String>)> = files
        .par_iter()
        .map(|path| {
            let profile = crate::timbre::analyze_file(path, Some(args.seconds));
            bar.inc(1);
            (path.clone(), profile)
        })
        .collect();
    bar.finish_and_clear();

    let mut profiles = Vec::new();
    let mut skipped = 0usize;
    for (path, result) in results {
        match result {
            Ok(p) => profiles.push(p),
            Err(e) => {
                skipped += 1;
                eprintln!("  skipped {}: {e}", path.display());
            }
        }
    }

    let Some(target) = Target::from_profiles(&profiles) else {
        eprintln!(
            "Need at least two analysable recordings; got {}.",
            profiles.len()
        );
        return ExitCode::FAILURE;
    };

    println!(
        "Reference profile from {} recordings ({skipped} skipped), first {:.0}s of each.",
        target.sources, args.seconds
    );
    println!(
        "  crest {:.1} ± {:.1} dB",
        target.mean.crest_db, target.deviation.crest_db
    );
    println!(
        "  dynamic range {:.1} ± {:.1} dB",
        target.mean.dynamic_range_db, target.deviation.dynamic_range_db
    );
    println!(
        "  flux {:.2} ± {:.2} dB/frame",
        target.mean.flux_db, target.deviation.flux_db
    );
    println!("  spectrum (dB about the mean band, low to high):");
    for b in 0..BAND_COUNT {
        println!(
            "    band {b:2}  {:+6.1} ± {:4.1}",
            target.mean.spectrum_db[b], target.deviation.spectrum_db[b]
        );
    }

    let json = match serde_json::to_string_pretty(&target) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("Failed to serialise the profile: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = std::fs::write(&args.out, json) {
        eprintln!("Failed to write '{}': {e}", args.out.display());
        return ExitCode::FAILURE;
    }
    println!("\nWrote {}", args.out.display());
    ExitCode::SUCCESS
}

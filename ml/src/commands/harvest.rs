//! Harvest real game songs into unlabeled note-event datasets for SSL pretraining →
//! `data/real_{train,val}.bin`. Each archive in `<rom_dir>` runs headlessly → `NoteEvent` windows,
//! split at the **song** level. Train windows use `--coverage ×` random offsets; val uses fixed
//! non-overlapping tiling for an honest held-out loss.
//!
//! `--annotate CODE=json` (repeatable) supplies weak is-music labels for a GBA game code from the
//! app's `song_names` JSON.

use clap::Args;
use optime_ml::data::{save_songs, GenConfig};
use optime_ml::harvest::{
    harvest_dir_full, sample_random_windows, slice_into_windows, Annotations,
};
use optime_ml::notes::Song;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct HarvestArgs {
    /// Directory of sound archives (ROMs / `.sdat` / `.gbaaudio`).
    pub rom_dir: PathBuf,
    /// Window length in frames (default from `GenConfig`).
    pub seq_len: Option<usize>,
    /// Fraction of songs held out for validation (default 0.1).
    pub val_fraction: Option<f64>,
    /// Random-offset overlapping-window multiplier for the train split (≥ 1, default 4).
    #[arg(long, default_value_t = 4.0)]
    pub coverage: f64,
    /// Weak is-music labels: `CODE=song_names.json`, repeatable.
    #[arg(long, value_parser = parse_annotate)]
    pub annotate: Vec<(String, PathBuf)>,
}

fn parse_annotate(spec: &str) -> Result<(String, PathBuf), String> {
    let (code, path) = spec
        .split_once('=')
        .ok_or_else(|| format!("--annotate expects CODE=path, got {spec:?}"))?;
    Ok((code.to_string(), PathBuf::from(path)))
}

pub fn run(args: HarvestArgs) {
    assert!(args.coverage >= 1.0, "--coverage must be ≥ 1");
    let dir = args.rom_dir.as_path();
    let seq_len = args.seq_len.unwrap_or_else(|| GenConfig::default().seq_len);
    let val_fraction = args.val_fraction.unwrap_or(0.1);
    let coverage = args.coverage;

    let annotations =
        Annotations::from_files(args.annotate.iter().map(|(c, p)| (c.as_str(), p.as_path())));

    println!(
        "harvesting {} (seq_len {seq_len}, train coverage {coverage}x)",
        dir.display()
    );
    // One (full note stream, is-music) per source song, so the split is at the
    // *song* level and train/val can be windowed differently afterward.
    let mut songs = harvest_dir_full(dir, &annotations).expect("read rom dir");
    if songs.is_empty() {
        eprintln!("no playable songs found in {}", dir.display());
        std::process::exit(1);
    }

    // Deterministic shuffle of *songs*, then split by song.
    let mut rng = StdRng::seed_from_u64(0x5EED_A110_0F50);
    songs.shuffle(&mut rng);
    let n_songs = songs.len();
    let n_val = ((n_songs as f64) * val_fraction)
        .round()
        .clamp(0.0, n_songs.saturating_sub(1) as f64) as usize;
    let val_songs = songs.split_off(n_songs - n_val);

    // Val: clean fixed-tiling windows. Train: random-offset overlapping windows.
    let val: Vec<Song> = val_songs
        .iter()
        .flat_map(|(notes, is_music)| slice_into_windows(notes, seq_len, *is_music))
        .collect();
    let train: Vec<Song> = songs
        .iter()
        .flat_map(|(notes, is_music)| {
            sample_random_windows(notes, seq_len, coverage, *is_music, &mut rng)
        })
        .collect();

    std::fs::create_dir_all("data").expect("create data dir");
    save_songs("data/real_train.bin", &train).expect("save real_train");
    save_songs("data/real_val.bin", &val).expect("save real_val");
    println!(
        "harvested {n_songs} songs → {coverage}x random-offset train ({}) + tiled val ({}) windows (song-level split) → data/real_train.bin, data/real_val.bin",
        train.len(),
        val.len()
    );
}

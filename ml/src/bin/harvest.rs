//! Harvest real game songs into unlabeled note-event datasets for
//! self-supervised pretraining.
//!
//! ```sh
//! cargo run --release --features harvest --bin harvest -- <rom_dir> [seq_len] [val_fraction] \
//!     [--annotate <GAME_CODE>=<song_names.json>]...
//! #   → data/real_train.bin, data/real_val.bin
//! ```
//!
//! Each ROM/audio archive in `<rom_dir>` is parsed with the engine's `load_all`,
//! every playable song is run headlessly, and its `SynthEvent` stream is turned
//! into `NoteEvent` windows on the same 4-frames-per-beat grid the synthetic
//! generator uses. The windows are shuffled deterministically and split into a
//! train/val set.
//!
//! `--annotate` (repeatable) supplies weak is-music labels for a GBA game code
//! from the app's `song_names` JSON, e.g.
//! `--annotate BPEE=../crates/optime-app/src/song_names/pokemon_emerald.json`.

use optime_ml::data::{save_songs, GenConfig};
use optime_ml::harvest::{harvest_dir, Annotations};
use optime_ml::notes::Song;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Split off repeatable `--annotate CODE=path` flags from the positional args.
    let mut annotate: Vec<(String, PathBuf)> = Vec::new();
    let mut positional: Vec<String> = Vec::new();
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        if a == "--annotate" {
            let spec = it.next().expect("--annotate needs CODE=path");
            let (code, path) = spec.split_once('=').expect("--annotate expects CODE=path");
            annotate.push((code.to_string(), PathBuf::from(path)));
        } else {
            positional.push(a.clone());
        }
    }
    if positional.is_empty() {
        eprintln!("usage: harvest <rom_dir> [seq_len] [val_fraction] [--annotate CODE=json]...");
        std::process::exit(2);
    }
    let dir = Path::new(&positional[0]);
    let seq_len: usize = positional
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(GenConfig::default().seq_len);
    let val_fraction: f64 = positional
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.1);

    let annotations =
        Annotations::from_files(annotate.iter().map(|(c, p)| (c.as_str(), p.as_path())));

    println!("harvesting {} (seq_len {seq_len})", dir.display());
    // One group of windows per source song, so the split is at the *song* level:
    // no two windows of the same song straddle the train/val boundary.
    let mut groups = harvest_dir(dir, seq_len, &annotations).expect("read rom dir");
    if groups.is_empty() {
        eprintln!("no playable songs found in {}", dir.display());
        std::process::exit(1);
    }

    // Deterministic shuffle of *songs*, then split by song.
    let mut rng = StdRng::seed_from_u64(0x5EED_A110_0F50);
    groups.shuffle(&mut rng);
    let n_songs = groups.len();
    let n_val = ((n_songs as f64) * val_fraction)
        .round()
        .clamp(0.0, n_songs.saturating_sub(1) as f64) as usize;
    let val_groups = groups.split_off(n_songs - n_val);
    let train: Vec<Song> = groups.into_iter().flatten().collect();
    let val: Vec<Song> = val_groups.into_iter().flatten().collect();

    std::fs::create_dir_all("data").expect("create data dir");
    save_songs("data/real_train.bin", &train).expect("save real_train");
    save_songs("data/real_val.bin", &val).expect("save real_val");
    println!(
        "harvested {n_songs} songs → {} train / {} val windows (song-level split) → data/real_train.bin, data/real_val.bin",
        train.len(),
        val.len()
    );
}

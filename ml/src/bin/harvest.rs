//! Harvest real game songs into a **whole-song** dataset for self-supervised
//! pretraining.
//!
//! ```sh
//! cargo run --release --features harvest --bin harvest -- <rom_dir> [max_frames] [val_fraction] \
//!     [--annotate <GAME_CODE>=<song_names.json>]...
//! #   → data/real_train.bin, data/real_val.bin   (whole songs, variable n_frames)
//! ```
//!
//! Each ROM/audio archive in `<rom_dir>` is parsed with the engine's `load_all`,
//! every playable song is run headlessly as **intro + loop + loop** (the run stops
//! at the second loop point, so the stream contains one full crossing of the loop
//! boundary), and stored **whole** — windowing/packing happens at load time
//! (`pack::window_dataset` for the fixed-window generations, `pack::pack_songs`
//! for the long-context ones). Songs longer than `max_frames − 1` frames are
//! truncated (one slot is reserved for the EOS token). Songs are shuffled
//! deterministically and split at the **song** level.
//!
//! After harvesting, the song-length distribution is printed — that is what sizes
//! the long-context model's `n_frames`.
//!
//! `--annotate` (repeatable) supplies weak is-music labels for a GBA game code
//! from the app's `song_names` JSON, e.g.
//! `--annotate BPEE=../crates/optime-app/src/song_names/pokemon_emerald.json`.

use optime_ml::data::save_songs;
use optime_ml::harvest::{harvest_dir_full, Annotations};
use optime_ml::notes::Song;
use optime_ml::pack::truncate_song;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use std::path::{Path, PathBuf};

fn print_length_stats(songs: &[Song]) {
    let mut lens: Vec<usize> = songs.iter().map(|s| s.n_frames).collect();
    lens.sort_unstable();
    let pct = |p: f64| lens[((lens.len() - 1) as f64 * p) as usize];
    println!(
        "song lengths (frames): min {}  median {}  p90 {}  p99 {}  max {}   ({} songs, {} looped)",
        lens.first().unwrap(),
        pct(0.5),
        pct(0.9),
        pct(0.99),
        lens.last().unwrap(),
        lens.len(),
        songs.iter().filter(|s| s.loop_frame.is_some()).count(),
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

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
        eprintln!("usage: harvest <rom_dir> [max_frames] [val_fraction] [--annotate CODE=json]...");
        std::process::exit(2);
    }
    let dir = Path::new(&positional[0]);
    let max_frames: usize = positional
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);
    let val_fraction: f64 = positional
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.1);

    let annotations =
        Annotations::from_files(annotate.iter().map(|(c, p)| (c.as_str(), p.as_path())));

    println!(
        "harvesting {} (whole songs, intro+loop+loop, truncated at {} frames)",
        dir.display(),
        max_frames - 1
    );
    let songs = harvest_dir_full(dir, &annotations).expect("read rom dir");
    if songs.is_empty() {
        eprintln!("no playable songs found in {}", dir.display());
        std::process::exit(1);
    }

    print_length_stats(&songs);
    let over: usize = songs.iter().filter(|s| s.n_frames > max_frames - 1).count();
    if over > 0 {
        println!(
            "truncating {over} songs longer than {} frames",
            max_frames - 1
        );
    }
    let mut songs: Vec<Song> = songs
        .iter()
        .map(|s| truncate_song(s, max_frames - 1))
        .collect();

    // Deterministic shuffle of songs, then split by song.
    let mut rng = StdRng::seed_from_u64(0x5EED_A110_0F50);
    songs.shuffle(&mut rng);
    let n_songs = songs.len();
    let n_val = ((n_songs as f64) * val_fraction)
        .round()
        .clamp(0.0, n_songs.saturating_sub(1) as f64) as usize;
    let val = songs.split_off(n_songs - n_val);

    std::fs::create_dir_all("data").expect("create data dir");
    save_songs("data/real_train.bin", &songs).expect("save real_train");
    save_songs("data/real_val.bin", &val).expect("save real_val");
    println!(
        "harvested {n_songs} whole songs → {} train + {} val (song-level split) → data/real_train.bin, data/real_val.bin",
        songs.len(),
        val.len()
    );
}

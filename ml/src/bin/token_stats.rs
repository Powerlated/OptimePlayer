//! Empirically size the event tokenizer's `MAX_TOKENS` cap from the real corpus.
//!
//! One event token = one note-on in a 128-frame window, so tokens-per-window =
//! notes-per-window. This prints the distribution over the harvested real windows
//! (`data/real_{train,val}.bin`) plus how many windows a few candidate caps would
//! truncate, so the cap can be chosen to cover ~all real music.
//!
//! Usage: cargo run --release --bin token_stats

use optime_ml::data::load_songs;

fn main() {
    let mut counts: Vec<usize> = Vec::new();
    for path in ["data/real_train.bin", "data/real_val.bin"] {
        match load_songs(path) {
            Ok(songs) => counts.extend(songs.iter().map(|s| s.notes.len())),
            Err(e) => eprintln!("skip {path}: {e}"),
        }
    }
    if counts.is_empty() {
        eprintln!("no windows found — run `harvest` first");
        std::process::exit(1);
    }
    counts.sort_unstable();
    let n = counts.len();
    let pct = |p: f64| counts[((p / 100.0 * n as f64) as usize).min(n - 1)];
    let mean = counts.iter().sum::<usize>() as f64 / n as f64;

    println!("windows: {n}");
    println!("notes/window  mean {mean:.1}");
    for p in [50.0, 90.0, 99.0, 99.5, 99.9] {
        println!("  p{p:<5} {}", pct(p));
    }
    println!("  max    {}", counts[n - 1]);
    println!(
        "also: mean tokens/frame = {:.2} (over 128 frames)",
        mean / 128.0
    );
    println!("truncation by candidate MAX_TOKENS:");
    for cap in [256usize, 384, 512, 640, 768, 1024] {
        let over = counts.iter().filter(|&&c| c > cap).count();
        println!(
            "  cap {cap:>4}: {over} windows truncated ({:.3}%)",
            100.0 * over as f64 / n as f64
        );
    }

    // Per-frame polyphony (notes SOUNDING in a frame): sizes MAX_POLY for the
    // hierarchical set-transformer pooling (its per-frame sequence length).
    let mut poly: Vec<usize> = Vec::new();
    for path in ["data/real_train.bin", "data/real_val.bin"] {
        if let Ok(songs) = load_songs(path) {
            for s in &songs {
                let mut per_frame = vec![0usize; s.n_frames];
                for note in &s.notes {
                    let end = (note.end_frame as usize).min(s.n_frames);
                    for slot in per_frame[(note.start_frame as usize)..end].iter_mut() {
                        *slot += 1;
                    }
                }
                poly.extend(per_frame);
            }
        }
    }
    poly.sort_unstable();
    let m = poly.len();
    let ppct = |p: f64| poly[((p / 100.0 * m as f64) as usize).min(m - 1)];
    println!(
        "\nsounding notes/frame  mean {:.2}",
        poly.iter().sum::<usize>() as f64 / m as f64
    );
    for p in [50.0, 90.0, 99.0, 99.9, 99.99] {
        println!("  p{p:<6} {}", ppct(p));
    }
    println!("  max      {}", poly[m - 1]);
    for cap in [16usize, 24, 32, 48, 64] {
        let over = poly.iter().filter(|&&c| c > cap).count();
        println!(
            "  MAX_POLY {cap:>3}: {over} frames clipped ({:.4}%)",
            100.0 * over as f64 / m as f64
        );
    }
}

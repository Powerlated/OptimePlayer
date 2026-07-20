//! Generate and retain the synthetic dataset.
//!
//! Usage:
//!   cargo run --release --bin generate_data -- [n_train] [n_val] [seq_len] [seed]
//!
//! Writes raw note-event songs (the retained data-of-record) to:
//!   data/train.bin, data/val.bin, data/gen_config.json

use optime_ml::data::{self, GenConfig};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = GenConfig::default();
    if let Some(v) = args.get(1) {
        cfg.n_train = v.parse().expect("n_train");
    }
    if let Some(v) = args.get(2) {
        cfg.n_val = v.parse().expect("n_val");
    }
    if let Some(v) = args.get(3) {
        cfg.seq_len = v.parse().expect("seq_len");
    }
    if let Some(v) = args.get(4) {
        cfg.seed = v.parse().expect("seed");
    }

    let out = PathBuf::from("data");
    std::fs::create_dir_all(&out).expect("create data dir");

    println!(
        "generating {} train + {} val songs, seq_len {} ...",
        cfg.n_train, cfg.n_val, cfg.seq_len
    );
    let (train, val) = data::build(&cfg);

    data::save_songs(out.join("train.bin"), &train).expect("save train");
    data::save_songs(out.join("val.bin"), &val).expect("save val");
    let cfg_json = serde_json::to_string_pretty(&cfg).unwrap();
    std::fs::write(out.join("gen_config.json"), cfg_json).expect("save config");

    // Quick stats.
    let notes: usize = train.iter().map(|s| s.notes.len()).sum();
    println!(
        "wrote data/train.bin ({} songs, {} note events), data/val.bin ({} songs)",
        train.len(),
        notes,
        val.len()
    );
}

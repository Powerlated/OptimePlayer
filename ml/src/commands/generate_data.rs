//! Generate and retain the synthetic dataset → `data/{train,val}.bin` + `data/gen_config.json`
//! (raw note-event songs, the retained data-of-record; features are derived on load). Clobbers;
//! deterministic from `seed`.

use clap::Args;
use optime_ml::data::{self, GenConfig};
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct GenerateDataArgs {
    /// Number of training songs (default from `GenConfig`).
    pub n_train: Option<usize>,
    /// Number of validation songs.
    pub n_val: Option<usize>,
    /// Window length in frames. Must match the model's; a dataset windowed at one length cannot
    /// train a model built for another.
    pub seq_len: Option<usize>,
    /// RNG seed.
    pub seed: Option<u64>,
}

pub fn run(args: GenerateDataArgs) {
    let mut cfg = GenConfig::default();
    if let Some(v) = args.n_train {
        cfg.n_train = v;
    }
    if let Some(v) = args.n_val {
        cfg.n_val = v;
    }
    if let Some(v) = args.seq_len {
        cfg.seq_len = v;
    }
    if let Some(v) = args.seed {
        cfg.seed = v;
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

    let notes: usize = train.iter().map(|s| s.notes.len()).sum();
    println!(
        "wrote data/train.bin ({} songs, {} note events), data/val.bin ({} songs)",
        train.len(),
        notes,
        val.len()
    );
}

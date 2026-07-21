//! Synthetic dataset: generation, on-disk retention, and the training example
//! shape. The *raw note-event songs* are the canonical retained data (they hold
//! all pitch + metadata); per-frame features are derived deterministically on
//! load via [`crate::features`], so the same extractor serves training and live
//! inference.

use crate::features::{self, FEATURE_DIM};
use crate::notes::{render_song, Song};
use crate::theory::Key;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Parameters for a synthetic dataset build.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenConfig {
    pub n_train: usize,
    pub n_val: usize,
    /// Frames per song (fixed sequence length for batching). Must match the
    /// backbones' positional-embedding table size — 256 frames = 64 beats = 32s at
    /// 120bpm. A dataset built at one window can't train a model built for another.
    pub seq_len: usize,
    pub seed: u64,
}

impl Default for GenConfig {
    fn default() -> Self {
        GenConfig {
            n_train: 4000,
            n_val: 500,
            seq_len: 256,
            seed: 0xC0FFEE,
        }
    }
}

/// One flattened training example ready for tensor conversion.
#[derive(Debug, Clone)]
pub struct Example {
    pub seq_len: usize,
    /// `seq_len * FEATURE_DIM`, row-major.
    pub features: Vec<f32>,
    pub key_label: usize,
    /// `seq_len` chord labels.
    pub chord_labels: Vec<usize>,
}

impl Example {
    pub fn from_song(song: &Song) -> Example {
        let grid = features::extract_song(song);
        debug_assert_eq!(grid.data.len(), song.n_frames * FEATURE_DIM);
        Example {
            seq_len: song.n_frames,
            features: grid.data,
            key_label: song.key_label,
            chord_labels: song.chord_labels.clone(),
        }
    }
}

/// Generate `n` songs, each a random key (uniform over 24) rendered to `seq_len`
/// frames, using a deterministic per-song RNG stream from `base_seed`.
pub fn generate_songs(n: usize, seq_len: usize, base_seed: u64) -> Vec<Song> {
    (0..n)
        .map(|i| {
            let mut rng =
                StdRng::seed_from_u64(base_seed.wrapping_add(i as u64).wrapping_mul(0x9E3779B9));
            let key = Key::from_label(rng.gen_range(0..24));
            render_song(&mut rng, &key, seq_len)
        })
        .collect()
}

/// Build train + val song sets from a config.
pub fn build(config: &GenConfig) -> (Vec<Song>, Vec<Song>) {
    let train = generate_songs(config.n_train, config.seq_len, config.seed);
    let val = generate_songs(config.n_val, config.seq_len, config.seed ^ 0xDEADBEEF);
    (train, val)
}

/// Serialize songs to a bincode file (the retained raw dataset).
pub fn save_songs<P: AsRef<Path>>(path: P, songs: &[Song]) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, songs).map_err(std::io::Error::other)?;
    Ok(())
}

/// Load songs previously written by [`save_songs`].
pub fn load_songs<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<Song>> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let songs = bincode::deserialize_from(reader).map_err(std::io::Error::other)?;
    Ok(songs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn examples_have_consistent_shape() {
        let songs = generate_songs(8, 64, 1);
        for s in &songs {
            let ex = Example::from_song(s);
            assert_eq!(ex.seq_len, 64);
            assert_eq!(ex.features.len(), 64 * FEATURE_DIM);
            assert_eq!(ex.chord_labels.len(), 64);
            assert!(ex.key_label < 24);
        }
    }

    #[test]
    fn roundtrip_serialization() {
        let dir = std::env::temp_dir();
        let path = dir.join("optime_ml_test_songs.bin");
        let songs = generate_songs(4, 32, 2);
        save_songs(&path, &songs).unwrap();
        let loaded = load_songs(&path).unwrap();
        assert_eq!(loaded.len(), songs.len());
        assert_eq!(loaded[0].chord_labels, songs[0].chord_labels);
        let _ = std::fs::remove_file(&path);
    }
}

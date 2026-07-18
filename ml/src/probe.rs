//! "Is-music" linear probe on a **frozen** SSL encoder.
//!
//! The self-supervised encoder (pretrained, and optionally fine-tuned) already
//! carries a representation of the note-event distribution. This module trains
//! *only* a small pooled binary head on top of it — the encoder is run once and
//! its pooled features are cached, so the shared representation never moves. The
//! weak labels come from the app's curated `song_names` JSON (see
//! [`crate::harvest::Annotations`]): curated GBA tracks are music, unlisted
//! song-table entries are not. This is the "SSL learns what's music; the
//! annotations only train the decoder" design.

use crate::backend::MlDevice;
use burn::module::{AutodiffModule, Module};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::CompactRecorder;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::path::Path;

use crate::backend::{Back, Inner};
use crate::features::{self, FEATURE_DIM};
use crate::m00_frame::KeyChordModel;
use crate::notes::Song;

/// Cached pooled encoder features + binary labels for the frozen probe.
pub struct MusicSet {
    pub d_model: usize,
    /// `n * d_model`, row-major.
    pub pooled: Vec<f32>,
    /// `n` labels (1 = music, 0 = not-music).
    pub labels: Vec<usize>,
}

impl MusicSet {
    pub fn len(&self) -> usize {
        self.labels.len()
    }
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }
}

/// Run the frozen encoder over every weakly-labeled song and cache its pooled
/// features. Songs with `is_music == None` are skipped.
pub fn build_music_set(
    model: &KeyChordModel<Back>,
    songs: &[Song],
    batch_size: usize,
    device: &MlDevice,
) -> MusicSet {
    let model = model.valid(); // eval mode: no dropout, no grad — a fixed encoder
    let d_model = model.d_model();
    let labeled: Vec<&Song> = songs.iter().filter(|s| s.is_music.is_some()).collect();

    let mut pooled = Vec::with_capacity(labeled.len() * d_model);
    let mut labels = Vec::with_capacity(labeled.len());

    for chunk in labeled.chunks(batch_size.max(1)) {
        let seq = chunk[0].n_frames;
        let mut data = Vec::with_capacity(chunk.len() * seq * FEATURE_DIM);
        for song in chunk {
            let grid = features::extract_song(song);
            data.extend_from_slice(&grid.data);
            labels.push(if song.is_music == Some(true) { 1 } else { 0 });
        }
        let feats = Tensor::<Inner, 3>::from_data(
            TensorData::new(data, [chunk.len(), seq, FEATURE_DIM]),
            device,
        );
        let p = model.pool_encode(feats); // [chunk, d_model]
        let v: Vec<f32> = p.into_data().to_vec().unwrap();
        pooled.extend_from_slice(&v);
    }

    MusicSet {
        d_model,
        pooled,
        labels,
    }
}

#[derive(Config, Debug)]
pub struct ProbeConfig {
    #[config(default = 30)]
    pub epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 1.0e-3)]
    pub lr: f64,
    #[config(default = 7)]
    pub seed: u64,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        ProbeConfig::new()
    }
}

/// Train the frozen is-music probe. `model` provides (and keeps) the frozen
/// encoder; only its `music_head` is updated. The updated model is written to
/// `out_dir` as `probe(.mpk)` + `probe.json`.
pub fn run(
    mut model: KeyChordModel<Back>,
    config: &ProbeConfig,
    model_config: &crate::m00_frame::ModelConfig,
    train: &MusicSet,
    val: &MusicSet,
    out_dir: &Path,
) {
    assert!(!train.is_empty(), "no weakly-labeled songs to probe on");
    let device = MlDevice::default();
    let mut optim = AdamConfig::new().init();
    let ce = CrossEntropyLossConfig::new().init(&device);
    let d = train.d_model;

    let n_pos = train.labels.iter().filter(|&&l| l == 1).count();
    println!(
        "probe: {} train ({} music / {} not) / {} val, d_model {d}",
        train.len(),
        n_pos,
        train.len() - n_pos,
        val.len()
    );

    std::fs::create_dir_all(out_dir).expect("create out dir");
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut order: Vec<usize> = (0..train.len()).collect();

    for epoch in 1..=config.epochs {
        order.shuffle(&mut rng);
        let mut running = 0.0f64;
        let mut nb = 0usize;
        for chunk in order.chunks(config.batch_size) {
            let mut pooled = Vec::with_capacity(chunk.len() * d);
            let mut labels = Vec::with_capacity(chunk.len());
            for &i in chunk {
                pooled.extend_from_slice(&train.pooled[i * d..(i + 1) * d]);
                labels.push(train.labels[i] as i64);
            }
            let feats =
                Tensor::<Back, 2>::from_data(TensorData::new(pooled, [chunk.len(), d]), &device);
            let logits = model.music_from_pooled(feats); // gradients reach only music_head
            let targets =
                Tensor::<Back, 1, Int>::from_data(TensorData::new(labels, [chunk.len()]), &device);
            let loss = ce.forward(logits, targets);

            running += loss.clone().into_scalar().elem::<f32>() as f64;
            nb += 1;
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(config.lr, model, grads);
        }
        let acc = accuracy(&model, val, &device);
        println!(
            "epoch {epoch:>3}/{}  loss {:.4}  |  val acc {:.1}%",
            config.epochs,
            running / nb.max(1) as f64,
            acc * 100.0
        );
    }

    let recorder = CompactRecorder::new();
    model
        .clone()
        .save_file(out_dir.join("probe"), &recorder)
        .expect("save probe weights");
    model_config
        .save(out_dir.join("probe.json"))
        .expect("save probe config");
    println!("saved is-music probe to {}", out_dir.display());
}

/// Overall classification accuracy of the probe on a cached feature set.
pub fn accuracy(model: &KeyChordModel<Back>, set: &MusicSet, device: &MlDevice) -> f64 {
    if set.is_empty() {
        return 0.0;
    }
    let model = model.valid();
    let d = set.d_model;
    let feats =
        Tensor::<Inner, 2>::from_data(TensorData::new(set.pooled.clone(), [set.len(), d]), device);
    let pred = model
        .music_from_pooled(feats)
        .argmax(1)
        .reshape([set.len()]);
    let pred: Vec<i64> = pred.into_data().to_vec().unwrap();
    let correct = pred
        .iter()
        .zip(&set.labels)
        .filter(|(&p, &l)| p as usize == l)
        .count();
    correct as f64 / set.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::m00_frame::ModelConfig;
    use crate::notes::{Instrument, NoteEvent};
    use crate::theory::NO_CHORD;

    fn labeled_song(pitch: u8, is_music: bool, seq: usize) -> Song {
        Song {
            key_label: 0,
            n_frames: seq,
            notes: vec![NoteEvent {
                start_frame: 0,
                end_frame: seq as u32,
                pitch,
                velocity: 1.0,
                instrument: Instrument::Harmony,
                track: 0,
                pan: 0.0,
            }],
            chord_labels: vec![NO_CHORD; seq],
            is_music: Some(is_music),
        }
    }

    #[test]
    fn build_music_set_caches_only_labeled() {
        let device = MlDevice::default();
        let model = ModelConfig::wired().init::<Back>(&device);
        let mut songs = vec![labeled_song(60, true, 16), labeled_song(61, false, 16)];
        // An unlabeled song must be skipped.
        let mut unlabeled = labeled_song(62, true, 16);
        unlabeled.is_music = None;
        songs.push(unlabeled);

        let set = build_music_set(&model, &songs, 8, &device);
        assert_eq!(set.len(), 2);
        assert_eq!(set.labels, vec![1, 0]);
        assert_eq!(set.pooled.len(), 2 * model.d_model());
    }
}

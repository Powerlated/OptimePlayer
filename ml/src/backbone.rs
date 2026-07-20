//! The contract every model generation implements, and the head outputs they all
//! emit.
//!
//! Three backbones — [`crate::m00_frame`] (hand-engineered per-frame features),
//! [`crate::m01_event`] (learned frame tokens, scatter-add pooled), and
//! [`crate::m02_hier`] (learned frame tokens, set-transformer pooled) — differ only
//! in how a window of note events becomes a `[batch, n_frames, d_model]` hidden
//! state. Everything downstream of that (factored root/quality/key heads, the
//! multi-task loss in [`crate::shared`], the training loop, inference) is identical,
//! so it lives once and is generic over this trait.
//!
//! [`ModelOutput`] and [`ArOutput`] live here rather than inside a backbone: every
//! generation produces them, so the newer ones would otherwise import their own
//! output type from generation 00.

use burn::config::Config;
use burn::module::Module;
use burn::prelude::*;
use burn::record::CompactRecorder;
use std::path::{Path, PathBuf};

use crate::notes::Song;

/// Model outputs: raw logits for every head. The chord prediction is **factored**
/// into dedicated root + quality logits (recombined into a joint label at inference).
#[derive(Debug, Clone)]
pub struct ModelOutput<B: Backend> {
    /// `[batch, seq, n_root_classes]` — none + 12 roots.
    pub root_logits: Tensor<B, 3>,
    /// `[batch, seq, n_quality_classes]` — none + 10 qualities.
    pub quality_logits: Tensor<B, 3>,
    /// `[batch, n_key_classes]`
    pub key_logits: Tensor<B, 2>,
}

/// Autoregressive-pretraining head outputs: per-frame next-frame content logits.
pub struct ArOutput<B: Backend> {
    /// `[batch, n_frames, N_PC]`
    pub pc_logits: Tensor<B, 3>,
    /// `[batch, n_frames, N_CHANNELS]`
    pub channel_logits: Tensor<B, 3>,
}

/// A chord/key model generation.
///
/// `Batch` is the CPU-side batch representation (index/label vectors, never
/// tensors), so it is backend-independent — the same `Batch` type serves a model on
/// the autodiff backend and its inner counterpart. That is what lets the training
/// loop build a batch once and use it for both the gradient step and evaluation.
pub trait Backbone<B: Backend>: Module<B> + Clone + Send + Sync + 'static {
    /// CPU-side batch: flattened labels + whatever index arrays the backbone needs.
    type Batch: Send + Sync;
    /// Persisted architecture config, saved next to the weights as `<name>.json`.
    type Cfg: Config;

    /// Artifact subdirectory under `models/`: `"00-frame"` | `"01-event"` | `"02-hier"`.
    const DIR: &'static str;
    /// Human-readable name for log lines.
    const NAME: &'static str;

    /// The config this generation is normally trained with.
    fn default_cfg() -> Self::Cfg;

    /// Instantiate from a config.
    fn init(cfg: &Self::Cfg, device: &B::Device) -> Self;

    /// Tokenize / featurize a slice of songs into one batch. All songs must share
    /// `n_frames`.
    fn build_batch(songs: &[Song]) -> Self::Batch;

    /// `(batch, n_frames)`.
    fn dims(batch: &Self::Batch) -> (usize, usize);

    /// Flattened per-frame joint chord labels (`batch * n_frames`).
    fn chord_labels(batch: &Self::Batch) -> &[usize];

    /// Per-window key labels (`batch`).
    fn key_labels(batch: &Self::Batch) -> &[usize];

    /// Supervised forward: per-frame factored chord logits + pooled key logits.
    fn forward_output(&self, batch: &Self::Batch, device: &B::Device) -> ModelOutput<B>;

    /// Estimated FLOPs to push **one window** through [`forward_output`], under the
    /// matmul-only convention in [`crate::flops`].
    ///
    /// `notes_per_window` is the mean note count of the data, because generation 01
    /// applies its φ projection per *note* — its cost is data-dependent. The other
    /// generations ignore it (00 reads a fixed feature grid; 02's set transformer
    /// runs over a dense `MAX_POLY`-wide grid whether or not the slots are filled).
    ///
    /// The AR pretraining path ([`ArBackbone::ar_forward`]) differs only in swapping
    /// the chord/key heads for the two AR heads — well under 1% of the trunk — so
    /// this stands in for both stages rather than splitting the contract.
    fn flops_per_window(cfg: &Self::Cfg, notes_per_window: usize) -> u64;
}

/// A backbone that supports **autoregressive next-frame** pretraining: predict the
/// next frame's sounding pitch-classes + channels from the causal hidden state.
///
/// Generation 00 deliberately does not implement this — its pretext is masked-frame
/// reconstruction ([`crate::pretrain::masked`]), a genuinely different objective
/// rather than the same loop in disguise.
pub trait ArBackbone<B: Backend>: Backbone<B> {
    /// Causal forward → next-frame content logits.
    fn ar_forward(&self, batch: &Self::Batch, device: &B::Device) -> ArOutput<B>;

    /// Per-frame sounding-content multi-hots `(pc, channel)` for the AR target. The
    /// pretrain loss shifts these by one frame.
    fn ar_targets(batch: &Self::Batch) -> (Vec<f32>, Vec<f32>);
}

/// Load a checkpoint written by [`save`]: weights at `prefix`, config at
/// `prefix.json`.
///
/// Used both to warm-start a fine-tune from a pretrained trunk (the supervised heads
/// are freshly initialised by [`Backbone::init`] and simply overwritten by whatever
/// the record holds) and to load a finished model for inference.
pub fn load<M, B>(prefix: &Path, device: &B::Device) -> M
where
    B: Backend,
    M: Backbone<B>,
{
    let cfg = M::Cfg::load(prefix.with_extension("json"))
        .unwrap_or_else(|e| panic!("load config {}.json: {e}", prefix.display()));
    M::init(&cfg, device)
        .load_file(prefix, &CompactRecorder::new(), device)
        .unwrap_or_else(|e| panic!("load weights {}: {e}", prefix.display()))
}

/// Save weights + config under `dir` as `name(.mpk)` + `name.json`.
pub fn save<M, B>(model: M, cfg: &M::Cfg, dir: &Path, name: &str)
where
    B: Backend,
    M: Backbone<B>,
{
    std::fs::create_dir_all(dir).expect("create out dir");
    model
        .save_file(dir.join(name), &CompactRecorder::new())
        .unwrap_or_else(|e| panic!("save weights {}: {e}", dir.display()));
    cfg.save(dir.join(format!("{name}.json")))
        .unwrap_or_else(|e| panic!("save config {}: {e}", dir.display()));
}

/// Per-epoch checkpoint: `<dir>/<name>-ep007(.mpk)` + `.json`, alongside the final
/// `<name>`. Zero-padded so a lexical sort is a chronological one.
///
/// Takes the model by reference and clones — a driver must keep training after the
/// write, unlike [`save`], which consumes the model at the end of a run.
pub fn save_epoch<M, B>(model: &M, cfg: &M::Cfg, dir: &Path, name: &str, epoch: usize) -> PathBuf
where
    B: Backend,
    M: Backbone<B>,
{
    let stem = format!("{name}-ep{epoch:03}");
    save::<M, B>(model.clone(), cfg, dir, &stem);
    dir.join(stem)
}

/// Artifact directory for a backbone: `<root>/<DIR>` (e.g. `models/02-hier`).
pub fn artifact_dir<M, B>(root: &Path) -> std::path::PathBuf
where
    B: Backend,
    M: Backbone<B>,
{
    root.join(M::DIR)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Inner, MlDevice};
    use crate::m00_frame::FrameModel;
    use crate::m01_event::EventModel;
    use crate::m02_hier::HierModel;
    use crate::notes::{Instrument, NoteEvent, Song};
    use crate::theory::{N_KEY_CLASSES, N_QUALITY_CLASSES, N_ROOT_CLASSES};

    fn small_song(n_frames: usize) -> Song {
        Song {
            key_label: 0,
            n_frames,
            notes: vec![
                NoteEvent {
                    start_frame: 0,
                    end_frame: 4,
                    pitch: 60,
                    velocity: 1.0,
                    instrument: Instrument::Bass,
                    track: 5,
                    pan: 0.0,
                },
                NoteEvent {
                    start_frame: 2,
                    end_frame: 3,
                    pitch: 67,
                    velocity: 0.5,
                    instrument: Instrument::Melody,
                    track: 3,
                    pan: 0.0,
                },
            ],
            chord_labels: vec![0; n_frames],
            is_music: None,
        }
    }

    /// The contract that makes one training loop / one `predict` possible for all
    /// three generations: identical output shapes and label accessors from an
    /// identical `&[Song]` input.
    fn check_shapes<M: Backbone<Inner>>() {
        let device = MlDevice::default();
        let songs = [small_song(8), small_song(8)];
        let batch = M::build_batch(&songs);
        let (b, nf) = M::dims(&batch);
        assert_eq!((b, nf), (2, 8), "{}: dims", M::NAME);
        assert_eq!(M::chord_labels(&batch).len(), b * nf, "{}: chord", M::NAME);
        assert_eq!(M::key_labels(&batch).len(), b, "{}: key", M::NAME);

        let model = M::init(&M::default_cfg(), &device);
        let out = model.forward_output(&batch, &device);
        assert_eq!(
            out.root_logits.dims(),
            [b, nf, N_ROOT_CLASSES],
            "{}",
            M::NAME
        );
        assert_eq!(
            out.quality_logits.dims(),
            [b, nf, N_QUALITY_CLASSES],
            "{}",
            M::NAME
        );
        assert_eq!(out.key_logits.dims(), [b, N_KEY_CLASSES], "{}", M::NAME);
    }

    #[test]
    fn all_backbones_share_one_output_contract() {
        check_shapes::<FrameModel<Inner>>();
        check_shapes::<EventModel<Inner>>();
        check_shapes::<HierModel<Inner>>();
    }
}

//! **Generation 00** â€” the hand-engineered frame-feature backbone.
//!
//! Input is [`crate::features`]'s 57-dim per-frame vector (chroma / bass / melody /
//! onset pitch-class blocks + scalars), projected and run through a bidirectional
//! transformer. The oldest of the three generations and the only one with a
//! masked-frame pretext ([`crate::pretrain::masked`]) and an is-music probe head.

mod batch;
mod model;

pub use batch::FrameBatch;
pub use model::{KeyChordModel, ModelConfig};

use burn::prelude::*;

use crate::backbone::{Backbone, ModelOutput};
use crate::flops;
use crate::notes::Song;

/// Generation-00 model, named for the [`Backbone`] it implements. A type alias, so
/// it is the same type as [`KeyChordModel`] â€” persisted records are unaffected.
pub type FrameModel<B> = KeyChordModel<B>;

impl<B: Backend> Backbone<B> for KeyChordModel<B> {
    type Batch = FrameBatch;
    type Cfg = ModelConfig;

    const DIR: &'static str = "00-frame";
    const NAME: &'static str = "frame";

    fn default_cfg() -> ModelConfig {
        ModelConfig::wired()
    }

    fn init(cfg: &ModelConfig, device: &B::Device) -> Self {
        cfg.init(device)
    }

    fn build_batch(songs: &[Song]) -> FrameBatch {
        FrameBatch::build(songs)
    }

    fn dims(batch: &FrameBatch) -> (usize, usize) {
        (batch.batch, batch.n_frames)
    }

    fn chord_labels(batch: &FrameBatch) -> &[usize] {
        &batch.chord_labels
    }

    fn key_labels(batch: &FrameBatch) -> &[usize] {
        &batch.key_labels
    }

    fn forward_output(&self, batch: &FrameBatch, device: &B::Device) -> ModelOutput<B> {
        self.forward(batch.tensor(device))
    }

    /// Fixed-cost: a `[max_seq_len, n_features]` grid through one projection, the
    /// trunk, and the heads. Independent of note count — the features are already
    /// pooled per frame. The recon + is-music heads are not on this path.
    fn flops_per_window(cfg: &ModelConfig, _notes_per_window: usize) -> u64 {
        let seq = cfg.max_seq_len;
        flops::matmul(seq, cfg.n_features, cfg.d_model)
            + flops::transformer_encoder(cfg.n_layers, seq, cfg.d_model, cfg.d_ff)
            + flops::chord_key_heads(seq, cfg.d_model)
    }
}

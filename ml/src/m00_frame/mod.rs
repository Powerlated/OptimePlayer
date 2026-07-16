//! **Generation 00** — the hand-engineered frame-feature backbone.
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
use crate::notes::Song;

/// Generation-00 model, named for the [`Backbone`] it implements. A type alias, so
/// it is the same type as [`KeyChordModel`] — persisted records are unaffected.
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
}

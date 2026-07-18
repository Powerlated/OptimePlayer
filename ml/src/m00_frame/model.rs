//! **Generation 00** — transformer encoder over the hand-engineered per-frame
//! feature grid ([`crate::features`]), mapping it to a per-frame **chord**
//! distribution and a single pooled **key** distribution for the whole excerpt.
//!
//! Positional information is supplied by a learned position embedding; the encoder
//! is a standard pre-norm multi-head transformer. This generation also carries the
//! masked-frame reconstruction head ([`crate::pretrain::masked`]) and the frozen
//! is-music probe head ([`crate::probe`]), neither of which the learned-token
//! generations have.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

use crate::transformer::{RopeEncoder, RopeEncoderConfig};

use crate::backbone::ModelOutput;
use crate::features::{FEATURE_DIM, PITCH_BLOCK_DIM};
use crate::theory::{N_KEY_CLASSES, N_QUALITY_CLASSES, N_ROOT_CLASSES};

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 128)]
    pub d_model: usize,
    #[config(default = 512)]
    pub d_ff: usize,
    #[config(default = 4)]
    pub n_heads: usize,
    #[config(default = 4)]
    pub n_layers: usize,
    /// Positional-embedding table size. Equal to the training/inference window
    /// (256 frames = 64 beats = 32s at 120bpm); the model never sees more frames
    /// than this, and the harvested data must be windowed to match.
    #[config(default = 256)]
    pub max_seq_len: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
    #[config(default = 57)]
    pub n_features: usize,
    /// Factored chord head: dedicated root logits (none + 12 roots) …
    #[config(default = 13)]
    pub n_root_classes: usize,
    /// … and dedicated quality logits (none + 10 qualities).
    #[config(default = 11)]
    pub n_quality_classes: usize,
    #[config(default = 24)]
    pub n_key_classes: usize,
}

impl ModelConfig {
    /// Config wired to the real feature/label dimensions of this crate.
    pub fn wired() -> Self {
        ModelConfig::new()
            .with_n_features(FEATURE_DIM)
            .with_n_root_classes(N_ROOT_CLASSES)
            .with_n_quality_classes(N_QUALITY_CLASSES)
            .with_n_key_classes(N_KEY_CLASSES)
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> KeyChordModel<B> {
        KeyChordModel {
            input_proj: LinearConfig::new(self.n_features, self.d_model).init(device),
            encoder: RopeEncoderConfig::new(
                self.d_model,
                self.d_ff,
                self.n_heads,
                self.n_layers,
                self.max_seq_len,
            )
            .with_dropout(self.dropout)
            .init(device),
            root_head: LinearConfig::new(self.d_model, self.n_root_classes).init(device),
            quality_head: LinearConfig::new(self.d_model, self.n_quality_classes).init(device),
            key_head: LinearConfig::new(self.d_model, self.n_key_classes).init(device),
            // Self-supervised reconstruction head: predicts the pitch-class blocks
            // of the (masked) input. Unused by the supervised heads; carried so a
            // pretrained encoder + its recon head round-trip through one record.
            recon_head: LinearConfig::new(self.d_model, PITCH_BLOCK_DIM).init(device),
            // Pooled "is-music" head (2 classes). Trained by the frozen linear
            // probe on weak song-name labels; inert for the other objectives.
            music_head: LinearConfig::new(self.d_model, 2).init(device),
            d_model: self.d_model,
        }
    }
}

#[derive(Module, Debug)]
pub struct KeyChordModel<B: Backend> {
    input_proj: Linear<B>,
    encoder: RopeEncoder<B>,
    root_head: Linear<B>,
    quality_head: Linear<B>,
    key_head: Linear<B>,
    recon_head: Linear<B>,
    music_head: Linear<B>,
    d_model: usize,
}

impl<B: Backend> KeyChordModel<B> {
    /// Shared trunk: project features, run the RoPE encoder. `features`:
    /// `[batch, seq, n_features]` → `[batch, seq, d_model]`. Both the supervised
    /// heads and the self-supervised reconstruction head sit on top of this, so
    /// pretraining and fine-tuning share one encoder.
    ///
    /// No additive position term: RoPE injects position inside attention.
    pub fn encode(&self, features: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.input_proj.forward(features); // [batch, seq, d_model]
        self.encoder.forward(x, false) // bidirectional
    }

    /// `features`: `[batch, seq, n_features]`.
    pub fn forward(&self, features: Tensor<B, 3>) -> ModelOutput<B> {
        let [batch, _, _] = features.dims();
        let encoded = self.encode(features); // [batch, seq, d_model]

        // Per-frame factored chord logits: dedicated root + quality heads.
        let root_logits = self.root_head.forward(encoded.clone());
        let quality_logits = self.quality_head.forward(encoded.clone());

        // Pooled (mean over time) key logits.
        let pooled = encoded.mean_dim(1).reshape([batch, self.d_model]); // [batch, d_model]
        let key_logits = self.key_head.forward(pooled);

        ModelOutput {
            root_logits,
            quality_logits,
            key_logits,
        }
    }

    /// Self-supervised head: reconstruct the pitch-class blocks of the input from
    /// the (partially masked) feature grid. `features`: `[batch, seq, n_features]`
    /// → `[batch, seq, PITCH_BLOCK_DIM]`. Used only during masked-frame pretraining.
    pub fn forward_ssl(&self, features: Tensor<B, 3>) -> Tensor<B, 3> {
        let encoded = self.encode(features);
        self.recon_head.forward(encoded)
    }

    /// Time-pooled encoder features `[batch, d_model]` (mean over the sequence) —
    /// the fixed representation the frozen is-music probe reads.
    pub fn pool_encode(&self, features: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, _, _] = features.dims();
        self.encode(features)
            .mean_dim(1)
            .reshape([batch, self.d_model])
    }

    /// Apply the is-music head to precomputed pooled features → `[batch, 2]`
    /// (not-music = 0, music = 1). The frozen linear probe pools the encoder once
    /// (see [`Self::pool_encode`]) and trains only this head over the cached
    /// features, so the shared representation never moves.
    pub fn music_from_pooled(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        self.music_head.forward(pooled)
    }

    /// Pooled "is-music" logits `[batch, 2]` straight from features (for inference).
    pub fn forward_music(&self, features: Tensor<B, 3>) -> Tensor<B, 2> {
        let pooled = self.pool_encode(features);
        self.music_head.forward(pooled)
    }

    /// The model dimension (pooled-feature width for the is-music probe).
    pub fn d_model(&self) -> usize {
        self.d_model
    }
}

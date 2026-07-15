//! Transformer encoder that maps a per-frame feature grid to two outputs:
//!   * a per-frame **chord** distribution (121 classes incl. no-chord), and
//!   * a single pooled **key** distribution (24 classes) for the whole excerpt.
//!
//! Built on the Burn framework. Positional information is supplied by a learned
//! position embedding; the encoder is a standard pre-norm multi-head transformer.

use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;

use crate::features::{FEATURE_DIM, PITCH_BLOCK_DIM};
use crate::theory::{N_CHORD_CLASSES, N_KEY_CLASSES};

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
    #[config(default = 512)]
    pub max_seq_len: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
    #[config(default = 57)]
    pub n_features: usize,
    #[config(default = 121)]
    pub n_chord_classes: usize,
    #[config(default = 24)]
    pub n_key_classes: usize,
}

impl ModelConfig {
    /// Config wired to the real feature/label dimensions of this crate.
    pub fn wired() -> Self {
        ModelConfig::new()
            .with_n_features(FEATURE_DIM)
            .with_n_chord_classes(N_CHORD_CLASSES)
            .with_n_key_classes(N_KEY_CLASSES)
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> KeyChordModel<B> {
        KeyChordModel {
            input_proj: LinearConfig::new(self.n_features, self.d_model).init(device),
            pos_emb: EmbeddingConfig::new(self.max_seq_len, self.d_model).init(device),
            encoder: TransformerEncoderConfig::new(
                self.d_model,
                self.d_ff,
                self.n_heads,
                self.n_layers,
            )
            .with_norm_first(true)
            .with_dropout(self.dropout)
            .init(device),
            chord_head: LinearConfig::new(self.d_model, self.n_chord_classes).init(device),
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

/// Model outputs: raw logits for both heads.
#[derive(Debug, Clone)]
pub struct ModelOutput<B: Backend> {
    /// `[batch, seq, n_chord_classes]`
    pub chord_logits: Tensor<B, 3>,
    /// `[batch, n_key_classes]`
    pub key_logits: Tensor<B, 2>,
}

#[derive(Module, Debug)]
pub struct KeyChordModel<B: Backend> {
    input_proj: Linear<B>,
    pos_emb: Embedding<B>,
    encoder: TransformerEncoder<B>,
    chord_head: Linear<B>,
    key_head: Linear<B>,
    recon_head: Linear<B>,
    music_head: Linear<B>,
    d_model: usize,
}

impl<B: Backend> KeyChordModel<B> {
    /// Shared trunk: project features, add the learned positional embedding, run
    /// the transformer encoder. `features`: `[batch, seq, n_features]` →
    /// `[batch, seq, d_model]`. Both the supervised heads and the self-supervised
    /// reconstruction head sit on top of this, so pretraining and fine-tuning
    /// share one encoder.
    pub fn encode(&self, features: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = features.device();
        let [_, seq, _] = features.dims();

        let x = self.input_proj.forward(features); // [batch, seq, d_model]
        let positions = Tensor::<B, 1, Int>::arange(0..seq as i64, &device).reshape([1, seq]);
        let pos = self.pos_emb.forward(positions); // [1, seq, d_model]
        let x = x + pos;

        self.encoder.forward(TransformerEncoderInput::new(x)) // [batch, seq, d_model]
    }

    /// `features`: `[batch, seq, n_features]`.
    pub fn forward(&self, features: Tensor<B, 3>) -> ModelOutput<B> {
        let [batch, _, _] = features.dims();
        let encoded = self.encode(features); // [batch, seq, d_model]

        // Per-frame chord logits.
        let chord_logits = self.chord_head.forward(encoded.clone());

        // Pooled (mean over time) key logits.
        let pooled = encoded.mean_dim(1).reshape([batch, self.d_model]); // [batch, d_model]
        let key_logits = self.key_head.forward(pooled);

        ModelOutput {
            chord_logits,
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

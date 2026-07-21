use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::gelu;
use burn::tensor::IndexingUpdateOp;

use crate::backbone::{ArBackbone, ArOutput, Backbone, ModelOutput};
use crate::flops;
use crate::kda::{KimiEncoder, KimiEncoderConfig};
use crate::m01_event::EventBatchData;
use crate::notes::Song;
use crate::theory::{N_KEY_CLASSES, N_QUALITY_CLASSES, N_ROOT_CLASSES};
use crate::tokenize::{DUR_BUCKETS, N_CHANNELS, N_PC, N_ROLES, PAN_BUCKETS, VEL_BUCKETS};

#[derive(Config, Debug)]
pub struct KdaModelConfig {
    #[config(default = 128)]
    pub d_model: usize,
    #[config(default = 512)]
    pub d_ff: usize,
    #[config(default = 4)]
    pub n_heads: usize,
    /// Total trunk depth (KDA + full-attention layers combined). Every 4th layer
    /// (indices 3, 7, …) is full attention; the rest KDA — see
    /// [`crate::kda::KimiEncoderConfig`].
    #[config(default = 4)]
    pub n_layers: usize,
    /// KDA chunkwise-scan chunk length.
    #[config(default = 64)]
    pub chunk_size: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
    /// Window length in frames (256 = 64 beats = 32s at 120bpm). Must match the
    /// dataset's windowing.
    #[config(default = 256)]
    pub n_frames: usize,
}

impl KdaModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> KdaKeyChordModel<B> {
        let d = self.d_model;
        let emb = |n: usize| EmbeddingConfig::new(n, d).init(device);
        KdaKeyChordModel {
            emb_pitch: emb(128),
            emb_pc: emb(N_PC),
            emb_channel: emb(N_CHANNELS),
            emb_vel: emb(VEL_BUCKETS),
            emb_pan: emb(PAN_BUCKETS),
            emb_dur: emb(DUR_BUCKETS),
            emb_role: emb(N_ROLES),
            phi: LinearConfig::new(d, d).init(device),
            encoder: KimiEncoderConfig::new(d, self.d_ff, self.n_heads, self.n_layers)
                .with_chunk_size(self.chunk_size)
                .with_dropout(self.dropout)
                .init(device),
            root_head: LinearConfig::new(d, N_ROOT_CLASSES).init(device),
            quality_head: LinearConfig::new(d, N_QUALITY_CLASSES).init(device),
            key_head: LinearConfig::new(d, N_KEY_CLASSES).init(device),
            ar_pc_head: LinearConfig::new(d, N_PC).init(device),
            ar_ch_head: LinearConfig::new(d, N_CHANNELS).init(device),
            d_model: d,
            n_frames: self.n_frames,
        }
    }
}

/// Same seven-field-embedding + φ + scatter-add frame-token front-end as
/// [`crate::m01_event::EventKeyChordModel`] (deliberately not reimplemented — see
/// module docs), with [`KimiEncoder`] (Kimi Linear / KDA hybrid trunk) in place of
/// [`crate::transformer::RopeEncoder`].
#[derive(Module, Debug)]
pub struct KdaKeyChordModel<B: Backend> {
    emb_pitch: Embedding<B>,
    emb_pc: Embedding<B>,
    emb_channel: Embedding<B>,
    emb_vel: Embedding<B>,
    emb_pan: Embedding<B>,
    emb_dur: Embedding<B>,
    emb_role: Embedding<B>,
    phi: Linear<B>,
    encoder: KimiEncoder<B>,
    root_head: Linear<B>,
    quality_head: Linear<B>,
    key_head: Linear<B>,
    ar_pc_head: Linear<B>,
    ar_ch_head: Linear<B>,
    d_model: usize,
    n_frames: usize,
}

impl<B: Backend> KdaKeyChordModel<B> {
    /// Embed the onset tokens and pool them into per-frame vectors `[batch, nf, d]`.
    /// Identical to `EventKeyChordModel::frame_tokens` — same fields, same scatter-add.
    fn frame_tokens(&self, data: &EventBatchData, device: &B::Device) -> Tensor<B, 3> {
        let d = self.d_model;
        let (b, nf) = (data.batch, data.n_frames);
        let base = Tensor::<B, 2>::zeros([b * nf, d], device);
        if data.n_total == 0 {
            return base.reshape([b, nf, d]);
        }

        let field = |emb: &Embedding<B>, idx: &[i64]| -> Tensor<B, 2> {
            let t =
                Tensor::<B, 1, Int>::from_data(TensorData::new(idx.to_vec(), [idx.len()]), device)
                    .reshape([1, idx.len()]);
            emb.forward(t).reshape([idx.len(), d])
        };
        let per_note = field(&self.emb_pitch, &data.pitch)
            + field(&self.emb_pc, &data.pc)
            + field(&self.emb_channel, &data.channel)
            + field(&self.emb_vel, &data.vel)
            + field(&self.emb_pan, &data.pan)
            + field(&self.emb_dur, &data.dur)
            + field(&self.emb_role, &data.role);
        let per_note = gelu(self.phi.forward(per_note)); // [n_total, d]

        let rows = Tensor::<B, 1, Int>::from_data(
            TensorData::new(data.target_row.clone(), [data.n_total]),
            device,
        );
        let idx = rows.reshape([data.n_total, 1]).repeat_dim(1, d);
        base.scatter(0, idx, per_note, IndexingUpdateOp::Add)
            .reshape([b, nf, d])
    }

    /// Main trunk over the frame tokens. `causal` affects only the encoder's
    /// full-attention layers — KDA is causal by construction regardless (see
    /// [`crate::kda`] module docs), so a bidirectional supervised pass here is only
    /// bidirectional at those layers. Documented deviation from the RoPE-trunk
    /// generations, which are uniformly causal-or-not throughout.
    fn trunk(&self, frame_tokens: Tensor<B, 3>, causal: bool) -> Tensor<B, 3> {
        self.encoder.forward(frame_tokens, causal)
    }
}

impl<B: Backend> Backbone<B> for KdaKeyChordModel<B> {
    type Batch = EventBatchData;
    type Cfg = KdaModelConfig;

    const DIR: &'static str = "03-kda";
    const NAME: &'static str = "kda";

    fn default_cfg() -> KdaModelConfig {
        KdaModelConfig::new()
    }

    fn init(cfg: &KdaModelConfig, device: &B::Device) -> Self {
        cfg.init(device)
    }

    fn build_batch(songs: &[Song]) -> EventBatchData {
        EventBatchData::build(songs)
    }

    fn dims(data: &EventBatchData) -> (usize, usize) {
        (data.batch, data.n_frames)
    }

    fn chord_labels(data: &EventBatchData) -> &[usize] {
        &data.chord_labels
    }

    fn key_labels(data: &EventBatchData) -> &[usize] {
        &data.key_labels
    }

    /// φ runs per note (same cost as m01's front-end); the trunk is
    /// [`flops::kda_layer`] per layer instead of [`flops::transformer_layer`].
    fn flops_per_window(cfg: &KdaModelConfig, notes_per_window: usize) -> u64 {
        let seq = cfg.n_frames;
        let trunk: u64 = (0..cfg.n_layers)
            .map(|i| {
                if (i + 1).is_multiple_of(4) {
                    flops::transformer_layer(seq, cfg.d_model, cfg.d_ff)
                } else {
                    flops::kda_layer(seq, cfg.d_model, cfg.d_ff, cfg.chunk_size)
                }
            })
            .sum();
        flops::matmul(notes_per_window, cfg.d_model, cfg.d_model)
            + trunk
            + flops::chord_key_heads(seq, cfg.d_model)
    }

    /// Supervised forward (bidirectional at the attention layers) → shared
    /// [`ModelOutput`].
    fn forward_output(&self, data: &EventBatchData, device: &B::Device) -> ModelOutput<B> {
        let hidden = self.trunk(self.frame_tokens(data, device), false);
        let root_logits = self.root_head.forward(hidden.clone());
        let quality_logits = self.quality_head.forward(hidden.clone());
        let pooled = hidden.mean_dim(1).reshape([data.batch, self.d_model]);
        let key_logits = self.key_head.forward(pooled);
        ModelOutput {
            root_logits,
            quality_logits,
            key_logits,
        }
    }
}

impl<B: Backend> ArBackbone<B> for KdaKeyChordModel<B> {
    /// AR pretraining forward (causal at the attention layers; KDA is always
    /// causal): per-frame next-frame content logits.
    fn ar_forward(&self, data: &EventBatchData, device: &B::Device) -> ArOutput<B> {
        let hidden = self.trunk(self.frame_tokens(data, device), true);
        ArOutput {
            pc_logits: self.ar_pc_head.forward(hidden.clone()),
            channel_logits: self.ar_ch_head.forward(hidden),
        }
    }

    fn ar_targets(data: &EventBatchData) -> (Vec<f32>, Vec<f32>) {
        crate::tokenize::ar_targets(&data.examples)
    }
}

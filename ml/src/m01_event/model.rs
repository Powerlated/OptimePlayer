//! **Generation 01** — frame-token transformer: one token per frame (a scatter-add
//! pool of the notes onsetting in that frame), a shared trunk used **causally** for
//! autoregressive next-frame pretraining and **bidirectionally** for the supervised
//! chord/key fine-tune, and the same factored root/quality/key heads as every other
//! generation (emits the shared [`ModelOutput`]).

use burn::nn::attention::generate_autoregressive_mask;
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::gelu;
use burn::tensor::IndexingUpdateOp;

use crate::backbone::{ArBackbone, ArOutput, Backbone, ModelOutput};
use crate::flops;
use crate::m01_event::batch::EventBatchData;
use crate::notes::Song;
use crate::theory::{N_KEY_CLASSES, N_QUALITY_CLASSES, N_ROOT_CLASSES};
use crate::tokenize::{DUR_BUCKETS, N_CHANNELS, N_PC, N_ROLES, PAN_BUCKETS, VEL_BUCKETS};

#[derive(Config, Debug)]
pub struct EventModelConfig {
    #[config(default = 128)]
    pub d_model: usize,
    #[config(default = 512)]
    pub d_ff: usize,
    #[config(default = 4)]
    pub n_heads: usize,
    #[config(default = 4)]
    pub n_layers: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
    /// Window length in frames (256 = 64 beats = 32s at 120bpm). Must match the
    /// dataset's windowing.
    #[config(default = 256)]
    pub n_frames: usize,
}

impl EventModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EventKeyChordModel<B> {
        let d = self.d_model;
        let emb = |n: usize| EmbeddingConfig::new(n, d).init(device);
        EventKeyChordModel {
            emb_pitch: emb(128),
            emb_pc: emb(N_PC),
            emb_channel: emb(N_CHANNELS),
            emb_vel: emb(VEL_BUCKETS),
            emb_pan: emb(PAN_BUCKETS),
            emb_dur: emb(DUR_BUCKETS),
            emb_role: emb(N_ROLES),
            phi: LinearConfig::new(d, d).init(device),
            frame_pos: EmbeddingConfig::new(self.n_frames, d).init(device),
            encoder: TransformerEncoderConfig::new(d, self.d_ff, self.n_heads, self.n_layers)
                .with_norm_first(true)
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

#[derive(Module, Debug)]
pub struct EventKeyChordModel<B: Backend> {
    emb_pitch: Embedding<B>,
    emb_pc: Embedding<B>,
    emb_channel: Embedding<B>,
    emb_vel: Embedding<B>,
    emb_pan: Embedding<B>,
    emb_dur: Embedding<B>,
    emb_role: Embedding<B>,
    phi: Linear<B>,
    frame_pos: Embedding<B>,
    encoder: TransformerEncoder<B>,
    root_head: Linear<B>,
    quality_head: Linear<B>,
    key_head: Linear<B>,
    ar_pc_head: Linear<B>,
    ar_ch_head: Linear<B>,
    d_model: usize,
    n_frames: usize,
}

impl<B: Backend> EventKeyChordModel<B> {
    /// Embed the onset tokens and pool them into per-frame vectors `[batch, nf, d]`.
    fn frame_tokens(&self, data: &EventBatchData, device: &B::Device) -> Tensor<B, 3> {
        let d = self.d_model;
        let (b, nf) = (data.batch, data.n_frames);
        let base = Tensor::<B, 2>::zeros([b * nf, d], device);
        if data.n_total == 0 {
            return base.reshape([b, nf, d]);
        }

        // Per-note embedding = sum of field embeddings → nonlinear φ.
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

        // Scatter-add each note into its frame slot (b*nf + onset). Duplicate rows
        // accumulate (Add), pooling all notes onsetting in the same frame.
        let rows = Tensor::<B, 1, Int>::from_data(
            TensorData::new(data.target_row.clone(), [data.n_total]),
            device,
        );
        let idx = rows.reshape([data.n_total, 1]).repeat_dim(1, d);
        base.scatter(0, idx, per_note, IndexingUpdateOp::Add)
            .reshape([b, nf, d])
    }

    /// Add frame positions and run the trunk (causal for AR pretraining,
    /// bidirectional for the supervised heads).
    fn trunk(&self, frame_tokens: Tensor<B, 3>, causal: bool, device: &B::Device) -> Tensor<B, 3> {
        let [b, nf, _] = frame_tokens.dims();
        let positions = Tensor::<B, 1, Int>::arange(0..nf as i64, device).reshape([1, nf]);
        let x = frame_tokens + self.frame_pos.forward(positions);
        let mut input = TransformerEncoderInput::new(x);
        if causal {
            input = input.mask_attn(generate_autoregressive_mask::<B>(b, nf, device));
        }
        self.encoder.forward(input)
    }
}

impl<B: Backend> Backbone<B> for EventKeyChordModel<B> {
    type Batch = EventBatchData;
    type Cfg = EventModelConfig;

    const DIR: &'static str = "01-event";
    const NAME: &'static str = "event";

    fn default_cfg() -> EventModelConfig {
        EventModelConfig::new()
    }

    fn init(cfg: &EventModelConfig, device: &B::Device) -> Self {
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

    /// φ runs **per note**, so this generation's cost genuinely depends on how many
    /// notes the window holds; the trunk and heads are fixed.
    fn flops_per_window(cfg: &EventModelConfig, notes_per_window: usize) -> u64 {
        let seq = cfg.n_frames;
        flops::matmul(notes_per_window, cfg.d_model, cfg.d_model)
            + flops::transformer_encoder(cfg.n_layers, seq, cfg.d_model, cfg.d_ff)
            + flops::chord_key_heads(seq, cfg.d_model)
    }

    /// Supervised forward (bidirectional): per-frame factored chord logits + pooled
    /// key logits, as the shared [`ModelOutput`].
    fn forward_output(&self, data: &EventBatchData, device: &B::Device) -> ModelOutput<B> {
        let hidden = self.trunk(self.frame_tokens(data, device), false, device);
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

impl<B: Backend> ArBackbone<B> for EventKeyChordModel<B> {
    /// AR pretraining forward (causal): per-frame next-frame content logits.
    fn ar_forward(&self, data: &EventBatchData, device: &B::Device) -> ArOutput<B> {
        let hidden = self.trunk(self.frame_tokens(data, device), true, device);
        ArOutput {
            pc_logits: self.ar_pc_head.forward(hidden.clone()),
            channel_logits: self.ar_ch_head.forward(hidden),
        }
    }

    fn ar_targets(data: &EventBatchData) -> (Vec<f32>, Vec<f32>) {
        crate::tokenize::ar_targets(&data.examples)
    }
}

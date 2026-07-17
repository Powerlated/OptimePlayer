//! **Generation 02** — a **set transformer** embeds each frame's variable-length set of *sounding*
//! notes into that frame's token (learned CLS-attention pooling, set capped at [`MAX_POLY`]), then
//! the main RoPE trunk runs over the frame tokens — causal for AR pretraining, bidirectional for
//! the fine-tune. The set-transformer variant of [`crate::m01_event`]'s scatter-add sum pool. Emits
//! the shared [`ModelOutput`]/[`ArOutput`], reusing `shared.rs` + the generic `dp_step`.

use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{Bool, IndexingUpdateOp};

use crate::backbone::{ArBackbone, ArOutput, Backbone, ModelOutput};
use crate::flops;
use crate::m02_hier::batch::{HierBatchData, MAX_POLY};
use crate::notes::Song;
use crate::theory::{N_KEY_CLASSES, N_QUALITY_CLASSES, N_ROOT_CLASSES};
use crate::tokenize::{DUR_BUCKETS, N_CHANNELS, N_PC, N_ROLES, PAN_BUCKETS, VEL_BUCKETS};
use crate::transformer::{RopeEncoder, RopeEncoderConfig};

#[derive(Config, Debug)]
pub struct HierModelConfig {
    #[config(default = 128)]
    pub d_model: usize,
    #[config(default = 512)]
    pub d_ff: usize,
    /// Set-attention FFN width. Smaller than main `d_ff`: the sub-encoder FFN over
    /// `[batch*n_frames, MAX_POLY, sub_d_ff]` is the dominant retained activation, so shrinking it
    /// cuts peak memory, and a ≤`MAX_POLY`-note pool needs little capacity.
    #[config(default = 256)]
    pub sub_d_ff: usize,
    #[config(default = 4)]
    pub n_heads: usize,
    /// Main (frame-level) transformer depth.
    #[config(default = 4)]
    pub n_layers: usize,
    /// Set-transformer depth. One layer suffices for a light CLS-attention over ≤[`MAX_POLY`]
    /// notes; more costs a full extra pass over the grid for little gain.
    #[config(default = 1)]
    pub n_sub_layers: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
    /// Window length in frames (256 = 64 beats = 32s at 120bpm). Must match the
    /// dataset's windowing.
    #[config(default = 256)]
    pub n_frames: usize,
}

impl HierModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> HierEventModel<B> {
        let d = self.d_model;
        let emb = |n: usize| EmbeddingConfig::new(n, d).init(device);
        // Position-free: a frame's notes are an unordered set, so slot index is meaningless and
        // RoPE would invent an order. Only the frame trunk is a sequence.
        let set_enc = |layers: usize, d_ff: usize| {
            TransformerEncoderConfig::new(d, d_ff, self.n_heads, layers)
                .with_norm_first(true)
                .with_dropout(self.dropout)
                .init(device)
        };
        HierEventModel {
            emb_pitch: emb(128),
            emb_pc: emb(N_PC),
            emb_channel: emb(N_CHANNELS),
            emb_vel: emb(VEL_BUCKETS),
            emb_pan: emb(PAN_BUCKETS),
            emb_dur: emb(DUR_BUCKETS),
            emb_role: emb(N_ROLES),
            emb_onset: emb(2),
            cls: emb(1),
            sub_encoder: set_enc(self.n_sub_layers, self.sub_d_ff),
            encoder: RopeEncoderConfig::new(
                d,
                self.d_ff,
                self.n_heads,
                self.n_layers,
                self.n_frames,
            )
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
pub struct HierEventModel<B: Backend> {
    emb_pitch: Embedding<B>,
    emb_pc: Embedding<B>,
    emb_channel: Embedding<B>,
    emb_vel: Embedding<B>,
    emb_pan: Embedding<B>,
    emb_dur: Embedding<B>,
    emb_role: Embedding<B>,
    emb_onset: Embedding<B>,
    cls: Embedding<B>,
    sub_encoder: TransformerEncoder<B>,
    encoder: RopeEncoder<B>,
    root_head: Linear<B>,
    quality_head: Linear<B>,
    key_head: Linear<B>,
    ar_pc_head: Linear<B>,
    ar_ch_head: Linear<B>,
    d_model: usize,
    n_frames: usize,
}

impl<B: Backend> HierEventModel<B> {
    /// Set-transformer pool → `[batch, n_frames, d]`: embed the flat note list, scatter-add each
    /// into its `(frame, pos)` slot to build the set grid once, prepend a CLS token per frame, run
    /// the padding-masked sub-encoder, take CLS as the frame token.
    fn frame_tokens(&self, data: &HierBatchData, device: &B::Device) -> Tensor<B, 3> {
        let d = self.d_model;
        let (b, nf) = (data.batch, data.n_frames);
        let bnf = b * nf;

        // Field-embedding sum runs over the SHORT note list `[n_snd, d]`, not the padded grid, so
        // the per-field gathers and `+`-chain temporaries are ~MAX_POLY× smaller.
        let grid = if data.n_snd == 0 {
            Tensor::<B, 2>::zeros([bnf * MAX_POLY, d], device).reshape([bnf, MAX_POLY, d])
        } else {
            let field = |emb: &Embedding<B>, idx: &[i64]| -> Tensor<B, 2> {
                let t = Tensor::<B, 1, Int>::from_data(
                    TensorData::new(idx.to_vec(), [idx.len()]),
                    device,
                )
                .reshape([1, idx.len()]);
                emb.forward(t).reshape([idx.len(), d])
            };
            let per_note = field(&self.emb_pitch, &data.pitch)
                + field(&self.emb_pc, &data.pc)
                + field(&self.emb_channel, &data.channel)
                + field(&self.emb_vel, &data.vel)
                + field(&self.emb_pan, &data.pan)
                + field(&self.emb_dur, &data.dur)
                + field(&self.emb_role, &data.role)
                + field(&self.emb_onset, &data.onset);

            // Scatter-add each note into its unique slot (Add on zeros == assign).
            let base = Tensor::<B, 2>::zeros([bnf * MAX_POLY, d], device);
            let rows = Tensor::<B, 1, Int>::from_data(
                TensorData::new(data.slot_row.clone(), [data.n_snd]),
                device,
            );
            let idx = rows.reshape([data.n_snd, 1]).repeat_dim(1, d);
            base.scatter(0, idx, per_note, IndexingUpdateOp::Add)
                .reshape([bnf, MAX_POLY, d])
        };

        // Prepend CLS (index 0) to each frame's set: [bnf, MAX_POLY+1, d].
        let cls = self
            .cls
            .forward(Tensor::<B, 1, Int>::zeros([bnf], device).reshape([bnf, 1]));
        let x = Tensor::cat(vec![cls, grid], 1);

        // Precomputed pad mask (CLS col 0 always valid; per-frame note cols filled in build).
        let mask = Tensor::<B, 2, Bool>::from_data(
            TensorData::new(data.pad_mask.clone(), [bnf, MAX_POLY + 1]),
            device,
        );

        let encoded = self
            .sub_encoder
            .forward(TransformerEncoderInput::new(x).mask_pad(mask));
        // CLS output = the frame token.
        encoded.slice([0..bnf, 0..1, 0..d]).reshape([b, nf, d])
    }

    /// Main trunk — causal for AR pretraining, bidirectional for the supervised heads. Position is
    /// RoPE inside attention, so no additive position embedding.
    fn trunk(&self, frame_tokens: Tensor<B, 3>, causal: bool) -> Tensor<B, 3> {
        self.encoder.forward(frame_tokens, causal)
    }
}

impl<B: Backend> Backbone<B> for HierEventModel<B> {
    type Batch = HierBatchData;
    type Cfg = HierModelConfig;

    const DIR: &'static str = "02-hier";
    const NAME: &'static str = "hier";

    fn default_cfg() -> HierModelConfig {
        HierModelConfig::new()
    }

    fn init(cfg: &HierModelConfig, device: &B::Device) -> Self {
        cfg.init(device)
    }

    fn build_batch(songs: &[Song]) -> HierBatchData {
        HierBatchData::build(songs)
    }

    fn dims(data: &HierBatchData) -> (usize, usize) {
        (data.batch, data.n_frames)
    }

    fn chord_labels(data: &HierBatchData) -> &[usize] {
        &data.chord_labels
    }

    fn key_labels(data: &HierBatchData) -> &[usize] {
        &data.key_labels
    }

    /// Two stacked transformers: the set encoder runs once per frame over the dense
    /// `MAX_POLY + 1` slot grid (CLS + slots) — a fixed cost, since the grid is
    /// materialised padded regardless of real polyphony — then the trunk over the
    /// frame tokens.
    fn flops_per_window(cfg: &HierModelConfig, _notes_per_window: usize) -> u64 {
        let seq = cfg.n_frames;
        let set_encoder = seq as u64
            * flops::transformer_encoder(cfg.n_sub_layers, MAX_POLY + 1, cfg.d_model, cfg.sub_d_ff);
        set_encoder
            + flops::transformer_encoder(cfg.n_layers, seq, cfg.d_model, cfg.d_ff)
            + flops::chord_key_heads(seq, cfg.d_model)
    }

    /// Supervised forward (bidirectional) → shared [`ModelOutput`].
    fn forward_output(&self, data: &HierBatchData, device: &B::Device) -> ModelOutput<B> {
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

impl<B: Backend> ArBackbone<B> for HierEventModel<B> {
    /// AR pretraining forward (causal) → next-frame content logits.
    fn ar_forward(&self, data: &HierBatchData, device: &B::Device) -> ArOutput<B> {
        let hidden = self.trunk(self.frame_tokens(data, device), true);
        ArOutput {
            pc_logits: self.ar_pc_head.forward(hidden.clone()),
            channel_logits: self.ar_ch_head.forward(hidden),
        }
    }

    fn ar_targets(data: &HierBatchData) -> (Vec<f32>, Vec<f32>) {
        crate::tokenize::ar_targets(&data.examples)
    }
}

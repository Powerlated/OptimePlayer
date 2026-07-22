use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::gelu;
use burn::tensor::IndexingUpdateOp;

use crate::backbone::{ArBackbone, ArOutput, Backbone, ModelOutput};
use crate::flops;
use crate::kda::{Docs, KimiEncoder, KimiEncoderConfig};
use crate::m01_event::{EventBatchData, SLOT_EOS, SLOT_FRAME};
use crate::notes::Song;
use crate::theory::{
    chord_label_to_root_quality, N_KEY_CLASSES, N_QUALITY_CLASSES, N_ROOT_CLASSES,
};
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
    /// [`crate::kda::KimiEncoderConfig`]. Default 6 (5 KDA + 1 attention).
    /// FLOP parity with m02 is handled at **run launch** (epoch count chosen so
    /// total training FLOPs match), not per window — at 2048-slot packed windows a
    /// per-window match is impossible.
    #[config(default = 6)]
    pub n_layers: usize,
    /// KDA chunkwise-scan chunk length.
    #[config(default = 64)]
    pub chunk_size: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
    /// **Nominal / maximum** sequence length in slots. m03 has no positional table
    /// (KDA carries position; the attention layers are NoPE), so batches run at
    /// whatever padded length they arrive at — this number sizes the *pretraining
    /// pack length* and the harvest truncation cap, and feeds the FLOP estimate.
    /// 2048 slots ≈ 512 beats ≈ 4.3 min at 120bpm.
    #[config(default = 2048)]
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
            emb_prev_root: emb(N_ROOT_CLASSES + 1),
            emb_prev_quality: emb(N_QUALITY_CLASSES + 1),
            emb_segment: emb(2),
            emb_eos: emb(1),
            encoder: KimiEncoderConfig::new(d, self.d_ff, self.n_heads, self.n_layers)
                .with_chunk_size(self.chunk_size)
                .with_dropout(self.dropout)
                .init(device),
            root_head: LinearConfig::new(d, N_ROOT_CLASSES).init(device),
            quality_head: LinearConfig::new(d, N_QUALITY_CLASSES).init(device),
            key_head: LinearConfig::new(d, N_KEY_CLASSES).init(device),
            ar_pc_head: LinearConfig::new(d, N_PC).init(device),
            ar_ch_head: LinearConfig::new(d, N_CHANNELS).init(device),
            ar_eos_head: LinearConfig::new(d, 1).init(device),
            d_model: d,
            n_frames: self.n_frames,
        }
    }
}

/// Same seven-field-embedding + φ + scatter-add frame-token front-end as
/// [`crate::m01_event::EventKeyChordModel`] (deliberately not reimplemented — see
/// module docs), with [`KimiEncoder`] (Kimi Linear / KDA hybrid trunk) in place of
/// [`crate::transformer::RopeEncoder`].
///
/// **Generative (read-then-generate) formulation.** The supervised pass is one fully
/// causal sequence of `2·n_frames` slots: the song's frame tokens, then one label
/// slot per frame. Label slot `t`'s input is `frame_token[t] + emb(prev root) +
/// emb(prev quality) + segment`, so every chord prediction conditions on the
/// **entire song** (through the causal KDA state — no bidirectional attention
/// needed) plus all previously decoded chords (a learned chord-LM prior).
/// Training teacher-forces the previous labels ([`Backbone::forward_output`], one
/// pass, so train-loop val metrics are teacher-forced); real inference decodes
/// greedily, one slot at a time ([`Backbone::infer_output`], `n_frames` re-forwards
/// — offline-only cost by design).
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
    /// Previous-label conditioning for the generative label slots. Index
    /// `N_ROOT_CLASSES` / `N_QUALITY_CLASSES` (one past the label spaces) is BOS.
    emb_prev_root: Embedding<B>,
    emb_prev_quality: Embedding<B>,
    /// Two learned vectors marking song-half vs. label-half slots (the attention
    /// layers are NoPE, so nothing else distinguishes the halves).
    emb_segment: Embedding<B>,
    /// The **EOS token**: added at each document's EOS slot (packed sequences put
    /// one after every song; the generative layout after the song's last frame).
    emb_eos: Embedding<B>,
    encoder: KimiEncoder<B>,
    root_head: Linear<B>,
    quality_head: Linear<B>,
    key_head: Linear<B>,
    ar_pc_head: Linear<B>,
    ar_ch_head: Linear<B>,
    /// "Next slot is EOS" logit for the packed AR pretext.
    ar_eos_head: Linear<B>,
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

    /// One learned segment vector (`0` = song half, `1` = label half), shaped for
    /// broadcast over `[b, seq, d]`.
    fn segment(&self, which: i64, device: &B::Device) -> Tensor<B, 3> {
        let idx = Tensor::<B, 1, Int>::from_data(TensorData::new(vec![which], [1]), device)
            .reshape([1, 1]);
        self.emb_segment.forward(idx).reshape([1, 1, self.d_model])
    }

    /// The song half's input: frame tokens + the EOS embedding at EOS slots +
    /// the song segment vector. Pad slots stay zero (+ segment).
    fn song_tokens(
        &self,
        data: &EventBatchData,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let (b, nf) = (data.batch, data.n_frames);
        let ft = self.frame_tokens(data, device);
        let eos_mask: Vec<f32> = data
            .slot_kind
            .iter()
            .map(|&k| if k == SLOT_EOS { 1.0 } else { 0.0 })
            .collect();
        let eos_mask = Tensor::<B, 1>::from_data(TensorData::new(eos_mask, [b * nf]), device)
            .reshape([b, nf, 1]);
        let eos_vec = self
            .emb_eos
            .forward(Tensor::<B, 1, Int>::zeros([1], device).reshape([1, 1]))
            .reshape([1, 1, self.d_model]);
        let song = ft.clone() + eos_mask.mul(eos_vec) + self.segment(0, device);
        (song, ft)
    }

    /// Document ids over the full generative sequence `song(nf) ++ labels(nf)`:
    /// the song half's ids verbatim; a label slot inherits its frame's document
    /// when the frame is real, else `-1` (dummy slot — masked everywhere).
    fn generative_doc_ids(data: &EventBatchData) -> Vec<i64> {
        let (b, nf) = (data.batch, data.n_frames);
        let mut ids = Vec::with_capacity(b * 2 * nf);
        for bi in 0..b {
            let row = &data.doc_id[bi * nf..(bi + 1) * nf];
            let kinds = &data.slot_kind[bi * nf..(bi + 1) * nf];
            ids.extend_from_slice(row);
            for f in 0..nf {
                ids.push(if kinds[f] == SLOT_FRAME { row[f] } else { -1 });
            }
        }
        // Row-major over [b, 2nf] — but we built [song_row ++ label_row] per b,
        // which IS row-major over the concatenated sequence. Rearrange: the loop
        // above already emits per-b contiguous rows of length 2nf.
        ids
    }

    /// Masked mean of the label-half hidden states over **real frame** slots
    /// (`[b, nf, d]` → `[b, d]`), the key head's pooling. Falls back to a plain
    /// mean when a row has no real frames (degenerate, but keeps it finite).
    fn pooled_key_input(
        &self,
        hidden: &Tensor<B, 3>,
        data: &EventBatchData,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let (b, nf) = (data.batch, data.n_frames);
        let w: Vec<f32> = data
            .slot_kind
            .iter()
            .map(|&k| if k == SLOT_FRAME { 1.0 } else { 0.0 })
            .collect();
        let w = Tensor::<B, 1>::from_data(TensorData::new(w, [b * nf]), device).reshape([b, nf, 1]);
        let denom = w.clone().sum_dim(1).clamp_min(1.0); // [b, 1, 1]
        hidden
            .clone()
            .mul(w)
            .sum_dim(1)
            .div(denom)
            .reshape([b, self.d_model])
    }

    /// Label-slot inputs `[b, nf, d]`: the frame token again (slot↔frame alignment —
    /// the attention layers are NoPE) + previous-label embeddings + label segment.
    /// `prev_root`/`prev_quality` are `b·nf` class indices (BOS at slot 0).
    fn label_slots(
        &self,
        frame_tokens: &Tensor<B, 3>,
        prev_root: Vec<i64>,
        prev_quality: Vec<i64>,
        b: usize,
        nf: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let idx = |v: Vec<i64>| {
            Tensor::<B, 1, Int>::from_data(TensorData::new(v, [b * nf]), device).reshape([b, nf])
        };
        frame_tokens.clone()
            + self.emb_prev_root.forward(idx(prev_root))
            + self.emb_prev_quality.forward(idx(prev_quality))
            + self.segment(1, device)
    }

    /// Run the causal trunk over `song ++ label` slots and return the label half's
    /// hidden states `[b, nf, d]`, with document structure applied throughout.
    fn generative_hidden(
        &self,
        song: Tensor<B, 3>,
        label_slots: Tensor<B, 3>,
        data: &EventBatchData,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let [b, nf, d] = song.dims();
        let doc_ids = Self::generative_doc_ids(data);
        let docs = Docs::from_doc_ids(&doc_ids, b, 2 * nf, device);
        let hidden =
            self.encoder
                .forward(Tensor::cat(vec![song, label_slots], 1), true, Some(&docs));
        hidden.slice([0..b, nf..2 * nf, 0..d])
    }

    /// BOS-shifted previous-label class indices for teacher forcing: slot 0 gets
    /// BOS, slot `t` gets label `t−1`.
    fn teacher_prev(chord_labels: &[usize], b: usize, nf: usize) -> (Vec<i64>, Vec<i64>) {
        let mut prev_root = Vec::with_capacity(b * nf);
        let mut prev_quality = Vec::with_capacity(b * nf);
        for w in 0..b {
            prev_root.push(N_ROOT_CLASSES as i64); // BOS
            prev_quality.push(N_QUALITY_CLASSES as i64);
            for t in 1..nf {
                let (r, q) = chord_label_to_root_quality(chord_labels[w * nf + t - 1]);
                prev_root.push(r as i64);
                prev_quality.push(q as i64);
            }
        }
        (prev_root, prev_quality)
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

    const VARLEN: bool = true;

    fn build_batch(songs: &[Song]) -> EventBatchData {
        EventBatchData::build_generative(songs)
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

    fn frame_mask(data: &EventBatchData) -> Option<&[f32]> {
        Some(&data.label_valid)
    }

    /// φ runs per note (same cost as m01's front-end); the trunk is
    /// [`flops::kda_layer`] per layer instead of [`flops::transformer_layer`], and
    /// runs over `2·n_frames` slots (song ++ labels — the generative formulation's
    /// teacher-forced pass; greedy decode costs ~`n_frames`× this, not modelled).
    fn flops_per_window(cfg: &KdaModelConfig, notes_per_window: usize) -> u64 {
        let seq = 2 * cfg.n_frames;
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
            + flops::chord_key_heads(cfg.n_frames, cfg.d_model)
    }

    /// Supervised forward = **teacher-forced generative pass**: one fully causal
    /// trunk run over `song ++ label` slots with the batch's own labels shifted in
    /// as the previous-chord conditioning. Val metrics computed from this are
    /// teacher-forced; honest decoding is [`Backbone::infer_output`].
    fn forward_output(&self, data: &EventBatchData, device: &B::Device) -> ModelOutput<B> {
        let (b, nf) = (data.batch, data.n_frames);
        let (song, ft) = self.song_tokens(data, device);
        let (prev_root, prev_quality) = Self::teacher_prev(&data.chord_labels, b, nf);
        let slots = self.label_slots(&ft, prev_root, prev_quality, b, nf, device);
        let hidden = self.generative_hidden(song, slots, data, device);
        let root_logits = self.root_head.forward(hidden.clone());
        let quality_logits = self.quality_head.forward(hidden.clone());
        let key_logits = self
            .key_head
            .forward(self.pooled_key_input(&hidden, data, device));
        ModelOutput {
            root_logits,
            quality_logits,
            key_logits,
        }
    }

    /// Greedy autoregressive decode with **incremental state**: one prefill over
    /// the song half (building the per-layer KDA-state + attention-KV cache),
    /// then one `forward_step` per label slot — O(nf) total trunk work instead of
    /// the O(nf²) re-forward decode this replaces. The step's argmax feeds the
    /// next slot's previous-label conditioning; the collected label-half hidden
    /// states feed the masked-pooled key head.
    fn infer_output(&self, data: &EventBatchData, device: &B::Device) -> ModelOutput<B> {
        let (b, nf) = (data.batch, data.n_frames);
        let d = self.d_model;
        let (song, ft) = self.song_tokens(data, device);

        let full_ids = Self::generative_doc_ids(data);
        // Per-row split into song-half and label-half doc ids.
        let song_ids: Vec<i64> = (0..b)
            .flat_map(|bi| full_ids[bi * 2 * nf..bi * 2 * nf + nf].to_vec())
            .collect();
        let label_ids: Vec<Vec<i64>> = (0..b)
            .map(|bi| full_ids[bi * 2 * nf + nf..(bi + 1) * 2 * nf].to_vec())
            .collect();

        let song_docs = Docs::from_doc_ids(&song_ids, b, nf, device);
        let (_, mut cache) = self.encoder.forward_prefill(song, Some(&song_docs));

        // Previous decoded classes (BOS at slot 0).
        let mut prev_root = vec![N_ROOT_CLASSES as i64; b];
        let mut prev_quality = vec![N_QUALITY_CLASSES as i64; b];
        let mut root_logit_steps = Vec::with_capacity(nf);
        let mut quality_logit_steps = Vec::with_capacity(nf);
        let mut hidden_steps = Vec::with_capacity(nf);

        #[allow(clippy::needless_range_loop)] // t indexes several parallel per-slot structures
        for t in 0..nf {
            let idx = |v: &[i64]| {
                Tensor::<B, 1, Int>::from_data(TensorData::new(v.to_vec(), [b]), device)
                    .reshape([b, 1])
            };
            let x_t = ft.clone().slice([0..b, t..t + 1, 0..d])
                + self.emb_prev_root.forward(idx(&prev_root))
                + self.emb_prev_quality.forward(idx(&prev_quality))
                + self.segment(1, device);
            let doc_t: Vec<i64> = (0..b).map(|bi| label_ids[bi][t]).collect();
            let h_t = self.encoder.forward_step(x_t, &doc_t, &mut cache);
            let rl = self.root_head.forward(h_t.clone()); // [b, 1, roots]
            let ql = self.quality_head.forward(h_t.clone());
            prev_root = rl
                .clone()
                .argmax(2)
                .reshape([b])
                .into_data()
                .to_vec()
                .unwrap();
            prev_quality = ql
                .clone()
                .argmax(2)
                .reshape([b])
                .into_data()
                .to_vec()
                .unwrap();
            root_logit_steps.push(rl);
            quality_logit_steps.push(ql);
            hidden_steps.push(h_t);
        }

        let hidden = Tensor::cat(hidden_steps, 1); // [b, nf, d]
        ModelOutput {
            root_logits: Tensor::cat(root_logit_steps, 1),
            quality_logits: Tensor::cat(quality_logit_steps, 1),
            key_logits: self
                .key_head
                .forward(self.pooled_key_input(&hidden, data, device)),
        }
    }
}

impl<B: Backend> ArBackbone<B> for KdaKeyChordModel<B> {
    /// AR pretraining forward (causal at the attention layers; KDA is always
    /// causal): per-frame next-frame content logits.
    fn ar_forward(&self, data: &EventBatchData, device: &B::Device) -> ArOutput<B> {
        // Song half only (no label slots), stamped with the song segment so the
        // pretrained trunk sees the same input distribution the fine-tune's song
        // half does.
        // Packed sequences bring document structure: block-diagonal attention +
        // KDA state reset at each song's start, EOS embeddings at the separators.
        let (b, nf) = (data.batch, data.n_frames);
        let (song, _ft) = self.song_tokens(data, device);
        let docs = Docs::from_doc_ids(&data.doc_id, b, nf, device);
        let hidden = self.encoder.forward(song, true, Some(&docs));
        ArOutput {
            pc_logits: self.ar_pc_head.forward(hidden.clone()),
            channel_logits: self.ar_ch_head.forward(hidden.clone()),
            eos_logits: Some(self.ar_eos_head.forward(hidden)),
        }
    }

    fn ar_targets(data: &EventBatchData) -> (Vec<f32>, Vec<f32>) {
        crate::tokenize::ar_targets(&data.examples)
    }

    fn ar_slot_kinds(data: &EventBatchData) -> Option<&[i64]> {
        Some(&data.slot_kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Inner;

    /// The greedy AR decode ([`Backbone::infer_output`]) must produce the same
    /// output shapes as the teacher-forced pass (the shared contract), with finite
    /// logits — it is a different code path (`n_frames` re-forwards) the contract
    /// test never touches.
    #[test]
    fn greedy_decode_matches_output_contract() {
        use crate::backend::MlDevice;
        use crate::notes::{Instrument, NoteEvent, Song};

        let device = MlDevice::default();
        let nf = 6;
        let song = Song {
            key_label: 0,
            n_frames: nf,
            notes: vec![NoteEvent {
                start_frame: 0,
                end_frame: 4,
                pitch: 60,
                velocity: 1.0,
                instrument: Instrument::Bass,
                track: 5,
                pan: 0.0,
            }],
            chord_labels: vec![0; nf],
            is_music: None,
            ..Song::default()
        };
        // The generative builder pads to n_frames + 1 (EOS slot).
        let batch = EventBatchData::build_generative(&[song]);
        let cfg = KdaModelConfig::new()
            .with_d_model(32)
            .with_d_ff(64)
            .with_n_layers(4)
            .with_chunk_size(16)
            .with_dropout(0.0);
        let model = cfg.init::<Inner>(&device);
        let out = Backbone::<Inner>::infer_output(&model, &batch, &device);
        let padded = nf + 1; // frames + EOS slot
        assert_eq!(out.root_logits.dims(), [1, padded, N_ROOT_CLASSES]);
        assert_eq!(out.quality_logits.dims(), [1, padded, N_QUALITY_CLASSES]);
        assert_eq!(out.key_logits.dims(), [1, N_KEY_CLASSES]);
        let data: Vec<f32> = out.root_logits.into_data().to_vec().unwrap();
        assert!(data.iter().all(|v| v.is_finite()), "NaN/Inf in decode");
    }
}

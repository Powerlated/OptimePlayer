//! Pre-norm transformer encoder with **rotary position embeddings** (RoPE).
//!
//! ## Why this exists instead of `burn::nn::transformer::TransformerEncoder`
//!
//! burn ships [`RotaryEncoding`], but neither its `MultiHeadAttention` nor its
//! `TransformerEncoder` exposes a hook to apply it. RoPE has to rotate **Q and K
//! inside attention**, after the head split — that is what makes a dot product
//! `qᵢ·kⱼ` depend only on the *relative* offset `i-j`.
//!
//! Rotating the layer input instead would not be RoPE: Q and K are arbitrary linear
//! projections of `x`, and a rotation does not commute with them, so `rot(x)Wq ·
//! rot(x)Wk` carries no relative-position guarantee. That shortcut is a different
//! (and much weaker) operation wearing RoPE's name. So the attention block is
//! written out here.
//!
//! ## What it replaces
//!
//! The learned position embedding every generation used to add to its frame tokens
//! (`m00`'s `pos_emb`, `m01`/`m02`'s `frame_pos`). RoPE injects position inside
//! attention instead, so there is no additive position term and no position table:
//! the trunk generalises past the window it trained on, and relative offsets are
//! encoded directly rather than learned per absolute slot.
//!
//! Only **sequence** trunks use this. `m02`'s within-frame set encoder must NOT:
//! a frame's notes are an unordered set, and slot index carries no order.

use burn::config::Config;
use burn::module::Module;
use burn::nn::{
    Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, RotaryEncoding,
    RotaryEncodingConfig,
};
use burn::prelude::*;
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::Bool;

#[derive(Config, Debug)]
pub struct RopeEncoderConfig {
    pub d_model: usize,
    pub d_ff: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    /// Longest sequence the rotation cache covers. RoPE itself extrapolates, but the
    /// cache is precomputed, so this bounds one forward pass.
    pub max_seq_len: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
    /// RoPE base frequency. 10000 is the RoFormer/Llama default.
    #[config(default = 10000.0)]
    pub theta: f32,
}

impl RopeEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RopeEncoder<B> {
        let head_dim = self.d_model / self.n_heads;
        assert!(
            self.d_model.is_multiple_of(self.n_heads),
            "d_model {} must divide by n_heads {}",
            self.d_model,
            self.n_heads
        );
        // RoPE rotates coordinate *pairs*, so a head's width must be even.
        assert!(
            head_dim.is_multiple_of(2),
            "RoPE needs an even head_dim, got {head_dim}"
        );
        RopeEncoder {
            layers: (0..self.n_layers)
                .map(|_| RopeEncoderLayer::init(self, device))
                .collect(),
            // The cache is keyed by head_dim: rotation happens per head, not on the
            // full d_model vector.
            rope: RotaryEncodingConfig::new(self.max_seq_len, head_dim)
                .with_theta(self.theta)
                .init(device),
            norm: LayerNormConfig::new(self.d_model).init(device),
            n_heads: self.n_heads,
        }
    }
}

/// Stack of pre-norm RoPE layers + a final norm.
#[derive(Module, Debug)]
pub struct RopeEncoder<B: Backend> {
    layers: Vec<RopeEncoderLayer<B>>,
    rope: RotaryEncoding<B>,
    norm: LayerNorm<B>,
    n_heads: usize,
}

impl<B: Backend> RopeEncoder<B> {
    /// `x`: `[batch, seq, d_model]` → same shape. `causal` masks each position from
    /// attending to later ones (AR pretraining); `false` is the bidirectional
    /// fine-tune.
    ///
    /// No padding mask: every trunk here runs over a fixed-length frame grid where
    /// all positions are real.
    pub fn forward(&self, x: Tensor<B, 3>, causal: bool) -> Tensor<B, 3> {
        let mask = causal.then(|| {
            let [b, n, _] = x.dims();
            causal_mask::<B>(b, self.n_heads, n, &x.device())
        });
        let mut h = x;
        for layer in &self.layers {
            h = layer.forward(h, &self.rope, self.n_heads, mask.clone());
        }
        self.norm.forward(h)
    }
}

/// `[batch, heads, seq, seq]`, true where position `i` must not see `j` (`j > i`).
fn causal_mask<B: Backend>(
    batch: usize,
    heads: usize,
    seq: usize,
    device: &B::Device,
) -> Tensor<B, 4, Bool> {
    let row = Tensor::<B, 1, Int>::arange(0..seq as i64, device).reshape([seq, 1]);
    let col = Tensor::<B, 1, Int>::arange(0..seq as i64, device).reshape([1, seq]);
    col.repeat_dim(0, seq)
        .greater(row.repeat_dim(1, seq))
        .reshape([1, 1, seq, seq])
        .repeat_dim(0, batch)
        .repeat_dim(1, heads)
}

#[derive(Module, Debug)]
struct RopeEncoderLayer<B: Backend> {
    norm_attn: LayerNorm<B>,
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    out: Linear<B>,
    norm_ff: LayerNorm<B>,
    ff_up: Linear<B>,
    ff_down: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> RopeEncoderLayer<B> {
    fn init(cfg: &RopeEncoderConfig, device: &B::Device) -> Self {
        let d = cfg.d_model;
        let lin = |i: usize, o: usize| LinearConfig::new(i, o).init(device);
        Self {
            norm_attn: LayerNormConfig::new(d).init(device),
            query: lin(d, d),
            key: lin(d, d),
            value: lin(d, d),
            out: lin(d, d),
            norm_ff: LayerNormConfig::new(d).init(device),
            ff_up: lin(d, cfg.d_ff),
            ff_down: lin(cfg.d_ff, d),
            dropout: DropoutConfig::new(cfg.dropout).init(),
        }
    }

    /// Pre-norm: `x + attn(norm(x))`, then `x + ff(norm(x))`.
    fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEncoding<B>,
        n_heads: usize,
        mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 3> {
        let h = self.attention(self.norm_attn.forward(x.clone()), rope, n_heads, mask);
        let x = x + self.dropout.forward(h);

        let f = self.norm_ff.forward(x.clone());
        let f = self.ff_down.forward(gelu(self.ff_up.forward(f)));
        x + self.dropout.forward(f)
    }

    fn attention(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEncoding<B>,
        n_heads: usize,
        mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 3> {
        let [b, n, d] = x.dims();
        let head_dim = d / n_heads;
        // [b, n, d] -> [b, heads, n, head_dim]: rope::apply reads the last two dims
        // as (seq, hidden), so the rotation lands per head exactly as intended.
        let split = |t: Tensor<B, 3>| t.reshape([b, n, n_heads, head_dim]).swap_dims(1, 2);

        let q = rope.forward(split(self.query.forward(x.clone())));
        let k = rope.forward(split(self.key.forward(x.clone())));
        let v = split(self.value.forward(x));

        let scores = q
            .matmul(k.swap_dims(2, 3))
            .div_scalar((head_dim as f32).sqrt());
        let scores = match mask {
            Some(m) => scores.mask_fill(m, f32::NEG_INFINITY),
            None => scores,
        };
        let attn = self.dropout.forward(softmax(scores, 3));

        let out = attn.matmul(v).swap_dims(1, 2).reshape([b, n, d]);
        self.out.forward(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Inner, MlDevice};

    fn cfg(seq: usize) -> RopeEncoderConfig {
        RopeEncoderConfig::new(32, 64, 4, 2, seq).with_dropout(0.0)
    }

    #[test]
    fn preserves_shape() {
        let device = MlDevice::default();
        let enc = cfg(16).init::<Inner>(&device);
        let x = Tensor::<Inner, 3>::zeros([2, 16, 32], &device);
        assert_eq!(enc.forward(x, false).dims(), [2, 16, 32]);
    }

    /// **The** property that makes this RoPE rather than a position table: after
    /// rotation, `q_i · k_j` depends only on the relative offset `i - j`. Pinned at
    /// the rotation itself (an encoder's outputs can't show it directly — token `i`
    /// of a shifted sequence sees a different *set* of predecessors, so its output
    /// legitimately differs).
    ///
    /// Uses `head_dim`, not `d_model`, which is the thing easy to get wrong: the
    /// rotation must be per head.
    #[test]
    fn rope_dot_product_depends_only_on_relative_offset() {
        let device = MlDevice::default();
        let head_dim = 32;
        let rope = RotaryEncodingConfig::new(64, head_dim).init::<Inner>(&device);

        let q: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.3).sin()).collect();
        let k: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.7).cos()).collect();
        let dot = |i: usize, j: usize| -> f32 {
            let tq = Tensor::<Inner, 3>::from_data(
                TensorData::new(q.clone(), [1, 1, head_dim]),
                &device,
            );
            let tk = Tensor::<Inner, 3>::from_data(
                TensorData::new(k.clone(), [1, 1, head_dim]),
                &device,
            );
            (rope.apply(tq, i) * rope.apply(tk, j)).sum().into_scalar()
        };

        // Same offset (-3) at four different absolute positions ⇒ same score.
        let base = dot(2, 5);
        for shift in [1usize, 3, 7, 20] {
            let moved = dot(2 + shift, 5 + shift);
            assert!(
                (base - moved).abs() < 1e-4,
                "offset -3 scored {base} at pos 2, {moved} after shifting {shift}"
            );
        }
        // A different offset must actually score differently, or the test above
        // would pass on a no-op rotation.
        assert!(
            (dot(2, 5) - dot(2, 6)).abs() > 1e-4,
            "offsets -3 and -4 scored the same; rotation isn't doing anything"
        );
    }

    /// A causal trunk must not let a position read the future.
    #[test]
    fn causal_mask_hides_later_tokens() {
        let device = MlDevice::default();
        let enc = cfg(8).init::<Inner>(&device);
        let d = 32;

        let base = vec![0.1f32; 8 * d];
        let mut changed = base.clone();
        // Perturb only the LAST token.
        for i in 0..d {
            changed[7 * d + i] = 5.0;
        }
        let t0 = Tensor::<Inner, 3>::from_data(TensorData::new(base, [1, 8, d]), &device);
        let t1 = Tensor::<Inner, 3>::from_data(TensorData::new(changed, [1, 8, d]), &device);

        let o0 = enc.forward(t0, true);
        let o1 = enc.forward(t1, true);
        // Token 0 precedes the change, so it must be untouched.
        let a: Vec<f32> = o0.slice([0..1, 0..1, 0..d]).into_data().to_vec().unwrap();
        let b: Vec<f32> = o1.slice([0..1, 0..1, 0..d]).into_data().to_vec().unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6, "future leaked into an earlier token");
        }
    }
}

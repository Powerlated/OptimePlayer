//! Kimi Linear attention — **KDA** (Kimi Delta Attention), from Kimi Linear
//! (arXiv 2510.26692). Gated DeltaNet with the scalar decay gate generalised to a
//! **per-channel (fine-grained)** one, plus a hybrid stack that intersperses KDA
//! layers with ordinary full attention.
//!
//! ## The recurrence (per head, state `S_t ∈ R^{d_k × d_v}`)
//!
//! ```text
//! S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ
//! o_t = S_tᵀ q_t
//! ```
//!
//! `α_t ∈ (0,1)^{d_k}` is the per-channel decay, `β_t ∈ [0,1]` a per-head scalar gate.
//! Worked entirely in log space: `g_t = log α_t`, never materialised as `α` and
//! re-logged.
//!
//! ## Neural parameterisation (per head, input `x_t ∈ R^d`)
//!
//! ```text
//! q_t, k_t = L2Norm(Swish(ShortConv(W_{q/k} x_t)))         ∈ R^{d_k}
//! v_t      = Swish(ShortConv(W_v x_t))                     ∈ R^{d_v}
//! g_t      = -exp(a_log) ⊙ softplus(W_α↑ W_α↓ x_t + b_α)   ∈ (-∞,0]^{d_k}  (= log α_t)
//! β_t      = Sigmoid(W_β x_t)                              ∈ [0,1]
//! o_t      = W_o( Sigmoid(W_g↑ W_g↓ x_t) ⊙ RMSNorm_headwise(KDA(q,k,v,g,β)) )
//! ```
//!
//! `ShortConv` = depthwise causal 1-D conv, kernel width 4 (Gated DeltaNet
//! convention): pad left 3, none right, `groups = channels`.
//!
//! `a_log` is a learned per-channel parameter (size `n_heads·d_k`), Mamba2-style:
//! initialised as `ln(A)`, `A ~ Uniform(1, 16)`. `b_α` is `alpha_up`'s bias, not a
//! separate `Param` — it adds inside the same affine map, so folding it into the
//! `Linear` is exact. It is **not** left at its default zero init: solved per-channel
//! so the *initial* decay is `α ≈ exp(-0.05) ≈ 0.951` (inside the paper's 0.9–0.99
//! target band) regardless of that channel's `A` draw — see [`KdaMixerConfig::init`].
//!
//! ## Hybrid architecture
//!
//! Kimi Linear repeats 3 KDA layers then 1 full-attention layer. The paper's
//! full-attention layer is MLA (multi-head latent attention); this simplifies to
//! **standard multi-head attention** — the KDA mixer is the point of this module, MLA
//! is an orthogonal KV-cache optimisation this codebase has no cache to spend. The
//! attention layers run **NoPE** (no positional encoding at all): position is carried
//! entirely by KDA's data-dependent decay, so nothing needs to be added at the
//! attention layers. Each layer is pre-norm residual: `x = x + mixer(norm(x));
//! x = x + ffn(norm(x))`.
//!
//! KDA is causal by construction (the recurrence only ever looks backward), so the
//! `causal` flag on [`KimiEncoder::forward`] affects **only** the attention layers —
//! the "bidirectional" supervised pass is therefore only bidirectional at those
//! layers. An accepted, documented deviation from the other generations' trunks
//! (which are uniformly causal-or-not throughout).

use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{
    Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig1d,
    RmsNorm, RmsNormConfig,
};
use burn::prelude::*;
use burn::tensor::activation::{gelu, sigmoid, silu, softmax, softplus};

use crate::transformer::causal_mask;

/// Depthwise causal short-conv kernel width (Gated DeltaNet convention).
const SHORT_CONV_KERNEL: usize = 4;
/// Floor under the L2-norm denominator (avoids a divide-by-zero on an all-zero
/// vector; never binds in practice since q/k are learned projections).
const L2_NORM_EPS: f32 = 1e-6;
/// Target initial magnitude of `-g_t` (so `alpha_t ≈ exp(-0.05) ≈ 0.951`, inside the
/// paper's 0.9–0.99 target band) regardless of a channel's Mamba2 `A` draw — see
/// [`KdaMixerConfig::init`].
const INIT_DECAY_EPS: f32 = 0.05;

// ============================================================================
// KdaMixer — one KDA token-mixing layer.
// ============================================================================

#[derive(Config, Debug)]
pub struct KdaMixerConfig {
    pub d_model: usize,
    pub n_heads: usize,
    /// Chunk length for the chunkwise WY/UT scan.
    #[config(default = 64)]
    pub chunk_size: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl KdaMixerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> KdaMixer<B> {
        assert!(
            self.d_model.is_multiple_of(self.n_heads),
            "d_model {} must divide by n_heads {}",
            self.d_model,
            self.n_heads
        );
        let d = self.d_model;
        let head_dim = d / self.n_heads;
        let lin = |i: usize, o: usize| LinearConfig::new(i, o).init(device);
        let causal_conv = || {
            Conv1dConfig::new(d, d, SHORT_CONV_KERNEL)
                .with_groups(d)
                .with_padding(PaddingConfig1d::Explicit(SHORT_CONV_KERNEL - 1, 0))
                .init(device)
        };

        // Mamba2/GDN decay init: `a_log = ln(A)`, `A ~ Uniform(1, 16)` per channel. Kept
        // as an explicit `Param` (not a `Linear` bias) because it *multiplies* the
        // projected pre-activation rather than adding to it.
        let mut rng = rand::thread_rng();
        let a: Vec<f32> = (0..d)
            .map(|_| rand::Rng::gen_range(&mut rng, 1.0f32..16.0f32))
            .collect();
        let a_log_data: Vec<f32> = a.iter().map(|v| v.ln()).collect();
        let a_log = Param::from_tensor(Tensor::<B, 1>::from_data(
            TensorData::new(a_log_data, [d]),
            device,
        ));

        let alpha_down = lin(d, head_dim);
        let mut alpha_up = lin(head_dim, d);
        // Bias the low-rank decay gate so `A_j * softplus(b_j) = INIT_DECAY_EPS` for
        // every channel `j`, independent of that channel's `A` draw:
        // `b_j = softplus⁻¹(EPS / A_j) = ln(exp(EPS / A_j) - 1)`.
        let bias_data: Vec<f32> = a
            .iter()
            .map(|&aj| {
                let y = (INIT_DECAY_EPS / aj) as f64;
                (y.exp() - 1.0).ln() as f32
            })
            .collect();
        alpha_up.bias = Some(Param::from_tensor(Tensor::<B, 1>::from_data(
            TensorData::new(bias_data, [d]),
            device,
        )));

        KdaMixer {
            q_proj: lin(d, d),
            k_proj: lin(d, d),
            v_proj: lin(d, d),
            conv_q: causal_conv(),
            conv_k: causal_conv(),
            conv_v: causal_conv(),
            alpha_down,
            alpha_up,
            a_log,
            beta_proj: lin(d, self.n_heads),
            gate_down: lin(d, head_dim),
            gate_up: lin(head_dim, d),
            o_proj: lin(d, d),
            norm: RmsNormConfig::new(head_dim).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            n_heads: self.n_heads,
            head_dim,
            chunk_size: self.chunk_size,
        }
    }
}

#[derive(Module, Debug)]
pub struct KdaMixer<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    conv_q: Conv1d<B>,
    conv_k: Conv1d<B>,
    conv_v: Conv1d<B>,
    alpha_down: Linear<B>,
    alpha_up: Linear<B>,
    a_log: Param<Tensor<B, 1>>,
    beta_proj: Linear<B>,
    gate_down: Linear<B>,
    gate_up: Linear<B>,
    o_proj: Linear<B>,
    /// Shared (not per-head) `RmsNorm(head_dim)`, applied headwise (see `forward`).
    norm: RmsNorm<B>,
    dropout: Dropout,
    n_heads: usize,
    head_dim: usize,
    chunk_size: usize,
}

impl<B: Backend> KdaMixer<B> {
    /// `x`: `[batch, seq, d_model]` → same shape. KDA is causal by construction, so
    /// unlike the attention layers there is no `causal` argument here.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, seq, d] = x.dims();
        let h = self.n_heads;
        let dh = self.head_dim;
        let device = x.device();

        let conv_branch = |lin: &Linear<B>, conv: &Conv1d<B>, x: Tensor<B, 3>| -> Tensor<B, 3> {
            conv.forward(lin.forward(x).swap_dims(1, 2)).swap_dims(1, 2)
        };
        let q = silu(conv_branch(&self.q_proj, &self.conv_q, x.clone()));
        let k = silu(conv_branch(&self.k_proj, &self.conv_k, x.clone()));
        let v = silu(conv_branch(&self.v_proj, &self.conv_v, x.clone()));

        // [b, seq, h*dim] -> [b, h, seq, dim] -> [b*h, seq, dim], so the chunkwise scan
        // (which only knows about a leading batch axis) treats every head as an
        // independent batch element.
        let to_heads = |t: Tensor<B, 3>, dim: usize| -> Tensor<B, 3> {
            t.reshape([b, seq, h, dim])
                .swap_dims(1, 2)
                .reshape([b * h, seq, dim])
        };

        let q = l2_norm_last(to_heads(q, dh));
        let k = l2_norm_last(to_heads(k, dh));
        let v = to_heads(v, dh);

        // Low-rank per-channel decay gate, in log space throughout: `g_t = log alpha_t
        // = -exp(a_log) * softplus(W_up W_down x_t + b_alpha)`. `b_alpha` lives on
        // `alpha_up`'s bias (see `KdaMixerConfig::init`).
        let raw_alpha = self.alpha_up.forward(self.alpha_down.forward(x.clone()));
        let decay_scale = self.a_log.val().exp().reshape([1, 1, d]);
        let g = softplus(raw_alpha, 1.0).mul(decay_scale).neg();
        let g = to_heads(g, dh);

        let beta = sigmoid(self.beta_proj.forward(x.clone())); // [b, seq, h]
        let beta = beta.swap_dims(1, 2).reshape([b * h, seq]);

        let o = chunk_kda(q, k, v, g, beta, self.chunk_size); // [b*h, seq, dh]

        // Headwise RMSNorm: reshape to expose (b, seq, h, dh) so the shared-gamma norm
        // runs per head, then flatten back for the gate + output projection.
        let o = o
            .reshape([b, h, seq, dh])
            .swap_dims(1, 2)
            .reshape([b, seq, h, dh]);
        let o = self.norm.forward(o).reshape([b, seq, d]);

        let gate = sigmoid(self.gate_up.forward(self.gate_down.forward(x.clone())));
        let _ = device;
        self.o_proj.forward(self.dropout.forward(gate.mul(o)))
    }
}

/// L2-normalise the last dim (per-head q/k vectors).
fn l2_norm_last<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let dim = D - 1;
    let norm = (x.clone() * x.clone())
        .sum_dim(dim)
        .sqrt()
        .clamp_min(L2_NORM_EPS);
    x.div(norm)
}

// ============================================================================
// Chunkwise KDA scan.
// ============================================================================

/// Chunkwise KDA scan — the paper's Appendix pseudocode, transcribed to burn
/// tensors via the WY/UT-transform chunked delta rule (same construction as Gated
/// DeltaNet, generalised from a scalar to a per-channel gate).
///
/// Shapes: `q`, `k`: `[BH, T, K]` (this fn applies the paper's `K^-1/2` query scale
/// itself), `v`: `[BH, T, V]`, `g`: `[BH, T, K]` (**raw per-step** log-decay, `<= 0` —
/// the chunk-local cumulative sum is taken inside), `beta`: `[BH, T]`. Returns
/// `[BH, T, V]`.
///
/// `T` need not be a multiple of `chunk_size`: the tail is zero-padded
/// (`q=k=v=g=β=0`) — an exact no-op on the recurrence, since `Diag(exp(0)) = I` and
/// `β=0` drops the rank-1 update — then the output is sliced back to `T`. This is
/// what makes short windows (e.g. an 8-frame test song against a 64-frame chunk)
/// correct rather than merely non-crashing.
fn chunk_kda<B: Backend>(
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    g: Tensor<B, 3>,
    beta: Tensor<B, 2>,
    chunk_size: usize,
) -> Tensor<B, 3> {
    let device = q.device();
    let [bh, seq, dk] = q.dims();
    let dv = v.dims()[2];
    let c = chunk_size;
    let seq_pad = seq.div_ceil(c) * c;
    let pad = seq_pad - seq;

    let pad3 = |x: Tensor<B, 3>, last: usize| -> Tensor<B, 3> {
        if pad == 0 {
            x
        } else {
            Tensor::cat(vec![x, Tensor::zeros([bh, pad, last], &device)], 1)
        }
    };
    let q = pad3(q.mul_scalar((dk as f64).powf(-0.5)), dk);
    let k = pad3(k, dk);
    let v = pad3(v, dv);
    let g = pad3(g, dk);
    let beta = if pad == 0 {
        beta
    } else {
        Tensor::cat(vec![beta, Tensor::zeros([bh, pad], &device)], 1)
    };

    let n = seq_pad / c;
    let take3 = |x: &Tensor<B, 3>, i: usize, last: usize| -> Tensor<B, 3> {
        x.clone().slice([0..bh, i * c..(i + 1) * c, 0..last])
    };
    let take2 = |x: &Tensor<B, 2>, i: usize| -> Tensor<B, 2> {
        x.clone().slice([0..bh, i * c..(i + 1) * c])
    };

    // Masks (data-independent, built once): `le[c,i]` true where `c <= i` (invalid for
    // the strictly-lower WY matrix — upper triangle *incl.* diagonal); `lt[c,j]` true
    // where `c < j` (invalid for the causal per-chunk output attention, which keeps
    // the diagonal).
    let le = tri_mask::<B>(c, false, &device);
    let lt = tri_mask::<B>(c, true, &device);
    let eye = eye_tensor::<B>(c, &device);

    let mut state = Tensor::<B, 3>::zeros([bh, dk, dv], &device);
    let mut outputs = Vec::with_capacity(n);

    for i in 0..n {
        let q_i = take3(&q, i, dk);
        let k_i = take3(&k, i, dk);
        let v_i = take3(&v, i, dv);
        let g_raw = take3(&g, i, dk);
        let beta_i = take2(&beta, i); // [bh, c]
        let g_i = g_raw.cumsum(1); // chunk-local cumulative log-decay, <= 0

        // --- WY/UT inverse transform for this chunk. ---
        // a0[c,i'] = sum_d k[c,d]*exp(g[c,d]-g[i',d])*k[i',d], zeroed where c<=i' — the
        // exponent is masked to -inf *before* `exp` (not zeroed after), so the
        // positive, overflow-prone exponents at c<=i' are never materialised.
        let a0 = pairwise_decayed_dot(&k_i, &g_i, &k_i, &g_i, &le);
        let beta_row = beta_i.clone().reshape([bh, c, 1]);
        let mut a = a0.mul(beta_row).neg(); // "-A.masked_fill(mask,0) * beta" (order harmless: 0*beta=0)

        // Forward substitution: row-by-row inverse of `I + L` (`L` = `a`'s strictly
        // lower part). Row `i` only ever reads already-finalised earlier rows, so a
        // plain sequential loop is exact — clarity over kernel speed, per this
        // codebase's convention for this class of scan.
        for row in 1..c {
            let cur = a.clone().slice([0..bh, row..row + 1, 0..c]); // [bh,1,c]
            let earlier = a.clone().slice([0..bh, 0..c, 0..row]); // [bh,c,row]
            let correction = cur.matmul(earlier); // [bh,1,row]
            let updated = a.clone().slice([0..bh, row..row + 1, 0..row]) + correction;
            a = a.slice_assign([0..bh, row..row + 1, 0..row], updated);
        }
        let a = (a + eye.clone()).mul(beta_i.clone().reshape([bh, 1, c])); // column-wise beta

        let w = a.clone().matmul(g_i.clone().exp().mul(k_i.clone())); // [bh,c,dk]
        let u = a.matmul(v_i.clone()); // [bh,c,dv]

        // --- Per-chunk causal output + state update. ---
        // a2[c,j] = sum_d q[c,d]*exp(g[c,d]-g[j,d])*k[j,d], zeroed where c<j (kept:
        // c>=j, i.e. the diagonal is included — this stage is causal, not strict).
        let a2 = pairwise_decayed_dot(&q_i, &g_i, &k_i, &g_i, &lt);
        let v_eff = u - w.matmul(state.clone()); // [bh,c,dv]
        let o_chunk =
            q_i.clone().mul(g_i.clone().exp()).matmul(state.clone()) + a2.matmul(v_eff.clone());
        outputs.push(o_chunk);

        // State carry: decay the old state to the end of the chunk, then add this
        // chunk's contribution decayed *forward* to the same point — every exponent
        // here is `g_last - g_c <= 0` (g is non-increasing within the chunk), so this
        // is safe without masking.
        let g_last = g_i.clone().slice([0..bh, c - 1..c, 0..dk]); // [bh,1,dk]
        let decay_to_end = (g_last.clone() - g_i).exp().mul(k_i); // [bh,c,dk]
        state = state.mul(g_last.reshape([bh, dk, 1]).exp())
            + decay_to_end.swap_dims(1, 2).matmul(v_eff);
    }

    let o = Tensor::cat(outputs, 1); // [bh, seq_pad, dv]
    o.slice([0..bh, 0..seq, 0..dv])
}

/// `out[c,i] = sum_d row[c,d] * exp(row_g[c,d] - col_g[i,d]) * col[i,d]`, masked to
/// `0` wherever `invalid[c,i]` — masked in the exponent (before `exp`) so an
/// otherwise-overflowing positive exponent is never computed. All tensors `[bh,c,d]`
/// except `invalid` (`[c,c]`); result `[bh,c,c]`.
fn pairwise_decayed_dot<B: Backend>(
    row: &Tensor<B, 3>,
    row_g: &Tensor<B, 3>,
    col: &Tensor<B, 3>,
    col_g: &Tensor<B, 3>,
    invalid: &Tensor<B, 2, Bool>,
) -> Tensor<B, 3> {
    let [bh, c, d] = row.dims();
    let g_row = row_g.clone().reshape([bh, c, 1, d]);
    let g_col = col_g.clone().reshape([bh, 1, c, d]);
    let diff = g_row - g_col; // [bh, c, c, d]
    let mask = invalid
        .clone()
        .reshape([1, c, c, 1])
        .repeat_dim(0, bh)
        .repeat_dim(3, d);
    let exp_diff = diff.mask_fill(mask, f32::NEG_INFINITY).exp();
    let k_row = row.clone().reshape([bh, c, 1, d]);
    let k_col = col.clone().reshape([bh, 1, c, d]);
    (k_row * exp_diff * k_col).sum_dim(3).reshape([bh, c, c])
}

/// `[C, C]` bool mask. `strict`=false: true where `row <= col` (upper incl.
/// diagonal). `strict`=true: true where `row < col` (strict upper).
fn tri_mask<B: Backend>(c: usize, strict: bool, device: &B::Device) -> Tensor<B, 2, Bool> {
    let row = Tensor::<B, 1, Int>::arange(0..c as i64, device).reshape([c, 1]);
    let col = Tensor::<B, 1, Int>::arange(0..c as i64, device).reshape([1, c]);
    let row = row.repeat_dim(1, c);
    let col = col.repeat_dim(0, c);
    if strict {
        col.greater(row)
    } else {
        col.greater_equal(row)
    }
}

/// `[1, C, C]` identity, broadcastable against a `[BH, C, C]` tensor.
fn eye_tensor<B: Backend>(c: usize, device: &B::Device) -> Tensor<B, 3> {
    let mut data = vec![0f32; c * c];
    for i in 0..c {
        data[i * c + i] = 1.0;
    }
    Tensor::<B, 2>::from_data(TensorData::new(data, [c, c]), device).reshape([1, c, c])
}

// ============================================================================
// KimiEncoder — the hybrid KDA / full-attention stack.
// ============================================================================

#[derive(Config, Debug)]
pub struct KimiEncoderConfig {
    pub d_model: usize,
    pub d_ff: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    /// KDA chunkwise-scan chunk length.
    #[config(default = 64)]
    pub chunk_size: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl KimiEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> KimiEncoder<B> {
        assert!(
            self.d_model.is_multiple_of(self.n_heads),
            "d_model {} must divide by n_heads {}",
            self.d_model,
            self.n_heads
        );
        let layers = (0..self.n_layers)
            .map(|i| {
                // Every 4th layer (indices 3, 7, …) is full attention; the rest KDA.
                let is_attn = (i + 1).is_multiple_of(4);
                let mixer = if is_attn {
                    MixerKind::Attn(AttnMixer {
                        query: LinearConfig::new(self.d_model, self.d_model).init(device),
                        key: LinearConfig::new(self.d_model, self.d_model).init(device),
                        value: LinearConfig::new(self.d_model, self.d_model).init(device),
                        out: LinearConfig::new(self.d_model, self.d_model).init(device),
                        n_heads: self.n_heads,
                    })
                } else {
                    MixerKind::Kda(
                        KdaMixerConfig::new(self.d_model, self.n_heads)
                            .with_chunk_size(self.chunk_size)
                            .with_dropout(self.dropout)
                            .init(device),
                    )
                };
                KimiLayer {
                    norm_mixer: LayerNormConfig::new(self.d_model).init(device),
                    mixer,
                    norm_ff: LayerNormConfig::new(self.d_model).init(device),
                    ff_up: LinearConfig::new(self.d_model, self.d_ff).init(device),
                    ff_down: LinearConfig::new(self.d_ff, self.d_model).init(device),
                    dropout: DropoutConfig::new(self.dropout).init(),
                }
            })
            .collect();
        KimiEncoder {
            layers,
            norm: LayerNormConfig::new(self.d_model).init(device),
            n_heads: self.n_heads,
        }
    }
}

#[derive(Module, Debug)]
pub struct KimiEncoder<B: Backend> {
    layers: Vec<KimiLayer<B>>,
    norm: LayerNorm<B>,
    n_heads: usize,
}

impl<B: Backend> KimiEncoder<B> {
    /// `x`: `[batch, seq, d_model]` → same shape. `causal` affects **only** the
    /// full-attention layers — KDA is causal by construction regardless.
    pub fn forward(&self, x: Tensor<B, 3>, causal: bool) -> Tensor<B, 3> {
        let mask = causal.then(|| {
            let [b, n, _] = x.dims();
            causal_mask::<B>(b, self.n_heads, n, &x.device())
        });
        let mut h = x;
        for layer in &self.layers {
            h = layer.forward(h, mask.clone());
        }
        self.norm.forward(h)
    }
}

// `KdaMixer` is meaningfully larger than `AttnMixer` (3 short convs + a low-rank
// decay gate vs. plain QKVO projections) — that asymmetry is inherent to a 3-KDA-
// layers-per-1-attention-layer hybrid, not a size accident to box away. Burn's
// `Module` derive isn't implemented for `Box<M>`, so boxing isn't available anyway.
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
enum MixerKind<B: Backend> {
    Kda(KdaMixer<B>),
    Attn(AttnMixer<B>),
}

#[derive(Module, Debug)]
struct KimiLayer<B: Backend> {
    norm_mixer: LayerNorm<B>,
    mixer: MixerKind<B>,
    norm_ff: LayerNorm<B>,
    ff_up: Linear<B>,
    ff_down: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> KimiLayer<B> {
    fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 4, Bool>>) -> Tensor<B, 3> {
        let normed = self.norm_mixer.forward(x.clone());
        let mixed = match &self.mixer {
            MixerKind::Kda(m) => m.forward(normed),
            MixerKind::Attn(m) => m.forward(normed, mask),
        };
        let x = x + self.dropout.forward(mixed);

        let f = self.norm_ff.forward(x.clone());
        let f = self.ff_down.forward(gelu(self.ff_up.forward(f)));
        x + self.dropout.forward(f)
    }
}

/// NoPE (no positional encoding) standard multi-head attention — the paper's
/// full-attention layer is MLA; this simplifies to plain MHA (see module docs).
#[derive(Module, Debug)]
struct AttnMixer<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    out: Linear<B>,
    n_heads: usize,
}

impl<B: Backend> AttnMixer<B> {
    fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 4, Bool>>) -> Tensor<B, 3> {
        let [b, n, d] = x.dims();
        let head_dim = d / self.n_heads;
        let split = |t: Tensor<B, 3>| t.reshape([b, n, self.n_heads, head_dim]).swap_dims(1, 2);

        let q = split(self.query.forward(x.clone()));
        let k = split(self.key.forward(x.clone()));
        let v = split(self.value.forward(x));

        let scores = q
            .matmul(k.swap_dims(2, 3))
            .div_scalar((head_dim as f32).sqrt());
        let scores = match mask {
            Some(m) => scores.mask_fill(m, f32::NEG_INFINITY),
            None => scores,
        };
        let attn = softmax(scores, 3);
        let out = attn.matmul(v).swap_dims(1, 2).reshape([b, n, d]);
        self.out.forward(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Inner, MlDevice};

    // ------------------------------------------------------------------
    // Naive per-token recurrence — the accuracy anchor the chunkwise scan
    // is transcribed against.
    // ------------------------------------------------------------------

    /// Plain nested-`Vec<f64>` recurrence:
    /// `S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T`,
    /// `o_t = S_t^T q_t`. Applies the same `K^-1/2` query scale `chunk_kda` does.
    ///
    /// Indexes `s[d][e]` by both `d` and `e` in lockstep with other same-shaped
    /// arrays throughout — an iterator rewrite of any one loop wouldn't carry the
    /// others along, so this stays index-based for clarity as a reference
    /// implementation.
    #[allow(clippy::needless_range_loop)]
    fn naive_recurrence(
        q: &[Vec<f64>],
        k: &[Vec<f64>],
        v: &[Vec<f64>],
        g: &[Vec<f64>], // raw per-step log-decay (not cumulative)
        beta: &[f64],
        dk: usize,
        dv: usize,
    ) -> Vec<Vec<f64>> {
        let t = q.len();
        let scale = (dk as f64).powf(-0.5);
        let mut s = vec![vec![0f64; dv]; dk]; // [dk, dv]
        let mut out = Vec::with_capacity(t);
        for step in 0..t {
            let alpha: Vec<f64> = g[step].iter().map(|&gi| gi.exp()).collect();
            // Diag(alpha) * S
            for d in 0..dk {
                for e in 0..dv {
                    s[d][e] *= alpha[d];
                }
            }
            // (I - beta k k^T) * (previous result), i.e. subtract beta*k*(k^T S)
            let kt_s: Vec<f64> = (0..dv)
                .map(|e| (0..dk).map(|d| k[step][d] * s[d][e]).sum())
                .collect();
            for d in 0..dk {
                for e in 0..dv {
                    s[d][e] -= beta[step] * k[step][d] * kt_s[e];
                }
            }
            // + beta * k * v^T
            for d in 0..dk {
                for e in 0..dv {
                    s[d][e] += beta[step] * k[step][d] * v[step][e];
                }
            }
            // o_t = S^T q_t
            let qs: Vec<f64> = q[step].iter().map(|&x| x * scale).collect();
            let o: Vec<f64> = (0..dv)
                .map(|e| (0..dk).map(|d| s[d][e] * qs[d]).sum())
                .collect();
            out.push(o);
        }
        out
    }

    /// Deterministic pseudo-random f32 in `(lo, hi)` from an index (no `rand` dep
    /// needed for a fixed, reproducible fixture).
    fn pseudo(seed: u64, lo: f32, hi: f32) -> f32 {
        // xorshift64
        let mut x = seed ^ 0x9E3779B97F4A7C15;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let frac = (x % 1_000_003) as f32 / 1_000_003.0;
        lo + frac * (hi - lo)
    }

    struct Fixture {
        q: Vec<Vec<f64>>,
        k: Vec<Vec<f64>>,
        v: Vec<Vec<f64>>,
        g: Vec<Vec<f64>>,
        beta: Vec<f64>,
    }

    /// One head's worth of deterministic test data. `q`/`k` are L2-normalised (as the
    /// real mixer's would be); `g` is a per-channel log-decay with `alpha in (0.8,
    /// 1.0)`; `beta in (0,1)`.
    fn fixture(t: usize, dk: usize, dv: usize, tag: u64) -> Fixture {
        let mut q = Vec::with_capacity(t);
        let mut k = Vec::with_capacity(t);
        let mut v = Vec::with_capacity(t);
        let mut g = Vec::with_capacity(t);
        let mut beta = Vec::with_capacity(t);
        let mut ctr = tag * 100_003;
        for _ in 0..t {
            let mut qi: Vec<f64> = (0..dk)
                .map(|_| {
                    ctr += 1;
                    pseudo(ctr, -1.0, 1.0) as f64
                })
                .collect();
            let mut ki: Vec<f64> = (0..dk)
                .map(|_| {
                    ctr += 1;
                    pseudo(ctr, -1.0, 1.0) as f64
                })
                .collect();
            let norm_q = qi.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-9);
            let norm_k = ki.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-9);
            for x in qi.iter_mut() {
                *x /= norm_q;
            }
            for x in ki.iter_mut() {
                *x /= norm_k;
            }
            let vi: Vec<f64> = (0..dv)
                .map(|_| {
                    ctr += 1;
                    pseudo(ctr, -1.0, 1.0) as f64
                })
                .collect();
            let gi: Vec<f64> = (0..dk)
                .map(|_| {
                    ctr += 1;
                    let alpha = pseudo(ctr, 0.8, 1.0) as f64;
                    alpha.ln()
                })
                .collect();
            ctr += 1;
            let b = pseudo(ctr, 0.05, 0.95) as f64;
            q.push(qi);
            k.push(ki);
            v.push(vi);
            g.push(gi);
            beta.push(b);
        }
        Fixture { q, k, v, g, beta }
    }

    fn to_tensor(rows: &[Vec<f64>]) -> (Tensor<Inner, 3>, usize, usize) {
        let t = rows.len();
        let d = rows[0].len();
        let flat: Vec<f32> = rows.iter().flatten().map(|&x| x as f32).collect();
        let device = MlDevice::default();
        (
            Tensor::<Inner, 3>::from_data(TensorData::new(flat, [1, t, d]), &device),
            t,
            d,
        )
    }

    fn run_chunked(f: &Fixture, chunk_size: usize) -> Vec<Vec<f32>> {
        let (q, t, _dk) = to_tensor(&f.q);
        let (k, _, _) = to_tensor(&f.k);
        let (v, _, dv) = to_tensor(&f.v);
        let (g, _, _) = to_tensor(&f.g);
        let device = MlDevice::default();
        let beta_flat: Vec<f32> = f.beta.iter().map(|&x| x as f32).collect();
        let beta = Tensor::<Inner, 2>::from_data(TensorData::new(beta_flat, [1, t]), &device);
        let o = chunk_kda::<Inner>(q, k, v, g, beta, chunk_size);
        let data: Vec<f32> = o.into_data().to_vec().unwrap();
        data.chunks(dv).map(|c| c.to_vec()).collect()
    }

    /// **The** accuracy anchor: the chunkwise WY/UT scan must reproduce the naive
    /// per-token recurrence, across multiple chunks (`T=96`, `C=32` → 3 full chunks).
    #[test]
    fn chunkwise_matches_naive_recurrence() {
        let (t, dk, dv, chunk) = (96, 8, 8, 32);
        let f = fixture(t, dk, dv, 1);
        let expect = naive_recurrence(&f.q, &f.k, &f.v, &f.g, &f.beta, dk, dv);
        let got = run_chunked(&f, chunk);

        assert_eq!(got.len(), expect.len());
        for (row_got, row_expect) in got.iter().zip(expect.iter()) {
            for (&g, &e) in row_got.iter().zip(row_expect.iter()) {
                let e = e as f32;
                let tol = 1e-4 + 1e-3 * e.abs();
                assert!(
                    (g - e).abs() < tol,
                    "mismatch: got {g}, expected {e} (diff {})",
                    (g - e).abs()
                );
            }
        }
    }

    /// Tail padding: `T=40` is not a multiple of `chunk_size=32` (one full chunk +
    /// a 8-frame remainder), so this exercises the zero-pad path specifically.
    #[test]
    fn tail_padding_matches_full_chunks() {
        let (t, dk, dv, chunk) = (40, 8, 8, 32);
        let f = fixture(t, dk, dv, 2);
        let expect = naive_recurrence(&f.q, &f.k, &f.v, &f.g, &f.beta, dk, dv);
        let got = run_chunked(&f, chunk);

        assert_eq!(got.len(), expect.len());
        for (row_got, row_expect) in got.iter().zip(expect.iter()) {
            for (&g, &e) in row_got.iter().zip(row_expect.iter()) {
                let e = e as f32;
                let tol = 1e-4 + 1e-3 * e.abs();
                assert!(
                    (g - e).abs() < tol,
                    "mismatch: got {g}, expected {e} (diff {})",
                    (g - e).abs()
                );
            }
        }
    }

    /// A window shorter than one chunk (`T=8 < chunk_size=64`, exactly the shape
    /// `small_song(8)` in `backbone.rs`'s shared contract test produces) must still
    /// match the naive recurrence — this is the padding path at its most extreme.
    #[test]
    fn shorter_than_one_chunk_matches_naive_recurrence() {
        let (t, dk, dv, chunk) = (8, 4, 4, 64);
        let f = fixture(t, dk, dv, 3);
        let expect = naive_recurrence(&f.q, &f.k, &f.v, &f.g, &f.beta, dk, dv);
        let got = run_chunked(&f, chunk);

        assert_eq!(got.len(), expect.len());
        for (row_got, row_expect) in got.iter().zip(expect.iter()) {
            for (&g, &e) in row_got.iter().zip(row_expect.iter()) {
                let e = e as f32;
                let tol = 1e-4 + 1e-3 * e.abs();
                assert!((g - e).abs() < tol, "mismatch: got {g}, expected {e}");
            }
        }
    }

    #[test]
    fn kda_mixer_preserves_shape_and_is_finite() {
        let device = MlDevice::default();
        let mixer = KdaMixerConfig::new(32, 4)
            .with_chunk_size(16)
            .with_dropout(0.0)
            .init::<Inner>(&device);
        let x = Tensor::<Inner, 3>::from_data(
            TensorData::new(
                (0..2 * 20 * 32)
                    .map(|i| pseudo(i as u64, -1.0, 1.0))
                    .collect::<Vec<f32>>(),
                [2, 20, 32],
            ),
            &device,
        );
        let out = mixer.forward(x);
        assert_eq!(out.dims(), [2, 20, 32]);
        let data: Vec<f32> = out.into_data().to_vec().unwrap();
        assert!(
            data.iter().all(|v| v.is_finite()),
            "NaN/Inf in mixer output"
        );
    }

    #[test]
    fn kimi_encoder_preserves_shape_and_is_finite() {
        let device = MlDevice::default();
        let enc = KimiEncoderConfig::new(32, 64, 4, 4)
            .with_chunk_size(16)
            .with_dropout(0.0)
            .init::<Inner>(&device);
        let x = Tensor::<Inner, 3>::from_data(
            TensorData::new(
                (0..2 * 20 * 32)
                    .map(|i| pseudo(i as u64 + 7, -1.0, 1.0))
                    .collect::<Vec<f32>>(),
                [2, 20, 32],
            ),
            &device,
        );
        for causal in [false, true] {
            let out = enc.forward(x.clone(), causal);
            assert_eq!(out.dims(), [2, 20, 32]);
            let data: Vec<f32> = out.into_data().to_vec().unwrap();
            assert!(
                data.iter().all(|v| v.is_finite()),
                "NaN/Inf in encoder output (causal={causal})"
            );
        }
    }
}

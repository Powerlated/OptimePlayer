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
/// Log-decay added at a **document start** to reset the KDA state: `exp(-60) ≈
/// 8.8e-27`, an effective zero against O(1) state entries, while keeping the
/// chunk-local cumulative sums well inside f32 precision (a hard `-inf` or `-1e9`
/// would destroy the *differences* `g_r − g_c` the scan is built on). Expressing
/// the reset through the decay itself means the chunkwise scan needs no other
/// change and stays exactly transcribable against the naive recurrence.
const RESET_LOG_DECAY: f32 = -60.0;

// ============================================================================
// Docs — document structure for packed / padded sequences.
// ============================================================================

/// Per-position document structure of a packed (multi-song) or padded sequence,
/// as the trunk consumes it:
///
/// * `attn_docs` `[b, seq]` (Int) — document id per slot, `-1` = pad. Attention
///   layers get a **block-diagonal document mask**: a slot attends only within
///   its own document (pad rows attend to themselves so softmax stays finite).
/// * `kda_reset` `[b, seq]` — 1.0 at each document's **first** slot: the KDA
///   per-channel decay is driven to ~0 there ([`RESET_LOG_DECAY`]), wiping the
///   state so nothing leaks across an EOS boundary.
/// * `kda_valid` `[b, seq]` — 0.0 at pad slots: decay is forced to 1 and β to 0
///   there, making pad an **exact no-op** on the KDA state (the same trick the
///   scan's own tail padding uses), so state flows *through* pad unchanged —
///   which is what lets the generative layout's label half continue its song's
///   document across the mid-sequence pad.
///
/// The depthwise short convs are document-masked too (see [`Docs::doc_ids`]):
/// without that, boundary-crossing conv taps contaminate k/v at a document's
/// first 3 slots and the contamination persists through the recurrence state.
#[derive(Debug, Clone)]
pub struct Docs<B: Backend> {
    pub attn_docs: Tensor<B, 2, Int>,
    pub kda_reset: Tensor<B, 2>,
    pub kda_valid: Tensor<B, 2>,
    /// Same-document masks for short-conv lags 1..K−1, built and uploaded once
    /// per batch rather than once per lag in every KDA layer.
    conv_tap_masks: Vec<Tensor<B, 3>>,
    /// Raw ids (`b*seq`, row-major), retained for incremental-cache construction.
    pub doc_ids: Vec<i64>,
}

impl<B: Backend> Docs<B> {
    /// Build from flattened per-slot document ids (`b*seq`, `-1` = pad). A reset
    /// is the first occurrence of each distinct non-negative id in its row — so a
    /// document resumed after a pad gap (the generative label half) does **not**
    /// reset again.
    pub fn from_doc_ids(doc_ids: &[i64], b: usize, seq: usize, device: &B::Device) -> Self {
        assert_eq!(doc_ids.len(), b * seq);
        let mut reset = vec![0.0f32; b * seq];
        let mut valid = vec![0.0f32; b * seq];
        for bi in 0..b {
            let row = &doc_ids[bi * seq..(bi + 1) * seq];
            let mut seen: Vec<i64> = Vec::new();
            for (f, &d) in row.iter().enumerate() {
                if d >= 0 {
                    valid[bi * seq + f] = 1.0;
                    if !seen.contains(&d) {
                        seen.push(d);
                        reset[bi * seq + f] = 1.0;
                    }
                }
            }
        }
        let conv_tap_masks = (1..SHORT_CONV_KERNEL)
            .map(|lag| {
                let mask: Vec<f32> = (0..b * seq)
                    .map(|i| {
                        let (bi, t) = (i / seq, i % seq);
                        if t >= lag && doc_ids[bi * seq + t - lag] == doc_ids[bi * seq + t] {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect();
                Tensor::<B, 1>::from_data(TensorData::new(mask, [b * seq]), device)
                    .reshape([b, seq, 1])
            })
            .collect();
        Docs {
            attn_docs: Tensor::from_data(TensorData::new(doc_ids.to_vec(), [b, seq]), device),
            kda_reset: Tensor::from_data(TensorData::new(reset, [b, seq]), device),
            kda_valid: Tensor::from_data(TensorData::new(valid, [b, seq]), device),
            conv_tap_masks,
            doc_ids: doc_ids.to_vec(),
        }
    }

    /// `[b, 1, seq, seq]` bool mask, true where attention is **disallowed** by
    /// document structure: different documents, or a pad partner (pad rows keep
    /// their diagonal so softmax stays finite).
    fn attn_mask(&self) -> Tensor<B, 4, Bool> {
        let [b, n] = self.attn_docs.dims();
        let device = self.attn_docs.device();
        let di = self.attn_docs.clone().reshape([b, n, 1]);
        let dj = self.attn_docs.clone().reshape([b, 1, n]);
        let same = di.clone().equal(dj).float(); // [b, n, n]
        let nonneg = di.greater_equal_elem(0).float(); // [b, n, 1]
        let eye = Tensor::<B, 2>::eye(n, &device).reshape([1, n, n]);
        let allow = same * nonneg + eye;
        allow.lower_elem(0.5).reshape([b, 1, n, n])
    }
}

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

/// Incremental-decode cache for one [`KdaMixer`]: the recurrence state plus the
/// last `SHORT_CONV_KERNEL − 1` **pre-conv** projections of each branch (what the
/// depthwise causal convs need to produce the next output column).
#[derive(Debug, Clone)]
pub struct KdaLayerCache<B: Backend> {
    /// `[b, K−1, d]` tails of `lin(x)` for the q/k/v conv branches.
    conv_q: Tensor<B, 3>,
    conv_k: Tensor<B, 3>,
    conv_v: Tensor<B, 3>,
    /// Document id of each tail slot (`b·(K−1)`, row-major; `-2` = before the
    /// sequence). A step masks tail taps from other documents, matching
    /// [`doc_masked_conv`].
    tail_docs: Vec<i64>,
    /// `[b·h, d_k, d_v]` recurrence state after the cached prefix.
    state: Tensor<B, 3>,
}

impl<B: Backend> KdaMixer<B> {
    /// `x`: `[batch, seq, d_model]` → same shape. KDA is causal by construction, so
    /// unlike the attention layers there is no `causal` argument here. `docs`
    /// injects document structure: state reset at doc starts, exact no-op at pad.
    pub fn forward(&self, x: Tensor<B, 3>, docs: Option<&Docs<B>>) -> Tensor<B, 3> {
        self.forward_inner(x, docs, false).0
    }

    /// [`Self::forward`] that also captures a [`KdaLayerCache`] for incremental
    /// decoding of a continuation.
    pub fn forward_prefill(
        &self,
        x: Tensor<B, 3>,
        docs: Option<&Docs<B>>,
    ) -> (Tensor<B, 3>, KdaLayerCache<B>) {
        let (o, cache) = self.forward_inner(x, docs, true);
        (o, cache.expect("capture requested"))
    }

    fn forward_inner(
        &self,
        x: Tensor<B, 3>,
        docs: Option<&Docs<B>>,
        capture: bool,
    ) -> (Tensor<B, 3>, Option<KdaLayerCache<B>>) {
        let [b, seq, d] = x.dims();
        let h = self.n_heads;
        let dh = self.head_dim;
        let km1 = SHORT_CONV_KERNEL - 1;

        // Pre-conv projections, kept separate so a prefill can cache their tails.
        let q_lin = self.q_proj.forward(x.clone());
        let k_lin = self.k_proj.forward(x.clone());
        let v_lin = self.v_proj.forward(x.clone());
        // With document structure, taps reaching across a boundary are zeroed
        // (doc-local convs); without it, the plain module path (bit-identical to
        // the historic forward).
        let conv_branch = |conv: &Conv1d<B>, u: &Tensor<B, 3>| -> Tensor<B, 3> {
            match docs {
                None => conv.forward(u.clone().swap_dims(1, 2)).swap_dims(1, 2),
                Some(dd) => doc_masked_conv(conv, u, dd),
            }
        };
        let q = silu(conv_branch(&self.conv_q, &q_lin));
        let k = silu(conv_branch(&self.conv_k, &k_lin));
        let v = silu(conv_branch(&self.conv_v, &v_lin));

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
        let mut g = softplus(raw_alpha, 1.0).mul(decay_scale).neg();
        let mut beta = sigmoid(self.beta_proj.forward(x.clone())); // [b, seq, h]

        // Document structure, expressed through the recurrence's own knobs:
        // pad → α=1 (g·valid) and β=0 (exact state no-op); doc start → α≈0
        // (g + RESET_LOG_DECAY), wiping the state before the new doc's first update.
        if let Some(docs) = docs {
            let valid = docs.kda_valid.clone().reshape([b, seq, 1]);
            let reset = docs.kda_reset.clone().reshape([b, seq, 1]);
            g = g.mul(valid.clone()) + reset.mul_scalar(RESET_LOG_DECAY);
            beta = beta.mul(valid);
        }
        let g = to_heads(g, dh);
        let beta = beta.swap_dims(1, 2).reshape([b * h, seq]);

        let (o, state) = chunk_kda(q, k, v, g, beta, self.chunk_size, None); // [b*h, seq, dh]

        // Headwise RMSNorm: reshape to expose (b, seq, h, dh) so the shared-gamma norm
        // runs per head, then flatten back for the gate + output projection.
        let o = o
            .reshape([b, h, seq, dh])
            .swap_dims(1, 2)
            .reshape([b, seq, h, dh]);
        let o = self.norm.forward(o).reshape([b, seq, d]);

        let gate = sigmoid(self.gate_up.forward(self.gate_down.forward(x.clone())));
        let out = self.o_proj.forward(self.dropout.forward(gate.mul(o)));

        let cache = capture.then(|| {
            // Conv tails: the last K−1 pre-conv columns, left-padded with zeros for
            // sequences shorter than that (matching the conv's own zero padding).
            let tail = |u: &Tensor<B, 3>| -> Tensor<B, 3> {
                if seq >= km1 {
                    u.clone().slice([0..b, seq - km1..seq, 0..d])
                } else {
                    Tensor::cat(
                        vec![Tensor::zeros([b, km1 - seq, d], &u.device()), u.clone()],
                        1,
                    )
                }
            };
            let tail_docs: Vec<i64> = (0..b)
                .flat_map(|bi| {
                    (0..km1).map(move |i| {
                        // Tail slot i corresponds to sequence position seq-km1+i.
                        let t = (seq + i).checked_sub(km1);
                        match (t, docs) {
                            (Some(t), Some(dd)) if t < seq => dd.doc_ids[bi * seq + t],
                            (Some(t), None) if t < seq => 0,
                            _ => -2, // before the sequence: never a valid tap
                        }
                    })
                })
                .collect();
            KdaLayerCache {
                conv_q: tail(&q_lin),
                conv_k: tail(&k_lin),
                conv_v: tail(&v_lin),
                tail_docs,
                state,
            }
        });
        (out, cache)
    }

    /// One-token incremental step (`x_t`: `[b, 1, d]`), advancing the cache. The
    /// slot is assumed to be an ordinary in-document token (no reset, not pad) —
    /// exactly what the generative decode's label slots are. `doc_t` (per row)
    /// masks conv taps from other documents, matching [`doc_masked_conv`].
    pub fn step(
        &self,
        x_t: Tensor<B, 3>,
        doc_t: &[i64],
        cache: &mut KdaLayerCache<B>,
    ) -> Tensor<B, 3> {
        let [b, _one, d] = x_t.dims();
        let h = self.n_heads;
        let dh = self.head_dim;
        let km1 = SHORT_CONV_KERNEL - 1;

        // Same-document mask over the cached tail slots (`[b, K−1, 1]`).
        let tap_mask: Vec<f32> = (0..b * km1)
            .map(|i| {
                if cache.tail_docs[i] == doc_t[i / km1] {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let tap_mask =
            Tensor::<B, 1>::from_data(TensorData::new(tap_mask, [b * km1]), &x_t.device())
                .reshape([b, km1, 1]);

        // Conv over exactly one kernel window (masked tail ++ current): with the
        // module's left padding the *last* output column is the padding-free
        // causal one, and zeroing a tap's input equals zeroing its contribution
        // (the conv is linear). The stored tail keeps the *true* values.
        let conv_step =
            |lin: &Linear<B>, conv: &Conv1d<B>, tail: &mut Tensor<B, 3>| -> Tensor<B, 3> {
                let u_t = lin.forward(x_t.clone()); // [b, 1, d]
                let masked_tail = tail.clone().mul(tap_mask.clone());
                let window = Tensor::cat(vec![masked_tail, u_t.clone()], 1); // [b, K, d]
                let full = SHORT_CONV_KERNEL;
                *tail = Tensor::cat(vec![tail.clone().slice([0..b, 1..km1, 0..d]), u_t], 1);
                let y = conv.forward(window.swap_dims(1, 2)).swap_dims(1, 2); // [b, K, d]
                y.slice([0..b, full - 1..full, 0..d])
            };
        let q = silu(conv_step(&self.q_proj, &self.conv_q, &mut cache.conv_q));
        let k = silu(conv_step(&self.k_proj, &self.conv_k, &mut cache.conv_k));
        let v = silu(conv_step(&self.v_proj, &self.conv_v, &mut cache.conv_v));
        for (bi, &dt) in doc_t.iter().enumerate().take(b) {
            cache
                .tail_docs
                .copy_within(bi * km1 + 1..(bi + 1) * km1, bi * km1);
            cache.tail_docs[(bi + 1) * km1 - 1] = dt;
        }

        let to_heads = |t: Tensor<B, 3>, dim: usize| -> Tensor<B, 3> {
            t.reshape([b, 1, h, dim])
                .swap_dims(1, 2)
                .reshape([b * h, 1, dim])
        };
        let q = l2_norm_last(to_heads(q, dh));
        let k = l2_norm_last(to_heads(k, dh));
        let v = to_heads(v, dh);

        let raw_alpha = self.alpha_up.forward(self.alpha_down.forward(x_t.clone()));
        let decay_scale = self.a_log.val().exp().reshape([1, 1, d]);
        let g = softplus(raw_alpha, 1.0).mul(decay_scale).neg();
        let alpha = to_heads(g, dh).exp(); // [bh, 1, dk]
        let beta = sigmoid(self.beta_proj.forward(x_t.clone()))
            .swap_dims(1, 2)
            .reshape([b * h, 1, 1]);

        // The naive recurrence, one step, batched over heads:
        // S ← Diag(α)S;  S ← S − β·k(kᵀS);  S ← S + β·k·vᵀ;  o = Sᵀq·K^{-1/2}.
        let bh = b * h;
        let s = cache.state.clone().mul(alpha.reshape([bh, dh, 1]));
        let k_col = k.clone().reshape([bh, dh, 1]);
        let kt_s = k.clone().matmul(s.clone()); // [bh, 1, dv]
        let s = s - k_col.clone().matmul(kt_s).mul(beta.clone());
        let s = s + k_col.matmul(v).mul(beta);
        cache.state = s.clone();
        let o = q.mul_scalar((dh as f64).powf(-0.5)).matmul(s); // [bh, 1, dh]

        let o = o
            .reshape([b, h, 1, dh])
            .swap_dims(1, 2)
            .reshape([b, 1, h, dh]);
        let o = self.norm.forward(o).reshape([b, 1, d]);
        let gate = sigmoid(self.gate_up.forward(self.gate_down.forward(x_t)));
        self.o_proj.forward(self.dropout.forward(gate.mul(o)))
    }
}

/// Depthwise causal conv with **per-lag same-document tap masking**: output
/// `y[t] = b + Σ_j w_j · u[t−(K−1−j)]`, where a tap is kept only if its source
/// slot exists and belongs to `t`'s document. Exactly the module's conv when no
/// boundary is in reach (the conv is linear, so zeroing a tap's input is zeroing
/// its contribution).
fn doc_masked_conv<B: Backend>(conv: &Conv1d<B>, u: &Tensor<B, 3>, docs: &Docs<B>) -> Tensor<B, 3> {
    let [b, seq, d] = u.dims();
    let device = u.device();
    let kernel = SHORT_CONV_KERNEL;
    let w = conv.weight.val(); // [d, 1, K] (depthwise)

    let mut acc = Tensor::<B, 3>::zeros([b, seq, d], &device);
    for j in 0..kernel {
        let lag = kernel - 1 - j;
        let wj = w.clone().slice([0..d, 0..1, j..j + 1]).reshape([1, 1, d]);
        let shifted = if lag == 0 {
            u.clone()
        } else {
            Tensor::cat(
                vec![
                    Tensor::zeros([b, lag, d], &device),
                    u.clone().slice([0..b, 0..seq - lag, 0..d]),
                ],
                1,
            )
        };
        let term = shifted.mul(wj);
        if lag == 0 {
            acc = acc + term;
        } else {
            acc = acc + term.mul(docs.conv_tap_masks[lag - 1].clone());
        }
    }
    match &conv.bias {
        Some(bias) => acc + bias.val().reshape([1, 1, d]),
        None => acc,
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
    state0: Option<Tensor<B, 3>>,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
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

    // Additive log-domain mask (data-independent, built once): `le_add[c,i] = −∞`
    // where `c ≤ i` (upper triangle *incl.* diagonal), else 0. Added to the decay
    // exponent *before* `exp` (so `exp(−∞)=0`): the masked, otherwise-overflow-prone
    // positive exponents are never materialised, and the backward pass stays finite
    // (grad `exp(−∞)=0`). Shape `[1,c,c,1]` → broadcasts over the `[bh,c,c,dk]`
    // exponent, so no `[bh,c,c,dk]` bool mask is built. Only the ≤ mask exists: a2's
    // diagonal (where `diff = 0` ⇒ `exp = 1`) is restored via `eye` instead of a
    // second full-size masked exp.
    let le_add = {
        let m = tri_mask::<B>(c, false, &device).reshape([1, c, c, 1]);
        Tensor::<B, 4>::zeros([1, c, c, 1], &device).mask_fill(m, f32::NEG_INFINITY)
    };
    let eye = Tensor::<B, 2>::eye(c, &device).reshape([1, c, c]);

    let mut state = state0.unwrap_or_else(|| Tensor::<B, 3>::zeros([bh, dk, dv], &device));
    let mut outputs = Vec::with_capacity(n);

    for i in 0..n {
        let q_i = take3(&q, i, dk);
        let k_i = take3(&k, i, dk);
        let v_i = take3(&v, i, dv);
        let g_raw = take3(&g, i, dk);
        let beta_i = take2(&beta, i); // [bh, c]
        let g_i = g_raw.cumsum(1); // chunk-local cumulative log-decay, <= 0
        let g_exp = g_i.clone().exp();

        // Shared decayed-pairwise structure: `diff[c,i',d] = g_i[c,d] − g_i[i',d]` (≤ 0
        // in every valid region, since g_i is non-increasing) drives BOTH the WY inverse
        // (a0, rows = k) and the causal output attention (a2, rows = q) — same decay, so
        // the strictly-lower masked `exp(diff)` (the dominant `[bh,c,c,dk]` elementwise
        // op) is formed **once** and reused; a2's diagonal (`diff = 0` ⇒ `exp = 1`,
        // entry = `q·k`) is added back via `eye`. `col` operand is `k[i']` for both.
        let g_r = g_i.clone().reshape([bh, c, 1, dk]);
        let g_c = g_i.clone().reshape([bh, 1, c, dk]);
        let decay = (g_r - g_c + le_add.clone()).exp(); // [bh,c,c,dk], strictly lower
        let k_c = k_i.clone().reshape([bh, 1, c, dk]);
        let dot = |rows: Tensor<B, 4>| -> Tensor<B, 3> {
            (rows * decay.clone() * k_c.clone())
                .sum_dim(3)
                .reshape([bh, c, c])
        };

        // --- WY/UT inverse transform for this chunk. ---
        // a0[c,i'] = Σ_d k[c,d]·exp(diff)·k[i',d], strictly-lower (masked `c ≤ i'`).
        let a0 = dot(k_i.clone().reshape([bh, c, 1, dk]));
        // a2[c,i'] = Σ_d q[c,d]·exp(diff)·k[i',d], causal *incl.* diagonal: strict-lower
        // part from the shared decay + `q·k` on the diagonal.
        let qk_diag = (q_i.clone() * k_i.clone()).sum_dim(2); // [bh,c,1]
        let a2 = dot(q_i.clone().reshape([bh, c, 1, dk])) + eye.clone().mul(qk_diag);

        let a = a0.mul(beta_i.clone().reshape([bh, c, 1])).neg(); // A = −a0·β, strictly lower

        // WY/UT inverse: the transform matrix is `(I − A)⁻¹ ⊙ β_col`. `A` is strictly
        // lower-triangular ⇒ `(I − A)` unit lower-triangular, inverted by
        // [`unit_lower_inverse`] (recursive 2×2 block inversion): numerically stable and
        // GPU-friendly. See that fn for why doubling/Neumann is NOT used here.
        let a = unit_lower_inverse(a).mul(beta_i.clone().reshape([bh, 1, c])); // column-wise β

        let w = a.clone().matmul(g_exp.clone().mul(k_i.clone())); // [bh,c,dk]
        let u = a.matmul(v_i.clone()); // [bh,c,dv]

        // --- Per-chunk causal output + state update (a2 formed above). ---
        let v_eff = u - w.matmul(state.clone()); // [bh,c,dv]
        let o_chunk =
            q_i.clone().mul(g_exp.clone()).matmul(state.clone()) + a2.matmul(v_eff.clone());
        outputs.push(o_chunk);

        // State carry: decay the old state to the end of the chunk, then add this
        // chunk's contribution decayed *forward* to the same point — every exponent
        // here is `g_last - g_c <= 0` (g is non-increasing within the chunk), so this
        // is safe without masking.
        let g_last = g_i.clone().slice([0..bh, c - 1..c, 0..dk]); // [bh,1,dk]
        let g_last_exp = g_exp.slice([0..bh, c - 1..c, 0..dk]);
        let decay_to_end = (g_last.clone() - g_i).exp().mul(k_i); // [bh,c,dk]
        state =
            state.mul(g_last_exp.reshape([bh, dk, 1])) + decay_to_end.swap_dims(1, 2).matmul(v_eff);
    }

    let o = Tensor::cat(outputs, 1); // [bh, seq_pad, dv]
                                     // The tail padding is a no-op on the recurrence, so `state` after the last
                                     // chunk *is* the state after the last real token — safe to hand to a decoder.
    (o.slice([0..bh, 0..seq, 0..dv]), state)
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

/// `(I − A)⁻¹` for a **strictly-lower-triangular** `A` (`[bh, c, c]`), by recursive
/// 2×2 block inversion of the unit lower-triangular `L = I − A`:
///
/// ```text
/// L = ⎡L₁₁  0 ⎤   L⁻¹ = ⎡    X₁₁      0  ⎤   X₁₁ = L₁₁⁻¹,  X₂₂ = L₂₂⁻¹,
///     ⎣L₂₁ L₂₂⎦         ⎣ X₂₂·A₂₁·X₁₁  X₂₂⎦   (L₂₁ = −A₂₁)
/// ```
///
/// recursing on the two diagonal blocks to a `1×1` base (`L=[1]`, inverse `[1]`).
/// The two sub-blocks are the **same size** (`c` is padded to a power of two, exact:
/// the pad region contributes an identity block with zero coupling), so each level
/// recurses **once on a batch-stacked pair** — O(log c) serial depth *and* O(log c)
/// kernel launches, not O(c) separate calls.
///
/// **Why not Neumann/doubling** (`(I−A)⁻¹ = Σ_{j<c} Aʲ`, since `A` is nilpotent): the
/// partial powers `Aʲ` grow ~`cʲ` and **overflow f32** (around `j≈31` for `c=64` with
/// `|A|~O(1)`) *before* nilpotency (`Aᶜ=0`) cancels them → inf·0 → **NaN** in training.
/// Block inversion only ever forms the true inverse's (bounded) sub-blocks, so it stays
/// finite exactly when the reference recurrence does. Tiny test chunks don't reach the
/// overflow, so the transcription test cannot see this — it is a scale-only failure.
fn unit_lower_inverse<B: Backend>(a: Tensor<B, 3>) -> Tensor<B, 3> {
    let device = a.device();
    let [bh, c, _] = a.dims();
    let p = c.next_power_of_two();
    if p == c {
        return unit_lower_inverse_pow2(a);
    }
    // Zero-pad to p×p: the pad block of L = I − A_pad is an identity with zero
    // coupling, so the padded inverse's top-left c×c block *is* the true inverse.
    let a = Tensor::cat(vec![a, Tensor::zeros([bh, p - c, c], &device)], 1);
    let a = Tensor::cat(vec![a, Tensor::zeros([bh, p, p - c], &device)], 2);
    unit_lower_inverse_pow2(a).slice([0..bh, 0..c, 0..c])
}

/// [`unit_lower_inverse`] body for power-of-two `c`: the two same-size diagonal
/// sub-blocks are stacked along the batch axis and recursed as **one** call.
fn unit_lower_inverse_pow2<B: Backend>(a: Tensor<B, 3>) -> Tensor<B, 3> {
    let device = a.device();
    let [bh, c, _] = a.dims();
    if c == 1 {
        // L = [[1]] (A strictly lower ⇒ A = 0); inverse is the identity.
        return Tensor::ones([bh, 1, 1], &device);
    }
    let m = c / 2;
    let a11 = a.clone().slice([0..bh, 0..m, 0..m]);
    let a22 = a.clone().slice([0..bh, m..c, m..c]);
    let a21 = a.slice([0..bh, m..c, 0..m]); // [bh, m, m]
    let x = unit_lower_inverse_pow2(Tensor::cat(vec![a11, a22], 0)); // [2bh, m, m]
    let x11 = x.clone().slice([0..bh, 0..m, 0..m]);
    let x22 = x.slice([bh..2 * bh, 0..m, 0..m]);
    let x21 = x22.clone().matmul(a21).matmul(x11.clone()); // X₂₂·A₂₁·X₁₁, [bh, m, m]
    let top = Tensor::cat(vec![x11, Tensor::zeros([bh, m, m], &device)], 2); // [bh, m, c]
    let bot = Tensor::cat(vec![x21, x22], 2); // [bh, m, c]
    Tensor::cat(vec![top, bot], 1) // [bh, c, c]
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

/// Incremental-decode cache for a whole [`KimiEncoder`]: one entry per layer.
#[derive(Debug, Clone)]
pub struct EncoderCache<B: Backend> {
    layers: Vec<LayerCache<B>>,
}

#[derive(Debug, Clone)]
enum LayerCache<B: Backend> {
    Kda(KdaLayerCache<B>),
    Attn(AttnLayerCache<B>),
}

impl<B: Backend> KimiEncoder<B> {
    /// `x`: `[batch, seq, d_model]` → same shape. `causal` affects **only** the
    /// full-attention layers — KDA is causal by construction regardless. `docs`
    /// adds document structure (block-diagonal attention, KDA reset at doc starts,
    /// state no-op at pad) for packed / padded sequences.
    pub fn forward(&self, x: Tensor<B, 3>, causal: bool, docs: Option<&Docs<B>>) -> Tensor<B, 3> {
        let mask = self.attn_mask(&x, causal, docs);
        let mut h = x;
        for layer in &self.layers {
            h = layer.forward(h, mask.clone(), docs);
        }
        self.norm.forward(h)
    }

    /// [`Self::forward`] that also captures an [`EncoderCache`] so a continuation
    /// can be decoded one token at a time with [`Self::forward_step`]. The prefix
    /// must be causal (it is the read half of a generative decode).
    pub fn forward_prefill(
        &self,
        x: Tensor<B, 3>,
        docs: Option<&Docs<B>>,
    ) -> (Tensor<B, 3>, EncoderCache<B>) {
        let mask = self.attn_mask(&x, true, docs);
        let mut h = x;
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (out, cache) = layer.forward_prefill(h, mask.clone(), docs);
            h = out;
            layers.push(cache);
        }
        (self.norm.forward(h), EncoderCache { layers })
    }

    /// One-token continuation step (`x_t`: `[b, 1, d]`), advancing the cache.
    /// `doc_t` is the token's document id per batch row (`[b]`) — attention layers
    /// only look at cached positions of the same document.
    pub fn forward_step(
        &self,
        x_t: Tensor<B, 3>,
        doc_t: &[i64],
        cache: &mut EncoderCache<B>,
    ) -> Tensor<B, 3> {
        let mut h = x_t;
        for (layer, lc) in self.layers.iter().zip(cache.layers.iter_mut()) {
            h = layer.forward_step(h, doc_t, lc);
        }
        self.norm.forward(h)
    }

    /// Combined causal + document attention mask (`None` when neither applies).
    fn attn_mask(
        &self,
        x: &Tensor<B, 3>,
        causal: bool,
        docs: Option<&Docs<B>>,
    ) -> Option<Tensor<B, 4, Bool>> {
        let [b, n, _] = x.dims();
        let device = x.device();
        let causal_m = causal.then(|| causal_mask::<B>(b, self.n_heads, n, &device));
        let doc_m = docs.map(|d| d.attn_mask().repeat_dim(1, self.n_heads));
        match (causal_m, doc_m) {
            (Some(c), Some(d)) => Some(bool_or(c, d)),
            (Some(c), None) => Some(c),
            (None, Some(d)) => Some(d),
            (None, None) => None,
        }
    }
}

/// Elementwise OR of two bool masks (burn has no direct bool-or on `Bool` tensors;
/// go through float addition).
fn bool_or<B: Backend, const D: usize>(
    a: Tensor<B, D, Bool>,
    b: Tensor<B, D, Bool>,
) -> Tensor<B, D, Bool> {
    (a.float() + b.float()).greater_elem(0.5)
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
    fn forward(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 4, Bool>>,
        docs: Option<&Docs<B>>,
    ) -> Tensor<B, 3> {
        let normed = self.norm_mixer.forward(x.clone());
        let mixed = match &self.mixer {
            MixerKind::Kda(m) => m.forward(normed, docs),
            MixerKind::Attn(m) => m.forward(normed, mask),
        };
        self.finish(x, mixed)
    }

    fn forward_prefill(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 4, Bool>>,
        docs: Option<&Docs<B>>,
    ) -> (Tensor<B, 3>, LayerCache<B>) {
        let normed = self.norm_mixer.forward(x.clone());
        let (mixed, cache) = match &self.mixer {
            MixerKind::Kda(m) => {
                let (o, c) = m.forward_prefill(normed, docs);
                (o, LayerCache::Kda(c))
            }
            MixerKind::Attn(m) => {
                let (o, c) = m.forward_prefill(normed, mask, docs);
                (o, LayerCache::Attn(c))
            }
        };
        (self.finish(x, mixed), cache)
    }

    fn forward_step(
        &self,
        x_t: Tensor<B, 3>,
        doc_t: &[i64],
        cache: &mut LayerCache<B>,
    ) -> Tensor<B, 3> {
        let normed = self.norm_mixer.forward(x_t.clone());
        let mixed = match (&self.mixer, cache) {
            (MixerKind::Kda(m), LayerCache::Kda(c)) => m.step(normed, doc_t, c),
            (MixerKind::Attn(m), LayerCache::Attn(c)) => m.step(normed, doc_t, c),
            _ => unreachable!("cache kind always matches its layer"),
        };
        self.finish(x_t, mixed)
    }

    /// Residual add + FFN sublayer (shared by all three paths).
    fn finish(&self, x: Tensor<B, 3>, mixed: Tensor<B, 3>) -> Tensor<B, 3> {
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

/// Incremental-decode cache for one [`AttnMixer`]: the cached keys/values and the
/// document id of every cached position (so a step can mask cross-document and
/// pad positions out of its attention).
#[derive(Debug, Clone)]
pub struct AttnLayerCache<B: Backend> {
    /// `[b, h, t, dh]`
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    /// Document id per cached position (`b*t`, row-major), `-1` = pad.
    docs: Vec<i64>,
}

impl<B: Backend> AttnMixer<B> {
    fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 4, Bool>>) -> Tensor<B, 3> {
        let [b, n, d] = x.dims();
        let head_dim = d / self.n_heads;
        let split = |t: Tensor<B, 3>| t.reshape([b, n, self.n_heads, head_dim]).swap_dims(1, 2);

        let q = split(self.query.forward(x.clone()));
        let k = split(self.key.forward(x.clone()));
        let v = split(self.value.forward(x));
        let out = self.attend(q, k, v, mask);
        self.out.forward(out.swap_dims(1, 2).reshape([b, n, d]))
    }

    fn forward_prefill(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 4, Bool>>,
        docs: Option<&Docs<B>>,
    ) -> (Tensor<B, 3>, AttnLayerCache<B>) {
        let [b, n, d] = x.dims();
        let head_dim = d / self.n_heads;
        let split = |t: Tensor<B, 3>| t.reshape([b, n, self.n_heads, head_dim]).swap_dims(1, 2);

        let q = split(self.query.forward(x.clone()));
        let k = split(self.key.forward(x.clone()));
        let v = split(self.value.forward(x));
        let doc_vec = match docs {
            Some(dd) => dd.attn_docs.clone().into_data().to_vec::<i64>().unwrap(),
            None => vec![0; b * n],
        };
        let cache = AttnLayerCache {
            k: k.clone(),
            v: v.clone(),
            docs: doc_vec,
        };
        let out = self.attend(q, k, v, mask);
        (
            self.out.forward(out.swap_dims(1, 2).reshape([b, n, d])),
            cache,
        )
    }

    /// One-token step: attend from `x_t` over the cached prefix + itself, masked
    /// to the token's own document.
    fn step(
        &self,
        x_t: Tensor<B, 3>,
        doc_t: &[i64],
        cache: &mut AttnLayerCache<B>,
    ) -> Tensor<B, 3> {
        let [b, _one, d] = x_t.dims();
        let head_dim = d / self.n_heads;
        let device = x_t.device();
        let split = |t: Tensor<B, 3>| t.reshape([b, 1, self.n_heads, head_dim]).swap_dims(1, 2);

        let q = split(self.query.forward(x_t.clone()));
        let k_t = split(self.key.forward(x_t.clone()));
        let v_t = split(self.value.forward(x_t));

        cache.k = Tensor::cat(vec![cache.k.clone(), k_t], 2);
        cache.v = Tensor::cat(vec![cache.v.clone(), v_t], 2);
        let t = cache.k.dims()[2];
        let t_old = t - 1;
        let mut new_docs = Vec::with_capacity(b * t);
        for (bi, &dt) in doc_t.iter().enumerate() {
            new_docs.extend_from_slice(&cache.docs[bi * t_old..(bi + 1) * t_old]);
            new_docs.push(dt);
        }
        cache.docs = new_docs;

        // Disallow cached positions from other documents (incl. pad, doc −1). The
        // token itself (last position, same doc) always stays visible.
        let disallow: Vec<f32> = (0..b * t)
            .map(|i| {
                let (bi, ti) = (i / t, i % t);
                if cache.docs[bi * t + ti] == doc_t[bi] {
                    0.0
                } else {
                    1.0
                }
            })
            .collect();
        let mask = Tensor::<B, 2>::from_data(TensorData::new(disallow, [b, t]), &device)
            .greater_elem(0.5)
            .reshape([b, 1, 1, t])
            .repeat_dim(1, self.n_heads);

        let out = self.attend(q, cache.k.clone(), cache.v.clone(), Some(mask));
        self.out.forward(out.swap_dims(1, 2).reshape([b, 1, d]))
    }

    /// Scaled-dot attention core (`q`: `[b,h,m,dh]`, `k`/`v`: `[b,h,t,dh]`).
    fn attend(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 4> {
        let head_dim = q.dims()[3];
        let scores = q
            .matmul(k.swap_dims(2, 3))
            .div_scalar((head_dim as f32).sqrt());
        let scores = match mask {
            Some(m) => scores.mask_fill(m, f32::NEG_INFINITY),
            None => scores,
        };
        softmax(scores, 3).matmul(v)
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
        let (o, _) = chunk_kda::<Inner>(q, k, v, g, beta, chunk_size, None);
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
        let out = mixer.forward(x, None);
        assert_eq!(out.dims(), [2, 20, 32]);
        let data: Vec<f32> = out.into_data().to_vec().unwrap();
        assert!(
            data.iter().all(|v| v.is_finite()),
            "NaN/Inf in mixer output"
        );
    }

    /// `unit_lower_inverse` at the **real chunk size** `c=64`: `(I − A)·inv` must be the
    /// identity and finite. This is the regression guard for the NaN that the removed
    /// Neumann/doubling inverse produced at scale — the small-`c` transcription tests
    /// above never reach the overflow regime, so this exercises the full block-inversion
    /// recursion (64→32→…→1) directly. `A` entries in `[−1, 1]` (the real bound: q/k are
    /// L2-normalised, so `|a0| ≤ 1`).
    #[test]
    fn unit_lower_inverse_is_identity_at_full_chunk() {
        let device = MlDevice::default();
        let c = 64;
        // Strictly-lower A with entries in (−1, 1).
        let mut data = vec![0f32; c * c];
        for row in 0..c {
            for col in 0..row {
                data[row * c + col] = pseudo((row * c + col) as u64 + 11, -1.0, 1.0);
            }
        }
        let a = Tensor::<Inner, 3>::from_data(TensorData::new(data, [1, c, c]), &device);
        let inv = unit_lower_inverse(a.clone());

        // (I − A) · inv  ==  I.
        let eye = {
            let mut e = vec![0f32; c * c];
            for i in 0..c {
                e[i * c + i] = 1.0;
            }
            Tensor::<Inner, 3>::from_data(TensorData::new(e, [1, c, c]), &device)
        };
        let l = eye.clone() - a;
        let prod = l.matmul(inv);
        let got: Vec<f32> = prod.into_data().to_vec().unwrap();
        let expect: Vec<f32> = eye.into_data().to_vec().unwrap();
        for (i, (&g, &e)) in got.iter().zip(expect.iter()).enumerate() {
            assert!(g.is_finite(), "non-finite inverse at {i}: {g}");
            assert!(
                (g - e).abs() < 1e-3,
                "(I−A)·inv ≠ I at {i}: got {g}, expected {e}"
            );
        }
    }

    /// The chunkwise scan across **two full `c=64` chunks** (`T=128`) must match the
    /// naive recurrence — the transcription anchor at the production chunk size, where
    /// the doubling inverse NaN'd.
    #[test]
    fn full_chunk_matches_naive_recurrence() {
        let (t, dk, dv, chunk) = (128, 16, 16, 64);
        let f = fixture(t, dk, dv, 4);
        let expect = naive_recurrence(&f.q, &f.k, &f.v, &f.g, &f.beta, dk, dv);
        let got = run_chunked(&f, chunk);
        assert_eq!(got.len(), expect.len());
        for (row_got, row_expect) in got.iter().zip(expect.iter()) {
            for (&g, &e) in row_got.iter().zip(row_expect.iter()) {
                let e = e as f32;
                let tol = 1e-4 + 1e-3 * e.abs();
                assert!(
                    g.is_finite() && (g - e).abs() < tol,
                    "got {g}, expected {e}"
                );
            }
        }
    }

    /// Document reset, expressed through the decay (`g += RESET_LOG_DECAY`), must
    /// reproduce a naive recurrence whose state is **hard-zeroed** at the reset
    /// positions — the transcription pin for the packed-sequence state reset.
    /// Includes a reset mid-chunk and one on a chunk boundary.
    #[test]
    #[allow(clippy::needless_range_loop)] // reference recurrence indexes arrays in lockstep
    fn decay_reset_matches_hard_state_zeroing() {
        let (t, dk, dv, chunk) = (48, 8, 8, 16);
        let f = fixture(t, dk, dv, 5);
        let resets = [0usize, 21, 32]; // doc starts (incl. boundary case 32 = 2*chunk)

        // Reference: naive recurrence with S zeroed before each reset step.
        let scale = (dk as f64).powf(-0.5);
        let mut s = vec![vec![0f64; dv]; dk];
        let mut expect = Vec::with_capacity(t);
        for step in 0..t {
            if resets.contains(&step) {
                s = vec![vec![0f64; dv]; dk];
            }
            let alpha: Vec<f64> = f.g[step].iter().map(|&gi| gi.exp()).collect();
            for d in 0..dk {
                for e in 0..dv {
                    s[d][e] *= alpha[d];
                }
            }
            let kt_s: Vec<f64> = (0..dv)
                .map(|e| (0..dk).map(|d| f.k[step][d] * s[d][e]).sum())
                .collect();
            for d in 0..dk {
                for e in 0..dv {
                    s[d][e] += f.beta[step] * f.k[step][d] * (f.v[step][e] - kt_s[e]);
                }
            }
            let o: Vec<f64> = (0..dv)
                .map(|e| (0..dk).map(|d| s[d][e] * f.q[step][d] * scale).sum())
                .collect();
            expect.push(o);
        }

        // Chunked scan with the reset folded into g.
        let mut f2 = Fixture {
            q: f.q.clone(),
            k: f.k.clone(),
            v: f.v.clone(),
            g: f.g.clone(),
            beta: f.beta.clone(),
        };
        for &r in &resets {
            for gd in f2.g[r].iter_mut() {
                *gd += RESET_LOG_DECAY as f64;
            }
        }
        let got = run_chunked(&f2, chunk);

        for (row_got, row_expect) in got.iter().zip(expect.iter()) {
            for (&g, &e) in row_got.iter().zip(row_expect.iter()) {
                let e = e as f32;
                let tol = 1e-4 + 1e-3 * e.abs();
                assert!((g - e).abs() < tol, "reset mismatch: got {g}, expected {e}");
            }
        }
    }

    /// The incremental decode path (prefill + per-token steps) must reproduce the
    /// full batched forward exactly — this is the KV-cache / KDA-state-cache
    /// correctness pin, covering the conv tails, the per-step recurrence, the
    /// attention KV append, and the document mask (a pad slot sits mid-prefix).
    #[test]
    fn incremental_decode_matches_full_forward() {
        let device = MlDevice::default();
        let d = 32;
        let enc = KimiEncoderConfig::new(d, 64, 4, 4)
            .with_chunk_size(8)
            .with_dropout(0.0)
            .init::<Inner>(&device);

        // 20-slot sequence: doc 0 (slots 0..8), EOS-ish, pad (slots 9..11, doc -1),
        // then doc 0 resumes (the generative label-half shape).
        let total = 20usize;
        let prefix = 12usize;
        let doc_ids: Vec<i64> = (0..total)
            .map(|i| if (9..12).contains(&i) { -1 } else { 0 })
            .collect();
        let x = Tensor::<Inner, 3>::from_data(
            TensorData::new(
                (0..2 * total * d)
                    .map(|i| pseudo(i as u64 + 99, -1.0, 1.0))
                    .collect::<Vec<f32>>(),
                [2, total, d],
            ),
            &device,
        );
        let docs = Docs::from_doc_ids(
            &doc_ids
                .iter()
                .cycle()
                .take(2 * total)
                .copied()
                .collect::<Vec<i64>>(),
            2,
            total,
            &device,
        );

        // Reference: one full causal pass over all 20 slots.
        let full = enc.forward(x.clone(), true, Some(&docs));

        // Incremental: prefill the first 12 slots, then step the remaining 8.
        let x_prefix = x.clone().slice([0..2, 0..prefix, 0..d]);
        let docs_prefix = Docs::from_doc_ids(
            &doc_ids[..prefix]
                .iter()
                .cycle()
                .take(2 * prefix)
                .copied()
                .collect::<Vec<i64>>(),
            2,
            prefix,
            &device,
        );
        let (h_prefix, mut cache) = enc.forward_prefill(x_prefix, Some(&docs_prefix));
        let mut steps = vec![h_prefix];
        #[allow(clippy::needless_range_loop)]
        for t in prefix..total {
            let x_t = x.clone().slice([0..2, t..t + 1, 0..d]);
            steps.push(enc.forward_step(x_t, &[doc_ids[t], doc_ids[t]], &mut cache));
        }
        let inc = Tensor::cat(steps, 1);

        let a: Vec<f32> = full.into_data().to_vec().unwrap();
        let b: Vec<f32> = inc.into_data().to_vec().unwrap();
        assert_eq!(a.len(), b.len());
        for (i, (&fa, &fb)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (fa - fb).abs() < 1e-3 + 1e-3 * fa.abs(),
                "incremental != full at {i}: {fb} vs {fa}"
            );
        }
    }

    /// Block-diagonal document masking: perturbing document A's tokens must not
    /// change document B's outputs at the attention layers *or* through the KDA
    /// state (which resets at B's start).
    #[test]
    fn doc_mask_isolates_documents() {
        let device = MlDevice::default();
        let d = 32;
        let enc = KimiEncoderConfig::new(d, 64, 4, 4)
            .with_chunk_size(8)
            .with_dropout(0.0)
            .init::<Inner>(&device);

        // Two docs: A = slots 0..6 (EOS at 6 belongs to A), B = 7..14, pad 15.
        let total = 16usize;
        let doc_ids: Vec<i64> = (0..total as i64)
            .map(|i| {
                if i <= 6 {
                    0
                } else if i <= 14 {
                    1
                } else {
                    -1
                }
            })
            .collect();
        let docs = Docs::from_doc_ids(&doc_ids, 1, total, &device);

        let base: Vec<f32> = (0..total * d)
            .map(|i| pseudo(i as u64 + 3, -1.0, 1.0))
            .collect();
        let mut perturbed = base.clone();
        for (i, p) in perturbed.iter_mut().take(3 * d).enumerate() {
            // Channel-varying perturbation of doc A's first three slots (a constant
            // shift would be erased exactly by the pre-norm LayerNorm).
            *p += 1.0 + ((i % 7) as f32) * 0.5;
        }
        let run = |data: Vec<f32>| -> Vec<f32> {
            let x = Tensor::<Inner, 3>::from_data(TensorData::new(data, [1, total, d]), &device);
            enc.forward(x, true, Some(&docs))
                .into_data()
                .to_vec()
                .unwrap()
        };
        let a = run(base);
        let b = run(perturbed);
        // Doc B's slots (7..=14) must be untouched, apart from the width-4 conv
        // leak at its first 3 slots (7, 8, 9) — check 10..=14.
        for slot in 10..15 {
            for c in 0..d {
                let i = slot * d + c;
                assert!(
                    (a[i] - b[i]).abs() < 1e-4,
                    "doc A perturbation leaked into doc B slot {slot}"
                );
            }
        }
        // Sanity: doc A's own outputs did change.
        assert!(
            (0..6 * d).any(|i| (a[i] - b[i]).abs() > 1e-3),
            "perturbation had no effect at all"
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
            let out = enc.forward(x.clone(), causal, None);
            assert_eq!(out.dims(), [2, 20, 32]);
            let data: Vec<f32> = out.into_data().to_vec().unwrap();
            assert!(
                data.iter().all(|v| v.is_finite()),
                "NaN/Inf in encoder output (causal={causal})"
            );
        }
    }
}

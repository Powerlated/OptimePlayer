//! Analytic FLOP estimates for one forward pass.
//!
//! **Convention.** One multiply-accumulate counts as 2 FLOPs, and only **matmuls**
//! are counted: the Q/K/V/O projections, the two attention products, the FFN, the
//! input projection, and the output heads. Softmax, LayerNorm, GELU, dropout,
//! residual adds, embedding gathers, and scatter-adds are elementwise or
//! memory-movement work rather than arithmetic on the matmul critical path, and are
//! excluded.
//!
//! So this is a **lower bound that tracks the real cost**, not a profile. Use it to
//! compare generations, window sizes, and configs against each other — not to
//! predict wall time. On a memory-bandwidth-bound machine (see [`crate::parallel`])
//! measured throughput will be far below what these numbers alone would suggest.

/// Dense matmul `[n, k] × [k, m]` = `2·n·k·m`.
pub fn matmul(n: usize, k: usize, m: usize) -> u64 {
    2 * n as u64 * k as u64 * m as u64
}

/// One pre-norm transformer encoder layer over `seq` tokens:
///
/// * Q/K/V/O projections — `8·seq·d²`
/// * attention `QKᵀ` then `AV` — `4·seq²·d` (the term that makes context quadratic)
/// * FFN up then down — `4·seq·d·d_ff`
///
/// Head count doesn't appear: heads repartition the same `d`, so the arithmetic is
/// unchanged. A causal mask doesn't either — the masked half is computed and then
/// discarded.
pub fn transformer_layer(seq: usize, d_model: usize, d_ff: usize) -> u64 {
    let proj = 4 * matmul(seq, d_model, d_model);
    let attn = matmul(seq, d_model, seq) + matmul(seq, seq, d_model);
    let ffn = matmul(seq, d_model, d_ff) + matmul(seq, d_ff, d_model);
    proj + attn + ffn
}

/// `n_layers` identical [`transformer_layer`]s.
pub fn transformer_encoder(n_layers: usize, seq: usize, d_model: usize, d_ff: usize) -> u64 {
    n_layers as u64 * transformer_layer(seq, d_model, d_ff)
}

/// One KDA (Kimi Delta Attention) mixer layer over `seq` frames, matmul-only lower
/// bound. `d_model` = total width, `chunk` = the chunkwise-scan chunk length.
///
/// Counted matmuls, per head-group (i.e. already summed over all heads, since a
/// head's `d_k`×`d_v` add up to `d_model` regardless of head count — same convention
/// as [`transformer_layer`]):
/// * q/k/v/gate/output projections — `5 * 2·seq·d_model²` (5 full `d_model→d_model`
///   affine maps: q, k, v, output gate's up-projection folded to `d_model` width as
///   an upper bound, and the final `W_o`).
/// * low-rank decay gate (`W_α↓`, `W_α↑`) — down to `head_dim`, back up: negligible
///   next to the above but included for completeness.
/// * the WY/UT transform's pairwise decayed dot products, two per chunk (the
///   within-chunk inverse-transform matrix and the causal output-stage matrix) —
///   `2 * (seq/chunk) * chunk² * d_model` FLOPs worth of contraction (every chunk
///   builds a `chunk × chunk` matrix contracting `d_model`-many channels, laid out
///   as one matmul per chunk in the real implementation).
/// * per-chunk `A @ (·)` products (`w`, `u`, state read/write) — `O(chunk² · d_model)`
///   per chunk, same order as the pairwise dot products, so folded into the same term
///   via a constant factor rather than re-derived term-by-term (this is a lower bound,
///   not a cycle-accurate count).
///
/// Quadratic in `seq` only through the chunk count (`seq/chunk` chunks, each
/// `O(chunk²)`) — i.e. linear in `seq` for fixed `chunk`, unlike full attention's
/// `seq²` term. That is the whole point of a linear-attention mixer.
pub fn kda_layer(seq: usize, d_model: usize, d_ff: usize, chunk: usize) -> u64 {
    let head_dim = chunk.min(d_model); // stand-in when the caller doesn't split heads out
    let proj = 5 * matmul(seq, d_model, d_model);
    let low_rank_gate = 2 * (matmul(seq, d_model, head_dim) + matmul(seq, head_dim, d_model));
    let n_chunks = seq.div_ceil(chunk).max(1);
    // Two `chunk×chunk` pairwise-dot builds + ~4 `chunk×chunk` matmuls (inverse-transform
    // apply, w, u, output/state) worth of contraction per chunk.
    let per_chunk = 6 * matmul(chunk, d_model, chunk);
    let chunkwise_scan = n_chunks as u64 * per_chunk;
    let ffn = matmul(seq, d_model, d_ff) + matmul(seq, d_ff, d_model);
    proj + low_rank_gate + chunkwise_scan + ffn
}

/// The shared per-frame chord heads + pooled key head every generation ends with.
pub fn chord_key_heads(seq: usize, d_model: usize) -> u64 {
    use crate::theory::{N_KEY_CLASSES, N_QUALITY_CLASSES, N_ROOT_CLASSES};
    matmul(seq, d_model, N_ROOT_CLASSES)
        + matmul(seq, d_model, N_QUALITY_CLASSES)
        // Key head sees one pooled vector per window, not one per frame.
        + matmul(1, d_model, N_KEY_CLASSES)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layer estimate must equal the closed form it documents.
    #[test]
    fn layer_matches_closed_form() {
        let (seq, d, d_ff) = (128usize, 128usize, 512usize);
        let expect = 8 * seq as u64 * (d * d) as u64
            + 4 * (seq * seq) as u64 * d as u64
            + 4 * (seq * d * d_ff) as u64;
        assert_eq!(transformer_layer(seq, d, d_ff), expect);
    }

    /// Attention is quadratic in context: doubling `seq` more than doubles the layer.
    #[test]
    fn attention_term_is_quadratic_in_context() {
        let one = transformer_layer(128, 128, 512);
        let two = transformer_layer(256, 128, 512);
        assert!(
            two > 2 * one,
            "quadratic term should dominate the linear ones"
        );
    }

    #[test]
    fn encoder_scales_with_depth() {
        assert_eq!(
            transformer_encoder(4, 128, 128, 512),
            4 * transformer_layer(128, 128, 512)
        );
    }

    /// KDA's chunkwise scan is **linear** in context for a fixed chunk length —
    /// every term (projections, FFN, and `n_chunks * per_chunk` with `n_chunks =
    /// seq/chunk`) scales linearly with `seq` when `chunk` evenly divides it, unlike
    /// full attention's quadratic `transformer_layer` pinned above.
    #[test]
    fn kda_layer_is_linear_in_context_for_fixed_chunk() {
        let one = kda_layer(128, 128, 512, 64);
        let two = kda_layer(256, 128, 512, 64);
        assert_eq!(
            two,
            2 * one,
            "kda cost should scale exactly linearly with seq"
        );
    }
}

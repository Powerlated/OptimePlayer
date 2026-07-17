//! Analytic FLOP estimates for one forward pass. 1 MAC = 2 FLOPs, **matmuls only** (softmax/norm/
//! GELU/gathers/scatter excluded) → a lower bound that tracks cost for comparing generations/window
//! sizes/configs, not a profile: the machine is memory-bandwidth-bound (see [`crate::parallel`]).

/// Dense matmul `[n, k] × [k, m]` = `2·n·k·m`.
pub fn matmul(n: usize, k: usize, m: usize) -> u64 {
    2 * n as u64 * k as u64 * m as u64
}

/// One pre-norm encoder layer over `seq` tokens: Q/K/V/O projections `8·seq·d²` + attention
/// `4·seq²·d` (the quadratic term) + FFN `4·seq·d·d_ff`. Head count and causal masking don't change
/// it (heads repartition `d`; the masked half is computed then discarded).
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
}

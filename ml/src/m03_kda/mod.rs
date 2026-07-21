//! **Generation 03** — Kimi Linear (KDA) hybrid trunk over [`crate::m01_event`]'s
//! frame tokens.
//!
//! Reuses generation 01's front-end verbatim (same seven field embeddings, φ,
//! scatter-add pooling) and its exact `EventBatchData` batch type (`type Batch =
//! EventBatchData`, not a copy), so the only thing this generation changes is the
//! trunk: [`crate::kda::KimiEncoder`] (3 KDA — Kimi Delta Attention — layers per 1
//! NoPE full-attention layer) in place of m01's pure-RoPE-attention `RopeEncoder`.
//!
//! The [`crate::backbone::Backbone`] / [`crate::backbone::ArBackbone`] impls live in
//! [`model`], next to the private fields they read.

mod model;

pub use model::{KdaKeyChordModel, KdaModelConfig};

/// Generation-03 model, named for the backbone it implements. A type alias, so it is
/// the same type as [`KdaKeyChordModel`] — persisted records are unaffected.
pub type KdaModel<B> = KdaKeyChordModel<B>;

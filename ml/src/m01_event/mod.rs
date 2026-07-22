//! **Generation 01** — learned frame tokens, pooled by a param-free scatter-add sum.
//!
//! Each frame token is the summed field embeddings of the notes onsetting in that
//! frame, passed through a nonlinear φ. The first generation to read the note stream
//! directly rather than a hand-engineered feature vector, and the first with an
//! autoregressive next-frame pretext ([`crate::pretrain::ar`]).
//!
//! The [`crate::backbone::Backbone`] / [`crate::backbone::ArBackbone`] impls live in
//! [`model`], next to the private fields they read.

mod batch;
mod model;

pub use batch::{EventBatchData, SLOT_EOS, SLOT_FRAME, SLOT_PAD};
pub use model::{EventKeyChordModel, EventModelConfig};

/// Generation-01 model, named for the backbone it implements. A type alias, so it is
/// the same type as [`EventKeyChordModel`] — persisted records are unaffected.
pub type EventModel<B> = EventKeyChordModel<B>;

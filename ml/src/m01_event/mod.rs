//! **Generation 01** — learned frame tokens, param-free scatter-add sum pooling: each token is the
//! summed field embeddings of the frame's onsetting notes through a nonlinear φ. First generation
//! to read the note stream directly, and first with an AR next-frame pretext ([`crate::pretrain::ar`]).
//! `Backbone`/`ArBackbone` impls live in [`model`].

mod batch;
mod model;

pub use batch::EventBatchData;
pub use model::{EventKeyChordModel, EventModelConfig};

/// Generation-01 model, named for the backbone it implements. A type alias, so it is
/// the same type as [`EventKeyChordModel`] — persisted records are unaffected.
pub type EventModel<B> = EventKeyChordModel<B>;

//! **Generation 02** — learned frame tokens, pooled by a set transformer.
//!
//! A second, smaller transformer embeds each frame's variable-length set of sounding
//! notes into that frame's CLS token via learned pad-masked attention; the main trunk
//! then runs over the 128 frame tokens. The bet over [`crate::m01_event`]'s sum pool:
//! attention can weight a frame's notes against each other (bass vs. top voice)
//! rather than collapsing them additively.
//!
//! The [`crate::backbone::Backbone`] / [`crate::backbone::ArBackbone`] impls live in
//! [`model`], next to the private fields they read.

mod batch;
mod model;

pub use batch::{HierBatchData, MAX_POLY};
pub use model::{HierEventModel, HierModelConfig};

/// Generation-02 model, named for the backbone it implements. A type alias, so it is
/// the same type as [`HierEventModel`] — persisted records are unaffected.
pub type HierModel<B> = HierEventModel<B>;

//! **Generation 02** — learned frame tokens, set-transformer pooling: a smaller transformer embeds
//! each frame's sounding-note set into that frame's CLS token via pad-masked attention, then the
//! main trunk runs over the frame tokens. The bet over [`crate::m01_event`]'s sum pool: attention
//! can weight a frame's notes against each other rather than collapsing them additively.
//! `Backbone`/`ArBackbone` impls live in [`model`].

mod batch;
mod model;

pub use batch::{HierBatchData, MAX_POLY};
pub use model::{HierEventModel, HierModelConfig};

/// Generation-02 model, named for the backbone it implements. A type alias, so it is
/// the same type as [`HierEventModel`] — persisted records are unaffected.
pub type HierModel<B> = HierEventModel<B>;

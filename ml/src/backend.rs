//! Compute-backend aliases, in one place so no other module names a concrete
//! backend.
//!
//! The pure-Rust `ndarray` CPU backend. (GPU/threaded-CPU backends were evaluated —
//! see git history — but on a model this small they don't help: intra-op
//! parallelism has nothing to fill and dispatch/launch overhead dominates. CPU
//! throughput comes from the data-parallel training loop in [`crate::parallel`]
//! instead.)
//!
//! These lived in `train.rs` until the backbones were unified, which made every
//! module — including [`crate::parallel`] — import its backend from the training
//! loop. They have no dependency of their own, so they sit at the bottom instead.

use burn::backend::Autodiff;
use burn::prelude::Backend;

/// Inference / non-autodiff backend.
pub type Inner = burn::backend::NdArray<f32>;

/// The device type for [`Inner`].
pub type MlDevice = burn::backend::ndarray::NdArrayDevice;

/// The autodiff training backend wrapping [`Inner`], with **gradient checkpointing**
/// (`BalancedCheckpointing`): elementwise/activation ops are recomputed in backward
/// instead of having their inputs retained on the autodiff graph. For the
/// hierarchical backbone the set transformer's `[batch*n_frames, MAX_POLY, d]` grid
/// creates large activation tensors; checkpointing cuts the peak memory they force.
/// The matmul outputs (compute-bound) are still retained, so this is a partial win —
/// the dominant lever remains shrinking that grid (smaller sub-encoder FFN, etc.).
pub type Back =
    Autodiff<Inner, burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing>;

/// The float element `B` actually computes in, e.g. `"f32"` — read off the type, so
/// a report can never claim a precision the run isn't using.
///
/// bf16 is not reachable from here: burn-ndarray implements `FloatNdArrayElement`
/// for f32/f64 only, cubecl's WGSL compiler rejects `FloatKind::BF16` outright, and
/// the Vulkan path registers it for storage/conversion, not arithmetic. It is a
/// CUDA-path element.
pub fn precision<B: Backend>() -> String {
    let name = std::any::type_name::<B::FloatElem>();
    name.rsplit("::").next().unwrap_or(name).to_string()
}

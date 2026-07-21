//! Compute-backend aliases, in one place so no other module names a concrete
//! backend.
//!
//! Two builds, selected by the `cuda` cargo feature; every bin follows because no
//! other module names a backend:
//!
//! * **default** — the pure-Rust `ndarray` CPU backend. Throughput comes from the
//!   data-parallel training loop in [`crate::parallel`], not from the backend:
//!   intra-op parallelism has nothing to fill on a model this small, so one stream
//!   tops out at ~2 cores.
//! * **`--features cuda`** — the CUDA backend (cubecl → nvrtc → PTX). Needs a CUDA
//!   toolkit, not just a driver: cubecl compiles its kernels at runtime via `nvrtc`.
//!   Here `/usr/local/cuda-13.2`; `cudarc` 0.19 supports CUDA 11.4–13.3.
//!
//! WGPU was measured *not* to help the frame/event backbones on an iGPU
//! (dispatch/launch overhead dominates), but [`crate::m02_hier`]'s set transformer is
//! far heavier — ~1.7 GFLOP/window at seq 256 — and does profit from a discrete GPU.
//!
//! These lived in `train.rs` until the backbones were unified, which made every
//! module — including [`crate::parallel`] — import its backend from the training
//! loop. They have no dependency of their own, so they sit at the bottom instead.

use burn::backend::Autodiff;
use burn::prelude::Backend;

/// Inference / non-autodiff backend.
///
/// Three builds, in priority order (cuda wins over gpu if both are on, which they never should
/// be): `cuda` → CUDA, `gpu` → WGPU (cubecl-wgsl → Vulkan/DX12/Metal), else the pure-Rust ndarray
/// CPU backend. Every bin follows because no other module names a backend.
#[cfg(feature = "cuda")]
pub type Inner = burn::backend::Cuda<f32, i64>;

/// The device type for [`Inner`].
#[cfg(feature = "cuda")]
pub type MlDevice = burn::backend::cuda::CudaDevice;

/// Inference / non-autodiff backend: WGPU (cubecl-wgsl). The `i64` int element CLAUDE.md pins for
/// CUDA isn't reachable here — WGSL has no first-class i64 — so this uses WGPU's default i32, and
/// the eval-count reads in [`crate::shared`] are written against `B::IntElem` to match.
#[cfg(all(feature = "gpu", not(feature = "cuda")))]
pub type Inner = burn::backend::Wgpu;

/// The device type for [`Inner`].
#[cfg(all(feature = "gpu", not(feature = "cuda")))]
pub type MlDevice = burn::backend::wgpu::WgpuDevice;

/// Inference / non-autodiff backend: the pure-Rust `ndarray` CPU backend. Throughput comes from
/// the data-parallel training loop in [`crate::parallel`], not from the backend: intra-op
/// parallelism has nothing to fill on a model this small, so one stream tops out at ~2 cores.
#[cfg(not(any(feature = "cuda", feature = "gpu")))]
pub type Inner = burn::backend::NdArray<f32>;

/// The device type for [`Inner`].
#[cfg(not(any(feature = "cuda", feature = "gpu")))]
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
/// bf16 is not reachable from the CPU build: burn-ndarray implements
/// `FloatNdArrayElement` for f32/f64 only, cubecl's WGSL compiler rejects
/// `FloatKind::BF16` outright, and the Vulkan path registers it for
/// storage/conversion, not arithmetic. It is a CUDA-path element — reachable under
/// `--features cuda`, but [`Inner`] pins f32 there too so the two builds stay
/// numerically comparable.
pub fn precision<B: Backend>() -> String {
    let name = std::any::type_name::<B::FloatElem>();
    name.rsplit("::").next().unwrap_or(name).to_string()
}

//! Compute-backend aliases, in one place so no other module names a concrete backend; every bin
//! follows. Selected by cargo feature: default `ndarray` CPU, `--features cuda` (cubecl→nvrtc→PTX,
//! needs a CUDA toolkit), `--features gpu` WGPU.

use burn::backend::Autodiff;
use burn::prelude::Backend;

#[cfg(feature = "cuda")]
pub type Inner = burn::backend::Cuda<f32, i64>;

#[cfg(feature = "cuda")]
pub type MlDevice = burn::backend::cuda::CudaDevice;

/// WGSL has no first-class i64, so WGPU uses its default i32; the eval-count reads in
/// [`crate::shared`] are written against `B::IntElem` to match.
#[cfg(all(feature = "gpu", not(feature = "cuda")))]
pub type Inner = burn::backend::Wgpu;

#[cfg(all(feature = "gpu", not(feature = "cuda")))]
pub type MlDevice = burn::backend::wgpu::WgpuDevice;

#[cfg(not(any(feature = "cuda", feature = "gpu")))]
pub type Inner = burn::backend::NdArray<f32>;

#[cfg(not(any(feature = "cuda", feature = "gpu")))]
pub type MlDevice = burn::backend::ndarray::NdArrayDevice;

/// Autodiff training backend over [`Inner`], with **gradient checkpointing**: activation ops are
/// recomputed in backward rather than retained, cutting the peak memory m02's
/// `[batch*n_frames, MAX_POLY, d]` grid forces. Matmul outputs are still retained (partial win).
pub type Back =
    Autodiff<Inner, burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing>;

/// The float element `B` actually computes in — read off the type, so a report can't claim a
/// precision the run isn't using. bf16 is unreachable from CPU/WGSL builds and pinned to f32 on
/// CUDA so both builds stay numerically comparable.
pub fn precision<B: Backend>() -> String {
    let name = std::any::type_name::<B::FloatElem>();
    name.rsplit("::").next().unwrap_or(name).to_string()
}

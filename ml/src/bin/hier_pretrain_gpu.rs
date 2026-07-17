//! Autoregressive pretraining for the hierarchical backbone on the **WGPU GPU
//! backend** (cubecl-wgsl → Vulkan/DX12/Metal; here an AMD Radeon iGPU).
//!
//! ```sh
//! cargo run --release --features gpu --bin hier_pretrain_gpu -- [epochs] [batch_size] [lr]
//! #   reads data/real_train.bin + data/real_val.bin (produced by `harvest`)
//! #   → models/02-hier/pretrained (+ .json)
//! ```
//!
//! Same pretext/model/output as `pretrain --backbone hier`, only swapping backend + driver: calls
//! [`ar::run_single_device`] since a GPU has nothing to shard (DP is a CPU-only trick). Output is
//! backend-agnostic ([`CompactRecorder`]), so it loads straight into the ndarray fine-tune.
//!
//! [`CompactRecorder`]: burn::record::CompactRecorder

use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use optime_ml::cli::Args;
use optime_ml::data::load_songs;
use optime_ml::m02_hier::{HierModel, HierModelConfig};
use optime_ml::pretrain::ar::{self, ArPretrainConfig};

/// GPU autodiff backend: WGPU + checkpointing (recompute elementwise ops in backward
/// so the set-transformer's activation tensors fit an iGPU's limited memory).
pub type GpuBack = Autodiff<Wgpu, BalancedCheckpointing>;

fn main() {
    let args = Args::parse();
    let config = ArPretrainConfig::default()
        .with_epochs(args.positional_or(0, 20))
        .with_batch_size(args.positional_or(1, 32))
        .with_lr(args.positional_or(2, 3.0e-4));

    let train =
        load_songs("data/real_train.bin").expect("load data/real_train.bin (run `harvest` first)");
    let val = load_songs("data/real_val.bin").unwrap_or_default();
    println!(
        "loaded {} train / {} val real windows (WGPU backend)",
        train.len(),
        val.len()
    );

    let device = WgpuDevice::default();
    ar::run_single_device::<HierModel<GpuBack>, GpuBack>(
        &config,
        &HierModelConfig::new(),
        &train,
        &val,
        &args.out_dir,
        &device,
    );
}

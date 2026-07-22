//! Autoregressive pretraining for the hierarchical backbone on the **WGPU GPU
//! backend** (cubecl-wgsl → Vulkan/DX12/Metal; here an AMD Radeon iGPU).
//!
//! ```sh
//! cargo run --release --features gpu --bin hier_pretrain_gpu -- [epochs] [batch_size] [lr]
//! #   reads data/real_train.bin + data/real_val.bin (produced by `harvest`)
//! #   → models/02-hier/pretrained (+ .json)
//! ```
//!
//! Same pretext, model, augmentation, and output as `pretrain --backbone hier` —
//! this bin only swaps the backend and the driver. It calls
//! [`ar::run_single_device`] because the rayon data-parallel step is a CPU-only trick
//! to fill cores (the ndarray backend tops out at ~2); a GPU runs one batch at a time
//! and has nothing to shard. Gradient checkpointing keeps the set transformer's large
//! `[batch×128, MAX_POLY, d]` activation grid off a memory-constrained iGPU.
//!
//! Output is backend-agnostic ([`CompactRecorder`]): weights saved here load straight
//! into the ndarray fine-tune (`train --backbone hier --pretrained …`).
//!
//! Note (per `parallel.rs`): GPU was measured *not* to help the smaller frame/event
//! backbones on this class of hardware — but the hierarchical set transformer is far
//! heavier, so it's worth re-measuring here.
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
    // Whole-song dataset → fixed 256-frame windows (hier is a fixed-window backbone).
    let train = optime_ml::pack::window_dataset(&train, 256);
    let val = optime_ml::pack::window_dataset(&val, 256);
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

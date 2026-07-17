//! Autoregressive pretraining for the hierarchical backbone on the **WGPU GPU backend**
//! (cubecl-wgsl → Vulkan/DX12/Metal). Same pretext/model/output as `pretrain --backbone hier`,
//! only swapping backend + driver: calls [`ar::run_single_device`] since a GPU has nothing to
//! shard (DP is a CPU-only trick). Output is backend-agnostic, so it loads straight into the
//! ndarray fine-tune. Reads `data/real_{train,val}.bin`; writes `<out-dir>/02-hier/pretrained`.

use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use clap::Args;
use optime_ml::data::load_songs;
use optime_ml::m02_hier::{HierModel, HierModelConfig};
use optime_ml::pretrain::ar::{self, ArPretrainConfig};
use std::path::PathBuf;

/// GPU autodiff backend: WGPU + checkpointing (recompute elementwise ops in backward so the
/// set-transformer's activation tensors fit an iGPU's limited memory).
pub type GpuBack = Autodiff<Wgpu, BalancedCheckpointing>;

#[derive(Args, Debug)]
pub struct PretrainGpuArgs {
    /// Epochs (default 20).
    pub epochs: Option<usize>,
    /// Batch size (default 32).
    pub batch_size: Option<usize>,
    /// Learning rate (default 3e-4).
    pub lr: Option<f64>,
    /// Model root the backbone's subdir hangs off.
    #[arg(long, default_value = "models")]
    pub out_dir: PathBuf,
}

pub fn run(args: PretrainGpuArgs) {
    let config = ArPretrainConfig::default()
        .with_epochs(args.epochs.unwrap_or(20))
        .with_batch_size(args.batch_size.unwrap_or(32))
        .with_lr(args.lr.unwrap_or(3.0e-4));

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

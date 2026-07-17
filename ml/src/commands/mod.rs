//! One subcommand per module (the old `src/bin/*` binaries), dispatched by `main.rs`. Harvest- and
//! GPU-only commands are feature-gated, so a build without the feature simply drops them (the old
//! per-bin `required-features`).

pub mod opts;

pub mod dashboard;
pub mod eval_real;
pub mod generate_data;
pub mod infer;
pub mod pretrain;
pub mod probe;
pub mod token_stats;
pub mod train;

#[cfg(feature = "harvest")]
pub mod chord_export;
#[cfg(feature = "harvest")]
pub mod eval_labeled;
#[cfg(feature = "harvest")]
pub mod harvest;
#[cfg(feature = "harvest")]
pub mod sft;

#[cfg(feature = "gpu")]
pub mod pretrain_gpu;

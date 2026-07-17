//! CPU data parallelism for training. The ndarray backend can't fill many cores on a model this
//! small (intra-op parallelism tops out ~2 cores), so each optimizer step shards the minibatch,
//! differentiates the shared weights per shard concurrently (rayon), sums the gradients, and takes
//! one step. Summed grads over disjoint shards = the full-batch gradient: exact synchronous DP-SGD.

use burn::module::{AutodiffModule, Module, ModuleVisitor, Param};
use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use rayon::prelude::*;

use crate::backend::{Back, Inner, MlDevice};

/// Default shard count = logical-core count, but **1 under `--features cuda`/`gpu`**: a GPU already
/// parallelizes within the batch, so sharding would just serialize launches. One shard makes
/// [`dp_step`] a plain single-device step.
pub fn default_shards() -> usize {
    if cfg!(feature = "cuda") || cfg!(feature = "gpu") {
        return 1;
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
}

/// Sums one gradient set into an accumulator. burn stores gradients rank-erased but visits each
/// param at its concrete rank `D`, which lets us fetch and sum matching tensors generically.
struct GradAdder {
    acc: GradientsParams,
    other: GradientsParams,
}

impl ModuleVisitor<Back> for GradAdder {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<Back, D>>) {
        let id = param.id;
        if let Some(b) = self.other.get::<Inner, D>(id) {
            match self.acc.get::<Inner, D>(id) {
                Some(a) => self.acc.register::<Inner, D>(id, a + b),
                None => self.acc.register::<Inner, D>(id, b),
            }
        }
    }
}

/// Add `other`'s gradients into `acc`, walking `model`'s params to recover ranks.
fn add_grads<M: Module<Back>>(
    model: &M,
    acc: GradientsParams,
    other: GradientsParams,
) -> GradientsParams {
    let mut adder = GradAdder { acc, other };
    model.visit(&mut adder);
    adder.acc
}

/// One data-parallel optimizer step. `indices` is split into up to `n_shards` shards, each
/// differentiated in parallel by `shard_grads`; the closure must scale so the **sum** over shards
/// is the intended full-batch quantity (e.g. divide each shard loss by `n_shards` for a batch mean).
pub fn dp_step<M, O, F>(
    model: M,
    optim: &mut O,
    lr: f64,
    device: &MlDevice,
    indices: &[usize],
    n_shards: usize,
    shard_grads: F,
) -> (M, f64)
where
    M: AutodiffModule<Back> + Clone + Send + Sync,
    O: Optimizer<M, Back>,
    F: Fn(&M, &[usize]) -> (GradientsParams, f64) + Sync,
{
    let n_shards = n_shards.max(1);
    let shard_size = indices.len().div_ceil(n_shards).max(1);
    let shards: Vec<&[usize]> = indices.chunks(shard_size).collect();

    // `fork` gives each shard an independent copy of the weights (same `ParamId`s, fresh autodiff
    // leaves — a plain clone deadlocks, sharing the tensors every thread differentiates). Only the
    // single Adam update is serial.
    let (grads, loss) = shards
        .into_par_iter()
        .map(|shard| {
            let m = model.clone().fork(device);
            let (g, l) = shard_grads(&m, shard);
            (Some(g), l)
        })
        .reduce(
            || (None, 0.0),
            |(ga, la), (gb, lb)| {
                let g = match (ga, gb) {
                    (Some(a), Some(b)) => Some(add_grads(&model, a, b)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                };
                (g, la + lb)
            },
        );

    let grads = grads.expect("at least one shard");
    let model = optim.step(lr, model, grads);
    (model, loss)
}

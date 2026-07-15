//! CPU data parallelism for training.
//!
//! burn's `ndarray` CPU backend can't spread a small transformer's autodiff across
//! many cores — intra-op parallelism has nothing to fill (most of the wall time is
//! tiny matmuls plus serial elementwise / softmax / layernorm / autodiff-graph
//! work), so a single training stream tops out at ~2 cores no matter the batch
//! size (threaded-CPU and GPU backends were measured too and don't help this model
//! size). We parallelize *across the batch* instead: each
//! optimizer step splits the minibatch into shards, differentiates the **shared**
//! weights on every shard concurrently (rayon), sums the per-shard gradients, and
//! takes one step. Summed gradients over disjoint shards equal the full-batch
//! gradient, so this is exact synchronous data-parallel SGD — and it scales to all
//! cores regardless of how small the model is.

use burn::module::{Module, ModuleVisitor, Param};
use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use rayon::prelude::*;

use crate::model::KeyChordModel;
use crate::train::{Back, Inner, MlDevice};

/// Default shard count for a data-parallel step: the machine's logical-core count.
pub fn default_shards() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
}

/// Adds one gradient set into an accumulator in place. burn stores gradients
/// rank-erased, but calls [`ModuleVisitor::visit_float`] with each parameter's
/// concrete rank `D`, which lets us fetch and sum the matching tensors generically.
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
fn add_grads(
    model: &KeyChordModel<Back>,
    acc: GradientsParams,
    other: GradientsParams,
) -> GradientsParams {
    let mut adder = GradAdder { acc, other };
    model.visit(&mut adder);
    adder.acc
}

/// One data-parallel optimizer step.
///
/// `indices` is the minibatch, split into up to `n_shards` roughly-equal shards.
/// Each shard's gradient is computed on a clone of `model` (clones share the same
/// weight tensors, so all shards differentiate the *same* parameters) via
/// `shard_grads`, in parallel across rayon threads. The closure returns each
/// shard's `GradientsParams` and its scalar loss contribution; it must scale so
/// that the **sum** over shards is the intended full-batch quantity (e.g. divide
/// each shard's loss by `n_shards` for a batch-mean objective). Returns the updated
/// model and the summed loss.
pub fn dp_step<O, F>(
    model: KeyChordModel<Back>,
    optim: &mut O,
    lr: f64,
    device: &MlDevice,
    indices: &[usize],
    n_shards: usize,
    shard_grads: F,
) -> (KeyChordModel<Back>, f64)
where
    O: Optimizer<KeyChordModel<Back>, Back>,
    F: Fn(&KeyChordModel<Back>, &[usize]) -> (GradientsParams, f64) + Sync,
{
    let n_shards = n_shards.max(1);
    let shard_size = indices.len().div_ceil(n_shards).max(1);
    let shards: Vec<&[usize]> = indices.chunks(shard_size).collect();

    // Per shard, in parallel: `fork` an **independent** copy of the weights (same
    // `ParamId`s, fresh autodiff leaves — cloning instead deadlocks, since clones
    // share the parameter tensors every thread differentiates), compute the shard
    // gradient, and reduce. Forking and the tree-reduction are parallel too, so the
    // only serial work per step is the single Adam update.
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

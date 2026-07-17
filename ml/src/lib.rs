//! optime-ml: a transformer that infers a song's global key and its per-frame chord from
//! synthesizer note events (pitch + velocity + instrument role + pan + onset) — the same data
//! OptimePlayer's `SynthEvent` stream emits.
//!
//! Three model generations differ *only* in how a window of note events becomes a hidden state
//! ([`m00_frame`] hand feature grid, [`m01_event`] scatter-add pooling, [`m02_hier`] set-transformer
//! pooling); heads, label space, loss, [`train`], and [`infer`] are shared. Adding a generation =
//! implement [`backbone::Backbone`] (+ optional [`backbone::ArBackbone`]), not copy the loop.
//!
//! Module index (one line each, alphabetical below):
//! [`theory`] label space · [`progression`]+[`notes`] synth gen · [`features`] m00 grid ·
//! [`tokenize`] m01/m02 tokens · [`data`] dataset · [`shared`] targets/loss/metrics ·
//! [`transformer`] RoPE encoder · [`train`] the one supervised loop · [`pretrain`] masked+AR ·
//! [`infer`] · [`parallel`] DP step · [`backend`]/[`backbone`]/[`flops`]/[`progress`] ·
//! [`probe`]/[`estimate`] refs · [`harvest`] (feature-gated, pulls the engine).

pub mod annotations;
pub mod backbone;
pub mod backend;
pub mod dashboard;
pub mod data;
pub mod estimate;
pub mod features;
pub mod flops;
#[cfg(feature = "harvest")]
pub mod harvest;
pub mod infer;
pub mod m00_frame;
pub mod m01_event;
pub mod m02_hier;
pub mod notes;
pub mod parallel;
pub mod pretrain;
pub mod probe;
pub mod progress;
pub mod progression;
pub mod shared;
pub mod theory;
pub mod tokenize;
pub mod train;
pub mod transformer;

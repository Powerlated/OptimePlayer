//! optime-ml: a transformer that infers a song's global key and its chord at any
//! point in time, from synthesizer note events (pitch + velocity + instrument
//! role + pan + onset) — the same data OptimePlayer's `SynthEvent` stream emits.
//!
//! ```text
//! progression + voicing ─► NoteEvent stream ─► backbone (m00 | m01 | m02)
//!                                           ─► chord head (per frame, root x quality)
//!                                           └─► key head  (pooled)
//! ```
//!
//! ## Model generations
//!
//! Numbered by the order they were built. They differ *only* in how a window of note
//! events becomes a hidden state; the heads, label space, loss, training loop, and
//! inference are shared.
//!
//! * [`m00_frame`] — hand-engineered 57-dim per-frame feature grid.
//! * [`m01_event`] — learned frame tokens, param-free scatter-add pooling.
//! * [`m02_hier`]  — learned frame tokens, set-transformer (CLS attention) pooling.
//!
//! Adding a generation means implementing [`backbone::Backbone`] (and optionally
//! [`backbone::ArBackbone`]) — not copying the training loop.
//!
//! ## Shared infrastructure
//!
//! * [`backbone`]    — the `Backbone`/`ArBackbone` contract + `ModelOutput`/`ArOutput`.
//! * [`backend`]     — compute-backend aliases (`Back`/`Inner`/`MlDevice`).
//! * [`cli`]         — shared bin argument parsing (`--backbone`, `--out-dir`).
//! * [`shared`]      — factored targets, multi-task loss, beat-aware smoothness, metrics.
//! * [`theory`]      — pitch classes, chords, keys, diatonic harmony, label space.
//! * [`progression`] — chord-progression generation (templates + Markov walk).
//! * [`notes`]       — arrangement/voicing into note events + per-frame labels.
//! * [`features`]    — note events → per-frame feature grid (generation 00's input).
//! * [`flops`]       — analytic matmul FLOP estimates for a forward pass.
//! * [`tokenize`]    — note events → per-note field indices (the learned-token input).
//! * [`data`]        — synthetic dataset generation, retention, example shape.
//! * [`train`]       — the one supervised multi-task loop, generic over a backbone.
//! * [`transformer`] — pre-norm encoder with RoPE (burn's has no rotary hook).
//! * [`pretrain`]    — self-supervised pretexts: masked-frame (m00) + autoregressive (m01/m02).
//! * [`parallel`]    — CPU data-parallel optimizer step.
//! * [`probe`]       — frozen-encoder "is-music" linear probe (weak song-name labels).
//! * [`estimate`]    — training-free chroma-template + Viterbi chord reference.
//! * [`infer`]       — run a trained model, produce a chord/key timeline.
//! * [`progress`]    — time-throttled in-epoch progress logging.
//!
//! Behind the `harvest` feature (pulls in the engine crate):
//! * [`harvest`]     — run device sequencers headlessly → unlabeled real songs.

pub mod annotations;
pub mod backbone;
pub mod backend;
pub mod cli;
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

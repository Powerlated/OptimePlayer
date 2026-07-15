//! optime-ml: a transformer that infers a song's global key and its chord at any
//! point in time, from synthesizer note events (pitch + velocity + instrument
//! role + pan + onset) — the same data OptimePlayer's `SynthEvent` stream emits.
//!
//! Pipeline:
//! ```text
//! progression + voicing ─► NoteEvent stream ─► per-frame feature grid
//!                        ─► Transformer encoder ─► chord head (per frame)
//!                                              └─► key head  (pooled)
//! ```
//!
//! Modules:
//! * [`theory`]      — pitch classes, chords, keys, diatonic harmony, label space.
//! * [`progression`] — chord-progression generation (templates + Markov walk).
//! * [`notes`]       — arrangement/voicing into note events + per-frame labels.
//! * [`features`]    — note events → per-frame feature grid (the model's input).
//! * [`data`]        — synthetic dataset generation, retention, example shape.
//! * [`model`]       — the Burn transformer (chord + key + reconstruction heads).
//! * [`train`]       — multi-task training loop (+ optional pretrained warm-start).
//! * [`pretrain`]    — self-supervised masked-frame pretraining on real songs.
//! * [`probe`]       — frozen-encoder "is-music" linear probe (weak song-name labels).
//! * [`estimate`]    — training-free chroma-template + Viterbi chord reference.
//! * [`infer`]       — run a trained model, produce a chord/key timeline.
//!
//! Behind the `harvest` feature (pulls in the engine crate):
//! * [`harvest`]     — run device sequencers headlessly → unlabeled real songs.

pub mod data;
pub mod estimate;
pub mod features;
#[cfg(feature = "harvest")]
pub mod harvest;
pub mod infer;
pub mod model;
pub mod notes;
pub mod parallel;
pub mod pretrain;
pub mod probe;
pub mod progression;
pub mod theory;
pub mod train;

//! Every out-of-the-box setting in one place.
//!
//! The `Default` impls for the persisted state ([`Persisted`](crate::persisted::Persisted) and
//! [`ResampleSettings`](crate::persisted::InstrumentResampleSettings)) read their values from here, so the
//! app's defaults are easy to find and tweak without hunting through the UI code.
//!
//! Note the distinction these constants keep that the engine does not: a *default* cutoff is the
//! listenable value the user starts on, whereas
//! [`ResampleMode::CUTOFF_OFF_HZ`](optime_core::InstrumentResampleMode::CUTOFF_OFF_HZ) is the transparent
//! "no extra filtering" position (and the slider's maximum). They are not the same number.

use crate::persisted::{RepeatMode, InstrumentResampleChoice, SortMode};

// ── Playback ────────────────────────────────────────────────────────────────
/// Start with shuffle off.
pub const SHUFFLE: bool = false;
/// Loop the queue forever by default.
pub const REPEAT: RepeatMode = RepeatMode::All;
/// Master volume (0..=1).
pub const VOLUME: f32 = 1.0;
/// Native song-list order until the user sorts.
pub const SORT_MODE: SortMode = SortMode::Default;
/// Sort ascending (A→Z, shortest→longest) by default.
pub const SORT_DESCENDING: bool = false;

// ── Stereo / mixing ─────────────────────────────────────────────────────────
pub const STEREO_SEPARATION: bool = true;
pub const FORCE_STEREO_SEPARATION: bool = false;
pub const BASS_MONO: bool = true;
/// Crossover below which the stereo expander keeps the signal centered.
pub const BASS_MONO_FREQ_HZ: f32 = 200.0;
/// Stereo-expander delay-change handling: 0 = immediate, 1 = hold during notes.
pub const DELAY_SMOOTHING_CHOICE: usize = 0;
/// Sample rate for the intermediate mixing step
pub const MIXER_SAMPLE_RATE: u32 = 48000;

// ── Tuning ──────────────────────────────────────────────────────────────────
/// 0 = equal temperament, 1 = pure (Pythagorean).
pub const TUNING_CHOICE: usize = 0;
/// Tonic for the pure tuning, in semitones from A.
pub const PURE_TONIC: i32 = 0;

// ── Resampling (per device) ──────────────────────────────────────────────────
/// Resample mode choice
pub const RESAMPLE_CHOICE: InstrumentResampleChoice = InstrumentResampleChoice::SincOutputNyquist;
/// Total source-tap count for the sinc/reconstruction kernel.
pub const SINC_TAPS: usize = 32;
/// Listenable default low-pass cutoff (Hz) for PSG (square/wave/noise) voices. The slider can
/// still be opened all the way to `ResampleMode::CUTOFF_OFF_HZ` (no extra filtering).
pub const PSG_CUTOFF_HZ: u32 = 15_000;
/// Listenable default low-pass cutoff (Hz) for sampled (DirectSound/SWAR) voices.
pub const SAMPLER_CUTOFF_HZ: u32 = 15_000;
/// Preserve the hardware's hard PSG on/off edges by default (don't slew the pops).
pub const SMOOTH_PSG_POPS: bool = false;

// ── Master high-shelf EQ (per device) ────────────────────────────────────────
/// Off by default — a transparent pass until the user dials in a shelf.
pub const SHELF_ENABLED: bool = false;
/// Filter order (even); the cascade has `order / 2` biquad sections.
pub const SHELF_ORDER: usize = 2;
/// Shelf resonance at the corner.
pub const SHELF_Q: f32 = 0.707;
/// Corner frequency (Hz).
pub const SHELF_CUTOFF_HZ: f32 = 4000.0;
/// Shelf gain (dB); 0 = flat.
pub const SHELF_GAIN_DB: f32 = 0.0;

//! Optime Player front-end: an `eframe`/`egui` application that runs natively and on the web,
//! driving the platform-independent [`optime_core`] DS sound engine.

mod annotation;
mod app;
mod audio;
mod chord_data;
mod media_controls;
mod persisted;
mod piano_roll;
mod player;
mod song_names;
mod theme;
mod visualizer;
mod wav;

#[cfg(target_arch = "wasm32")]
mod web;

pub use app::OptimeApp;

/// Number of sequence tracks (re-exported from the core engine).
pub const TRACK_COUNT: usize = optime_core::TRACK_COUNT;

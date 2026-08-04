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

pub const TRACK_COUNT: usize = optime_core::TRACK_COUNT;

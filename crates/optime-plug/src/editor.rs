//! The plugin's egui editor.
//!
//! Not cosmetic — the plugin is unusable without it. A host's generic parameter view shows only
//! *parameters*, and the two things that decide what the engine even plays (which ROM, which
//! voicegroup) are persisted state rather than automatable floats. This is the only way to set them.
//!
//! Styling comes from `optime-ui`, the same crate the player app themes itself with, so the two
//! look like one product.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use nice_plug::prelude::*;
use nice_plug_egui::{EguiState, create_egui_editor, widgets::ParamSlider};
use optime_core::SoundData;
use optime_core::SynthController;
use optime_core::devices::gba::GbaRom;
use optime_core::devices::gba::param_player::{ParamPlayer, TrackParams as EngineTrackParams};

use crate::params::{OptimePlugParams, TRACKS};

/// The handoff between the editor and the audio thread.
///
/// Loading a ROM means file I/O, parsing, decoding, and allocation — none of which may happen on
/// the audio thread. The background worker is allowed to block, so it builds the whole engine and
/// parks it here; `process` swaps it in with a `try_lock` and never waits.
#[derive(Default)]
pub struct RomSlot {
    /// A freshly built engine waiting to be picked up.
    pub pending: Mutex<Option<SynthController>>,
    /// The output rate, published by `initialize` so the worker can build at the right rate.
    pub sample_rate: AtomicU32,
    /// Human-readable status for the editor to show.
    pub status: Mutex<String>,
    /// Whether a file dialog is already up, so repeated clicks can't stack modal windows.
    pub dialog_open: AtomicBool,
    /// The parsed ROM: the song list the editor shows, and what each rebuild starts from.
    pub rom: Mutex<Option<Arc<GbaRom>>>,
    /// Playable song ids, cached for the editor.
    pub songs: Mutex<Vec<u32>>,
    /// An armed song engine (the real bytecode player) waiting for the transport to roll from bar 1.
    ///
    /// Separate from [`Self::pending`]: `pending` is the note-driven player's swaps (ROM load, song
    /// change, capture stop), while this holds a [`GbaPlayer`] parked by [`Task::ArmCapture`] until
    /// `process` sees playback and pulls it in. Keeping them apart means an armed-but-unstarted
    /// capture never yanks the running engine, and disarming is just `take()` with no rebuild.
    pub capture_engine: Mutex<Option<SynthController>>,
}

impl RomSlot {
    /// Opens the file dialog, then loads whatever was picked.
    ///
    /// **Must not run on the GUI thread.** `IFileDialog::Show` pumps a modal message loop on its
    /// calling thread; from inside egui's draw callback that re-enters the window proc, re-enters
    /// the in-progress draw, and panics on egui's already-borrowed `Context` — which, unwinding
    /// through the VST3 FFI boundary, takes the host down with it. Hence the background task.
    pub fn pick_and_load(&self, params: &OptimePlugParams) {
        if self.dialog_open.swap(true, Ordering::SeqCst) {
            return;
        }
        if let Some(file) = rfd::FileDialog::new()
            .add_filter("GBA ROM / audio extract", &["gba", "gbaaudio"])
            .pick_file()
        {
            let path = file.display().to_string();
            *params.rom_path.write().unwrap() = path.clone();
            self.open(&path, params);
        }
        self.dialog_open.store(false, Ordering::SeqCst);
    }

    /// Reads and parses `path`, then selects the persisted song.
    ///
    /// Any failure is reported into [`Self::status`] rather than silently leaving the old engine in
    /// place — a plugin that quietly plays the previous ROM after you picked a new one is worse
    /// than one that says it failed.
    pub fn open(&self, path: &str, params: &OptimePlugParams) {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                *self.status.lock().unwrap() = format!("Could not read: {e}");
                return;
            }
        };
        let Some(rom) = GbaRom::parse(&bytes) else {
            *self.status.lock().unwrap() = "Not an MP2K ROM (no song table found)".to_owned();
            return;
        };
        let songs = rom.song_ids();
        if songs.is_empty() {
            *self.status.lock().unwrap() = "No playable songs in the table".to_owned();
            return;
        }
        *self.status.lock().unwrap() = format!("{} songs", songs.len());

        // Keep the persisted song if the ROM still has it, else fall back to the first.
        let want = *params.song_id.read().unwrap();
        let song = if songs.contains(&want) {
            want
        } else {
            songs[0]
        };
        *params.song_id.write().unwrap() = song;

        *self.songs.lock().unwrap() = songs;
        *self.rom.lock().unwrap() = Some(Arc::new(rom));
        self.select_song(song, params);
    }

    /// Points the engine at `song`: takes its voicegroup from the song header and rebuilds.
    ///
    /// The voicegroup is a pointer, so it can't be a parameter — but it also shouldn't be typed in
    /// by hand, because the song header already knows it.
    pub fn select_song(&self, song: u32, params: &OptimePlugParams) {
        let rom = self.rom.lock().unwrap().clone();
        let Some(rom) = rom else { return };
        let Some(header) = rom.song_header(song) else {
            *self.status.lock().unwrap() = format!("Song {song} has no valid header");
            return;
        };
        *params.song_id.write().unwrap() = song;
        *params.voicegroup.write().unwrap() = header.voicegroup as u32;
        self.rebuild(params);
    }

    /// Rebuilds the note-driven engine (the normal, MIDI-playing mode) and parks it.
    pub fn rebuild(&self, params: &OptimePlugParams) {
        let rate = self.sample_rate.load(Ordering::Relaxed);
        let rom = self.rom.lock().unwrap().clone();
        let (Some(rom), true) = (rom, rate != 0) else {
            return;
        };
        let voicegroup = *params.voicegroup.read().unwrap() as usize;
        let player = ParamPlayer::new(rom.data.clone(), voicegroup, TRACKS);
        *self.pending.lock().unwrap() = Some(SynthController::with_player(
            f64::from(rate),
            Box::new(player),
        ));
    }

    /// Builds a *song* engine — the real bytecode player — and parks it armed, not running. This
    /// is what rip capture listens to once the transport rolls from bar 1.
    pub fn build_capture_engine(&self, params: &OptimePlugParams) -> bool {
        let rate = self.sample_rate.load(Ordering::Relaxed);
        let rom = self.rom.lock().unwrap().clone();
        let (Some(rom), true) = (rom, rate != 0) else {
            return false;
        };
        let song = *params.song_id.read().unwrap();
        let Some(mut controller) = SynthController::new(f64::from(rate), rom.as_ref(), song) else {
            *self.status.lock().unwrap() = format!("Song {song} would not start");
            return false;
        };
        // Only a capture engine records its notes; the note-driven one would just echo the DAW's
        // own notes straight back at it.
        controller.set_record_notes(true);
        *self.capture_engine.lock().unwrap() = Some(controller);
        true
    }
}

/// The live engine state rip capture publishes for the editor to record as automation.
///
/// **Why this shape.** VST3 records automation through `IComponentHandler::performEdit`, which
/// nice-plug exposes only as `ParamSetter` on the GUI thread. So the audio thread cannot write
/// automation itself; it can only publish what the song is doing, and let the editor perform it.
///
/// It publishes a *snapshot*, not an event queue, because automation is sampled rather than
/// streamed: `performEdit` timestamps at "now" regardless, the editor runs at ~60 Hz and MP2K's
/// control rate is ~59.73 Hz, so one snapshot per GUI frame loses essentially nothing — and a
/// snapshot cannot overflow or drop the way a queue can.
#[derive(Default)]
pub struct CaptureState {
    /// Whether capture is armed: the song engine is built and parked, but `process` hasn't seen
    /// the transport roll from bar 1 yet. Disjoint from `active` (which means it's running).
    pub armed: AtomicBool,
    /// Whether capture is running (the engine is a song player, not the note-driven one).
    pub active: AtomicBool,
    /// Latest per-track registers from the running song.
    pub tracks: Mutex<Vec<EngineTrackParams>>,
    /// Bumped on every refresh, so the editor can skip a frame that brought nothing new.
    pub generation: AtomicU64,
    /// The song's own tempo in BPM (`f32` bits), published so the editor can tell the user to match
    /// the project to it.
    ///
    /// This matters: during capture the song runs on its *own* `TEMPO` commands while the host
    /// timestamps the automation at the *project* tempo. Mismatch them and the lanes land at the
    /// wrong bar positions relative to the notes.
    pub song_bpm: AtomicU32,
}

/// Editor-only state (nothing the engine reads).
struct EditorState {
    /// Which track's controls are shown. 16 tracks × 22 params is far too many at once, and a
    /// host's generic view already lists them all for automation — this is for *playing*.
    track: usize,
    /// The last values rip capture wrote, so only genuine changes are performed. Without this every
    /// frame would `performEdit` all 352 parameters and bury the host in automation points.
    last_written: Option<Vec<EngineTrackParams>>,
    /// The capture generation already consumed.
    last_generation: u64,
}

pub fn create(
    params: Arc<OptimePlugParams>,
    rom: Arc<RomSlot>,
    capture: Arc<CaptureState>,
    editor_state: Arc<EguiState>,
    async_executor: AsyncExecutor<crate::OptimePlug>,
) -> Option<Box<dyn Editor>> {
    let initial = EditorState {
        track: 0,
        last_written: None,
        last_generation: 0,
    };

    create_egui_editor(
        editor_state,
        initial,
        Default::default(),
        // Runs once when the window opens: adopt the app's theme.
        |ctx, _queue, _state| optime_ui::apply(ctx),
        move |ui, setter, _queue, state| {
            egui::Frame::NONE
                .fill(optime_ui::BG)
                .inner_margin(egui::Margin::same(12))
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 6.0;
                    // Perform any automation the running song produced, before drawing: the widgets
                    // below should show the values we just wrote.
                    drain_capture(&capture, setter, &params, state);
                    header(ui, &params, &rom, &async_executor);
                    ui.separator();
                    rip_capture(ui, &capture, &rom, &async_executor);
                    ui.separator();
                    globals(ui, &params, setter);
                    ui.separator();
                    track_selector(ui, state);
                    ui.add_space(2.0);
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        track_params(ui, &params, setter, state.track);
                    });
                });
        },
    )
}

/// ROM picker, voicegroup, and load status.
fn header(
    ui: &mut egui::Ui,
    params: &Arc<OptimePlugParams>,
    rom: &Arc<RomSlot>,
    async_executor: &AsyncExecutor<crate::OptimePlug>,
) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Optime Player").size(18.0).strong());
        ui.label(
            egui::RichText::new("GBA · MP2K")
                .size(11.5)
                .color(optime_ui::TEXT_DIM),
        );
    });

    ui.horizontal(|ui| {
        // Hand the dialog to a background thread — see `RomSlot::pick_and_load`.
        let busy = rom.dialog_open.load(Ordering::Relaxed);
        if ui
            .add_enabled(!busy, egui::Button::new("Load ROM…"))
            .clicked()
        {
            async_executor.execute_background(crate::Task::PickRom);
        }

        let path = params.rom_path.read().unwrap().clone();
        let name = std::path::Path::new(&path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "no ROM loaded".to_owned());
        ui.label(egui::RichText::new(name).color(optime_ui::TEXT_DIM));
    });

    ui.horizontal(|ui| {
        ui.label("Song");
        let songs = rom.songs.lock().unwrap().clone();
        let current = *params.song_id.read().unwrap();
        if songs.is_empty() {
            ui.label(
                egui::RichText::new("load a ROM first")
                    .size(11.5)
                    .color(optime_ui::TEXT_DIM),
            );
        } else {
            let mut pick = current;
            egui::ComboBox::from_id_salt("song")
                .selected_text(format!("{current}"))
                .width(90.0)
                .show_ui(ui, |ui| {
                    for id in &songs {
                        ui.selectable_value(&mut pick, *id, format!("{id}"));
                    }
                });
            if pick != current {
                *params.song_id.write().unwrap() = pick;
                // Rebuilding re-reads the ROM: off the GUI thread, like the dialog.
                async_executor.execute_background(crate::Task::SelectSong);
            }
            ui.label(
                egui::RichText::new(format!(
                    "voicegroup {:#x}",
                    *params.voicegroup.read().unwrap()
                ))
                .size(11.5)
                .color(optime_ui::TEXT_DIM),
            );
        }

        let status = rom.status.lock().unwrap().clone();
        if !status.is_empty() {
            ui.label(
                egui::RichText::new(status)
                    .size(11.5)
                    .color(optime_ui::TEXT_DIM),
            );
        }
    });
}

/// The rip-capture panel.
fn rip_capture(
    ui: &mut egui::Ui,
    capture: &Arc<CaptureState>,
    rom: &Arc<RomSlot>,
    async_executor: &AsyncExecutor<crate::OptimePlug>,
) {
    optime_ui::section_header(ui, "Rip capture");
    let armed = capture.armed.load(Ordering::Relaxed);
    let active = capture.active.load(Ordering::Relaxed);
    let have_rom = rom.rom.lock().unwrap().is_some();

    ui.horizontal(|ui| {
        // Three states, one button: idle arms, armed disarms, running stops. ArmCapture builds and
        // parks the song engine; the song itself waits for the transport (see lib.rs `process`).
        let (label, task) = if active {
            ("Stop capture", crate::Task::StopCapture)
        } else if armed {
            ("Disarm", crate::Task::StopCapture)
        } else {
            ("ARM CAPTURE", crate::Task::ArmCapture)
        };
        if ui.add_enabled(have_rom, egui::Button::new(label)).clicked() {
            async_executor.execute_background(task);
        }
        if active {
            ui.label(
                egui::RichText::new("● playing the song — notes + automation streaming")
                    .size(11.5)
                    .color(optime_ui::ACCENT),
            );
        } else if armed {
            ui.label(
                egui::RichText::new("● armed — press play from bar 1")
                    .size(11.5)
                    .color(optime_ui::ACCENT),
            );
        }
    });
    let bpm = f32::from_bits(capture.song_bpm.load(Ordering::Relaxed));
    if bpm > 0.0 {
        ui.label(
            egui::RichText::new(format!(
                "Song tempo is {bpm:.1} BPM — set the project tempo to match, or the captured\
                 lanes will not line up with the notes."
            ))
            .size(10.5)
            .color(optime_ui::ACCENT),
        );
    }
    ui.label(
        egui::RichText::new(
            "Arm automation write on this channel, press ARM CAPTURE, then roll the transport\
             from bar 1. The song plays from the ROM: its notes stream out as MIDI (the host\
             records them into a clip) and every register move is recorded as automation. Stop,\
             and your MIDI drives the same engine through the lanes it just wrote.",
        )
        .size(10.5)
        .color(optime_ui::TEXT_DIM),
    );
}

/// Performs whatever the running song changed since the last frame as host automation.
///
/// This is the whole trick: the audio thread can't write automation (nice-plug only exposes
/// `performEdit` through `ParamSetter`, on this thread), so it publishes a snapshot and the editor
/// performs it. `begin`/`set`/`end` per changed parameter is exactly the gesture a knob-drag makes,
/// which is what a host records in write/latch mode.
///
/// Only *changes* are performed. Performing all 352 every frame would bury the host in redundant
/// automation points and swamp the parameter queue.
fn drain_capture(
    capture: &Arc<CaptureState>,
    setter: &ParamSetter,
    params: &Arc<OptimePlugParams>,
    state: &mut EditorState,
) {
    if !capture.active.load(Ordering::Relaxed) {
        state.last_written = None;
        return;
    }
    let generation = capture.generation.load(Ordering::Acquire);
    if generation == state.last_generation {
        return;
    }
    // `try_lock`: the audio thread may be mid-publish. Missing a frame is fine — the next one
    // carries a newer snapshot, and blocking the GUI on the audio thread is never fine.
    let live: Vec<EngineTrackParams> = {
        let Ok(live) = capture.tracks.try_lock() else {
            return;
        };
        live.clone()
    };
    state.last_generation = generation;

    let previous = state.last_written.take();
    for (t, now) in live.iter().enumerate() {
        if t >= TRACKS {
            break;
        }
        let before = previous.as_ref().map(|p| p[t]);
        write_track(setter, &params.tracks[t], now, before.as_ref());
    }
    state.last_written = Some(live);
}

/// Performs the fields of one track that differ from `before` (or all of them, first time).
fn write_track(
    setter: &ParamSetter,
    p: &crate::params::TrackParams,
    now: &EngineTrackParams,
    before: Option<&EngineTrackParams>,
) {
    macro_rules! write_if_changed {
        ($param:expr, $field:ident, $conv:expr) => {
            if before.is_none_or(|b| b.$field != now.$field) {
                let value = $conv(now.$field);
                setter.begin_set_parameter($param);
                setter.set_parameter($param, value);
                setter.end_set_parameter($param);
            }
        };
    }

    write_if_changed!(&p.prog, prog, i32::from);
    write_if_changed!(&p.vol, vol, i32::from);
    write_if_changed!(&p.pan, pan, i32::from);
    write_if_changed!(&p.bend, bend, i32::from);
    write_if_changed!(&p.bend_range, bend_range, i32::from);
    write_if_changed!(&p.mod_, mod_, i32::from);
    write_if_changed!(&p.lfo_speed, lfo_speed, i32::from);
    write_if_changed!(&p.lfo_delay, lfo_delay, i32::from);
    write_if_changed!(&p.tune, tune, i32::from);
    write_if_changed!(&p.key_shift, key_shift, i32::from);
    write_if_changed!(&p.priority, priority, i32::from);
    write_if_changed!(&p.tone_override, tone_override, |v| v);
    write_if_changed!(&p.kind, kind, i32::from);
    write_if_changed!(&p.attack, attack, i32::from);
    write_if_changed!(&p.decay, decay, i32::from);
    write_if_changed!(&p.sustain, sustain, i32::from);
    write_if_changed!(&p.release, release, i32::from);
    write_if_changed!(&p.length, length, i32::from);
    write_if_changed!(&p.pan_sweep, pan_sweep, i32::from);
    write_if_changed!(&p.echo_volume, echo_volume, i32::from);
    write_if_changed!(&p.echo_length, echo_length, i32::from);

    // `modT` is an enum parameter, so it needs the variant rather than the raw register.
    if before.is_none_or(|b| b.mod_type != now.mod_type) {
        let value = match now.mod_type {
            1 => crate::params::ModType::Tremolo,
            2 => crate::params::ModType::AutoPan,
            _ => crate::params::ModType::Vibrato,
        };
        setter.begin_set_parameter(&p.mod_type);
        setter.set_parameter(&p.mod_type, value);
        setter.end_set_parameter(&p.mod_type);
    }
}

fn globals(ui: &mut egui::Ui, params: &Arc<OptimePlugParams>, setter: &ParamSetter) {
    optime_ui::section_header(ui, "Global");
    ui.horizontal(|ui| {
        ui.label("Preset");
        ui.add(ParamSlider::for_param(&params.preset, setter).with_width(120.0));
    });
    row(ui, "Master volume", &params.master_volume, setter);
    row(ui, "Reverb", &params.reverb, setter);
    row(ui, "Max channels", &params.max_chans, setter);
    ui.label(
        egui::RichText::new(
            "Max channels is fidelity, not performance: the engine steals by priority within it, \
             so it decides which notes drop. Games use 5–8.",
        )
        .size(10.5)
        .color(optime_ui::TEXT_DIM),
    );
}

fn track_selector(ui: &mut egui::Ui, state: &mut EditorState) {
    optime_ui::section_header(ui, "Tracks");
    ui.horizontal_wrapped(|ui| {
        for t in 0..TRACKS {
            let selected = state.track == t;
            if ui
                .selectable_label(selected, format!("{}", t + 1))
                .on_hover_text(format!("MIDI channel {}", t + 1))
                .clicked()
            {
                state.track = t;
            }
        }
    });
}

fn track_params(
    ui: &mut egui::Ui,
    params: &Arc<OptimePlugParams>,
    setter: &ParamSetter,
    track: usize,
) {
    let t = &params.tracks[track];

    optime_ui::section_header(ui, "Voice");
    row(ui, "Program", &t.prog, setter);
    row(ui, "Volume", &t.vol, setter);
    row(ui, "Pan", &t.pan, setter);
    row(ui, "Priority", &t.priority, setter);

    optime_ui::section_header(ui, "Pitch");
    row(ui, "Bend", &t.bend, setter);
    row(ui, "Bend range", &t.bend_range, setter);
    row(ui, "Tune", &t.tune, setter);
    row(ui, "Key shift", &t.key_shift, setter);

    optime_ui::section_header(ui, "LFO");
    ui.horizontal(|ui| {
        ui.label("Mod type");
        ui.add(ParamSlider::for_param(&t.mod_type, setter).with_width(120.0));
    });
    row(ui, "Mod depth", &t.mod_, setter);
    row(ui, "LFO speed", &t.lfo_speed, setter);
    row(ui, "LFO delay", &t.lfo_delay, setter);

    optime_ui::section_header(ui, "Tone override");
    ui.horizontal(|ui| {
        ui.add(ParamSlider::for_param(&t.tone_override, setter).with_width(60.0));
        ui.label(
            egui::RichText::new("Off = use the voicegroup's tone verbatim, as a song does.")
                .size(10.5)
                .color(optime_ui::TEXT_DIM),
        );
    });
    ui.add_enabled_ui(t.tone_override.value(), |ui| {
        row(ui, "Type", &t.kind, setter);
        row(ui, "Attack", &t.attack, setter);
        row(ui, "Decay", &t.decay, setter);
        row(ui, "Sustain", &t.sustain, setter);
        row(ui, "Release", &t.release, setter);
        row(ui, "Length", &t.length, setter);
        row(ui, "Sweep", &t.pan_sweep, setter);
    });

    optime_ui::section_header(ui, "Pseudo-echo");
    row(ui, "Echo volume", &t.echo_volume, setter);
    row(ui, "Echo length", &t.echo_length, setter);
}

/// One labelled parameter slider.
fn row<P: Param>(ui: &mut egui::Ui, label: &str, param: &P, setter: &ParamSetter) {
    ui.horizontal(|ui| {
        ui.add_sized(
            [110.0, 18.0],
            egui::Label::new(label).halign(egui::Align::LEFT),
        );
        ui.add(ParamSlider::for_param(param, setter).with_width(220.0));
    });
}

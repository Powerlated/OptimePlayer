//! The egui application: song list, transport, settings, visualizer, and live keyboard input.

use std::sync::{Arc, Mutex};

use optime_core::{Controller, Sdat, SynthConfig, TuningSystem};

use crate::visualizer::{self, VisSnapshot};
use crate::{audio::AudioEngine, player, TRACK_COUNT};

/// One entry in the flattened song list.
struct Song {
    sdat_index: usize,
    sseq_id: u32,
    label: String,
}

/// Demo SDATs available to load. Native reads from `demos/`; web fetches them at runtime.
const DEMOS: &[(&str, &str)] = &[
    ("Super Mario 64 DS", "super-mario-64-ds"),
    ("New Super Mario Bros.", "new-super-mario-bros"),
    ("Pokémon Platinum", "pokemon-platinum"),
    ("Pokémon HeartGold", "pokemon-heartgold"),
    ("Ace Attorney", "ace-attorney"),
];

/// The application state.
pub struct OptimeApp {
    audio: Option<AudioEngine>,
    /// Set once audio init has been attempted and failed, so we stop retrying.
    audio_failed: bool,
    sample_rate: f64,

    sdats: Vec<Sdat>,
    songs: Vec<Song>,
    current_song: Option<usize>,

    // UI mirrors of [`SynthConfig`].
    stereo_separation: bool,
    force_stereo_separation: bool,
    bass_mono: bool,
    bass_mono_freq: f32,
    tuning_choice: usize,
    pure_tonic: i32,
    track_enables: [bool; TRACK_COUNT],

    paused: bool,
    status: String,

    /// Cross-thread inbox for asynchronously-loaded file bytes (file picker).
    pending_file: Arc<Mutex<Option<Vec<u8>>>>,
    /// Keys currently held, to debounce auto-repeat for note input.
    held_notes: [bool; 128],
}

impl OptimeApp {
    /// Builds the app and loads the first demo. Native starts audio immediately; web defers it
    /// until the first user gesture (browser autoplay policy — see [`Self::ensure_audio`]).
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let audio = AudioEngine::new();
        #[cfg(target_arch = "wasm32")]
        let audio: Option<AudioEngine> = None;
        let sample_rate = audio.as_ref().map(|a| a.sample_rate).unwrap_or(48_000.0);

        let mut app = Self {
            audio,
            audio_failed: false,
            sample_rate,
            sdats: Vec::new(),
            songs: Vec::new(),
            current_song: None,
            stereo_separation: true,
            force_stereo_separation: true,
            bass_mono: true,
            bass_mono_freq: 200.0,
            tuning_choice: 0,
            pure_tonic: 0,
            track_enables: [true; TRACK_COUNT],
            paused: false,
            status: "Load a ROM, an SDAT, or a demo to begin.".to_owned(),
            pending_file: Arc::new(Mutex::new(None)),
            held_notes: [false; 128],
        };
        app.try_load_first_demo();
        app
    }

    fn try_load_first_demo(&mut self) {
        self.request_demo(DEMOS[0].1, DEMOS[0].0);
    }

    /// Lazily starts the audio engine. On the web the `AudioContext` may only begin after a user
    /// gesture, so creation is deferred until the first interaction; once started, any
    /// already-selected song is (re)loaded into the new engine.
    fn ensure_audio(&mut self, ctx: &egui::Context) {
        if self.audio.is_some() || self.audio_failed {
            return;
        }
        #[cfg(target_arch = "wasm32")]
        {
            let interacted = ctx.input(|i| {
                i.pointer.any_pressed()
                    || i.events
                        .iter()
                        .any(|e| matches!(e, egui::Event::Key { .. } | egui::Event::Text(_)))
            });
            if !interacted {
                return;
            }
        }
        match AudioEngine::new() {
            Some(audio) => {
                self.sample_rate = audio.sample_rate;
                self.audio = Some(audio);
                if let Some(i) = self.current_song {
                    self.play_song(i);
                }
            }
            None => {
                self.audio_failed = true;
                self.status = "No audio output device available; playback disabled.".to_owned();
                log::error!("no audio output device available");
            }
        }
        let _ = ctx;
    }

    /// Loads a demo SDAT. Native reads from `demos/`; web fetches it (copied into the deploy by
    /// Trunk) into [`Self::pending_file`].
    fn request_demo(&mut self, stem: &str, label: &str) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            match std::fs::read(format!("demos/{stem}.sdat")) {
                Ok(bytes) => self.load_bytes(&bytes, label),
                Err(_) => self.status = format!("Demo '{label}' not found in demos/."),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.status = format!("Loading {label}…");
            let inbox = self.pending_file.clone();
            let url = format!("{stem}.sdat");
            wasm_bindgen_futures::spawn_local(async move {
                if let Some(bytes) = crate::web::fetch_bytes(&url).await {
                    if let Ok(mut slot) = inbox.lock() {
                        *slot = Some(bytes);
                    }
                }
            });
        }
    }

    /// The active synth config built from the UI mirrors.
    fn config(&self) -> SynthConfig {
        let tuning = if self.tuning_choice == 0 {
            TuningSystem::Equal
        } else {
            TuningSystem::Pure {
                tonic: self.pure_tonic,
            }
        };
        SynthConfig {
            stereo_separation: self.stereo_separation,
            force_stereo_separation: self.force_stereo_separation,
            bass_mono: self.bass_mono,
            bass_mono_freq: self.bass_mono_freq as f64,
            tuning,
            track_enables: self.track_enables,
        }
    }

    /// Parses SDATs from `bytes` and rebuilds the song list.
    fn load_bytes(&mut self, bytes: &[u8], source: &str) {
        let sdats = Sdat::load_all(bytes);
        if sdats.is_empty() {
            self.status = format!("No SDAT found in {source}.");
            return;
        }
        self.sdats = sdats;
        self.songs.clear();
        for (i, sdat) in self.sdats.iter().enumerate() {
            for &id in &sdat.sseq_list {
                let name = sdat
                    .sseq_id_to_name
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| format!("SSEQ {id}"));
                self.songs.push(Song {
                    sdat_index: i,
                    sseq_id: id,
                    label: format!("{name} (#{id})"),
                });
            }
        }
        self.current_song = None;
        self.status = format!("Loaded {source}: {} songs.", self.songs.len());
    }

    /// Starts playing the song at `index` in the flattened list.
    fn play_song(&mut self, index: usize) {
        let Some(song) = self.songs.get(index) else {
            return;
        };
        let sdat = &self.sdats[song.sdat_index];
        match Controller::new(self.sample_rate, sdat, song.sseq_id) {
            Some(controller) => {
                self.current_song = Some(index);
                self.paused = false;
                self.status = format!("Playing: {}", song.label);
                if let Some(audio) = &self.audio {
                    if let Ok(mut st) = audio.shared.lock() {
                        st.config = self.config();
                        st.paused = false;
                        st.controller = Some(controller);
                    }
                }
            }
            None => self.status = format!("Failed to load: {}", song.label),
        }
    }

    fn restart(&mut self) {
        if let Some(i) = self.current_song {
            self.play_song(i);
        }
    }

    fn step_song(&mut self, delta: isize) {
        if let Some(i) = self.current_song {
            let next = i as isize + delta;
            if next >= 0 && (next as usize) < self.songs.len() {
                self.play_song(next as usize);
            }
        }
    }

    /// Opens a native file dialog; on web spawns the async picker. Loaded bytes arrive via
    /// [`Self::pending_file`].
    fn open_file_dialog(&mut self) {
        let inbox = self.pending_file.clone();
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::spawn(move || {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("DS sound", &["nds", "sdat"])
                    .pick_file()
                {
                    if let Ok(bytes) = std::fs::read(path) {
                        if let Ok(mut slot) = inbox.lock() {
                            *slot = Some(bytes);
                        }
                    }
                }
            });
        }
        #[cfg(target_arch = "wasm32")]
        {
            wasm_bindgen_futures::spawn_local(async move {
                if let Some(handle) = rfd::AsyncFileDialog::new()
                    .add_filter("DS sound", &["nds", "sdat"])
                    .pick_file()
                    .await
                {
                    let bytes = handle.read().await;
                    if let Ok(mut slot) = inbox.lock() {
                        *slot = Some(bytes);
                    }
                }
            });
        }
    }

    /// Drains a pending picked file, if any.
    fn poll_pending_file(&mut self) {
        let bytes = self.pending_file.lock().ok().and_then(|mut s| s.take());
        if let Some(bytes) = bytes {
            self.load_bytes(&bytes, "selected file");
        }
    }

    /// Renders the current song offline to WAV and saves (native) / downloads (web).
    fn export_wav(&mut self) {
        let Some(i) = self.current_song else {
            self.status = "Nothing to export.".to_owned();
            return;
        };
        let song = &self.songs[i];
        let sdat = &self.sdats[song.sdat_index];
        let samples = player::render_to_samples(sdat, song.sseq_id, &self.config());
        let wav = crate::wav::encode_stereo_i16(&samples, player::EXPORT_SAMPLE_RATE);
        let name = song.label.replace([' ', '#', '(', ')', '/'], "_");
        save_bytes(&format!("{name}.wav"), &wav);
        self.status = format!("Exported {name}.wav ({} frames).", samples.len());
    }

    /// Pushes UI config into the audio thread and pulls a note snapshot for the visualizer.
    fn sync_audio(&mut self, ctx: &egui::Context) -> VisSnapshot {
        let config = self.config();
        let mut snap = VisSnapshot::default();
        // Clone the shared handle so we drop the borrow of `self.audio` and can still touch
        // `self.held_notes` while holding the lock.
        let Some(shared) = self.audio.as_ref().map(|a| a.shared.clone()) else {
            return snap;
        };
        let Ok(mut st) = shared.lock() else {
            return snap;
        };
        st.config = config.clone();
        st.paused = self.paused;

        if let Some(controller) = &mut st.controller {
            snap.active = true;
            snap.active_track = controller.active_keyboard_track_num;
            for t in 0..TRACK_COUNT {
                for n in 0..128 {
                    snap.notes_on[t][n] = controller.notes_on[t][n] != 0;
                    snap.notes_kbd[t][n] = controller.notes_on_keyboard[t][n] != 0;
                }
            }
            handle_keyboard(ctx, controller, &config, &mut self.held_notes);
        }
        snap
    }
}

impl eframe::App for OptimeApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.ensure_audio(ctx);
        self.poll_pending_file();

        // Arrow-key song navigation (sequence switching).
        let (left, right) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::ArrowRight),
            )
        });
        if left {
            self.step_song(-1);
        }
        if right {
            self.step_song(1);
        }

        let snap = self.sync_audio(ctx);

        egui::SidePanel::left("songs")
            .resizable(true)
            .default_width(260.0)
            .show(ctx, |ui| {
                ui.heading("Optime Player");
                ui.add_space(4.0);
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Open ROM / SDAT…").clicked() {
                        self.open_file_dialog();
                    }
                });
                ui.collapsing("Demos", |ui| {
                    let mut requested = None;
                    for (label, stem) in DEMOS {
                        if ui.button(*label).clicked() {
                            requested = Some((*stem, *label));
                        }
                    }
                    if let Some((stem, label)) = requested {
                        self.request_demo(stem, label);
                    }
                });
                ui.separator();
                ui.label("Songs");
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let mut to_play = None;
                    for (i, song) in self.songs.iter().enumerate() {
                        let selected = self.current_song == Some(i);
                        if ui.selectable_label(selected, &song.label).clicked() {
                            to_play = Some(i);
                        }
                    }
                    if let Some(i) = to_play {
                        self.play_song(i);
                    }
                });
            });

        egui::TopBottomPanel::top("transport").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let pause_label = if self.paused {
                    "▶ Resume"
                } else {
                    "⏸ Pause"
                };
                if ui.button(pause_label).clicked() {
                    self.paused = !self.paused;
                }
                if ui.button("⟲ Restart").clicked() {
                    self.restart();
                }
                if ui.button("⏮").on_hover_text("Previous (←)").clicked() {
                    self.step_song(-1);
                }
                if ui.button("⏭").on_hover_text("Next (→)").clicked() {
                    self.step_song(1);
                }
                ui.separator();
                if ui.button("💾 Export WAV").clicked() {
                    self.export_wav();
                }
            });
        });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
            });
        });

        egui::SidePanel::right("settings")
            .default_width(220.0)
            .show(ctx, |ui| {
                ui.heading("Settings");
                ui.checkbox(&mut self.stereo_separation, "Stereo separation");
                ui.add_enabled_ui(self.stereo_separation, |ui| {
                    ui.checkbox(&mut self.force_stereo_separation, "Force stereo separation");
                    ui.checkbox(&mut self.bass_mono, "Keep bass centered");
                    ui.add_enabled(
                        self.bass_mono,
                        egui::Slider::new(&mut self.bass_mono_freq, 40.0..=800.0)
                            .text("Bass crossover")
                            .suffix(" Hz")
                            .logarithmic(true),
                    )
                    .on_hover_text(
                        "Frequencies below this stay glued to the center; \
                         mids and treble are widened.",
                    );
                });
                ui.separator();
                ui.label("Tuning system");
                egui::ComboBox::from_id_salt("tuning")
                    .selected_text(if self.tuning_choice == 0 {
                        "Equal temperament"
                    } else {
                        "Pure (Pythagorean)"
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.tuning_choice, 0, "Equal temperament");
                        ui.selectable_value(&mut self.tuning_choice, 1, "Pure (Pythagorean)");
                    });
                if self.tuning_choice == 1 {
                    ui.add(
                        egui::Slider::new(&mut self.pure_tonic, 0..=11)
                            .text("Tonic (semitones from A)"),
                    );
                }
                ui.separator();
                ui.label("Live keyboard");
                ui.label("Click a track row to capture, then play z–m / q–p.");
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::both().show(ui, |ui| {
                let mut active_track = snap.active_track;
                visualizer::draw(ui, &snap, &mut self.track_enables, &mut active_track);
                // Apply any track-selection change back to the controller.
                if active_track != snap.active_track {
                    if let Some(audio) = &self.audio {
                        if let Ok(mut st) = audio.shared.lock() {
                            if let Some(c) = &mut st.controller {
                                c.active_keyboard_track_num = active_track;
                            }
                        }
                    }
                }
            });
        });

        // Keep animating the visualizer.
        ctx.request_repaint();
    }
}

/// Saves bytes to disk (native, via dialog) or triggers a browser download (web).
fn save_bytes(filename: &str, bytes: &[u8]) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Some(path) = rfd::FileDialog::new().set_file_name(filename).save_file() {
            let _ = std::fs::write(path, bytes);
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        crate::web::download(filename, bytes);
    }
}

/// Maps held computer-keyboard keys to live notes on the controller's active keyboard track.
fn handle_keyboard(
    ctx: &egui::Context,
    controller: &mut Controller,
    config: &SynthConfig,
    held_notes: &mut [bool; 128],
) {
    let Some(track) = controller.active_keyboard_track_num else {
        return;
    };
    ctx.input(|i| {
        for event in &i.events {
            if let egui::Event::Key {
                key,
                pressed,
                repeat,
                ..
            } = event
            {
                if *repeat {
                    continue;
                }
                if let Some(note) = key_to_note(*key) {
                    let n = note as usize;
                    if *pressed && !held_notes[n] {
                        held_notes[n] = true;
                        controller.play_keyboard_note(track, note, 127, 2000, config);
                    } else if !*pressed && held_notes[n] {
                        held_notes[n] = false;
                        controller.release_keyboard_note(track, note);
                    }
                }
            }
        }
    });
}

/// Maps an egui key to a MIDI note, mirroring the legacy two-row keyboard layout.
fn key_to_note(key: egui::Key) -> Option<u8> {
    use egui::Key::*;
    Some(match key {
        // Lower row: z = middle C (60).
        Z => 60,
        S => 61,
        X => 62,
        D => 63,
        C => 64,
        V => 65,
        G => 66,
        B => 67,
        H => 68,
        N => 69,
        J => 70,
        M => 71,
        Comma => 72,
        L => 73,
        Period => 74,
        Semicolon => 75,
        Slash => 76,
        // Upper row: q = C (72).
        Q => 72,
        Num2 => 73,
        W => 74,
        Num3 => 75,
        E => 76,
        R => 77,
        Num5 => 78,
        T => 79,
        Num6 => 80,
        Y => 81,
        Num7 => 82,
        U => 83,
        I => 84,
        Num9 => 85,
        O => 86,
        Num0 => 87,
        P => 88,
        OpenBracket => 89,
        Equals => 90,
        CloseBracket => 91,
        _ => return None,
    })
}

//! The egui application: song list, transport, settings, visualizer, and live keyboard input.

use std::sync::{Arc, Mutex};

use optime_core::{Controller, FsVisController, ResampleMode, Sdat, SynthConfig, TuningSystem};

use crate::piano_roll::PianoRoll;
use crate::visualizer::{self, VisSnapshot};
use crate::{audio::AudioEngine, player, TRACK_COUNT};

/// One entry in the flattened song list.
struct Song {
    sdat_index: usize,
    sseq_id: u32,
    label: String,
}

/// Which visualizer the central panel shows.
#[derive(PartialEq, Eq, Clone, Copy)]
enum VisTab {
    /// The streaming FL-Studio-style piano roll.
    PianoRoll,
    /// The legacy per-track keyboard grid (track enables + live-input selection).
    Tracks,
}

/// Demo SDATs available to load. Native reads from `demos/`; web fetches them at runtime.
const DEMOS: &[(&str, &str)] = &[
    ("Super Mario 64 DS", "super-mario-64-ds"),
    ("New Super Mario Bros.", "new-super-mario-bros"),
    ("Pokémon Platinum", "pokemon-platinum"),
    ("Pokémon HeartGold", "pokemon-heartgold"),
    ("Pokémon Black 2", "pokemon-black-2"),
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

    /// Resampling mode index: 0=Nearest, 1=Linear, 2=SincOutputNyquist, 3=SincSampleNyquist.
    resample_choice: usize,
    /// Total source-tap count for the sinc kernel (the kernel spans `sinc_taps` source samples,
    /// i.e. `sinc_taps / 2` per side, regardless of the resampling ratio).
    sinc_taps: usize,

    paused: bool,
    status: String,

    /// When a song ends (sequence finished or looped twice), fade out and advance to the next.
    autoplay: bool,
    /// Pick the next song at random instead of in list order.
    shuffle: bool,
    /// Loops completed by the current song (counted from `Controller::jumps`).
    loop_count: u32,
    /// xorshift64 state for shuffle.
    rng: u64,

    /// Mobile layout: `true` shows the library (playlists + songs) instead of Now Playing.
    mobile_library_open: bool,
    /// Accumulated horizontal swipe distance on the Now Playing view.
    swipe_dx: f32,

    /// Cross-thread inbox for asynchronously-loaded file bytes (file picker).
    pending_file: Arc<Mutex<Option<Vec<u8>>>>,
    /// Keys currently held, to debounce auto-repeat for note input.
    held_notes: [bool; 128],

    /// Whether the crossover-filter analysis popup is open.
    crossover_plot_open: bool,
    /// Whether the sinc-resampler analysis popup is open.
    sinc_plot_open: bool,

    /// Which visualizer tab is active.
    vis_tab: VisTab,
    /// Streaming piano-roll state (note timeline, smoothed scroll clock).
    piano_roll: PianoRoll,
    /// Parallel look-ahead sequence runner feeding upcoming notes to the piano roll.
    look_ahead: Option<FsVisController>,
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
            resample_choice: 2,
            sinc_taps: 32,
            paused: false,
            status: "Load a ROM, an SDAT, or a demo to begin.".to_owned(),
            autoplay: true,
            shuffle: false,
            loop_count: 0,
            rng: 0x9E37_79B9_7F4A_7C15,
            mobile_library_open: false,
            swipe_dx: 0.0,
            pending_file: Arc::new(Mutex::new(None)),
            held_notes: [false; 128],
            crossover_plot_open: false,
            sinc_plot_open: false,
            vis_tab: VisTab::PianoRoll,
            piano_roll: PianoRoll::default(),
            look_ahead: None,
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
        let resample = match self.resample_choice {
            1 => ResampleMode::Linear,
            2 => ResampleMode::SincOutputNyquist {
                half_taps: (self.sinc_taps / 2).max(1),
            },
            3 => ResampleMode::SincSampleNyquist {
                half_taps: (self.sinc_taps / 2).max(1),
            },
            _ => ResampleMode::NearestNeighbor,
        };
        SynthConfig {
            stereo_separation: self.stereo_separation,
            force_stereo_separation: self.force_stereo_separation,
            bass_mono: self.bass_mono,
            bass_mono_freq: self.bass_mono_freq as f64,
            tuning,
            track_enables: self.track_enables,
            resample,
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
                self.loop_count = 0;
                self.piano_roll.clear();
                // Parallel look-ahead runner; we drive it forward ourselves each frame.
                self.look_ahead = FsVisController::new(sdat, song.sseq_id, 0);
                self.status = format!("Playing: {}", song.label);
                if let Some(audio) = &self.audio {
                    if let Ok(mut st) = audio.shared.lock() {
                        st.config = self.config();
                        st.paused = false;
                        st.fade_gain = 1.0;
                        st.fade_step = 0.0;
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

    /// Steps to the previous/next song in queue order: random when shuffle is on, otherwise list
    /// order with wraparound.
    fn step_song(&mut self, delta: isize) {
        let n = self.songs.len();
        if n == 0 {
            return;
        }
        let next = if self.shuffle && n > 1 {
            loop {
                // xorshift64
                self.rng ^= self.rng << 13;
                self.rng ^= self.rng >> 7;
                self.rng ^= self.rng << 17;
                let r = (self.rng % n as u64) as usize;
                if Some(r) != self.current_song {
                    break r;
                }
            }
        } else {
            let cur = self.current_song.unwrap_or(0) as isize;
            (cur + delta).rem_euclid(n as isize) as usize
        };
        self.play_song(next);
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
    /// Returns the snapshot and whether autoplay should advance to the next song.
    fn sync_audio(&mut self, ctx: &egui::Context) -> (VisSnapshot, bool) {
        let config = self.config();
        let mut snap = VisSnapshot::default();
        let mut advance = false;
        // Clone the shared handle so we drop the borrow of `self.audio` and can still touch
        // `self.held_notes` while holding the lock.
        let Some(shared) = self.audio.as_ref().map(|a| a.shared.clone()) else {
            return (snap, advance);
        };
        let Ok(mut st) = shared.lock() else {
            return (snap, advance);
        };
        st.config = config.clone();
        st.paused = self.paused;

        // Autoplay: fade out once the sequence has finished (or looped twice), then advance.
        if self.autoplay {
            if let Some(controller) = &mut st.controller {
                if controller.jumps > 0 {
                    controller.jumps = 0;
                    self.loop_count += 1;
                }
                let ended = std::mem::take(&mut controller.fading_start);
                if st.fade_step == 0.0 && (ended || self.loop_count >= 2) {
                    // 3-second fade at the device sample rate.
                    st.fade_step = 1.0 / (3.0 * self.sample_rate as f32);
                }
            }
            if st.controller.is_some() && st.fade_step > 0.0 && st.fade_gain <= 0.0 {
                advance = true;
            }
        }

        if let Some(controller) = &mut st.controller {
            snap.active = true;
            snap.active_track = controller.active_keyboard_track_num;
            snap.ticks = controller.sequence.ticks_elapsed;
            snap.bpm = controller.sequence.tracks[0].bpm;
            for t in 0..TRACK_COUNT {
                for n in 0..128 {
                    snap.notes_on[t][n] = controller.notes_on[t][n] != 0;
                    snap.notes_kbd[t][n] = controller.notes_on_keyboard[t][n] != 0;
                }
            }
            handle_keyboard(ctx, controller, &config, &mut self.held_notes);
        }
        (snap, advance)
    }

    /// The compact phone layout: a Spotify-style Now Playing screen with swipe navigation, plus
    /// a library view (playlists = demo SDATs, then the song list).
    fn mobile_ui(&mut self, ctx: &egui::Context, snap: &VisSnapshot) {
        egui::TopBottomPanel::top("m_top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.mobile_library_open, false, "🎵 Now Playing");
                ui.selectable_value(&mut self.mobile_library_open, true, "📚 Library");
            });
        });

        if self.mobile_library_open {
            self.mobile_library(ctx);
            return;
        }

        egui::TopBottomPanel::bottom("m_transport").show(ctx, |ui| {
            ui.add_space(6.0);
            // Title + status above the controls.
            let title = self
                .current_song
                .and_then(|i| self.songs.get(i))
                .map(|s| s.label.clone())
                .unwrap_or_else(|| "No song loaded".to_owned());
            ui.vertical_centered(|ui| {
                ui.label(egui::RichText::new(title).strong().size(16.0));
                ui.label(egui::RichText::new(&self.status).weak().size(11.0));
            });
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let spacing = (ui.available_width() - 5.0 * 44.0).max(0.0) / 6.0;
                ui.spacing_mut().item_spacing.x = spacing;
                ui.add_space(spacing);
                let big = egui::vec2(44.0, 36.0);
                let shuffle = ui.add_sized(big, egui::Button::new("🔀").selected(self.shuffle));
                if shuffle.clicked() {
                    self.shuffle = !self.shuffle;
                }
                if ui.add_sized(big, egui::Button::new("⏮")).clicked() {
                    self.step_song(-1);
                }
                let pause_icon = if self.paused || self.current_song.is_none() {
                    "▶"
                } else {
                    "⏸"
                };
                if ui
                    .add_sized(egui::vec2(56.0, 44.0), egui::Button::new(pause_icon))
                    .clicked()
                {
                    self.paused = !self.paused;
                }
                if ui.add_sized(big, egui::Button::new("⏭")).clicked() {
                    self.step_song(1);
                }
                if ui
                    .add_sized(big, egui::Button::new("🔁").selected(self.autoplay))
                    .on_hover_text("Autoplay next song")
                    .clicked()
                {
                    self.autoplay = !self.autoplay;
                }
            });
            ui.add_space(8.0);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // The piano roll doubles as the album art; swipe horizontally to change songs.
            let rect = ui.max_rect();
            self.piano_roll.draw(ui, snap.active);
            let resp = ui.interact(rect, egui::Id::new("swipe"), egui::Sense::drag());
            if resp.dragged() {
                self.swipe_dx += resp.drag_delta().x;
            }
            if resp.drag_stopped() {
                if self.swipe_dx <= -60.0 {
                    self.step_song(1); // swipe left → next
                } else if self.swipe_dx >= 60.0 {
                    self.step_song(-1); // swipe right → previous
                }
                self.swipe_dx = 0.0;
            }
        });
    }

    /// The mobile library: demo playlists, file open, and the song list.
    fn mobile_library(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                if ui.button("📂 Open ROM / SDAT…").clicked() {
                    self.open_file_dialog();
                }
                ui.collapsing("Playlists (demos)", |ui| {
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
                let mut to_play = None;
                for (i, song) in self.songs.iter().enumerate() {
                    let selected = self.current_song == Some(i);
                    if ui.selectable_label(selected, &song.label).clicked() {
                        to_play = Some(i);
                    }
                }
                if let Some(i) = to_play {
                    self.play_song(i);
                    self.mobile_library_open = false;
                }
            });
        });
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

        let (snap, advance) = self.sync_audio(ctx);
        if advance {
            self.step_song(1);
        }

        // Advance the piano roll's smoothed playhead (frozen when paused / no song), then drive
        // the look-ahead runner so it stays buffered ahead of the playhead, and pull its notes.
        let playing = snap.active && !self.paused;
        let dt = ctx.input(|i| i.stable_dt) as f64;
        self.piano_roll.advance(&snap, dt, playing);
        if let Some(look) = &mut self.look_ahead {
            if playing {
                let target = self.piano_roll.display_tick().ceil() as u32
                    + crate::piano_roll::RUN_AHEAD_TICKS;
                // Bounded catch-up so a stalled/zero-BPM sequence can't spin forever.
                let mut guard = 0u32;
                while look.sequence.ticks_elapsed < target && guard < 200_000 {
                    look.tick();
                    guard += 1;
                }
            }
            self.piano_roll.ingest(look);
        }

        // Narrow screens (phones) get the Spotify-style mobile layout.
        if ctx.screen_rect().width() < 600.0 {
            self.mobile_ui(ctx, &snap);
            #[cfg(target_arch = "wasm32")]
            ctx.request_repaint_after(std::time::Duration::from_millis(33));
            #[cfg(not(target_arch = "wasm32"))]
            ctx.request_repaint();
            return;
        }

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
                ui.toggle_value(&mut self.shuffle, "🔀 Shuffle");
                ui.toggle_value(&mut self.autoplay, "🔁 Autoplay")
                    .on_hover_text("Fade out and advance when a song ends or loops twice");
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
                    ui.horizontal(|ui| {
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
                        if ui
                            .add_enabled(self.bass_mono, egui::Button::new("📈"))
                            .on_hover_text("Analyze crossover filter")
                            .clicked()
                        {
                            self.crossover_plot_open = true;
                        }
                    });
                });
                ui.separator();
                ui.label("Resampling");
                ui.horizontal(|ui| {
                    egui::ComboBox::from_id_salt("resample")
                        .selected_text(match self.resample_choice {
                            1 => "Linear",
                            2 => "Sinc – output Nyquist (crunch)",
                            3 => "Sinc – sample Nyquist (clean)",
                            _ => "Nearest (DS hardware)",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.resample_choice,
                                0,
                                "Nearest (DS hardware)",
                            );
                            ui.selectable_value(&mut self.resample_choice, 1, "Linear");
                            ui.selectable_value(
                                &mut self.resample_choice,
                                2,
                                "Sinc – output Nyquist (crunch)",
                            );
                            ui.selectable_value(
                                &mut self.resample_choice,
                                3,
                                "Sinc – sample Nyquist (clean)",
                            );
                        });
                    if ui
                        .add_enabled(self.resample_choice >= 2, egui::Button::new("📈"))
                        .on_hover_text("Analyze sinc kernel")
                        .clicked()
                    {
                        self.sinc_plot_open = true;
                    }
                });
                let is_sinc = self.resample_choice >= 2;
                ui.add_enabled_ui(is_sinc, |ui| {
                    ui.add(
                        egui::Slider::new(&mut self.sinc_taps, 4..=128)
                            .step_by(2.0)
                            .text("Sinc taps")
                            .logarithmic(false),
                    )
                    .on_hover_text(
                        "Number of source samples the kernel spans — fixed regardless of pitch, \
                         so CPU cost is constant per voice. More taps → sharper cutoff and better \
                         stopband rejection, at higher CPU cost.",
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

        // Analysis popups (shown every frame while open).
        crate::filter_plot::show_crossover_window(
            ctx,
            &mut self.crossover_plot_open,
            self.sample_rate,
            self.bass_mono_freq as f64,
        );
        if self.resample_choice >= 2 {
            let resample_mode = self.config().resample;
            crate::filter_plot::show_sinc_window(ctx, &mut self.sinc_plot_open, resample_mode);
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.vis_tab, VisTab::PianoRoll, "🎹 Piano Roll");
                ui.selectable_value(&mut self.vis_tab, VisTab::Tracks, "🎚 Tracks");
            });
            ui.separator();

            match self.vis_tab {
                VisTab::PianoRoll => {
                    self.piano_roll.draw(ui, snap.active);
                }
                VisTab::Tracks => {
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
                }
            }
        });

        // Keep animating the visualizer. On the web, cpal generates audio on the *main thread*
        // (ScriptProcessorNode), so an unthrottled full-rate repaint starves the audio callback
        // and causes dropouts. Cap the visualizer to ~30 fps there to leave the main thread for
        // audio; native keeps the audio callback on its own thread, so repaint as fast as it can.
        #[cfg(target_arch = "wasm32")]
        ctx.request_repaint_after(std::time::Duration::from_millis(33));
        #[cfg(not(target_arch = "wasm32"))]
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

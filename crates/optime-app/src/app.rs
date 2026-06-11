//! The egui application: song list, transport, settings, visualizer, and live keyboard input.

use std::sync::{Arc, Mutex};

use optime_core::{Controller, FsVisController, ResampleMode, Sdat, SynthConfig, TuningSystem};

use crate::library::{Library, Persisted, RepeatMode, TrackRef};
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

/// Cross-thread inbox for asynchronously-loaded file bytes: (source key, bytes).
type FileInbox = Arc<Mutex<Option<(String, Vec<u8>)>>>;

/// The screens reachable from the mobile bottom navigation bar.
#[derive(PartialEq, Eq, Clone, Copy)]
enum MobileTab {
    NowPlaying,
    Library,
    Playlists,
    Settings,
}

/// Which library collection is open in the library browser.
#[derive(PartialEq, Eq, Clone, Copy)]
enum LibraryView {
    /// The list of collections (liked / recent / playlists).
    Root,
    Liked,
    Recent,
    /// A user playlist, by index into `library.playlists`.
    Playlist(usize),
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

    /// What happens when a song ends (after fade-out): stop, advance, or replay.
    repeat: RepeatMode,
    /// Pick the next song at random instead of in list order.
    shuffle: bool,
    /// Master volume (0..=1).
    volume: f32,
    /// Loops completed by the current song (counted from `Controller::jumps`).
    loop_count: u32,
    /// xorshift64 state for shuffle.
    rng: u64,

    /// The user's persistent library (playlists, likes, history).
    library: Library,
    /// Which library collection the browser shows.
    library_view: LibraryView,
    /// Text buffer for the "new playlist" name field.
    new_playlist_name: String,
    /// Source key (demo stem or user file name) of the currently loaded SDATs.
    current_source: String,
    /// When playing from a playlist/collection: its tracks and the current position.
    queue: Option<(Vec<TrackRef>, usize)>,
    /// A track waiting for its source archive to finish loading (cross-source playlist jump
    /// or session restore).
    pending_play: Option<TrackRef>,
    /// Restore-on-launch: start the restored track paused instead of blasting audio.
    resume_paused: bool,

    /// Which mobile screen the bottom navigation has selected.
    mobile_tab: MobileTab,
    /// Horizontal slide of the Now Playing visualizer, in points: follows the finger during a
    /// swipe, then animates back to 0 (the new song sliding in after a committed swipe).
    swipe_offset: f32,
    /// Volume attenuation tied to the swipe: 1.0 centered, → 0.0 as the view leaves the screen.
    swipe_gain: f32,

    /// Cross-thread inbox for asynchronously-loaded file bytes: (source key, bytes).
    pending_file: FileInbox,
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
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let audio = AudioEngine::new();
        #[cfg(target_arch = "wasm32")]
        let audio: Option<AudioEngine> = None;
        let sample_rate = audio.as_ref().map(|a| a.sample_rate).unwrap_or(48_000.0);

        let p: Persisted = cc
            .storage
            .and_then(|s| eframe::get_value(s, eframe::APP_KEY))
            .unwrap_or_default();

        let mut app = Self {
            audio,
            audio_failed: false,
            sample_rate,
            sdats: Vec::new(),
            songs: Vec::new(),
            current_song: None,
            stereo_separation: p.stereo_separation,
            force_stereo_separation: p.force_stereo_separation,
            bass_mono: p.bass_mono,
            bass_mono_freq: p.bass_mono_freq,
            tuning_choice: p.tuning_choice,
            pure_tonic: p.pure_tonic,
            track_enables: [true; TRACK_COUNT],
            resample_choice: p.resample_choice,
            sinc_taps: p.sinc_taps,
            paused: false,
            status: "Load a ROM, an SDAT, or a demo to begin.".to_owned(),
            repeat: p.repeat,
            shuffle: p.shuffle,
            volume: p.volume.clamp(0.0, 1.0),
            loop_count: 0,
            rng: 0x9E37_79B9_7F4A_7C15,
            library: p.library,
            library_view: LibraryView::Root,
            new_playlist_name: String::new(),
            current_source: String::new(),
            queue: None,
            pending_play: None,
            resume_paused: false,
            mobile_tab: MobileTab::NowPlaying,
            swipe_offset: 0.0,
            swipe_gain: 1.0,
            pending_file: Arc::new(Mutex::new(None)),
            held_notes: [false; 128],
            crossover_plot_open: false,
            sinc_plot_open: false,
            vis_tab: VisTab::PianoRoll,
            piano_roll: PianoRoll::default(),
            look_ahead: None,
        };
        // Resume where the last session left off (paused), if the last track was from a demo
        // we can re-fetch; otherwise fall back to the first demo.
        match p.last_track {
            Some(t) if DEMOS.iter().any(|(_, stem)| *stem == t.source) => {
                app.resume_paused = true;
                app.pending_play = Some(t.clone());
                let label = DEMOS
                    .iter()
                    .find(|(_, stem)| *stem == t.source)
                    .map(|(l, _)| *l)
                    .unwrap_or("demo");
                app.request_demo(&t.source, label);
            }
            _ => app.request_demo(DEMOS[0].1, DEMOS[0].0),
        }
        app
    }

    /// The serializable bundle written to eframe storage.
    fn persisted(&self) -> Persisted {
        Persisted {
            library: self.library.clone(),
            shuffle: self.shuffle,
            repeat: self.repeat,
            volume: self.volume,
            last_track: self.current_track_ref(),
            stereo_separation: self.stereo_separation,
            force_stereo_separation: self.force_stereo_separation,
            bass_mono: self.bass_mono,
            bass_mono_freq: self.bass_mono_freq,
            tuning_choice: self.tuning_choice,
            pure_tonic: self.pure_tonic,
            resample_choice: self.resample_choice,
            sinc_taps: self.sinc_taps,
        }
    }

    /// The persistent reference for the song at list index `i`, if it exists.
    fn track_ref(&self, i: usize) -> Option<TrackRef> {
        self.songs.get(i).map(|s| TrackRef {
            source: self.current_source.clone(),
            sseq_id: s.sseq_id,
            label: s.label.clone(),
        })
    }

    /// The persistent reference for the currently playing song.
    fn current_track_ref(&self) -> Option<TrackRef> {
        self.current_song.and_then(|i| self.track_ref(i))
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
                Ok(bytes) => self.load_bytes(&bytes, stem, label),
                Err(_) => self.status = format!("Demo '{label}' not found in demos/."),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.status = format!("Loading {label}…");
            let inbox = self.pending_file.clone();
            let key = stem.to_owned();
            let url = format!("{stem}.sdat");
            wasm_bindgen_futures::spawn_local(async move {
                if let Some(bytes) = crate::web::fetch_bytes(&url).await {
                    if let Ok(mut slot) = inbox.lock() {
                        *slot = Some((key, bytes));
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

    /// Parses SDATs from `bytes` and rebuilds the song list. `key` is the persistent source
    /// identity (demo stem or user file name); `source` is the display name for status text.
    fn load_bytes(&mut self, bytes: &[u8], key: &str, source: &str) {
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
        self.current_source = key.to_owned();
        self.status = format!("Loaded {source}: {} songs.", self.songs.len());

        // A track was waiting for this source (playlist jump / session restore).
        if let Some(t) = self.pending_play.take() {
            if t.source == self.current_source {
                if let Some(i) = self.songs.iter().position(|s| s.sseq_id == t.sseq_id) {
                    self.play_song_keep_queue(i);
                    if std::mem::take(&mut self.resume_paused) {
                        self.paused = true;
                    }
                }
            } else {
                self.pending_play = Some(t);
            }
        }
    }

    /// Starts playing the song at `index`, leaving any playlist queue: navigation reverts to
    /// the full song list.
    fn play_song(&mut self, index: usize) {
        self.queue = None;
        self.play_song_keep_queue(index);
    }

    /// Starts playing the song at `index` in the flattened list without touching the queue.
    fn play_song_keep_queue(&mut self, index: usize) {
        let Some(song) = self.songs.get(index) else {
            return;
        };
        let sdat = &self.sdats[song.sdat_index];
        match Controller::new(self.sample_rate, sdat, song.sseq_id) {
            Some(controller) => {
                self.current_song = Some(index);
                self.paused = false;
                self.loop_count = 0;
                if let Some(t) = self.track_ref(index) {
                    self.library.push_recent(&t);
                }
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
            self.play_song_keep_queue(i);
        }
    }

    /// A pseudo-random index in `0..n`, avoiding `current` when possible (xorshift64).
    fn rand_index(&mut self, n: usize, current: Option<usize>) -> usize {
        loop {
            self.rng ^= self.rng << 13;
            self.rng ^= self.rng >> 7;
            self.rng ^= self.rng << 17;
            let r = (self.rng % n as u64) as usize;
            if n == 1 || Some(r) != current {
                return r;
            }
        }
    }

    /// Steps to the previous/next song in queue order: within the active playlist queue if one
    /// is set, otherwise the full list. Random when shuffle is on, else wraparound order.
    fn step_song(&mut self, delta: isize) {
        if let Some((tracks, pos)) = self.queue.clone() {
            if tracks.is_empty() {
                self.queue = None;
            } else {
                let next = if self.shuffle && tracks.len() > 1 {
                    self.rand_index(tracks.len(), Some(pos))
                } else {
                    (pos as isize + delta).rem_euclid(tracks.len() as isize) as usize
                };
                self.queue = Some((tracks.clone(), next));
                self.play_ref(tracks[next].clone());
                return;
            }
        }
        let n = self.songs.len();
        if n == 0 {
            return;
        }
        let next = if self.shuffle && n > 1 {
            self.rand_index(n, self.current_song)
        } else {
            let cur = self.current_song.unwrap_or(0) as isize;
            (cur + delta).rem_euclid(n as isize) as usize
        };
        self.play_song(next);
    }

    /// Plays a persistent track reference, loading its source archive first if needed (demo
    /// sources are auto-fetched; user files must be re-opened manually).
    fn play_ref(&mut self, t: TrackRef) {
        if t.source == self.current_source {
            if let Some(i) = self.songs.iter().position(|s| s.sseq_id == t.sseq_id) {
                self.play_song_keep_queue(i);
            } else {
                self.status = format!("Track not found in current archive: {}", t.label);
            }
            return;
        }
        if let Some((label, stem)) = DEMOS.iter().find(|(_, stem)| *stem == t.source) {
            let (label, stem) = (*label, *stem);
            self.pending_play = Some(t);
            self.request_demo(stem, label);
        } else {
            self.status = format!("Open '{}' to play: {}", t.source, t.label);
        }
    }

    /// Starts playing `tracks` as the active queue, from position `pos`.
    fn play_queue(&mut self, tracks: Vec<TrackRef>, pos: usize) {
        if tracks.is_empty() {
            return;
        }
        let pos = pos.min(tracks.len() - 1);
        let t = tracks[pos].clone();
        self.queue = Some((tracks, pos));
        self.play_ref(t);
    }

    /// What to do when the current song has finished its fade-out.
    fn handle_song_end(&mut self) {
        match self.repeat {
            RepeatMode::One => self.restart(),
            RepeatMode::All => self.step_song(1),
            RepeatMode::Off => {
                let at_end = match &self.queue {
                    Some((tracks, pos)) => pos + 1 >= tracks.len(),
                    None => self.current_song.is_none_or(|i| i + 1 >= self.songs.len()),
                };
                if at_end && !self.shuffle {
                    // Reload the song (resetting the fade) and leave it paused at the start.
                    self.restart();
                    self.paused = true;
                    self.status = "End of queue.".to_owned();
                } else {
                    self.step_song(1);
                }
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
                    let key = path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "file".to_owned());
                    if let Ok(bytes) = std::fs::read(path) {
                        if let Ok(mut slot) = inbox.lock() {
                            *slot = Some((key, bytes));
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
                    let key = handle.file_name();
                    let bytes = handle.read().await;
                    if let Ok(mut slot) = inbox.lock() {
                        *slot = Some((key, bytes));
                    }
                }
            });
        }
    }

    /// Drains a pending picked/fetched file, if any.
    fn poll_pending_file(&mut self) {
        let pending = self.pending_file.lock().ok().and_then(|mut s| s.take());
        if let Some((key, bytes)) = pending {
            let display = key.clone();
            self.load_bytes(&bytes, &key, &display);
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
        // Swipe attenuation: the song gets quieter as its visualizer slides offscreen.
        st.volume = self.volume * self.swipe_gain;

        // End-of-song: fade out once the sequence has finished (or looped twice), then let
        // [`Self::handle_song_end`] apply the repeat mode.
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

    /// The full song list with like / add-to-playlist context menus (right-click on desktop,
    /// long-press on touch). Returns `true` if a song was started.
    fn song_list_ui(&mut self, ui: &mut egui::Ui) -> bool {
        enum Action {
            Play(usize),
            Like(usize),
            Add(usize, usize),
        }
        let mut action = None;
        for (i, song) in self.songs.iter().enumerate() {
            let selected = self.current_song == Some(i);
            let resp = ui.selectable_label(selected, &song.label);
            if resp.clicked() {
                action = Some(Action::Play(i));
            }
            resp.context_menu(|ui| {
                let liked = self.track_ref(i).is_some_and(|t| self.library.is_liked(&t));
                let like_label = if liked { "💔 Unlike" } else { "❤ Like" };
                if ui.button(like_label).clicked() {
                    action = Some(Action::Like(i));
                    ui.close_menu();
                }
                ui.menu_button("➕ Add to playlist", |ui| {
                    if self.library.playlists.is_empty() {
                        ui.label("No playlists yet — create one in the Library.");
                    }
                    for (p, pl) in self.library.playlists.iter().enumerate() {
                        if ui.button(&pl.name).clicked() {
                            action = Some(Action::Add(i, p));
                            ui.close_menu();
                        }
                    }
                });
            });
        }
        match action {
            Some(Action::Play(i)) => {
                self.play_song(i);
                return true;
            }
            Some(Action::Like(i)) => {
                if let Some(t) = self.track_ref(i) {
                    self.library.toggle_liked(&t);
                }
            }
            Some(Action::Add(i, p)) => {
                if let Some(t) = self.track_ref(i) {
                    let pl = &mut self.library.playlists[p];
                    if !pl.tracks.iter().any(|x| x.same_song(&t)) {
                        pl.tracks.push(t);
                    }
                }
            }
            None => {}
        }
        false
    }

    /// The library browser (liked / recent / playlists). Returns `true` if playback started.
    fn library_ui(&mut self, ui: &mut egui::Ui) -> bool {
        match self.library_view {
            LibraryView::Root => {
                self.library_root_ui(ui);
                false
            }
            view => self.collection_ui(ui, view),
        }
    }

    /// The library root: collection buttons plus playlist management.
    fn library_root_ui(&mut self, ui: &mut egui::Ui) {
        if ui
            .button(format!("❤ Liked Songs ({})", self.library.liked.len()))
            .clicked()
        {
            self.library_view = LibraryView::Liked;
        }
        if ui.button("🕘 Recently Played").clicked() {
            self.library_view = LibraryView::Recent;
        }
        ui.add_space(4.0);
        ui.label("Playlists");
        let mut open = None;
        let mut delete = None;
        for (p, pl) in self.library.playlists.iter().enumerate() {
            ui.horizontal(|ui| {
                if ui
                    .button(format!("🎵 {} ({})", pl.name, pl.tracks.len()))
                    .clicked()
                {
                    open = Some(p);
                }
                if ui
                    .small_button("🗑")
                    .on_hover_text("Delete playlist")
                    .clicked()
                {
                    delete = Some(p);
                }
            });
        }
        if let Some(p) = open {
            self.library_view = LibraryView::Playlist(p);
        }
        if let Some(p) = delete {
            self.library.playlists.remove(p);
        }
        ui.horizontal(|ui| {
            let edit = egui::TextEdit::singleline(&mut self.new_playlist_name)
                .hint_text("New playlist…")
                .desired_width(120.0);
            ui.add(edit);
            let name = self.new_playlist_name.trim().to_owned();
            if ui.button("➕").clicked() && !name.is_empty() {
                self.library.playlists.push(crate::library::Playlist {
                    name,
                    tracks: Vec::new(),
                });
                self.new_playlist_name.clear();
            }
        });
    }

    /// One open collection (liked / recent / a playlist): play all, play/remove single tracks.
    /// Returns `true` if playback started.
    fn collection_ui(&mut self, ui: &mut egui::Ui, view: LibraryView) -> bool {
        let (title, tracks, removable) = match view {
            LibraryView::Liked => ("❤ Liked Songs", self.library.liked.clone(), true),
            LibraryView::Recent => ("🕘 Recently Played", self.library.recent.clone(), false),
            LibraryView::Playlist(p) => match self.library.playlists.get(p) {
                Some(pl) => (pl.name.as_str(), pl.tracks.clone(), true),
                None => {
                    self.library_view = LibraryView::Root;
                    return false;
                }
            },
            LibraryView::Root => return false,
        };
        let title = title.to_owned();
        let mut started = false;
        ui.horizontal(|ui| {
            if ui.button("⬅").clicked() {
                self.library_view = LibraryView::Root;
            }
            ui.label(egui::RichText::new(format!("{title} ({})", tracks.len())).strong());
        });
        if !tracks.is_empty() && ui.button("▶ Play all").clicked() {
            let pos = if self.shuffle {
                self.rand_index(tracks.len(), None)
            } else {
                0
            };
            self.play_queue(tracks.clone(), pos);
            started = true;
        }
        let current = self.current_track_ref();
        let mut play = None;
        let mut remove = None;
        for (i, t) in tracks.iter().enumerate() {
            ui.horizontal(|ui| {
                let here = current.as_ref().is_some_and(|c| c.same_song(t));
                if ui.selectable_label(here, &t.label).clicked() {
                    play = Some(i);
                }
                if removable && ui.small_button("❌").clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = play {
            self.play_queue(tracks.clone(), i);
            started = true;
        }
        if let Some(i) = remove {
            match view {
                LibraryView::Liked => {
                    let t = tracks[i].clone();
                    self.library.liked.retain(|x| !x.same_song(&t));
                }
                LibraryView::Playlist(p) => {
                    if let Some(pl) = self.library.playlists.get_mut(p) {
                        pl.tracks.remove(i);
                    }
                }
                _ => {}
            }
        }
        started
    }

    /// The synthesis settings (shared between the desktop side panel and the mobile tab).
    fn settings_ui(&mut self, ui: &mut egui::Ui) {
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
                    ui.selectable_value(&mut self.resample_choice, 0, "Nearest (DS hardware)");
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
                egui::Slider::new(&mut self.pure_tonic, 0..=11).text("Tonic (semitones from A)"),
            );
        }
    }

    /// The compact phone layout: a bottom navigation bar (Now Playing / Library / Playlists /
    /// Settings) with a floating mini-player above it on every screen except Now Playing.
    fn mobile_ui(&mut self, ctx: &egui::Context, snap: &VisSnapshot) {
        egui::TopBottomPanel::bottom("m_nav")
            .show_separator_line(false)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                let tabs = [
                    (MobileTab::NowPlaying, "▶", "Playing"),
                    (MobileTab::Library, "📚", "Library"),
                    (MobileTab::Playlists, "🎵", "Playlists"),
                    (MobileTab::Settings, "⚙", "Settings"),
                ];
                ui.columns(tabs.len(), |cols| {
                    for (col, (tab, icon, label)) in cols.iter_mut().zip(tabs) {
                        col.vertical_centered(|ui| {
                            let selected = self.mobile_tab == tab;
                            let text = egui::RichText::new(format!("{icon}\n{label}")).size(13.0);
                            let text = if selected { text.strong() } else { text.weak() };
                            if ui.add(egui::Button::new(text).frame(false)).clicked() {
                                self.mobile_tab = tab;
                            }
                        });
                    }
                });
                ui.add_space(4.0);
            });

        if self.mobile_tab != MobileTab::NowPlaying && self.current_song.is_some() {
            egui::TopBottomPanel::bottom("m_mini")
                .show_separator_line(false)
                .show(ctx, |ui| self.mini_player(ui));
        }

        match self.mobile_tab {
            MobileTab::NowPlaying => self.mobile_now_playing(ctx, snap),
            MobileTab::Library => self.mobile_library(ctx),
            MobileTab::Playlists => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.heading("Playlists");
                        ui.add_space(4.0);
                        self.library_ui(ui);
                    });
                });
            }
            MobileTab::Settings => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        self.settings_ui(ui);
                        ui.separator();
                        if ui.button("💾 Export WAV").clicked() {
                            self.export_wav();
                        }
                        ui.add_space(4.0);
                        ui.label(egui::RichText::new(&self.status).weak().size(11.0));
                    });
                });
            }
        }
    }

    /// The Spotify-style floating mini-player: animated EQ bars, song title, and transport;
    /// tapping it opens the Now Playing screen.
    fn mini_player(&mut self, ui: &mut egui::Ui) {
        let height = 48.0;
        let (rect, resp) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), height),
            egui::Sense::click(),
        );
        let painter = ui.painter_at(rect);
        let visuals = ui.visuals();
        let bg = if resp.hovered() {
            visuals.widgets.hovered.bg_fill
        } else {
            visuals.widgets.inactive.bg_fill
        };
        painter.rect_filled(rect, 10.0, bg);

        // Animated EQ bars (frozen when paused).
        let accent = visuals.selection.bg_fill;
        let t = ui.input(|i| i.time);
        let playing = !self.paused;
        let bar_w = 4.0;
        let max_h = height - 16.0;
        for b in 0..4 {
            let phase = b as f64 * 1.3;
            let level = if playing {
                ((t * (4.0 + b as f64 * 0.9) + phase).sin() * 0.5 + 0.5) as f32
            } else {
                0.15
            };
            let h = 4.0 + level * (max_h - 4.0);
            let x = rect.left() + 12.0 + b as f32 * (bar_w + 3.0);
            let bar = egui::Rect::from_min_max(
                egui::pos2(x, rect.center().y + max_h / 2.0 - h),
                egui::pos2(x + bar_w, rect.center().y + max_h / 2.0),
            );
            painter.rect_filled(bar, 2.0, accent);
        }
        if playing {
            ui.ctx().request_repaint();
        }

        // Song title.
        let title = self
            .current_song
            .and_then(|i| self.songs.get(i))
            .map(|s| s.label.clone())
            .unwrap_or_default();
        let text_rect = egui::Rect::from_min_max(
            egui::pos2(rect.left() + 48.0, rect.top()),
            egui::pos2(rect.right() - 88.0, rect.bottom()),
        );
        painter.text(
            text_rect.left_center(),
            egui::Align2::LEFT_CENTER,
            title,
            egui::FontId::proportional(14.0),
            visuals.text_color(),
        );

        // Transport buttons layered on the right edge of the bar.
        let btn = egui::vec2(34.0, 34.0);
        let pause_icon = if self.paused { "▶" } else { "⏸" };
        let pause_rect =
            egui::Rect::from_center_size(egui::pos2(rect.right() - 62.0, rect.center().y), btn);
        if ui
            .put(pause_rect, egui::Button::new(pause_icon).frame(false))
            .clicked()
        {
            self.paused = !self.paused;
        }
        let next_rect =
            egui::Rect::from_center_size(egui::pos2(rect.right() - 26.0, rect.center().y), btn);
        if ui
            .put(next_rect, egui::Button::new("⏭").frame(false))
            .clicked()
        {
            self.step_song(1);
        }

        if resp.clicked() {
            self.mobile_tab = MobileTab::NowPlaying;
        }
    }

    /// The full-screen Now Playing view: visualizer with animated swipe navigation plus the
    /// large transport.
    fn mobile_now_playing(&mut self, ctx: &egui::Context, snap: &VisSnapshot) {
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
                let spacing = (ui.available_width() - 6.0 * 44.0).max(0.0) / 7.0;
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
                let repeat_icon = match self.repeat {
                    RepeatMode::One => "🔂",
                    _ => "🔁",
                };
                if ui
                    .add_sized(
                        big,
                        egui::Button::new(repeat_icon).selected(self.repeat != RepeatMode::Off),
                    )
                    .on_hover_text("Repeat: off / all / one")
                    .clicked()
                {
                    self.repeat = self.repeat.next();
                }
                if let Some(t) = self.current_track_ref() {
                    let liked = self.library.is_liked(&t);
                    let heart = if liked { "❤" } else { "🤍" };
                    if ui
                        .add_sized(big, egui::Button::new(heart).selected(liked))
                        .clicked()
                    {
                        self.library.toggle_liked(&t);
                    }
                }
            });
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("🔊");
                ui.add(
                    egui::Slider::new(&mut self.volume, 0.0..=1.0)
                        .show_value(false)
                        .trailing_fill(true),
                );
            });
            ui.add_space(8.0);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // The piano roll doubles as the album art; swipe horizontally to change songs.
            // While dragging, the roll follows the finger; past the threshold the old song
            // slides out and the next one slides in from the opposite edge, with the volume
            // dipping in proportion to how far offscreen the view is.
            let rect = ui.max_rect();
            let w = rect.width().max(1.0);
            let resp = ui.interact(rect, egui::Id::new("swipe"), egui::Sense::drag());
            let dt = ui.input(|i| i.stable_dt).min(0.1);
            if resp.dragged() {
                self.swipe_offset += resp.drag_delta().x;
            } else if resp.drag_stopped() {
                if self.swipe_offset <= -0.25 * w {
                    // Swiped left → next song slides in from the right.
                    self.step_song(1);
                    self.swipe_offset += w;
                } else if self.swipe_offset >= 0.25 * w {
                    // Swiped right → previous song slides in from the left.
                    self.step_song(-1);
                    self.swipe_offset -= w;
                }
            } else if self.swipe_offset != 0.0 {
                // Spring back / slide in.
                self.swipe_offset *= (-12.0 * dt).exp();
                if self.swipe_offset.abs() < 0.5 {
                    self.swipe_offset = 0.0;
                }
                ui.ctx().request_repaint();
            }
            self.swipe_gain = 1.0 - (self.swipe_offset.abs() / w).clamp(0.0, 1.0);

            let child_rect = rect.translate(egui::vec2(self.swipe_offset, 0.0));
            let mut child = ui.new_child(egui::UiBuilder::new().max_rect(child_rect));
            child.set_clip_rect(rect);
            self.piano_roll.draw(&mut child, snap.active);
        });
    }

    /// The mobile library: file open, demo archives, and the song list. Selecting a song starts
    /// it in the mini-player rather than jumping to the visualizer.
    fn mobile_library(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Library");
                ui.add_space(4.0);
                if ui.button("📂 Open ROM / SDAT…").clicked() {
                    self.open_file_dialog();
                }
                ui.collapsing("Demo archives", |ui| {
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
                ui.label("All songs");
                self.song_list_ui(ui);
            });
        });
    }
}

impl eframe::App for OptimeApp {
    /// Persists the library, playback prefs, and synth settings (native: disk; web: localStorage).
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.persisted());
    }

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
            self.handle_song_end();
        }
        // Re-derived each frame by the Now Playing swipe; full volume everywhere else.
        self.swipe_gain = 1.0;

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

        // Analysis popups (shown every frame while open) — reachable from both layouts.
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
                ui.collapsing("Library", |ui| {
                    self.library_ui(ui);
                });
                ui.separator();
                ui.label("Songs");
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.song_list_ui(ui);
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
                if ui
                    .button(self.repeat.label())
                    .on_hover_text("Repeat mode: off / all / one")
                    .clicked()
                {
                    self.repeat = self.repeat.next();
                }
                if let Some(t) = self.current_track_ref() {
                    let liked = self.library.is_liked(&t);
                    if ui
                        .button(if liked { "❤" } else { "🤍" })
                        .on_hover_text("Like")
                        .clicked()
                    {
                        self.library.toggle_liked(&t);
                    }
                }
                ui.separator();
                ui.label("🔊");
                ui.add(
                    egui::Slider::new(&mut self.volume, 0.0..=1.0)
                        .show_value(false)
                        .trailing_fill(true),
                );
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
                self.settings_ui(ui);
                ui.separator();
                ui.label("Live keyboard");
                ui.label("Click a track row to capture, then play z–m / q–p.");
            });

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

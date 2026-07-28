//! The egui application: song list, transport, settings, visualizer.

use std::sync::{Arc, Mutex};

use optime_core::devices::gba::GbaRom;
use optime_core::{FsVisController, InstrumentResampleMode, SoundData, load_all};

#[cfg(target_arch = "wasm32")]
use crate::web::get_track_ref_from_query_string;
#[cfg(target_arch = "wasm32")]
use crate::web::update_query_string;

use crate::annotation::{AnnotationState, BounceJob};
use crate::chord_data;
use crate::media_controls::{self, MediaAction};
use crate::persisted::{
    InstrumentResampleChoice, PerDeviceSettings, Persisted, PopSmoothingEdge, RepeatMode, SortMode,
    TrackRef,
};
use crate::piano_roll::PianoRoll;
use crate::player::{PlaybackCommand, PlaylistEntry};
use crate::song_names;
use crate::visualizer::{self, VisSnapshot};
use crate::{TRACK_COUNT, audio::AudioEngine, player};

/// The four-option resampling-algorithm combo box, shared by the resampling settings sections.
fn resample_combo(ui: &mut egui::Ui, id_salt: &str, choice: &mut InstrumentResampleChoice) {
    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt(id_salt)
            .selected_text(choice.text())
            .show_ui(ui, |ui| {
                for option in [
                    InstrumentResampleChoice::Nearest,
                    InstrumentResampleChoice::Linear,
                    InstrumentResampleChoice::SincOutputNyquist,
                    InstrumentResampleChoice::SincSampleNyquist,
                ] {
                    let text = option.text();
                    ui.selectable_value(choice, option, text);
                }
            });
    });
}

/// The "Sinc taps" slider, shared by the resampling settings sections (shown for the sinc modes).
fn sinc_taps_slider(ui: &mut egui::Ui, sinc_taps: &mut usize) {
    ui.add(
        egui::Slider::new(sinc_taps, 4..=128)
            .step_by(2.0)
            .text("Sinc taps"),
    )
    .on_hover_text(
        "Number of source samples the kernel spans — fixed regardless of pitch, so CPU cost is \
         constant. More taps → sharper cutoff and better stopband rejection, at higher CPU cost.",
    );
}

/// The "Default" listing order for a **sparse** curated table. Each input is `(sparse_index,
/// native_order)`; songs with a `sparse_index` are *anchored* to that absolute output position and
/// the rest *fill* the remaining slots in native order. Returns the songs' original indices in their
/// new order. Implemented as a merge: at each output slot it emits the next anchored song once its
/// target index is reached (or once no fillers remain), otherwise the next filler — robust to
/// colliding or out-of-range `sparse_index` values (which simply land at the next free slot / the
/// end).
fn sparse_default_order(keys: &[(Option<usize>, usize)]) -> Vec<usize> {
    let n = keys.len();
    let mut anchored: Vec<usize> = (0..n).filter(|&i| keys[i].0.is_some()).collect();
    anchored.sort_by_key(|&i| (keys[i].0.unwrap(), keys[i].1));
    let mut fillers: Vec<usize> = (0..n).filter(|&i| keys[i].0.is_none()).collect();
    fillers.sort_by_key(|&i| keys[i].1);

    let mut order = Vec::with_capacity(n);
    let (mut ai, mut fi) = (0, 0);
    for pos in 0..n {
        let take_anchored =
            ai < anchored.len() && (fi >= fillers.len() || keys[anchored[ai]].0.unwrap() <= pos);
        if take_anchored {
            order.push(anchored[ai]);
            ai += 1;
        } else {
            order.push(fillers[fi]);
            fi += 1;
        }
    }
    order.extend(anchored[ai..].iter().chain(fillers[fi..].iter()).copied());
    order
}

/// Renders a song's id as a dim, monospaced `#123` tag — shared by the library list and the
/// song-name editor so the number is styled identically in both. `min_digits` right-pads the number
/// (monospaced) so a column of tags shares one width and the song names line up; pass 0 for none.
fn song_id_tag(
    ui: &mut egui::Ui,
    song_id: u32,
    color: egui::Color32,
    min_digits: usize,
) -> egui::Response {
    ui.label(
        egui::RichText::new(format!("#{song_id:>min_digits$}"))
            .monospace()
            .color(color),
    )
}

/// The song-title half of a library row: a static iOS-style row (browse list) or an editable name
/// field (song-name editor). Selects which widget [`song_id_and_title`] draws after the id tag.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
enum SongTitle<'a> {
    /// Read-only row with the length, status badges, and selection highlight; click to play.
    Static {
        name: &'a str,
        length: Option<&'a str>,
        badges: &'a [(&'a str, egui::Color32)],
        selected: bool,
    },
    /// Editable single-line name field (the editor); the caller wires up rename/reset.
    Edit { name: &'a mut String },
}

/// The two responses [`song_id_and_title`] hands back: the id tag (for a hover) and the title widget
/// (click-to-play in browse mode, rename/reset in edit mode).
struct SongRow {
    /// Only the native song-name editor reads this (for the curated-state hover).
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    id: egui::Response,
    title: egui::Response,
}

/// Renders one library row's leading `#id` tag and the song title, shared by the browse list and the
/// song-name editor so both lay out and pad the number identically. The caller supplies the
/// surrounding layout (transport/menu controls) and wires up the returned responses.
fn song_id_and_title(
    ui: &mut egui::Ui,
    song_id: u32,
    id_digits: usize,
    id_color: egui::Color32,
    title: SongTitle<'_>,
) -> SongRow {
    let id = song_id_tag(ui, song_id, id_color, id_digits);
    let title = match title {
        SongTitle::Static {
            name,
            length,
            badges,
            selected,
        } => {
            let row_w = ui.available_width();
            crate::theme::ios_row_ext(ui, row_w, None, name, length, badges, selected, false)
        }
        SongTitle::Edit { name } => {
            ui.add(egui::TextEdit::singleline(name).desired_width(f32::INFINITY))
        }
    };
    SongRow { id, title }
}

/// One entry in the flattened song list.
struct Song {
    archive_index: usize,
    song_id: u32,
    /// The bare song name (no `#id` suffix), used as the alphabetical sort key.
    name: String,
    /// The display label (`name (#id)`).
    label: String,
    /// The archive's native listing position, used as a stable tie-breaker for the other sorts and
    /// as the "Default" order for songs with no OST track number.
    order: usize,
    /// The song's position in the game's curated listing order (OST tracks first in album order,
    /// then the rest), when the game has a curated table. In "Default" sort these songs are listed
    /// first, in this order; songs without one follow, in native order.
    ost_order: Option<usize>,
    /// The song's absolute "Default"-sort position when the curated table is **sparse** (only a few
    /// of the game's songs are labeled): the labeled song is placed at this list index among the
    /// unlabeled ones, instead of being grouped at the front by [`Self::ost_order`]. `None` for
    /// dense/album-ordered tables, which keep the front-grouped behavior.
    sparse_index: Option<usize>,
    /// Playback length in seconds, once computed (lazily, in the background).
    length: Option<f64>,
    /// Whether the length has been computed yet (a computed-but-failed length stays `None`).
    length_computed: bool,
    /// Whether this song belongs in the curated song-names JSON: `true` if it came from the loaded
    /// table, or it's been renamed/reordered in "Edit song names" mode. Only these songs are written
    /// back on save, so untouched (e.g. ROM-embedded-name) songs stay out of the file. Only read by
    /// the native-only editor.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    curated: bool,
}

/// Where the song-name editor should move a song (see [`OptimeApp::reposition_song`]). Movement is
/// through the unlabeled slots only; other curated songs stay pinned.
#[cfg(not(target_arch = "wasm32"))]
enum CuratedTarget {
    /// Towards the start by one movable slot.
    Up,
    /// Towards the end by one movable slot.
    Down,
    /// To the movable slot implied by a typed 0-based list position.
    Abs(usize),
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

/// The adjacent song's pre-rendered piano roll, shown sliding in during a swipe.
struct SwipePreview {
    /// +1 = next (dragging left), -1 = previous (dragging right).
    dir: isize,
    /// A roll pre-filled with the target song's opening notes.
    roll: PianoRoll,
    /// The look-ahead runner that filled `roll`; handed to the app on commit.
    look: Option<FsVisController>,
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

/// Demo files available to load. Native reads from `demos/`; web fetches them at runtime.
const DEMOS: &[(&str, &str)] = &[
    ("Super Mario 64 DS", "super-mario-64-ds.sdat"),
    ("New Super Mario Bros.", "new-super-mario-bros.sdat"),
    ("Mother 3", "mother-3.gbaaudio"),
    ("Pokémon Emerald", "pokemon-emerald.gbaaudio"),
    ("Pokémon Platinum", "pokemon-platinum.sdat"),
    ("Pokémon HeartGold", "pokemon-heartgold.sdat"),
    ("Pokémon Black 2", "pokemon-black-2.sdat"),
    ("Ace Attorney", "ace-attorney.sdat"),
];

// Extra, **local-only** demo entries generated at build time from the optional, gitignored
// `local_extras.txt` manifest (see `build.rs`); empty in a clean clone. Defines `LOCAL_DEMOS`.
include!(concat!(env!("OUT_DIR"), "/local_demos.rs"));

/// All demo entries: the committed [`DEMOS`] plus any local-only ones from `local_extras.txt`.
fn all_demos() -> impl Iterator<Item = &'static (&'static str, &'static str)> {
    DEMOS.iter().chain(LOCAL_DEMOS.iter())
}

/// The application state.
pub struct OptimeApp {
    audio: Option<AudioEngine>,
    /// Set once audio init has been attempted and failed, so we stop retrying.
    audio_failed: bool,
    sample_rate: f64,

    archives: Vec<Arc<dyn SoundData>>,
    songs: Vec<Song>,
    current_song: Option<usize>,

    paused: bool,
    status: String,

    /// OS media-transport controls (Bluetooth/keyboard media keys); lazily created once the window
    /// handle is available, `None` if unsupported (e.g. web, or no handle yet).
    media: Option<media_controls::MediaControls>,
    /// Whether media-control creation has been attempted (so it's tried only once).
    media_tried: bool,

    // Live piano-roll track mutes, injected into the per-device config each frame (see
    // [`Self::config`]).
    track_enables: [bool; TRACK_COUNT],

    /// Saved state that persists across sessions
    p: Persisted,

    /// xorshift64 state for shuffle.
    rng: u64,

    /// Cached computed song lengths, keyed by (source key, song id), so re-sorting and
    /// reloading don't recompute. `None` = the length couldn't be determined.
    length_cache: std::collections::HashMap<(String, u32), Option<f64>>,
    /// Set when the song list needs (re)sorting (after a load or a sort-mode change).
    needs_sort: bool,
    /// Which library collection the browser shows.
    library_view: LibraryView,
    /// Text buffer for the "new playlist" name field.
    new_playlist_name: String,
    /// Source key (demo stem or user file name) of the currently loaded archives.
    current_source: String,
    /// The current playlist in *natural* (unshuffled) order — the song list snapshot or a
    /// collection's tracks. The audio thread owns the live (possibly shuffled) order + position;
    /// this is what the UI re-materializes from on a shuffle toggle.
    playlist: Vec<TrackRef>,
    /// Positions in [`Self::playlist`] in the order last sent to the audio thread. It turns that
    /// thread's index back into a natural-order one. Searching by identity cannot do this job: when
    /// the same song sits in the list twice, it finds whichever copy comes first.
    playlist_order: Vec<usize>,
    /// Last `playback.status_gen` the UI reconciled, to detect audio-thread-driven advances.
    last_status_gen: u64,
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
    /// The adjacent song's roll shown while dragging (becomes the live roll on commit).
    swipe_preview: Option<SwipePreview>,
    /// The old song's roll sliding out after a committed swipe: (roll, exit side −1/+1).
    swipe_out: Option<(PianoRoll, f32)>,

    /// Rolling history of the audio-callback DSP load, for the top-bar meter.
    cpu_history: std::collections::VecDeque<f32>,
    /// Rolling history of the active synthesizer voice count.
    voice_history: std::collections::VecDeque<f32>,
    /// Smoothed high-band-compressor gain reduction (dB, ≤ 0), sampled from the audio thread in
    /// [`Self::sync_audio`] for the small GR bar inside the high-band-compressor settings block.
    high_comp_gr_db: f32,

    /// Cross-thread inbox for asynchronously-loaded file bytes: (source key, bytes).
    pending_file: FileInbox,

    /// Which visualizer tab is active.
    vis_tab: VisTab,
    /// Streaming piano-roll state (note timeline, smoothed scroll clock).
    piano_roll: PianoRoll,
    /// Parallel look-ahead sequence runner feeding upcoming notes to the piano roll.
    look_ahead: Option<FsVisController>,
    /// Whole-track note timeline for the overview bar, rendered once when the song loads.
    overview: Option<optime_core::SongOverview>,
    /// GPU texture of [`Self::overview`], (re)built lazily from the active [`egui::Context`].
    overview_tex: Option<egui::TextureHandle>,
    /// Whether the "Stats for Nerds" sample-DC-offset window is open.
    stats_open: bool,
    /// Cached per-sample DC-offset stats for the current song, recomputed when the window opens
    /// or the song changes. The key is the `(archive_index, song_id)` they were computed for.
    stats_cache: Option<((usize, u32), Vec<optime_core::WaveformDcStat>)>,

    /// Session-only (never persisted) "Edit song names" mode: when on, the library list rows become
    /// editable (rename + reorder) and a Save button writes the curated titles + order back to the
    /// source-tree JSON. A maintainer dev-tool, meaningful only when run from the checkout.
    #[cfg(not(target_arch = "wasm32"))]
    song_edit_enabled: bool,
    /// The adjustable target filename the editor's "Save to JSON" writes to (under
    /// `song_names::source_json_dir()`). Seeded from the loaded game's table, then user-editable.
    #[cfg(not(target_arch = "wasm32"))]
    song_edit_filename: String,
    /// The source key [`Self::song_edit_filename`] was seeded for, so it re-seeds when a different
    /// game is loaded (but keeps the user's edits while the same game stays loaded).
    #[cfg(not(target_arch = "wasm32"))]
    song_edit_filename_for: String,

    /// Session-only chord-annotation dev-tool state (see [`crate::annotation`]).
    annotation: AnnotationState,
    /// The chord symbol being typed for the selected span.
    annotation_entry: String,
}

impl OptimeApp {
    /// Builds the app and loads the first demo. Native starts audio immediately; web defers it
    /// until the first user gesture (browser autoplay policy — see [`Self::ensure_audio`]).
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        crate::theme::apply(&cc.egui_ctx);
        #[cfg(not(target_arch = "wasm32"))]
        let audio = AudioEngine::new();
        #[cfg(target_arch = "wasm32")]
        let audio: Option<AudioEngine> = None;
        let sample_rate = audio.as_ref().map(|a| a.sample_rate).unwrap_or(48_000.0);

        let p: Persisted = cc
            .storage
            .and_then(|s| eframe::get_value(s, eframe::APP_KEY))
            .unwrap_or_default();

        #[cfg(target_arch = "wasm32")]
        let track = get_track_ref_from_query_string().or_else(|| p.last_track.clone());
        #[cfg(not(target_arch = "wasm32"))]
        let track = p.last_track.clone();

        let mut app = Self {
            audio,
            audio_failed: false,
            sample_rate,
            archives: Vec::new(),
            songs: Vec::new(),
            current_song: None,
            p,
            track_enables: [true; TRACK_COUNT],
            paused: false,
            media: None,
            media_tried: false,
            status: "Load a ROM, an SDAT, or a demo to begin.".to_owned(),
            rng: 0x9E37_79B9_7F4A_7C15,
            length_cache: std::collections::HashMap::new(),
            needs_sort: false,
            library_view: LibraryView::Root,
            new_playlist_name: String::new(),
            current_source: String::new(),
            playlist: Vec::new(),
            playlist_order: Vec::new(),
            last_status_gen: 0,
            pending_play: None,
            resume_paused: false,
            mobile_tab: MobileTab::NowPlaying,
            swipe_offset: 0.0,
            swipe_gain: 1.0,
            swipe_preview: None,
            swipe_out: None,
            cpu_history: std::collections::VecDeque::new(),
            voice_history: std::collections::VecDeque::new(),
            high_comp_gr_db: 0.0,
            pending_file: Arc::new(Mutex::new(None)),
            vis_tab: VisTab::PianoRoll,
            piano_roll: PianoRoll::default(),
            look_ahead: None,
            overview: None,
            overview_tex: None,
            stats_open: false,
            stats_cache: None,
            #[cfg(not(target_arch = "wasm32"))]
            song_edit_enabled: false,
            #[cfg(not(target_arch = "wasm32"))]
            song_edit_filename: String::new(),
            #[cfg(not(target_arch = "wasm32"))]
            song_edit_filename_for: String::new(),
            annotation: AnnotationState::default(),
            annotation_entry: String::new(),
        };

        // Resume where the last session left off (paused), if the last track was from a demo
        // we can re-fetch; otherwise fall back to the first demo.
        match track {
            Some(t) if all_demos().any(|(_, stem)| *stem == t.source) => {
                app.resume_paused = true;
                app.pending_play = Some(t.clone());
                let label = all_demos()
                    .find(|(_, stem)| *stem == t.source)
                    .map(|(l, _)| *l)
                    .unwrap_or("demo");
                app.request_demo(&t.source, label);
            }
            _ => app.request_demo(DEMOS[0].1, DEMOS[0].0),
        }
        app
    }

    /// The persistent reference for the song at list index `i`, if it exists.
    fn track_ref(&self, i: usize) -> Option<TrackRef> {
        self.songs.get(i).map(|s| TrackRef {
            source: self.current_source.clone(),
            song_id: s.song_id,
            label: s.label.clone(),
        })
    }

    /// The persistent reference for the currently playing song.
    fn current_track_ref(&self) -> Option<TrackRef> {
        self.current_song.and_then(|i| self.track_ref(i))
    }

    /// Whether the current song plays on the GBA (vs the DS). Defaults to the DS when nothing is
    /// loaded, so the settings panel has a sensible target before a song is picked.
    fn current_is_gba(&self) -> bool {
        self.current_song
            .and_then(|i| self.songs.get(i))
            .is_some_and(|s| self.archives[s.archive_index].as_any().is::<GbaRom>())
    }

    /// Human-readable name of the current song's console.
    fn current_device_name(&self) -> &'static str {
        if self.current_is_gba() {
            "Game Boy Advance"
        } else {
            "Nintendo DS"
        }
    }

    /// The persisted synth/audio settings for the current song's console.
    fn device_settings(&self) -> &PerDeviceSettings {
        if self.current_is_gba() {
            &self.p.gba
        } else {
            &self.p.nds
        }
    }

    /// Mutable access to the current console's synth/audio settings (for the settings UI).
    fn device_settings_mut(&mut self) -> &mut PerDeviceSettings {
        if self.current_is_gba() {
            &mut self.p.gba
        } else {
            &mut self.p.nds
        }
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
                    self.play_from_list(i);
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

    /// Recovers web audio after iOS suspends the `AudioContext` on background. cpal owns the
    /// context internally and never resumes it, so when the callback has stalled while we should
    /// be playing we re-`play()` it each frame (cheap) and, on the next user gesture iOS requires,
    /// rebuild the stream over the same shared state so playback continues from where it left off.
    #[cfg(target_arch = "wasm32")]
    fn keep_audio_alive(&mut self, ctx: &egui::Context) {
        let should_play = self.current_song.is_some() && !self.paused;
        let stalled = should_play && self.audio.as_ref().is_some_and(|a| a.callback_age() > 0.4);
        if !stalled {
            return;
        }
        let interacted = ctx.input(|i| {
            i.pointer.any_pressed()
                || i.events.iter().any(|e| {
                    matches!(
                        e,
                        egui::Event::Key { .. }
                            | egui::Event::Text(_)
                            | egui::Event::Touch { .. }
                            | egui::Event::PointerButton { .. }
                    )
                })
        });
        if let Some(audio) = self.audio.as_mut() {
            audio.resume();
            if interacted {
                audio.rebuild();
            }
        }
        // Keep polling (outside of any pointer animation) until the stream recovers.
        ctx.request_repaint();
    }

    /// Loads a demo. Native reads from `demos/`; web fetches it (copied into the deploy by
    /// Trunk) into [`Self::pending_file`].
    fn request_demo(&mut self, stem: &str, label: &str) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            match std::fs::read(format!("demos/{stem}")) {
                Ok(bytes) => self.load_bytes(&bytes, stem, label),
                Err(_) => self.status = format!("Demo '{label}' not found in demos/."),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.status = format!("Loading {label}…");
            let inbox = self.pending_file.clone();
            let key = stem.to_owned();
            let url = stem.to_string();
            wasm_bindgen_futures::spawn_local(async move {
                if let Some(bytes) = crate::web::fetch_bytes(&url).await
                    && let Ok(mut slot) = inbox.lock()
                {
                    *slot = Some((key, bytes));
                }
            });
        }
    }

    /// The active synth config: the current device's persisted settings (which the synthesis layer
    /// consumes directly via its resolver methods) with the live piano-roll track mutes injected.
    fn config(&self) -> PerDeviceSettings {
        let mut c = self.device_settings().clone();
        c.track_enables = self.track_enables;
        c
    }

    /// Parses sound archives from `bytes` (DS `.nds`/`.sdat`, or a GBA ROM) and rebuilds the
    /// song list. Gzip-compressed input (e.g. a `*.sdat.gz` / `*.gbaaudio.gz` demo, or a `.gz` the
    /// user opens) is transparently decompressed first. `key` is the persistent source identity
    /// (demo stem or user file name); `source` is the display name for status text.
    fn load_bytes(&mut self, bytes: &[u8], key: &str, source: &str) {
        // Decompress if the payload is gzip (magic `1f 8b`); otherwise use the bytes as-is.
        let decompressed;
        let bytes: &[u8] = if bytes.starts_with(&[0x1f, 0x8b]) {
            use std::io::Read;
            let mut out = Vec::new();
            if let Err(e) = flate2::read::GzDecoder::new(bytes).read_to_end(&mut out) {
                self.status = format!("Failed to decompress {source}: {e}");
                return;
            }
            decompressed = out;
            &decompressed
        } else {
            bytes
        };
        let archives = load_all(bytes);
        if archives.is_empty() {
            self.status = format!("No songs found in {source} (not an SDAT, NDS, or GBA ROM).");
            return;
        }
        self.archives = archives.into_iter().map(Arc::from).collect();
        self.songs.clear();
        for (i, data) in self.archives.iter().enumerate() {
            // GBA ROMs carry no song names; supply curated titles + OST order by game code.
            let game_code = data
                .as_any()
                .downcast_ref::<GbaRom>()
                .and_then(GbaRom::game_code);
            for id in data.song_ids() {
                let meta = song_names::lookup(Some(key), game_code.as_deref(), id);
                // A curated title (the editor's output) wins over the ROM's embedded name.
                let name = meta
                    .as_ref()
                    .map(|m| m.title.clone())
                    .or_else(|| data.song_name(id))
                    .unwrap_or_else(|| format!("Song {id}"));
                let ost_order = meta.as_ref().map(|m| m.order);
                let sparse_index = meta.as_ref().and_then(|m| m.sparse_index);
                let label = format!("{name} (#{id})");
                // Reuse any previously computed length for this exact song.
                let (length, length_computed) = match self.length_cache.get(&(key.to_owned(), id)) {
                    Some(v) => (*v, true),
                    None => (None, false),
                };
                self.songs.push(Song {
                    archive_index: i,
                    song_id: id,
                    name,
                    label,
                    order: self.songs.len(),
                    ost_order,
                    sparse_index,
                    length,
                    length_computed,
                    curated: ost_order.is_some(),
                });
            }
        }
        self.current_song = None;
        self.current_source = key.to_owned();
        self.needs_sort = true;
        self.status = format!("Loaded {source}: {} songs.", self.songs.len());

        // A track was waiting for this source (playlist jump / session restore).
        if let Some(t) = self.pending_play.take() {
            if t.source == self.current_source {
                // Session restore with no playlist context yet: default to the full song list.
                if self.playlist.is_empty() {
                    self.playlist = (0..self.songs.len())
                        .filter_map(|i| self.track_ref(i))
                        .collect();
                }
                self.start_playlist(Some(t));
                if std::mem::take(&mut self.resume_paused) {
                    self.paused = true;
                }
            } else {
                self.pending_play = Some(t);
            }
        }
    }

    /// Keeps the song list ordered and the length column filled: lazily computes a few song
    /// lengths per frame (cached), and (re)applies the sort once it's safe to. Length sort waits
    /// until every length is known so the list doesn't reshuffle as values trickle in.
    fn update_library_order(&mut self, ctx: &egui::Context) {
        // Compute a bounded number of still-unknown lengths this frame.
        const LENGTH_BUDGET: usize = 6;
        let mut did_work = 0;
        for idx in 0..self.songs.len() {
            if self.songs[idx].length_computed {
                continue;
            }
            let song_id = self.songs[idx].song_id;
            let archive_index = self.songs[idx].archive_index;
            let key = (self.current_source.clone(), song_id);
            let val = match self.length_cache.get(&key) {
                Some(v) => *v,
                None => {
                    let v = self.archives[archive_index].song_length_seconds(song_id);
                    self.length_cache.insert(key, v);
                    v
                }
            };
            self.songs[idx].length = val;
            self.songs[idx].length_computed = true;
            did_work += 1;
            if did_work >= LENGTH_BUDGET {
                break;
            }
        }
        let all_known = self.songs.iter().all(|s| s.length_computed);

        // While editing song names, the list is in manual order — don't let a sort reshuffle it.
        if self.needs_sort && !self.song_edit_active() {
            // Length sort needs every length first; the others can sort right away.
            if self.p.sort_mode != SortMode::Length || all_known {
                self.apply_sort();
                self.needs_sort = false;
            }
        }
        // Keep stepping until all lengths are known (so the column fills and a pending length
        // sort eventually lands), and while a sort is still outstanding.
        if !all_known || self.needs_sort {
            ctx.request_repaint();
        }
    }

    /// Reorders [`Self::songs`] per [`Self::sort_mode`] and [`Self::sort_descending`],
    /// preserving which song is current.
    fn apply_sort(&mut self) {
        use std::cmp::Ordering;
        let current = self
            .current_song
            .and_then(|i| self.songs.get(i))
            .map(|s| (s.archive_index, s.song_id));
        // Copy out so the sort closure doesn't borrow `self` alongside `&mut self.songs`.
        let mode = self.p.sort_mode;
        let desc = self.p.sort_descending;
        if mode == SortMode::Default && self.songs.iter().any(|s| s.sparse_index.is_some()) {
            // Sparse curated table: place each labeled song at its absolute index, with the unlabeled
            // songs filling the gaps in native order (the comparator below would instead clump the
            // labeled songs at the front).
            self.apply_sparse_default(desc);
        } else {
            // The ascending key comparison for the active mode (Default keeps native order).
            let key_cmp = |a: &Song, b: &Song| -> Ordering {
                match mode {
                    // OST tracks first, in album order; everything else after, in native order.
                    SortMode::Default => match (a.ost_order, b.ost_order) {
                        (Some(x), Some(y)) => x.cmp(&y),
                        (Some(_), None) => Ordering::Less,
                        (None, Some(_)) => Ordering::Greater,
                        (None, None) => a.order.cmp(&b.order),
                    },
                    SortMode::Name => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
                    SortMode::Length => a
                        .length
                        .unwrap_or(f64::INFINITY)
                        .partial_cmp(&b.length.unwrap_or(f64::INFINITY))
                        .unwrap_or(Ordering::Equal),
                }
            };
            self.songs.sort_by(|a, b| {
                let primary = key_cmp(a, b);
                let primary = if desc { primary.reverse() } else { primary };
                // Native order breaks ties (and, for Default, is the whole ordering); it follows the
                // ascending/descending direction too so a descending sort fully reverses the list.
                let tie = a.order.cmp(&b.order);
                primary.then(if desc { tie.reverse() } else { tie })
            });
        }
        if let Some((ai, sid)) = current {
            self.current_song = self
                .songs
                .iter()
                .position(|s| s.archive_index == ai && s.song_id == sid);
        }
    }

    /// Reorders [`Self::songs`] for a **sparse** curated table: each song with a `sparse_index` is
    /// placed at that absolute list position, and the remaining (unlabeled) songs fill the gaps in
    /// native order (see [`sparse_default_order`]). `desc` reverses the whole list.
    fn apply_sparse_default(&mut self, desc: bool) {
        let keys: Vec<(Option<usize>, usize)> = self
            .songs
            .iter()
            .map(|s| (s.sparse_index, s.order))
            .collect();
        let mut order = sparse_default_order(&keys);
        if desc {
            order.reverse();
        }

        // Apply the permutation (Song isn't Clone, so move each out of an `Option` slot).
        let mut slots: Vec<Option<Song>> = self.songs.drain(..).map(Some).collect();
        self.songs = order
            .into_iter()
            .map(|i| slots[i].take().unwrap())
            .collect();
    }

    /// Plays the song at flattened-list index `index`, making the whole song list the playlist.
    fn play_from_list(&mut self, index: usize) {
        let Some(start) = self.track_ref(index) else {
            return;
        };
        self.playlist = (0..self.songs.len())
            .filter_map(|i| self.track_ref(i))
            .collect();
        self.start_playlist(Some(start));
    }

    /// Plays `tracks` as the playlist, starting at the track at `pos`. Starting by position means
    /// that when the same song appears several times, playback begins at the copy the user clicked.
    fn play_collection_at(&mut self, tracks: Vec<TrackRef>, pos: usize) {
        if tracks.is_empty() {
            return;
        }
        let pos = pos.min(tracks.len() - 1);
        self.playlist = tracks;
        self.start_playlist_at(Some(pos));
    }

    /// Plays `tracks` from the top (a random entry first when shuffle is on).
    fn play_all_collection(&mut self, tracks: Vec<TrackRef>) {
        if tracks.is_empty() {
            return;
        }
        self.playlist = tracks;
        self.start_playlist(None);
    }

    /// (Re)materializes `self.playlist` (applying shuffle once, here — the whole order is fixed at
    /// this point) and sends it to the audio thread, starting at `start` (or the first entry after
    /// shuffling when `None`). If the chosen start entry's source isn't loaded, defers: fetches the
    /// source and resumes from [`Self::load_bytes`] when it arrives.
    fn start_playlist(&mut self, start: Option<TrackRef>) {
        let start = start.and_then(|s| self.playlist.iter().position(|t| t.same_song(&s)));
        self.start_playlist_at(start);
    }

    /// The same as [`Self::start_playlist`], but starting at a position in [`Self::playlist`]. A
    /// caller that knows which row was clicked should use this: a position names one copy of a
    /// song, where a [`TrackRef`] matches every copy of it.
    fn start_playlist_at(&mut self, start: Option<usize>) {
        self.paused = false;
        if self.playlist.is_empty() {
            return;
        }
        let natural = self.playlist.clone();
        let start = start.map(|s| s.min(natural.len() - 1));
        let order = if self.p.shuffle {
            self.shuffled_order(natural.len(), start)
        } else {
            (0..natural.len()).collect()
        };
        let index = match start {
            Some(s) => order.iter().position(|&i| i == s).unwrap_or(0),
            None => 0,
        };
        let ordered: Vec<TrackRef> = order.iter().map(|&i| natural[i].clone()).collect();
        self.playlist_order = order;
        let start_track = ordered[index].clone();
        if start_track.source != self.current_source {
            // The chosen source isn't loaded — fetch it (demo) and resume on arrival.
            self.pending_play = Some(start_track.clone());
            if let Some((label, stem)) = all_demos().find(|(_, stem)| *stem == start_track.source) {
                let (label, stem) = (*label, *stem);
                self.request_demo(stem, label);
            } else {
                self.status = format!(
                    "Open '{}' to play: {}",
                    start_track.source, start_track.label
                );
            }
            return;
        }
        if self.audio.is_some() {
            let entries = self.build_entries(&ordered);
            self.send_command(PlaybackCommand::SetPlaylist { entries, index });
        } else {
            // No audio engine yet (web defers it until the first user gesture). A `SetPlaylist`
            // command would be dropped here, so reflect the start track in the UI directly — the
            // highlight + piano roll appear immediately, and `ensure_audio` starts playback from
            // `current_song` once the user interacts and the engine comes up.
            self.on_now_playing(start_track);
        }
    }

    /// A fresh shuffle (Fisher–Yates, xorshift64) of the playlist positions `0..len`, with `keep` —
    /// the position that should start playing — lifted out first and put at the front so it plays
    /// immediately and the rest follow shuffled. Shuffling positions rather than tracks keeps two
    /// copies of the same song distinct, so the one the user picked is the one that plays.
    fn shuffled_order(&mut self, len: usize, keep: Option<usize>) -> Vec<usize> {
        let mut out: Vec<usize> = (0..len).filter(|i| Some(*i) != keep).collect();
        for i in (1..out.len()).rev() {
            self.rng ^= self.rng << 13;
            self.rng ^= self.rng >> 7;
            self.rng ^= self.rng << 17;
            let j = (self.rng % (i as u64 + 1)) as usize;
            out.swap(i, j);
        }
        if let Some(k) = keep {
            out.insert(0, k);
        }
        out
    }

    /// Where the song the audio thread is playing sits in [`Self::playlist`], mapped back through
    /// the order last sent so that the right copy is found when a song appears more than once.
    /// Falls back to matching by identity before the audio engine exists.
    fn current_playlist_position(&self) -> Option<usize> {
        if let Some(audio) = &self.audio
            && let Ok(st) = audio.shared.lock()
            && let Some(&natural) = self.playlist_order.get(st.playback.index)
        {
            return Some(natural);
        }
        let cur = self.current_track_ref()?;
        self.playlist.iter().position(|t| t.same_song(&cur))
    }

    /// Maps playlist tracks to [`PlaylistEntry`]s, attaching the loaded source archive (or `None`
    /// for a track whose source isn't loaded — the audio thread then asks the UI to fetch it).
    fn build_entries(&self, tracks: &[TrackRef]) -> Vec<PlaylistEntry> {
        tracks
            .iter()
            .map(|t| PlaylistEntry {
                archive: self.resolve_archive(t),
                track: t.clone(),
            })
            .collect()
    }

    /// The loaded source archive for `t`, or `None` if `t` belongs to a source that isn't loaded.
    fn resolve_archive(&self, t: &TrackRef) -> Option<Arc<dyn SoundData>> {
        if t.source != self.current_source {
            return None;
        }
        self.songs
            .iter()
            .find(|s| s.song_id == t.song_id)
            .map(|s| self.archives[s.archive_index].clone())
    }

    /// Pushes a command plus the latest level state (config / paused / repeat) to the audio thread.
    fn send_command(&mut self, cmd: PlaybackCommand) {
        let config = self.config();
        if let Some(audio) = &self.audio
            && let Ok(mut st) = audio.shared.lock()
        {
            st.config = config;
            st.paused = self.paused;
            st.playback.repeat = self.p.repeat;
            st.playback.commands.push_back(cmd);
        }
    }

    /// Skips to the next (`delta >= 0`) or previous entry.
    fn send_transport(&mut self, delta: isize) {
        self.paused = false;
        self.send_command(if delta >= 0 {
            PlaybackCommand::Next
        } else {
            PlaybackCommand::Prev
        });
    }

    /// Restarts the currently-playing entry from the top.
    fn restart(&mut self) {
        self.paused = false;
        // Annotation mode parks the playlist entirely, so a `PlayAt` would just sit in the queue
        // unread: restarting there means rewinding the bounce, which is an index.
        if self.annotation.enabled {
            self.annotation_seek(0.0);
            if let Some(audio) = &self.audio
                && let Ok(mut st) = audio.shared.lock()
            {
                st.paused = false;
                // The chord under the top of the song must strike again even if the playhead was
                // already inside that same section.
                if let Some(c) = &mut st.bounce.chords {
                    c.reset();
                }
            }
            return;
        }
        let config = self.config();
        if let Some(audio) = &self.audio
            && let Ok(mut st) = audio.shared.lock()
        {
            let i = st.playback.index;
            st.config = config;
            st.paused = false;
            st.playback.repeat = self.p.repeat;
            st.playback.commands.push_back(PlaybackCommand::PlayAt(i));
        }
    }

    /// Re-materializes the playlist order for the new shuffle state, keeping the current song
    /// playing (no restart). Call *after* flipping `self.p.shuffle`.
    fn toggle_shuffle(&mut self) {
        if self.playlist.is_empty() {
            return;
        }
        let cur = self.current_playlist_position();
        let natural = self.playlist.clone();
        let order = if self.p.shuffle {
            self.shuffled_order(natural.len(), cur)
        } else {
            (0..natural.len()).collect()
        };
        let index = cur
            .and_then(|c| order.iter().position(|&i| i == c))
            .unwrap_or(0);
        let ordered: Vec<TrackRef> = order.iter().map(|&i| natural[i].clone()).collect();
        self.playlist_order = order;
        let entries = self.build_entries(&ordered);
        self.send_command(PlaybackCommand::Reorder { entries, index });
    }

    /// Pushes the (already-updated) repeat mode to the audio thread.
    fn push_repeat(&mut self) {
        if let Some(audio) = &self.audio
            && let Ok(mut st) = audio.shared.lock()
        {
            st.playback.repeat = self.p.repeat;
        }
    }

    /// Drives the OS media-transport controls: lazily creates them once the window handle is
    /// available, applies any transport commands the user triggered (media keys / AirPods taps),
    /// then pushes the current track + playback state to the system "now playing" display.
    fn handle_media_controls(&mut self, ctx: &egui::Context, frame: &eframe::Frame) {
        if !self.media_tried {
            self.media_tried = true;
            self.media = media_controls::MediaControls::new(ctx, frame);
        }
        let Some(actions) = self.media.as_mut().map(|m| m.poll()) else {
            return;
        };
        for action in actions {
            match action {
                MediaAction::Next => self.send_transport(1),
                MediaAction::Prev => self.send_transport(-1),
                MediaAction::PlayPause => self.paused = !self.paused,
                MediaAction::Play => self.paused = false,
                MediaAction::Pause | MediaAction::Stop => self.paused = true,
            }
        }
        let title = self
            .current_song
            .and_then(|i| self.songs.get(i))
            .map(|s| s.name.clone());
        let playing = !self.paused && self.current_song.is_some();
        if let (Some(title), Some(media)) = (title, self.media.as_mut()) {
            media.set_now_playing(&title, "Optime Player", playing);
        }
    }

    /// Builds the swipe preview for the entry a step in `delta` direction would land on (read from
    /// the audio-owned playlist, shuffle already applied): a piano roll pre-filled with that song's
    /// opening notes via its own look-ahead runner.
    fn build_swipe_preview(&mut self, delta: isize) -> Option<SwipePreview> {
        // Resolve the adjacent entry to an archive + song within the loaded source (a track from
        // another source previews as an empty roll).
        let resolved = {
            let shared = self.audio.as_ref().map(|a| a.shared.clone())?;
            let st = shared.lock().ok()?;
            let n = st.playback.entries.len();
            if n == 0 {
                None
            } else {
                let target = (st.playback.index as isize + delta).rem_euclid(n as isize) as usize;
                let t = &st.playback.entries[target].track;
                if t.source == self.current_source {
                    self.songs
                        .iter()
                        .find(|s| s.song_id == t.song_id)
                        .map(|s| (s.archive_index, s.song_id))
                } else {
                    None
                }
            }
        };
        let mut roll = PianoRoll::default();
        let look = resolved.and_then(|(archive_index, song_id)| {
            let mut look = FsVisController::new(&*self.archives[archive_index], song_id)?;
            // Pre-buffer the opening notes (bounded like the live look-ahead drive).
            let mut guard = 0u32;
            while look.steps_elapsed() < crate::piano_roll::RUN_AHEAD_TICKS && guard < 200_000 {
                look.tick();
                guard += 1;
            }
            roll.ingest(&look);
            roll.set_chords(self.chord_spans(archive_index, song_id));
            Some(look)
        });
        Some(SwipePreview {
            dir: delta,
            roll,
            look,
        })
    }

    /// Reflects advances the audio thread performed on its own (so auto-advance / fade-out keep
    /// working while the UI's repaint loop is frozen — e.g. a hidden tab): when the audio thread's
    /// `status_gen` changes, update the highlight, look-ahead, status, recents and media session to
    /// match its authoritative `playback.index`. Does not touch the audio controller.
    fn reconcile_playback(&mut self) {
        let Some(shared) = self.audio.as_ref().map(|a| a.shared.clone()) else {
            return;
        };
        let (needs_ui, stopped, track) = {
            let Ok(mut st) = shared.lock() else {
                return;
            };
            if st.playback.status_gen == self.last_status_gen {
                return;
            }
            self.last_status_gen = st.playback.status_gen;
            let needs_ui = std::mem::take(&mut st.playback.needs_ui);
            let stopped = std::mem::take(&mut st.playback.stopped);
            let track = st
                .playback
                .entries
                .get(st.playback.index)
                .map(|e| e.track.clone());
            (needs_ui, stopped, track)
        };
        if needs_ui {
            // The audio thread couldn't decode this entry. If its source isn't loaded, fetch it
            // and resume; if the source *is* loaded the song is simply missing — stop, rather than
            // re-sending it and looping (each retry would re-trigger `needs_ui`).
            if let Some(t) = track {
                if t.source == self.current_source {
                    self.paused = true;
                    self.status = format!("Track not found: {}", t.label);
                } else {
                    self.start_playlist(Some(t));
                }
            }
            return;
        }
        if stopped {
            // End of queue under RepeatMode::Off.
            self.paused = true;
            self.status = "End of queue.".to_owned();
            return;
        }
        if let Some(t) = track {
            self.on_now_playing(t);
        }
    }

    /// Pre-inferred chord-lane spans for a song: `(start_step, end_step, label)`, the label already
    /// baked as `"<roman> (<name>)"` (or `"N.C."`) offline. Empty when the game has no chord table
    /// (only Pokémon Emerald ships one), so the roll hides the lane.
    fn chord_spans(&self, archive_index: usize, song_id: u32) -> Vec<crate::piano_roll::LaneSpan> {
        let game_code = self.archives[archive_index]
            .as_any()
            .downcast_ref::<GbaRom>()
            .and_then(GbaRom::game_code);
        let Some(chords) =
            chord_data::lookup(Some(&self.current_source), game_code.as_deref(), song_id)
        else {
            return Vec::new();
        };
        chords
            .segments
            .iter()
            .map(|s| crate::piano_roll::LaneSpan {
                start: s.start_step,
                end: s.end_step,
                label: s.label.clone(),
                // Pre-inferred: text only. Only hand-authored spans get a block.
                fill: None,
            })
            .collect()
    }

    /// Updates the UI visuals for a newly-playing track. The audio thread already installed the
    /// controller; this only refreshes the highlight, look-ahead, overview, status and history.
    fn on_now_playing(&mut self, t: TrackRef) {
        self.p.library.push_recent(&t);
        self.piano_roll.clear();
        self.look_ahead = None;
        self.overview = None;
        self.overview_tex = None;
        let idx = if t.source == self.current_source {
            self.songs.iter().position(|s| s.song_id == t.song_id)
        } else {
            None
        };
        self.current_song = idx;
        if let Some(i) = idx {
            let (archive_index, song_id, label) = {
                let s = &self.songs[i];
                (s.archive_index, s.song_id, s.label.clone())
            };
            self.look_ahead = FsVisController::new(&*self.archives[archive_index], song_id);
            self.status = format!("Playing: {label}");
            self.overview = FsVisController::overview(&*self.archives[archive_index], song_id);
            self.piano_roll
                .set_chords(self.chord_spans(archive_index, song_id));
            // The bar grid comes from the device's steps-per-beat; the meter/pickup are the
            // annotator's (defaults until a song's annotation supplies them).
            if let Some(ov) = &self.overview {
                let grid = self.annotation.grid_for(song_id, ov.steps_per_beat);
                self.piano_roll.set_grid(grid);
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = update_query_string(t);
        }
    }

    /// The per-frame annotation sync: collect a finished bounce, follow the transport's playhead,
    /// and push the hand-authored spans into the roll's chord lane.
    fn sync_annotation(&mut self, ctx: &egui::Context) {
        self.poll_bounce(ctx);
        // The song can change under annotation mode (the library list still works).
        if let Some(i) = self.current_song {
            let song_id = self.songs[i].song_id;
            if self.annotation.bounce_for != Some(song_id) {
                self.request_bounce();
                if let Some(ov) = &self.overview {
                    let total = ov.total_steps;
                    self.piano_roll.zoom_to_fit(total);
                }
            }
        }
        if let Some(ov) = &self.overview {
            self.piano_roll.ingest_overview(ov);
        }
        // The app's own transport owns play/pause here too — annotation adds no second button, so
        // there is exactly one source of truth for "is it rolling".
        let rolling = !self.paused && self.annotation.bounce.is_some();
        let pos = self
            .audio
            .as_ref()
            .and_then(|a| {
                a.shared.lock().ok().map(|mut st| {
                    st.bounce.playing = rolling;
                    st.bounce.pos
                })
            })
            .unwrap_or(0);
        if let Some(b) = &self.annotation.bounce {
            self.piano_roll
                .set_display_tick(b.step_of_frame(pos) as f64);
        }
        if rolling {
            ctx.request_repaint();
        }
        // Follow the playhead with the chord voice. Driven from the UI frame rather than the audio
        // callback, which has no idea what a span is: chords change at most once a bar (~2 s), so
        // 16 ms of granularity is far below anything audible.
        if let Some(song_id) = self.current_song.map(|i| self.songs[i].song_id) {
            let step = self.piano_roll.display_tick();
            let chord = self.chord_at(song_id, step);
            if let Some(audio) = &self.audio
                && let Ok(mut st) = audio.shared.lock()
                && let Some(c) = &mut st.bounce.chords
            {
                if rolling {
                    c.set_chord(chord);
                } else {
                    // Stopped: let go of the section chord (it is held until the section changes,
                    // and a stopped playhead never will) and forget it, so resuming re-sounds it.
                    c.stop_following();
                }
            }
        }

        // The lane shows what has been *annotated*, not what a model guessed — this set is the
        // eval data, so a prediction must never appear alongside it and get mistaken for a label.
        let spans: Vec<crate::piano_roll::LaneSpan> = self
            .current_song
            .map(|i| self.songs[i].song_id)
            .and_then(|id| self.annotation.song(id))
            .map(|a| {
                a.spans
                    .iter()
                    .map(|s| {
                        let (label, fill) = match s.chord {
                            Some(c) => (
                                c.symbol(),
                                crate::piano_roll::chord_color(
                                    Some(c.root),
                                    // Uncertain quality reads dimmer: the ribbon should show which
                                    // labels are only half-trusted without opening anything.
                                    c.quality_uncertain,
                                ),
                            ),
                            None => (
                                "N.C.".to_string(),
                                crate::piano_roll::chord_color(None, false),
                            ),
                        };
                        crate::piano_roll::LaneSpan {
                            start: s.start_step as f64,
                            end: s.end_step as f64,
                            label,
                            fill: Some(fill),
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        self.piano_roll.set_chords(spans);
        self.stash_annotations();
    }

    /// Snaps a step to the grid: bars by default, beats when the annotator asks.
    fn snap_step(&self, step: f64) -> f64 {
        let g = self.piano_roll.grid();
        let unit = self.snap_unit();
        if unit <= 0.0 {
            return step.max(0.0);
        }
        let rel = step - g.offset_steps;
        (g.offset_steps + (rel / unit).round() * unit).max(0.0)
    }

    /// The snap unit in steps: a bar, or a beat when beat-snap is on.
    fn snap_unit(&self) -> f64 {
        let g = self.piano_roll.grid();
        if self.annotation.beat_snap {
            g.steps_per_beat
        } else {
            g.bar_steps().unwrap_or(g.steps_per_beat)
        }
    }

    /// The start of the snap unit *containing* `step` — floor, not round.
    ///
    /// Distinct from [`Self::snap_step`] on purpose: dragging an edge wants the *nearest* boundary,
    /// but clicking a spot means "this bar", and rounding would grab the next bar over from
    /// anywhere past its midpoint.
    fn snap_floor(&self, step: f64) -> f64 {
        let g = self.piano_roll.grid();
        let unit = self.snap_unit();
        if unit <= 0.0 {
            return step.max(0.0);
        }
        let rel = step - g.offset_steps;
        (g.offset_steps + (rel / unit).floor() * unit).max(0.0)
    }

    /// Interprets the roll's pointer input as annotation edits.
    ///
    /// Dragging in the chord lane defines a snapped range; releasing selects it (creating the span
    /// on first label). Clicking elsewhere moves the playhead. The roll reports geometry only — the
    /// model lives here.
    fn annotation_roll_input(&mut self, input: crate::piano_roll::RollInput) {
        let Some(song_id) = self.current_song.map(|i| self.songs[i].song_id) else {
            return;
        };
        // A drag paints a range; show it live, then open the picker on release. `drag_step` (not
        // `hover_step`) tracks the pointer, so a drag that wanders off the roll still resolves.
        if let (Some(a), Some(b)) = (input.drag_start_step, input.drag_step)
            && (input.dragging || input.drag_stopped)
        {
            let (s, e) = (self.snap_step(a), self.snap_step(b));
            let (s, e) = (s.min(e), s.max(e));
            // A drag with no movement still selects one whole snap unit.
            let e = if e > s { e } else { self.snap_next(s) };
            self.piano_roll.set_selection(Some((s, e)));
            if input.drag_stopped {
                self.select_or_create_span(song_id, s, e);
                // The picker comes to the region just dragged, rather than making the eye travel to
                // a toolbar and back for every bar.
                self.annotation.picker_at = input.pointer_pos;
                self.annotation.picker_just_opened = true;
            }
            return;
        }
        // A plain right-click (no drag) opens the picker without needing a drag at all — the common
        // case is one bar, and dragging a single bar to label it is busywork.
        if input.secondary_clicked
            && let Some(step) = input.hover_step
        {
            // On an existing span, take that span's whole range: right-clicking a block means "this
            // chord", and re-labelling or clearing it should act on what you clicked, not on
            // whichever grid unit happens to sit under the cursor.
            let existing = self.annotation.song(song_id).and_then(|a| {
                a.spans
                    .iter()
                    .find(|sp| (sp.start_step as f64) <= step && (sp.end_step as f64) > step)
                    .map(|sp| (sp.start_step, sp.end_step))
            });
            let (s, e) = existing.unwrap_or_else(|| {
                let s = self.snap_floor(step);
                (s as u32, self.snap_next(s) as u32)
            });
            self.select_or_create_span(song_id, s as f64, e as f64);
            self.piano_roll.set_selection(Some((s as f64, e as f64)));
            self.annotation.picker_at = input.pointer_pos;
            self.annotation.picker_just_opened = true;
            // Right-clicking a labelled bar plays its chord: the fastest possible check of whether
            // what you wrote is what you meant.
            if let Some(section) = self.chord_at(song_id, step) {
                self.strike_chord(section);
            }
            return;
        }
        // Left button is purely the transport: click or drag anywhere to scrub.
        if let Some(step) = input.scrub_step {
            self.annotation_seek(step);
        }
    }

    /// The next snap boundary after `step`.
    fn snap_next(&self, step: f64) -> f64 {
        step + self.snap_unit().max(1.0)
    }

    /// Selects the span covering `[start, end)`, or remembers the range so the next typed chord
    /// creates one. Blank-slate authoring: nothing is written until a label is given.
    fn select_or_create_span(&mut self, song_id: u32, start: f64, end: f64) {
        let (s, e) = (start as u32, end as u32);
        let existing = self.annotation.song(song_id).and_then(|a| {
            a.spans
                .iter()
                .position(|sp| sp.start_step < e && sp.end_step > s)
        });
        self.annotation.pending_range = Some((s, e));
        // Seed the entry box with whatever is already there, so retyping a chord is an edit.
        self.annotation_entry = existing
            .and_then(|i| self.annotation.song(song_id).map(|a| a.spans[i]))
            .map(|sp| match sp.chord {
                Some(c) => c.symbol(),
                None => "NC".into(),
            })
            .unwrap_or_default();
    }

    /// Applies the typed chord symbol to the selected range.
    fn apply_typed_chord(&mut self, song_id: u32) {
        let Some((s, e)) = self.annotation.pending_range else {
            self.annotation.status = "select a range first".into();
            return;
        };
        // A typo must not overwrite a good label with a wrong one.
        let Some(chord) = crate::annotation::model::parse_chord(&self.annotation_entry) else {
            self.annotation.status = format!("unrecognised chord {:?}", self.annotation_entry);
            return;
        };
        self.annotation.insert_span(
            song_id,
            crate::annotation::model::Span {
                start_step: s,
                end_step: e,
                chord,
            },
        );
        self.annotation.status.clear();
        // Step on to the next range so labelling a run of bars is type-enter-type-enter.
        let next_s = e;
        let next_e = self.snap_next(next_s as f64) as u32;
        self.annotation.pending_range = Some((next_s, next_e));
        self.piano_roll
            .set_selection(Some((next_s as f64, next_e as f64)));
    }

    /// The annotation toolbar: transport, meter, grid offset, snap, chord entry, save.
    fn annotation_toolbar_ui(&mut self, ui: &mut egui::Ui) {
        let Some(song_id) = self.current_song.map(|i| self.songs[i].song_id) else {
            ui.label("no song loaded");
            return;
        };
        let ready = self.annotation.bounce.is_some();

        ui.horizontal_wrapped(|ui| {
            // --- Transport -------------------------------------------------------------------
            // No play/pause here: the app's own transport button drives the bounce (see
            // `sync_annotation`). A second one would be a second source of truth for the same state.
            let mut looping = self
                .audio
                .as_ref()
                .and_then(|a| {
                    a.shared
                        .lock()
                        .ok()
                        .map(|st| st.bounce.loop_range.is_some())
                })
                .unwrap_or(false);
            if ui
                .add_enabled(ready, egui::Checkbox::new(&mut looping, "Loop selection"))
                .on_hover_text("Repeat the selected range — seamless, because the song is bounced.")
                .changed()
            {
                self.set_annotation_loop(looping);
            }
            ui.separator();

            // --- Grid ------------------------------------------------------------------------
            let mut bpb = self.annotation.grid_for(song_id, 0.0).beats_per_bar;
            if ui
                .add(
                    egui::DragValue::new(&mut bpb)
                        .range(2..=12)
                        .prefix("meter "),
                )
                .on_hover_text("Beats per bar.")
                .changed()
            {
                self.annotation.song_mut(song_id).beats_per_bar = bpb.clamp(2, 12);
                self.annotation.dirty = true;
                self.refresh_grid(song_id);
            }
            if ui
                .button("Bar 1 here")
                .on_hover_text(
                    "Move the grid so bar 1 starts at the playhead — for songs that open with a \
                     pickup.",
                )
                .clicked()
            {
                let t = self.piano_roll.display_tick().max(0.0);
                let g = self.piano_roll.grid();
                // Keep the offset inside one bar; the grid repeats, so a whole-bar shift is a no-op.
                let bar = g.bar_steps().unwrap_or(1.0);
                self.annotation.song_mut(song_id).grid_offset_steps = (t % bar).round() as i64;
                self.annotation.dirty = true;
                self.refresh_grid(song_id);
            }
            if ui.button("Reset grid").clicked() {
                self.annotation.song_mut(song_id).grid_offset_steps = 0;
                self.annotation.dirty = true;
                self.refresh_grid(song_id);
            }
            ui.checkbox(&mut self.annotation.beat_snap, "Beat snap")
                .on_hover_text("Snap to beats instead of bars (for a bar with two chords).");
            let mut chords_on = self
                .audio
                .as_ref()
                .and_then(|a| a.shared.lock().ok().map(|st| st.bounce.chords_on))
                .unwrap_or(false);
            if ui
                .checkbox(&mut chords_on, "Play chords")
                .on_hover_text(
                    "Play the annotated chord over the song, on a piano lifted from the ROM. \
                     Off by default — it fights busy passages, and sometimes the track alone is \
                     easier to judge.",
                )
                .changed()
                && let Some(audio) = &self.audio
                && let Ok(mut st) = audio.shared.lock()
            {
                st.bounce.chords_on = chords_on;
            }
            // Chord audition level — a quiet reference instrument can be lifted, a loud one tamed,
            // without relabelling. Read on every output frame, so it needs no re-strike.
            let prev_gain = self.annotation.chord_gain;
            ui.add(
                egui::Slider::new(&mut self.annotation.chord_gain, 0.0..=1.0)
                    .fixed_decimals(2)
                    .text("chord vol"),
            )
            .on_hover_text("Audition gain for chord playback.");
            if self.annotation.chord_gain != prev_gain
                && let Some(audio) = &self.audio
                && let Ok(mut st) = audio.shared.lock()
                && let Some(c) = &mut st.bounce.chords
            {
                c.set_gain(self.annotation.chord_gain);
            }
            // Which inversion chords voice in. Re-voices the sounding chord at once, so flipping
            // through root / 1st / 2nd / 3rd is itself an ear check.
            let mut inv = self.annotation.chord_inversion;
            egui::ComboBox::from_label("inversion")
                .selected_text(match inv {
                    1 => "1st",
                    2 => "2nd",
                    3 => "3rd",
                    _ => "root",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut inv, 0, "root");
                    ui.selectable_value(&mut inv, 1, "1st");
                    ui.selectable_value(&mut inv, 2, "2nd");
                    ui.selectable_value(&mut inv, 3, "3rd");
                });
            if inv != self.annotation.chord_inversion {
                self.annotation.chord_inversion = inv;
                if let Some(audio) = &self.audio
                    && let Ok(mut st) = audio.shared.lock()
                    && let Some(c) = &mut st.bounce.chords
                {
                    c.set_inversion(inv);
                }
            }
            ui.separator();
        });

        ui.horizontal_wrapped(|ui| {
            let (done, total) = self.annotation.labeled_bars(
                song_id,
                self.piano_roll.grid(),
                self.overview.as_ref().map(|o| o.total_steps).unwrap_or(0),
            );
            ui.label(
                egui::RichText::new(format!("{done}/{total} bars"))
                    .monospace()
                    .color(crate::theme::TEXT_DIM),
            );
            if self.annotation.dirty {
                ui.label(egui::RichText::new("● unsaved").color(crate::theme::ACCENT));
            }
            if ui.button("💾 Save to JSON").clicked() {
                self.save_annotations();
            }
            if !self.annotation.status.is_empty() {
                ui.label(
                    egui::RichText::new(&self.annotation.status)
                        .small()
                        .color(crate::theme::TEXT_DIM),
                );
            }
        });
    }

    /// The chord picker: pops up at the region you just dragged, listing every chord you can use.
    ///
    /// A 12×10 grid of every root × every quality — one click per label, no typing, no menu diving.
    /// It shows the *whole* vocabulary on purpose: the set is exactly what the ml label space can
    /// represent, so anything absent here is a chord the model could never be scored on, and the
    /// annotator should never have to wonder which those are.
    fn chord_picker_ui(&mut self, ctx: &egui::Context, song_id: u32) {
        use crate::annotation::model::Quality;
        let Some((px, py)) = self.annotation.picker_at else {
            return;
        };
        let mut close = false;
        let mut pick: Option<Option<crate::annotation::model::Chord>> = None;
        let mut uncertain = self.annotation.picker_uncertain;
        // The chord whose button is under the cursor this frame, if any — auditioned below once it
        // changes (so a glide within one cell doesn't re-strike, but moving to a neighbour does).
        let mut hovered: Option<crate::annotation::model::Chord> = None;

        let area = egui::Area::new(egui::Id::new("chord_picker"))
            .order(egui::Order::Foreground)
            // Keep the whole grid on screen when the drag ends near an edge.
            .constrain(true)
            .fixed_pos(egui::pos2(px, py))
            .show(ctx, |ui| {
                egui::Frame::popup(&ctx.style()).show(ui, |ui| {
                    ui.set_max_width(430.0);
                    ui.horizontal(|ui| {
                        let (s, e) = self.annotation.pending_range.unwrap_or((0, 0));
                        let bars = self
                            .piano_roll
                            .grid()
                            .bar_beat_at(s as f64)
                            .map(|(b, _)| format!("bar {b}"))
                            .unwrap_or_default();
                        ui.label(
                            egui::RichText::new(format!("{bars}  ({} steps)", e.saturating_sub(s)))
                                .small()
                                .color(crate::theme::TEXT_DIM),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            // `❌`, not `✕`: the loaded fonts (Inter / SF Pro + egui's emoji font)
                            // have no U+2715, so it renders as tofu. This is the glyph the Stats
                            // window's close button already uses.
                            if ui.small_button("❌").clicked() {
                                close = true;
                            }
                        });
                    });
                    ui.checkbox(&mut uncertain, "quality uncertain (scores root only)");
                    ui.separator();

                    egui::Grid::new("chord_grid")
                        .spacing([2.0, 2.0])
                        .show(ui, |ui| {
                            // Header: the quality of each column.
                            ui.label("");
                            for q in Quality::ALL {
                                let name = if q.suffix().is_empty() {
                                    "maj"
                                } else {
                                    q.suffix()
                                };
                                ui.label(
                                    egui::RichText::new(name)
                                        .small()
                                        .color(crate::theme::TEXT_DIM),
                                );
                            }
                            ui.end_row();

                            for root in 0..12u8 {
                                let chord = |q: Quality| crate::annotation::model::Chord {
                                    root,
                                    quality: q,
                                    quality_uncertain: uncertain,
                                };
                                // Row header doubles as the colour key used in the ribbon.
                                let swatch = crate::piano_roll::chord_color(Some(root), false);
                                ui.label(
                                    egui::RichText::new(chord(Quality::Major).symbol())
                                        .monospace()
                                        .color(swatch),
                                );
                                for q in Quality::ALL {
                                    let c = chord(q);
                                    let btn = egui::Button::new(
                                        egui::RichText::new(c.symbol()).monospace().size(11.0),
                                    )
                                    .min_size(egui::vec2(34.0, 18.0))
                                    .fill(crate::piano_roll::chord_color(Some(root), true));
                                    let resp = ui.add(btn);
                                    if resp.clicked() {
                                        pick = Some(Some(c));
                                    }
                                    // Hovering a cell auditions its chord, so the picker is an
                                    // ear-led scan of the vocabulary rather than guess-and-click.
                                    if resp.hovered() {
                                        hovered = Some(c);
                                    }
                                }
                                ui.end_row();
                            }
                        });

                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui
                            .button("N.C.")
                            .on_hover_text("No chord here — a real annotation, not a gap.")
                            .clicked()
                        {
                            pick = Some(None);
                        }
                        if ui
                            .button("🗑 clear")
                            .on_hover_text("Remove any label over this range")
                            .clicked()
                        {
                            self.delete_selected_span(song_id);
                            close = true;
                        }
                        // Typing stays available: it beats the grid once you know the vocabulary.
                        let entry = ui.add(
                            egui::TextEdit::singleline(&mut self.annotation_entry)
                                .desired_width(70.0)
                                .hint_text("Am7"),
                        );
                        if entry.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                            self.apply_typed_chord(song_id);
                            close = true;
                        }
                    });
                });
            });

        self.annotation.picker_uncertain = uncertain;
        // Audition the chord whose cell the cursor entered — only when it differs from last frame's
        // cell, so dwelling on one button rings once and a glide across the grid plays each in turn.
        // Compared on `(root, quality)`: `quality_uncertain` flipping isn't a musical change.
        if let Some(c) = hovered {
            let key = (c.root % 12, c.quality);
            let same = self
                .annotation
                .picker_hovered
                .map(|p| (p.root % 12, p.quality) == key)
                .unwrap_or(false);
            if !same {
                let start = self.annotation.pending_range.map(|(s, _)| s).unwrap_or(0);
                self.strike_chord((start, c));
            }
        }
        self.annotation.picker_hovered = hovered;
        if let Some(chord) = pick {
            self.set_selected_chord(song_id, chord);
            close = true;
        }
        // Escape, or a press anywhere outside the popup, dismisses it. (A *press*, not a release:
        // the drag-release that opened the picker lands on the same frame, and testing releases
        // would close it instantly.)
        //
        // Never on the opening frame: a right-click's *press* arrives in the same frame the popup
        // first appears, and a popup constrained away from a screen edge doesn't sit under the
        // cursor — so it would dismiss itself the instant it opened.
        let rect = area.response.rect;
        let just_opened = std::mem::take(&mut self.annotation.picker_just_opened);
        let pressed_outside = !just_opened
            && ctx.input(|i| {
                i.pointer.any_pressed()
                    && i.pointer.interact_pos().is_some_and(|p| !rect.contains(p))
            });
        if pressed_outside || ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            close = true;
        }
        if close {
            self.annotation.picker_at = None;
            self.annotation.picker_hovered = None;
        }
    }

    /// Writes `chord` over the selected range and advances the selection to the next one, so a run
    /// of bars is drag-once-then-click-click-click.
    fn set_selected_chord(&mut self, song_id: u32, chord: Option<crate::annotation::model::Chord>) {
        let Some((s, e)) = self.annotation.pending_range else {
            return;
        };
        self.annotation.insert_span(
            song_id,
            crate::annotation::model::Span {
                start_step: s,
                end_step: e,
                chord,
            },
        );
        self.annotation.status.clear();
        let next_s = e;
        let next_e = self.snap_next(next_s as f64) as u32;
        self.annotation.pending_range = Some((next_s, next_e));
        self.piano_roll
            .set_selection(Some((next_s as f64, next_e as f64)));
    }

    /// Sounds `section`'s chord now through the audition voice. Passing the span, not just the
    /// chord, lets the voice recognise the playhead arriving at the same section and hold it rather
    /// than restrike it.
    fn strike_chord(&mut self, section: (u32, crate::annotation::model::Chord)) {
        if let Some(audio) = &self.audio
            && let Ok(mut st) = audio.shared.lock()
        {
            st.bounce.chords_on = true; // an explicit audition overrides the toggle
            if let Some(c) = &mut st.bounce.chords {
                c.strike(section);
            }
        }
    }

    /// Re-installs the grid after the meter/offset changed.
    fn refresh_grid(&mut self, song_id: u32) {
        let spb = self
            .overview
            .as_ref()
            .map(|o| o.steps_per_beat)
            .unwrap_or(0.0);
        let grid = self.annotation.grid_for(song_id, spb);
        self.piano_roll.set_grid(grid);
    }

    /// Arms or clears the loop over the current selection. Arming also starts the transport — the
    /// only reason to loop a range is to hear it.
    fn set_annotation_loop(&mut self, on: bool) {
        let range = self.annotation.pending_range.filter(|_| on);
        let (Some(b), Some(audio)) = (&self.annotation.bounce, &self.audio) else {
            return;
        };
        let frames = range.map(|(s, e)| (b.frame_of_step(s), b.frame_of_step(e)));
        if let Ok(mut st) = audio.shared.lock() {
            st.bounce.loop_range = frames;
            if let Some((start, _)) = frames {
                st.bounce.pos = start;
            }
        }
        if frames.is_some() {
            // `playing` is derived from `paused` each frame, so unpause rather than force the flag.
            self.paused = false;
        }
    }

    /// Deletes the selected span.
    fn delete_selected_span(&mut self, song_id: u32) {
        let Some((s, e)) = self.annotation.pending_range else {
            return;
        };
        if let Some(song) = self.annotation.songs.get_mut(&song_id) {
            let before = song.spans.len();
            song.spans
                .retain(|sp| !(sp.start_step < e && sp.end_step > s));
            if song.spans.len() != before {
                self.annotation.dirty = true;
            }
        }
        self.annotation_entry.clear();
    }

    /// Turns annotation mode on/off.
    ///
    /// On: the live playlist is parked (installing a bounce makes the callback ignore it entirely)
    /// and the current song is re-rendered to memory. Off: the bounce is dropped and live playback
    /// picks up exactly where it was.
    fn set_annotation_mode(&mut self, on: bool) {
        self.annotation.enabled = on;
        self.piano_roll.set_edit(on);
        self.piano_roll.set_selection(None);
        if on {
            self.load_annotations();
            self.capture_chord_instrument();
            // Hand the output to the annotation mixer up front, not when the bounce lands: chord
            // playback has to work while the render is still going.
            if let Some(audio) = &self.audio
                && let Ok(mut st) = audio.shared.lock()
            {
                st.bounce.active = true;
                st.bounce.buffer = None;
                st.bounce.pos = 0;
                st.bounce.loop_range = None;
            }
            self.request_bounce();
            if let Some(ov) = &self.overview {
                let total = ov.total_steps;
                self.piano_roll.zoom_to_fit(total);
            }
        } else {
            self.annotation.bounce = None;
            self.annotation.bounce_for = None;
            self.annotation.job = None;
            // `audio` is `None` when no output device exists — the app still runs, so annotation
            // must not assume a stream.
            if let Some(audio) = &self.audio
                && let Ok(mut st) = audio.shared.lock()
            {
                st.bounce.active = false;
                st.bounce.buffer = None;
                st.bounce.playing = false;
                st.bounce.loop_range = None;
            }
            // Restore the pre-inferred lane the roll shows outside annotation.
            if let Some(i) = self.current_song {
                let (a, s) = (self.songs[i].archive_index, self.songs[i].song_id);
                self.piano_roll.set_chords(self.chord_spans(a, s));
            }
        }
    }

    /// Lifts the chord-audition voice out of the ROM.
    ///
    /// Prefers Emerald's song 435 ("Trainers' School"), which opens on a clean piano; falls back to
    /// the song being annotated so chord playback still works on a ROM without that id.
    /// Arms the chord-audition voicer with the embedded piano reference instrument, inheriting the
    /// user's current gain/inversion. The instrument ships with the app (`include_bytes!`'d from
    /// `chord_data/piano_c4.wav`), so this no longer surveys the loaded ROM — the audition always
    /// has a known-good piano regardless of the track, instead of grabbing a bass thump on a sparse
    /// one.
    fn capture_chord_instrument(&mut self) {
        let instrument = crate::annotation::chord_voice::embedded_piano();
        if let Some(audio) = &self.audio
            && let Ok(mut st) = audio.shared.lock()
        {
            let mut v = crate::annotation::ChordVoicer::new(instrument);
            // A freshly armed voicer must inherit the user's current audition settings, not the
            // defaults — otherwise switching songs snaps the volume back and loses the inversion.
            v.set_gain(self.annotation.chord_gain);
            v.set_inversion(self.annotation.chord_inversion);
            st.bounce.chords = Some(v);
        }
    }

    /// The annotated chord covering `step`, if any, with the start step of the span it came from —
    /// the voice keys its retrigger on the span, so two neighbouring bars labelled the same chord
    /// each get struck.
    fn chord_at(&self, song_id: u32, step: f64) -> Option<(u32, crate::annotation::model::Chord)> {
        self.annotation.song(song_id)?.spans.iter().find_map(|sp| {
            ((sp.start_step as f64) <= step && (sp.end_step as f64) > step)
                .then_some(sp.chord)
                .flatten()
                .map(|c| (sp.start_step, c))
        })
    }

    /// Starts rendering the current song to memory. The work is sliced across frames by
    /// [`Self::poll_bounce`] rather than threaded — annotation runs on the web too, where there is
    /// no thread to spawn and a multi-second block would freeze the tab.
    fn request_bounce(&mut self) {
        let Some(i) = self.current_song else { return };
        let Some(ov) = &self.overview else { return };
        let (archive_index, song_id) = (self.songs[i].archive_index, self.songs[i].song_id);
        if self.annotation.bounce_for == Some(song_id) {
            return;
        }
        let total_steps = ov.total_steps;
        let archive = self.archives[archive_index].clone();
        let config = self.device_settings().clone();
        self.annotation.bounce = None;
        self.annotation.bounce_for = Some(song_id);
        self.annotation.job = BounceJob::new(&*archive, song_id, total_steps, config);
        self.annotation.status = "rendering…".into();
        // The old buffer must go now: it belongs to a different song.
        if let Some(audio) = &self.audio
            && let Ok(mut st) = audio.shared.lock()
        {
            st.bounce.buffer = None;
            st.bounce.playing = false;
            st.bounce.loop_range = None;
        }
    }

    /// Advances an in-flight bounce by one frame's worth of work, installing it when done.
    fn poll_bounce(&mut self, ctx: &egui::Context) {
        /// Frames of audio rendered per UI frame. ~0.4 s of audio per tick keeps a 60 Hz UI
        /// responsive while still bouncing a 3-minute song in a few seconds.
        const BUDGET: usize = 12_000;

        let Some(job) = &mut self.annotation.job else {
            return;
        };
        job.step(BUDGET);
        if !job.is_done() {
            self.annotation.status = format!("rendering… {:.0}%", job.progress() * 100.0);
            ctx.request_repaint(); // keep the slices coming
            return;
        }
        let bounce = Arc::new(self.annotation.job.take().unwrap().finish());
        self.annotation.status = format!("{:.1}s buffered", bounce.frames() as f64 / 32768.0);
        self.annotation.bounce = Some(bounce.clone());
        if let Some(audio) = &self.audio
            && let Ok(mut st) = audio.shared.lock()
        {
            st.bounce.buffer = Some(bounce);
            st.bounce.pos = 0;
            st.bounce.playing = false;
            st.bounce.loop_range = None;
            st.resampler.reset();
        }
    }

    /// Moves the annotation playhead to `step` (a scrub). Instant: the bounce makes this an index.
    fn annotation_seek(&mut self, step: f64) {
        let Some(b) = &self.annotation.bounce else {
            return;
        };
        let frame = b.frame_of_step(step.max(0.0) as u32);
        if let Some(audio) = &self.audio
            && let Ok(mut st) = audio.shared.lock()
        {
            st.bounce.pos = frame;
        }
        // The roll follows the cursor even with no audio device — the app runs without one.
        self.piano_roll.set_display_tick(step.max(0.0));
    }

    /// The annotation JSON path for the loaded source.
    #[cfg(not(target_arch = "wasm32"))]
    fn annotation_path(&self) -> std::path::PathBuf {
        crate::annotation::source_dir().join(crate::annotation::filename_for(&self.current_source))
    }

    /// Loads the current game's labels, once per source.
    #[cfg(not(target_arch = "wasm32"))]
    fn load_annotations(&mut self) {
        if self.annotation.loaded_for.as_deref() == Some(self.current_source.as_str()) {
            return;
        }
        let path = self.annotation_path();
        match crate::annotation::load_file(&path) {
            Ok(Some(file)) => {
                self.annotation.songs = file.songs.into_iter().map(|s| (s.song_id, s)).collect();
                self.annotation.status = format!("loaded {}", path.display());
            }
            Ok(None) => {
                // No committed JSON for this source — recover the last session's unsaved work
                // (a "Bar 1 here" / meter edit the user didn't think to save) from the
                // eframe-storage stash. The JSON remains the record; the stash only speaks when
                // there is no record yet, so a hand-edited file is never overwritten.
                if let Some(file) = self.p.annotations.get(&self.current_source).cloned() {
                    self.annotation.songs =
                        file.songs.into_iter().map(|s| (s.song_id, s)).collect();
                    self.annotation.status = "recovered unsaved edits from last session".into();
                } else {
                    self.annotation.songs.clear();
                    self.annotation.status = "no labels yet".into();
                }
            }
            // Never fall back to "empty" on a bad file: that would look like no labels, and the
            // next save would overwrite them.
            Err(e) => {
                self.annotation.songs.clear();
                self.annotation.status = format!("⚠ {e}");
                self.annotation.loaded_for = None;
                return;
            }
        }
        self.annotation.loaded_for = Some(self.current_source.clone());
        self.annotation.dirty = false;
    }

    /// The current game's labels as the file that gets written/downloaded.
    fn annotation_file(&self) -> crate::annotation::model::GameAnnotations {
        let game_code = self
            .archives
            .first()
            .and_then(|a| a.as_any().downcast_ref::<GbaRom>())
            .and_then(GbaRom::game_code);
        let spb = self
            .overview
            .as_ref()
            .map(|o| o.steps_per_beat)
            .unwrap_or(0.0);
        self.annotation
            .to_file(self.current_source.clone(), game_code, spb)
    }

    /// Persists the current game's labels.
    ///
    /// Native writes `ml/annotations/*.json` in the source tree — that file *is* the training data.
    /// Web has no filesystem, so it downloads the same JSON for the user to drop in; the schema is
    /// identical either way, which is the point of the format being the contract.
    #[cfg(not(target_arch = "wasm32"))]
    fn save_annotations(&mut self) {
        let file = self.annotation_file();
        let path = self.annotation_path();
        match crate::annotation::save_file(&path, &file) {
            Ok(()) => {
                self.annotation.dirty = false;
                self.annotation.status = format!("saved {}", path.display());
            }
            Err(e) => self.annotation.status = format!("⚠ {e}"),
        }
    }

    /// Downloads the current game's labels as JSON (web has no source tree to write to).
    ///
    /// The working copy already lives in eframe storage (see [`Self::stash_annotations`]), so this
    /// is an *export* rather than the only thing standing between the user and data loss.
    #[cfg(target_arch = "wasm32")]
    fn save_annotations(&mut self) {
        let file = self.annotation_file();
        let name = crate::annotation::filename_for(&self.current_source);
        match serde_json::to_string_pretty(&file) {
            Ok(json) => {
                crate::web::download(&name, json.as_bytes());
                self.annotation.dirty = false;
                self.annotation.status = format!("downloaded {name}");
            }
            Err(e) => self.annotation.status = format!("⚠ encode: {e}"),
        }
    }

    /// Keeps the working copy in eframe storage so a refresh (web) or restart (native) can't
    /// destroy unsaved work — most often a grid/meter edit ("Bar 1 here", meter, beat snap) the
    /// user doesn't think to commit, since it isn't a chord label.
    ///
    /// This is **session recovery, not the record**: the source-tree JSON (native) and the
    /// downloaded file (web) stay authoritative. The stash only fills in when there is no
    /// committed file yet, so it can never silently overwrite a hand-edited record.
    fn stash_annotations(&mut self) {
        if !self.annotation.dirty {
            return;
        }
        let file = self.annotation_file();
        self.p.annotations.insert(self.current_source.clone(), file);
    }

    /// Web: restore the working copy from eframe storage.
    #[cfg(target_arch = "wasm32")]
    fn load_annotations(&mut self) {
        if self.annotation.loaded_for.as_deref() == Some(self.current_source.as_str()) {
            return;
        }
        self.annotation.songs = self
            .p
            .annotations
            .get(&self.current_source)
            .map(|f| f.songs.iter().map(|s| (s.song_id, s.clone())).collect())
            .unwrap_or_default();
        self.annotation.status = if self.annotation.songs.is_empty() {
            "no labels yet".into()
        } else {
            format!("{} songs from browser storage", self.annotation.songs.len())
        };
        self.annotation.loaded_for = Some(self.current_source.clone());
        self.annotation.dirty = false;
    }

    /// Opens a native file dialog; on web spawns the async picker. Loaded bytes arrive via
    /// [`Self::pending_file`].
    fn open_file_dialog(&mut self) {
        let inbox = self.pending_file.clone();
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::spawn(move || {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("DS / GBA sound", &["nds", "sdat", "gba", "gbaaudio", "gz"])
                    .pick_file()
                {
                    let key = path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "file".to_owned());
                    if let Ok(bytes) = std::fs::read(path)
                        && let Ok(mut slot) = inbox.lock()
                    {
                        *slot = Some((key, bytes));
                    }
                }
            });
        }
        #[cfg(target_arch = "wasm32")]
        {
            wasm_bindgen_futures::spawn_local(async move {
                if let Some(handle) = rfd::AsyncFileDialog::new()
                    .add_filter("DS / GBA sound", &["nds", "sdat", "gba", "gbaaudio", "gz"])
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
        let data = &*self.archives[song.archive_index];
        let samples = player::render_to_samples(data, song.song_id, &self.config());
        let wav = crate::wav::encode_stereo_i16(&samples, player::EXPORT_SAMPLE_RATE);
        let name = song.label.replace([' ', '#', '(', ')', '/'], "_");
        save_bytes(&format!("{name}.wav"), &wav);
        self.status = format!("Exported {name}.wav ({} frames).", samples.len());
    }

    /// Whether a GBA ROM is currently loaded (enables the audio-data export button).
    fn gba_loaded(&self) -> bool {
        self.archives.iter().any(|a| a.as_any().is::<GbaRom>())
    }

    /// Exports the loaded GBA ROM's audio data as an audio-only `.gba` image: everything the
    /// MP2K engine can't reach from the song table is stripped, so the file can be shipped
    /// (e.g. in `demos/`) without bundling the game's code or art.
    fn export_gba_audio(&mut self) {
        let Some(rom) = self
            .archives
            .iter()
            .find_map(|a| a.as_any().downcast_ref::<GbaRom>())
        else {
            self.status = "No GBA ROM loaded.".to_owned();
            return;
        };
        let image = rom.extract_audio();
        let stem = self
            .current_source
            .trim_end_matches(".gba")
            .replace([' ', '#', '(', ')', '/'], "_");
        let name = format!("{stem}-audio.gba");
        save_bytes(&name, &image);
        self.status = format!(
            "Exported {name} ({:.1} MB of audio-only ROM).",
            image.len() as f64 / 1_048_576.0
        );
    }

    /// Pushes UI config (config / paused / volume) into the audio thread and pulls a note snapshot
    /// for the visualizer. The fade-out + advance now live entirely in the audio callback (see
    /// [`crate::audio`]); the UI reflects the result via [`Self::reconcile_playback`].
    fn sync_audio(&mut self) -> VisSnapshot {
        let config = self.config();
        let mut snap = VisSnapshot::default();
        let Some(shared) = self.audio.as_ref().map(|a| a.shared.clone()) else {
            return snap;
        };
        let Ok(mut st) = shared.lock() else {
            return snap;
        };
        st.config = config;
        st.paused = self.paused;
        st.playback.repeat = self.p.repeat;
        // Swipe attenuation: the song gets quieter as its visualizer slides offscreen.
        st.volume = self.p.volume * self.swipe_gain;

        // Sample the performance meters (drawn in the top bar).
        const METER_SAMPLES: usize = 128;
        self.cpu_history.push_back(st.dsp_load);
        self.voice_history.push_back(st.voices as f32);
        if self.cpu_history.len() > METER_SAMPLES {
            self.cpu_history.pop_front();
        }
        if self.voice_history.len() > METER_SAMPLES {
            self.voice_history.pop_front();
        }

        // Sample the high-band-compressor GR meter (drawn inside its settings block).
        self.high_comp_gr_db = st.high_comp_gr_db;

        if let Some(controller) = &st.controller {
            snap.active = true;
            snap.steps = controller.steps_elapsed();
            snap.step_rate = controller.step_rate();
            snap.bpm = controller.current_bpm();
            for t in 0..TRACK_COUNT {
                for n in 0..128 {
                    snap.notes_on[t][n] = controller.notes_on[t][n] != 0;
                }
            }
        }
        snap
    }

    /// The desktop piano-roll panel: a pre-rendered whole-track overview bar with the visible
    /// window highlighted, the current tempo marking beneath it, then the scrolling roll.
    fn piano_roll_panel(&mut self, ui: &mut egui::Ui, snap: &VisSnapshot) {
        // Lazily (re)build the overview texture from the song loaded in `on_now_playing`.
        if self.overview_tex.is_none()
            && let Some(ov) = &self.overview
        {
            let img = crate::piano_roll::overview_image(ov, 1024, 72);
            self.overview_tex = Some(ui.ctx().load_texture(
                "piano_overview",
                img,
                egui::TextureOptions::LINEAR,
            ));
        }

        // The overview bar (whole track) with the on-screen window highlighted. In annotation mode
        // it is also the scrub control — dragging it seeks, which only works because the song is
        // bounced (the sequencer itself cannot seek).
        let mut scrub_to: Option<f64> = None;
        if let (Some(tex), Some(ov)) = (&self.overview_tex, &self.overview) {
            let bar_h = 32.0;
            let scrubbable = self.annotation.enabled;
            let sense = if scrubbable {
                egui::Sense::click_and_drag()
            } else {
                egui::Sense::hover()
            };
            let (bar, resp) =
                ui.allocate_exact_size(egui::vec2(ui.available_width(), bar_h), sense);
            let painter = ui.painter_at(bar);
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            painter.image(tex.id(), bar, uv, egui::Color32::WHITE);

            let total = ov.total_steps.max(1) as f64;
            let (vs, ve) = self.piano_roll.visible_range();
            let frac = |t: f64| (t / total).clamp(0.0, 1.0) as f32;
            let x0 = bar.min.x + frac(vs) * bar.width();
            let x1 = bar.min.x + frac(ve) * bar.width();
            let win =
                egui::Rect::from_min_max(egui::pos2(x0, bar.min.y), egui::pos2(x1, bar.max.y));
            painter.rect_filled(win, 0.0, egui::Color32::from_white_alpha(36));
            painter.rect_stroke(
                win,
                0.0,
                egui::Stroke::new(1.0_f32, egui::Color32::from_white_alpha(170)),
            );
            painter.rect_stroke(bar, 2.0, egui::Stroke::new(1.0_f32, crate::theme::HAIRLINE));

            if scrubbable {
                // The playhead, so the bar reads as a transport rather than a thumbnail.
                let px = bar.min.x + frac(self.piano_roll.display_tick()) * bar.width();
                painter.line_segment(
                    [egui::pos2(px, bar.min.y), egui::pos2(px, bar.max.y)],
                    egui::Stroke::new(1.0_f32, crate::theme::ACCENT),
                );
                if (resp.dragged() || resp.clicked())
                    && let Some(p) = resp.interact_pointer_pos()
                {
                    let f = ((p.x - bar.min.x) / bar.width().max(1.0)).clamp(0.0, 1.0) as f64;
                    scrub_to = Some(f * total);
                }
            }
        }
        if let Some(step) = scrub_to {
            self.annotation_seek(step);
        }

        // The current tempo marking and musical position, below the bar.
        ui.horizontal(|ui| {
            if snap.active && snap.bpm > 0.0 {
                ui.label(
                    egui::RichText::new(format!("\u{2669} = {}", snap.bpm.round() as i64))
                        .monospace()
                        .color(egui::Color32::from_gray(0xc0)),
                );
            }
            // Where the playhead is in musical terms — the coordinate an annotator thinks in.
            if let Some((bar, beat)) = self
                .piano_roll
                .grid()
                .bar_beat_at(self.piano_roll.display_tick())
            {
                ui.label(
                    egui::RichText::new(format!("Bar {bar} · {beat}"))
                        .monospace()
                        .color(crate::theme::TEXT_DIM),
                );
            }
        });
        ui.add_space(2.0);

        if self.annotation.enabled {
            self.annotation_toolbar_ui(ui);
            ui.add_space(2.0);
        }

        // The scrolling roll fills the rest of the panel.
        let input = self.piano_roll.draw(ui, snap.active);
        if self.annotation.enabled {
            self.annotation_roll_input(input);
            if let Some(song_id) = self.current_song.map(|i| self.songs[i].song_id) {
                self.chord_picker_ui(ui.ctx(), song_id);
            }
        }
    }

    /// The full song list with like / add-to-playlist menus and status badges.
    /// Returns `true` if a song was started.
    fn song_list_ui(&mut self, ui: &mut egui::Ui) -> bool {
        // "Edit song names" mode replaces the list with an inline title/order editor (native only).
        #[cfg(not(target_arch = "wasm32"))]
        if self.song_edit_enabled {
            return self.song_edit_list_ui(ui);
        }
        enum Action {
            Play(usize),
            Like(usize),
            Add(usize, usize),
        }
        let mut action = None;
        // The Like / Add-to-playlist menu body, shared between the per-row `…` button (works
        // by tap — touch has no right-click) and the desktop context menu.
        let song_menu = |ui: &mut egui::Ui,
                         i: usize,
                         liked: bool,
                         playlists: &[crate::persisted::Playlist],
                         action: &mut Option<Action>| {
            let like_label = if liked { "💔 Unlike" } else { "❤ Like" };
            if ui.button(like_label).clicked() {
                *action = Some(Action::Like(i));
                ui.close_menu();
            }
            ui.menu_button("➕ Add to playlist", |ui| {
                if playlists.is_empty() {
                    ui.label("No playlists yet — create one in Playlists.");
                }
                for (p, pl) in playlists.iter().enumerate() {
                    if ui.button(&pl.name).clicked() {
                        *action = Some(Action::Add(i, p));
                        ui.close_menu();
                    }
                }
            });
        };
        // Sort selector (by native order, name, or computed length) with an ascending/descending
        // direction toggle.
        let mut mode = self.p.sort_mode;
        let mut desc = self.p.sort_descending;
        ui.horizontal(|ui| {
            ui.label("Sort:");
            egui::ComboBox::from_id_salt("song_sort")
                .selected_text(mode.label())
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut mode, SortMode::Default, SortMode::Default.label());
                    ui.selectable_value(&mut mode, SortMode::Name, SortMode::Name.label());
                    ui.selectable_value(&mut mode, SortMode::Length, SortMode::Length.label());
                });
            // "▲" ascending / "▼" descending, toggled on click.
            let arrow = if desc { "▼" } else { "▲" };
            if ui
                .add(egui::Button::new(arrow).frame(false))
                .on_hover_text(if desc {
                    "Descending — click for ascending"
                } else {
                    "Ascending — click for descending"
                })
                .clicked()
            {
                desc = !desc;
            }
        });
        if mode != self.p.sort_mode || desc != self.p.sort_descending {
            self.p.sort_mode = mode;
            self.p.sort_descending = desc;
            self.needs_sort = true;
        }
        ui.add_space(2.0);
        ui.spacing_mut().item_spacing.y = 0.0;
        // Pad every id to the widest song's digit count so the names line up in one column.
        let id_digits = self
            .songs
            .iter()
            .map(|s| s.song_id)
            .max()
            .unwrap_or(0)
            .to_string()
            .len();
        for (i, song) in self.songs.iter().enumerate() {
            let selected = self.current_song == Some(i);
            let length = song.length.map(fmt_duration);
            let track = TrackRef {
                source: self.current_source.clone(),
                song_id: song.song_id,
                label: String::new(),
            };
            let liked = self.p.library.is_liked(&track);
            let in_playlist = self
                .p
                .library
                .playlists
                .iter()
                .any(|pl| pl.tracks.iter().any(|x| x.same_song(&track)));
            // Status badges: green playlist marker, accent heart.
            let mut badges: Vec<(&str, egui::Color32)> = Vec::new();
            if liked {
                badges.push(("❤", crate::theme::ACCENT));
            }
            if in_playlist {
                badges.push(("🎵", egui::Color32::from_rgb(0x32, 0xd7, 0x4b)));
            }
            // Right-to-left so the trailing menu button takes its true size first and the
            // row label fills exactly the remainder — the row can never overflow the panel
            // (overflow makes a resizable SidePanel grow every frame).
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), 42.0),
                egui::Layout::right_to_left(egui::Align::Center),
                |ui| {
                    // "…" — U+22EF is missing from egui's bundled fonts and renders as tofu.
                    ui.menu_button(
                        egui::RichText::new("…")
                            .size(22.0)
                            .color(crate::theme::TEXT_DIM),
                        |ui| song_menu(ui, i, liked, &self.p.library.playlists, &mut action),
                    );
                    // The id sits to the left of the name (as in the editor): place it first in a
                    // left-to-right sub-area, then let the row fill the remaining width.
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        let resp = song_id_and_title(
                            ui,
                            song.song_id,
                            id_digits,
                            crate::theme::TEXT_DIM,
                            SongTitle::Static {
                                name: &song.name,
                                length: length.as_deref(),
                                badges: &badges,
                                selected,
                            },
                        )
                        .title;
                        if resp.clicked() {
                            action = Some(Action::Play(i));
                        }
                        resp.context_menu(|ui| {
                            song_menu(ui, i, liked, &self.p.library.playlists, &mut action)
                        });
                    });
                },
            );
        }
        match action {
            Some(Action::Play(i)) => {
                self.play_from_list(i);
                return true;
            }
            Some(Action::Like(i)) => {
                if let Some(t) = self.track_ref(i) {
                    self.p.library.toggle_liked(&t);
                }
            }
            Some(Action::Add(i, p)) => self.add_song_to_playlist(i, p),
            None => {}
        }
        false
    }

    /// The library list in "Edit song names" mode: every loaded song as an editable row (rename +
    /// reorder), plus a "Save to JSON" header that writes the titles and order back to the
    /// source-tree JSON. Operates directly on [`Self::songs`] in `Vec` order (sorting is suspended
    /// while editing). Native only.
    #[cfg(not(target_arch = "wasm32"))]
    fn song_edit_list_ui(&mut self, ui: &mut egui::Ui) -> bool {
        /// One deferred edit applied after the row loop (so we never reorder mid-iteration).
        enum Edit {
            Play(usize),
            MoveUp(usize),
            MoveDown(usize),
            MoveTo(usize, usize),
            Save,
        }
        let mut edit: Option<Edit> = None;

        // Seed the (adjustable) target filename from the loaded game's table the first time, and
        // again whenever a different game is loaded — but keep the user's edits in between.
        if self.song_edit_filename_for != self.current_source {
            let game_code = self
                .archives
                .first()
                .and_then(|a| a.as_any().downcast_ref::<GbaRom>())
                .and_then(GbaRom::game_code);
            let (file, _) =
                song_names::target_filename(Some(&self.current_source), game_code.as_deref());
            self.song_edit_filename = file;
            self.song_edit_filename_for = self.current_source.clone();
        }
        let path = song_names::source_json_dir().join(&self.song_edit_filename);
        let wired = song_names::is_known_filename(&self.song_edit_filename);

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Editing song names").strong());
            if ui
                .button("💾 Save to JSON")
                .on_hover_text(format!("Write to {}", path.display()))
                .clicked()
            {
                edit = Some(Edit::Save);
            }
        });
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("File:")
                    .small()
                    .color(crate::theme::TEXT_DIM),
            );
            ui.add(
                egui::TextEdit::singleline(&mut self.song_edit_filename)
                    .desired_width(f32::INFINITY)
                    .hint_text("filename.json"),
            )
            .on_hover_text(path.display().to_string());
        });
        if !wired {
            ui.label(
                egui::RichText::new(
                    "Not yet wired: after saving, add this file to JSONS_BY_GBA_GAME_CODE / \
                     JSONS_BY_GAME_FILENAME in song_names/mod.rs so it loads next build.",
                )
                .small()
                .color(crate::theme::ACCENT),
            );
        }
        ui.separator();
        ui.spacing_mut().item_spacing.y = 2.0;

        let len = self.songs.len();
        for i in 0..len {
            let song_id = self.songs[i].song_id;
            // Curated songs (accent-coloured id) are the ones a save will write.
            let curated = self.songs[i].curated;
            // The currently-playing song's play button is shown red.
            let is_current = self.current_song == Some(i);
            let mut pos = i + 1;
            ui.horizontal(|ui| {
                let play = egui::RichText::new("▶");
                let play = if is_current {
                    play.color(egui::Color32::from_rgb(0xff, 0x45, 0x45))
                } else {
                    play
                };
                if ui
                    .add(egui::Button::new(play).small())
                    .on_hover_text("Play")
                    .clicked()
                {
                    edit = Some(Edit::Play(i));
                }
                if ui
                    .add_enabled(i > 0, egui::Button::new("↑").small())
                    .clicked()
                {
                    edit = Some(Edit::MoveUp(i));
                }
                if ui
                    .add_enabled(i + 1 < len, egui::Button::new("↓").small())
                    .clicked()
                {
                    edit = Some(Edit::MoveDown(i));
                }
                ui.add(egui::DragValue::new(&mut pos).range(1..=len.max(1)))
                    .on_hover_text("Position — drag or type to move this song");
                let id_color = if curated {
                    crate::theme::ACCENT
                } else {
                    crate::theme::TEXT_DIM
                };
                let row = song_id_and_title(
                    ui,
                    song_id,
                    0,
                    id_color,
                    SongTitle::Edit {
                        name: &mut self.songs[i].name,
                    },
                );
                row.id.on_hover_text(if curated {
                    "Curated — written on save"
                } else {
                    "Not in the saved table until you rename or move it"
                });
                let resp = row.title;
                if resp.changed() && !self.songs[i].name.trim().is_empty() {
                    let s = &mut self.songs[i];
                    s.label = format!("{} (#{})", s.name, s.song_id);
                    s.curated = true; // a rename opts the song into the saved table
                }
                // Clearing the field resets the song to its default (ROM/derived) title and drops it
                // from the curated table — applied on commit so the field can be cleared and retyped.
                if resp.lost_focus() && self.songs[i].name.trim().is_empty() {
                    let default = self.default_song_name(i);
                    let s = &mut self.songs[i];
                    s.name = default;
                    s.label = format!("{} (#{})", s.name, s.song_id);
                    s.curated = false;
                    s.sparse_index = None;
                }
            });
            // A DragValue change is detected by the position drifting from this row's index.
            if pos != i + 1 {
                edit = Some(Edit::MoveTo(i, pos.saturating_sub(1).min(len - 1)));
            }
        }

        match edit {
            Some(Edit::Play(i)) => {
                self.play_from_list(i);
                return true;
            }
            // The moved song travels through the unlabeled slots only; every *other* curated song
            // stays pinned to its list position, so reordering never shifts a curated song. The
            // moved song opts into the saved table.
            Some(Edit::MoveUp(i)) => self.reposition_song(i, CuratedTarget::Up),
            Some(Edit::MoveDown(i)) => self.reposition_song(i, CuratedTarget::Down),
            Some(Edit::MoveTo(from, to)) => self.reposition_song(from, CuratedTarget::Abs(to)),
            Some(Edit::Save) => self.save_song_names(),
            None => {}
        }
        false
    }

    /// The song's default (non-curated) title: the ROM-embedded name if any, otherwise `Song {id}` —
    /// i.e. the title it would have with no curated-JSON entry. Used to reset a cleared row.
    #[cfg(not(target_arch = "wasm32"))]
    fn default_song_name(&self, i: usize) -> String {
        let s = &self.songs[i];
        self.archives[s.archive_index]
            .song_name(s.song_id)
            .unwrap_or_else(|| format!("Song {}", s.song_id))
    }

    /// Moves the song at `moved_slot` to a new list position by bubbling it through its own slot plus
    /// the unlabeled slots — every *other* curated (labeled) song is pinned, so a reorder never
    /// shifts a curated song; only unlabeled songs flow around the move. The target is a neighbouring
    /// movable slot (`Up`/`Down`) or, for a typed list position (`Abs`), just past the unlabeled
    /// songs still ahead of it. (Labels the moved song, and preserves the current-song highlight via
    /// the same identity-remap [`Self::apply_sort`] uses.)
    #[cfg(not(target_arch = "wasm32"))]
    fn reposition_song(&mut self, moved_slot: usize, target: CuratedTarget) {
        self.songs[moved_slot].curated = true;
        // The slots the move may touch: the moved song's own slot plus every unlabeled song's. Other
        // curated songs are absent, so they're never swapped — they keep their exact list position.
        let slots: Vec<usize> = (0..self.songs.len())
            .filter(|&j| j == moved_slot || !self.songs[j].curated)
            .collect();
        let k = slots.len();
        let from_rank = slots.iter().position(|&s| s == moved_slot).unwrap();
        let to_rank = match target {
            CuratedTarget::Up => from_rank.saturating_sub(1),
            CuratedTarget::Down => (from_rank + 1).min(k - 1),
            // Land just past the unlabeled songs still ahead of the typed position.
            CuratedTarget::Abs(to) => slots
                .iter()
                .filter(|&&s| s < to && s != moved_slot)
                .count()
                .min(k - 1),
        };
        if to_rank == from_rank {
            return;
        }
        let playing = self
            .current_song
            .and_then(|c| self.songs.get(c))
            .map(|s| (s.archive_index, s.song_id));
        // Bubble the song one movable slot at a time (each swap exchanges it with an unlabeled song,
        // never another curated song).
        if to_rank > from_rank {
            for r in from_rank..to_rank {
                self.songs.swap(slots[r], slots[r + 1]);
            }
        } else {
            for r in (to_rank..from_rank).rev() {
                self.songs.swap(slots[r], slots[r + 1]);
            }
        }
        if let Some((ai, sid)) = playing {
            self.current_song = self
                .songs
                .iter()
                .position(|s| s.archive_index == ai && s.song_id == sid);
        }
    }

    /// Writes the current [`Self::songs`] titles and order (the editor's state) back to the matching
    /// source-tree JSON, as `[{ "songId", "title", "sparseIndex" }]`. Only the **curated** songs (the
    /// ones from the loaded table plus any renamed/reordered this session) are written, each with its
    /// absolute list position (`sparseIndex`) so a sparse table keeps its songs interspersed among the
    /// unlabeled ones on reload — untouched ROM-named songs stay out of the file. Native dev-tool:
    /// only works when the app runs from the checkout. Also mirrors the saved positions into
    /// `sparse_index` so the order survives a later Default re-sort this session.
    #[cfg(not(target_arch = "wasm32"))]
    fn save_song_names(&mut self) {
        #[derive(serde::Serialize)]
        struct Out<'a> {
            #[serde(rename = "songId")]
            song_id: u32,
            title: &'a str,
            #[serde(rename = "sparseIndex")]
            sparse_index: usize,
        }
        let entries: Vec<Out> = self
            .songs
            .iter()
            .enumerate()
            .filter(|(_, s)| s.curated)
            .map(|(idx, s)| Out {
                song_id: s.song_id,
                title: &s.name,
                sparse_index: idx,
            })
            .collect();
        if entries.is_empty() {
            self.status =
                "Nothing to save — rename or reorder a song first (untouched songs aren't written)."
                    .to_owned();
            return;
        }
        let n = entries.len();
        // Write to the editor's adjustable filename (seeded from the loaded game's table).
        let file = self.song_edit_filename.trim().to_owned();
        if file.is_empty() {
            self.status = "Enter a filename to save to.".to_owned();
            return;
        }
        let wired = song_names::is_known_filename(&file);
        let path = song_names::source_json_dir().join(&file);

        let json = match serde_json::to_string_pretty(&entries) {
            Ok(mut j) => {
                j.push('\n');
                j
            }
            Err(e) => {
                self.status = format!("Failed to serialize song names: {e}");
                return;
            }
        };
        drop(entries);
        if let Err(e) = std::fs::write(&path, json) {
            self.status = format!("Failed to write {}: {e}", path.display());
            return;
        }
        // Mirror the saved positions in-session: each curated song keeps its absolute list index as
        // its `sparse_index`, so the Default re-sort puts it right back where it is now.
        for i in 0..self.songs.len() {
            self.songs[i].sparse_index = self.songs[i].curated.then_some(i);
        }
        self.status = if wired {
            format!("Saved {n} song names to {}.", path.display())
        } else {
            format!(
                "Saved {n} song names to {} — add it to JSONS_BY_* in song_names/mod.rs to load it \
                 next build.",
                path.display()
            )
        };
    }

    /// Whether the inline song-name editor is active (always `false` on web, where it isn't built).
    fn song_edit_active(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.song_edit_enabled
        }
        #[cfg(target_arch = "wasm32")]
        {
            false
        }
    }

    /// Adds song `i` to the end of playlist `p` and reports the result in the status line. A song
    /// already in the list is added again rather than refused: a playlist is an ordered programme,
    /// and a track may legitimately recur in one.
    fn add_song_to_playlist(&mut self, i: usize, p: usize) {
        let Some(t) = self.track_ref(i) else { return };
        let Some(pl) = self.p.library.playlists.get_mut(p) else {
            return;
        };
        let copies = pl.tracks.iter().filter(|x| x.same_song(&t)).count();
        self.status = if copies > 0 {
            format!("Added to {} ({} times now).", pl.name, copies + 1)
        } else {
            format!("Added to {}.", pl.name)
        };
        pl.tracks.push(t);
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

    /// The library root: collection rows plus playlist management, iOS-grouped-list style.
    fn library_root_ui(&mut self, ui: &mut egui::Ui) {
        use crate::theme::{ios_row, section_header};
        ui.spacing_mut().item_spacing.y = 0.0;
        let w = ui.available_width();
        let liked_title = format!("Liked Songs ({})", self.p.library.liked.len());
        if ios_row(ui, w, Some("❤"), &liked_title, &[], false, true).clicked() {
            self.library_view = LibraryView::Liked;
        }
        if ios_row(ui, w, Some("🕘"), "Recently Played", &[], false, true).clicked() {
            self.library_view = LibraryView::Recent;
        }
        section_header(ui, "Playlists");
        let mut open = None;
        let mut delete = None;
        for (p, pl) in self.p.library.playlists.iter().enumerate() {
            // RTL: trash button sized first, row fills the exact remainder (no overflow).
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), 42.0),
                egui::Layout::right_to_left(egui::Align::Center),
                |ui| {
                    if ui
                        .add(egui::Button::new("🗑").frame(false))
                        .on_hover_text("Delete playlist")
                        .clicked()
                    {
                        delete = Some(p);
                    }
                    let title = format!("{} ({})", pl.name, pl.tracks.len());
                    let row_w = ui.available_width();
                    if ios_row(ui, row_w, Some("🎵"), &title, &[], false, true).clicked() {
                        open = Some(p);
                    }
                },
            );
        }
        if let Some(p) = open {
            self.library_view = LibraryView::Playlist(p);
        }
        if let Some(p) = delete {
            self.p.library.playlists.remove(p);
        }
        ui.add_space(8.0);
        ui.allocate_ui_with_layout(
            egui::vec2(ui.available_width(), 30.0),
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| {
                let add = ui.button("➕");
                let edit = egui::TextEdit::singleline(&mut self.new_playlist_name)
                    .hint_text("New playlist…")
                    .desired_width(ui.available_width());
                ui.add(edit);
                let name = self.new_playlist_name.trim().to_owned();
                if add.clicked() && !name.is_empty() {
                    self.p.library.playlists.push(crate::persisted::Playlist {
                        name,
                        tracks: Vec::new(),
                    });
                    self.new_playlist_name.clear();
                }
            },
        );
    }

    /// One open collection (liked / recent / a playlist): play all, play/remove single tracks.
    /// Returns `true` if playback started.
    fn collection_ui(&mut self, ui: &mut egui::Ui, view: LibraryView) -> bool {
        let (title, tracks, removable) = match view {
            LibraryView::Liked => ("❤ Liked Songs", self.p.library.liked.clone(), true),
            LibraryView::Recent => ("🕘 Recently Played", self.p.library.recent.clone(), false),
            LibraryView::Playlist(p) => match self.p.library.playlists.get(p) {
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
            if ui
                .add(egui::Button::new(egui::RichText::new("‹ Back").size(15.0)).frame(false))
                .clicked()
            {
                self.library_view = LibraryView::Root;
            }
            ui.label(
                egui::RichText::new(format!("{title} ({})", tracks.len()))
                    .strong()
                    .size(17.0),
            );
        });
        ui.add_space(4.0);
        if !tracks.is_empty()
            && ui
                .add(
                    egui::Button::new(
                        egui::RichText::new("▶ Play all").color(egui::Color32::WHITE),
                    )
                    .fill(crate::theme::ACCENT)
                    .min_size(egui::vec2(ui.available_width(), 36.0)),
                )
                .clicked()
        {
            self.play_all_collection(tracks.clone());
            started = true;
        }
        ui.add_space(4.0);
        ui.spacing_mut().item_spacing.y = 0.0;
        let current = self.current_track_ref();
        let mut play = None;
        let mut remove = None;
        for (i, t) in tracks.iter().enumerate() {
            // RTL: remove button sized first, row fills the exact remainder (no overflow).
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), 42.0),
                egui::Layout::right_to_left(egui::Align::Center),
                |ui| {
                    if removable
                        && ui
                            .add(egui::Button::new("❌").frame(false))
                            .on_hover_text("Remove")
                            .clicked()
                    {
                        remove = Some(i);
                    }
                    let here = current.as_ref().is_some_and(|c| c.same_song(t));
                    let row_w = ui.available_width();
                    if crate::theme::ios_row(ui, row_w, None, &t.label, &[], here, false).clicked()
                    {
                        play = Some(i);
                    }
                },
            );
        }
        if let Some(i) = play {
            self.play_collection_at(tracks.clone(), i);
            started = true;
        }
        if let Some(i) = remove {
            match view {
                LibraryView::Liked => {
                    let t = tracks[i].clone();
                    self.p.library.liked.retain(|x| !x.same_song(&t));
                }
                LibraryView::Playlist(p) => {
                    if let Some(pl) = self.p.library.playlists.get_mut(p) {
                        pl.tracks.remove(i);
                    }
                }
                _ => {}
            }
        }
        started
    }

    /// FL-Studio-style top-bar performance meters: 🖥 DSP load and 🎶 active voices, each a
    /// small scrolling graph with the exact numbers on hover.
    fn meters_ui(&self, ui: &mut egui::Ui) {
        let cpu = self.cpu_history.back().copied().unwrap_or(0.0);
        let danger = ui.visuals().error_fg_color;
        let accent = crate::theme::ACCENT;
        let color = if cpu > 0.85 { danger } else { accent };
        draw_meter(
            ui,
            "🖥",
            &self.cpu_history,
            1.0,
            color,
            format!("{:.0}%", cpu * 100.0),
            format!("DSP load: {:.0}%", cpu * 100.0),
        );
        let voices = self.voice_history.back().copied().unwrap_or(0.0);
        let scale = self.voice_history.iter().fold(16.0f32, |m, &v| m.max(v));
        draw_meter(
            ui,
            "🎶",
            &self.voice_history,
            scale,
            accent,
            format!("{voices:.0}"),
            format!("Voices: {voices:.0}"),
        );
    }

    /// The synthesis settings (shared between the desktop side panel and the mobile tab).
    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        // The settings panel edits whichever console the current song plays on; the DS and GBA
        // keep independent copies.
        let device_name = self.current_device_name();
        let is_gba = self.current_is_gba();
        // Snapshot before the mutable `d` borrow below — the high-band-compressor GR bar reads
        // this inside the HBC sub-block.
        let gr_db = self.high_comp_gr_db;
        let d = self.device_settings_mut();
        ui.heading("Settings");
        ui.label(format!("Settings are stored independently for each supported emulated device. You are currently editing the settings for: {device_name}"));
        // GBA presets: a one-click swap of the whole settings block. "Enhanced" is the polished
        // default; "Original" is the raw m4a chain (8-bit mixer, linear/nearest resampling, no DSP).
        if is_gba {
            ui.horizontal(|ui| {
                ui.label("Presets:");
                let te = d.track_enables;
                if ui
                    .button("Enhanced")
                    .on_hover_text(
                        "The polished default: stereo widening, clean sinc voices, crunchy mixer \
                         upsample and a high-shelf de-harsher.",
                    )
                    .clicked()
                {
                    *d = PerDeviceSettings::enhanced_gba();
                    d.track_enables = te;
                }
                if ui
                    .button("Original")
                    .on_hover_text(
                        "The raw m4a signal chain: no stereo widening or smoothing, the intermediate \
                         mixer at the GBA-native 13379 Hz crushed to 8-bit, nearest-neighbour mixer \
                         upsample and no EQ.",
                    )
                    .clicked()
                {
                    *d = PerDeviceSettings::original_gba();
                    d.track_enables = te;
                }
            });
        }
        ui.separator();
        ui.checkbox(
            &mut d.stereo_separation,
            "Stereo separation: Apply a stereo widener to panned instruments",
        );
        ui.add_enabled_ui(d.stereo_separation, |ui| {
            egui::ComboBox::from_id_salt("delay_smoothing")
                .selected_text(match d.delay_smoothing_choice {
                    1 => "No delay change during notes",
                    _ => "No smoothing",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut d.delay_smoothing_choice, 0, "No smoothing")
                        .on_hover_text("Pan changes move the widening delays immediately.");
                    ui.selectable_value(
                        &mut d.delay_smoothing_choice,
                        1,
                        "No delay change during notes",
                    )
                        .on_hover_text(
                            "Defer widening-delay changes until the track is silent, so they \
                             never pop in the middle of a playing note.",
                        );
                });
                ui.checkbox(
                    &mut d.force_stereo_separation,
                    "Force stereo separation: Apply a contrived stereo widener to instruments that are center-panned",
                );
                ui.checkbox(&mut d.bass_mono, "Keep bass centered");
                ui.horizontal(|ui| {
                    ui.add_enabled(
                        d.bass_mono,
                        egui::Slider::new(&mut d.bass_mono_freq, 40.0..=800.0)
                            .text("Bass crossover")
                            .suffix(" Hz")
                            .logarithmic(true),
                    )
                        .on_hover_text(
                            "Frequencies below this stay glued to the center; \
                        mids and treble are widened.",
                        );
                });
            });
        ui.separator();
        // Every de-click control lives here, whatever stage of the chain it acts on: the two
        // pop-smoothing kinds, the ramp time and edge they share, pan smoothing, and the stereo
        // widener's delay handling.
        ui.label("Smoothing (anti-pop & clicks)");
        ui.checkbox(
            &mut d.instrument_resample.smooth_psg_pops,
            "Smooth PSG pops",
        )
        .on_hover_text(
            "Slew PSG channel gains so notes turning abruptly on and off don't click. Unchecked \
                preserves the hardware's hard edges.",
        );
        ui.checkbox(
            &mut d.instrument_resample.smooth_sample_pops,
            "Smooth sample pops",
        )
        .on_hover_text(
            "Slew sampled (DirectSound / SWAR) voice gains so notes starting or cut mid-waveform \
                don't click. Unchecked preserves the original edges.",
        );
        let any_pop_smoothing =
            d.instrument_resample.smooth_psg_pops || d.instrument_resample.smooth_sample_pops;
        ui.add_enabled_ui(any_pop_smoothing, |ui| {
            ui.horizontal(|ui| {
                ui.label("Smooth on");
                egui::ComboBox::from_id_salt("pop_smooth_edge")
                    .selected_text(d.instrument_resample.pop_smooth_edge.text())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut d.instrument_resample.pop_smooth_edge,
                            PopSmoothingEdge::Attack,
                            PopSmoothingEdge::Attack.text(),
                        )
                        .on_hover_text(
                            "Ramp only the rise, so a note fades in but is still cut off instantly.",
                        );
                        ui.selectable_value(
                            &mut d.instrument_resample.pop_smooth_edge,
                            PopSmoothingEdge::Release,
                            PopSmoothingEdge::Release.text(),
                        )
                        .on_hover_text(
                            "Ramp only the fall, so a note keeps its hard attack but is never cut \
                             off mid-waveform.",
                        );
                        ui.selectable_value(
                            &mut d.instrument_resample.pop_smooth_edge,
                            PopSmoothingEdge::Both,
                            PopSmoothingEdge::Both.text(),
                        )
                        .on_hover_text("Ramp both edges of every note.");
                    })
                    .response
                    .on_hover_text(
                        "Which edge of a note the de-click ramp applies to. The other edge keeps \
                         the hardware's instant jump.",
                    );
            });
            ui.add(
                egui::Slider::new(&mut d.instrument_resample.pop_slew_ms, 0.1..=20.0)
                    .text("De-click slew time")
                    .suffix(" ms")
                    .logarithmic(true),
            )
            .on_hover_text(
                "How long the de-click gain ramp takes for the smoothed voices above. Longer is \
                    gentler (fewer clicks) but softens note attacks; ~2 ms is a click-free default.",
            );
        });
        ui.checkbox(
            &mut d.smooth_pan,
            "Smooth panning changes (slew the pan to avoid clicks)",
        )
        .on_hover_text(
            "Ramp the left/right pan gains over a few milliseconds on a pan change instead of \
             stepping them, removing the click an abrupt pan jump would otherwise make.",
        );
        ui.separator();
        ui.label("Instrument-to-mixer resampling");
        resample_combo(
            ui,
            "instrument-to-mixer-resampling",
            &mut d.instrument_resample.choice,
        );
        // Only the options of the selected mode are shown.
        if matches!(
            d.instrument_resample.choice,
            InstrumentResampleChoice::SincOutputNyquist
                | InstrumentResampleChoice::SincSampleNyquist
        ) {
            sinc_taps_slider(ui, &mut d.instrument_resample.sinc_taps);
        }
        if d.instrument_resample.choice == InstrumentResampleChoice::SincOutputNyquist {
            ui.add(
                egui::Slider::new(
                    &mut d.instrument_resample.psg_cutoff_hz,
                    1000..=InstrumentResampleMode::CUTOFF_OFF_HZ,
                )
                .text("PSG cutoff")
                .suffix(" Hz")
                .logarithmic(true),
            )
            .on_hover_text("Low-pass cutoff for the PSG (square/wave/noise) channels.");
            ui.add(
                egui::Slider::new(
                    &mut d.instrument_resample.sampler_cutoff_hz,
                    1000..=InstrumentResampleMode::CUTOFF_OFF_HZ,
                )
                .text("Sampler cutoff")
                .suffix(" Hz")
                .logarithmic(true),
            )
            .on_hover_text("Low-pass cutoff for the sampled (DirectSound / SWAR) channels.");
        }
        if is_gba {
            ui.checkbox(&mut d.remove_sample_dc_offset, "Remove sample DC offset")
                .on_hover_text(
                    "Subtract each DirectSound sample's average level so it sits centered on zero, \
                    matching the GBA's AC-coupled output and removing the thump a biased sample \
                    makes when its envelope opens or closes. See \"Stats for Nerds\" for how far \
                    each sample is shifted. Off preserves the raw ROM data.",
                );
            ui.add_enabled(
                d.use_mixer,
                egui::Checkbox::new(&mut d.mp2k_reverb, "MP2K reverb"),
            )
            .on_hover_text(
                "Apply the GBA sound engine's built-in reverb (the SoundMainRAM pre-pass): a \
                    mono feedback echo on the sampled bus, using the amount each song requests. \
                    Requires the intermediate mixer (below). PSG voices stay dry, as on hardware.",
            );
        }
        ui.separator();
        ui.label("Mixer settings");
        {
            ui.checkbox(
                    &mut d.use_mixer,
                    "Use intermediate mixer for sampled instruments",
                )
                .on_hover_text(
                    "Route the sampled (non-PSG) instruments through an intermediate mixer \
                        running at the mixer rate below, then resample that bus up to the output \
                        rate. PSG (square/wave/noise) voices bypass it and play at the output rate. \
                        Emulates hardware that mixes its sampled channels at a low rate.",
                );
            ui.add_enabled(
                d.use_mixer,
                egui::Slider::new(&mut d.mixer_sample_rate, 10000..=48000)
                    .step_by(1.0)
                    .text("Mixer rate")
                    .suffix(" Hz")
                    .logarithmic(false),
            );
            ui.add_enabled(
                d.use_mixer,
                egui::Checkbox::new(&mut d.bitcrush_mixer, "Bitcrush the mixer bus"),
            )
            .on_hover_text(
                "Quantize the intermediate mixer bus to a reduced bit depth, the way the GBA's m4a \
                    software mixer renders DirectSound into an 8-bit PCM buffer: truncated, and \
                    wrapping in two's complement on overflow (as the hardware accumulator does), not \
                    clipped. Only the sampled bus is affected; PSG voices bypass the mixer.",
            );
            ui.add_enabled(
                d.use_mixer && d.bitcrush_mixer,
                egui::Slider::new(&mut d.bitcrush_bits, 1..=16)
                    .text("Bit depth")
                    .suffix("-bit"),
            )
            .on_hover_text("8-bit matches GBA hardware; lower is grittier.");
        }
        ui.separator();

        ui.label("Mixer-to-output resampling");
        let use_mixer = d.use_mixer;
        let ms = &mut d.mixer_resample;
        let psg_crunch_compensation = &mut d.psg_crunch_compensation;
        ui.add_enabled_ui(use_mixer, |ui| {
            resample_combo(ui, "mixer-to-output-resampling", &mut ms.choice);
            // Same per-selected-mode controls as the instrument stage, minus the PSG-specific ones
            // (the bus is a finished mix): the sinc modes show taps, crunch shows a single cutoff.
            if matches!(
                ms.choice,
                InstrumentResampleChoice::SincOutputNyquist
                    | InstrumentResampleChoice::SincSampleNyquist
            ) {
                sinc_taps_slider(ui, &mut ms.sinc_taps);
            }
            if ms.choice == InstrumentResampleChoice::SincOutputNyquist {
                ui.add(
                    egui::Slider::new(
                        &mut ms.cutoff_hz,
                        1000..=InstrumentResampleMode::CUTOFF_OFF_HZ,
                    )
                    .text("Cutoff")
                    .suffix(" Hz")
                    .logarithmic(true),
                )
                .on_hover_text("Low-pass cutoff for the mixer bus in crunch mode.");

                ui.checkbox(
                    psg_crunch_compensation,
                    "Compensate PSG level for crunch high-end loss",
                )
                .on_hover_text(
                    "Crunch resampling darkens the DirectSound bus' high end (less aliasing \
                        energy), leaving the PSG voices sitting too loud. This colours the PSG bus \
                        with the same measured high-frequency rolloff so the two stay balanced.",
                );
            }
        });
        ui.label(
            egui::RichText::new(
                "Note: the output here is a fixed 32768 Hz, emulating the hardware DACs. A further \
                resampling stage converts that to your audio device's rate, which isn't adjustable \
                here.",
            )
            .weak()
            .small(),
        );
        ui.separator();

        // Master high-shelf EQ — per device, like the resampling settings above.
        ui.label("Master high-shelf EQ");
        {
            ui.checkbox(&mut d.shelf.enabled, "Enable high-shelf")
                .on_hover_text(
                    "A master high-shelf EQ on the final mix. Negative gain tames harsh highs / \
                    click brightness; positive adds air.",
                );
            ui.add_enabled_ui(d.shelf.enabled, |ui| {
                ui.add(
                    egui::Slider::new(&mut d.shelf.gain_db, -24.0..=24.0)
                        .text("Gain")
                        .suffix(" dB"),
                );
                ui.add(
                    egui::Slider::new(&mut d.shelf.cutoff_hz, 500.0..=16000.0)
                        .text("Cutoff")
                        .suffix(" Hz")
                        .logarithmic(true),
                );
                ui.add(egui::Slider::new(&mut d.shelf.q, 0.1..=2.0).text("Q"));
                ui.add(
                    egui::Slider::new(&mut d.shelf.order, 2..=16)
                        .step_by(2.0)
                        .text("Order"),
                )
                .on_hover_text(
                    "Higher order steepens the shelf transition (more biquad sections).",
                );
            });
        }
        ui.separator();
        // Per-bus high-band compressor (PSG and sampled). Each bus gets its own enable so a
        // user can tame DirectSound highs without touching PSG, or vice-versa. Both require
        // the intermediate mixer (the buses are only isolated there).
        ui.label("High-band compressor");
        let hbc = &mut d.high_band_compress;
        let use_mixer = d.use_mixer;
        ui.add_enabled_ui(use_mixer, |ui| {
            ui.checkbox(&mut hbc.enabled_psg, "Apply to PSG voices")
                .on_hover_text(
                    "Run the high-band compressor on the PSG (square/wave/noise) bus. PSG \
                        voices bypass the intermediate mixer, so this runs at the output rate.",
                );
            ui.checkbox(
                &mut hbc.enabled_sampled,
                "Apply to sampled (DirectSound) voices",
            )
            .on_hover_text(
                "Run the high-band compressor on the sampled (DirectSound / SWAR) bus, at \
                        the intermediate mixer rate. Catches harsh transients the static \
                        high-shelf above can't track dynamically.",
            );
            let any = hbc.enabled_psg || hbc.enabled_sampled;
            ui.add_enabled_ui(any, |ui| {
                ui.add(
                    egui::Slider::new(&mut hbc.cutoff_hz, 500.0..=16000.0)
                        .text("Cutoff")
                        .suffix(" Hz")
                        .logarithmic(true),
                )
                .on_hover_text(
                    "LPF/HPF split frequency. Only signal above this is compressed; the \
                        rest passes untouched.",
                );
                ui.add(
                    egui::Slider::new(&mut hbc.threshold_db, -120.0..=0.0)
                        .text("Threshold")
                        .suffix(" dB"),
                )
                .on_hover_text("Sidechain level above which the high band is compressed.");
                ui.add(
                    egui::Slider::new(&mut hbc.ratio, 1.0..=20.0)
                        .text("Ratio")
                        .logarithmic(true),
                )
                .on_hover_text(
                    "Compression ratio above the threshold. 1:1 = no compression; higher = \
                        more aggressive taming.",
                );
                ui.add(
                    egui::Slider::new(&mut hbc.attack_ms, 0.1..=50.0)
                        .text("Attack")
                        .suffix(" ms")
                        .logarithmic(true),
                )
                .on_hover_text(
                    "How fast the compressor reacts to a rising sidechain. Fast catches \
                        transients; slow lets them through.",
                );
                ui.add(
                    egui::Slider::new(&mut hbc.release_ms, 1.0..=500.0)
                        .text("Release")
                        .suffix(" ms")
                        .logarithmic(true),
                )
                .on_hover_text(
                    "How fast the compressor recovers after the sidechain drops. Too fast \
                        pumps; too slow squashes the next transient.",
                );
                ui.add(
                    egui::Slider::new(&mut hbc.makeup_db, 0.0..=24.0)
                        .text("Makeup")
                        .suffix(" dB"),
                )
                .on_hover_text(
                    "Gain applied after compression. Use it to bring the tamed signal back \
                        up to the desired level.",
                );
                draw_gr_bar(ui, gr_db);
            });
        });
        ui.separator();
        ui.label("Tuning system");
        egui::ComboBox::from_id_salt("tuning")
            .selected_text(if d.tuning_choice == 0 {
                "Equal temperament"
            } else {
                "Pure (Pythagorean)"
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut d.tuning_choice, 0, "Equal temperament");
                ui.selectable_value(&mut d.tuning_choice, 1, "Pure (Pythagorean)");
            });
        if d.tuning_choice == 1 {
            ui.add(egui::Slider::new(&mut d.pure_tonic, 0..=11).text("Tonic (semitones from A)"));
        }
        // `d` (the device-settings borrow) is no longer used past here, so we can touch `self`.
        ui.separator();
        if ui
            .button("📊 Stats for Nerds")
            .on_hover_text("Per-sample DC-offset analysis for the current song.")
            .clicked()
        {
            self.stats_open = true;
            self.stats_cache = None; // recompute for whatever is playing now
        }
        // Maintainer dev-tool: turn the library list into an inline title/order editor that writes
        // the curated names back to this crate's source-tree JSON (only when run from the checkout).
        #[cfg(not(target_arch = "wasm32"))]
        {
            ui.separator();
            ui.checkbox(&mut self.song_edit_enabled, "Edit song names")
                .on_hover_text(
                "Make the song list editable (rename + reorder), then \"Save to JSON\" writes the \
                 titles and order back to crates/optime-app/src/song_names/*.json. A maintainer \
                 tool — the saved file is baked in on the next build; only works when the app is \
                 run from the source checkout.",
            );
        }

        // Chord annotation. Available on the web too — the labelling itself needs nothing but the
        // engine, and only where the JSON *lands* differs (source tree vs browser download).
        ui.separator();
        let mut annotating = self.annotation.enabled;
        let changed = ui
            .checkbox(&mut annotating, "Data annotation mode")
            .on_hover_text(
                "Turn the piano roll into a chord-labelling platform: scroll to zoom (shift+wheel \
                 or middle-drag to pan), drag the overview bar to scrub, drag in the chord lane to \
                 select a range, then type a chord symbol (Am7, F#m, Csus4, NC; trailing ? = \
                 quality uncertain). The song is rendered to memory on entry so bars can be looped \
                 seamlessly (a few seconds; ~47 MB for 3 minutes).",
            )
            .changed();
        if changed {
            self.set_annotation_mode(annotating);
        }

        // Build provenance: which commit this binary was built from, and when. Embedded at compile
        // time by build.rs (falls back to "unknown" when git isn't available).
        ui.separator();
        ui.label(
            egui::RichText::new(format!(
                "Build {} · {}",
                env!("OPTIME_GIT_HASH"),
                env!("OPTIME_BUILD_TIME"),
            ))
            .weak()
            .small(),
        )
        .on_hover_text(format!("Commit date: {}", env!("OPTIME_COMMIT_DATE")));
    }

    /// The DC-offset stats for the current song, computed (and cached) on demand. `None` when no
    /// song is loaded.
    fn current_waveform_stats(&mut self) -> Option<&[optime_core::WaveformDcStat]> {
        let key = self
            .current_song
            .and_then(|i| self.songs.get(i))
            .map(|s| (s.archive_index, s.song_id))?;
        if self.stats_cache.as_ref().map(|(k, _)| *k) != Some(key) {
            let stats = self.archives[key.0].waveform_dc_stats(key.1);
            self.stats_cache = Some((key, stats));
        }
        self.stats_cache.as_ref().map(|(_, v)| v.as_slice())
    }

    /// The "Stats for Nerds" window: every decoded sample for the current song with how far it was
    /// shifted to remove its DC offset, most-shifted first.
    fn stats_ui(&mut self, ctx: &egui::Context) {
        if !self.stats_open {
            return;
        }
        let mut open = true;
        let stats: Vec<optime_core::WaveformDcStat> = self
            .current_waveform_stats()
            .map(|s| s.to_vec())
            .unwrap_or_default();
        egui::Window::new("📊 Stats for Nerds")
            .open(&mut open)
            .default_width(480.0)
            .default_height(420.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.label(
                    "DirectSound samples decoded for the current song. \"DC shift\" is how far \
                     each sample's mean was offset from zero (as a percentage of full scale), \
                     removed on decode to match the GBA's AC-coupled output. Most-shifted first.",
                );
                ui.separator();
                if stats.is_empty() {
                    ui.label(
                        "No analyzable samples — load a GBA song (the DS and DSE engines aren't \
                         analyzed here).",
                    );
                    return;
                }
                ui.label(format!("{} samples", stats.len()));
                egui::ScrollArea::vertical().show(ui, |ui| {
                    egui::Grid::new("dc_stats_grid")
                        .num_columns(4)
                        .striped(true)
                        .spacing([16.0, 4.0])
                        .show(ui, |ui| {
                            ui.strong("Sample");
                            ui.strong("DC shift");
                            ui.strong("Length");
                            ui.strong("Rate");
                            ui.end_row();
                            for s in &stats {
                                ui.monospace(&s.label);
                                ui.label(format!("{:.3}%", s.dc_shift * 100.0));
                                ui.label(format!("{}", s.length));
                                ui.label(format!("{:.0} Hz", s.sample_rate));
                                ui.end_row();
                            }
                        });
                });
            });
        self.stats_open = open;
    }

    /// The compact phone layout: a bottom navigation bar (Now Playing / Library / Playlists /
    /// Settings) with a floating mini-player above it on every screen except Now Playing.
    fn mobile_ui(&mut self, ctx: &egui::Context, snap: &VisSnapshot) {
        egui::TopBottomPanel::top("m_meters")
            .show_separator_line(false)
            .show(ctx, |ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.allocate_ui_with_layout(
                        egui::vec2(220.0, 20.0),
                        egui::Layout::left_to_right(egui::Align::Center),
                        |ui| self.meters_ui(ui),
                    );
                });
            });

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
                // iOS-style tab bar: hairline on top, accent tint on the active tab.
                let top = ui.max_rect().top();
                ui.painter().line_segment(
                    [
                        egui::pos2(ui.max_rect().left(), top),
                        egui::pos2(ui.max_rect().right(), top),
                    ],
                    egui::Stroke::new(0.5_f32, crate::theme::HAIRLINE),
                );
                ui.columns(tabs.len(), |cols| {
                    for (col, (tab, icon, label)) in cols.iter_mut().zip(tabs) {
                        let selected = self.mobile_tab == tab;
                        let (rect, resp) = col.allocate_exact_size(
                            egui::vec2(col.available_width(), 50.0),
                            egui::Sense::click(),
                        );
                        let color = if selected {
                            crate::theme::ACCENT
                        } else {
                            crate::theme::TEXT_DIM
                        };
                        // Painter-drawn so the label is exactly centered under the icon.
                        let painter = col.painter_at(rect);
                        painter.text(
                            rect.center() - egui::vec2(0.0, 10.0),
                            egui::Align2::CENTER_CENTER,
                            icon,
                            egui::FontId::proportional(18.0),
                            color,
                        );
                        painter.text(
                            rect.center() + egui::vec2(0.0, 13.0),
                            egui::Align2::CENTER_CENTER,
                            label,
                            egui::FontId::proportional(11.0),
                            color,
                        );
                        if resp.clicked() {
                            self.mobile_tab = tab;
                        }
                    }
                });
                ui.add_space(4.0);
            });

        if self.mobile_tab != MobileTab::NowPlaying && self.current_song.is_some() {
            egui::TopBottomPanel::bottom("m_mini")
                .show_separator_line(false)
                .frame(
                    egui::Frame::none()
                        .fill(crate::theme::BG)
                        .inner_margin(egui::Margin::symmetric(10.0, 4.0)),
                )
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
                        if ui
                            .add_enabled(
                                self.current_song.is_some(),
                                egui::Button::new("💾 Export WAV"),
                            )
                            .clicked()
                        {
                            self.export_wav();
                        }
                        if ui
                            .add_enabled(
                                self.gba_loaded(),
                                egui::Button::new("🎵 Export GBA audio data"),
                            )
                            .on_hover_text(
                                "Save an audio-only copy of the loaded GBA ROM: game code, \
                                 sprites, and everything else are stripped; only the music \
                                 data the player needs is kept.",
                            )
                            .clicked()
                        {
                            self.export_gba_audio();
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
        let height = 52.0;
        let (rect, resp) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), height),
            egui::Sense::click(),
        );
        // Floating-card look: soft drop shadow, rounded fill.
        let painter = ui.painter();
        painter.rect_filled(
            rect.translate(egui::vec2(0.0, 2.0)).expand(2.0),
            14.0,
            egui::Color32::from_black_alpha(70),
        );
        let painter = ui.painter_at(rect);
        let bg = if resp.hovered() {
            crate::theme::CARD_HI
        } else {
            crate::theme::CARD
        };
        painter.rect_filled(rect, 12.0, bg);

        // A small EQ-bar "now playing" indicator. Deliberately static (a fixed silhouette when
        // playing, flat when paused) rather than animated: the mini-player only appears on the
        // non-piano-roll tabs, which idle at 1 Hz to save battery, so animating it here would
        // force continuous repaints and defeat that.
        let accent = crate::theme::ACCENT;
        let playing = !self.paused;
        let bar_w = 4.0;
        let max_h = height - 16.0;
        let levels = [0.7, 0.4, 0.9, 0.55];
        for (b, &level) in levels.iter().enumerate() {
            let level = if playing { level } else { 0.15 };
            let h = 4.0 + level * (max_h - 4.0);
            let x = rect.left() + 12.0 + b as f32 * (bar_w + 3.0);
            let bar = egui::Rect::from_min_max(
                egui::pos2(x, rect.center().y + max_h / 2.0 - h),
                egui::pos2(x + bar_w, rect.center().y + max_h / 2.0),
            );
            painter.rect_filled(bar, 2.0, accent);
        }

        // Song title.
        let title = self
            .current_song
            .and_then(|i| self.songs.get(i))
            .map(|s| s.label.clone())
            .unwrap_or_default();
        let text_rect = egui::Rect::from_min_max(
            egui::pos2(rect.left() + 48.0, rect.top()),
            egui::pos2(rect.right() - 92.0, rect.bottom()),
        );
        // Clipped to its own rect so long titles never run under the buttons.
        ui.painter_at(text_rect).text(
            text_rect.left_center(),
            egui::Align2::LEFT_CENTER,
            title,
            egui::FontId::proportional(14.5),
            crate::theme::TEXT,
        );

        // Transport buttons layered on the right edge of the bar.
        let btn = egui::vec2(36.0, 36.0);
        let pause_icon = if self.paused { "▶" } else { "⏸" };
        let pause_rect =
            egui::Rect::from_center_size(egui::pos2(rect.right() - 64.0, rect.center().y), btn);
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
            self.send_transport(1);
        }

        if resp.clicked() {
            self.mobile_tab = MobileTab::NowPlaying;
        }
    }

    /// The full-screen Now Playing view: visualizer with animated swipe navigation plus the
    /// large transport.
    fn mobile_now_playing(&mut self, ctx: &egui::Context, snap: &VisSnapshot) {
        egui::TopBottomPanel::bottom("m_transport").show(ctx, |ui| {
            use crate::theme::icon_button;
            ui.add_space(8.0);
            // Title + status above the controls.
            let title = self
                .current_song
                .and_then(|i| self.songs.get(i))
                .map(|s| s.label.clone())
                .unwrap_or_else(|| "No song loaded".to_owned());
            ui.vertical_centered(|ui| {
                ui.label(
                    egui::RichText::new(title)
                        .strong()
                        .size(18.0)
                        .color(crate::theme::TEXT),
                );
                ui.label(
                    egui::RichText::new(&self.status)
                        .size(11.5)
                        .color(crate::theme::TEXT_DIM),
                );
            });
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                // iOS-style transport: five small circular buttons around a large filled
                // play/pause disc, evenly spaced. The heart slot is always present so the
                // row never shifts.
                let have_songs = !self.songs.is_empty();
                let small = 42.0;
                let big = 58.0;
                let total = 5.0 * small + big;
                let spacing = (ui.available_width() - total).max(0.0) / 7.0;
                ui.spacing_mut().item_spacing.x = spacing;
                // Pin the row height to the largest button up front, so the small buttons
                // placed before the big play disc center to the same height as those after
                // it (otherwise egui center-aligns them against the shorter row they see).
                ui.set_min_height(big);
                ui.add_space(spacing);

                if icon_button(ui, "🔀", small, 18.0, false, self.p.shuffle, true).clicked() {
                    self.p.shuffle = !self.p.shuffle;
                    self.toggle_shuffle();
                }
                if icon_button(ui, "⏮", small, 20.0, false, false, have_songs).clicked() {
                    self.send_transport(-1);
                }
                let pause_icon = if self.paused || self.current_song.is_none() {
                    "▶"
                } else {
                    "⏸"
                };
                if icon_button(ui, pause_icon, big, 26.0, true, false, have_songs).clicked() {
                    self.paused = !self.paused;
                }
                if icon_button(ui, "⏭", small, 20.0, false, false, have_songs).clicked() {
                    self.send_transport(1);
                }
                let repeat_icon = match self.p.repeat {
                    RepeatMode::One => "🔂",
                    _ => "🔁",
                };
                let repeat = icon_button(
                    ui,
                    repeat_icon,
                    small,
                    18.0,
                    false,
                    self.p.repeat != RepeatMode::Off,
                    true,
                );
                if repeat.on_hover_text("Repeat: off / all / one").clicked() {
                    self.p.repeat = self.p.repeat.next();
                    self.push_repeat();
                }
                let current = self.current_track_ref();
                let liked = current.as_ref().is_some_and(|t| self.p.library.is_liked(t));
                // Same glyph either way (the outline heart isn't in egui's fonts); the
                // active flag tints it accent when liked, white otherwise.
                let heart = "❤";
                if icon_button(ui, heart, small, 18.0, false, liked, current.is_some()).clicked()
                    && let Some(t) = current
                {
                    self.p.library.toggle_liked(&t);
                }
            });
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.add_space(8.0);
                ui.label(egui::RichText::new("🔈").color(crate::theme::TEXT_DIM));
                ui.spacing_mut().slider_width = (ui.available_width() - 40.0).max(60.0);
                ui.add(
                    egui::Slider::new(&mut self.p.volume, 0.0..=1.0)
                        .show_value(false)
                        .trailing_fill(true),
                );
                ui.label(egui::RichText::new("🔊").color(crate::theme::TEXT_DIM));
            });
            ui.add_space(10.0);
        });

        let card_frame = egui::Frame::none()
            .fill(crate::theme::BG)
            .inner_margin(egui::Margin::symmetric(10.0, 8.0));
        egui::CentralPanel::default()
            .frame(card_frame)
            .show(ctx, |ui| {
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
                    // Keep a preview of the song the swipe is heading toward, so it is already
                    // visible (notes and all) next to the outgoing roll.
                    let dir: isize = if self.swipe_offset < -4.0 {
                        1
                    } else if self.swipe_offset > 4.0 {
                        -1
                    } else {
                        0
                    };
                    if dir == 0 {
                        self.swipe_preview = None;
                    } else if self.swipe_preview.as_ref().map(|p| p.dir) != Some(dir) {
                        self.swipe_preview = self.build_swipe_preview(dir);
                    }
                } else if resp.drag_stopped() {
                    if self.swipe_offset.abs() >= 0.25 * w {
                        // Committed: the old roll keeps sliding out while the preview (already
                        // populated with the next song's notes) becomes the live roll.
                        let exit_side = self.swipe_offset.signum();
                        let old_roll = std::mem::take(&mut self.piano_roll);
                        self.swipe_out = Some((old_roll, exit_side));
                        if let Some(p) = self.swipe_preview.take() {
                            // `dir`: +1 = next (dragged left), -1 = previous.
                            self.send_transport(p.dir);
                            self.piano_roll = p.roll;
                            if let Some(look) = p.look {
                                self.look_ahead = Some(look);
                            }
                        } else {
                            self.send_transport(if exit_side < 0.0 { 1 } else { -1 });
                        }
                        self.swipe_offset -= exit_side * w;
                    } else {
                        self.swipe_preview = None;
                    }
                } else if self.swipe_offset != 0.0 {
                    // Spring back / slide in.
                    self.swipe_offset *= (-12.0 * dt).exp();
                    if self.swipe_offset.abs() < 0.5 {
                        self.swipe_offset = 0.0;
                        self.swipe_preview = None;
                        self.swipe_out = None;
                    }
                    ui.ctx().request_repaint();
                }
                self.swipe_gain = 1.0 - (self.swipe_offset.abs() / w).clamp(0.0, 1.0);

                // The live roll, offset by the swipe.
                let child_rect = rect.translate(egui::vec2(self.swipe_offset, 0.0));
                let mut child = ui.new_child(egui::UiBuilder::new().max_rect(child_rect));
                child.set_clip_rect(rect);
                self.piano_roll.draw(&mut child, snap.active);

                // The incoming song's preview alongside it while dragging.
                let offset = self.swipe_offset;
                if let Some(p) = &mut self.swipe_preview {
                    let r = rect.translate(egui::vec2(offset + p.dir as f32 * w, 0.0));
                    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(r));
                    child.set_clip_rect(rect);
                    p.roll.draw(&mut child, true);
                }
                // The old song's roll sliding out after a committed swipe.
                if let Some((roll, side)) = &mut self.swipe_out {
                    let r = rect.translate(egui::vec2(offset + *side * w, 0.0));
                    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(r));
                    child.set_clip_rect(rect);
                    roll.draw(&mut child, true);
                }

                // Round the corners of the (rectangularly-clipped) roll by masking the four
                // square corners with the panel background, then draw the card outline on top.
                crate::theme::mask_rounded_corners(ui.painter(), rect, 12.0, crate::theme::BG);
                ui.painter().rect_stroke(
                    rect,
                    12.0,
                    egui::Stroke::new(1.0_f32, crate::theme::HAIRLINE),
                );
            });
    }

    /// The mobile library: file open, demo archives, and the song list. Selecting a song starts
    /// it in the mini-player rather than jumping to the visualizer.
    fn mobile_library(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Library");
                ui.add_space(6.0);
                if ui.button("📂 Open ROM / SDAT / GBA…").clicked() {
                    self.open_file_dialog();
                }
                crate::theme::section_header(ui, "Demo archives");
                {
                    let prev_spacing = ui.spacing().item_spacing.y;
                    ui.spacing_mut().item_spacing.y = 0.0;
                    let mut requested = None;
                    let active = &self.current_source;
                    for (label, stem) in all_demos() {
                        let selected = active == stem;
                        let w = ui.available_width();
                        if crate::theme::ios_row(ui, w, Some("💿"), label, &[], selected, true)
                            .clicked()
                        {
                            requested = Some((*stem, *label));
                        }
                    }
                    ui.spacing_mut().item_spacing.y = prev_spacing;
                    if let Some((stem, label)) = requested {
                        self.request_demo(stem, label);
                    }
                }
                crate::theme::section_header(ui, "All songs");
                self.song_list_ui(ui);
            });
        });
    }

    /// All keyboard shortcuts pass through here so they share one inhibition rule: whenever egui is
    /// routing keystrokes to a focused widget (e.g. typing a title in the song-name editor, or any
    /// text field / active drag-value), every shortcut is suppressed — the keystroke belongs to that
    /// widget. Otherwise each shortcut *consumes* its key so it can't also reach a background widget.
    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        // A focused text field (or any widget wanting keyboard input) inhibits all shortcuts.
        if ctx.wants_keyboard_input() {
            return;
        }
        let (prev, next, toggle) = ctx.input_mut(|i| {
            (
                i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft),
                i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight),
                i.consume_key(egui::Modifiers::NONE, egui::Key::Space),
            )
        });
        // Arrow keys switch songs; spacebar toggles play/pause (starting the first song if idle).
        if prev {
            self.send_transport(-1);
        }
        if next {
            self.send_transport(1);
        }
        if toggle {
            if self.current_song.is_some() {
                self.paused = !self.paused;
            } else if !self.songs.is_empty() {
                self.play_from_list(0);
            }
        }
    }
}

impl eframe::App for OptimeApp {
    /// Persists the library, playback prefs, and synth settings (native: disk; web: localStorage).
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        // Capture the currently playing song so a refresh/relaunch restores it.
        if let Some(track) = self.current_track_ref() {
            self.p.last_track = Some(track);
        }
        eframe::set_value(storage, eframe::APP_KEY, &self.p);
    }

    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.ensure_audio(ctx);
        #[cfg(target_arch = "wasm32")]
        self.keep_audio_alive(ctx);
        self.poll_pending_file();
        self.update_library_order(ctx);

        // All keyboard shortcuts (song nav + play/pause), with the focused-widget inhibition.
        self.handle_shortcuts(ctx);

        self.handle_media_controls(ctx, frame);

        // Apply any advance the audio thread performed on its own (keeps working while frozen),
        // then sync config + pull the visualizer snapshot for the now-current song.
        self.reconcile_playback();
        let snap = self.sync_audio();
        // Re-derived each frame by the Now Playing swipe; full volume everywhere else.
        self.swipe_gain = 1.0;

        // Advance the piano roll's smoothed playhead (frozen when paused / no song), then drive
        // the look-ahead runner so it stays buffered ahead of the playhead, and pull its notes.
        let playing = snap.active && !self.paused;
        let dt = ctx.input(|i| i.stable_dt) as f64;
        self.piano_roll.advance(&snap, dt, playing);
        if self.annotation.enabled {
            // Annotation drives everything from the bounce: notes from the whole-song overview
            // (the look-ahead's rolling window can't serve an editor that scrubs backwards) and the
            // playhead from the buffer position.
            self.sync_annotation(ctx);
        } else if let Some(look) = &mut self.look_ahead {
            let target =
                self.piano_roll.display_tick().ceil() as u32 + crate::piano_roll::RUN_AHEAD_TICKS;
            // Bounded catch-up so a stalled/zero-BPM sequence can't spin forever.
            let mut guard = 0u32;
            while look.steps_elapsed() < target && guard < 200_000 {
                look.tick();
                guard += 1;
            }
            self.piano_roll.ingest(look);
        }

        // The "Stats for Nerds" window floats over either layout.
        self.stats_ui(ctx);

        // Narrow screens (phones) get the Spotify-style mobile layout.
        if ctx.screen_rect().width() < 600.0 {
            self.mobile_ui(ctx, &snap);
            // On mobile, to save battery, only repaint the Playing and Settings tab for animating
            // the piano roll and compressor gain reduction bar, respectively.
            if matches!(self.mobile_tab, MobileTab::NowPlaying | MobileTab::Settings) {
                #[cfg(target_arch = "wasm32")]
                ctx.request_repaint_after(std::time::Duration::from_millis(33));
                #[cfg(not(target_arch = "wasm32"))]
                ctx.request_repaint();
            } else {
                ctx.request_repaint_after(std::time::Duration::from_secs(1));
            }
            return;
        }

        egui::SidePanel::left("songs")
            .resizable(true)
            .default_width(260.0)
            .width_range(200.0..=600.0)
            .show(ctx, |ui| {
                ui.heading("Optime Player");
                ui.add_space(4.0);
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Open ROM / SDAT / GBA…").clicked() {
                        self.open_file_dialog();
                    }
                });
                ui.collapsing("Demos", |ui| {
                    let mut requested = None;
                    for (label, stem) in all_demos() {
                        let mut text = egui::RichText::new(*label);
                        if self.current_source == *stem {
                            text = text.color(egui::Color32::RED);
                        }
                        if ui.button(text).clicked() {
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
                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .show(ui, |ui| {
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
                    self.send_transport(-1);
                }
                if ui.button("⏭").on_hover_text("Next (→)").clicked() {
                    self.send_transport(1);
                }
                ui.separator();
                if ui.toggle_value(&mut self.p.shuffle, "🔀 Shuffle").clicked() {
                    self.toggle_shuffle();
                }
                if ui
                    .button(self.p.repeat.label())
                    .on_hover_text("Repeat mode: off / all / one")
                    .clicked()
                {
                    self.p.repeat = self.p.repeat.next();
                    self.push_repeat();
                }
                if let Some(t) = self.current_track_ref() {
                    let liked = self.p.library.is_liked(&t);
                    let heart_color = if liked {
                        crate::theme::ACCENT
                    } else {
                        crate::theme::TEXT_DIM
                    };
                    if ui
                        .button(egui::RichText::new("❤").color(heart_color))
                        .on_hover_text("Like")
                        .clicked()
                    {
                        self.p.library.toggle_liked(&t);
                    }
                }
                ui.separator();
                ui.label("🔊");
                ui.add(
                    egui::Slider::new(&mut self.p.volume, 0.0..=1.0)
                        .show_value(false)
                        .trailing_fill(true),
                );
                ui.separator();
                if ui
                    .add_enabled(
                        self.current_song.is_some(),
                        egui::Button::new("💾 Export WAV"),
                    )
                    .clicked()
                {
                    self.export_wav();
                }
                if ui
                    .add_enabled(self.gba_loaded(), egui::Button::new("🎵 Export GBA audio"))
                    .on_hover_text(
                        "Save an audio-only copy of the loaded GBA ROM: game code, sprites, \
                         and everything else are stripped; only the music data the player \
                         needs is kept.",
                    )
                    .clicked()
                {
                    self.export_gba_audio();
                }
                // Meters live at the right edge; skip them when the bar is too narrow
                // rather than overlapping the left-to-right content.
                const METERS_W: f32 = 220.0;
                if ui.available_width() >= METERS_W {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.allocate_ui_with_layout(
                            egui::vec2(METERS_W, 20.0),
                            egui::Layout::left_to_right(egui::Align::Center),
                            |ui| self.meters_ui(ui),
                        );
                    });
                }
            });
        });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
            });
        });

        egui::SidePanel::right("settings")
            .default_width(240.0)
            .width_range(220.0..=340.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.settings_ui(ui);
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.vis_tab, VisTab::PianoRoll, "🎹 Piano Roll");
                ui.selectable_value(&mut self.vis_tab, VisTab::Tracks, "Tracks");
            });
            ui.separator();

            match self.vis_tab {
                VisTab::PianoRoll => {
                    self.piano_roll_panel(ui, &snap);
                }
                VisTab::Tracks => {
                    egui::ScrollArea::both().show(ui, |ui| {
                        visualizer::draw(ui, &snap, &mut self.track_enables);
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

/// A small horizontal gain-reduction meter for the high-band compressor: 0 dB at the right edge,
/// `−GR_FULL_SCALE_DB` at the left, fill grows leftward as `gr_db` goes more negative. Industry
/// convention (compression-only, no makeup); silent (0 dB) when neither bus is active.
fn draw_gr_bar(ui: &mut egui::Ui, gr_db: f32) {
    /// Full-scale span (left edge = `−this` dB). 
    const GR_FULL_SCALE_DB: f32 = 96.0;
    let accent = crate::theme::ACCENT;
    let warn = ui.visuals().warn_fg_color;
    let frac = (gr_db.abs() / GR_FULL_SCALE_DB).clamp(0.0, 1.0);
    let color = if frac > 0.75 { warn } else { accent };
    ui.horizontal(|ui| {
        ui.label("GR");
        let (rect, _resp) = ui.allocate_exact_size(egui::vec2(180.0, 10.0), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
        // Fill grows from the right (0 dB anchor) leftward as reduction deepens.
        let fill_w = rect.width() * frac;
        let fill_rect = egui::Rect::from_min_size(
            egui::pos2(rect.right() - fill_w, rect.top()),
            egui::vec2(fill_w, rect.height()),
        );
        if fill_w > 0.0 {
            painter.rect_filled(fill_rect, 2.0, color);
        }
        // Tick at the −12 dB mark.
        let tick_x = rect.right() - rect.width() * (12.0 / GR_FULL_SCALE_DB);
        painter.line_segment(
            [
                egui::pos2(tick_x, rect.top()),
                egui::pos2(tick_x, rect.bottom()),
            ],
            egui::Stroke::new(1.0_f32, ui.visuals().noninteractive().fg_stroke.color),
        );
        ui.label(format!("{gr_db:+.1} dB")).on_hover_text(
            "Live gain reduction applied by the high-band compressor (excludes makeup). \
                Updates as the audio plays; peaks catch instantly, releases over ~200 ms.",
        );
    });
}

/// One scrolling history graph for the top-bar meters: an icon label, a filled line plot, and
/// the current value drawn inside the plot.
#[allow(clippy::too_many_arguments)]
fn draw_meter(
    ui: &mut egui::Ui,
    icon: &str,
    values: &std::collections::VecDeque<f32>,
    max: f32,
    color: egui::Color32,
    value_text: String,
    hover: String,
) {
    ui.label(icon);
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(72.0, 18.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 3.0, ui.visuals().extreme_bg_color);
    if values.len() >= 2 {
        let n = values.len();
        let pts: Vec<egui::Pos2> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let x = rect.left() + rect.width() * i as f32 / (n - 1) as f32;
                let y = rect.bottom() - rect.height() * (v / max.max(1e-6)).clamp(0.0, 1.0);
                egui::pos2(x, y)
            })
            .collect();
        // Soft fill under the curve, then the line itself.
        let fill = color.linear_multiply(0.25);
        for p in &pts {
            painter.line_segment(
                [*p, egui::pos2(p.x, rect.bottom())],
                egui::Stroke::new(1.0_f32, fill),
            );
        }
        painter.add(egui::Shape::line(pts, egui::Stroke::new(1.5_f32, color)));
    }
    painter.text(
        egui::pos2(rect.right() - 3.0, rect.center().y),
        egui::Align2::RIGHT_CENTER,
        value_text,
        egui::FontId::monospace(10.0),
        ui.visuals().strong_text_color(),
    );
    resp.on_hover_text(hover);
}

/// Saves bytes to disk (native, via dialog) or triggers a browser download (web).
/// Formats a song length in seconds as `M:SS`.
fn fmt_duration(secs: f64) -> String {
    let total = secs.round().max(0.0) as u64;
    format!("{}:{:02}", total / 60, total % 60)
}

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

#[cfg(test)]
mod tests {
    use super::sparse_default_order;

    // Helper: songs identified by native order; `anchored[i]` gives song i its sparse_index.
    fn keys(anchored: &[(usize, usize)], n: usize) -> Vec<(Option<usize>, usize)> {
        (0..n)
            .map(|i| {
                let sparse = anchored.iter().find(|(idx, _)| *idx == i).map(|(_, s)| *s);
                (sparse, i)
            })
            .collect()
    }

    #[test]
    fn anchored_song_holds_its_absolute_position() {
        // Song 2 anchored at index 2 stays put among the unlabeled ones.
        assert_eq!(
            sparse_default_order(&keys(&[(2, 2)], 5)),
            vec![0, 1, 2, 3, 4]
        );
        // Anchored at the front: it jumps to 0, the rest keep native order.
        assert_eq!(sparse_default_order(&keys(&[(2, 0)], 4)), vec![2, 0, 1, 3]);
        // Anchored near the end.
        assert_eq!(sparse_default_order(&keys(&[(0, 3)], 4)), vec![1, 2, 3, 0]);
    }

    #[test]
    fn two_anchored_intersperse_and_collisions_resolve() {
        // Two anchors at 1 and 3, three fillers fill 0,2,4.
        assert_eq!(
            sparse_default_order(&keys(&[(0, 1), (4, 3)], 5)),
            vec![1, 0, 2, 4, 3]
        );
        // Colliding targets (both want 2): first lands at 2, the other at the next free slot.
        assert_eq!(
            sparse_default_order(&keys(&[(0, 2), (1, 2)], 5)),
            vec![2, 3, 0, 1, 4]
        );
        // Out-of-range index parks at the end.
        assert_eq!(sparse_default_order(&keys(&[(1, 99)], 3)), vec![0, 2, 1]);
    }

    #[test]
    fn no_anchors_keeps_native_order() {
        assert_eq!(sparse_default_order(&keys(&[], 4)), vec![0, 1, 2, 3]);
    }
}

//! The user's persistent library — playlists, liked songs, play history — plus the
//! serializable bundle of everything saved to eframe storage between sessions.

pub use optime_core::{InstrumentResampleChoice, PerDeviceSettings, PopSmoothingEdge};

/// What happens when the current song ends (Spotify-style repeat cycle).
#[derive(Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum RepeatMode {
    /// Play through the queue once, then stop.
    Off,
    /// Advance and wrap around the queue forever.
    #[default]
    All,
    /// Replay the current song.
    One,
}

impl RepeatMode {
    /// The next mode in the toggle cycle (Off → All → One → Off).
    pub fn next(self) -> Self {
        match self {
            RepeatMode::Off => RepeatMode::All,
            RepeatMode::All => RepeatMode::One,
            RepeatMode::One => RepeatMode::Off,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            RepeatMode::Off => "🔁 Off",
            RepeatMode::All => "🔁 All",
            RepeatMode::One => "🔂 One",
        }
    }
}

/// How the loaded archive's song list is ordered in the library.
#[derive(Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum SortMode {
    /// The archive's native listing order.
    #[default]
    Default,
    /// Alphabetical by song name.
    Name,
    /// Shortest first, by computed playback length.
    Length,
}

impl SortMode {
    pub fn label(self) -> &'static str {
        match self {
            SortMode::Default => "Default order",
            SortMode::Name => "Name",
            SortMode::Length => "Length",
        }
    }
}

/// A song reference that survives restarts: the source archive key (a demo stem, or the file
/// name of a user-opened ROM/SDAT) plus the SSEQ id within it.
#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TrackRef {
    pub source: String,
    /// The song id within the source. Serialized as `sseq_id` (and used as the `?sseq_id=` share
    /// URL param) for storage compatibility; despite the on-disk name it is the song id for any
    /// device, not just DS SSEQ.
    #[serde(rename = "sseq_id")]
    pub song_id: u32,
    pub label: String,
}

impl TrackRef {
    /// Identity ignores the display label (which may change with SYMB availability).
    pub fn same_song(&self, other: &TrackRef) -> bool {
        self.source == other.source && self.song_id == other.song_id
    }
}

/// A user-curated, ordered list of tracks.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Playlist {
    pub name: String,
    pub tracks: Vec<TrackRef>,
}

/// Maximum entries kept in the recently-played history.
const RECENT_CAP: usize = 30;

/// Everything the user has curated.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Library {
    pub playlists: Vec<Playlist>,
    pub liked: Vec<TrackRef>,
    /// Most recent first.
    pub recent: Vec<TrackRef>,
}

impl Library {
    pub fn is_liked(&self, t: &TrackRef) -> bool {
        self.liked.iter().any(|x| x.same_song(t))
    }

    pub fn toggle_liked(&mut self, t: &TrackRef) {
        if let Some(i) = self.liked.iter().position(|x| x.same_song(t)) {
            self.liked.remove(i);
        } else {
            self.liked.push(t.clone());
        }
    }

    /// Records a play at the front of the history (deduplicated, capped).
    pub fn push_recent(&mut self, t: &TrackRef) {
        self.recent.retain(|x| !x.same_song(t));
        self.recent.insert(0, t.clone());
        self.recent.truncate(RECENT_CAP);
    }
}

/// The full app state saved to (and restored from) eframe storage.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Persisted {
    pub library: Library,
    pub shuffle: bool,
    pub repeat: RepeatMode,
    pub volume: f32,
    /// The song that was playing when the app last saved; restored (paused) on launch.
    pub last_track: Option<TrackRef>,

    /// Synth/audio settings for Nintendo DS playback.
    pub nds: PerDeviceSettings,
    /// Synth/audio settings for Game Boy Advance playback.
    pub gba: PerDeviceSettings,

    /// How the song list is sorted.
    pub sort_mode: SortMode,
    /// Whether the sort runs in descending order.
    pub sort_descending: bool,

    /// Hand-authored chord labels, keyed by source archive, kept in eframe storage as a
    /// **session-recovery working copy**.
    ///
    /// The record differs by platform: native commits `ml/annotations/*.json` in the source tree
    /// (the training data of record); web has no source tree, so "Save" there only offers a
    /// download. In both cases this stash captures unsaved edits — including non-label work like a
    /// "Bar 1 here" grid/meter change — so a refresh or restart doesn't drop them. It only ever
    /// fills in when no committed file exists (native load falls back to it on `Ok(None)`), so a
    /// hand-edited record is never overwritten. Defaulted so existing stored blobs load unchanged.
    #[serde(default)]
    pub annotations: std::collections::HashMap<String, crate::annotation::model::GameAnnotations>,
}

impl Default for Persisted {
    fn default() -> Self {
        Self {
            library: Library::default(),
            // Start with shuffle off; loop the queue forever.
            shuffle: false,
            repeat: RepeatMode::All,
            volume: 1.0,
            last_track: None,

            // Native song-list order, ascending, until the user sorts.
            sort_mode: SortMode::Default,
            sort_descending: false,
            annotations: std::collections::HashMap::new(),

            // The high-quality presets are owned by the engine (so offline tools share them); the
            // app's runtime-only piano-roll track mutes are injected per frame over these.
            nds: PerDeviceSettings::high_quality_nintendo_ds(),
            gba: PerDeviceSettings::enhanced_gba(),
        }
    }
}

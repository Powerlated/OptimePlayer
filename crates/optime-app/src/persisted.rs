//! Everything the app keeps between runs: library, playlists, per-device settings, and annotations.

pub use optime_core::{InstrumentResampleChoice, PerDeviceSettings, PopSmoothingEdge};

#[derive(Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum RepeatMode {
    Off,
    #[default]
    All,
    One,
}

impl RepeatMode {
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

#[derive(Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum SortMode {
    #[default]
    Default,
    Name,
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

#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TrackRef {
    pub source: String,
    #[serde(rename = "sseq_id")]
    pub song_id: u32,
    pub label: String,
}

impl TrackRef {
    pub fn same_song(&self, other: &TrackRef) -> bool {
        self.source == other.source && self.song_id == other.song_id
    }
}

#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Playlist {
    pub name: String,
    pub tracks: Vec<TrackRef>,
}

const RECENT_CAP: usize = 30;

#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Library {
    pub playlists: Vec<Playlist>,
    pub liked: Vec<TrackRef>,
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

    pub fn push_recent(&mut self, t: &TrackRef) {
        self.recent.retain(|x| !x.same_song(t));
        self.recent.insert(0, t.clone());
        self.recent.truncate(RECENT_CAP);
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Persisted {
    pub library: Library,
    pub shuffle: bool,
    pub repeat: RepeatMode,
    pub volume: f32,
    pub last_track: Option<TrackRef>,

    pub nds: PerDeviceSettings,
    pub gba: PerDeviceSettings,

    pub sort_mode: SortMode,
    pub sort_descending: bool,

    #[serde(default)]
    pub annotations: std::collections::HashMap<String, crate::annotation::model::GameAnnotations>,
}

impl Default for Persisted {
    fn default() -> Self {
        Self {
            library: Library::default(),
            shuffle: false,
            repeat: RepeatMode::All,
            volume: 1.0,
            last_track: None,

            sort_mode: SortMode::Default,
            sort_descending: false,
            annotations: std::collections::HashMap::new(),

            nds: PerDeviceSettings::high_quality_nintendo_ds(),
            gba: PerDeviceSettings::enhanced_gba(),
        }
    }
}

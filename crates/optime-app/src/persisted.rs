//! The user's persistent library — playlists, liked songs, play history — plus the
//! serializable bundle of everything saved to eframe storage between sessions.

pub use optime_core::{
    HighShelf, InstrumentResampleChoice, InstrumentResampleSettings, MixerResampleSettings,
    PerDeviceSettings,
};

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
    pub sseq_id: u32,
    pub label: String,
}

impl TrackRef {
    /// Identity ignores the display label (which may change with SYMB availability).
    pub fn same_song(&self, other: &TrackRef) -> bool {
        self.source == other.source && self.sseq_id == other.sseq_id
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

            nds: PerDeviceSettings {
                stereo_separation: true,
                force_stereo_separation: false,
                delay_smoothing_choice: 1,
                bass_mono: true,
                bass_mono_freq: 200.0,
                tuning_choice: 0,
                pure_tonic: 0,
                instrument_resample: InstrumentResampleSettings {
                    choice: InstrumentResampleChoice::SincOutputNyquist,
                    sinc_taps: 32,
                    psg_cutoff_hz: 15_000,
                    sampler_cutoff_hz: 15_000,
                    smooth_psg_pops: false,
                    smooth_sample_pops: false,
                },
                use_mixer: true,
                mixer_sample_rate: 32768,
                psg_crunch_compensation: true,
                mixer_resample: MixerResampleSettings {
                    choice: InstrumentResampleChoice::SincSampleNyquist,
                    sinc_taps: 32,
                    cutoff_hz: 15_000,
                },
                shelf: HighShelf {
                    enabled: true,
                    order: 2,
                    q: 0.5,
                    cutoff_hz: 12700.0,
                    gain_db: -10.0,
                },
            },
            gba: PerDeviceSettings {
                stereo_separation: true,
                force_stereo_separation: false,
                delay_smoothing_choice: 1,
                bass_mono: true,
                bass_mono_freq: 200.0,
                tuning_choice: 0,
                pure_tonic: 0,
                instrument_resample: InstrumentResampleSettings {
                    choice: InstrumentResampleChoice::SincOutputNyquist,
                    sinc_taps: 32,
                    psg_cutoff_hz: 15_000,
                    sampler_cutoff_hz: 15_000,
                    smooth_psg_pops: true,
                    smooth_sample_pops: true,
                },
                use_mixer: true,
                mixer_sample_rate: 13379,
                psg_crunch_compensation: true,
                mixer_resample: MixerResampleSettings {
                    choice: InstrumentResampleChoice::SincOutputNyquist,
                    sinc_taps: 32,
                    cutoff_hz: 15_000,
                },
                shelf: HighShelf {
                    enabled: true,
                    order: 2,
                    q: 0.5,
                    cutoff_hz: 12700.0,
                    gain_db: -20.0,
                },
            },
        }
    }
}

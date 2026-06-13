//! The user's persistent library — playlists, liked songs, play history — plus the
//! serializable bundle of everything saved to eframe storage between sessions.

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

/// Per-device resampling settings — each console keeps its own, so e.g. the DS can play
/// Crunchy while the GBA plays Authentic.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ResampleSettings {
    /// Mode index: 0=Nearest, 1=Linear, 2=Crunchy sinc, 3=Clean sinc, 4=Authentic.
    pub choice: usize,
    /// Total source-tap count for the sinc/reconstruction kernel.
    pub sinc_taps: usize,
    /// Crunchy-mode low-pass cutoff (Hz) for PSG voices.
    pub psg_cutoff_hz: u32,
    /// Crunchy-mode low-pass cutoff (Hz) for DirectSound/sampled voices.
    pub sampler_cutoff_hz: u32,
    /// Authentic-mode low-pass cutoff (Hz) on the final reconstruction.
    pub authentic_cutoff_hz: u32,
    /// Crunchy-Authentic-mode low-pass cutoff (Hz) on the final reconstruction.
    pub crunchy_authentic_cutoff_hz: u32,
    /// Crunchy-mode option: smooth out PSG on/off pops instead of preserving the clicks.
    pub smooth_psg_pops: bool,
}

impl Default for ResampleSettings {
    fn default() -> Self {
        use crate::default_settings as d;
        Self {
            choice: d::RESAMPLE_CHOICE,
            sinc_taps: d::SINC_TAPS,
            psg_cutoff_hz: d::PSG_CUTOFF_HZ,
            sampler_cutoff_hz: d::SAMPLER_CUTOFF_HZ,
            authentic_cutoff_hz: d::AUTHENTIC_CUTOFF_HZ,
            crunchy_authentic_cutoff_hz: d::CRUNCHY_AUTHENTIC_CUTOFF_HZ,
            smooth_psg_pops: d::SMOOTH_PSG_POPS,
        }
    }
}

/// Per-device master high-shelf EQ settings (one for the DS, one for the GBA), persisted.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ShelfSettings {
    /// Whether the shelf is applied.
    pub enabled: bool,
    /// Filter order (even); higher steepens the transition.
    pub order: usize,
    /// Resonance at the corner.
    pub q: f32,
    /// Corner frequency (Hz).
    pub cutoff_hz: f32,
    /// Shelf gain (dB); negative cuts the highs, positive boosts them.
    pub gain_db: f32,
}

impl Default for ShelfSettings {
    fn default() -> Self {
        use crate::default_settings as d;
        Self {
            enabled: d::SHELF_ENABLED,
            order: d::SHELF_ORDER,
            q: d::SHELF_Q,
            cutoff_hz: d::SHELF_CUTOFF_HZ,
            gain_db: d::SHELF_GAIN_DB,
        }
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

    // Synth settings.
    pub stereo_separation: bool,
    pub force_stereo_separation: bool,
    pub bass_mono: bool,
    pub bass_mono_freq: f32,
    pub tuning_choice: usize,
    pub pure_tonic: i32,
    /// Nintendo DS resampling settings.
    pub nds_resample: ResampleSettings,
    /// GBA resampling settings.
    pub gba_resample: ResampleSettings,
    /// Nintendo DS master high-shelf EQ.
    pub nds_shelf: ShelfSettings,
    /// GBA master high-shelf EQ.
    pub gba_shelf: ShelfSettings,
    /// Stereo-expander delay-change handling: 0 = immediate, 1 = hold during notes.
    pub delay_smoothing_choice: usize,
    /// How the song list is sorted.
    pub sort_mode: SortMode,
    /// Whether the sort runs in descending order.
    pub sort_descending: bool,
}

impl Default for Persisted {
    fn default() -> Self {
        use crate::default_settings as d;
        Self {
            library: Library::default(),
            shuffle: d::SHUFFLE,
            repeat: d::REPEAT,
            volume: d::VOLUME,
            last_track: None,
            stereo_separation: d::STEREO_SEPARATION,
            force_stereo_separation: d::FORCE_STEREO_SEPARATION,
            bass_mono: d::BASS_MONO,
            bass_mono_freq: d::BASS_MONO_FREQ_HZ,
            tuning_choice: d::TUNING_CHOICE,
            pure_tonic: d::PURE_TONIC,
            nds_resample: ResampleSettings::default(),
            gba_resample: ResampleSettings::default(),
            nds_shelf: ShelfSettings::default(),
            gba_shelf: ShelfSettings::default(),
            delay_smoothing_choice: d::DELAY_SMOOTHING_CHOICE,
            sort_mode: d::SORT_MODE,
            sort_descending: d::SORT_DESCENDING,
        }
    }
}

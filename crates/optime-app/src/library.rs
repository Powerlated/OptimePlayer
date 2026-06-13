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
    /// Crunchy-mode option: smooth out PSG on/off pops instead of preserving the clicks.
    pub smooth_psg_pops: bool,
}

impl Default for ResampleSettings {
    fn default() -> Self {
        Self {
            choice: 2,
            sinc_taps: 32,
            psg_cutoff_hz: optime_core::ResampleMode::CUTOFF_OFF_HZ,
            sampler_cutoff_hz: optime_core::ResampleMode::CUTOFF_OFF_HZ,
            authentic_cutoff_hz: optime_core::ResampleMode::CUTOFF_OFF_HZ,
            smooth_psg_pops: false,
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
    /// Stereo-expander delay-change handling: 0 = immediate, 1 = hold during notes.
    pub delay_smoothing_choice: usize,
}

impl Default for Persisted {
    fn default() -> Self {
        Self {
            library: Library::default(),
            shuffle: false,
            repeat: RepeatMode::All,
            volume: 1.0,
            last_track: None,
            stereo_separation: true,
            force_stereo_separation: true,
            bass_mono: true,
            bass_mono_freq: 200.0,
            tuning_choice: 0,
            pure_tonic: 0,
            nds_resample: ResampleSettings::default(),
            gba_resample: ResampleSettings::default(),
            delay_smoothing_choice: 0,
        }
    }
}

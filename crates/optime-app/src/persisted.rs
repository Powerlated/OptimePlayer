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

#[derive(Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum InstrumentResampleChoice {
    Nearest,
    Linear,
    SincOutputNyquist,
    SincSampleNyquist,
}

impl InstrumentResampleChoice {
    pub fn text(&self) -> &'static str {
        match self {
            InstrumentResampleChoice::Nearest => "Nearest neighbour",
            InstrumentResampleChoice::Linear => "Linear",
            InstrumentResampleChoice::SincOutputNyquist => "Sinc – output Nyquist (crunch)",
            InstrumentResampleChoice::SincSampleNyquist => "Sinc – sample Nyquist (clean)",
        }
    }
}

/// Per-device resampling settings — each console keeps its own, so e.g. the DS can play
/// Crunchy sinc while the GBA plays Clean sinc.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct InstrumentResampleSettings {
    /// Resampling choice enum
    pub choice: InstrumentResampleChoice,
    /// Total source-tap count for the sinc/reconstruction kernel.
    pub sinc_taps: usize,
    /// Crunchy-mode low-pass cutoff (Hz) for PSG voices.
    pub psg_cutoff_hz: u32,
    /// Crunchy-mode low-pass cutoff (Hz) for DirectSound/sampled voices.
    pub sampler_cutoff_hz: u32,
    /// Smooth out PSG on/off pops (a gain slew) instead of preserving the clicks. Applies in
    /// every resampling mode.
    pub smooth_psg_pops: bool,
    /// Smooth out sampled (DirectSound/SWAR) voice pops/clicks. Applies in every resampling mode.
    pub smooth_sample_pops: bool,
}

/// Mixer-to-output resampling settings. Reuses the same algorithm choice as the per-instrument
/// stage ([`InstrumentResampleChoice`]); the bus is a finished mix (no PSG/sampled split), so the
/// crunch mode carries a single `cutoff_hz` rather than the per-kind PSG/sampler cutoffs.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct MixerResampleSettings {
    /// Resampling choice enum (shared with the instrument stage).
    pub choice: InstrumentResampleChoice,
    /// Total source-tap count for the sinc/reconstruction kernel.
    pub sinc_taps: usize,
    /// Crunchy-mode low-pass cutoff (Hz) for the bus.
    pub cutoff_hz: u32,
}

/// Per-device master high-shelf EQ settings (one for the DS, one for the GBA), persisted.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    pub instrument_resample: InstrumentResampleSettings,
    pub mixer_resample: MixerResampleSettings,
    pub shelf: ShelfSettings,
    /// Stereo-expander delay-change handling: 0 = immediate, 1 = hold during notes.
    pub delay_smoothing_choice: usize,
    /// How the song list is sorted.
    pub sort_mode: SortMode,
    /// Whether the sort runs in descending order.
    pub sort_descending: bool,

    pub mixer_sample_rate: u32,
    /// Route sampled (non-PSG) voices through the intermediate mixer (then upsample to output).
    pub use_mixer: bool,

    pub psg_crunch_compensation: bool,
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
            stereo_separation: true,
            force_stereo_separation: false,
            bass_mono: true,
            // Crossover below which the stereo expander keeps the signal centered.
            bass_mono_freq: 200.0,
            // 0 = equal temperament, 1 = pure (Pythagorean); tonic in semitones from A.
            tuning_choice: 0,
            pure_tonic: 0,
            instrument_resample: InstrumentResampleSettings {
                choice: InstrumentResampleChoice::SincOutputNyquist,
                // Total source-tap count for the sinc/reconstruction kernel.
                sinc_taps: 32,
                // Listenable default low-pass cutoffs (Hz). The slider can still be opened all the
                // way to `ResampleMode::CUTOFF_OFF_HZ` (no extra filtering).
                psg_cutoff_hz: 15_000,
                sampler_cutoff_hz: 15_000,
                // Preserve the hardware's hard on/off edges by default (don't slew the pops).
                smooth_psg_pops: false,
                smooth_sample_pops: false,
            },
            mixer_resample: MixerResampleSettings {
                // Mixer-to-output resampling: clean reconstruction is the sane default for
                // upsampling a bus.
                choice: InstrumentResampleChoice::SincSampleNyquist,
                sinc_taps: 32,
                cutoff_hz: 15_000,
            },
            shelf: ShelfSettings {
                // Off by default — a transparent pass until the user dials in a shelf.
                enabled: false,
                // Filter order (even); the cascade has `order / 2` biquad sections.
                order: 2,
                q: 0.707,
                cutoff_hz: 4000.0,
                gain_db: 0.0,
            },
            // Stereo-expander delay-change handling: 0 = immediate, 1 = hold during notes.
            delay_smoothing_choice: 0,
            // Native song-list order, ascending, until the user sorts.
            sort_mode: SortMode::Default,
            sort_descending: false,
            // Sample rate for the intermediate mixing step; off by default.
            mixer_sample_rate: 48000,
            use_mixer: false,
            psg_crunch_compensation: false,
        }
    }
}

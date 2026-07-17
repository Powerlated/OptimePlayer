//! The emulated sound devices.
//!
//! Each console lives in its own folder and follows the same shape, so the data flow is the
//! same everywhere:
//!
//! ```text
//! ROM bytes ──► devices::<console>::parse  ──► SoundData (archive: songs + instruments)
//! SoundData ──► devices::<console>::Player ──► SynthEvent stream ──► SynthController (voices/mixing)
//! ```
//!
//! - [`nds`] — SDAT archives, the SSEQ sequencer, and the DS ADSR/LFO hardware model.
//! - [`gba`] — GBA ROMs running the MP2K ("Sappy") engine from `pret/pokeemerald`.
//! - [`dse`] — Procyon Studios' DSE engine (SMDL/SWDL), used by PMD: Explorers of Sky; from
//!   `pret/pmd-sky`. A full [`SoundData`]/[`DevicePlayer`] backend: SMDL sequencer, the volume
//!   envelope, and the ROM-exact note→frequency and square-law volume of the voice-update code.
//!
//! The standardized messages are defined in [`messages`]; the controller never reaches into a
//! device, and a device never touches voices directly.

pub mod dse;
pub mod gba;
pub mod nds;

pub use crate::synth_controller::messages::{SynthEvent, TickFeedback, VoiceId, VoicePitch};

use crate::PerDeviceSettings;

/// DC-offset statistic for one decoded PCM sample, for the app's "Stats for Nerds" view.
///
/// Real GB/GBA output is AC-coupled (a DC-blocking high-pass on the way out), so a sample's
/// constant offset is filtered away on hardware. The engine removes it at decode time; this
/// records how much had to be shifted.
#[derive(Debug, Clone)]
pub struct WaveformDcStat {
    /// Human label for the sample (GBA: the wave's ROM address).
    pub label: String,
    /// The DC offset that was removed, as a fraction of full scale (`|mean|`, 0.0..=1.0).
    pub dc_shift: f32,
    /// Number of PCM samples.
    pub length: usize,
    /// Playback sample rate in Hz.
    pub sample_rate: f64,
}

/// A loaded, parsed sound archive for some device — everything needed to list and start songs.
///
/// Each console backend implements this trait; use [`load_all`] to parse bytes into archives.
/// The trait object is `Send + Sync` so it can be shared across threads via `Arc<dyn SoundData>`.
pub trait SoundData: Send + Sync {
    /// Ids of the playable songs, in listing order. Only songs that will actually start when
    /// selected are listed: GBA song tables are full of empty placeholders and the odd
    /// malformed entry, which are filtered with the same header validation
    /// [`Self::make_player`] performs.
    fn song_ids(&self) -> Vec<u32>;

    /// Creates a player for song `id`, or `None` if the song is missing / malformed.
    fn make_player(&self, id: u32) -> Option<Box<dyn DevicePlayer>>;

    /// Display name for song `id`, if the archive carries one. Defaults to `None` when the
    /// archive carries no embedded names (the app may supply curated titles separately).
    fn song_name(&self, _id: u32) -> Option<String> {
        None
    }

    /// DC-offset stats for every PCM sample reachable from song `id`, sorted by the amount of DC
    /// shift (most shifted first). Returns an empty list for archives that don't analyse samples.
    fn waveform_dc_stats(&self, _id: u32) -> Vec<WaveformDcStat> {
        Vec::new()
    }

    /// Returns this archive as `&dyn Any`, enabling downcasts to the concrete type in tests,
    /// examples, and app code that needs console-specific fields (e.g. `GbaRom::game_code()`).
    fn as_any(&self) -> &dyn std::any::Any;

    /// Playback length of song `id` in seconds, or `None` if the song is missing/malformed.
    ///
    /// Defined for the library's length column and length sort:
    /// - **Repeating** songs (those that loop): the intro plus two passes of the repeating
    ///   section — i.e. up to the second loop point.
    /// - **Non-repeating** songs: the full play-through, up to the end.
    ///
    /// Computed by running the device sequencer headlessly (no audio) and timing how many
    /// fixed-rate device ticks elapse before the second [`SynthEvent::Looped`] or the
    /// [`SynthEvent::Ended`]. A song that neither loops nor ends is capped at 15 minutes.
    fn song_length_seconds(&self, id: u32) -> Option<f64> {
        let mut player = self.make_player(id)?;
        let tick_rate = player.tick_rate();
        let config = PerDeviceSettings::neutral();
        let mut feedback = TickFeedback::default();
        let mut events = Vec::new();
        let max_ticks = (tick_rate * 15.0 * 60.0) as u64;
        let mut ticks: u64 = 0;
        let mut loops = 0u32;
        let mut end_ticks = None;
        while ticks < max_ticks {
            events.clear();
            player.tick(&mut feedback, &config, &mut events);
            ticks += 1;
            for ev in &events {
                match ev {
                    SynthEvent::Looped => {
                        loops += 1;
                        if loops >= 2 {
                            end_ticks = Some(ticks);
                        }
                    }
                    SynthEvent::Ended => end_ticks = Some(ticks),
                    _ => {}
                }
            }
            if end_ticks.is_some() {
                break;
            }
        }
        Some(end_ticks.unwrap_or(ticks) as f64 / tick_rate)
    }
}

/// A running device player: the sequencer + envelope model of one console, generating
/// [`SynthEvent`]s for the [`SynthController`](crate::SynthController).
pub trait DevicePlayer: Send {
    /// The device master-clock rate in Hz (cycles per second).
    fn clock_rate(&self) -> f64;

    /// Device clock cycles between ticks (DS: sequencer timer period; GBA: one VBlank frame).
    fn cycles_per_tick(&self) -> f64;

    /// Sequencer steps executed so far — the note-timeline position for visualizers
    /// (DS: SSEQ ticks; GBA: MP2K tempo steps).
    fn steps_elapsed(&self) -> u32;

    /// The current *sequencer* step rate in steps per second (tempo-dependent), used by
    /// visualizers to convert note timestamps to wall time.
    fn step_rate(&self) -> f64;

    /// Sequencer steps per quarter-note beat (DS SSEQ: 48 ticks; GBA MP2K: 24 steps). Lets a
    /// visualizer convert the tempo-dependent [`Self::step_rate`] into a musical BPM:
    /// `bpm = step_rate * 60 / steps_per_beat`.
    fn steps_per_beat(&self) -> f64;

    /// Advances the device by one tick, draining `feedback` and appending events to `events`.
    fn tick(
        &mut self,
        feedback: &mut TickFeedback,
        config: &PerDeviceSettings,
        events: &mut Vec<SynthEvent>,
    );

    /// Device ticks per second.
    fn tick_rate(&self) -> f64 {
        self.clock_rate() / self.cycles_per_tick()
    }

    /// Downcast hook for driving a console-specific player that a caller *owns the far end of*.
    ///
    /// Same rule as [`SoundData::as_any`]: console-specific operations stay off this trait, and a
    /// caller that needs one downcasts. The GBA's parameter-driven player is the case that wants it
    /// — the [`SynthController`](crate::SynthController) owns the `Box<dyn DevicePlayer>`, but the
    /// VST3 plugin still has to push each block's notes and parameters into it.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Parses every sound archive found in `bytes` (`.nds`/`.sdat` containers, a DSE ROM, or a
/// GBA ROM). DSE is probed before GBA: a PMD `.nds` has no SDAT, so the SDAT scan comes up
/// empty and the `swdl`/`smdl` scan identifies it.
pub fn load_all(bytes: &[u8]) -> Vec<Box<dyn SoundData>> {
    let sdats = nds::Sdat::load_all(bytes);
    if !sdats.is_empty() {
        return sdats
            .into_iter()
            .map(|sdat| -> Box<dyn SoundData> { Box::new(sdat) })
            .collect();
    }
    if let Some(dse) = dse::DseSoundData::load_all(bytes) {
        return vec![Box::new(dse)];
    }
    match gba::GbaRom::parse(bytes) {
        Some(rom) => vec![Box::new(rom)],
        None => Vec::new(),
    }
}

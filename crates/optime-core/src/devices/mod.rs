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
//! - [`nintendo_ds`] — SDAT archives, the SSEQ sequencer, and the DS ADSR/LFO hardware model.
//! - [`gba`] — GBA ROMs running the MP2K ("Sappy") engine from `pret/pokeemerald`.
//!
//! The standardized messages are defined in [`messages`]; the controller never reaches into a
//! device, and a device never touches voices directly.

pub mod gba;
pub mod nintendo_ds;

pub use crate::synth_controller::messages::{SynthEvent, TickFeedback, VoiceId, VoicePitch};

use crate::synth_controller::SynthConfig;

/// A loaded, parsed sound archive for some device — everything needed to list and start songs.
pub enum SoundData {
    /// A Nintendo DS SDAT sound archive.
    NintendoDs(Box<nintendo_ds::Sdat>),
    /// A GBA ROM with a located MP2K song table.
    Gba(gba::GbaRom),
}

impl SoundData {
    /// Parses every sound archive found in `bytes` (`.nds`/`.sdat` containers, or a GBA ROM).
    pub fn load_all(bytes: &[u8]) -> Vec<SoundData> {
        let sdats = nintendo_ds::Sdat::load_all(bytes);
        if !sdats.is_empty() {
            return sdats
                .into_iter()
                .map(|sdat| SoundData::NintendoDs(Box::new(sdat)))
                .collect();
        }
        match gba::GbaRom::parse(bytes) {
            Some(rom) => vec![SoundData::Gba(rom)],
            None => Vec::new(),
        }
    }

    /// Ids of the playable songs, in listing order. Only songs that will actually start when
    /// selected are listed: GBA song tables are full of empty placeholders and the odd
    /// malformed entry, which are filtered with the same header validation
    /// [`Self::make_player`] performs.
    pub fn song_ids(&self) -> Vec<u32> {
        match self {
            SoundData::NintendoDs(sdat) => sdat.sseq_list.clone(),
            SoundData::Gba(rom) => (0..rom.song_count() as u32)
                .filter(|&id| rom.song_header(id).is_some())
                .collect(),
        }
    }

    /// Display name for song `id`, if the archive carries one. GBA ROMs have no embedded song
    /// names (the table is numbered); the app supplies curated titles keyed by
    /// [`Self::gba_game_code`].
    pub fn song_name(&self, id: u32) -> Option<String> {
        match self {
            SoundData::NintendoDs(sdat) => sdat.sseq_id_to_name.get(&id).cloned(),
            SoundData::Gba(_) => None,
        }
    }

    /// The GBA ROM's 4-character game code (header offset 0xAC), e.g. `"BPEE"` for Pokémon Emerald.
    /// `None` for DS archives. The app uses it to pick curated song-name tables.
    pub fn gba_game_code(&self) -> Option<String> {
        match self {
            SoundData::Gba(rom) => rom.game_code(),
            SoundData::NintendoDs(_) => None,
        }
    }

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
    pub fn song_length_seconds(&self, id: u32) -> Option<f64> {
        let mut player = self.make_player(id)?;
        let tick_rate = player.tick_rate();
        let config = SynthConfig::default();
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

    /// Creates a player for song `id`.
    pub fn make_player(&self, id: u32) -> Option<DevicePlayer> {
        match self {
            SoundData::NintendoDs(sdat) => Some(DevicePlayer::NintendoDs(Box::new(
                nintendo_ds::NdsPlayer::new(sdat, id)?,
            ))),
            SoundData::Gba(rom) => Some(DevicePlayer::Gba(Box::new(gba::GbaPlayer::new(rom, id)?))),
        }
    }
}

/// A running device player: the sequencer + envelope model of one console, generating
/// [`SynthEvent`]s for the [`SynthController`](crate::SynthController).
pub enum DevicePlayer {
    NintendoDs(Box<nintendo_ds::NdsPlayer>),
    Gba(Box<gba::GbaPlayer>),
}

impl DevicePlayer {
    /// The device master-clock rate in Hz (cycles per second).
    pub fn clock_rate(&self) -> f64 {
        match self {
            DevicePlayer::NintendoDs(_) => crate::DS_CLOCK_RATE as f64,
            DevicePlayer::Gba(_) => gba::GBA_CLOCK_RATE as f64,
        }
    }

    /// Device clock cycles between ticks (DS: sequencer timer period; GBA: one VBlank frame).
    pub fn cycles_per_tick(&self) -> f64 {
        match self {
            DevicePlayer::NintendoDs(_) => crate::CYCLES_PER_TICK as f64,
            DevicePlayer::Gba(_) => gba::CYCLES_PER_FRAME as f64,
        }
    }

    /// Device ticks per second.
    pub fn tick_rate(&self) -> f64 {
        self.clock_rate() / self.cycles_per_tick()
    }

    /// Sequencer steps executed so far — the note-timeline position for visualizers
    /// (DS: SSEQ ticks; GBA: MP2K tempo steps).
    pub fn steps_elapsed(&self) -> u32 {
        match self {
            DevicePlayer::NintendoDs(p) => p.steps_elapsed(),
            DevicePlayer::Gba(p) => p.steps_elapsed(),
        }
    }

    /// The current *sequencer* step rate in steps per second (tempo-dependent), used by
    /// visualizers to convert note timestamps to wall time.
    pub fn step_rate(&self) -> f64 {
        match self {
            DevicePlayer::NintendoDs(p) => p.step_rate(),
            DevicePlayer::Gba(p) => p.step_rate(),
        }
    }

    /// Advances the device by one tick, draining `feedback` and appending events to `events`.
    pub fn tick(
        &mut self,
        feedback: &mut TickFeedback,
        config: &SynthConfig,
        events: &mut Vec<SynthEvent>,
    ) {
        match self {
            DevicePlayer::NintendoDs(p) => p.tick(feedback, config, events),
            DevicePlayer::Gba(p) => p.tick(feedback, config, events),
        }
    }
}

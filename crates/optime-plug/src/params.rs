//! The VST3 parameter surface: MP2K's engine registers, as host-automatable controls.
//!
//! The split here is not arbitrary — it is `m4a_1::control_command`'s own. MP2K sorts its track
//! commands into three groups, and each maps to a different place:
//!
//! - **Flow control** (`GOTO`, `PATT`, `PEND`, `REPT`, `MEMACC`, `FINE`) — the sequencer. A DAW's
//!   clip replaces it wholesale, so none of it appears here.
//! - **Per-track control** (`0xBA`–`0xC8`) — [`TrackParams`]'s first block.
//! - **Extended commands** (`XCMD` `0xCD` → `ply_x*`) — [`TrackParams`]'s tone-override block.
//!
//! What is deliberately *absent* matters as much as what is here. `volMR`, `volML`, `keyM`, `pitM`,
//! `modM`, `lfoSpeedC`, `pitX`, `volX` and friends are **derived**: `TrkVolPitSet` and `lfo_step`
//! recompute them from the parameters below on every frame. A control for one of those would be a
//! knob the engine overwrites ~60 times a second — visible, automatable, and inert. Likewise the
//! ROM pointers (`tone.wav`, the voicegroup base) can't be normalized floats, so they are persisted
//! state instead.

use std::sync::{Arc, RwLock};

use nice_plug::prelude::*;
use nice_plug_egui::EguiState;
use optime_core::PerDeviceSettings;
use optime_core::devices::gba::param_player::TrackParams as EngineTrackParams;

/// How many MP2K tracks one instance carries. The engine's own maximum, and one per MIDI channel.
pub const TRACKS: usize = 16;

/// `0xC5` MODT — which target the track's LFO modulates.
#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModType {
    /// Pitch (MP2K's `modT` 0).
    #[id = "vibrato"]
    Vibrato,
    /// Volume (`modT` 1).
    #[id = "tremolo"]
    Tremolo,
    /// Pan (`modT` 2).
    #[id = "autopan"]
    AutoPan,
}

/// The DSP colouring applied to the final mix.
#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    /// The raw m4a signal chain: 8-bit crushed mixer at 13379 Hz, no widening, no EQ.
    #[id = "original"]
    Original,
    /// The polished chain: clean sinc voices, stereo separation, high-shelf de-harsher.
    #[id = "enhanced"]
    Enhanced,
}

impl Preset {
    /// Resolves to the engine settings. These are the app's own presets — shared, not restated, so
    /// the plugin and the player can't drift apart.
    pub fn settings(self) -> PerDeviceSettings {
        match self {
            Preset::Original => PerDeviceSettings::original_gba(),
            Preset::Enhanced => PerDeviceSettings::enhanced_gba(),
        }
    }
}

/// One track's controls. Instantiated 16×, which `#[nested(array, …)]` suffixes into `vol_1`…
/// `vol_16` and groups as "Track 1"…"Track 16".
#[derive(Params)]
pub struct TrackParams {
    // --- Per-track control commands (`0xBA`–`0xC8`) ---
    /// `0xBD` VOICE.
    #[id = "prog"]
    pub prog: IntParam,
    /// `0xBE` VOL.
    #[id = "vol"]
    pub vol: IntParam,
    /// `0xBF` PAN. MP2K centres this at `0x40`, so the stored value is signed.
    #[id = "pan"]
    pub pan: IntParam,
    /// `0xC0` BEND.
    #[id = "bend"]
    pub bend: IntParam,
    /// `0xC1` BENDR.
    #[id = "bendr"]
    pub bend_range: IntParam,
    /// `0xC4` MOD.
    #[id = "mod"]
    pub mod_: IntParam,
    /// `0xC2` LFOS.
    #[id = "lfos"]
    pub lfo_speed: IntParam,
    /// `0xC3` LFODL.
    #[id = "lfodl"]
    pub lfo_delay: IntParam,
    /// `0xC5` MODT.
    #[id = "modt"]
    pub mod_type: EnumParam<ModType>,
    /// `0xC8` TUNE.
    #[id = "tune"]
    pub tune: IntParam,
    /// `0xBC` KEYSH.
    #[id = "keysh"]
    pub key_shift: IntParam,
    /// `0xBA` PRIO.
    #[id = "prio"]
    pub priority: IntParam,

    // --- Extended commands (`XCMD` `0xCD`) ---
    /// Whether the tone overrides below apply, or the voicegroup's record is used verbatim.
    #[id = "toneovr"]
    pub tone_override: BoolParam,
    /// `ply_xtype`.
    #[id = "type"]
    pub kind: IntParam,
    /// `ply_xatta`.
    #[id = "atk"]
    pub attack: IntParam,
    /// `ply_xdeca`.
    #[id = "dec"]
    pub decay: IntParam,
    /// `ply_xsust`.
    #[id = "sus"]
    pub sustain: IntParam,
    /// `ply_xrele`.
    #[id = "rel"]
    pub release: IntParam,
    /// `ply_xleng`.
    #[id = "leng"]
    pub length: IntParam,
    /// `ply_xswee`.
    #[id = "swee"]
    pub pan_sweep: IntParam,
    /// `ply_xiecv`.
    #[id = "iecv"]
    pub echo_volume: IntParam,
    /// `ply_xiecl`.
    #[id = "iecl"]
    pub echo_length: IntParam,
}

/// `IntParam::new` with a linear range, the shape every MP2K register wants (they are raw byte
/// registers, so there is nothing to skew).
fn int(name: &str, default: i32, min: i32, max: i32) -> IntParam {
    IntParam::new(name, default, IntRange::Linear { min, max })
}

impl Default for TrackParams {
    fn default() -> Self {
        // Take the defaults from the engine's own track reset, so an untouched plugin track starts
        // exactly where a freshly started song track does rather than at some plugin-invented zero.
        let d = EngineTrackParams::default();
        TrackParams {
            prog: int("Program", d.prog.into(), 0, 127),
            vol: int("Volume", d.vol.into(), 0, 127),
            pan: int("Pan", d.pan.into(), -64, 63),
            bend: int("Bend", d.bend.into(), -64, 63),
            bend_range: int("Bend Range", d.bend_range.into(), 0, 127),
            mod_: int("Mod Depth", d.mod_.into(), 0, 127),
            lfo_speed: int("LFO Speed", d.lfo_speed.into(), 0, 127),
            lfo_delay: int("LFO Delay", d.lfo_delay.into(), 0, 127),
            mod_type: EnumParam::new("Mod Type", ModType::Vibrato),
            tune: int("Tune", d.tune.into(), -64, 63),
            key_shift: int("Key Shift", d.key_shift.into(), -128, 127),
            priority: int("Priority", d.priority.into(), 0, 127),

            tone_override: BoolParam::new("Tone Override", d.tone_override),
            kind: int("Tone Type", d.kind.into(), 0, 255),
            attack: int("Attack", d.attack.into(), 0, 255),
            decay: int("Decay", d.decay.into(), 0, 255),
            sustain: int("Sustain", d.sustain.into(), 0, 255),
            release: int("Release", d.release.into(), 0, 255),
            length: int("Length", d.length.into(), 0, 255),
            pan_sweep: int("Sweep", d.pan_sweep.into(), 0, 255),
            echo_volume: int("Echo Volume", d.echo_volume.into(), 0, 255),
            echo_length: int("Echo Length", d.echo_length.into(), 0, 255),
        }
    }
}

impl TrackParams {
    /// Reads the current values into the engine's parameter struct.
    pub fn to_engine(&self) -> EngineTrackParams {
        EngineTrackParams {
            prog: self.prog.value() as u8,
            vol: self.vol.value() as u8,
            pan: self.pan.value() as i8,
            bend: self.bend.value() as i8,
            bend_range: self.bend_range.value() as u8,
            mod_: self.mod_.value() as u8,
            lfo_speed: self.lfo_speed.value() as u8,
            lfo_delay: self.lfo_delay.value() as u8,
            mod_type: match self.mod_type.value() {
                ModType::Vibrato => 0,
                ModType::Tremolo => 1,
                ModType::AutoPan => 2,
            },
            tune: self.tune.value() as i8,
            key_shift: self.key_shift.value() as i8,
            priority: self.priority.value() as u8,

            tone_override: self.tone_override.value(),
            kind: self.kind.value() as u8,
            attack: self.attack.value() as u8,
            decay: self.decay.value() as u8,
            sustain: self.sustain.value() as u8,
            release: self.release.value() as u8,
            length: self.length.value() as u8,
            pan_sweep: self.pan_sweep.value() as u8,
            echo_volume: self.echo_volume.value() as u8,
            echo_length: self.echo_length.value() as u8,
        }
    }
}

/// The whole plugin's parameters plus its persisted (non-automatable) state.
#[derive(Params)]
pub struct OptimePlugParams {
    /// Editor size/open state, persisted alongside the parameters.
    #[persist = "editor-state"]
    pub editor_state: Arc<EguiState>,

    #[nested(array, group = "Track")]
    pub tracks: [TrackParams; TRACKS],

    /// `SOUND_MODE_MASVOL`.
    #[id = "masvol"]
    pub master_volume: IntParam,
    /// `SOUND_MODE_REVERB`.
    #[id = "reverb"]
    pub reverb: IntParam,
    /// `SOUND_MODE_MAXCHN`.
    ///
    /// A fidelity control, not a performance one. `alloc_direct_sound` steals by priority *within*
    /// this count, so it decides which notes get dropped. Games configure 5–8; the default of 8
    /// matches, and raising it keeps notes real hardware would have stolen.
    #[id = "maxchan"]
    pub max_chans: IntParam,
    /// Which DSP chain colours the output.
    #[id = "preset"]
    pub preset: EnumParam<Preset>,

    /// Path to the GBA ROM (or `.gbaaudio` extract) the samples come from.
    ///
    /// Persisted state rather than a parameter: it is not a number a host can automate, and the ROM
    /// is the user's own file, which this deliberately does not embed in the project.
    #[persist = "rom-path"]
    pub rom_path: RwLock<String>,

    /// Which song in the ROM's table the voicegroup comes from, and what rip capture plays.
    ///
    /// State, not a parameter: the valid range depends on the loaded ROM, and VST3 parameter ranges
    /// are fixed at instantiation.
    #[persist = "song-id"]
    pub song_id: RwLock<u32>,

    /// ROM *offset* of the voicegroup `prog` indexes into. Also state — it is a pointer.
    ///
    /// Normally taken from the selected song's header rather than typed.
    #[persist = "voicegroup"]
    pub voicegroup: RwLock<u32>,
}

impl Default for OptimePlugParams {
    fn default() -> Self {
        OptimePlugParams {
            editor_state: EguiState::from_size(720, 520),
            tracks: std::array::from_fn(|_| TrackParams::default()),
            master_volume: int("Master Volume", 12, 0, 15),
            reverb: int("Reverb", 0, 0, 127),
            max_chans: int("Max Channels", 8, 1, 12),
            preset: EnumParam::new("Preset", Preset::Enhanced),
            rom_path: RwLock::new(String::new()),
            song_id: RwLock::new(0),
            voicegroup: RwLock::new(0),
        }
    }
}

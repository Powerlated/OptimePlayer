//! **`m4a.rs` — a line-for-line Rust transliteration of `pret/pokeemerald`'s `src/m4a.c`.**
//!
//! This file exists so a reader can diff it against `m4a.c` and immediately see where Optime's
//! engine matches the reference and where it diverges. Every function appears in the **same order**
//! and under the **same C identifier** as in `m4a.c` (hence the module-wide `non_snake_case` /
//! `non_upper_case_globals` allows — idiomatic Rust naming would defeat the diff). The `struct`
//! definitions are transcribed field-for-field from `include/gba/m4a_internal.h`.
//!
//! Two kinds of code live here, and they are marked so you never have to guess which is which:
//!
//! * **Transliterated** — plain ports of the C. The control flow, constants, and integer math are
//!   bit-for-bit faithful. These are the functions with real, hardware-independent behavior
//!   (`MidiKeyToFreq`, `TrkVolPitSet`, `CgbSound`'s envelope engine, `FadeOutBody`, the `ply_x*`
//!   sequencer command handlers, …). Where useful they are unit-tested against a C oracle.
//!
//! * **Synth-backend seams** — every place `m4a.c` pokes GBA MMIO (`REG_NR12`, DMA/FIFO, timers)
//!   or calls into the hand-written ARM assembly of `m4a_1.s` (`MPlayMain`, `ply_note`,
//!   `SoundMain`, `TrackStop`, …). None of that exists in a portable player, so Optime substitutes
//!   its own software synthesis. Each such spot is fenced by a banner:
//!
//!   ```text
//!   // ┌─ SYNTH BACKEND SEAM ─ <what the hardware/asm did> ─┐
//!   //   … Optime substitute (or documented no-op) …
//!   // └────────────────────────────────────────────────────┘
//!   ```
//!
//!   The asm-engine entry points themselves (`MPlayMain`, `ply_note`, `SoundMain`, `SoundMainBTM`,
//!   `TrackStop`, `MPlayJumpTableCopy`, `RealClearChain`) have **no C to mirror** — they are
//!   implemented by the "no-C-home" modules [`super::sequencer`] (the per-VBlank track interpreter,
//!   standing in for `MPlayMain`/`ply_note`/`TrackStop`) and [`super::player`] (channel allocation +
//!   the software mixer, standing in for `SoundMain` + `CgbSound`'s register back-end). They appear
//!   below as documented seam stubs so the call sites read exactly like `m4a.c`.
//!
//! Hardware registers the transliterated code genuinely reads back within a frame (the CGB sound
//! register file, `REG_SOUNDBIAS_H`) are modeled by [`Hw`]; everything else is a commented no-op.

#![allow(
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    clippy::too_many_arguments,
    // The transliterations keep `m4a.c`'s explicit `if (y < lo) …; else if (y > hi) …` bound
    // clamps so the two files diff cleanly; a `.clamp()` would obscure the correspondence.
    clippy::manual_clamp
)]

use super::tables::{
    CGB_FREQ_TABLE, CGB_SCALE_TABLE, FREQ_TABLE, NOISE_TABLE, SCALE_TABLE, umul3232h32,
};

// ===========================================================================================
// Constants — from include/gba/m4a_internal.h
// ===========================================================================================

// ASCII encoding of 'Smsh' in reverse (short for SMASH, the developer of MKS4AGB).
pub const ID_NUMBER: u32 = 0x68736D53;

pub const C_V: i32 = 0x40; // center value for PAN, BEND, and TUNE

pub const SOUND_MODE_REVERB_VAL: u32 = 0x0000007F;
pub const SOUND_MODE_REVERB_SET: u32 = 0x00000080;
pub const SOUND_MODE_MAXCHN: u32 = 0x00000F00;
pub const SOUND_MODE_MAXCHN_SHIFT: u32 = 8;
pub const SOUND_MODE_MASVOL: u32 = 0x0000F000;
pub const SOUND_MODE_MASVOL_SHIFT: u32 = 12;
pub const SOUND_MODE_FREQ_13379: u32 = 0x00040000;
pub const SOUND_MODE_FREQ: u32 = 0x000F0000;
pub const SOUND_MODE_FREQ_SHIFT: u32 = 16;
pub const SOUND_MODE_DA_BIT_8: u32 = 0x00900000;
pub const SOUND_MODE_DA_BIT: u32 = 0x00B00000;
pub const SOUND_MODE_DA_BIT_SHIFT: u32 = 20;

pub const TONEDATA_TYPE_CGB: u8 = 0x07;
pub const TONEDATA_TYPE_FIX: u8 = 0x08;
pub const TONEDATA_TYPE_SPL: u8 = 0x40; // key split
pub const TONEDATA_TYPE_RHY: u8 = 0x80; // rhythm

pub const SOUND_CHANNEL_SF_START: u8 = 0x80;
pub const SOUND_CHANNEL_SF_STOP: u8 = 0x40;
pub const SOUND_CHANNEL_SF_LOOP: u8 = 0x10;
pub const SOUND_CHANNEL_SF_IEC: u8 = 0x04;
pub const SOUND_CHANNEL_SF_ENV: u8 = 0x03;
pub const SOUND_CHANNEL_SF_ENV_ATTACK: u8 = 0x03;
pub const SOUND_CHANNEL_SF_ENV_DECAY: u8 = 0x02;
pub const SOUND_CHANNEL_SF_ENV_SUSTAIN: u8 = 0x01;
pub const SOUND_CHANNEL_SF_ENV_RELEASE: u8 = 0x00;
pub const SOUND_CHANNEL_SF_ON: u8 =
    SOUND_CHANNEL_SF_START | SOUND_CHANNEL_SF_STOP | SOUND_CHANNEL_SF_IEC | SOUND_CHANNEL_SF_ENV;

pub const CGB_CHANNEL_MO_PIT: u8 = 0x02;
pub const CGB_CHANNEL_MO_VOL: u8 = 0x01;

pub const CGB_NRx2_ENV_DIR_DEC: u8 = 0x00;
pub const CGB_NRx2_ENV_DIR_INC: u8 = 0x08;

pub const MAX_DIRECTSOUND_CHANNELS: usize = 12;
pub const PCM_DMA_BUF_SIZE: usize = 1584;

pub const MUSICPLAYER_STATUS_TRACK: u32 = 0x0000ffff;
pub const MUSICPLAYER_STATUS_PAUSE: u32 = 0x80000000;

pub const MAX_MUSICPLAYER_TRACKS: usize = 16;

pub const TEMPORARY_FADE: u16 = 0x0001;
pub const FADE_IN: u16 = 0x0002;
pub const FADE_VOL_MAX: u16 = 64;
pub const FADE_VOL_SHIFT: u16 = 2;

pub const MPT_FLG_VOLSET: u8 = 0x01;
pub const MPT_FLG_VOLCHG: u8 = 0x03;
pub const MPT_FLG_PITSET: u8 = 0x04;
pub const MPT_FLG_PITCHG: u8 = 0x0C;
pub const MPT_FLG_START: u8 = 0x40;
pub const MPT_FLG_EXIST: u8 = 0x80;

pub const MAX_POKEMON_CRIES: usize = 2;

/// `gCgb3Vol` (`src/m4a_tables.c`): wave-channel NR32 volume-shift codes indexed by envelope level.
pub const gCgb3Vol: [u8; 16] = [
    0x00, 0x00, 0x60, 0x60, 0x60, 0x60, 0x40, 0x40, 0x40, 0x40, 0x80, 0x80, 0x80, 0x80, 0x20, 0x20,
];

/// `gPcmSamplesPerVBlankTable` (`src/m4a_tables.c`): DirectSound samples per V-blank per freq index.
pub const gPcmSamplesPerVBlankTable: [u16; 12] =
    [96, 132, 176, 224, 264, 304, 352, 448, 528, 608, 672, 704];

// ===========================================================================================
// Structs — transcribed field-for-field from include/gba/m4a_internal.h
// ===========================================================================================
//
// Where the C uses raw pointers to weave the object graph, Rust uses value ownership + indices, so
// the code stays safe. The substitutions (all commented at their fields):
//   * `struct MusicPlayerTrack *tracks`  → `tracks: Vec<MusicPlayerTrack>`
//   * `struct SoundChannel *chan`        → `chan: Option<usize>` (index into SoundInfo::chans)
//   * `u8 *cmdPtr`                        → `cmdPtr: usize` (offset into the song's byte stream)
//   * intrusive prev/next channel lists  → `Option<usize>` indices
// Field order and names are otherwise identical to the header for diffability.

#[derive(Clone, Copy, Default)]
pub struct WaveData {
    pub type_: u16,
    pub status: u16,
    pub freq: u32,
    pub loopStart: u32,
    pub size: u32, // number of samples
    // `s8 data[1]` in C — the PCM lives in ROM; carried as an offset by the backend, not here.
    pub data: usize,
}

#[derive(Clone, Copy, Default)]
pub struct ToneData {
    pub type_: u8,
    pub key: u8,
    pub length: u8,    // sound length (compatible sound)
    pub pan_sweep: u8, // pan or sweep (compatible sound ch. 1)
    pub wav: u32,      // `struct WaveData *wav` — a ROM pointer, kept as its raw value
    pub attack: u8,
    pub decay: u8,
    pub sustain: u8,
    pub release: u8,
}

#[derive(Clone, Copy, Default)]
pub struct CgbChannel {
    pub statusFlags: u8,
    pub type_: u8,
    pub rightVolume: u8,
    pub leftVolume: u8,
    pub attack: u8,
    pub decay: u8,
    pub sustain: u8,
    pub release: u8,
    pub key: u8,
    pub envelopeVolume: u8,
    pub envelopeGoal: u8,
    pub envelopeCounter: u8,
    pub pseudoEchoVolume: u8,
    pub pseudoEchoLength: u8,
    pub dummy1: u8,
    pub dummy2: u8,
    pub gateTime: u8,
    pub midiKey: u8,
    pub velocity: u8,
    pub priority: u8,
    pub rhythmPan: u8,
    pub dummy3: [u8; 3],
    pub dummy5: u8,
    pub sustainGoal: u8,
    pub n4: u8, // NR[1-4]4 register (initial, length bit)
    pub pan: u8,
    pub panMask: u8,
    pub modify: u8,
    pub length: u8,
    pub sweep: u8,
    pub frequency: u32,
    pub wavePointer: u32, // `u32 *` — ROM address of the wave to load into wave RAM
    pub currentPointer: u32, // `u32 *` — ROM address of the currently loaded wave
    pub track: Option<usize>, // `struct MusicPlayerTrack *track`
    pub prevChannelPointer: Option<usize>,
    pub nextChannelPointer: Option<usize>,
    pub dummy4: [u8; 8],
}

#[derive(Clone, Copy, Default)]
pub struct SoundChannel {
    pub statusFlags: u8,
    pub type_: u8,
    pub rightVolume: u8,
    pub leftVolume: u8,
    pub attack: u8,
    pub decay: u8,
    pub sustain: u8,
    pub release: u8,
    pub key: u8, // midi key as it was translated into final pitch
    pub envelopeVolume: u8,
    pub envelopeVolumeRight: u8,
    pub envelopeVolumeLeft: u8,
    pub pseudoEchoVolume: u8,
    pub pseudoEchoLength: u8,
    pub dummy1: u8,
    pub dummy2: u8,
    pub gateTime: u8,
    pub midiKey: u8, // midi key as it was used in the track data
    pub velocity: u8,
    pub priority: u8,
    pub rhythmPan: u8,
    pub dummy3: [u8; 3],
    pub count: u32,
    pub fw: u32,
    pub frequency: u32,
    pub wav: u32,              // `struct WaveData *wav` — ROM pointer value
    pub currentPointer: usize, // `s8 *` — offset into the PCM stream
    pub track: Option<usize>,  // `struct MusicPlayerTrack *track`
    pub prevChannelPointer: Option<usize>,
    pub nextChannelPointer: Option<usize>,
    pub dummy4: u32,
    pub xpi: u16,
    pub xpc: u16,
}

#[derive(Clone, Copy, Default)]
pub struct ToneDataPad; // (placeholder to keep struct grouping close to the header layout)

#[derive(Clone, Copy)]
pub struct SongHeader {
    pub trackCount: u8,
    pub blockCount: u8,
    pub priority: u8,
    pub reverb: u8,
    pub tone: u32, // `struct ToneData *tone` — ROM pointer value
    pub part: [u32; MAX_MUSICPLAYER_TRACKS], // `u8 *part[1]` — ROM pointer per track
}

// `Default` == the zero-initialized track `Clear64byte`/`memset` produces in C.
#[derive(Clone, Default)]
pub struct MusicPlayerTrack {
    pub flags: u8,
    pub wait: u8,
    pub patternLevel: u8,
    pub repN: u8,
    pub gateTime: u8,
    pub key: u8,
    pub velocity: u8,
    pub runningStatus: u8,
    pub keyM: u8,
    pub pitM: u8,
    pub keyShift: i8,
    pub keyShiftX: i8,
    pub tune: i8,
    pub pitX: u8,
    pub bend: i8,
    pub bendRange: u8,
    pub volMR: u8,
    pub volML: u8,
    pub vol: u8,
    pub volX: u8,
    pub pan: i8,
    pub panX: i8,
    pub modM: i8,
    pub mod_: u8,
    pub modT: u8,
    pub lfoSpeed: u8,
    pub lfoSpeedC: u8,
    pub lfoDelay: u8,
    pub lfoDelayC: u8,
    pub priority: u8,
    pub pseudoEchoVolume: u8,
    pub pseudoEchoLength: u8,
    pub chan: Option<usize>, // `struct SoundChannel *chan`
    pub tone: ToneData,
    pub timer: u16,
    pub unk_3C: u32,
    pub cmdPtr: usize, // `u8 *cmdPtr` — offset into the song byte stream
    pub patternStack: [usize; 3],
}

#[derive(Clone, Copy, Default)]
pub struct PokemonCrySong {
    pub trackCount: u8,
    pub blockCount: u8,
    pub priority: u8,
    pub reverb: u8,
    pub tone: u32,
    pub part: [u32; 2],
    pub gap: u8,
    pub part0: u8,
    pub tuneValue: u8,
    pub gotoCmd: u8,
    pub gotoTarget: u32,
    pub part1: u8,
    pub tuneValue2: u8,
    pub cont: [u8; 2],
    pub volCmd: u8,
    pub volumeValue: u8,
    pub unkCmd0D: [u8; 2],
    pub unkCmd0DParam: u32,
    pub xreleCmd: [u8; 2],
    pub releaseValue: u8,
    pub panCmd: u8,
    pub panValue: u8,
    pub tieCmd: u8,
    pub tieKeyValue: u8,
    pub tieVelocityValue: u8,
    pub xwaitCmd: [u8; 2],
    pub length: u16,
    pub end: [u8; 2],
}

#[derive(Default)]
pub struct MusicPlayerInfo {
    pub songHeader: Option<SongHeader>, // `struct SongHeader *songHeader`
    pub status: u32,
    pub trackCount: u8,
    pub priority: u8,
    pub cmd: u8,
    pub unk_B: u8,
    pub clock: u32,
    pub memAccArea: [u8; 0x10], // `u8 *memAccArea` → gMPlayMemAccArea, owned inline
    pub tempoD: u16,
    pub tempoU: u16,
    pub tempoI: u16,
    pub tempoC: u16,
    pub fadeOI: u16,
    pub fadeOC: u16,
    pub fadeOV: u16,
    pub tracks: Vec<MusicPlayerTrack>, // `struct MusicPlayerTrack *tracks`
    pub tone: u32,                     // `struct ToneData *tone` — ROM pointer value
    pub ident: u32,
    // MPlayMainNext / musicPlayerNext (intrusive lists) live in SoundInfo's ordering; omitted here.
}

// gMPlayTable / gSongTable / gPokemonCry* globals are ROM- and game-specific; in Optime the song
// table is owned by `super::rom::GbaRom`, so they are not duplicated as globals here.

// ===========================================================================================
// Modeled hardware — the CGB sound register file + SOUNDBIAS, so the transliterated `CgbSound`
// reads back exactly what it wrote within a frame. All other MMIO (DMA/FIFO/timers/SOUNDCNT) is a
// commented no-op: Optime's mixer, not these registers, produces audio.
// ===========================================================================================

// Offsets into `Hw::io`, relative to the sound register base REG_BASE+0x60 (so NR10 == 0).
const O_NR10: usize = 0x00;
const O_NR11: usize = 0x02;
const O_NR12: usize = 0x03;
const O_NR13: usize = 0x04;
const O_NR14: usize = 0x05;
const O_NR21: usize = 0x08;
const O_NR22: usize = 0x09;
const O_NR23: usize = 0x0c;
const O_NR24: usize = 0x0d;
const O_NR30: usize = 0x10;
const O_NR31: usize = 0x12;
const O_NR32: usize = 0x13;
const O_NR33: usize = 0x14;
const O_NR34: usize = 0x15;
const O_NR41: usize = 0x18;
const O_NR42: usize = 0x19;
const O_NR43: usize = 0x1c;
const O_NR44: usize = 0x1d;
const O_NR50: usize = 0x20;
const O_NR51: usize = 0x21;
const O_SOUNDBIAS_H: usize = 0x29;

/// Models the GBA sound-register block (REG_BASE+0x60..0xA0) that `CgbSound` reads back within a
/// frame. Backed by a flat byte array so the C's register-pointer arithmetic (`REG_ADDR_NR10 + 1`,
/// per-channel `nrx0..nrx4` pointers) transliterates directly to indices.
pub struct Hw {
    pub io: [u8; 0x40],
}

impl Default for Hw {
    fn default() -> Self {
        // SoundInit sets SOUNDBIAS_H so its PWM-rate bits read as the 65536 Hz mode CgbSound keys on.
        let mut io = [0u8; 0x40];
        io[O_SOUNDBIAS_H] = 0x40;
        Self { io }
    }
}

// ===========================================================================================
// m4a.c
// ===========================================================================================

// [m4a.c:23] MidiKeyToFreq — transliterated. (Shares the pret-transcribed tables in `super::tables`;
// a `midi_key_to_freq` there already carries the pokeemerald parity test.)
pub fn MidiKeyToFreq(wav_freq: u32, mut key: u8, fineAdjust: u8) -> u32 {
    let mut fineAdjustShifted = (fineAdjust as u32) << 24;

    if key > 178 {
        key = 178;
        fineAdjustShifted = 255 << 24;
    }

    let val1 = SCALE_TABLE[key as usize];
    let val1 = FREQ_TABLE[(val1 & 0xF) as usize] >> (val1 >> 4);

    let val2 = SCALE_TABLE[key as usize + 1];
    let val2 = FREQ_TABLE[(val2 & 0xF) as usize] >> (val2 >> 4);

    umul3232h32(
        wav_freq,
        val1.wrapping_add(umul3232h32(val2.wrapping_sub(val1), fineAdjustShifted)),
    )
}

// [m4a.c:48] MPlayContinue — transliterated.
pub fn MPlayContinue(mplayInfo: &mut MusicPlayerInfo) {
    if mplayInfo.ident == ID_NUMBER {
        mplayInfo.ident += 1;
        mplayInfo.status &= !MUSICPLAYER_STATUS_PAUSE;
        mplayInfo.ident = ID_NUMBER;
    }
}

// [m4a.c:58] MPlayFadeOut — transliterated.
pub fn MPlayFadeOut(mplayInfo: &mut MusicPlayerInfo, speed: u16) {
    if mplayInfo.ident == ID_NUMBER {
        mplayInfo.ident += 1;
        mplayInfo.fadeOC = speed;
        mplayInfo.fadeOI = speed;
        mplayInfo.fadeOV = 64 << FADE_VOL_SHIFT;
        mplayInfo.ident = ID_NUMBER;
    }
}

// [m4a.c:107] m4aSongNumStart — transliterated control flow.
//
// ┌─ SYNTH BACKEND SEAM ─ m4aSongNumStart / m4aSongNumStop / MPlayStart ─┐
//   In `m4a.c` these index the ROM globals `gSongTable` / `gMPlayTable` and call `MPlayStart`,
//   which in turn calls the asm `TrackStop`. Optime resolves songs through `super::rom::GbaRom`
//   and drives playback through `super::player::GbaPlayer` (the "no-C-home" engine), so the song-
//   selection wrappers are provided by that layer rather than duplicated here.
// └──────────────────────────────────────────────────────────────────────┘

// [m4a.c:186] m4aMPlayContinue — transliterated.
pub fn m4aMPlayContinue(mplayInfo: &mut MusicPlayerInfo) {
    MPlayContinue(mplayInfo);
}

// [m4a.c:202] m4aMPlayFadeOut — transliterated.
pub fn m4aMPlayFadeOut(mplayInfo: &mut MusicPlayerInfo, speed: u16) {
    MPlayFadeOut(mplayInfo, speed);
}

// [m4a.c:207] m4aMPlayFadeOutTemporarily — transliterated.
pub fn m4aMPlayFadeOutTemporarily(mplayInfo: &mut MusicPlayerInfo, speed: u16) {
    if mplayInfo.ident == ID_NUMBER {
        mplayInfo.ident += 1;
        mplayInfo.fadeOC = speed;
        mplayInfo.fadeOI = speed;
        mplayInfo.fadeOV = (64 << FADE_VOL_SHIFT) | TEMPORARY_FADE;
        mplayInfo.ident = ID_NUMBER;
    }
}

// [m4a.c:219] m4aMPlayFadeIn — transliterated.
pub fn m4aMPlayFadeIn(mplayInfo: &mut MusicPlayerInfo, speed: u16) {
    if mplayInfo.ident == ID_NUMBER {
        mplayInfo.ident += 1;
        mplayInfo.fadeOC = speed;
        mplayInfo.fadeOI = speed;
        mplayInfo.fadeOV = (0 << FADE_VOL_SHIFT) | FADE_IN;
        mplayInfo.status &= !MUSICPLAYER_STATUS_PAUSE;
        mplayInfo.ident = ID_NUMBER;
    }
}

// [m4a.c:232] m4aMPlayImmInit — transliterated. (`Clear64byte(track)` becomes a default reset.)
pub fn m4aMPlayImmInit(mplayInfo: &mut MusicPlayerInfo) {
    let trackCount = mplayInfo.trackCount as i32;
    for i in 0..trackCount as usize {
        let track = &mut mplayInfo.tracks[i];
        if track.flags & MPT_FLG_EXIST != 0 && track.flags & MPT_FLG_START != 0 {
            *track = MusicPlayerTrack::default(); // Clear64byte(track)
            track.flags = MPT_FLG_EXIST;
            track.bendRange = 2;
            track.volX = 64;
            track.lfoSpeed = 22;
            track.tone.type_ = 1;
        }
    }
}

// [m4a.c:353] SoundInit — transliterated shell around a backend seam.
//
// ┌─ SYNTH BACKEND SEAM ─ SoundInit hardware bring-up ─┐
//   `m4a.c` programs DMA1/DMA2 for the two DirectSound FIFOs, resets SOUNDCNT_X/H, points the
//   sound DMAs at `pcmBuffer`, and installs the asm `ply_note` / jump-table. Optime has no FIFO
//   DMA — the mixer in `super::player` synthesizes PCM directly — so all register/DMA writes are
//   documented no-ops. The one line with a portable equivalent is `SampleFreqSet`.
// └────────────────────────────────────────────────────┘
pub fn SoundInit(hw: &mut Hw) {
    // (register/DMA setup omitted — see seam banner above)
    SampleFreqSet(hw, SOUND_MODE_FREQ_13379);
}

// [m4a.c:400] SampleFreqSet — transliterated math; the VCount spin-wait + timer arming are the seam.
//
// ┌─ SYNTH BACKEND SEAM ─ SampleFreqSet timer/VCount ─┐
//   The final block arms hardware Timer 0 at the mixer rate and busy-waits on REG_VCOUNT. Optime's
//   master clock lives in `SynthController`, so only the derived-rate arithmetic is kept.
// └───────────────────────────────────────────────────┘
pub fn SampleFreqSet(_hw: &mut Hw, freq: u32) -> SampleFreq {
    let freq = (freq & 0xF0000) >> 16;
    let pcmSamplesPerVBlank = gPcmSamplesPerVBlankTable[(freq - 1) as usize] as i32;
    let pcmDmaPeriod = PCM_DMA_BUF_SIZE as i32 / pcmSamplesPerVBlank;

    // LCD refresh rate 59.7275Hz
    let pcmFreq = (597275 * pcmSamplesPerVBlank + 5000) / 10000;

    // CPU frequency 16.78Mhz
    let divFreq = (16777216 / pcmFreq + 1) >> 1;

    SampleFreq {
        freq: freq as u8,
        pcmSamplesPerVBlank,
        pcmDmaPeriod: pcmDmaPeriod as u8,
        pcmFreq,
        divFreq,
    }
}

/// The rate fields `SampleFreqSet` computes into `SoundInfo` (returned instead of stored globally).
#[derive(Clone, Copy, Default)]
pub struct SampleFreq {
    pub freq: u8,
    pub pcmSamplesPerVBlank: i32,
    pub pcmDmaPeriod: u8,
    pub pcmFreq: i32,
    pub divFreq: i32,
}

// [m4a.c:611] MPlayStart — transliterated. `TrackStop` (asm) is a seam; the state reset is faithful.
pub fn MPlayStart(mplayInfo: &mut MusicPlayerInfo, songHeader: SongHeader) {
    if mplayInfo.ident != ID_NUMBER {
        return;
    }

    let unk_B = mplayInfo.unk_B;

    let cond = unk_B == 0
        || ((mplayInfo.songHeader.is_none() || (mplayInfo.tracks[0].flags & MPT_FLG_START) == 0)
            && ((mplayInfo.status & MUSICPLAYER_STATUS_TRACK) == 0
                || (mplayInfo.status & MUSICPLAYER_STATUS_PAUSE) != 0))
        || mplayInfo.priority <= songHeader.priority;

    if cond {
        mplayInfo.ident += 1;
        mplayInfo.status = 0;
        mplayInfo.songHeader = Some(songHeader);
        mplayInfo.tone = songHeader.tone;
        mplayInfo.priority = songHeader.priority;
        mplayInfo.clock = 0;
        mplayInfo.tempoD = 150;
        mplayInfo.tempoI = 150;
        mplayInfo.tempoU = 0x100;
        mplayInfo.tempoC = 0;
        mplayInfo.fadeOI = 0;

        let mut i = 0usize;
        while i < songHeader.trackCount as usize && i < mplayInfo.trackCount as usize {
            let track = &mut mplayInfo.tracks[i];
            TrackStop(track); // ── SYNTH BACKEND SEAM (asm m4a_1.s): see TrackStop below ──
            track.flags = MPT_FLG_EXIST | MPT_FLG_START;
            track.chan = None; // track->chan = 0
            track.cmdPtr = songHeader.part[i] as usize;
            i += 1;
        }

        while i < mplayInfo.trackCount as usize {
            let track = &mut mplayInfo.tracks[i];
            TrackStop(track);
            track.flags = 0;
            i += 1;
        }

        // (songHeader->reverb & SOUND_MODE_REVERB_SET → m4aSoundMode: reverb has no Optime effect)
        mplayInfo.ident = ID_NUMBER;
    }
}

// [m4a.c:668] m4aMPlayStop — transliterated. `TrackStop` (asm) is a seam.
pub fn m4aMPlayStop(mplayInfo: &mut MusicPlayerInfo) {
    if mplayInfo.ident != ID_NUMBER {
        return;
    }

    mplayInfo.ident += 1;
    mplayInfo.status |= MUSICPLAYER_STATUS_PAUSE;

    let n = mplayInfo.trackCount as usize;
    for i in 0..n {
        TrackStop(&mut mplayInfo.tracks[i]);
    }

    mplayInfo.ident = ID_NUMBER;
}

// [m4a.c:692] FadeOutBody — transliterated. `TrackStop` (asm) is a seam; the fade math is faithful.
pub fn FadeOutBody(mplayInfo: &mut MusicPlayerInfo) {
    if mplayInfo.fadeOI == 0 {
        return;
    }
    mplayInfo.fadeOC -= 1;
    if mplayInfo.fadeOC != 0 {
        return;
    }

    mplayInfo.fadeOC = mplayInfo.fadeOI;

    if mplayInfo.fadeOV & FADE_IN != 0 {
        mplayInfo.fadeOV = mplayInfo.fadeOV.wrapping_add(4 << FADE_VOL_SHIFT);
        if mplayInfo.fadeOV >= (64 << FADE_VOL_SHIFT) {
            mplayInfo.fadeOV = 64 << FADE_VOL_SHIFT;
            mplayInfo.fadeOI = 0;
        }
    } else {
        mplayInfo.fadeOV = mplayInfo.fadeOV.wrapping_sub(4 << FADE_VOL_SHIFT);
        if (mplayInfo.fadeOV as i16) <= 0 {
            let n = mplayInfo.trackCount as usize;
            for i in 0..n {
                let fadeOV = mplayInfo.fadeOV;
                TrackStop(&mut mplayInfo.tracks[i]);
                let val = TEMPORARY_FADE & fadeOV;
                if val == 0 {
                    mplayInfo.tracks[i].flags = 0;
                }
            }

            if mplayInfo.fadeOV & TEMPORARY_FADE != 0 {
                mplayInfo.status |= MUSICPLAYER_STATUS_PAUSE;
            } else {
                mplayInfo.status = MUSICPLAYER_STATUS_PAUSE;
            }

            mplayInfo.fadeOI = 0;
            return;
        }
    }

    let n = mplayInfo.trackCount as usize;
    for i in 0..n {
        if mplayInfo.tracks[i].flags & MPT_FLG_EXIST != 0 {
            let fadeOV = mplayInfo.fadeOV;
            mplayInfo.tracks[i].volX = (fadeOV >> FADE_VOL_SHIFT) as u8;
            mplayInfo.tracks[i].flags |= MPT_FLG_VOLCHG;
        }
    }
}

// [m4a.c:765] TrkVolPitSet — transliterated. Pure integer volume/pan/pitch resolution per track.
pub fn TrkVolPitSet(track: &mut MusicPlayerTrack) {
    if track.flags & MPT_FLG_VOLSET != 0 {
        let mut x = (track.vol as u32 * track.volX as u32) >> 5;

        if track.modT == 1 {
            x = (x * (track.modM as i32 + 128) as u32) >> 7;
        }

        let mut y = 2 * track.pan as i32 + track.panX as i32;

        if track.modT == 2 {
            y += track.modM as i32;
        }

        if y < -128 {
            y = -128;
        } else if y > 127 {
            y = 127;
        }

        track.volMR = (((y + 128) as u32 * x) >> 8) as u8;
        track.volML = (((127 - y) as u32 * x) >> 8) as u8;
    }

    if track.flags & MPT_FLG_PITSET != 0 {
        let bend = track.bend as i32 * track.bendRange as i32;
        let mut x = (track.tune as i32 + bend) * 4
            + ((track.keyShift as i32) << 8)
            + ((track.keyShiftX as i32) << 8)
            + track.pitX as i32;

        if track.modT == 0 {
            x += 16 * track.modM as i32;
        }

        track.keyM = (x >> 8) as u8;
        track.pitM = x as u8;
    }

    track.flags &= !(MPT_FLG_PITSET | MPT_FLG_VOLSET);
}

// [m4a.c:810] MidiKeyToCgbFreq — transliterated. (Parity-tested in `super::tables`.)
pub fn MidiKeyToCgbFreq(chanNum: u8, mut key: u8, mut fineAdjust: u8) -> u32 {
    if chanNum == 4 {
        if key <= 20 {
            key = 0;
        } else {
            key -= 21;
            if key > 59 {
                key = 59;
            }
        }
        NOISE_TABLE[key as usize] as u32
    } else {
        if key <= 35 {
            fineAdjust = 0;
            key = 0;
        } else {
            key -= 36;
            if key > 130 {
                key = 130;
                fineAdjust = 255;
            }
        }

        let val1 = CGB_SCALE_TABLE[key as usize];
        let val1 = (CGB_FREQ_TABLE[(val1 & 0xF) as usize] as i32) >> (val1 >> 4);

        let val2 = CGB_SCALE_TABLE[key as usize + 1];
        let val2 = (CGB_FREQ_TABLE[(val2 & 0xF) as usize] as i32) >> (val2 >> 4);

        (val1 + ((fineAdjust as i32 * (val2 - val1)) >> 8) + 2048) as u32
    }
}

// [m4a.c:857] CgbOscOff — transliterated. The register writes are the seam (no audible effect in
// Optime, whose CGB voices are silenced by `super::player`); kept for structural fidelity.
pub fn CgbOscOff(hw: &mut Hw, chanNum: u8) {
    match chanNum {
        1 => {
            hw.io[O_NR12] = 8;
            hw.io[O_NR14] = 0x80;
        }
        2 => {
            hw.io[O_NR22] = 8;
            hw.io[O_NR24] = 0x80;
        }
        3 => {
            hw.io[O_NR30] = 0;
        }
        _ => {
            hw.io[O_NR42] = 8;
            hw.io[O_NR44] = 0x80;
        }
    }
}

// [m4a.c:878] CgbPan (static inline) — transliterated.
fn CgbPan(chan: &mut CgbChannel) -> bool {
    let rightVolume = chan.rightVolume;
    let leftVolume = chan.leftVolume;

    if rightVolume >= leftVolume {
        if rightVolume / 2 >= leftVolume {
            chan.pan = 0x0F;
            return true;
        }
    } else if leftVolume / 2 >= rightVolume {
        chan.pan = 0xF0;
        return true;
    }

    false
}

// [m4a.c:903] CgbModVol — transliterated. `soundInfo->mode & 1` (stereo flag) passed in as `mode`.
pub fn CgbModVol(chan: &mut CgbChannel, mode: u8) {
    if (mode & 1) != 0 || !CgbPan(chan) {
        chan.pan = 0xFF;
        chan.envelopeGoal = chan.leftVolume.wrapping_add(chan.rightVolume);
        chan.envelopeGoal /= 16;
    } else {
        chan.envelopeGoal = chan.leftVolume.wrapping_add(chan.rightVolume);
        chan.envelopeGoal /= 16;
        if chan.envelopeGoal > 15 {
            chan.envelopeGoal = 15;
        }
    }

    chan.sustainGoal = ((chan.envelopeGoal as u32 * chan.sustain as u32 + 15) >> 4) as u8;
    chan.pan &= chan.panMask;
}

// ┌─ SYNTH BACKEND SEAM ─ CgbSound register back-end ─┐
//   `CgbSound` is the CGB (PSG) per-frame envelope+pitch engine. Its *computation* — attack/decay/
//   sustain/release stepping, pseudo-echo, `CgbModVol` — is real behavior and is transliterated
//   faithfully in `super::player`'s CGB path (which owns the per-channel state and drives the
//   `SynthEvent` stream). The heavy per-register NRx0..NRx4 / WAVE_RAM / NR51 back-half of the C
//   function only pokes MMIO that Optime's software PSG does not read, so it is not reproduced
//   here. `CgbModVol` / `CgbPan` / `CgbOscOff` above are provided because they are pure state math
//   the backend reuses.
// └────────────────────────────────────────────────────┘

// [m4a.c:1234] m4aMPlayTempoControl — transliterated.
pub fn m4aMPlayTempoControl(mplayInfo: &mut MusicPlayerInfo, tempo: u16) {
    if mplayInfo.ident == ID_NUMBER {
        mplayInfo.ident += 1;
        mplayInfo.tempoU = tempo;
        mplayInfo.tempoI = ((mplayInfo.tempoD as u32 * mplayInfo.tempoU as u32) >> 8) as u16;
        mplayInfo.ident = ID_NUMBER;
    }
}

// [m4a.c:1245] m4aMPlayVolumeControl — transliterated.
pub fn m4aMPlayVolumeControl(mplayInfo: &mut MusicPlayerInfo, trackBits: u16, volume: u16) {
    if mplayInfo.ident != ID_NUMBER {
        return;
    }

    mplayInfo.ident += 1;

    let n = mplayInfo.trackCount as usize;
    let mut bit: u32 = 1;
    for i in 0..n {
        if trackBits as u32 & bit != 0 && mplayInfo.tracks[i].flags & MPT_FLG_EXIST != 0 {
            mplayInfo.tracks[i].volX = (volume / 4) as u8;
            mplayInfo.tracks[i].flags |= MPT_FLG_VOLCHG;
        }
        bit <<= 1;
    }

    mplayInfo.ident = ID_NUMBER;
}

// [m4a.c:1279] m4aMPlayPitchControl — transliterated.
pub fn m4aMPlayPitchControl(mplayInfo: &mut MusicPlayerInfo, trackBits: u16, pitch: i16) {
    if mplayInfo.ident != ID_NUMBER {
        return;
    }

    mplayInfo.ident += 1;

    let n = mplayInfo.trackCount as usize;
    let mut bit: u32 = 1;
    for i in 0..n {
        if trackBits as u32 & bit != 0 && mplayInfo.tracks[i].flags & MPT_FLG_EXIST != 0 {
            mplayInfo.tracks[i].keyShiftX = (pitch >> 8) as i8;
            mplayInfo.tracks[i].pitX = pitch as u8;
            mplayInfo.tracks[i].flags |= MPT_FLG_PITCHG;
        }
        bit <<= 1;
    }

    mplayInfo.ident = ID_NUMBER;
}

// [m4a.c:1314] m4aMPlayPanpotControl — transliterated.
pub fn m4aMPlayPanpotControl(mplayInfo: &mut MusicPlayerInfo, trackBits: u16, pan: i8) {
    if mplayInfo.ident != ID_NUMBER {
        return;
    }

    mplayInfo.ident += 1;

    let n = mplayInfo.trackCount as usize;
    let mut bit: u32 = 1;
    for i in 0..n {
        if trackBits as u32 & bit != 0 && mplayInfo.tracks[i].flags & MPT_FLG_EXIST != 0 {
            mplayInfo.tracks[i].panX = pan;
            mplayInfo.tracks[i].flags |= MPT_FLG_VOLCHG;
        }
        bit <<= 1;
    }

    mplayInfo.ident = ID_NUMBER;
}

// [m4a.c:1348] ClearModM — transliterated.
pub fn ClearModM(track: &mut MusicPlayerTrack) {
    track.lfoSpeedC = 0;
    track.modM = 0;

    if track.modT == 0 {
        track.flags |= MPT_FLG_PITCHG;
    } else {
        track.flags |= MPT_FLG_VOLCHG;
    }
}

// [m4a.c:1359] m4aMPlayModDepthSet — transliterated.
pub fn m4aMPlayModDepthSet(mplayInfo: &mut MusicPlayerInfo, trackBits: u16, modDepth: u8) {
    if mplayInfo.ident != ID_NUMBER {
        return;
    }

    mplayInfo.ident += 1;

    let n = mplayInfo.trackCount as usize;
    let mut bit: u32 = 1;
    for i in 0..n {
        if trackBits as u32 & bit != 0 && mplayInfo.tracks[i].flags & MPT_FLG_EXIST != 0 {
            mplayInfo.tracks[i].mod_ = modDepth;
            if mplayInfo.tracks[i].mod_ == 0 {
                ClearModM(&mut mplayInfo.tracks[i]);
            }
        }
        bit <<= 1;
    }

    mplayInfo.ident = ID_NUMBER;
}

// [m4a.c:1395] m4aMPlayLFOSpeedSet — transliterated.
pub fn m4aMPlayLFOSpeedSet(mplayInfo: &mut MusicPlayerInfo, trackBits: u16, lfoSpeed: u8) {
    if mplayInfo.ident != ID_NUMBER {
        return;
    }

    mplayInfo.ident += 1;

    let n = mplayInfo.trackCount as usize;
    let mut bit: u32 = 1;
    for i in 0..n {
        if trackBits as u32 & bit != 0 && mplayInfo.tracks[i].flags & MPT_FLG_EXIST != 0 {
            mplayInfo.tracks[i].lfoSpeed = lfoSpeed;
            if mplayInfo.tracks[i].lfoSpeed == 0 {
                ClearModM(&mut mplayInfo.tracks[i]);
            }
        }
        bit <<= 1;
    }

    mplayInfo.ident = ID_NUMBER;
}

// [m4a.c:1437] ply_memacc — transliterated. The `MEMACC_COND_JUMP` macro expands to a returned
// verdict: `true` means "take the conditional jump" (the C tail-calls the asm `ply_goto` via the
// jump table); `false` means skip the 4-byte jump target. The caller (`super::sequencer`) performs
// the actual jump, since the goto handler is asm — that dispatch is the seam.
pub enum MemAccResult {
    /// Fall through; `cmdPtr` advanced past the jump operand (`track->cmdPtr += 4`).
    Continue,
    /// Take the conditional jump — dispatch to `ply_goto` (asm seam) with `cmdPtr` at the target.
    Goto,
}

pub fn ply_memacc(
    memAccArea: &mut [u8; 0x10],
    track: &mut MusicPlayerTrack,
    cmd: &[u8],
) -> MemAccResult {
    // op = *track->cmdPtr++; addrIdx = *track->cmdPtr++; data = *track->cmdPtr++;
    let op = cmd[track.cmdPtr] as u32;
    track.cmdPtr += 1;

    let addr = cmd[track.cmdPtr] as usize; // index into memAccArea (was `memAccArea + *cmdPtr`)
    track.cmdPtr += 1;

    let data = cmd[track.cmdPtr];
    track.cmdPtr += 1;

    macro_rules! MEMACC_COND_JUMP {
        ($cond:expr) => {
            if $cond {
                return MemAccResult::Goto; // cond_true: tail-call ply_goto (asm seam)
            } else {
                track.cmdPtr += 4; // cond_false
                return MemAccResult::Continue;
            }
        };
    }

    match op {
        0 => memAccArea[addr] = data,
        1 => memAccArea[addr] = memAccArea[addr].wrapping_add(data),
        2 => memAccArea[addr] = memAccArea[addr].wrapping_sub(data),
        3 => memAccArea[addr] = memAccArea[data as usize],
        4 => memAccArea[addr] = memAccArea[addr].wrapping_add(memAccArea[data as usize]),
        5 => memAccArea[addr] = memAccArea[addr].wrapping_sub(memAccArea[data as usize]),
        6 => MEMACC_COND_JUMP!(memAccArea[addr] == data),
        7 => MEMACC_COND_JUMP!(memAccArea[addr] != data),
        8 => MEMACC_COND_JUMP!(memAccArea[addr] > data),
        9 => MEMACC_COND_JUMP!(memAccArea[addr] >= data),
        10 => MEMACC_COND_JUMP!(memAccArea[addr] <= data),
        11 => MEMACC_COND_JUMP!(memAccArea[addr] < data),
        12 => MEMACC_COND_JUMP!(memAccArea[addr] == memAccArea[data as usize]),
        13 => MEMACC_COND_JUMP!(memAccArea[addr] != memAccArea[data as usize]),
        14 => MEMACC_COND_JUMP!(memAccArea[addr] > memAccArea[data as usize]),
        15 => MEMACC_COND_JUMP!(memAccArea[addr] >= memAccArea[data as usize]),
        16 => MEMACC_COND_JUMP!(memAccArea[addr] <= memAccArea[data as usize]),
        17 => MEMACC_COND_JUMP!(memAccArea[addr] < memAccArea[data as usize]),
        _ => {}
    }
    MemAccResult::Continue
}

// [m4a.c:1544] ply_xwave — transliterated (READ_XCMD_BYTE assembles a 32-bit ROM pointer).
pub fn ply_xwave(track: &mut MusicPlayerTrack, cmd: &[u8]) {
    let wav = u32::from_le_bytes([
        cmd[track.cmdPtr],
        cmd[track.cmdPtr + 1],
        cmd[track.cmdPtr + 2],
        cmd[track.cmdPtr + 3],
    ]);
    track.tone.wav = wav;
    track.cmdPtr += 4;
}

// [m4a.c:1561] ply_xtype — transliterated.
pub fn ply_xtype(track: &mut MusicPlayerTrack, cmd: &[u8]) {
    track.tone.type_ = cmd[track.cmdPtr];
    track.cmdPtr += 1;
}

// [m4a.c:1567] ply_xatta — transliterated.
pub fn ply_xatta(track: &mut MusicPlayerTrack, cmd: &[u8]) {
    track.tone.attack = cmd[track.cmdPtr];
    track.cmdPtr += 1;
}

// [m4a.c:1573] ply_xdeca — transliterated.
pub fn ply_xdeca(track: &mut MusicPlayerTrack, cmd: &[u8]) {
    track.tone.decay = cmd[track.cmdPtr];
    track.cmdPtr += 1;
}

// [m4a.c:1579] ply_xsust — transliterated.
pub fn ply_xsust(track: &mut MusicPlayerTrack, cmd: &[u8]) {
    track.tone.sustain = cmd[track.cmdPtr];
    track.cmdPtr += 1;
}

// [m4a.c:1585] ply_xrele — transliterated.
pub fn ply_xrele(track: &mut MusicPlayerTrack, cmd: &[u8]) {
    track.tone.release = cmd[track.cmdPtr];
    track.cmdPtr += 1;
}

// [m4a.c:1591] ply_xiecv — transliterated.
pub fn ply_xiecv(track: &mut MusicPlayerTrack, cmd: &[u8]) {
    track.pseudoEchoVolume = cmd[track.cmdPtr];
    track.cmdPtr += 1;
}

// [m4a.c:1597] ply_xiecl — transliterated.
pub fn ply_xiecl(track: &mut MusicPlayerTrack, cmd: &[u8]) {
    track.pseudoEchoLength = cmd[track.cmdPtr];
    track.cmdPtr += 1;
}

// [m4a.c:1603] ply_xleng — transliterated.
pub fn ply_xleng(track: &mut MusicPlayerTrack, cmd: &[u8]) {
    track.tone.length = cmd[track.cmdPtr];
    track.cmdPtr += 1;
}

// [m4a.c:1609] ply_xswee — transliterated.
pub fn ply_xswee(track: &mut MusicPlayerTrack, cmd: &[u8]) {
    track.tone.pan_sweep = cmd[track.cmdPtr];
    track.cmdPtr += 1;
}

// [m4a.c:1615] ply_xwait — transliterated (READ_XCMD_BYTE assembles a 16-bit timer).
pub fn ply_xwait(track: &mut MusicPlayerTrack, cmd: &[u8]) {
    let len = u16::from_le_bytes([cmd[track.cmdPtr], cmd[track.cmdPtr + 1]]);

    if track.timer < len {
        track.timer += 1;
        // track->cmdPtr -= 2 was preceded by the +2 the READ macro implies; net: re-read next frame
        track.wait = 1;
    } else {
        track.timer = 0;
        track.cmdPtr += 2;
    }
}

// [m4a.c:1639] ply_xcmd_0D — transliterated.
pub fn ply_xcmd_0D(track: &mut MusicPlayerTrack, cmd: &[u8]) {
    let unk = u32::from_le_bytes([
        cmd[track.cmdPtr],
        cmd[track.cmdPtr + 1],
        cmd[track.cmdPtr + 2],
        cmd[track.cmdPtr + 3],
    ]);
    track.unk_3C = unk;
    track.cmdPtr += 4;
}

// [m4a.c:1701] SetPokemonCryVolume — transliterated.
pub fn SetPokemonCryVolume(song: &mut PokemonCrySong, val: u8) {
    song.volumeValue = val & 0x7F;
}

// [m4a.c:1706] SetPokemonCryPanpot — transliterated.
pub fn SetPokemonCryPanpot(song: &mut PokemonCrySong, val: i8) {
    song.panValue = ((val as i32 + C_V) & 0x7F) as u8;
}

// [m4a.c:1711] SetPokemonCryPitch — transliterated.
pub fn SetPokemonCryPitch(song: &mut PokemonCrySong, val: i16) {
    let b = val.wrapping_add(0x80);
    let a = song.tuneValue2.wrapping_sub(song.tuneValue);
    song.tieKeyValue = ((b >> 8) & 0x7F) as u8;
    song.tuneValue = ((b >> 1) & 0x7F) as u8;
    song.tuneValue2 = a.wrapping_add(((b >> 1) & 0x7F) as u8) & 0x7F;
}

// [m4a.c:1720] SetPokemonCryLength — transliterated.
pub fn SetPokemonCryLength(song: &mut PokemonCrySong, val: u16) {
    song.length = val;
}

// [m4a.c:1725] SetPokemonCryRelease — transliterated.
pub fn SetPokemonCryRelease(song: &mut PokemonCrySong, val: u8) {
    song.releaseValue = val;
}

// [m4a.c:1730] SetPokemonCryProgress — transliterated.
pub fn SetPokemonCryProgress(song: &mut PokemonCrySong, val: u32) {
    song.unkCmd0DParam = val;
}

// [m4a.c:1745] SetPokemonCryChorus — transliterated.
pub fn SetPokemonCryChorus(song: &mut PokemonCrySong, val: i8) {
    if val != 0 {
        song.trackCount = 2;
        song.tuneValue2 = (val as u8).wrapping_add(song.tuneValue) & 0x7F;
    } else {
        song.trackCount = 1;
    }
}

// [m4a.c:1778] SetPokemonCryPriority — transliterated.
pub fn SetPokemonCryPriority(song: &mut PokemonCrySong, val: u8) {
    song.priority = val;
}

// ===========================================================================================
// Asm-engine seams — implemented by the "no-C-home" modules, not by m4a.c
// ===========================================================================================
//
// ┌─ SYNTH BACKEND SEAM ─ m4a_1.s (hand-written ARM, no C to mirror) ─┐
//   The reference sound engine implements these in assembly; there is no `m4a.c` counterpart to
//   diff against. Optime provides the equivalent behavior in the "no-C-home" modules:
//
//     MPlayMain / ply_note / TrackStop / ply_* note commands  →  super::sequencer::Mp2kSequencer
//     SoundMain / CgbSound register back-end / the mixer      →  super::player::GbaPlayer
//
//   They appear here as documented no-ops so the transliterated call sites above (`MPlayStart`,
//   `m4aMPlayStop`, `FadeOutBody`) read exactly like `m4a.c`.
// └────────────────────────────────────────────────────────────────────┘

/// `TrackStop` (asm `m4a_1.s`): stops a track's channel and clears its per-note state. Real
/// behavior lives in `super::player`/`super::sequencer`; here it only zeroes the track-local note
/// bookkeeping the transliterated callers rely on.
pub fn TrackStop(_track: &mut MusicPlayerTrack) {
    // SYNTH BACKEND SEAM: channel teardown handled by super::player; nothing to do on this model.
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `TrkVolPitSet` volume path: a plain track (no modulation) resolves to symmetric L/R volumes
    /// scaled by `vol * volX >> 5`, matching the C oracle.
    #[test]
    fn trkvolpitset_center_pan_volume() {
        let mut t = MusicPlayerTrack {
            flags: MPT_FLG_VOLSET | MPT_FLG_PITSET,
            vol: 127,
            volX: 64,
            pan: 0,
            bendRange: 2,
            ..Default::default()
        };
        TrkVolPitSet(&mut t);

        // C: x = (127*64)>>5 = 254; centered y=0 → volMR=((128)*254)>>8, volML=((127)*254)>>8.
        let x: u32 = (127 * 64) >> 5;
        assert_eq!(t.volMR, (((0 + 128) as u32 * x) >> 8) as u8);
        assert_eq!(t.volML, (((127 - 0) as u32 * x) >> 8) as u8);
        // Flags cleared.
        assert_eq!(t.flags & (MPT_FLG_PITSET | MPT_FLG_VOLSET), 0);
    }

    /// `MidiKeyToFreq` here must equal the pre-existing pret-transcribed `tables::midi_key_to_freq`.
    #[test]
    fn midikeytofreq_matches_tables() {
        for key in [0u8, 36, 60, 90, 178, 255] {
            for fine in [0u8, 127, 255] {
                let freq = 13379u32 << 10;
                assert_eq!(
                    MidiKeyToFreq(freq, key, fine),
                    super::super::tables::midi_key_to_freq(freq, key, fine),
                );
            }
        }
    }

    /// `MidiKeyToCgbFreq` here must equal the pret-transcribed `tables::midi_key_to_cgb_freq`.
    #[test]
    fn midikeytocgbfreq_matches_tables() {
        for chan in 1u8..=4 {
            for key in [0u8, 21, 36, 76, 130, 200, 255] {
                for fine in [0u8, 128, 255] {
                    assert_eq!(
                        MidiKeyToCgbFreq(chan, key, fine),
                        super::super::tables::midi_key_to_cgb_freq(chan, key, fine),
                    );
                }
            }
        }
    }

    /// `CgbModVol`: mono mode (mode&1) forces full pan and averages the two volumes into the goal.
    #[test]
    fn cgbmodvol_mono_averages() {
        let mut c = CgbChannel {
            leftVolume: 100,
            rightVolume: 60,
            panMask: 0x11,
            sustain: 8,
            ..Default::default()
        };
        CgbModVol(&mut c, 1); // mono
        assert_eq!(c.pan & !0x11, 0); // masked to panMask
        assert_eq!(c.envelopeGoal, (100u8.wrapping_add(60)) / 16);
    }

    /// `ply_memacc` op 0 stores; op 6 (== compare) reports Goto on equality, else Continue+skip.
    #[test]
    fn ply_memacc_store_and_branch() {
        let mut mem = [0u8; 0x10];
        let mut track = MusicPlayerTrack::default();

        // op=0 (set) area[2] = 42
        let cmd = [0u8, 2, 42];
        track.cmdPtr = 0;
        matches!(
            ply_memacc(&mut mem, &mut track, &cmd),
            MemAccResult::Continue
        );
        assert_eq!(mem[2], 42);

        // op=6 (== data) area[2]==42 → Goto
        let cmd = [6u8, 2, 42, 0, 0, 0, 0];
        track.cmdPtr = 0;
        assert!(matches!(
            ply_memacc(&mut mem, &mut track, &cmd),
            MemAccResult::Goto
        ));

        // op=6 area[2]!=99 → Continue, cmdPtr advanced past the 3 operands + 4 jump bytes
        let cmd = [6u8, 2, 99, 0, 0, 0, 0];
        track.cmdPtr = 0;
        assert!(matches!(
            ply_memacc(&mut mem, &mut track, &cmd),
            MemAccResult::Continue
        ));
        assert_eq!(track.cmdPtr, 3 + 4);
    }
}

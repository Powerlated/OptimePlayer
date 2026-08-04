#![allow(
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    clippy::too_many_arguments,
    clippy::manual_clamp
)]

use super::m4a_tables::{
    CGB_FREQ_TABLE, CGB_SCALE_TABLE, FREQ_TABLE, NOISE_TABLE, SCALE_TABLE, umul3232h32,
};

pub const ID_NUMBER: u32 = 0x68736D53;

pub const C_V: i32 = 0x40;

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
pub const TONEDATA_TYPE_SPL: u8 = 0x40;
pub const TONEDATA_TYPE_RHY: u8 = 0x80;

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

pub const gCgb3Vol: [u8; 16] = [
    0x00, 0x00, 0x60, 0x60, 0x60, 0x60, 0x40, 0x40, 0x40, 0x40, 0x80, 0x80, 0x80, 0x80, 0x20, 0x20,
];

pub const gPcmSamplesPerVBlankTable: [u16; 12] =
    [96, 132, 176, 224, 264, 304, 352, 448, 528, 608, 672, 704];

#[derive(Clone, Copy, Default)]
pub struct WaveData {
    pub type_: u16,
    pub status: u16,
    pub freq: u32,
    pub loopStart: u32,
    pub size: u32,
    pub data: usize,
}

impl WaveData {
    pub fn read(rom: &[u8], wav_addr: u32) -> Option<WaveData> {
        use crate::util::{read_u16, read_u32};
        let off = super::rom::ptr_to_offset(wav_addr, rom.len())?;
        let size = read_u32(rom, off + 12);
        let loopStart = read_u32(rom, off + 8);
        let data = off + 16;
        if size as usize > rom.len().saturating_sub(data) || loopStart > size {
            return None;
        }
        Some(WaveData {
            type_: read_u16(rom, off),
            status: read_u16(rom, off + 2),
            freq: read_u32(rom, off + 4),
            loopStart,
            size,
            data,
        })
    }

    pub fn looping(&self) -> bool {
        (self.status >> 8) & 0x40 != 0
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ToneData {
    pub kind: u8,
    pub key: u8,
    pub length: u8,
    pub pan_sweep: u8,
    pub wav: u32,
    pub attack: u8,
    pub decay: u8,
    pub sustain: u8,
    pub release: u8,
}

impl ToneData {
    pub fn read(rom: &[u8], offset: usize) -> ToneData {
        use crate::util::{read_u8, read_u32};
        ToneData {
            kind: read_u8(rom, offset),
            key: read_u8(rom, offset + 1),
            length: read_u8(rom, offset + 2),
            pan_sweep: read_u8(rom, offset + 3),
            wav: read_u32(rom, offset + 4),
            attack: read_u8(rom, offset + 8),
            decay: read_u8(rom, offset + 9),
            sustain: read_u8(rom, offset + 10),
            release: read_u8(rom, offset + 11),
        }
    }
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
    pub n4: u8,
    pub pan: u8,
    pub panMask: u8,
    pub modify: u8,
    pub length: u8,
    pub sweep: u8,
    pub frequency: u32,
    pub wavePointer: u32,
    pub currentPointer: u32,
    pub track: Option<usize>,
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
    pub key: u8,
    pub envelopeVolume: u8,
    pub envelopeVolumeRight: u8,
    pub envelopeVolumeLeft: u8,
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
    pub count: u32,
    pub fw: u32,
    pub frequency: u32,
    pub wav: u32,
    pub currentPointer: usize,
    pub track: Option<usize>,
    pub prevChannelPointer: Option<usize>,
    pub nextChannelPointer: Option<usize>,
    pub dummy4: u32,
    pub xpi: u16,
    pub xpc: u16,
}

#[derive(Clone, Copy, Default)]
pub struct ToneDataPad;

#[derive(Clone, Copy, Default)]
pub struct SongHeader {
    pub trackCount: u8,
    pub blockCount: u8,
    pub priority: u8,
    pub reverb: u8,
    pub tone: u32,
    pub part: [u32; MAX_MUSICPLAYER_TRACKS],
}

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
    pub chan: Option<usize>,
    pub tone: ToneData,
    pub timer: u16,
    pub unk_3C: u32,
    pub cmdPtr: usize,
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
    pub songHeader: Option<SongHeader>,
    pub status: u32,
    pub trackCount: u8,
    pub priority: u8,
    pub cmd: u8,
    pub unk_B: u8,
    pub clock: u32,
    pub memAccArea: [u8; 0x10],
    pub tempoD: u16,
    pub tempoU: u16,
    pub tempoI: u16,
    pub tempoC: u16,
    pub fadeOI: u16,
    pub fadeOC: u16,
    pub fadeOV: u16,
    pub tracks: Vec<MusicPlayerTrack>,
    pub tone: u32,
    pub ident: u32,
}

#[derive(Clone, Default)]
pub struct SoundInfo {
    pub ident: u32,
    pub reverb: u8,
    pub maxChans: u8,
    pub masterVolume: u8,
    pub mode: u32,
    pub c15: u8,
    pub cgbChans: [CgbChannel; 4],
    pub chans: [SoundChannel; MAX_DIRECTSOUND_CHANNELS],
}

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

pub struct Hw {
    pub io: [u8; 0x40],
}

impl Default for Hw {
    fn default() -> Self {
        let mut io = [0u8; 0x40];
        io[O_SOUNDBIAS_H] = 0x40;
        Self { io }
    }
}

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

pub fn MPlayContinue(mplayInfo: &mut MusicPlayerInfo) {
    if mplayInfo.ident == ID_NUMBER {
        mplayInfo.ident += 1;
        mplayInfo.status &= !MUSICPLAYER_STATUS_PAUSE;
        mplayInfo.ident = ID_NUMBER;
    }
}

pub fn MPlayFadeOut(mplayInfo: &mut MusicPlayerInfo, speed: u16) {
    if mplayInfo.ident == ID_NUMBER {
        mplayInfo.ident += 1;
        mplayInfo.fadeOC = speed;
        mplayInfo.fadeOI = speed;
        mplayInfo.fadeOV = 64 << FADE_VOL_SHIFT;
        mplayInfo.ident = ID_NUMBER;
    }
}

pub fn m4aMPlayContinue(mplayInfo: &mut MusicPlayerInfo) {
    MPlayContinue(mplayInfo);
}

pub fn m4aMPlayFadeOut(mplayInfo: &mut MusicPlayerInfo, speed: u16) {
    MPlayFadeOut(mplayInfo, speed);
}

pub fn m4aMPlayFadeOutTemporarily(mplayInfo: &mut MusicPlayerInfo, speed: u16) {
    if mplayInfo.ident == ID_NUMBER {
        mplayInfo.ident += 1;
        mplayInfo.fadeOC = speed;
        mplayInfo.fadeOI = speed;
        mplayInfo.fadeOV = (64 << FADE_VOL_SHIFT) | TEMPORARY_FADE;
        mplayInfo.ident = ID_NUMBER;
    }
}

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

pub fn m4aMPlayImmInit(mplayInfo: &mut MusicPlayerInfo) {
    let trackCount = mplayInfo.trackCount as i32;
    for i in 0..trackCount as usize {
        let track = &mut mplayInfo.tracks[i];
        if track.flags & MPT_FLG_EXIST != 0 && track.flags & MPT_FLG_START != 0 {
            *track = MusicPlayerTrack::default();
            track.flags = MPT_FLG_EXIST;
            track.bendRange = 2;
            track.volX = 64;
            track.lfoSpeed = 22;
            track.tone.kind = 1;
        }
    }
}

pub fn SoundInit(hw: &mut Hw) {
    SampleFreqSet(hw, SOUND_MODE_FREQ_13379);
}

pub fn SampleFreqSet(_hw: &mut Hw, freq: u32) -> SampleFreq {
    let freq = (freq & 0xF0000) >> 16;
    let pcmSamplesPerVBlank = gPcmSamplesPerVBlankTable[(freq - 1) as usize] as i32;
    let pcmDmaPeriod = PCM_DMA_BUF_SIZE as i32 / pcmSamplesPerVBlank;

    let pcmFreq = (597275 * pcmSamplesPerVBlank + 5000) / 10000;

    let divFreq = (16777216 / pcmFreq + 1) >> 1;

    SampleFreq {
        freq: freq as u8,
        pcmSamplesPerVBlank,
        pcmDmaPeriod: pcmDmaPeriod as u8,
        pcmFreq,
        divFreq,
    }
}

#[derive(Clone, Copy, Default)]
pub struct SampleFreq {
    pub freq: u8,
    pub pcmSamplesPerVBlank: i32,
    pub pcmDmaPeriod: u8,
    pub pcmFreq: i32,
    pub divFreq: i32,
}

pub fn reverb_from_song_header(song_reverb: u8) -> Option<u8> {
    (u32::from(song_reverb) & SOUND_MODE_REVERB_SET != 0)
        .then(|| (u32::from(song_reverb) & SOUND_MODE_REVERB_VAL) as u8)
}

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
            TrackStop(track);
            track.flags = MPT_FLG_EXIST | MPT_FLG_START;
            track.chan = None;
            track.cmdPtr = songHeader.part[i] as usize;
            i += 1;
        }

        while i < mplayInfo.trackCount as usize {
            let track = &mut mplayInfo.tracks[i];
            TrackStop(track);
            track.flags = 0;
            i += 1;
        }

        mplayInfo.ident = ID_NUMBER;
    }
}

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

pub fn CgbSound(soundInfo: &mut SoundInfo) {
    if soundInfo.c15 != 0 {
        soundInfo.c15 -= 1;
    } else {
        soundInfo.c15 = 14;
    }
    let double_step = soundInfo.c15 == 0;
    let mode = (soundInfo.mode & 1) as u8;

    for ch in 1..=4u8 {
        let channels = &mut soundInfo.cgbChans[(ch - 1) as usize];
        if channels.statusFlags & SOUND_CHANNEL_SF_ON == 0 {
            continue;
        }
        let mut off = CgbSound_Channel(channels, ch, mode);
        if !off && double_step && channels.statusFlags & SOUND_CHANNEL_SF_IEC == 0 {
            off = CgbSound_Channel(channels, ch, mode);
        }
        if off {
            channels.statusFlags = 0;
        }
    }
}

fn CgbSound_Channel(c: &mut CgbChannel, _ch: u8, mode: u8) -> bool {
    enum Flow {
        StepRepeat,
        StepComplete,
        DecayStart,
        SustainStart,
        PseudoEchoStart,
    }
    let mut flow = if c.statusFlags & SOUND_CHANNEL_SF_START != 0 {
        if c.statusFlags & SOUND_CHANNEL_SF_STOP == 0 {
            c.statusFlags = SOUND_CHANNEL_SF_ENV_ATTACK;
            c.modify = CGB_CHANNEL_MO_PIT | CGB_CHANNEL_MO_VOL;
            CgbModVol(c, mode);
            c.envelopeCounter = c.attack;
            if c.attack as i8 != 0 {
                c.envelopeVolume = 0;
                Flow::StepComplete
            } else {
                Flow::DecayStart
            }
        } else {
            return true;
        }
    } else if c.statusFlags & SOUND_CHANNEL_SF_IEC != 0 {
        c.pseudoEchoLength = c.pseudoEchoLength.wrapping_sub(1);
        if (c.pseudoEchoLength as i8) <= 0 {
            return true;
        }
        return false;
    } else if c.statusFlags & SOUND_CHANNEL_SF_STOP != 0
        && c.statusFlags & SOUND_CHANNEL_SF_ENV != 0
    {
        c.statusFlags &= !SOUND_CHANNEL_SF_ENV;
        c.envelopeCounter = c.release;
        if c.release as i8 != 0 {
            c.modify |= CGB_CHANNEL_MO_VOL;
            Flow::StepComplete
        } else {
            Flow::PseudoEchoStart
        }
    } else {
        Flow::StepRepeat
    };

    loop {
        match flow {
            Flow::StepRepeat => {
                if c.envelopeCounter == 0 {
                    CgbModVol(c, mode);
                    match c.statusFlags & SOUND_CHANNEL_SF_ENV {
                        SOUND_CHANNEL_SF_ENV_RELEASE => {
                            c.envelopeVolume = c.envelopeVolume.wrapping_sub(1);
                            if c.envelopeVolume as i8 <= 0 {
                                flow = Flow::PseudoEchoStart;
                                continue;
                            }
                            c.envelopeCounter = c.release;
                        }
                        SOUND_CHANNEL_SF_ENV_SUSTAIN => {
                            c.envelopeVolume = c.sustainGoal;
                            c.envelopeCounter = 7;
                        }
                        SOUND_CHANNEL_SF_ENV_DECAY => {
                            c.envelopeVolume = c.envelopeVolume.wrapping_sub(1);
                            if (c.envelopeVolume as i8) <= c.sustainGoal as i8 {
                                flow = Flow::SustainStart;
                                continue;
                            }
                            c.envelopeCounter = c.decay;
                        }
                        _ => {
                            c.envelopeVolume = c.envelopeVolume.wrapping_add(1);
                            if c.envelopeVolume >= c.envelopeGoal {
                                flow = Flow::DecayStart;
                                continue;
                            }
                            c.envelopeCounter = c.attack;
                        }
                    }
                }
                flow = Flow::StepComplete;
            }
            Flow::DecayStart => {
                c.statusFlags = c.statusFlags.wrapping_sub(1);
                c.envelopeCounter = c.decay;
                if c.envelopeCounter != 0 {
                    c.modify |= CGB_CHANNEL_MO_VOL;
                    c.envelopeVolume = c.envelopeGoal;
                    flow = Flow::StepComplete;
                } else {
                    flow = Flow::SustainStart;
                }
            }
            Flow::SustainStart => {
                if c.sustain == 0 {
                    c.statusFlags &= !SOUND_CHANNEL_SF_ENV;
                    flow = Flow::PseudoEchoStart;
                } else {
                    c.statusFlags = c.statusFlags.wrapping_sub(1);
                    c.modify |= CGB_CHANNEL_MO_VOL;
                    c.envelopeVolume = c.sustainGoal;
                    c.envelopeCounter = 7;
                    flow = Flow::StepComplete;
                }
            }
            Flow::PseudoEchoStart => {
                c.envelopeVolume =
                    (((c.envelopeGoal as u32 * c.pseudoEchoVolume as u32) + 0xFF) >> 8) as u8;
                if c.envelopeVolume != 0 {
                    c.statusFlags |= SOUND_CHANNEL_SF_IEC;
                    c.modify |= CGB_CHANNEL_MO_VOL;
                    return false;
                }
                return true;
            }
            Flow::StepComplete => {
                c.envelopeCounter = c.envelopeCounter.wrapping_sub(1);
                return false;
            }
        }
    }
}

pub fn m4aMPlayTempoControl(mplayInfo: &mut MusicPlayerInfo, tempo: u16) {
    if mplayInfo.ident == ID_NUMBER {
        mplayInfo.ident += 1;
        mplayInfo.tempoU = tempo;
        mplayInfo.tempoI = ((mplayInfo.tempoD as u32 * mplayInfo.tempoU as u32) >> 8) as u16;
        mplayInfo.ident = ID_NUMBER;
    }
}

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

pub fn ClearModM(track: &mut MusicPlayerTrack) {
    track.lfoSpeedC = 0;
    track.modM = 0;

    if track.modT == 0 {
        track.flags |= MPT_FLG_PITCHG;
    } else {
        track.flags |= MPT_FLG_VOLCHG;
    }
}

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

pub enum MemAccResult {
    Continue,
    Goto,
}

pub fn ply_memacc(
    memAccArea: &mut [u8; 0x10],
    track: &mut MusicPlayerTrack,
    cmd: &[u8],
) -> MemAccResult {
    let op = cmd[track.cmdPtr] as u32;
    track.cmdPtr += 1;

    let addr = (cmd[track.cmdPtr] as usize) & 0xF;
    track.cmdPtr += 1;

    let data = cmd[track.cmdPtr];
    track.cmdPtr += 1;

    let lhs = memAccArea[addr];
    let rhs = memAccArea[(data as usize) & 0xF];

    macro_rules! MEMACC_COND_JUMP {
        ($cond:expr) => {
            if $cond {
                return MemAccResult::Goto;
            } else {
                track.cmdPtr += 4;
                return MemAccResult::Continue;
            }
        };
    }

    match op {
        0 => memAccArea[addr] = data,
        1 => memAccArea[addr] = lhs.wrapping_add(data),
        2 => memAccArea[addr] = lhs.wrapping_sub(data),
        3 => memAccArea[addr] = rhs,
        4 => memAccArea[addr] = lhs.wrapping_add(rhs),
        5 => memAccArea[addr] = lhs.wrapping_sub(rhs),
        6 => MEMACC_COND_JUMP!(lhs == data),
        7 => MEMACC_COND_JUMP!(lhs != data),
        8 => MEMACC_COND_JUMP!(lhs > data),
        9 => MEMACC_COND_JUMP!(lhs >= data),
        10 => MEMACC_COND_JUMP!(lhs <= data),
        11 => MEMACC_COND_JUMP!(lhs < data),
        12 => MEMACC_COND_JUMP!(lhs == rhs),
        13 => MEMACC_COND_JUMP!(lhs != rhs),
        14 => MEMACC_COND_JUMP!(lhs > rhs),
        15 => MEMACC_COND_JUMP!(lhs >= rhs),
        16 => MEMACC_COND_JUMP!(lhs <= rhs),
        17 => MEMACC_COND_JUMP!(lhs < rhs),
        _ => {}
    }
    MemAccResult::Continue
}

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

pub fn ply_xtype(track: &mut MusicPlayerTrack, cmd: &[u8]) {
    track.tone.kind = cmd[track.cmdPtr];
    track.cmdPtr += 1;
}

pub fn ply_xatta(track: &mut MusicPlayerTrack, cmd: &[u8]) {
    track.tone.attack = cmd[track.cmdPtr];
    track.cmdPtr += 1;
}

pub fn ply_xdeca(track: &mut MusicPlayerTrack, cmd: &[u8]) {
    track.tone.decay = cmd[track.cmdPtr];
    track.cmdPtr += 1;
}

pub fn ply_xsust(track: &mut MusicPlayerTrack, cmd: &[u8]) {
    track.tone.sustain = cmd[track.cmdPtr];
    track.cmdPtr += 1;
}

pub fn ply_xrele(track: &mut MusicPlayerTrack, cmd: &[u8]) {
    track.tone.release = cmd[track.cmdPtr];
    track.cmdPtr += 1;
}

pub fn ply_xiecv(track: &mut MusicPlayerTrack, cmd: &[u8]) {
    track.pseudoEchoVolume = cmd[track.cmdPtr];
    track.cmdPtr += 1;
}

pub fn ply_xiecl(track: &mut MusicPlayerTrack, cmd: &[u8]) {
    track.pseudoEchoLength = cmd[track.cmdPtr];
    track.cmdPtr += 1;
}

pub fn ply_xleng(track: &mut MusicPlayerTrack, cmd: &[u8]) {
    track.tone.length = cmd[track.cmdPtr];
    track.cmdPtr += 1;
}

pub fn ply_xswee(track: &mut MusicPlayerTrack, cmd: &[u8]) {
    track.tone.pan_sweep = cmd[track.cmdPtr];
    track.cmdPtr += 1;
}

pub fn ply_xwait(track: &mut MusicPlayerTrack, cmd: &[u8]) {
    let len = u16::from_le_bytes([cmd[track.cmdPtr], cmd[track.cmdPtr + 1]]);

    if track.timer < len {
        track.timer += 1;
        track.cmdPtr -= 2;
        track.wait = 1;
    } else {
        track.timer = 0;
        track.cmdPtr += 2;
    }
}

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

pub fn SetPokemonCryVolume(song: &mut PokemonCrySong, val: u8) {
    song.volumeValue = val & 0x7F;
}

pub fn SetPokemonCryPanpot(song: &mut PokemonCrySong, val: i8) {
    song.panValue = ((val as i32 + C_V) & 0x7F) as u8;
}

pub fn SetPokemonCryPitch(song: &mut PokemonCrySong, val: i16) {
    let b = val.wrapping_add(0x80);
    let a = song.tuneValue2.wrapping_sub(song.tuneValue);
    song.tieKeyValue = ((b >> 8) & 0x7F) as u8;
    song.tuneValue = ((b >> 1) & 0x7F) as u8;
    song.tuneValue2 = a.wrapping_add(((b >> 1) & 0x7F) as u8) & 0x7F;
}

pub fn SetPokemonCryLength(song: &mut PokemonCrySong, val: u16) {
    song.length = val;
}

pub fn SetPokemonCryRelease(song: &mut PokemonCrySong, val: u8) {
    song.releaseValue = val;
}

pub fn SetPokemonCryProgress(song: &mut PokemonCrySong, val: u32) {
    song.unkCmd0DParam = val;
}

pub fn SetPokemonCryChorus(song: &mut PokemonCrySong, val: i8) {
    if val != 0 {
        song.trackCount = 2;
        song.tuneValue2 = (val as u8).wrapping_add(song.tuneValue) & 0x7F;
    } else {
        song.trackCount = 1;
    }
}

pub fn SetPokemonCryPriority(song: &mut PokemonCrySong, val: u8) {
    song.priority = val;
}

pub fn TrackStop(_track: &mut MusicPlayerTrack) {}

#[cfg(test)]
mod tests {
    use super::*;

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

        let x: u32 = (127 * 64) >> 5;
        let pan: i32 = 0;
        assert_eq!(t.volMR, (((0x80 + pan) as u32 * x) >> 8) as u8);
        assert_eq!(t.volML, (((0x7F - pan) as u32 * x) >> 8) as u8);
        assert_eq!(t.flags & (MPT_FLG_PITSET | MPT_FLG_VOLSET), 0);
    }

    #[test]
    fn midikeytofreq_matches_tables() {
        for key in [0u8, 36, 60, 90, 178, 255] {
            for fine in [0u8, 127, 255] {
                let freq = 13379u32 << 10;
                assert_eq!(
                    MidiKeyToFreq(freq, key, fine),
                    super::super::m4a_tables::midi_key_to_freq(freq, key, fine),
                );
            }
        }
    }

    #[test]
    fn midikeytocgbfreq_matches_tables() {
        for chan in 1u8..=4 {
            for key in [0u8, 21, 36, 76, 130, 200, 255] {
                for fine in [0u8, 128, 255] {
                    assert_eq!(
                        MidiKeyToCgbFreq(chan, key, fine),
                        super::super::m4a_tables::midi_key_to_cgb_freq(chan, key, fine),
                    );
                }
            }
        }
    }

    #[test]
    fn cgbmodvol_mono_averages() {
        let mut c = CgbChannel {
            leftVolume: 100,
            rightVolume: 60,
            panMask: 0x11,
            sustain: 8,
            ..Default::default()
        };
        CgbModVol(&mut c, 1);
        assert_eq!(c.pan & !0x11, 0);
        assert_eq!(c.envelopeGoal, (100u8.wrapping_add(60)) / 16);
    }

    #[test]
    fn cgb_envelope_attacks_sustains_and_releases() {
        let mut si = SoundInfo {
            cgbChans: [CgbChannel::default(); 4],
            ..SoundInfo::default()
        };
        si.cgbChans[1] = CgbChannel {
            statusFlags: SOUND_CHANNEL_SF_START,
            type_: 2,
            leftVolume: 127,
            rightVolume: 127,
            attack: 2,
            decay: 4,
            sustain: 8,
            release: 2,
            ..CgbChannel::default()
        };
        let mut peak = 0u8;
        for _ in 0..80 {
            CgbSound(&mut si);
            peak = peak.max(si.cgbChans[1].envelopeVolume);
        }
        assert_eq!(peak, 15, "attack reaches the full envelope goal");
        assert_eq!(
            si.cgbChans[1].envelopeVolume, 8,
            "held note settles at the sustain floor"
        );
        assert_ne!(si.cgbChans[1].statusFlags, 0, "still sounding while held");

        si.cgbChans[1].statusFlags |= SOUND_CHANNEL_SF_STOP;
        let mut turned_off = false;
        for _ in 0..200 {
            CgbSound(&mut si);
            if si.cgbChans[1].statusFlags == 0 {
                turned_off = true;
                break;
            }
        }
        assert!(turned_off, "a released, echo-less CGB channel shuts off");
    }

    #[test]
    fn ply_memacc_store_and_branch() {
        let mut mem = [0u8; 0x10];
        let mut track = MusicPlayerTrack::default();

        let cmd = [0u8, 2, 42];
        track.cmdPtr = 0;
        matches!(
            ply_memacc(&mut mem, &mut track, &cmd),
            MemAccResult::Continue
        );
        assert_eq!(mem[2], 42);

        let cmd = [6u8, 2, 42, 0, 0, 0, 0];
        track.cmdPtr = 0;
        assert!(matches!(
            ply_memacc(&mut mem, &mut track, &cmd),
            MemAccResult::Goto
        ));

        let cmd = [6u8, 2, 99, 0, 0, 0, 0];
        track.cmdPtr = 0;
        assert!(matches!(
            ply_memacc(&mut mem, &mut track, &cmd),
            MemAccResult::Continue
        ));
        assert_eq!(track.cmdPtr, 3 + 4);
    }
}

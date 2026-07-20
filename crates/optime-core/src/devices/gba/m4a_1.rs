//! A Rust transliteration of `pret/pokeemerald`'s `src/m4a_1.s` — the hand-written ARM sound
//! engine, which has no C to mirror (so it lives here, not in [`super::m4a`], the `m4a.c` port).
//!
//! This is the set of routines actually implemented in `m4a_1.s`, translated onto [`super::m4a`]'s
//! faithful hardware structs (`MusicPlayerInfo`, `MusicPlayerTrack`, `SoundInfo`, `SoundChannel`,
//! `CgbChannel`) and calling straight into `m4a.rs`'s C helpers (`TrkVolPitSet`, `ClearModM`,
//! `ply_memacc`, `ply_x*`, `MidiKeyToFreq`, `MidiKeyToCgbFreq`, `CgbSound`) — no logic duplicated:
//!
//! * [`MPlayMain`] — the per-VBlank track interpreter (`ply_note` + the `ply_*` command handlers,
//!   gate countdown, `ply_lfos`, and the end-of-frame volume/pitch refresh).
//! * [`SoundMain`] — the envelope pass: `SoundMainRAM` (DirectSound) + the `CgbSound` call. The
//!   PCM-DMA/FIFO *mixing* half of `SoundMain`/`SoundMainRAM` is what Optime's software synth
//!   replaces, so only the envelope/pitch state computation is reproduced here.
//!
//! The engine only mutates hardware state; it emits no audio and no [`crate::SynthEvent`]s. The
//! glue in [`super`] drives it per frame and reads the resulting channel state out.

#![allow(non_snake_case, non_upper_case_globals)]

use super::m4a::{
    self, CGB_CHANNEL_MO_PIT, CGB_CHANNEL_MO_VOL, CgbChannel, MPT_FLG_EXIST, MPT_FLG_PITCHG,
    MPT_FLG_START, MPT_FLG_VOLCHG, MusicPlayerInfo, MusicPlayerTrack, SOUND_CHANNEL_SF_ENV,
    SOUND_CHANNEL_SF_ENV_ATTACK, SOUND_CHANNEL_SF_ENV_DECAY, SOUND_CHANNEL_SF_IEC,
    SOUND_CHANNEL_SF_ON, SOUND_CHANNEL_SF_START, SOUND_CHANNEL_SF_STOP, SoundChannel, SoundInfo,
    TONEDATA_TYPE_CGB, TONEDATA_TYPE_RHY, TONEDATA_TYPE_SPL, ToneData,
};
use super::m4a_tables::{CLOCK_TABLE, midi_key_to_cgb_freq, midi_key_to_freq};
use super::rom::ptr_to_offset;
use crate::util::{read_u8, read_u32};

/// The tempo accumulator threshold: one sequencer step runs per `tempoC >= 150`.
const TEMPO_STEP: u16 = 150;

/// `TONEDATA_P_S_PAN` — the bias subtracted from a rhythm voice's `pan_sweep` byte before doubling.
const TONEDATA_P_S_PAN: u8 = 0xC0;

/// What one [`MPlayMain`] frame reports back to the glue that the hardware would not track itself.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameResult {
    /// A backward song-loop `GOTO` (or `REPT 0`) fired this frame — the glue emits one `Looped`.
    pub looped: bool,
}

/// Which channel-pool array a resolved note landed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    Ds(usize),
    Cgb(usize),
}

/// `chan.key + track.keyM`, clamped at zero (`ply_note` / `MPlayMain`). `keyM` is applied as a
/// signed offset (`i8`).
fn add_key(key: u8, key_m: u8) -> u8 {
    (i32::from(key) + i32::from(key_m as i8)).max(0) as u8
}

/// Whether a track slot is live (`MPT_FLG_EXIST`).
fn exists(track: &MusicPlayerTrack) -> bool {
    track.flags & MPT_FLG_EXIST != 0
}

/// `WaveData::freq` (offset 4) at the ROM pointer `wav`, for `MidiKeyToFreq`.
fn wav_freq(rom: &[u8], wav: u32) -> u32 {
    ptr_to_offset(wav, rom.len()).map_or(0, |o| read_u32(rom, o + 4))
}

/// `ChnVolSetAsm` (`m4a_1.s:1508`): per-channel right/left volumes from velocity, rhythm pan, and
/// the track's mixer volumes.
fn chn_vol_set(velocity: u8, rhythm_pan: i8, vol_mr: u8, vol_ml: u8) -> (u8, u8) {
    let pan = i32::from(rhythm_pan);
    let right = ((0x80 + pan) * i32::from(velocity) * i32::from(vol_mr)) >> 14;
    let left = ((0x7F - pan) * i32::from(velocity) * i32::from(vol_ml)) >> 14;
    (right.min(0xFF) as u8, left.min(0xFF) as u8)
}

// ===========================================================================================
// MPlayMain — the per-VBlank track interpreter (m4a_1.s:1129)
// ===========================================================================================

/// `MPlayMain`: advances the tempo clock, running `tempoI / 150` sequencer steps this frame, then
/// applies the end-of-frame volume/pitch refresh. Returns per-frame [`FrameResult`] the glue needs.
pub fn MPlayMain(mp: &mut MusicPlayerInfo, si: &mut SoundInfo, rom: &[u8]) -> FrameResult {
    let mut result = FrameResult::default();

    // status high bit set == paused/stopped: nothing to do.
    if mp.status & 0x8000_0000 != 0 {
        return result;
    }

    mp.tempoC += mp.tempoI;
    while mp.tempoC >= TEMPO_STEP {
        mp.tempoC -= TEMPO_STEP;
        step(mp, si, rom, &mut result);
        if mp.status & 0x8000_0000 != 0 {
            break;
        }
    }

    refresh_changed_tracks(mp, si, rom);
    result
}

/// One sequencer step over every track (the `tempoC >= 150` body of `MPlayMain`).
fn step(mp: &mut MusicPlayerInfo, si: &mut SoundInfo, rom: &[u8], result: &mut FrameResult) {
    let mut any_exists = false;
    for t in 0..mp.tracks.len() {
        // MPT_FLG_START: reset the track to its defaults on the first frame, preserving cmdPtr
        // (in the C, `Clear64byte` doesn't reach `cmdPtr`/`patternStack`, which sit past 0x40).
        if mp.tracks[t].flags & MPT_FLG_START != 0 {
            let cmd_ptr = mp.tracks[t].cmdPtr;
            let mut tr = MusicPlayerTrack {
                flags: MPT_FLG_EXIST,
                bendRange: 2,
                volX: 0x40,
                lfoSpeed: 0x16,
                cmdPtr: cmd_ptr,
                ..MusicPlayerTrack::default()
            };
            tr.tone.kind = 1;
            mp.tracks[t] = tr;
        }
        if !exists(&mp.tracks[t]) {
            continue;
        }

        gate_tick(si, t);

        // Run commands until the track owes a wait; the guard stops malformed data hanging us.
        let mut guard = 0u32;
        while mp.tracks[t].wait == 0 {
            execute_command(mp, si, rom, t, result);
            guard += 1;
            if !exists(&mp.tracks[t]) {
                break;
            }
            if guard > 100_000 {
                fine(mp, si, t);
                break;
            }
        }
        if !exists(&mp.tracks[t]) {
            continue;
        }
        any_exists = true;
        mp.tracks[t].wait -= 1;
        lfo_step(&mut mp.tracks[t]);
    }

    mp.clock += 1;
    if !any_exists {
        mp.status = 0x8000_0000;
    } else {
        mp.status = 1;
    }
}

/// `MPlayMain`'s per-channel gate countdown for track `t`: expired gates begin the release; a
/// channel the envelope already shut off (`statusFlags == 0`) is detached.
fn gate_tick(si: &mut SoundInfo, t: usize) {
    let tick = |c: &mut ChannelHdr| {
        if c.status() & SOUND_CHANNEL_SF_ON == 0 {
            c.set_track(None); // ClearChain
            return;
        }
        if c.gate_time() != 0 {
            let g = c.gate_time() - 1;
            c.set_gate_time(g);
            if g == 0 {
                c.set_status(c.status() | SOUND_CHANNEL_SF_STOP);
            }
        }
    };
    for i in 0..si.maxChans as usize {
        if si.chans[i].track == Some(t) {
            tick(&mut ChannelHdr::Ds(&mut si.chans[i]));
        }
    }
    for i in 0..4 {
        if si.cgbChans[i].track == Some(t) {
            tick(&mut ChannelHdr::Cgb(&mut si.cgbChans[i]));
        }
    }
}

/// Fetches and executes one command (the dispatch loop of `MPlayMain`).
fn execute_command(
    mp: &mut MusicPlayerInfo,
    si: &mut SoundInfo,
    rom: &[u8],
    t: usize,
    result: &mut FrameResult,
) {
    let mut cmd = read_u8(rom, mp.tracks[t].cmdPtr);
    if cmd < 0x80 {
        cmd = mp.tracks[t].runningStatus;
    } else {
        mp.tracks[t].cmdPtr += 1;
        if cmd >= 0xBD {
            mp.tracks[t].runningStatus = cmd;
        }
    }

    if cmd >= 0xCF {
        ply_note(mp, si, rom, t, cmd - 0xCF);
    } else if cmd > 0xB0 {
        control_command(mp, si, rom, t, cmd, result);
    } else if cmd >= 0x80 {
        // W00..W96 rest.
        mp.tracks[t].wait = CLOCK_TABLE[(cmd - 0x80) as usize];
    } else {
        fine(mp, si, t);
    }
}

/// Reads a one-byte command operand at the PC.
fn arg(mp: &mut MusicPlayerInfo, rom: &[u8], t: usize) -> u8 {
    let v = read_u8(rom, mp.tracks[t].cmdPtr);
    mp.tracks[t].cmdPtr += 1;
    v
}

/// The `0xB1..=0xCE` control commands (`gMPlayJumpTable`).
fn control_command(
    mp: &mut MusicPlayerInfo,
    si: &mut SoundInfo,
    rom: &[u8],
    t: usize,
    cmd: u8,
    result: &mut FrameResult,
) {
    match cmd {
        0xB2 => ply_goto(mp, si, rom, t, true, result), // GOTO: the song's loop point
        0xB3 => {
            // PATT: call a pattern.
            let level = mp.tracks[t].patternLevel as usize;
            if level >= 3 {
                fine(mp, si, t);
            } else {
                mp.tracks[t].patternStack[level] = mp.tracks[t].cmdPtr + 4;
                mp.tracks[t].patternLevel += 1;
                ply_goto(mp, si, rom, t, false, result);
            }
        }
        0xB4 => {
            // PEND: return from a pattern.
            if mp.tracks[t].patternLevel > 0 {
                mp.tracks[t].patternLevel -= 1;
                mp.tracks[t].cmdPtr = mp.tracks[t].patternStack[mp.tracks[t].patternLevel as usize];
            }
        }
        0xB5 => {
            // REPT: repeat a section `count` times (0 = forever = the song looping).
            let count = read_u8(rom, mp.tracks[t].cmdPtr);
            if count == 0 {
                mp.tracks[t].cmdPtr += 1;
                ply_goto(mp, si, rom, t, true, result);
            } else {
                mp.tracks[t].repN = mp.tracks[t].repN.wrapping_add(1);
                let rep_n = mp.tracks[t].repN;
                mp.tracks[t].cmdPtr += 1;
                if rep_n < count {
                    ply_goto(mp, si, rom, t, false, result);
                } else {
                    mp.tracks[t].repN = 0;
                    mp.tracks[t].cmdPtr += 4;
                }
            }
        }
        0xB9 => {
            // MEMACC: dispatched to m4a's byte machine, then the tail-call goto on a true branch.
            let mut track = std::mem::take(&mut mp.tracks[t]);
            let r = m4a::ply_memacc(&mut mp.memAccArea, &mut track, rom);
            mp.tracks[t] = track;
            if matches!(r, m4a::MemAccResult::Goto) {
                ply_goto(mp, si, rom, t, false, result);
            }
        }
        0xBA => mp.tracks[t].priority = arg(mp, rom, t),
        0xBB => {
            // TEMPO: value is half the step rate (tempoI = tempoD, tempoU = 0x100).
            let v = u16::from(arg(mp, rom, t));
            mp.tempoI = v * 2;
        }
        0xBC => {
            mp.tracks[t].keyShift = arg(mp, rom, t) as i8;
            mp.tracks[t].flags |= MPT_FLG_PITCHG;
        }
        0xBD => {
            // VOICE: copy the program's ToneData from the voicegroup.
            let program = arg(mp, rom, t) as usize;
            let voicegroup = mp.tone as usize;
            mp.tracks[t].tone = ToneData::read(rom, voicegroup + program * 12);
        }
        0xBE => {
            mp.tracks[t].vol = arg(mp, rom, t);
            mp.tracks[t].flags |= MPT_FLG_VOLCHG;
        }
        0xBF => {
            mp.tracks[t].pan = arg(mp, rom, t).wrapping_sub(0x40) as i8;
            mp.tracks[t].flags |= MPT_FLG_VOLCHG;
        }
        0xC0 => {
            mp.tracks[t].bend = arg(mp, rom, t).wrapping_sub(0x40) as i8;
            mp.tracks[t].flags |= MPT_FLG_PITCHG;
        }
        0xC1 => {
            mp.tracks[t].bendRange = arg(mp, rom, t);
            mp.tracks[t].flags |= MPT_FLG_PITCHG;
        }
        0xC2 => {
            mp.tracks[t].lfoSpeed = arg(mp, rom, t);
            if mp.tracks[t].lfoSpeed == 0 {
                m4a::ClearModM(&mut mp.tracks[t]);
            }
        }
        0xC3 => mp.tracks[t].lfoDelay = arg(mp, rom, t),
        0xC4 => {
            mp.tracks[t].mod_ = arg(mp, rom, t);
            if mp.tracks[t].mod_ == 0 {
                m4a::ClearModM(&mut mp.tracks[t]);
            }
        }
        0xC5 => {
            let v = arg(mp, rom, t);
            if mp.tracks[t].modT != v {
                mp.tracks[t].modT = v;
                mp.tracks[t].flags |= MPT_FLG_VOLCHG | MPT_FLG_PITCHG;
            }
        }
        0xC8 => {
            mp.tracks[t].tune = arg(mp, rom, t).wrapping_sub(0x40) as i8;
            mp.tracks[t].flags |= MPT_FLG_PITCHG;
        }
        0xCC => {
            // PORT: writes a raw GB sound register — consume the 2 operands, no effect.
            mp.tracks[t].cmdPtr += 2;
        }
        0xCD => xcmd(mp, si, rom, t),
        0xCE => end_tie(mp, si, rom, t),
        // FINE and every unassigned jump-table slot stop the track.
        _ => fine(mp, si, t),
    }
}

/// `ply_xcmd` (`m4a.c:1523`): extended commands, dispatched via `gXcmdTable` into the `m4a`
/// `ply_x*` handlers.
fn xcmd(mp: &mut MusicPlayerInfo, si: &mut SoundInfo, rom: &[u8], t: usize) {
    let n = arg(mp, rom, t);
    match n {
        1 => m4a::ply_xwave(&mut mp.tracks[t], rom),
        2 => m4a::ply_xtype(&mut mp.tracks[t], rom),
        4 => m4a::ply_xatta(&mut mp.tracks[t], rom),
        5 => m4a::ply_xdeca(&mut mp.tracks[t], rom),
        6 => m4a::ply_xsust(&mut mp.tracks[t], rom),
        7 => m4a::ply_xrele(&mut mp.tracks[t], rom),
        8 => m4a::ply_xiecv(&mut mp.tracks[t], rom),
        9 => m4a::ply_xiecl(&mut mp.tracks[t], rom),
        10 => m4a::ply_xleng(&mut mp.tracks[t], rom),
        11 => m4a::ply_xswee(&mut mp.tracks[t], rom),
        12 => m4a::ply_xwait(&mut mp.tracks[t], rom),
        13 => m4a::ply_xcmd_0D(&mut mp.tracks[t], rom),
        // gXcmdTable[0] and [3] are `ply_xxx`, which tail-calls gMPlayJumpTable[0] = `ply_fine`.
        0 | 3 => fine(mp, si, t),
        // Out-of-table indices (≥ 14) read past `gXcmdTable` in the C (UB); Optime safely stops
        // the track by clearing its EXIST flag.
        _ => mp.tracks[t].flags &= !MPT_FLG_EXIST,
    }
}

/// `ply_goto` (`m4a_1.s:831`): read the 4-byte target and jump. A backward jump that is the song
/// loop point (`GOTO` / `REPT 0`) reports one loop.
fn ply_goto(
    mp: &mut MusicPlayerInfo,
    si: &mut SoundInfo,
    rom: &[u8],
    t: usize,
    loop_point: bool,
    result: &mut FrameResult,
) {
    let target = read_u32(rom, mp.tracks[t].cmdPtr);
    match ptr_to_offset(target, rom.len()) {
        Some(offset) => {
            if loop_point && offset < mp.tracks[t].cmdPtr {
                result.looped = true;
            }
            mp.tracks[t].cmdPtr = offset;
        }
        None => fine(mp, si, t),
    }
}

/// `ply_fine` (`m4a_1.s:750`): release the track's channels and stop executing it.
fn fine(mp: &mut MusicPlayerInfo, si: &mut SoundInfo, t: usize) {
    let orphan = |c: &mut ChannelHdr| {
        if c.status() & SOUND_CHANNEL_SF_ON != 0 {
            c.set_status(c.status() | SOUND_CHANNEL_SF_STOP);
        }
        c.set_track(None); // RealClearChain detaches the channel from the track
    };
    for i in 0..si.maxChans as usize {
        if si.chans[i].track == Some(t) {
            orphan(&mut ChannelHdr::Ds(&mut si.chans[i]));
        }
    }
    for i in 0..4 {
        if si.cgbChans[i].track == Some(t) {
            orphan(&mut ChannelHdr::Cgb(&mut si.cgbChans[i]));
        }
    }
    mp.tracks[t].flags = 0;
}

/// `ply_endtie` (`m4a_1.s:1818`): release the track's first still-sounding channel playing `key`.
fn end_tie(mp: &mut MusicPlayerInfo, si: &mut SoundInfo, rom: &[u8], t: usize) {
    let byte = read_u8(rom, mp.tracks[t].cmdPtr);
    let key = if byte < 0x80 {
        mp.tracks[t].key = byte;
        mp.tracks[t].cmdPtr += 1;
        byte
    } else {
        mp.tracks[t].key
    };
    // SF_START|SF_ENV set and SF_STOP clear == a sounding, unreleased channel.
    let mask = SOUND_CHANNEL_SF_START | SOUND_CHANNEL_SF_ENV;
    let matching =
        |st: u8, midi: u8| st & mask != 0 && st & SOUND_CHANNEL_SF_STOP == 0 && midi == key;
    for i in 0..si.maxChans as usize {
        let c = &mut si.chans[i];
        if c.track == Some(t) && matching(c.statusFlags, c.midiKey) {
            c.statusFlags |= SOUND_CHANNEL_SF_STOP;
            return;
        }
    }
    for i in 0..4 {
        let c = &mut si.cgbChans[i];
        if c.track == Some(t) && matching(c.statusFlags, c.midiKey) {
            c.statusFlags |= SOUND_CHANNEL_SF_STOP;
            return;
        }
    }
}

/// `ply_lfos` (`m4a_1.s:1887`): the per-step LFO ("mod") tick, folded into the `MPlayMain` step.
fn lfo_step(tr: &mut MusicPlayerTrack) {
    if tr.lfoSpeed == 0 || tr.mod_ == 0 {
        return;
    }
    if tr.lfoDelayC != 0 {
        tr.lfoDelayC -= 1;
        return;
    }
    tr.lfoSpeedC = tr.lfoSpeedC.wrapping_add(tr.lfoSpeed);
    let triangle: i32 = if (tr.lfoSpeedC.wrapping_sub(0x40) as i8) < 0 {
        i32::from(tr.lfoSpeedC as i8)
    } else {
        0x80 - i32::from(tr.lfoSpeedC)
    };
    let value = ((i32::from(tr.mod_) * triangle) >> 6) as i8;
    if value as u8 != tr.modM as u8 {
        tr.modM = value;
        if tr.modT == 0 {
            tr.flags |= MPT_FLG_PITCHG;
        } else {
            tr.flags |= MPT_FLG_VOLCHG;
        }
    }
}

// ===========================================================================================
// ply_note — instrument resolution + channel allocation (m4a_1.s:1538)
// ===========================================================================================

/// `ply_note`: gate/key/velocity parsing, key-split / rhythm resolution, then DirectSound / CGB
/// channel allocation into `SoundInfo` and note setup.
fn ply_note(mp: &mut MusicPlayerInfo, si: &mut SoundInfo, rom: &[u8], t: usize, n: u8) {
    mp.tracks[t].gateTime = CLOCK_TABLE[n as usize];

    let mut byte = read_u8(rom, mp.tracks[t].cmdPtr);
    if byte < 0x80 {
        mp.tracks[t].key = byte;
        mp.tracks[t].cmdPtr += 1;
        byte = read_u8(rom, mp.tracks[t].cmdPtr);
        if byte < 0x80 {
            mp.tracks[t].velocity = byte;
            mp.tracks[t].cmdPtr += 1;
            byte = read_u8(rom, mp.tracks[t].cmdPtr);
            if byte < 0x80 {
                mp.tracks[t].gateTime = mp.tracks[t].gateTime.wrapping_add(byte);
                mp.tracks[t].cmdPtr += 1;
            }
        }
    }

    // --- resolve the tone (key-split / rhythm sub-voicegroups) ---
    let played_key = mp.tracks[t].key;
    let mut tone = mp.tracks[t].tone;
    let mut resolved_key = played_key;
    let mut rhythm_pan: i8 = 0;
    if tone.kind & (TONEDATA_TYPE_RHY | TONEDATA_TYPE_SPL) != 0 {
        let index = if tone.kind & TONEDATA_TYPE_SPL != 0 {
            let table = u32::from_le_bytes([tone.attack, tone.decay, tone.sustain, tone.release]);
            match ptr_to_offset(table, rom.len()) {
                Some(o) => read_u8(rom, o + played_key as usize),
                None => return,
            }
        } else {
            played_key
        };
        let group = match ptr_to_offset(tone.wav, rom.len()) {
            Some(o) => o,
            None => return,
        };
        let sub = ToneData::read(rom, group + index as usize * 12);
        if sub.kind & (TONEDATA_TYPE_RHY | TONEDATA_TYPE_SPL) != 0 {
            return;
        }
        if tone.kind & TONEDATA_TYPE_RHY != 0 {
            resolved_key = sub.key;
            if sub.pan_sweep & 0x80 != 0 {
                rhythm_pan = (sub.pan_sweep.wrapping_sub(TONEDATA_P_S_PAN) as i8).wrapping_mul(2);
            }
        }
        tone = sub;
    }

    let priority = mp.tracks[t].priority.saturating_add(mp.priority);
    let cgb = tone.kind & TONEDATA_TYPE_CGB;

    // --- allocation ---
    let slot = if cgb != 0 {
        let idx = (cgb - 1) as usize;
        let ex = &si.cgbChans[idx];
        if ex.statusFlags & SOUND_CHANNEL_SF_ON != 0 && ex.statusFlags & SOUND_CHANNEL_SF_STOP == 0
        {
            if ex.priority > priority {
                return;
            }
            if ex.priority == priority && ex.track.is_some_and(|tt| tt < t) {
                return;
            }
        }
        Slot::Cgb(idx)
    } else {
        match alloc_direct_sound(si, priority, t) {
            Some(i) => Slot::Ds(i),
            None => return,
        }
    };

    // --- note setup (shared) ---
    let lfo_delay = mp.tracks[t].lfoDelay;
    mp.tracks[t].lfoDelayC = lfo_delay;
    if lfo_delay != 0 {
        m4a::ClearModM(&mut mp.tracks[t]);
    }
    m4a::TrkVolPitSet(&mut mp.tracks[t]);
    let tr = &mp.tracks[t];
    let (right, left) = chn_vol_set(tr.velocity, rhythm_pan, tr.volMR, tr.volML);
    let key2 = add_key(resolved_key, tr.keyM);
    let gate = tr.gateTime;
    let velocity = tr.velocity;
    let midi_key = tr.key;
    let echo_vol = tr.pseudoEchoVolume;
    let echo_len = tr.pseudoEchoLength;
    let pit_m = tr.pitM;

    match slot {
        Slot::Ds(i) => {
            let c = &mut si.chans[i];
            *c = SoundChannel {
                statusFlags: SOUND_CHANNEL_SF_START,
                type_: tone.kind,
                rightVolume: right,
                leftVolume: left,
                attack: tone.attack,
                decay: tone.decay,
                sustain: tone.sustain,
                release: tone.release,
                key: resolved_key,
                midiKey: midi_key,
                velocity,
                priority,
                rhythmPan: rhythm_pan as u8,
                gateTime: gate,
                pseudoEchoVolume: echo_vol,
                pseudoEchoLength: echo_len,
                wav: tone.wav,
                frequency: midi_key_to_freq(wav_freq(rom, tone.wav), key2, pit_m),
                track: Some(t),
                ..SoundChannel::default()
            };
        }
        Slot::Cgb(i) => {
            let c = &mut si.cgbChans[i];
            *c = CgbChannel {
                statusFlags: SOUND_CHANNEL_SF_START,
                type_: tone.kind,
                rightVolume: right,
                leftVolume: left,
                attack: tone.attack,
                decay: tone.decay,
                sustain: tone.sustain,
                release: tone.release,
                key: resolved_key,
                midiKey: midi_key,
                velocity,
                priority,
                rhythmPan: rhythm_pan as u8,
                gateTime: gate,
                pseudoEchoVolume: echo_vol,
                pseudoEchoLength: echo_len,
                length: tone.length,
                sweep: cgb_sweep(&tone),
                wavePointer: tone.wav,
                frequency: midi_key_to_cgb_freq(cgb, key2, pit_m),
                track: Some(t),
                ..CgbChannel::default()
            };
        }
    }

    mp.tracks[t].flags &= 0xF0;
}

/// The CGB `sweep` byte a note-on stores (`ply_note`): the tone's `pan_sweep` unless bit 7 is set
/// or bits 4–6 are clear, in which case a fixed `8`.
fn cgb_sweep(tone: &ToneData) -> u8 {
    if tone.pan_sweep & 0x80 == 0 && tone.pan_sweep & 0x70 != 0 {
        tone.pan_sweep
    } else {
        8
    }
}

/// The DirectSound channel allocator from `ply_note` (`m4a_1.s:1677`): the first free channel wins;
/// otherwise steal the lowest-priority channel, preferring releasing ones, ties to the latest
/// track. A new note that loses every comparison is dropped.
fn alloc_direct_sound(si: &SoundInfo, priority: u8, track: usize) -> Option<usize> {
    let mut best: Option<usize> = None;
    let mut best_priority = priority;
    let mut best_track = track;
    let mut found_releasing = false;

    for i in 0..si.maxChans as usize {
        let c = &si.chans[i];
        if c.statusFlags & SOUND_CHANNEL_SF_ON == 0 {
            return Some(i); // free channel
        }
        if c.statusFlags & SOUND_CHANNEL_SF_STOP != 0 {
            // releasing
            if !found_releasing {
                found_releasing = true;
                best_priority = c.priority;
                best_track = c.track.unwrap_or(track);
                best = Some(i);
                continue;
            }
        } else if found_releasing {
            continue; // held channels ignored once a releasing one is found
        }
        let c_track = c.track.unwrap_or(track);
        if c.priority < best_priority {
            best_priority = c.priority;
            best_track = c_track;
            best = Some(i);
        } else if c.priority == best_priority && c_track >= best_track {
            best_track = c_track;
            best = Some(i);
        }
    }
    best
}

/// The end-of-frame `MPT_FLG_VOLCHG`/`PITCHG` pass (`MPlayMain` `_081DD9C4`): recompute track
/// mixers ([`m4a::TrkVolPitSet`]) and refresh every channel's volume scaling and pitch.
fn refresh_changed_tracks(mp: &mut MusicPlayerInfo, si: &mut SoundInfo, rom: &[u8]) {
    for t in 0..mp.tracks.len() {
        if !exists(&mp.tracks[t]) {
            continue;
        }
        let flags = mp.tracks[t].flags;
        let vol_chg = flags & MPT_FLG_VOLCHG != 0;
        let pit_chg = flags & MPT_FLG_PITCHG != 0;
        if !vol_chg && !pit_chg {
            continue;
        }
        m4a::TrkVolPitSet(&mut mp.tracks[t]);
        let tr = &mp.tracks[t];
        let (vol_mr, vol_ml, key_m, pit_m) = (tr.volMR, tr.volML, tr.keyM, tr.pitM);

        for i in 0..si.maxChans as usize {
            if si.chans[i].track != Some(t) {
                continue;
            }
            let c = &mut si.chans[i];
            if c.statusFlags & SOUND_CHANNEL_SF_ON == 0 {
                c.track = None;
                continue;
            }
            if vol_chg {
                let (r, l) = chn_vol_set(c.velocity, c.rhythmPan as i8, vol_mr, vol_ml);
                c.rightVolume = r;
                c.leftVolume = l;
            }
            if pit_chg {
                let key2 = add_key(c.key, key_m);
                c.frequency = midi_key_to_freq(wav_freq(rom, c.wav), key2, pit_m);
            }
        }
        for i in 0..4 {
            if si.cgbChans[i].track != Some(t) {
                continue;
            }
            let c = &mut si.cgbChans[i];
            if c.statusFlags & SOUND_CHANNEL_SF_ON == 0 {
                c.track = None;
                continue;
            }
            if vol_chg {
                let (r, l) = chn_vol_set(c.velocity, c.rhythmPan as i8, vol_mr, vol_ml);
                c.rightVolume = r;
                c.leftVolume = l;
                c.modify |= CGB_CHANNEL_MO_VOL;
            }
            if pit_chg {
                let key2 = add_key(c.key, key_m);
                c.frequency = midi_key_to_cgb_freq(c.type_ & TONEDATA_TYPE_CGB, key2, pit_m);
                c.modify |= CGB_CHANNEL_MO_PIT;
            }
        }

        mp.tracks[t].flags &= 0xF0;
    }
}

// ===========================================================================================
// SoundMain — the envelope pass (m4a_1.s:88 SoundMainRAM + the CgbSound call)
// ===========================================================================================

/// `SoundMain`: step the DirectSound envelopes (`SoundMainRAM`) and the CGB envelopes
/// ([`m4a::CgbSound`]). The PCM/FIFO mixing half of the reference is Optime's synth's job.
pub fn SoundMain(si: &mut SoundInfo) {
    sound_main_ram(si);
    m4a::CgbSound(si);
}

/// `SoundMainRAM`'s per-DirectSound-channel envelope step (`m4a_1.s:172`). The mixing loop that
/// follows in the reference is a backend seam (Optime's synth renders the PCM).
fn sound_main_ram(si: &mut SoundInfo) {
    let master = si.masterVolume as u32;
    for i in 0..si.maxChans as usize {
        let c = &mut si.chans[i];
        if c.statusFlags & SOUND_CHANNEL_SF_ON == 0 {
            continue;
        }
        if direct_sound_env(c) {
            c.statusFlags = 0;
            continue;
        }
        // _081DD006: master-volume scaling then per-side velocity/pan volumes.
        let uvol = ((master + 1) * c.envelopeVolume as u32) >> 4;
        c.envelopeVolumeRight = ((c.rightVolume as u32 * uvol) >> 8) as u8;
        c.envelopeVolumeLeft = ((c.leftVolume as u32 * uvol) >> 8) as u8;
    }
}

/// One DirectSound envelope frame. Returns `true` if the channel shut off (`statusFlags` cleared).
fn direct_sound_env(c: &mut SoundChannel) -> bool {
    if c.statusFlags & SOUND_CHANNEL_SF_START != 0 {
        if c.statusFlags & SOUND_CHANNEL_SF_STOP != 0 {
            return true;
        }
        c.statusFlags = SOUND_CHANNEL_SF_ENV_ATTACK;
        c.envelopeVolume = 0;
        // fall into the attack step.
        let mut env = c.attack as u32;
        if env >= 0xFF {
            env = 0xFF;
            c.statusFlags -= 1;
        }
        c.envelopeVolume = env as u8;
        return false;
    }

    let mut env = c.envelopeVolume as u32;
    if c.statusFlags & SOUND_CHANNEL_SF_IEC != 0 {
        let orig = c.pseudoEchoLength;
        c.pseudoEchoLength = orig.wrapping_sub(1);
        if orig <= 1 {
            return true;
        }
    } else if c.statusFlags & SOUND_CHANNEL_SF_STOP != 0 {
        env = (env * c.release as u32) >> 8;
        if env <= c.pseudoEchoVolume as u32 {
            if c.pseudoEchoVolume == 0 {
                return true;
            }
            env = c.pseudoEchoVolume as u32;
            c.statusFlags |= SOUND_CHANNEL_SF_IEC;
        }
    } else if c.statusFlags & SOUND_CHANNEL_SF_ENV == SOUND_CHANNEL_SF_ENV_DECAY {
        env = (env * c.decay as u32) >> 8;
        let sustain = c.sustain as u32;
        if env <= sustain {
            if sustain == 0 {
                if c.pseudoEchoVolume == 0 {
                    return true;
                }
                env = c.pseudoEchoVolume as u32;
                c.statusFlags |= SOUND_CHANNEL_SF_IEC;
            } else {
                env = sustain;
                c.statusFlags -= 1; // decay -> sustain
            }
        }
    } else if c.statusFlags & SOUND_CHANNEL_SF_ENV == SOUND_CHANNEL_SF_ENV_ATTACK {
        env += c.attack as u32;
        if env >= 0xFF {
            env = 0xFF;
            c.statusFlags -= 1;
        }
    }
    c.envelopeVolume = env as u8;
    false
}

// ===========================================================================================
// A tiny read/write shim over the two channel structs so the per-track loops stay generic.
// ===========================================================================================

enum ChannelHdr<'a> {
    Ds(&'a mut SoundChannel),
    Cgb(&'a mut CgbChannel),
}

impl ChannelHdr<'_> {
    fn status(&self) -> u8 {
        match self {
            ChannelHdr::Ds(c) => c.statusFlags,
            ChannelHdr::Cgb(c) => c.statusFlags,
        }
    }
    fn set_status(&mut self, v: u8) {
        match self {
            ChannelHdr::Ds(c) => c.statusFlags = v,
            ChannelHdr::Cgb(c) => c.statusFlags = v,
        }
    }
    fn gate_time(&self) -> u8 {
        match self {
            ChannelHdr::Ds(c) => c.gateTime,
            ChannelHdr::Cgb(c) => c.gateTime,
        }
    }
    fn set_gate_time(&mut self, v: u8) {
        match self {
            ChannelHdr::Ds(c) => c.gateTime = v,
            ChannelHdr::Cgb(c) => c.gateTime = v,
        }
    }
    fn set_track(&mut self, v: Option<usize>) {
        match self {
            ChannelHdr::Ds(c) => c.track = v,
            ChannelHdr::Cgb(c) => c.track = v,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `SOUND_CHANNEL_SF_*` raw statuses used by the oracles (from `m4a_internal.h`).
    const SF_START: u8 = 0x80;
    const SF_STOP: u8 = 0x40;
    const SF_IEC: u8 = 0x04;
    const SF_ENV: u8 = 0x03;
    const SF_ENV_DECAY: u8 = 0x02;
    const SF_ENV_ATTACK: u8 = 0x03;

    /// Direct transcription of `ChnVolSetAsm` (`m4a_1.s:1508`).
    fn c_chn_vol_set(velocity: u8, rhythm_pan: i8, vol_mr: u8, vol_ml: u8) -> (u8, u8) {
        let mut right =
            ((0x80 + i32::from(rhythm_pan)) * i32::from(velocity) * i32::from(vol_mr)) >> 14;
        if right > 0xFF {
            right = 0xFF;
        }
        let mut left =
            ((0x7F - i32::from(rhythm_pan)) * i32::from(velocity) * i32::from(vol_ml)) >> 14;
        if left > 0xFF {
            left = 0xFF;
        }
        (right as u8, left as u8)
    }

    #[test]
    fn chn_vol_set_matches_pokeemerald() {
        for velocity in [0u8, 1, 64, 100, 127, 255] {
            for pan in [-128i8, -64, -2, 0, 2, 63, 126] {
                for vol_mr in [0u8, 1, 90, 178, 255] {
                    for vol_ml in [0u8, 1, 90, 178, 255] {
                        assert_eq!(
                            chn_vol_set(velocity, pan, vol_mr, vol_ml),
                            c_chn_vol_set(velocity, pan, vol_mr, vol_ml),
                            "velocity={velocity} pan={pan} volMR={vol_mr} volML={vol_ml}"
                        );
                    }
                }
            }
        }
    }

    /// The oracle DirectSound channel: raw `statusFlags` state (`SoundMainRAM`, `m4a_1.s:172`).
    struct COracle {
        status: u8,
        env: u8,
        adsr: [u8; 4],
        echo_volume: u8,
        echo_length: u8,
    }

    /// Direct transcription of the `SoundMainRAM` envelope section. Returns `false` once off.
    fn c_sound_main_ram_env(c: &mut COracle) -> bool {
        let [attack, decay, sustain, release] = c.adsr;
        if c.status & SF_START != 0 {
            if c.status & SF_STOP != 0 {
                c.status = 0;
                return false;
            }
            c.status = SF_ENV_ATTACK;
            let env = u32::from(attack);
            if env >= 0xFF {
                c.env = 0xFF;
                c.status -= 1;
            } else {
                c.env = env as u8;
            }
            return true;
        }
        let mut env = u32::from(c.env);
        if c.status & SF_IEC != 0 {
            let orig = c.echo_length;
            c.echo_length = orig.wrapping_sub(1);
            if orig <= 1 {
                c.status = 0;
                return false;
            }
        } else if c.status & SF_STOP != 0 {
            env = (env * u32::from(release)) >> 8;
            if env <= u32::from(c.echo_volume) {
                if c.echo_volume == 0 {
                    c.status = 0;
                    return false;
                }
                env = u32::from(c.echo_volume);
                c.status |= SF_IEC;
            }
        } else if c.status & SF_ENV == SF_ENV_DECAY {
            env = (env * u32::from(decay)) >> 8;
            if env <= u32::from(sustain) {
                env = u32::from(sustain);
                if sustain == 0 {
                    if c.echo_volume == 0 {
                        c.status = 0;
                        return false;
                    }
                    env = u32::from(c.echo_volume);
                    c.status |= SF_IEC;
                } else {
                    c.status -= 1;
                }
            }
        } else if c.status & SF_ENV == SF_ENV_ATTACK {
            env += u32::from(attack);
            if env >= 0xFF {
                env = 0xFF;
                c.status -= 1;
            }
        }
        c.env = env as u8;
        true
    }

    #[test]
    fn direct_sound_envelope_matches_pokeemerald() {
        for attack in [0u8, 1, 9, 80, 255] {
            for decay in [0u8, 128, 235, 255] {
                for sustain in [0u8, 77, 255] {
                    for release in [0u8, 128, 245] {
                        for (echo_v, echo_l) in [(0u8, 0u8), (40, 1), (40, 5), (40, 0xC0)] {
                            for release_frame in [0u32, 1, 3, 25] {
                                ds_env_scenario(
                                    [attack, decay, sustain, release],
                                    echo_v,
                                    echo_l,
                                    release_frame,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    fn ds_env_scenario(adsr: [u8; 4], echo_v: u8, echo_l: u8, release_frame: u32) {
        let label = format!("adsr={adsr:?} echo=({echo_v},{echo_l}) rel={release_frame}");
        let mut ours = SoundChannel {
            statusFlags: SF_START,
            attack: adsr[0],
            decay: adsr[1],
            sustain: adsr[2],
            release: adsr[3],
            pseudoEchoVolume: echo_v,
            pseudoEchoLength: echo_l,
            ..SoundChannel::default()
        };
        let mut oracle = COracle {
            status: SF_START,
            env: 0,
            adsr,
            echo_volume: echo_v,
            echo_length: echo_l,
        };
        for frame in 0..600u32 {
            if frame == release_frame {
                ours.statusFlags |= SF_STOP;
                oracle.status |= SF_STOP;
            }
            let off = direct_sound_env(&mut ours);
            if off {
                ours.statusFlags = 0;
            }
            let oracle_alive = c_sound_main_ram_env(&mut oracle);
            assert_eq!(!off, oracle_alive, "{label}: alive @ {frame}");
            if off {
                return;
            }
            assert_eq!(ours.envelopeVolume, oracle.env, "{label}: env @ {frame}");
        }
    }

    /// The oracle LFO state (`ply_lfos`, `m4a_1.s:1887`).
    #[derive(Default)]
    struct CLfo {
        speed_c: u8,
        delay_c: u8,
        mod_m: i8,
        flag_pitch: bool,
        flag_volume: bool,
    }

    fn c_lfo_step(t: &mut CLfo, speed: u8, depth: u8, mod_t: u8) {
        if speed == 0 || depth == 0 {
            return;
        }
        if t.delay_c != 0 {
            t.delay_c -= 1;
            return;
        }
        t.speed_c = t.speed_c.wrapping_add(speed);
        let triangle: i32 = if (t.speed_c.wrapping_sub(0x40) as i8) < 0 {
            i32::from(t.speed_c as i8)
        } else {
            0x80 - i32::from(t.speed_c)
        };
        let value = (i32::from(depth) * triangle) >> 6;
        if value as u8 != t.mod_m as u8 {
            t.mod_m = value as i8;
            if mod_t == 0 {
                t.flag_pitch = true;
            } else {
                t.flag_volume = true;
            }
        }
    }

    #[test]
    fn lfo_step_matches_pokeemerald() {
        for speed in [0u8, 1, 22, 64, 130, 255] {
            for depth in [0u8, 1, 12, 127, 255] {
                for delay in [0u8, 3] {
                    for mod_t in [0u8, 1] {
                        let mut tr = MusicPlayerTrack {
                            lfoSpeed: speed,
                            mod_: depth,
                            lfoDelayC: delay,
                            modT: mod_t,
                            ..MusicPlayerTrack::default()
                        };
                        let mut oracle = CLfo {
                            delay_c: delay,
                            ..CLfo::default()
                        };
                        for stp in 0..600u32 {
                            tr.flags = 0;
                            oracle.flag_pitch = false;
                            oracle.flag_volume = false;
                            lfo_step(&mut tr);
                            c_lfo_step(&mut oracle, speed, depth, mod_t);
                            assert_eq!(
                                (tr.modM, tr.lfoSpeedC, tr.lfoDelayC),
                                (oracle.mod_m, oracle.speed_c, oracle.delay_c),
                                "speed={speed} depth={depth} delay={delay} modT={mod_t} step={stp}"
                            );
                            assert_eq!(
                                (
                                    tr.flags & MPT_FLG_PITCHG != 0,
                                    tr.flags & MPT_FLG_VOLCHG != 0
                                ),
                                (oracle.flag_pitch, oracle.flag_volume),
                                "flags: speed={speed} depth={depth} delay={delay} modT={mod_t}"
                            );
                        }
                    }
                }
            }
        }
    }
}

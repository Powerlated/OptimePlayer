//! Audio-only ROM extraction: produces an image of a GBA ROM where every byte that the MP2K
//! engine cannot reach from the song table is zeroed, so the audio data can be shipped (e.g.
//! in `demos/`) without bundling the game's code, sprites, text, or anything else.
//!
//! All kept bytes stay at their original offsets — MP2K data is full of absolute
//! `0x08000000`-based pointers — so the extracted image plays bit-identically to the original
//! ROM. The reachable set is computed by:
//!
//! - walking every song's track bytecode statically (the command set has fixed argument
//!   lengths, so the walk follows `GOTO`/`PATT`/`REPT`/`MEMACC` jump targets without running
//!   the song),
//! - marking every voicegroup reachable from the song headers, all 128 entries deep,
//!   following key-split tables and rhythm/key-split sub-voicegroups,
//! - marking the sample data behind every DirectSound `WaveData` and CGB programmable wave.

use std::collections::HashSet;

use super::rom::{ptr_to_offset, GbaRom};
use super::voice::{ToneData, WaveData};
use crate::devices::SampleDcStat;
use crate::util::{read_u32, read_u8};

/// `TONEDATA_TYPE_*` bits (mirrors `voice.rs`).
const TYPE_CGB: u8 = 0x07;
const TYPE_CMP_REV: u8 = 0x30;
const TYPE_SPL: u8 = 0x40;
const TYPE_RHY: u8 = 0x80;

/// Voicegroups are arrays without a length; the engine indexes them with 7-bit programs and
/// keys, so 128 entries covers everything reachable.
const VOICEGROUP_ENTRIES: usize = 128;

/// Key-split / rhythm sub-voicegroups can nest in principle; real data uses one level.
const MAX_GROUP_DEPTH: usize = 4;

/// Builds the audio-only image of `rom`: every byte unreachable from the song table is zeroed
/// and the image is truncated past the last reachable byte (4-byte aligned). The result still
/// parses as a GBA ROM with the same song table and plays identically.
pub fn extract_audio(rom: &GbaRom) -> Vec<u8> {
    let data: &[u8] = &rom.data;
    let mut marks = Marks::new(data.len());

    // The GBA header's identity block (title/codes + the fixed 0x96 check byte) so the result
    // is still recognized as a GBA ROM. Everything before it (entry point, logo bitmap) stays
    // zeroed.
    marks.mark(0xA0, 0xC0);

    // The song table itself.
    let song_count = rom.song_count();
    marks.mark(rom.song_table, rom.song_table + song_count * 8);

    let mut groups = GroupWalker::new();
    for id in 0..song_count as u32 {
        // Placeholder entries: keep the trackCount-0 header byte row (the table needs it to
        // stay enumerable) — it is 8 zero-ish bytes at most.
        let entry = rom.song_table + id as usize * 8;
        if let Some(header_off) = ptr_to_offset(read_u32(data, entry), data.len()) {
            marks.mark(header_off, header_off + 8);
        }
        let Some(header) = rom.song_header(id) else {
            continue;
        };
        marks.mark(
            header.offset,
            header.offset + 8 + header.track_count as usize * 4,
        );
        groups.walk(data, &mut marks, header.voicegroup, 0);
        for i in 0..header.track_count as usize {
            let ptr = read_u32(data, header.offset + 8 + i * 4);
            if let Some(start) = ptr_to_offset(ptr, data.len()) {
                walk_track(data, &mut marks, start);
            }
        }
    }

    marks.into_image(data)
}

/// The reachable-byte set.
struct Marks {
    kept: Vec<bool>,
}

impl Marks {
    fn new(len: usize) -> Self {
        Marks {
            kept: vec![false; len],
        }
    }

    /// Marks `[start, end)`, clamped to the image.
    fn mark(&mut self, start: usize, end: usize) {
        let end = end.min(self.kept.len());
        if start < end {
            self.kept[start..end].fill(true);
        }
    }

    /// Marks a DirectSound wave (header + PCM) and, defensively, the 16 bytes a CGB
    /// programmable wave would occupy — `xWAVE` overrides don't say which kind follows.
    fn mark_wave(&mut self, rom: &[u8], wav_ptr: u32) {
        if let Some(wave) = WaveData::read(rom, wav_ptr) {
            let start = wave.data_offset - 16;
            self.mark(start, wave.data_offset + wave.size as usize);
        } else if let Some(off) = ptr_to_offset(wav_ptr, rom.len()) {
            self.mark(off, off + 16);
        }
    }

    /// The final image: marked bytes copied, everything else zero, truncated past the last
    /// marked byte (4-byte aligned).
    fn into_image(self, rom: &[u8]) -> Vec<u8> {
        let last = self.kept.iter().rposition(|&k| k).map_or(0, |i| i + 1);
        let len = (last + 3) & !3;
        let mut out = vec![0u8; len.min(rom.len())];
        for (i, &keep) in self.kept[..out.len()].iter().enumerate() {
            if keep {
                out[i] = rom[i];
            }
        }
        out
    }
}

/// Marks every voicegroup reachable from `group`: all 128 ToneData entries, key-split tables,
/// sub-voicegroups, and the wave data behind DirectSound / CGB-wave entries.
struct GroupWalker {
    visited: HashSet<usize>,
}

impl GroupWalker {
    fn new() -> Self {
        GroupWalker {
            visited: HashSet::new(),
        }
    }

    fn walk(&mut self, rom: &[u8], marks: &mut Marks, group: usize, depth: usize) {
        if depth > MAX_GROUP_DEPTH || !self.visited.insert(group) {
            return;
        }
        for i in 0..VOICEGROUP_ENTRIES {
            let off = group + i * 12;
            if off + 12 > rom.len() {
                break;
            }
            marks.mark(off, off + 12);
            let tone = ToneData::read(rom, off);
            if tone.kind & (TYPE_RHY | TYPE_SPL) != 0 {
                if tone.kind & TYPE_SPL != 0 {
                    // The key-split table pointer lives in the ADSR word.
                    let table =
                        u32::from_le_bytes([tone.attack, tone.decay, tone.sustain, tone.release]);
                    if let Some(table_off) = ptr_to_offset(table, rom.len()) {
                        marks.mark(table_off, table_off + 128);
                    }
                }
                if let Some(sub) = ptr_to_offset(tone.wav, rom.len()) {
                    self.walk(rom, marks, sub, depth + 1);
                }
            } else {
                match tone.kind & TYPE_CGB {
                    // DirectSound PCM (uncompressed only — same constraint as playback).
                    0 if tone.kind & TYPE_CMP_REV == 0 => {
                        if let Some(wave) = WaveData::read(rom, tone.wav) {
                            marks
                                .mark(wave.data_offset - 16, wave.data_offset + wave.size as usize);
                        }
                    }
                    // CGB programmable wave: 32 packed 4-bit samples.
                    3 => {
                        if let Some(woff) = ptr_to_offset(tone.wav, rom.len()) {
                            marks.mark(woff, woff + 16);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

/// Statically walks one track's bytecode from `start`, marking every byte the interpreter can
/// fetch. Argument lengths mirror `Mp2kSequencer::execute_command` exactly, including running
/// status; jump targets are walked as further branches.
fn walk_track(rom: &[u8], marks: &mut Marks, start: usize) {
    // Branches are (offset, running_status) pairs; the status decides how a data byte
    // (< 0x80) is consumed, so it is part of the visited key.
    let mut work: Vec<(usize, u8)> = vec![(start, 0)];
    let mut visited: HashSet<(usize, u8)> = HashSet::new();

    while let Some((mut p, mut status)) = work.pop() {
        loop {
            if p >= rom.len() || !visited.insert((p, status)) {
                break;
            }
            let byte = read_u8(rom, p);
            let cmd = if byte < 0x80 {
                status // running status: the byte itself is the first argument
            } else {
                marks.mark(p, p + 1);
                p += 1;
                if byte >= 0xBD {
                    status = byte;
                }
                byte
            };

            match cmd {
                // Notes (and TIE): up to three optional bytes < 0x80 (key, velocity, gate
                // extension).
                0xCF..=0xFF => {
                    for _ in 0..3 {
                        if read_u8(rom, p) < 0x80 && p < rom.len() {
                            marks.mark(p, p + 1);
                            p += 1;
                        } else {
                            break;
                        }
                    }
                }
                // W00..W96 rests.
                0x80..=0xB0 => {}
                // GOTO: unconditional jump — branch continues at the target only.
                0xB2 => {
                    follow_target(rom, marks, &mut work, status, p);
                    break;
                }
                // PATT: call; walk the pattern and continue past the call (PEND returns).
                0xB3 => {
                    follow_target(rom, marks, &mut work, status, p);
                    p += 4;
                }
                // PEND: returns to the caller (whose branch already continues past its PATT);
                // also fall through for level-0 execution.
                0xB4 => {}
                // REPT: count byte + target; both paths are walked.
                0xB5 => {
                    marks.mark(p, p + 1);
                    p += 1;
                    follow_target(rom, marks, &mut work, status, p);
                    p += 4;
                }
                // MEMACC: op/addr/data, plus a jump target for the conditional ops (6..=17).
                0xB9 => {
                    let op = read_u8(rom, p);
                    marks.mark(p, p + 3);
                    p += 3;
                    if (6..=17).contains(&op) {
                        follow_target(rom, marks, &mut work, status, p);
                        p += 4;
                    }
                }
                // One-operand controls: PRIO TEMPO KEYSH VOICE VOL PAN BEND BENDR LFOS LFODL
                // MOD MODT TUNE.
                0xBA..=0xC5 | 0xC8 => {
                    marks.mark(p, p + 1);
                    p += 1;
                }
                // PORT: two raw register operands.
                0xCC => {
                    marks.mark(p, p + 2);
                    p += 2;
                }
                // XCMD: sub-command byte selects the operand length.
                0xCD => {
                    let n = read_u8(rom, p);
                    marks.mark(p, p + 1);
                    p += 1;
                    match n {
                        // xWAVE: a wave pointer — mark the wave data it references too.
                        1 => {
                            let wav = read_u32(rom, p);
                            marks.mark(p, p + 4);
                            p += 4;
                            marks.mark_wave(rom, wav);
                        }
                        13 => {
                            marks.mark(p, p + 4);
                            p += 4;
                        }
                        12 => {
                            marks.mark(p, p + 2);
                            p += 2;
                        }
                        2 | 4..=11 => {
                            marks.mark(p, p + 1);
                            p += 1;
                        }
                        // Unknown XCMDs stop the track.
                        _ => break,
                    }
                }
                // EOT: optional explicit key byte.
                0xCE => {
                    if read_u8(rom, p) < 0x80 && p < rom.len() {
                        marks.mark(p, p + 1);
                        p += 1;
                    }
                }
                // FINE, the unassigned command slots, and a data byte with no running status:
                // the track stops here.
                _ => break,
            }
        }
    }
}

/// Marks the 4-byte pointer at `p` and queues its target as a new branch.
fn follow_target(rom: &[u8], marks: &mut Marks, work: &mut Vec<(usize, u8)>, status: u8, p: usize) {
    marks.mark(p, p + 4);
    if let Some(target) = ptr_to_offset(read_u32(rom, p), rom.len()) {
        work.push((target, status));
    }
}

/// DC-offset stats for every DirectSound sample reachable from song `song_id`'s voicegroup,
/// deduped by wave address and sorted by DC shift (most shifted first). Mirrors playback: each
/// sample is decoded the same way [`super::player`] does, and the DC shift recorded is exactly
/// what the player subtracts.
pub fn sample_dc_stats(rom: &GbaRom, song_id: u32) -> Vec<SampleDcStat> {
    let data: &[u8] = &rom.data;
    let Some(header) = rom.song_header(song_id) else {
        return Vec::new();
    };
    let mut addrs = Vec::new();
    let mut visited = HashSet::new();
    collect_directsound_waves(data, header.voicegroup, 0, &mut visited, &mut addrs);

    let mut seen = HashSet::new();
    let mut stats = Vec::new();
    for wav_addr in addrs {
        if !seen.insert(wav_addr) {
            continue; // a wave referenced by several voices is one sample
        }
        let Some(wav) = WaveData::read(data, wav_addr) else {
            continue;
        };
        let raw = &data[wav.data_offset..wav.data_offset + wav.size as usize];
        let pcm = crate::waveform::decode_pcm8(raw);
        if pcm.is_empty() {
            continue;
        }
        let mean = pcm.iter().map(|&v| f64::from(v)).sum::<f64>() / pcm.len() as f64;
        stats.push(SampleDcStat {
            label: format!("0x{wav_addr:08X}"),
            dc_shift: mean.abs() as f32,
            length: pcm.len(),
            sample_rate: f64::from(wav.freq) / 1024.0,
        });
    }
    stats.sort_by(|a, b| b.dc_shift.total_cmp(&a.dc_shift));
    stats
}

/// Collects the ROM addresses of every uncompressed DirectSound wave reachable from voicegroup
/// `group` (following key-split / rhythm sub-voicegroups), the same traversal [`GroupWalker`]
/// uses for the audio-only image.
fn collect_directsound_waves(
    rom: &[u8],
    group: usize,
    depth: usize,
    visited: &mut HashSet<usize>,
    out: &mut Vec<u32>,
) {
    if depth > MAX_GROUP_DEPTH || !visited.insert(group) {
        return;
    }
    for i in 0..VOICEGROUP_ENTRIES {
        let off = group + i * 12;
        if off + 12 > rom.len() {
            break;
        }
        let tone = ToneData::read(rom, off);
        if tone.kind & (TYPE_RHY | TYPE_SPL) != 0 {
            if let Some(sub) = ptr_to_offset(tone.wav, rom.len()) {
                collect_directsound_waves(rom, sub, depth + 1, visited, out);
            }
        } else if tone.kind & TYPE_CGB == 0 && tone.kind & TYPE_CMP_REV == 0 {
            // An uncompressed DirectSound PCM voice (same constraint as playback).
            out.push(tone.wav);
        }
    }
}

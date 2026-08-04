use std::collections::HashSet;

use super::m4a::{ToneData, WaveData};
use super::rom::{GbaRom, ptr_to_offset};
use crate::devices::WaveformDcStat;
use crate::util::{read_u8, read_u32};

const TYPE_CGB: u8 = 0x07;
const TYPE_CMP_REV: u8 = 0x30;
const TYPE_SPL: u8 = 0x40;
const TYPE_RHY: u8 = 0x80;

const VOICEGROUP_ENTRIES: usize = 128;

const MAX_GROUP_DEPTH: usize = 4;

pub fn extract_audio(rom: &GbaRom) -> Vec<u8> {
    let data: &[u8] = &rom.data;
    let mut marks = Marks::new(data.len());

    marks.mark(0xA0, 0xC0);

    let song_count = rom.song_count();
    marks.mark(rom.song_table, rom.song_table + song_count * 8);

    let mut groups = GroupWalker::new();
    for id in 0..song_count as u32 {
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

struct Marks {
    kept: Vec<bool>,
}

impl Marks {
    fn new(len: usize) -> Self {
        Marks {
            kept: vec![false; len],
        }
    }

    fn mark(&mut self, start: usize, end: usize) {
        let end = end.min(self.kept.len());
        if start < end {
            self.kept[start..end].fill(true);
        }
    }

    fn mark_wave(&mut self, rom: &[u8], wav_ptr: u32) {
        if let Some(wave) = WaveData::read(rom, wav_ptr) {
            let start = wave.data - 16;
            self.mark(start, wave.data + wave.size as usize);
        } else if let Some(off) = ptr_to_offset(wav_ptr, rom.len()) {
            self.mark(off, off + 16);
        }
    }

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
                    0 if tone.kind & TYPE_CMP_REV == 0 => {
                        if let Some(wave) = WaveData::read(rom, tone.wav) {
                            marks.mark(wave.data - 16, wave.data + wave.size as usize);
                        }
                    }
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

fn walk_track(rom: &[u8], marks: &mut Marks, start: usize) {
    let mut work: Vec<(usize, u8)> = vec![(start, 0)];
    let mut visited: HashSet<(usize, u8)> = HashSet::new();

    while let Some((mut p, mut status)) = work.pop() {
        loop {
            if p >= rom.len() || !visited.insert((p, status)) {
                break;
            }
            let byte = read_u8(rom, p);
            let cmd = if byte < 0x80 {
                status
            } else {
                marks.mark(p, p + 1);
                p += 1;
                if byte >= 0xBD {
                    status = byte;
                }
                byte
            };

            match cmd {
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
                0x80..=0xB0 => {}
                0xB2 => {
                    follow_target(rom, marks, &mut work, status, p);
                    break;
                }
                0xB3 => {
                    follow_target(rom, marks, &mut work, status, p);
                    p += 4;
                }
                0xB4 => {}
                0xB5 => {
                    marks.mark(p, p + 1);
                    p += 1;
                    follow_target(rom, marks, &mut work, status, p);
                    p += 4;
                }
                0xB9 => {
                    let op = read_u8(rom, p);
                    marks.mark(p, p + 3);
                    p += 3;
                    if (6..=17).contains(&op) {
                        follow_target(rom, marks, &mut work, status, p);
                        p += 4;
                    }
                }
                0xBA..=0xC5 | 0xC8 => {
                    marks.mark(p, p + 1);
                    p += 1;
                }
                0xCC => {
                    marks.mark(p, p + 2);
                    p += 2;
                }
                0xCD => {
                    let n = read_u8(rom, p);
                    marks.mark(p, p + 1);
                    p += 1;
                    match n {
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
                        _ => break,
                    }
                }
                0xCE => {
                    if read_u8(rom, p) < 0x80 && p < rom.len() {
                        marks.mark(p, p + 1);
                        p += 1;
                    }
                }
                _ => break,
            }
        }
    }
}

fn follow_target(rom: &[u8], marks: &mut Marks, work: &mut Vec<(usize, u8)>, status: u8, p: usize) {
    marks.mark(p, p + 4);
    if let Some(target) = ptr_to_offset(read_u32(rom, p), rom.len()) {
        work.push((target, status));
    }
}

pub fn waveform_dc_stats(rom: &GbaRom, song_id: u32) -> Vec<WaveformDcStat> {
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
            continue;
        }
        let Some(wav) = WaveData::read(data, wav_addr) else {
            continue;
        };
        let raw = &data[wav.data..wav.data + wav.size as usize];
        let pcm = crate::waveform::decode_pcm8(raw);
        if pcm.is_empty() {
            continue;
        }
        let mean = pcm.iter().map(|&v| f64::from(v)).sum::<f64>() / pcm.len() as f64;
        stats.push(WaveformDcStat {
            label: format!("0x{wav_addr:08X}"),
            dc_shift: mean.abs() as f32,
            length: pcm.len(),
            sample_rate: f64::from(wav.freq) / 1024.0,
        });
    }
    stats.sort_by(|a, b| b.dc_shift.total_cmp(&a.dc_shift));
    stats
}

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
            out.push(tone.wav);
        }
    }
}

//! Parses a GBA ROM: locates the song table and reads song headers out of it.

use std::sync::Arc;

use crate::util::{read_u8, read_u16, read_u32};

const MAX_SONGS: usize = 2048;

const SELECT_SONG_SIGNATURE: [u8; 30] = [
    0x00, 0xB5, 0x00, 0x04, 0x07, 0x4A, 0x08, 0x49, 0x40, 0x0B, 0x40, 0x18, 0x83, 0x88, 0x59, 0x00,
    0xC9, 0x18, 0x89, 0x00, 0x89, 0x18, 0x0A, 0x68, 0x01, 0x68, 0x10, 0x1C, 0x00, 0xF0,
];

pub(crate) fn ptr_to_offset(ptr: u32, rom_len: usize) -> Option<usize> {
    let bank = ptr >> 24;
    if bank != 0x08 && bank != 0x09 {
        return None;
    }
    let offset = (ptr - 0x0800_0000) as usize;
    (offset < rom_len).then_some(offset)
}

#[derive(Debug, Clone, Copy)]
pub struct SongHeader {
    pub offset: usize,
    pub track_count: u8,
    pub priority: u8,
    pub reverb: u8,
    pub voicegroup: usize,
}

pub struct GbaRom {
    pub data: Arc<[u8]>,
    pub song_table: usize,
    song_count: usize,
}

impl GbaRom {
    pub fn parse(bytes: &[u8]) -> Option<GbaRom> {
        if bytes.len() < 0xC0 || read_u8(bytes, 0xB2) != 0x96 {
            return None;
        }
        let song_table = find_table_by_signature(bytes).or_else(|| find_table_by_scan(bytes))?;
        let song_count = count_songs(bytes, song_table);
        if song_count == 0 {
            return None;
        }
        Some(GbaRom {
            data: Arc::from(bytes.to_vec()),
            song_table,
            song_count,
        })
    }

    pub fn song_count(&self) -> usize {
        self.song_count
    }

    pub fn game_code(&self) -> Option<String> {
        let raw = self.data.get(0xAC..0xB0)?;
        if !raw.iter().all(|&b| b.is_ascii_graphic()) {
            return None;
        }
        Some(raw.iter().map(|&b| b as char).collect())
    }

    pub fn extract_audio(&self) -> Vec<u8> {
        super::extract::extract_audio(self)
    }

    pub fn song_header(&self, id: u32) -> Option<SongHeader> {
        if id as usize >= self.song_count {
            return None;
        }
        let entry = self.song_table + id as usize * 8;
        let header_ptr = read_u32(&self.data, entry);
        let offset = ptr_to_offset(header_ptr, self.data.len())?;
        parse_song_header(&self.data, offset)
    }
}

fn parse_song_header(rom: &[u8], offset: usize) -> Option<SongHeader> {
    let track_count = read_u8(rom, offset);
    if track_count == 0 || track_count > crate::TRACK_COUNT as u8 {
        return None;
    }
    let voicegroup = ptr_to_offset(read_u32(rom, offset + 4), rom.len())?;
    for i in 0..track_count as usize {
        ptr_to_offset(read_u32(rom, offset + 8 + i * 4), rom.len())?;
    }
    Some(SongHeader {
        offset,
        track_count,
        priority: read_u8(rom, offset + 2),
        reverb: read_u8(rom, offset + 3),
        voicegroup,
    })
}

fn entry_is_valid(rom: &[u8], offset: usize) -> bool {
    let header_ptr = read_u32(rom, offset);
    let ms = read_u16(rom, offset + 4);
    let me = read_u16(rom, offset + 6);
    if ms > 15 || me > 15 {
        return false;
    }
    let Some(header) = ptr_to_offset(header_ptr, rom.len()) else {
        return false;
    };
    read_u8(rom, header) == 0 || parse_song_header(rom, header).is_some()
}

fn find_table_by_signature(rom: &[u8]) -> Option<usize> {
    let positions = crate::util::search_for_sequence(rom, &SELECT_SONG_SIGNATURE);
    for pos in positions {
        let table_ptr = read_u32(rom, pos + 40);
        if let Some(table) = ptr_to_offset(table_ptr, rom.len())
            && entry_is_valid(rom, table)
        {
            return Some(table);
        }
    }
    None
}

fn find_table_by_scan(rom: &[u8]) -> Option<usize> {
    const MIN_RUN: usize = 8;
    let mut best: Option<(usize, usize)> = None;
    let mut offset = 0usize;
    while offset + 8 <= rom.len() {
        if entry_is_valid(rom, offset) {
            let start = offset;
            let mut run = 0usize;
            let mut real_songs = 0usize;
            while offset + 8 <= rom.len() && run < MAX_SONGS && entry_is_valid(rom, offset) {
                if read_u8(
                    rom,
                    ptr_to_offset(read_u32(rom, offset), rom.len()).unwrap_or(0),
                ) > 0
                {
                    real_songs += 1;
                }
                run += 1;
                offset += 8;
            }
            if run >= MIN_RUN && real_songs >= 4 && best.is_none_or(|(_, b)| run > b) {
                best = Some((start, run));
            }
        } else {
            offset += 4;
        }
    }
    best.map(|(start, _)| start)
}

fn count_songs(rom: &[u8], table: usize) -> usize {
    let mut n = 0;
    while n < MAX_SONGS && table + n * 8 + 8 <= rom.len() && entry_is_valid(rom, table + n * 8) {
        n += 1;
    }
    n
}

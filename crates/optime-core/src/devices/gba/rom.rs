//! GBA ROM parsing: locating the MP2K (`m4a`/"Sappy") song table and enumerating songs.
//!
//! The song table is `gSongTable` from `pret/pokeemerald`: an array of 8-byte entries
//! `{ SongHeader *header, u16 ms, u16 me }`, each header being
//! `{ u8 trackCount, u8 blockCount, u8 priority, u8 reverb, ToneData *tone, u8 *part[n] }`.

use std::sync::Arc;

use crate::util::{read_u8, read_u16, read_u32};

/// Maximum songs we will enumerate from a table (guards against runaway scans).
const MAX_SONGS: usize = 2048;

/// `m4aSongNumStart` compiled by agbcc — the byte signature used (as in saptapper /
/// gba-mus-ripper) to locate the song table: its literal pool holds the `gSongTable` pointer
/// 40 bytes after the function start.
const SELECT_SONG_SIGNATURE: [u8; 30] = [
    0x00, 0xB5, 0x00, 0x04, 0x07, 0x4A, 0x08, 0x49, 0x40, 0x0B, 0x40, 0x18, 0x83, 0x88, 0x59, 0x00,
    0xC9, 0x18, 0x89, 0x00, 0x89, 0x18, 0x0A, 0x68, 0x01, 0x68, 0x10, 0x1C, 0x00, 0xF0,
];

/// Converts a GBA ROM-space pointer (0x08000000-based) to an offset into `rom`.
pub(crate) fn ptr_to_offset(ptr: u32, rom_len: usize) -> Option<usize> {
    let bank = ptr >> 24;
    if bank != 0x08 && bank != 0x09 {
        return None;
    }
    let offset = (ptr - 0x0800_0000) as usize;
    (offset < rom_len).then_some(offset)
}

/// One parsed song-table entry.
#[derive(Debug, Clone, Copy)]
pub struct SongHeader {
    /// Offset of the header within the ROM.
    pub offset: usize,
    /// Number of sequence tracks (0 for the empty placeholder songs).
    pub track_count: u8,
    /// Player priority byte from the header.
    pub priority: u8,
    /// Reverb byte from the header (`SOUND_MODE_REVERB_SET | amount`); `m4aSoundMode` applies it.
    pub reverb: u8,
    /// Offset of the song's voicegroup (ToneData array).
    pub voicegroup: usize,
}

/// A GBA ROM with a located MP2K song table.
pub struct GbaRom {
    /// The ROM bytes.
    pub data: Arc<[u8]>,
    /// Offset of the song table.
    pub song_table: usize,
    song_count: usize,
}

impl GbaRom {
    /// Parses `bytes` as a GBA ROM, locating the song table by code signature with a
    /// brute-force table scan as fallback. Returns `None` if no song table is found.
    pub fn parse(bytes: &[u8]) -> Option<GbaRom> {
        // GBA header sanity: fixed value 0x96 at 0xB2.
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

    /// Number of entries in the song table.
    pub fn song_count(&self) -> usize {
        self.song_count
    }

    /// The 4-character ASCII game code from the ROM header (offset 0xAC) — e.g. `"BPEE"` for
    /// Pokémon Emerald (USA/Europe). Used to select curated song-name tables. Returns `None` if the
    /// bytes there aren't printable ASCII (e.g. a hand-built test ROM). The header survives the
    /// audio-only [`extract_audio`](Self::extract_audio) image, so it works on exported audio too.
    pub fn game_code(&self) -> Option<String> {
        let raw = self.data.get(0xAC..0xB0)?;
        if !raw.iter().all(|&b| b.is_ascii_graphic()) {
            return None;
        }
        Some(raw.iter().map(|&b| b as char).collect())
    }

    /// Builds an audio-only image of this ROM: everything the MP2K engine cannot reach from
    /// the song table is zeroed, so the result carries no game code or art but plays
    /// identically. See [`extract_audio`](super::extract_audio).
    pub fn extract_audio(&self) -> Vec<u8> {
        super::extract::extract_audio(self)
    }

    /// Parses the header of song `id`, if the entry is valid and non-empty.
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

/// Validates and reads the song header at `offset`.
fn parse_song_header(rom: &[u8], offset: usize) -> Option<SongHeader> {
    let track_count = read_u8(rom, offset);
    if track_count == 0 || track_count > crate::TRACK_COUNT as u8 {
        return None;
    }
    let voicegroup = ptr_to_offset(read_u32(rom, offset + 4), rom.len())?;
    // Every track's start pointer must land in the ROM.
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

/// Whether `offset` looks like a valid song-table entry (header pointer + small player ids).
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
    // Empty placeholder songs (trackCount 0) are common and valid table entries.
    read_u8(rom, header) == 0 || parse_song_header(rom, header).is_some()
}

/// Locates the song table via the `m4aSongNumStart` code signature.
fn find_table_by_signature(rom: &[u8]) -> Option<usize> {
    let positions = crate::util::search_for_sequence(rom, &SELECT_SONG_SIGNATURE);
    for pos in positions {
        let table_ptr = read_u32(rom, pos + 40);
        if let Some(table) = ptr_to_offset(table_ptr, rom.len()) {
            if entry_is_valid(rom, table) {
                return Some(table);
            }
        }
    }
    None
}

/// Fallback: scans the ROM for the longest run of consecutive valid song-table entries.
fn find_table_by_scan(rom: &[u8]) -> Option<usize> {
    const MIN_RUN: usize = 8;
    let mut best: Option<(usize, usize)> = None; // (start, run length)
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
            // Require a healthy run with at least a few non-empty songs in it.
            if run >= MIN_RUN && real_songs >= 4 && best.is_none_or(|(_, b)| run > b) {
                best = Some((start, run));
            }
        } else {
            offset += 4;
        }
    }
    best.map(|(start, _)| start)
}

/// Counts consecutive valid entries from the table start.
fn count_songs(rom: &[u8], table: usize) -> usize {
    let mut n = 0;
    while n < MAX_SONGS && table + n * 8 + 8 <= rom.len() && entry_is_valid(rom, table + n * 8) {
        n += 1;
    }
    n
}

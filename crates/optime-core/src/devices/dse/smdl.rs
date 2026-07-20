//! The DSE SMDL music-sequence container.
//!
//! Layout transcribed from the real Explorers of Sky sequences (`files/SOUND/BGM/*.smd` in
//! `pret/pmd-sky`). An SMDL is a 0x40-byte header, a 0x40-byte `song` chunk (which carries the
//! ticks-per-quarter-note and track count), then one `trk ` chunk per track, ending in `eoc\0`.
//!
//! Each `trk ` chunk is a 16-byte `ChunkHeader` (label, … `u32` length at +0x0C) then a payload
//! that begins with a 4-byte preamble (`track_id`, `channel_id`, two unknown) followed by the
//! raw event bytecode decoded by [`super::events`].

use crate::util::{read_u16, read_u32};

const HEADER_LEN: usize = 0x40;
const SONG_CHUNK_LEN: usize = 0x40;
const CHUNK_HEADER_LEN: usize = 0x10;
const TRACK_PREAMBLE_LEN: usize = 4;

/// One track within an SMDL: its ids and the raw event bytes (after the 4-byte preamble).
#[derive(Debug, Clone)]
pub struct Track {
    pub track_id: u8,
    pub channel_id: u8,
    /// Raw event bytecode for [`super::events::decode_track`].
    pub events: Vec<u8>,
}

/// A parsed SMDL sequence.
#[derive(Debug, Clone)]
pub struct Smdl {
    /// Internal file name from the header (+0x20).
    pub name: String,
    pub version: u16,
    /// Ticks per quarter note (the `song` chunk's TPQN, usually 48).
    pub tpqn: u16,
    pub tracks: Vec<Track>,
}

impl Smdl {
    /// Parses a single SMDL sequence from `data` (which must start at the `smdl` magic).
    pub fn parse(data: &[u8]) -> Option<Smdl> {
        if data.len() < HEADER_LEN + SONG_CHUNK_LEN || &data[0..4] != b"smdl" {
            return None;
        }
        let version = read_u16(data, 0x0C);
        let name = read_name(data, 0x20, 16);

        // The `song` chunk immediately follows the header at 0x40.
        if &data[HEADER_LEN..HEADER_LEN + 4] != b"song" {
            return None;
        }
        // TPQN is a u16 at +0x12 within the song chunk (offset 0x52 in the file).
        let tpqn = read_u16(data, HEADER_LEN + 0x12);

        // Track chunks start after the 0x40-byte song chunk.
        let mut tracks = Vec::new();
        let mut pos = HEADER_LEN + SONG_CHUNK_LEN;
        while pos + CHUNK_HEADER_LEN <= data.len() {
            let label = &data[pos..pos + 4];
            if label == b"eoc\0" {
                break;
            }
            if label != b"trk " {
                break;
            }
            let len = read_u32(data, pos + 0x0C) as usize;
            let payload_start = pos + CHUNK_HEADER_LEN;
            let payload_end = (payload_start + len).min(data.len());
            if payload_start + TRACK_PREAMBLE_LEN <= payload_end {
                tracks.push(Track {
                    track_id: data[payload_start],
                    channel_id: data[payload_start + 1],
                    events: data[payload_start + TRACK_PREAMBLE_LEN..payload_end].to_vec(),
                });
            }
            // Track chunks are padded with 0x98 up to the next 4-byte boundary.
            pos = (payload_end + 0x3) & !0x3;
        }

        Some(Smdl {
            name,
            version,
            tpqn,
            tracks,
        })
    }
}

fn read_name(data: &[u8], offset: usize, max: usize) -> String {
    let mut s = String::new();
    for i in 0..max {
        let b = data.get(offset + i).copied().unwrap_or(0);
        if b == 0 || b == 0xAA {
            break;
        }
        s.push(b as char);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a minimal valid SMDL: header + song chunk + one trk chunk + eoc.
    fn synthetic_smdl(events: &[u8]) -> Vec<u8> {
        let mut d = vec![0u8; HEADER_LEN];
        d[0..4].copy_from_slice(b"smdl");
        d[0x0C..0x0E].copy_from_slice(&0x0415u16.to_le_bytes());

        // song chunk (0x40 bytes), TPQN=48 at +0x12.
        let mut song = vec![0u8; SONG_CHUNK_LEN];
        song[0..4].copy_from_slice(b"song");
        song[0x12..0x14].copy_from_slice(&48u16.to_le_bytes());
        d.extend_from_slice(&song);

        // trk chunk: header + preamble + events, padded to 16.
        let mut trk = Vec::new();
        trk.extend_from_slice(b"trk ");
        trk.extend_from_slice(&[0, 0, 0, 0]); // label params
        trk.extend_from_slice(&[0, 0, 0, 0]); // chunkbeg
        let payload_len = TRACK_PREAMBLE_LEN + events.len();
        trk.extend_from_slice(&(payload_len as u32).to_le_bytes());
        trk.extend_from_slice(&[3, 1, 0, 0]); // preamble: track_id=3, channel_id=1
        trk.extend_from_slice(events);
        while trk.len() % 16 != 0 {
            trk.push(0x98); // pad
        }
        d.extend_from_slice(&trk);

        // eoc chunk.
        d.extend_from_slice(b"eoc\0");
        d.extend_from_slice(&[0u8; 12]);
        d
    }

    #[test]
    fn parses_synthetic_smdl() {
        let smdl = Smdl::parse(&synthetic_smdl(&[0x83, 0x98])).unwrap();
        assert_eq!(smdl.version, 0x0415);
        assert_eq!(smdl.tpqn, 48);
        assert_eq!(smdl.tracks.len(), 1);
        assert_eq!(smdl.tracks[0].track_id, 3);
        assert_eq!(smdl.tracks[0].channel_id, 1);
        assert_eq!(&smdl.tracks[0].events[0..2], &[0x83, 0x98]);
    }

    #[test]
    fn rejects_non_smdl() {
        assert!(Smdl::parse(b"not an smdl file at all .........").is_none());
    }
}

//! Parses the SMDL sequence container: header, song chunk, and track chunks.

use crate::util::{read_u16, read_u32};

const HEADER_LEN: usize = 0x40;
const SONG_CHUNK_LEN: usize = 0x40;
const CHUNK_HEADER_LEN: usize = 0x10;
const TRACK_PREAMBLE_LEN: usize = 4;

#[derive(Debug, Clone)]
pub struct Track {
    pub track_id: u8,
    pub channel_id: u8,
    pub events: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct Smdl {
    pub name: String,
    pub version: u16,
    pub tpqn: u16,
    pub tracks: Vec<Track>,
}

impl Smdl {
    pub fn parse(data: &[u8]) -> Option<Smdl> {
        if data.len() < HEADER_LEN + SONG_CHUNK_LEN || &data[0..4] != b"smdl" {
            return None;
        }
        let version = read_u16(data, 0x0C);
        let name = read_name(data, 0x20, 16);

        if &data[HEADER_LEN..HEADER_LEN + 4] != b"song" {
            return None;
        }
        let tpqn = read_u16(data, HEADER_LEN + 0x12);

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

    fn synthetic_smdl(events: &[u8]) -> Vec<u8> {
        let mut d = vec![0u8; HEADER_LEN];
        d[0..4].copy_from_slice(b"smdl");
        d[0x0C..0x0E].copy_from_slice(&0x0415u16.to_le_bytes());

        let mut song = vec![0u8; SONG_CHUNK_LEN];
        song[0..4].copy_from_slice(b"song");
        song[0x12..0x14].copy_from_slice(&48u16.to_le_bytes());
        d.extend_from_slice(&song);

        let mut trk = Vec::new();
        trk.extend_from_slice(b"trk ");
        trk.extend_from_slice(&[0, 0, 0, 0]);
        trk.extend_from_slice(&[0, 0, 0, 0]);
        let payload_len = TRACK_PREAMBLE_LEN + events.len();
        trk.extend_from_slice(&(payload_len as u32).to_le_bytes());
        trk.extend_from_slice(&[3, 1, 0, 0]);
        trk.extend_from_slice(events);
        while trk.len() % 16 != 0 {
            trk.push(0x98);
        }
        d.extend_from_slice(&trk);

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

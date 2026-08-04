//! Chord tables baked into the binary and looked up per game, feeding the piano roll's chord lane.

use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct ChordSpan {
    pub start_step: f64,
    pub end_step: f64,
    pub label: String,
}

#[derive(Debug, Clone)]
pub struct SongChords {
    pub segments: Vec<ChordSpan>,
}

const CHORDS_BY_GBA_GAME_CODE: &[(&str, &[u8])] =
    &[("BPEE", include_bytes!("pokemon_emerald.ocd"))];

const CHORDS_BY_GAME_FILENAME: &[(&str, &[u8])] = &[];

fn parsed() -> &'static HashMap<&'static str, HashMap<u32, SongChords>> {
    static CACHE: OnceLock<HashMap<&'static str, HashMap<u32, SongChords>>> = OnceLock::new();
    CACHE.get_or_init(|| {
        CHORDS_BY_GBA_GAME_CODE
            .iter()
            .chain(CHORDS_BY_GAME_FILENAME.iter())
            .filter_map(|&(id, bytes)| Some((id, parse(bytes)?)))
            .collect()
    })
}

fn table_for(
    filename: Option<&str>,
    game_code: Option<&str>,
) -> Option<&'static HashMap<u32, SongChords>> {
    let tables = parsed();
    filename
        .and_then(|f| tables.get(f))
        .or_else(|| game_code.and_then(|c| tables.get(c)))
}

pub fn lookup(
    filename: Option<&str>,
    game_code: Option<&str>,
    song_id: u32,
) -> Option<&'static SongChords> {
    table_for(filename, game_code)?.get(&song_id)
}

fn parse(bytes: &[u8]) -> Option<HashMap<u32, SongChords>> {
    let mut r = Reader::new(bytes);
    if r.take(4)? != b"OCHD" || r.u8()? != 1 {
        return None;
    }
    let song_count = r.u32()?;
    let label_count = r.u32()?;

    let mut labels: Vec<&str> = Vec::with_capacity(label_count as usize);
    for _ in 0..label_count {
        let len = r.u16()? as usize;
        labels.push(std::str::from_utf8(r.take(len)?).ok()?);
    }

    let mut songs = HashMap::with_capacity(song_count as usize);
    for _ in 0..song_count {
        let song_id = r.u32()?;
        let end_step = r.u32()? as f64;
        let seg_count = r.u32()? as usize;
        let mut raw = Vec::with_capacity(seg_count);
        for _ in 0..seg_count {
            let start = r.u32()? as f64;
            let idx = r.u16()? as usize;
            raw.push((start, *labels.get(idx)?));
        }
        let mut segments = Vec::with_capacity(seg_count);
        for i in 0..raw.len() {
            let end = raw.get(i + 1).map(|(s, _)| *s).unwrap_or(end_step);
            segments.push(ChordSpan {
                start_step: raw[i].0,
                end_step: end,
                label: raw[i].1.to_string(),
            });
        }
        songs.insert(song_id, SongChords { segments });
    }
    Some(songs)
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.bytes.get(self.pos..self.pos + n)?;
        self.pos += n;
        Some(s)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_tables_parse() {
        assert!(lookup(None, Some("BPEE"), u32::MAX).is_none());
        let bpee = parsed().get("BPEE").expect("Emerald table must parse");
        assert!(!bpee.is_empty(), "Emerald table must have songs");
        let (_, song) = bpee.iter().next().unwrap();
        assert!(!song.segments.is_empty());
        for w in song.segments.windows(2) {
            assert_eq!(
                w[0].end_step, w[1].start_step,
                "segments must be contiguous"
            );
        }
        assert!(song.segments.iter().all(|s| !s.label.is_empty()));
    }
}

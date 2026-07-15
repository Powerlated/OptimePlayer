//! Pre-inferred chord labels for the piano-roll chord lane — the offline output of
//! the `optime-ml` model (`ml/src/bin/chord_export.rs`).
//!
//! The model is too heavy to run live (and absent from the wasm build), so the
//! chord timeline is baked once on PC and shipped as a compact **bespoke binary**
//! (`.ocd`), embedded here via `include_bytes!` and keyed by GBA game code (and, in
//! principle, source filename for game-code-less games — that map is empty for now).
//! Currently only Pokémon Emerald (`BPEE`) is covered.
//!
//! A song's entry is a list of step-timed [`ChordSpan`]s, each a ready-to-draw label
//! (`"<roman> (<name>)"`, e.g. `V7 (G7)`, or `"N.C."` for no-chord). All music theory
//! (roman spelling relative to the inferred key) lives in the offline tool; the app
//! just maps segment indices to interned label strings. See [`crate::piano_roll`].
//!
//! ## `.ocd` binary format (little-endian)
//!
//! ```text
//! magic        "OCHD"            (4 bytes)
//! version      u8                (= 1)
//! song_count   u32
//! label_count  u32
//! labels       label_count × { len: u16, utf8: [u8; len] }   — the dictionary
//! songs        song_count × {
//!     song_id    u32
//!     end_step   u32             — final boundary (last segment's end)
//!     seg_count  u32
//!     segments   seg_count × { start_step: u32, label_idx: u16 }
//! }
//! ```
//!
//! Segments are contiguous (the model labels every frame), so each segment's `end`
//! is the next segment's `start`, and the last segment's end is `end_step` — only
//! starts are stored.

use std::collections::HashMap;
use std::sync::OnceLock;

/// One chord segment on the sequencer-**step** timeline (the piano roll's native
/// time axis), with its ready-to-draw label.
#[derive(Debug, Clone)]
pub struct ChordSpan {
    pub start_step: f64,
    pub end_step: f64,
    /// Display label, e.g. `"V7 (G7)"` or `"N.C."`.
    pub label: String,
}

/// The inferred chord timeline for one song.
#[derive(Debug, Clone)]
pub struct SongChords {
    pub segments: Vec<ChordSpan>,
}

/// `(GBA game code, embedded `.ocd` bytes)` chord tables.
const CHORDS_BY_GBA_GAME_CODE: &[(&str, &[u8])] =
    &[("BPEE", include_bytes!("pokemon_emerald.ocd"))];

/// `(source-filename key, embedded `.ocd` bytes)` tables, for game-code-less games.
/// Empty for now; kept so local ROM-hack chord tables can be wired in later.
const CHORDS_BY_GAME_FILENAME: &[(&str, &[u8])] = &[];

/// Lazily parsed cache: table identifier (`"BPEE"` / filename key) → song map. Built
/// once from every embedded `.ocd` on first lookup; malformed tables are skipped.
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

/// The parsed song table covering a game, filename mapping taking precedence over
/// game code (mirrors [`crate::song_names`]).
fn table_for(
    filename: Option<&str>,
    game_code: Option<&str>,
) -> Option<&'static HashMap<u32, SongChords>> {
    let tables = parsed();
    filename
        .and_then(|f| tables.get(f))
        .or_else(|| game_code.and_then(|c| tables.get(c)))
}

/// Looks up the pre-inferred chord timeline for `song_id`, identified by the loaded
/// source `filename` and/or GBA `game_code`. `None` if the game has no table or the
/// id isn't covered. Returns a `'static` reference (the cache lives forever).
pub fn lookup(
    filename: Option<&str>,
    game_code: Option<&str>,
    song_id: u32,
) -> Option<&'static SongChords> {
    table_for(filename, game_code)?.get(&song_id)
}

/// Parses one `.ocd` payload into `song_id → SongChords`. Returns `None` on a bad
/// magic/version or any truncation.
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
        // Read (start_step, label_idx) pairs, then reconstruct each end from the
        // next start (contiguous) and the last from `end_step`.
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

/// Minimal little-endian byte cursor; every read is bounds-checked (returns `None`).
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
        // Every embedded `.ocd` must parse, and the covered games must be present.
        assert!(lookup(None, Some("BPEE"), u32::MAX).is_none()); // triggers parse; bad id → None
        let bpee = parsed().get("BPEE").expect("Emerald table must parse");
        assert!(!bpee.is_empty(), "Emerald table must have songs");
        // Spot-check one song's spans: contiguous, ascending, non-empty labels.
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

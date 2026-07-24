//! Curated, human-readable song titles (and official-soundtrack ordering) for games whose ROMs
//! carry no song names of their own — GBA MP2K song tables are just numbered.
//!
//! This is **presentation metadata**, deliberately kept in the app rather than `optime-core` (which
//! stays free of UI/cosmetic, game-specific data). Each table is a JSON file in this directory
//! (`[{ "songId", "title" }]`, embedded via `include_str!`). [`lookup`] selects one by the loaded
//! source filename ([`JSONS_BY_GAME_FILENAME`], for DS / game-code-less games) or, failing that, by
//! the GBA game code from the ROM header ([`JSONS_BY_GBA_GAME_CODE`]). A song's **index in that
//! array** is its listing order, used as the "Default" sort position, so the OST plays back in album
//! order with everything else following.
//!
//! The desktop app's "Edit song names" mode round-trips these files: [`source_json_dir`] and
//! [`target_filename`] tell it which on-disk JSON to write a game's edited titles + order back to.

use serde::Deserialize;

/// One parsed JSON entry: a song id, its display title, and (for sparse tables) its absolute list
/// position. The entry's position in the array is its listing order when no `sparseIndex` is given.
#[derive(Deserialize)]
struct SongEntry {
    #[serde(rename = "songId")]
    song_id: u32,
    title: String,
    /// The song's absolute position in the full song list, kept when only a few of a game's songs
    /// are curated so they hold their place among the many unlabeled ones (an array index alone
    /// would just rank them relative to each other). Absent in dense/album-ordered tables.
    #[serde(rename = "sparseIndex", default)]
    sparse_index: Option<usize>,
}

/// Curated metadata resolved for one song.
pub struct SongMeta {
    /// Human-readable display title.
    pub title: String,
    /// The song's position in the game's listing order (its index in the JSON array): OST tracks
    /// first in album order, then the rest. Used as the "Default" sort key when there's no
    /// [`Self::sparse_index`].
    pub order: usize,
    /// The song's absolute position in the full song list, if the table stores one (sparse tables);
    /// places the song at that index among the unlabeled ones instead of grouping it at the front.
    pub sparse_index: Option<usize>,
}

// `(source-filename key, on-disk filename, unparsed JSON)` for games identified by the loaded
// source filename rather than a game code (e.g. DS games, or GBA ROM hacks sharing a base game
// code). Matched first, so a per-file table can override a game-code one. Generated at build time
// from the optional, gitignored `local_extras.txt` manifest (see `build.rs`) so a personal checkout
// can carry ROM-hack metadata that is never committed; empty in a clean clone.
include!(concat!(env!("OUT_DIR"), "/local_filename_tables.rs"));

/// `(GBA game code, on-disk filename, unparsed JSON)` song tables.
const JSONS_BY_GBA_GAME_CODE: &[(&str, &str, &str)] = &[
    (
        "BPEE",
        "pokemon_emerald.json",
        include_str!("pokemon_emerald.json"),
    ),
    ("A3UJ", "mother_3.json", include_str!("mother_3.json")),
];

/// The unparsed JSON table for a game, filename mapping taking precedence over game code.
fn json_for(filename: Option<&str>, game_code: Option<&str>) -> Option<&'static str> {
    let key = filename.map(normalize_source);
    filename
        .and_then(|f| JSONS_BY_GAME_FILENAME.iter().find(|(k, ..)| *k == f))
        .or_else(|| {
            // Also try the normalized stem so a loaded "Some Game.gba" matches "some_game".
            key.as_deref()
                .and_then(|k| JSONS_BY_GAME_FILENAME.iter().find(|(name, ..)| *name == k))
        })
        .or_else(|| {
            game_code.and_then(|c| JSONS_BY_GBA_GAME_CODE.iter().find(|(code, ..)| *code == c))
        })
        .map(|(_, _, json)| *json)
}

/// The `Vec<SongEntry>` table for a game, in listing order (album tracks first).
fn table_for(filename: Option<&str>, game_code: Option<&str>) -> Option<Vec<SongEntry>> {
    json_for(filename, game_code).and_then(|json| serde_json::from_str::<Vec<SongEntry>>(json).ok())
}

/// Looks up curated metadata for `song_id`, identified by the loaded source `filename` and/or the
/// GBA `game_code` (e.g. `"BPEE"`). `None` if the game has no table or the id isn't in it. A table
/// may list one song id more than once, since an album can use the same ROM song in several places;
/// the song then takes the title and listing position of its first entry.
pub fn lookup(filename: Option<&str>, game_code: Option<&str>, song_id: u32) -> Option<SongMeta> {
    let table = table_for(filename, game_code)?;
    let order = table.iter().position(|e| e.song_id == song_id)?;
    Some(SongMeta {
        title: table[order].title.clone(),
        order,
        sparse_index: table[order].sparse_index,
    })
}

/// The source-tree directory holding the song-name JSON files, resolved from the compile-time crate
/// manifest path (so it's independent of the process's working directory). Only meaningful when the
/// app is run from the checkout — the desktop "Edit song names" mode writes back here.
#[cfg(not(target_arch = "wasm32"))]
pub fn source_json_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/song_names")
}

/// The on-disk JSON filename a game's edited titles should be written to, and whether that file is
/// already wired into the `include_str!` maps (so it loads on the next build). For a known game this
/// is its existing file; for a brand-new one it's derived from the game code or source stem and
/// `wired` is `false` (the maintainer must add a one-line `const` entry for it to load).
#[cfg(not(target_arch = "wasm32"))]
pub fn target_filename(filename: Option<&str>, game_code: Option<&str>) -> (String, bool) {
    let key = filename.map(normalize_source);
    if let Some((_, file, _)) = filename
        .and_then(|f| JSONS_BY_GAME_FILENAME.iter().find(|(k, ..)| *k == f))
        .or_else(|| {
            key.as_deref()
                .and_then(|k| JSONS_BY_GAME_FILENAME.iter().find(|(name, ..)| *name == k))
        })
        .or_else(|| {
            game_code.and_then(|c| JSONS_BY_GBA_GAME_CODE.iter().find(|(code, ..)| *code == c))
        })
    {
        return ((*file).to_owned(), true);
    }
    // Unknown game: derive a stable, filesystem-safe filename (game code preferred, else stem).
    let stem = game_code
        .map(normalize_source)
        .or(key)
        .unwrap_or_else(|| "untitled".to_owned());
    (format!("{stem}.json"), false)
}

/// Whether `name` is an on-disk JSON filename already wired into the `include_str!` maps (so saving
/// to it updates a table that loads on the next build, rather than a brand-new file needing a
/// one-line `const` addition). Used by the editor's adjustable-filename field.
#[cfg(not(target_arch = "wasm32"))]
pub fn is_known_filename(name: &str) -> bool {
    JSONS_BY_GAME_FILENAME
        .iter()
        .chain(JSONS_BY_GBA_GAME_CODE.iter())
        .any(|(_, file, _)| *file == name)
}

/// Lower-cases and reduces `source` to a filesystem-safe stem: drops any extension and collapses
/// every run of non-alphanumeric characters to a single `_`.
fn normalize_source(source: &str) -> String {
    let stem = source.rsplit_once('.').map_or(source, |(s, _)| s);
    let mut out = String::with_capacity(stem.len());
    let mut prev_us = false;
    for ch in stem.chars() {
        if ch.is_ascii_alphanumeric() {
            out.extend(ch.to_lowercase());
            prev_us = false;
        } else if !prev_us {
            out.push('_');
            prev_us = true;
        }
    }
    out.trim_matches('_').to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_emerald_songs_resolve() {
        // The album's first mapped track sorts first, and the rest follow it in album order.
        let opening = lookup(None, Some("BPEE"), 414).unwrap();
        assert_eq!(opening.title, "Introductions");
        assert_eq!(opening.order, 0);
        assert!(lookup(None, Some("BPEE"), 442).unwrap().order > opening.order);
        let title = lookup(None, Some("BPEE"), 413).unwrap();
        assert_eq!(title.title, "Title Screen: Main Theme");
        // An OST track sorts before a non-OST one (FRLG port).
        let cycling = lookup(None, Some("BPEE"), 403).unwrap().order;
        let union = lookup(None, Some("BPEE"), 539).unwrap();
        assert_eq!(union.title, "Union Room");
        assert!(cycling < union.order, "OST track must precede a port");
        // Unknown game / id.
        assert!(lookup(None, Some("XXXX"), 413).is_none());
        assert!(lookup(None, Some("BPEE"), 99999).is_none());
    }

    #[test]
    fn mother3_sound_player_titles_resolve() {
        // Real MP2K song ids (via the in-ROM Sound Player slot->song-id table), in player order.
        assert_eq!(
            lookup(None, Some("A3UJ"), 32).unwrap().title,
            "MOTHER 3 Love Theme"
        );
        assert_eq!(
            lookup(None, Some("A3UJ"), 410).unwrap().title,
            "Unfounded Revenge"
        );
        assert_eq!(
            lookup(None, Some("A3UJ"), 1518).unwrap().title,
            "Curtain Call"
        );
        assert_eq!(
            lookup(None, Some("A3UJ"), 1938).unwrap().title,
            "Battle Against the Masked Man"
        );
        // "Let's Begin!" (Sound Player slot 1) lists before "Memory of Life" (the last slot).
        let begin = lookup(None, Some("A3UJ"), 58).unwrap().order;
        let memory = lookup(None, Some("A3UJ"), 51).unwrap().order;
        assert!(begin < memory);
    }

    #[test]
    fn embedded_tables_parse_and_are_non_empty() {
        for (_, file, json) in JSONS_BY_GBA_GAME_CODE.iter() {
            let table: Vec<SongEntry> = serde_json::from_str(json)
                .unwrap_or_else(|e| panic!("{file} must be valid JSON: {e}"));
            assert!(!table.is_empty(), "{file} must list at least one song");
            assert!(
                table.iter().all(|e| !e.title.trim().is_empty()),
                "every song in {file} must have a title"
            );
        }
    }

    #[test]
    fn target_filename_resolves_known_and_derives_unknown() {
        assert_eq!(
            target_filename(None, Some("BPEE")),
            ("pokemon_emerald.json".to_owned(), true)
        );
        // Unknown game code: derived, not yet wired.
        assert_eq!(
            target_filename(None, Some("XXXX")),
            ("xxxx.json".to_owned(), false)
        );
        // DS-style source filename, no game code: normalized stem.
        let (file, wired) = target_filename(Some("Pokémon Platinum.nds"), None);
        assert_eq!(file, "pok_mon_platinum.json");
        assert!(!wired);
    }
}

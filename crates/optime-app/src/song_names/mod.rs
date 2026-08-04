use serde::Deserialize;

#[derive(Deserialize)]
struct SongEntry {
    #[serde(rename = "songId")]
    song_id: u32,
    title: String,
    #[serde(rename = "sparseIndex", default)]
    sparse_index: Option<usize>,
}

pub struct SongMeta {
    pub title: String,
    pub order: usize,
    pub sparse_index: Option<usize>,
}

include!(concat!(env!("OUT_DIR"), "/local_filename_tables.rs"));

const JSONS_BY_GBA_GAME_CODE: &[(&str, &str, &str)] = &[
    (
        "BPEE",
        "pokemon_emerald.json",
        include_str!("pokemon_emerald.json"),
    ),
    ("A3UJ", "mother_3.json", include_str!("mother_3.json")),
];

fn json_for(filename: Option<&str>, game_code: Option<&str>) -> Option<&'static str> {
    let key = filename.map(normalize_source);
    filename
        .and_then(|f| JSONS_BY_GAME_FILENAME.iter().find(|(k, ..)| *k == f))
        .or_else(|| {
            key.as_deref()
                .and_then(|k| JSONS_BY_GAME_FILENAME.iter().find(|(name, ..)| *name == k))
        })
        .or_else(|| {
            game_code.and_then(|c| JSONS_BY_GBA_GAME_CODE.iter().find(|(code, ..)| *code == c))
        })
        .map(|(_, _, json)| *json)
}

fn table_for(filename: Option<&str>, game_code: Option<&str>) -> Option<Vec<SongEntry>> {
    json_for(filename, game_code).and_then(|json| serde_json::from_str::<Vec<SongEntry>>(json).ok())
}

pub fn lookup(filename: Option<&str>, game_code: Option<&str>, song_id: u32) -> Option<SongMeta> {
    let table = table_for(filename, game_code)?;
    let order = table.iter().position(|e| e.song_id == song_id)?;
    Some(SongMeta {
        title: table[order].title.clone(),
        order,
        sparse_index: table[order].sparse_index,
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn source_json_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/song_names")
}

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
    let stem = game_code
        .map(normalize_source)
        .or(key)
        .unwrap_or_else(|| "untitled".to_owned());
    (format!("{stem}.json"), false)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn is_known_filename(name: &str) -> bool {
    JSONS_BY_GAME_FILENAME
        .iter()
        .chain(JSONS_BY_GBA_GAME_CODE.iter())
        .any(|(_, file, _)| *file == name)
}

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
        let opening = lookup(None, Some("BPEE"), 414).unwrap();
        assert_eq!(opening.title, "Introductions");
        assert_eq!(opening.order, 0);
        assert!(lookup(None, Some("BPEE"), 442).unwrap().order > opening.order);
        let title = lookup(None, Some("BPEE"), 413).unwrap();
        assert_eq!(title.title, "Title Screen: Main Theme");
        let cycling = lookup(None, Some("BPEE"), 403).unwrap().order;
        let union = lookup(None, Some("BPEE"), 539).unwrap();
        assert_eq!(union.title, "Union Room");
        assert!(cycling < union.order, "Emerald album must precede FireRed");
        let frontier = lookup(None, Some("BPEE"), 457).unwrap();
        assert_eq!(frontier.title, "Battle Frontier");
        assert!(
            union.order < frontier.order,
            "FireRed album must precede the leftovers"
        );
        let pallet = lookup(None, Some("BPEE"), 512).unwrap();
        assert_eq!(pallet.title, "Pallet Town");
        assert!(pallet.order < lookup(None, Some("BPEE"), 498).unwrap().order);
        let level_up = lookup(None, Some("BPEE"), 367).unwrap();
        let firered_start = lookup(None, Some("BPEE"), 489).unwrap().order;
        assert!(level_up.order < firered_start);
        assert!(lookup(None, Some("XXXX"), 413).is_none());
        assert!(lookup(None, Some("BPEE"), 99999).is_none());
    }

    #[test]
    fn mother3_sound_player_titles_resolve() {
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
        assert_eq!(
            target_filename(None, Some("XXXX")),
            ("xxxx.json".to_owned(), false)
        );
        let (file, wired) = target_filename(Some("Pokémon Platinum.nds"), None);
        assert_eq!(file, "pok_mon_platinum.json");
        assert!(!wired);
    }
}

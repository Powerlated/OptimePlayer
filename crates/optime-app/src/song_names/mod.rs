//! Curated, human-readable song titles (and official-soundtrack ordering) for games whose ROMs
//! carry no song names of their own — GBA MP2K song tables are just numbered.
//!
//! This is **presentation metadata**, deliberately kept in the app rather than `optime-core` (which
//! stays free of UI/cosmetic, game-specific data). Each game lives in its own submodule (e.g.
//! [`pokemon_emerald`]) holding a `(song_id, title)` table in listing order — album tracks first,
//! then the rest — selected here by the GBA game code from the ROM header
//! (`SoundData::gba_game_code`). A song's index in that table is its "Default" sort position, so
//! the OST plays back in album order with everything else following.

mod pokemon_emerald;

/// Curated metadata for one song.
pub struct SongMeta {
    /// Human-readable display title.
    pub title: &'static str,
    /// The song's position in the game's listing order (its index in the per-game table): OST
    /// tracks first in album order, then the rest. Used as the "Default" sort key.
    pub order: usize,
}

/// The `(song_id, title)` table for a GBA game code, in listing order (album tracks first).
fn table_for(game_code: &str) -> Option<&'static [(u32, &'static str)]> {
    match game_code {
        // Pokémon Emerald (USA/Europe).
        "BPEE" => Some(pokemon_emerald::SONGS),
        _ => None,
    }
}

/// Looks up curated metadata for `song_id` of the game identified by `game_code` (e.g. `"BPEE"`).
pub fn lookup(game_code: &str, song_id: u32) -> Option<SongMeta> {
    let table = table_for(game_code)?;
    let order = table.iter().position(|&(id, _)| id == song_id)?;
    Some(SongMeta {
        title: table[order].1,
        order,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emerald_ids_are_unique() {
        let mut ids: Vec<u32> = pokemon_emerald::SONGS.iter().map(|&(id, _)| id).collect();
        ids.sort_unstable();
        let n = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), n, "song ids must be unique");
    }

    #[test]
    fn known_emerald_songs_resolve() {
        // The album's first mapped track sorts first.
        assert_eq!(lookup("BPEE", 442).unwrap().order, 0);
        let title = lookup("BPEE", 413).unwrap();
        assert_eq!(title.title, "Title Screen: Main Theme");
        // An OST track sorts before a non-OST one (FRLG port).
        let cycling = lookup("BPEE", 403).unwrap().order;
        let union = lookup("BPEE", 539).unwrap();
        assert_eq!(union.title, "Union Room");
        assert!(cycling < union.order, "OST track must precede a port");
        // Unknown game / id.
        assert!(lookup("XXXX", 413).is_none());
        assert!(lookup("BPEE", 99999).is_none());
    }
}

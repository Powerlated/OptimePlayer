//! Integration tests that parse the real demo SDAT archives shipped in `demos/`.

use std::path::PathBuf;

use optime_core::{load_all, Sdat};

fn demo_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../demos")
        .join(name)
}

fn load(name: &str) -> Vec<Sdat> {
    let bytes = std::fs::read(demo_path(name)).expect("demo file should exist");
    Sdat::load_all(&bytes)
}

#[test]
fn parses_super_mario_64_ds() {
    let sdats = load("super-mario-64-ds.sdat");
    assert!(!sdats.is_empty(), "should find at least one SDAT");
    let sdat = &sdats[0];

    // The sequence list and name dictionaries should be populated and consistent.
    assert!(!sdat.sseq_list.is_empty());
    assert_eq!(sdat.sseq_name_to_id.len(), sdat.sseq_id_to_name.len());

    // The fixture's song must be present and round-trip name<->id.
    let id = sdat
        .sseq_name_to_id
        .get("NCS_BGM_PERFECT")
        .copied()
        .expect("NCS_BGM_PERFECT should exist");
    assert_eq!(
        sdat.sseq_id_to_name.get(&id).map(String::as_str),
        Some("NCS_BGM_PERFECT")
    );

    // Its INFO record and the bank it references should resolve.
    let info = sdat.sseq_infos[id as usize].as_ref().unwrap();
    assert!(
        sdat.file(info.file_id).is_some(),
        "SSEQ file should be in the FAT"
    );
    assert!(
        sdat.instrument_banks
            .get(info.bank as usize)
            .map(Option::is_some)
            .unwrap_or(false),
        "the referenced bank should be decoded"
    );
}

#[test]
fn parses_multiple_demos() {
    for name in [
        "new-super-mario-bros.sdat",
        "pokemon-platinum.sdat",
        "ace-attorney.sdat",
    ] {
        let sdats = load(name);
        assert!(!sdats.is_empty(), "{name}: expected an SDAT");
        let sdat = &sdats[0];
        assert!(!sdat.sseq_list.is_empty(), "{name}: expected sequences");
        // Every listed sequence must have an INFO record.
        for &id in &sdat.sseq_list {
            assert!(sdat.sseq_infos[id as usize].is_some(), "{name}: id {id}");
        }
    }
}

#[test]
fn computes_song_lengths() {
    let bytes = std::fs::read(demo_path("super-mario-64-ds.sdat")).unwrap();
    let archives = load_all(&bytes);
    let data = &*archives[0];
    let ids = data.song_ids();
    assert!(!ids.is_empty());

    // Each playable song should resolve to a finite, positive length under the 15-minute cap
    // (the demo's songs all loop or end well before then).
    for &id in ids.iter().take(8) {
        let len = data
            .song_length_seconds(id)
            .unwrap_or_else(|| panic!("song {id} should have a length"));
        assert!(
            len > 0.0 && len.is_finite() && len < 15.0 * 60.0,
            "song {id}: implausible length {len}s"
        );
    }
}

#[test]
fn lookahead_overview_collects_notes() {
    // The look-ahead now runs the real device player and reads its `SynthEvent` stream; the
    // whole-song `overview` should yield a non-empty note timeline with sane durations.
    let bytes = std::fs::read(demo_path("super-mario-64-ds.sdat")).unwrap();
    let archives = load_all(&bytes);
    let data = &*archives[0];
    let id = data.song_ids()[0];

    let overview =
        optime_core::FsVisController::overview(data, id).expect("overview for the first song");
    assert!(!overview.notes.is_empty(), "the song should contain notes");
    assert!(overview.total_steps > 0);
    assert!(!overview.tempos.is_empty(), "at least the starting tempo");
    // Every resolved note bar must fit inside the timeline.
    for n in &overview.notes {
        assert!(
            n.timestamp + n.duration <= overview.total_steps,
            "note runs past the timeline: ts={} dur={} total={}",
            n.timestamp,
            n.duration,
            overview.total_steps
        );
    }
}

#[test]
fn rejects_garbage() {
    assert!(Sdat::load_all(&[0u8; 64]).is_empty());
    assert!(Sdat::load_all(b"not an sdat at all").is_empty());
}

//! Integration tests that parse the real demo SDAT archives shipped in `demos/`.

use std::path::PathBuf;

use optime_core::Sdat;

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
fn rejects_garbage() {
    assert!(Sdat::load_all(&[0u8; 64]).is_empty());
    assert!(Sdat::load_all(b"not an sdat at all").is_empty());
}

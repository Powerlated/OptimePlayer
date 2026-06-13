//! End-to-end GBA (MP2K) test against a synthetic in-memory ROM: a one-track song playing a
//! single DirectSound note. Exercises ROM parsing (heuristic song-table scan), the sequencer,
//! the player's channel/envelope model, and the shared synthesis path.

use optime_core::{SoundData, SynthConfig, SynthController};

/// GBA ROM-space base address.
const ROM_BASE: u32 = 0x0800_0000;

// Synthetic ROM layout.
const SONG_TABLE: usize = 0x200;
const SONG_HEADER: usize = 0x300;
/// A trackCount-0 placeholder header (a valid table entry, but not a playable song).
const EMPTY_HEADER: usize = 0x380;
const VOICEGROUP: usize = 0x400;
const TRACK: usize = 0x500;
const WAVE: usize = 0x600;
const ROM_LEN: usize = 0x800;

fn put_u32(rom: &mut [u8], offset: usize, value: u32) {
    rom[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn ptr(offset: usize) -> u32 {
    ROM_BASE + offset as u32
}

/// Builds a minimal MP2K ROM: an 8-entry song table where every entry points at the same
/// one-track song (the table scanner needs a run of ≥8 valid entries with ≥4 non-empty songs).
fn build_rom() -> Vec<u8> {
    let mut rom = vec![0u8; ROM_LEN];
    // GBA header sanity byte.
    rom[0xB2] = 0x96;

    // Song table: { SongHeader *header, u16 ms, u16 me } × 8. The last entry points at an
    // empty placeholder header (trackCount 0), as real song tables are littered with.
    for i in 0..7 {
        put_u32(&mut rom, SONG_TABLE + i * 8, ptr(SONG_HEADER));
        // ms/me stay 0.
    }
    put_u32(&mut rom, SONG_TABLE + 7 * 8, ptr(EMPTY_HEADER));

    // SongHeader: trackCount, blockCount, priority, reverb, tone*, part[0]*.
    rom[SONG_HEADER] = 1;
    put_u32(&mut rom, SONG_HEADER + 4, ptr(VOICEGROUP));
    put_u32(&mut rom, SONG_HEADER + 8, ptr(TRACK));

    // Voicegroup, program 0: a plain DirectSound tone with an instant-attack flat envelope.
    rom[VOICEGROUP] = 0x00; // type = DirectSound
    rom[VOICEGROUP + 1] = 60; // base key
    put_u32(&mut rom, VOICEGROUP + 4, ptr(WAVE));
    rom[VOICEGROUP + 8] = 255; // attack: instant
    rom[VOICEGROUP + 9] = 0; // decay (unused: sustain is full)
    rom[VOICEGROUP + 10] = 255; // sustain: full
    rom[VOICEGROUP + 11] = 200; // release

    // WaveData: looping flag in byte 3, freq = rate << 10, then 64 bytes of s8 square PCM.
    rom[WAVE + 3] = 0x40;
    put_u32(&mut rom, WAVE + 4, 13379 << 10);
    put_u32(&mut rom, WAVE + 8, 0); // loop start
    put_u32(&mut rom, WAVE + 12, 64); // size
    for i in 0..64 {
        rom[WAVE + 16 + i] = if i < 32 {
            100u8
        } else {
            156u8 /* -100 as s8 */
        };
    }

    // Track: TEMPO 75 (one step per frame), VOICE 0, VOL 100,
    // N24 key=60 vel=100, W24, FINE.
    let t = TRACK;
    let track_bytes: &[u8] = &[
        0xBB, 75, // TEMPO
        0xBD, 0, // VOICE
        0xBE, 100, // VOL
        0xE7, 60, 100,  // note, gate = CLOCK_TABLE[0xE7 - 0xCF] = 24 steps
        0x98, // W24 rest
        0xB1, // FINE
    ];
    rom[t..t + track_bytes.len()].copy_from_slice(track_bytes);

    rom
}

#[test]
fn parses_synthetic_rom() {
    let rom = build_rom();
    let archives = SoundData::load_all(&rom);
    assert_eq!(archives.len(), 1, "one GBA archive expected");
    let data = &archives[0];
    assert!(matches!(data, SoundData::Gba(_)));
    let ids = data.song_ids();
    assert_eq!(
        ids,
        (0..7).collect::<Vec<u32>>(),
        "only playable songs are listed; the trackCount-0 placeholder (id 7) is filtered"
    );
    assert!(
        data.make_player(7).is_none(),
        "the placeholder must indeed be unplayable"
    );
}

#[test]
fn renders_audio_and_ends() {
    let rom = build_rom();
    let data = SoundData::load_all(&rom).remove(0);
    let sr = 32768.0;
    let mut controller = SynthController::new(sr, &data, 0).expect("song 0 should load");
    let config = SynthConfig::default();

    // The note is 24 steps at 1 step/frame ≈ 0.4 s; render 2 s so the song finishes.
    let mut out = vec![0.0f32; 2 * (2.0 * sr) as usize];
    controller.fill(&mut out, &config);

    let peak = out.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    assert!(peak > 0.01, "the note should be audible, peak={peak}");
    assert!(
        controller.fading_start,
        "the song should report its end after FINE"
    );
}

#[test]
fn renders_audio_in_authentic_mode() {
    let rom = build_rom();
    let data = SoundData::load_all(&rom).remove(0);
    let sr = 48_000.0;
    let mut controller = SynthController::new(sr, &data, 0).expect("song 0 should load");
    let config = SynthConfig {
        resample: optime_core::ResampleMode::Authentic { half_taps: 16 },
        ..SynthConfig::default()
    };

    let mut out = vec![0.0f32; 2 * sr as usize];
    controller.fill(&mut out, &config);
    let peak = out.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    assert!(
        peak > 0.01,
        "the note should be audible through the hardware chain, peak={peak}"
    );
}

#[test]
fn lookahead_sees_the_note() {
    let rom = build_rom();
    let data = SoundData::load_all(&rom).remove(0);
    let mut look = optime_core::FsVisController::new(&data, 0).expect("song 0 should load");
    for _ in 0..64 {
        look.tick();
    }
    let notes: Vec<_> = (0..look.notes.entries())
        .filter_map(|i| look.notes.peek(i))
        .collect();
    assert_eq!(notes.len(), 1, "exactly one note in the song");
    assert_eq!(notes[0].key, 60);
    assert_eq!(notes[0].velocity, 100);
    assert_eq!(notes[0].duration, 24);
    assert_eq!(notes[0].track, 0);
}

#[test]
fn rejects_non_mp2k_rom() {
    // Valid header byte but no song table anywhere.
    let mut rom = vec![0u8; ROM_LEN];
    rom[0xB2] = 0x96;
    assert!(SoundData::load_all(&rom).is_empty());
}

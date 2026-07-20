//! End-to-end GBA (MP2K) test against a synthetic in-memory ROM: a one-track song playing a
//! single DirectSound note. Exercises ROM parsing (heuristic song-table scan), the sequencer,
//! the player's channel/envelope model, and the shared synthesis path.

use optime_core::devices::gba::GbaRom;
use optime_core::{
    LoopAndTransitionOptions, PerDeviceSettings, PlaybackEvent, SoundData, SynthController,
    load_all,
};

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
    let archives = load_all(&rom);
    assert_eq!(archives.len(), 1, "one GBA archive expected");
    let data = &*archives[0];
    assert!(
        data.as_any().downcast_ref::<GbaRom>().is_some(),
        "expected a GBA archive"
    );
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
    let data: Box<dyn SoundData> = load_all(&rom).remove(0);
    let sr = 32768.0;
    let mut controller = SynthController::new(sr, &*data, 0).expect("song 0 should load");
    // Fade out when the sequence signals its end, so the end-of-song detection is observable as a
    // pumped `TransitionStarted` message.
    controller.set_loop_and_transition(LoopAndTransitionOptions {
        fade_on_end: true,
        fade_seconds: 1.0,
        ..LoopAndTransitionOptions::none()
    });
    let config = PerDeviceSettings::neutral();

    // The note is 24 steps at 1 step/frame ≈ 0.4 s; render 2 s so the song finishes.
    let mut out = vec![0.0f32; 2 * (2.0 * sr) as usize];
    controller.fill(&mut out, &config);

    let peak = out.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    assert!(peak > 0.01, "the note should be audible, peak={peak}");
    let msgs: Vec<_> = controller.take_messages().collect();
    assert!(
        msgs.contains(&PlaybackEvent::TransitionStarted),
        "the song should report its end (a fade transition) after FINE: {msgs:?}"
    );
}

#[test]
fn lookahead_sees_the_note() {
    let rom = build_rom();
    let data: Box<dyn SoundData> = load_all(&rom).remove(0);
    let mut look = optime_core::FsVisController::new(&*data, 0).expect("song 0 should load");
    for _ in 0..64 {
        look.tick();
    }
    let notes: Vec<_> = (0..look.notes.entries())
        .filter_map(|i| look.notes.peek(i))
        .collect();
    assert_eq!(notes.len(), 1, "exactly one note in the song");
    assert_eq!(notes[0].key, 60);
    assert_eq!(notes[0].duration, 24);
    assert_eq!(notes[0].track, 0);
}

#[test]
fn remove_sample_dc_offset_is_opt_in_and_centers_the_signal() {
    // A DC-biased looping wave: every sample is +100 (s8), mean ≈ +0.78 of full scale.
    let mut rom = build_rom();
    for i in 0..64 {
        rom[WAVE + 16 + i] = 100;
    }
    let data: Box<dyn SoundData> = load_all(&rom).remove(0);
    let sr = 32768.0;

    let mean_of = |remove: bool| -> f32 {
        let config = PerDeviceSettings {
            remove_sample_dc_offset: remove,
            ..PerDeviceSettings::neutral()
        };
        let mut controller = SynthController::new(sr, &*data, 0).expect("song 0 should load");
        // Render the sustained portion of the note (skip the very start) and average it.
        let mut out = vec![0.0f32; (0.3 * sr) as usize * 2];
        controller.fill(&mut out, &config);
        let tail = &out[out.len() / 2..];
        tail.iter().sum::<f32>() / tail.len() as f32
    };

    // Off (the default): the biased sample keeps its DC, so the rendered signal is offset.
    assert!(
        mean_of(false).abs() > 0.05,
        "with DC removal off the biased sample should leave a clear offset"
    );
    // On: the sample is centered, so the steady-state mean collapses toward zero.
    assert!(
        mean_of(true).abs() < 0.01,
        "with DC removal on the rendered signal should be centered"
    );
}

#[test]
fn sample_dc_stats_report_the_offset_removed() {
    // Give the wave a known DC bias: every sample is +50 (s8), so its mean is 50/128 of full
    // scale — exactly what the decoder removes and the stat must report.
    let mut rom = build_rom();
    for i in 0..64 {
        rom[WAVE + 16 + i] = 50;
    }
    let data: Box<dyn SoundData> = load_all(&rom).remove(0);
    let stats = data.waveform_dc_stats(0);

    assert_eq!(
        stats.len(),
        1,
        "the song reaches exactly one DirectSound wave"
    );
    let s = &stats[0];
    assert_eq!(s.label, format!("0x{:08X}", ptr(WAVE)));
    assert_eq!(s.length, 64);
    assert!((s.sample_rate - 13379.0).abs() < 1.0);
    assert!(
        (s.dc_shift - 50.0 / 128.0).abs() < 1e-4,
        "dc_shift should be the sample mean as a fraction of full scale, got {}",
        s.dc_shift
    );

    // The symmetric ±100 square in the default ROM is already centered: ~zero shift.
    let centered = load_all(&build_rom()).remove(0).waveform_dc_stats(0);
    assert!(
        centered[0].dc_shift < 1e-6,
        "a symmetric wave has no DC offset"
    );
}

#[test]
fn audio_extraction_strips_non_audio_and_plays_identically() {
    let mut rom = build_rom();
    // Plant non-audio "game data" the extractor must not ship.
    rom[0x100] = 0xAB;
    rom[0x1F0] = 0xCD;

    let data: Box<dyn SoundData> = load_all(&rom).remove(0);
    let Some(gba) = data.as_any().downcast_ref::<GbaRom>() else {
        panic!("expected a GBA archive")
    };
    let extracted = gba.extract_audio();

    // The junk bytes are gone (zeroed or truncated away).
    assert!(extracted.len() <= rom.len());
    assert_eq!(extracted.get(0x100).copied().unwrap_or(0), 0);
    assert_eq!(extracted.get(0x1F0).copied().unwrap_or(0), 0);

    // The extracted image is still a loadable GBA archive with the same songs.
    let stripped: Box<dyn SoundData> = load_all(&extracted).remove(0);
    assert_eq!(stripped.song_ids(), data.song_ids());

    // And it renders bit-identically to the original ROM.
    let sr = 32768.0;
    let config = PerDeviceSettings::neutral();
    let mut original = SynthController::new(sr, &*data, 0).expect("song 0 in the original");
    let mut audio_only = SynthController::new(sr, &*stripped, 0).expect("song 0 in the extract");
    let mut a = vec![0.0f32; 2 * sr as usize];
    let mut b = vec![0.0f32; 2 * sr as usize];
    original.fill(&mut a, &config);
    audio_only.fill(&mut b, &config);
    assert_eq!(a, b, "extracted audio must play identically");
}

#[test]
fn rejects_non_mp2k_rom() {
    // Valid header byte but no song table anywhere.
    let mut rom = vec![0u8; ROM_LEN];
    rom[0xB2] = 0x96;
    assert!(load_all(&rom).is_empty());
}

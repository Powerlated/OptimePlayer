use optime_core::devices::gba::GbaRom;
use optime_core::{
    LoopAndTransitionOptions, PerDeviceSettings, PlaybackEvent, SoundData, SynthController,
    load_all,
};

const ROM_BASE: u32 = 0x0800_0000;

const SONG_TABLE: usize = 0x200;
const SONG_HEADER: usize = 0x300;
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

fn build_rom() -> Vec<u8> {
    let mut rom = vec![0u8; ROM_LEN];
    rom[0xB2] = 0x96;

    for i in 0..7 {
        put_u32(&mut rom, SONG_TABLE + i * 8, ptr(SONG_HEADER));
    }
    put_u32(&mut rom, SONG_TABLE + 7 * 8, ptr(EMPTY_HEADER));

    rom[SONG_HEADER] = 1;
    put_u32(&mut rom, SONG_HEADER + 4, ptr(VOICEGROUP));
    put_u32(&mut rom, SONG_HEADER + 8, ptr(TRACK));

    rom[VOICEGROUP] = 0x00;
    rom[VOICEGROUP + 1] = 60;
    put_u32(&mut rom, VOICEGROUP + 4, ptr(WAVE));
    rom[VOICEGROUP + 8] = 255;
    rom[VOICEGROUP + 9] = 0;
    rom[VOICEGROUP + 10] = 255;
    rom[VOICEGROUP + 11] = 200;

    rom[WAVE + 3] = 0x40;
    put_u32(&mut rom, WAVE + 4, 13379 << 10);
    put_u32(&mut rom, WAVE + 8, 0);
    put_u32(&mut rom, WAVE + 12, 64);
    for i in 0..64 {
        rom[WAVE + 16 + i] = if i < 32 { 100u8 } else { 156u8 };
    }

    let t = TRACK;
    let track_bytes: &[u8] = &[0xBB, 75, 0xBD, 0, 0xBE, 100, 0xE7, 60, 100, 0x98, 0xB1];
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
    controller.set_loop_and_transition(LoopAndTransitionOptions {
        fade_on_end: true,
        fade_seconds: 1.0,
        ..LoopAndTransitionOptions::none()
    });
    let config = PerDeviceSettings::neutral();

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
        let mut out = vec![0.0f32; (0.3 * sr) as usize * 2];
        controller.fill(&mut out, &config);
        let tail = &out[out.len() / 2..];
        tail.iter().sum::<f32>() / tail.len() as f32
    };

    assert!(
        mean_of(false).abs() > 0.05,
        "with DC removal off the biased sample should leave a clear offset"
    );
    assert!(
        mean_of(true).abs() < 0.01,
        "with DC removal on the rendered signal should be centered"
    );
}

#[test]
fn sample_dc_stats_report_the_offset_removed() {
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

    let centered = load_all(&build_rom()).remove(0).waveform_dc_stats(0);
    assert!(
        centered[0].dc_shift < 1e-6,
        "a symmetric wave has no DC offset"
    );
}

#[test]
fn audio_extraction_strips_non_audio_and_plays_identically() {
    let mut rom = build_rom();
    rom[0x100] = 0xAB;
    rom[0x1F0] = 0xCD;

    let data: Box<dyn SoundData> = load_all(&rom).remove(0);
    let Some(gba) = data.as_any().downcast_ref::<GbaRom>() else {
        panic!("expected a GBA archive")
    };
    let extracted = gba.extract_audio();

    assert!(extracted.len() <= rom.len());
    assert_eq!(extracted.get(0x100).copied().unwrap_or(0), 0);
    assert_eq!(extracted.get(0x1F0).copied().unwrap_or(0), 0);

    let stripped: Box<dyn SoundData> = load_all(&extracted).remove(0);
    assert_eq!(stripped.song_ids(), data.song_ids());

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
    let mut rom = vec![0u8; ROM_LEN];
    rom[0xB2] = 0x96;
    assert!(load_all(&rom).is_empty());
}

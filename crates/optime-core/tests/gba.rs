//! End-to-end GBA (MP2K) test against a synthetic in-memory ROM: a one-track song playing a
//! single DirectSound note. Exercises ROM parsing (heuristic song-table scan), the sequencer,
//! the player's channel/envelope model, and the shared synthesis path.

use optime_core::devices::gba::GbaRom;
use optime_core::devices::gba::param_player::{ParamPlayer, TrackParams};
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

/// The parameter-driven player (the VST3 seam) starts a note through the same `ply_note_with` the
/// sequencer uses, so a DAW note sounds like a song note: same voicegroup, same envelope, same
/// channel allocation. Only the driver differs — there is no bytecode here at all.
#[test]
fn param_player_renders_a_note_without_any_bytecode() {
    let rom: std::sync::Arc<[u8]> = build_rom().into();
    let mut player = ParamPlayer::new(rom, VOICEGROUP, 16);
    player.set_track_params(
        0,
        &TrackParams {
            vol: 100,
            ..TrackParams::default()
        },
    );
    player.note_on(0, 60, 100);

    let sr = 32768.0;
    let mut controller = SynthController::with_player(sr, Box::new(player));
    let config = PerDeviceSettings::neutral();

    let mut out = vec![0.0f32; 2 * (0.5 * sr) as usize];
    controller.fill(&mut out, &config);

    let peak = out.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    assert!(peak > 0.01, "the note should be audible, peak={peak}");
}

/// MP2K bakes a gate time into a note and `gate_tick` releases the channel when it expires. A DAW
/// note-off is asynchronous, so notes start with gate 0 — MP2K's own tie — and hold until released.
///
/// This pins that: the same note that the bytecode song ends after 24 steps (≈0.4 s, see
/// `renders_audio_and_ends`) is still sounding here well past that, and only stops once
/// `note_off` runs the engine's release.
#[test]
fn param_player_holds_a_note_until_note_off() {
    let rom: std::sync::Arc<[u8]> = build_rom().into();
    let sr = 32768.0;
    let config = PerDeviceSettings::neutral();

    let mut player = ParamPlayer::new(rom, VOICEGROUP, 16);
    player.set_track_params(
        0,
        &TrackParams {
            vol: 100,
            ..TrackParams::default()
        },
    );
    player.note_on(0, 60, 100);
    let mut controller = SynthController::with_player(sr, Box::new(player));

    // Well past the 24-step gate the bytecode version of this note would have expired at.
    let mut held = vec![0.0f32; 2 * (1.5 * sr) as usize];
    controller.fill(&mut held, &config);
    let tail = &held[held.len() - 2 * (sr as usize / 10)..];
    let held_peak = tail.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    assert!(
        held_peak > 0.01,
        "a gate-0 note should still be sounding at 1.5 s, peak={held_peak}"
    );

    // Release it, then let the envelope's release (200) run out.
    controller
        .player_mut()
        .as_any_mut()
        .downcast_mut::<ParamPlayer>()
        .expect("the controller should still hold the ParamPlayer")
        .note_off(0, 60);
    let mut released = vec![0.0f32; 2 * (2.0 * sr) as usize];
    controller.fill(&mut released, &config);
    let tail = &released[released.len() - 2 * (sr as usize / 10)..];
    let released_peak = tail.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    assert!(
        released_peak < 0.001,
        "after note_off the release should have run out, peak={released_peak}"
    );
}

/// Rip capture reads a running song's registers back out as parameters, so `from_track` must be a
/// true inverse of `set_track_params`: what the engine holds after applying a parameter set must
/// read back as that same set. If these two drift, a capture records something the plugin cannot
/// reproduce — which is the whole point of the feature.
#[test]
fn track_params_round_trip_through_the_engine() {
    let rom: std::sync::Arc<[u8]> = build_rom().into();
    let mut player = ParamPlayer::new(rom, VOICEGROUP, 16);

    let want = TrackParams {
        vol: 90,
        pan: -20,
        bend: 15,
        bend_range: 7,
        mod_: 30,
        lfo_speed: 40,
        lfo_delay: 5,
        mod_type: 1,
        tune: -8,
        key_shift: 3,
        priority: 60,
        echo_volume: 12,
        echo_length: 4,
        // The tone fields only reach the engine with the override on (see the test below), so a
        // round trip has to enable it to cover them.
        tone_override: true,
        kind: 0,
        attack: 200,
        decay: 10,
        sustain: 100,
        release: 50,
        length: 3,
        pan_sweep: 9,
        ..TrackParams::default()
    };
    player.set_track_params(0, &want);

    // `prog`/`tone_override` are not stored on the track (MP2K discards the VOICE index), so they
    // are supplied — that asymmetry is exactly why `from_track` takes them.
    let got = TrackParams::from_track(&player.tracks()[0], want.prog, want.tone_override);
    assert_eq!(
        got, want,
        "engine registers must read back as the params that set them"
    );
}

/// With `tone_override` off, the voicegroup's record wins and the tone parameters are inert — they
/// are *not* what the engine plays. That asymmetry is deliberate (an untouched instance must sound
/// like the ROM), and it is why `from_track` reports the engine's real tone rather than echoing the
/// parameters back: a capture has to record what is actually sounding.
#[test]
fn tone_params_are_inert_until_tone_override_is_on() {
    let rom: std::sync::Arc<[u8]> = build_rom().into();
    let mut player = ParamPlayer::new(rom, VOICEGROUP, 16);

    player.set_track_params(
        0,
        &TrackParams {
            attack: 7,
            release: 9,
            tone_override: false,
            ..TrackParams::default()
        },
    );
    let tone = player.tracks()[0].tone;
    // The synthetic voicegroup's program 0: attack 255, release 200.
    assert_eq!(tone.attack, 255, "the voicegroup's attack, not the param's");
    assert_eq!(
        tone.release, 200,
        "the voicegroup's release, not the param's"
    );

    player.set_track_params(
        0,
        &TrackParams {
            attack: 7,
            release: 9,
            tone_override: true,
            ..TrackParams::default()
        },
    );
    let tone = player.tracks()[0].tone;
    assert_eq!(tone.attack, 7, "the override should now win");
    assert_eq!(tone.release, 9, "the override should now win");
}

/// `VOICE` throws the program index away, so capture recovers it by matching the tone against the
/// voicegroup. A tone that isn't in the voicegroup (i.e. an `XCMD` edited it) must report `None`
/// rather than a wrong index — that's the signal to capture it as a tone override instead.
#[test]
fn program_of_recovers_the_voice_index_and_rejects_an_edited_tone() {
    use optime_core::devices::gba::m4a::ToneData;
    use optime_core::devices::gba::param_player::program_of;

    let rom = build_rom();
    let tone = ToneData::read(&rom, VOICEGROUP);
    assert_eq!(
        program_of(&rom, VOICEGROUP, &tone),
        Some(0),
        "program 0's own record must resolve back to 0"
    );

    let edited = ToneData {
        attack: tone.attack.wrapping_sub(1),
        ..tone
    };
    assert_eq!(
        program_of(&rom, VOICEGROUP, &edited),
        None,
        "a tone no voicegroup entry matches means XCMD edited it"
    );
}

/// The note stream must carry the sequence's own velocity.
///
/// `NoteStarted::volume` is the *envelope* level at note-on, which is 0 on GBA (an attack starts
/// from silence) — so it can never stand in for velocity. A MIDI export needs the note's dynamic,
/// and on GBA velocity is read per *channel*, so it stays right even when a track plays a chord
/// (consecutive `N` commands in one frame, each with its own velocity).
#[test]
fn note_started_carries_the_sequences_velocity() {
    let rom = build_rom();
    let data: Box<dyn SoundData> = load_all(&rom).remove(0);
    let mut player = data.make_player(0).expect("song 0 should load");
    let config = PerDeviceSettings::neutral();
    let mut feedback = optime_core::devices::TickFeedback::default();
    let mut events = Vec::new();

    // The track is `N24 key=60 vel=100`; find the note-on it produces.
    let mut started = None;
    for _ in 0..16 {
        events.clear();
        player.tick(&mut feedback, &config, &mut events);
        for ev in &events {
            if let optime_core::devices::SynthEvent::NoteStarted {
                key,
                velocity,
                volume,
                ..
            } = ev
            {
                started = Some((*key, *velocity, *volume));
            }
        }
        if started.is_some() {
            break;
        }
    }

    let (key, velocity, volume) = started.expect("the song's note should start");
    assert_eq!(key, 60, "the key the track asked for");
    assert_eq!(velocity, 100, "the velocity the track asked for");
    assert_eq!(
        volume, 0.0,
        "and `volume` is the envelope's starting level — which is exactly why velocity is needed"
    );
}

/// The note tap is what turns a playing song back into a MIDI clip: rip capture records what it
/// reports. So it must produce a matched on/off pair carrying the sequence's real velocity, and it
/// must be silent unless asked — plain playback must not pay for it.
#[test]
fn note_taps_report_the_songs_notes_with_velocity() {
    let rom = build_rom();
    let data: Box<dyn SoundData> = load_all(&rom).remove(0);
    let sr = 32768.0;
    let config = PerDeviceSettings::neutral();

    // Off by default.
    let mut quiet = SynthController::new(sr, &*data, 0).expect("song 0 should load");
    let mut out = vec![0.0f32; 2 * (1.0 * sr) as usize];
    quiet.fill(&mut out, &config);
    assert_eq!(
        quiet.take_note_taps().count(),
        0,
        "recording must be opt-in — plain playback taps nothing"
    );

    let mut controller = SynthController::new(sr, &*data, 0).expect("song 0 should load");
    controller.set_record_notes(true);

    // The song is `N24 key=60 vel=100` (24 steps ≈ 0.4 s at 1 step/frame), then FINE.
    let mut out = vec![0.0f32; 2 * (1.0 * sr) as usize];
    controller.fill(&mut out, &config);
    let taps: Vec<_> = controller.take_note_taps().collect();

    let ons: Vec<_> = taps.iter().filter(|t| t.velocity.is_some()).collect();
    let offs: Vec<_> = taps.iter().filter(|t| t.velocity.is_none()).collect();
    assert_eq!(ons.len(), 1, "one note-on: {taps:?}");
    assert_eq!(offs.len(), 1, "and a matching note-off: {taps:?}");

    assert_eq!(ons[0].track, 0);
    assert_eq!(ons[0].key, 60, "the key the song plays");
    assert_eq!(
        ons[0].velocity,
        Some(100),
        "the velocity the song plays it at"
    );
    assert_eq!(offs[0].key, 60, "the off must name the same key");
    assert!(
        offs[0].frame > ons[0].frame,
        "the note must end after it starts: on@{} off@{}",
        ons[0].frame,
        offs[0].frame
    );

    // Draining is destructive — a host performs each tap exactly once.
    assert_eq!(controller.take_note_taps().count(), 0, "taps drain");
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

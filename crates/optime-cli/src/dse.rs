//! DSE (Explorers of Sky) decode spike.
//!
//! Proves the SMDL/SWDL decode path end to end against real data: parses a main bank + a song
//! sequence, decodes the song's samples to `.wav`, and disassembles each track's bytecode using
//! the decomp-derived event table.
//!
//! The `files` mode reads the SWDL/SMDL files directly; the `rom` mode runs the same data through
//! the public `SoundData`/`DevicePlayer` pipeline the app uses.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args as ClapArgs, Subcommand};
use optime_core::devices::dse::{DseEvent, Smdl, Swdl, decode_track};
use optime_core::synth_controller::messages::TickFeedback;
use optime_core::waveform::Waveform;
use optime_core::{FsVisController, PerDeviceSettings, SynthController, SynthEvent, load_all};

#[derive(ClapArgs)]
#[command(about = "Dump a DSE (Explorers of Sky) bank + sequence, proving the decode path.")]
pub struct Args {
    #[command(subcommand)]
    mode: Mode,
}

#[derive(Subcommand)]
enum Mode {
    /// Parse a main bank and a song sequence from loose SWDL/SMDL files.
    Files {
        /// The main bank `bgm.swd`, which holds the sample data.
        main_bank: PathBuf,
        /// The song `bgm####.smd`.
        song: PathBuf,
        /// The per-song `bgm####.swd`, parsed for program info.
        song_bank: Option<PathBuf>,
        /// Directory the decoded sample WAVs are written to.
        #[arg(default_value = ".")]
        out_dir: String,
    },
    /// Run a PMD `.nds` ROM through the full `SoundData`/`DevicePlayer` pipeline.
    Rom {
        /// The PMD `.nds` ROM.
        rom: PathBuf,
        /// Which song to play.
        #[arg(default_value_t = 0)]
        song_id: u32,
    },
}

pub fn run(args: Args) -> ExitCode {
    let (main_bank_path, song_path, song_bank_path, out_dir) = match args.mode {
        Mode::Rom { rom, song_id } => {
            rom_mode(&rom, song_id);
            return ExitCode::SUCCESS;
        }
        Mode::Files {
            main_bank,
            song,
            song_bank,
            out_dir,
        } => (main_bank, song, song_bank, out_dir),
    };

    let main_bank = std::fs::read(main_bank_path).expect("read main bank .swd");
    let smd = std::fs::read(song_path).expect("read song .smd");
    let song_bank = song_bank_path.map(|p| std::fs::read(p).expect("read song .swd"));

    // --- Main bank ---
    let main = Swdl::parse(&main_bank).expect("parse main bank");
    println!("== MAIN BANK  '{}'  (v{:#06x}) ==", main.name, main.version);
    println!(
        "   {} samples, pcmd payload {} bytes",
        main.waveforms.len(),
        main.pcmd.len()
    );
    let rates: Vec<u32> = main
        .waveforms
        .iter()
        .take(8)
        .map(|s| s.sample_rate)
        .collect();
    println!("   first sample rates (Hz): {rates:?}");
    println!();

    // --- Song bank (optional) ---
    if let Some(bytes) = &song_bank
        && let Some(bank) = Swdl::parse(bytes)
    {
        println!(
            "== SONG BANK  '{}'  ({} wavi refs, {} programs) ==",
            bank.name,
            bank.waveforms.len(),
            bank.programs.len()
        );
        for prog in bank.programs.iter().take(4) {
            println!(
                "   program {:>3}: {} split(s), vol {}",
                prog.id,
                prog.splits.len(),
                prog.volume
            );
            // The faithful pitch model: a split's key_base (~ -1745 = the classic ctune -7 in
            // 8.8 fixed point) plus note_delta give the absolute playback rate per key.
            if let Some(split) = prog.splits.first() {
                let note_key =
                    i32::from(split.key_base) + (i32::from(split.note_delta) << 8) + (60i32 << 8);
                let hz = optime_core::devices::dse::note_key_to_hz(note_key);
                println!(
                    "      split0: key_base={} note_delta={} wave={} -> key 60 plays at {hz:.0} Hz",
                    split.key_base, split.note_delta, split.wave_index
                );
            }
        }
        println!();
    }

    // --- Song sequence ---
    let song = Smdl::parse(&smd).expect("parse song .smd");
    println!(
        "== SONG  '{}'  (v{:#06x}, TPQN={}, {} tracks) ==\n",
        song.name,
        song.version,
        song.tpqn,
        song.tracks.len()
    );

    // Disassemble the first music track (track 0 is usually a tiny control track).
    for track in song.tracks.iter().take(3) {
        let events = decode_track(&track.events, 4);
        println!(
            "--- track {} (channel {}): {} bytes, {} events ---",
            track.track_id,
            track.channel_id,
            track.events.len(),
            events.len()
        );
        for ev in events.iter().take(24) {
            print_event(ev);
        }
        if events.len() > 24 {
            println!("    … ({} more)", events.len() - 24);
        }
        println!();
    }

    // --- Run the sequencer over the song to prove the interpreter ---
    {
        use optime_core::devices::dse::{DseSequencer, SeqOp};
        let mut seq = DseSequencer::new(&song);
        let mut ops = Vec::new();
        let mut all = Vec::new();
        // ~20s of sequencer ticks at this song's tempo (TPQN * a few hundred beats).
        for _ in 0..8000 {
            ops.clear();
            seq.seq_tick(&mut ops);
            all.append(&mut ops);
            if seq.ended {
                break;
            }
        }
        let mut notes_per_track = [0u32; 16];
        let mut tempos = Vec::new();
        let mut programs = 0;
        for op in &all {
            match op {
                SeqOp::NoteOn { track, .. } => notes_per_track[*track] += 1,
                SeqOp::Tempo { bpm } => tempos.push(*bpm),
                SeqOp::Program { .. } => programs += 1,
                _ => {}
            }
        }
        let total: u32 = notes_per_track.iter().sum();
        println!(
            "== SEQUENCER: {} ticks, {} notes total, {} program changes, tempos {:?} ==",
            seq.ticks_elapsed, total, programs, tempos
        );
        println!("   notes/track: {notes_per_track:?}");
        println!("   final bpm {}, ended={}\n", seq.bpm, seq.ended);
    }

    // --- Tick a player to prove the LFO path (vibrato / tremolo / auto-pan) ---
    if let Some(bytes) = &song_bank
        && let Some(bank) = Swdl::parse(bytes)
    {
        use optime_core::DevicePlayer as _;
        use optime_core::devices::dse::DsePlayer;
        use std::sync::Arc;
        let mut player = DsePlayer::new(&song, Arc::new(bank), Arc::new(main.clone()));
        let mut feedback = TickFeedback::default();
        let cfg = PerDeviceSettings::neutral();
        let mut events = Vec::new();
        let (mut vib, mut bends, mut max_bend, mut pans, mut notes) =
            (0u32, 0u32, 0.0f64, 0u32, 0u32);
        for _ in 0..2000 {
            events.clear();
            player.tick(&mut feedback, &cfg, &mut events);
            for ev in &events {
                match ev {
                    SynthEvent::NoteStarted { .. } => notes += 1,
                    SynthEvent::VoiceDetune { semitones, .. } if semitones.abs() > 1e-9 => {
                        vib += 1;
                    }
                    SynthEvent::TrackDetune { semitones, .. } if semitones.abs() > 1e-9 => {
                        bends += 1;
                        max_bend = max_bend.max(semitones.abs());
                    }
                    SynthEvent::TrackPan { .. } => pans += 1,
                    _ => {}
                }
            }
        }
        println!(
            "== effects over ~20s: {notes} notes, {vib} vibrato (VoiceDetune), {bends} pitch \
                 bends (TrackDetune, max {max_bend:.2} st), {pans} pan updates ==\n"
        );
    }

    // --- Decode a few samples to WAV to prove the sample path ---
    println!("== decoding samples to WAV in '{out_dir}' ==");
    std::fs::create_dir_all(&out_dir).expect("create out_dir");
    let mut written = 0;
    for info in main.waveforms.iter().take(6) {
        match main.decode_waveform(info, &main.pcmd) {
            Some(waveform) => {
                let path = format!("{out_dir}/dse_sample_{:03}.wav", info.id);
                write_wav(&path, &waveform);
                println!(
                    "   sample {:>3}: {:?}, {} Hz, root {}, {} samples -> {}",
                    info.id,
                    info.format,
                    waveform.sample_rate as u32,
                    info.root_key,
                    waveform.data.len(),
                    Path::new(&path).file_name().unwrap().to_string_lossy(),
                );
                written += 1;
            }
            None => println!(
                "   sample {:>3}: {:?} (skipped/undecodable)",
                info.id, info.format
            ),
        }
    }
    println!("\nwrote {written} WAV file(s).");
    ExitCode::SUCCESS
}

/// Loads a PMD `.nds` ROM through the public `SoundData` API and ticks `DevicePlayer` for song
/// `song_id`, reporting the standardized synth events the audio layer would consume.
fn rom_mode(path: &Path, song_id: u32) {
    let bytes = std::fs::read(path).expect("read ROM");
    let archives = load_all(&bytes);
    let Some(data) = archives.first() else {
        eprintln!("No DSE/SDAT/GBA archive found in {}", path.display());
        std::process::exit(1);
    };
    let ids = data.song_ids();
    println!("== SoundData: {} songs ==", ids.len());
    for &id in ids.iter().take(8) {
        println!(
            "   song {id}: {}",
            data.song_name(id).unwrap_or_else(|| "(unnamed)".into())
        );
    }
    println!(
        "   ... selecting song {song_id}: {}\n",
        data.song_name(song_id)
            .unwrap_or_else(|| "(unnamed)".into())
    );

    let Some(mut player) = data.make_player(song_id) else {
        eprintln!("song {song_id} is out of range (only {} songs)", ids.len());
        std::process::exit(1);
    };
    let config = PerDeviceSettings::neutral();
    let mut feedback = TickFeedback::default();
    let mut events = Vec::new();

    let (mut started, mut stopped, mut released) = (0u32, 0u32, 0u32);
    let (mut min_vol, mut max_vol) = (f64::INFINITY, 0.0f64);
    let mut keys: Vec<u8> = Vec::new();
    // ~10 seconds at the ~100 Hz driver tick.
    for _ in 0..1000 {
        events.clear();
        player.tick(&mut feedback, &config, &mut events);
        for ev in &events {
            match ev {
                SynthEvent::NoteStarted { key, volume, .. } => {
                    started += 1;
                    keys.push(*key);
                    min_vol = min_vol.min(*volume);
                    max_vol = max_vol.max(*volume);
                }
                SynthEvent::VoiceStopped { .. } => stopped += 1,
                SynthEvent::NoteReleased { .. } => released += 1,
                _ => {}
            }
        }
    }
    let (klo, khi) = (
        keys.iter().copied().min().unwrap_or(0),
        keys.iter().copied().max().unwrap_or(0),
    );
    println!(
        "== DevicePlayer over ~10s ({} ticks) ==",
        player.steps_elapsed()
    );
    println!("   NoteStarted={started}, NoteReleased={released}, VoiceStopped={stopped}");
    println!(
        "   note key range {klo}..={khi}, NoteStarted volume range {min_vol:.3}..={max_vol:.3}"
    );
    println!(
        "   step_rate {:.1} Hz (~{:.0} BPM)",
        player.step_rate(),
        player.step_rate() * 60.0 / 48.0
    );

    // The look-ahead visualizer path: render the whole-song overview the piano roll uses.
    if let Some(overview) = FsVisController::overview(&**data, song_id) {
        let dur: u32 = overview.notes.iter().map(|n| n.duration).sum();
        println!("== look-ahead overview ==");
        println!(
            "   {} notes over {} steps, {} tempo change(s), avg note {} steps",
            overview.notes.len(),
            overview.total_steps,
            overview.tempos.len(),
            dur.checked_div(overview.notes.len() as u32).unwrap_or(0),
        );
    }

    // Render ~10s of real mixed audio through the SynthController and report the output level,
    // to check the per-voice volume calibration doesn't drive the mix into hard clipping.
    if let Some(mut ctrl) = SynthController::new(32_768.0, &**data, song_id) {
        let mut buf = vec![0f32; 32_768 * 2 * 10];
        ctrl.fill(&mut buf, &config);
        let peak = buf.iter().fold(0f32, |m, &s| m.max(s.abs()));
        let clipped = buf.iter().filter(|&&s| s.abs() >= 1.0).count();
        let rms = (buf
            .iter()
            .map(|&s| f64::from(s) * f64::from(s))
            .sum::<f64>()
            / buf.len() as f64)
            .sqrt();
        println!("== mixed output over ~10s ==");
        println!(
            "   peak {peak:.3}, rms {rms:.3}, {} clipped samples ({:.2}%)",
            clipped,
            100.0 * clipped as f64 / buf.len() as f64
        );
    }
}

fn print_event(ev: &DseEvent) {
    match ev {
        DseEvent::Note {
            velocity,
            key,
            duration,
            ..
        } => println!(
            "    Note   key={key:>3} vel={velocity:>3} dur={}",
            duration
                .map(|d| d.to_string())
                .unwrap_or_else(|| "(prev)".into())
        ),
        DseEvent::Pause { ticks } => println!("    Pause  {ticks} ticks"),
        DseEvent::Control {
            opcode,
            name,
            operands,
        } => println!("    {name} ({opcode:#04x}) {operands:02x?}"),
        DseEvent::Invalid { opcode } => println!("    <invalid {opcode:#04x}>"),
    }
}

/// Writes a mono 16-bit PCM WAV from a normalized [`Waveform`].
fn write_wav(path: &str, waveform: &Waveform) {
    let rate = waveform.sample_rate as u32;
    let n = waveform.data.len();
    let data_bytes = (n * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_bytes as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    for &s in &waveform.data {
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(path, out).expect("write wav");
}

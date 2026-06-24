//! DSE (Explorers of Sky) decode spike.
//!
//! Proves the SMDL/SWDL decode path end to end against real data: parses a main bank + a song
//! sequence, decodes the song's samples to `.wav`, and disassembles each track's bytecode using
//! the decomp-derived event table.
//!
//! Usage:
//!   cargo run -p optime-core --example dump_dse -- <bgm.swd> <bgm####.smd> [bgm####.swd] [out_dir]
//!
//! `bgm.swd` is the main bank (sample data); `bgm####.smd` is the song; the optional per-song
//! `.swd` is parsed for program info. Decoded WAVs are written to `out_dir` (default: cwd).

use optime_core::devices::dse::{decode_track, DseEvent, Smdl, Swdl};
use optime_core::sample::Sample;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!(
            "usage: dump_dse <bgm.swd> <bgm####.smd> [bgm####.swd] [out_dir]\n\
             e.g.  dump_dse '/d/Git/pmd-sky/files/SOUND/BGM/bgm.swd' \\\n\
                          '/d/Git/pmd-sky/files/SOUND/BGM/bgm0001.smd'"
        );
        std::process::exit(2);
    }

    let main_bank = std::fs::read(&args[0]).expect("read main bank .swd");
    let smd = std::fs::read(&args[1]).expect("read song .smd");
    let song_bank = args
        .get(2)
        .map(|p| std::fs::read(p).expect("read song .swd"));
    let out_dir = args.get(3).cloned().unwrap_or_else(|| ".".to_string());

    // --- Main bank ---
    let main = Swdl::parse(&main_bank).expect("parse main bank");
    println!("== MAIN BANK  '{}'  (v{:#06x}) ==", main.name, main.version);
    println!(
        "   {} samples, pcmd payload {} bytes",
        main.samples.len(),
        main.pcmd.len()
    );
    let rates: Vec<u32> = main.samples.iter().take(8).map(|s| s.sample_rate).collect();
    println!("   first sample rates (Hz): {rates:?}");
    println!();

    // --- Song bank (optional) ---
    if let Some(bytes) = &song_bank {
        if let Some(bank) = Swdl::parse(bytes) {
            println!(
                "== SONG BANK  '{}'  ({} wavi refs, {} programs) ==",
                bank.name,
                bank.samples.len(),
                bank.programs.len()
            );
            for prog in bank.programs.iter().take(4) {
                println!(
                    "   program {:>3}: {} split(s), vol {}",
                    prog.id,
                    prog.splits.len(),
                    prog.volume
                );
            }
            println!();
        }
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

    // --- Decode a few samples to WAV to prove the sample path ---
    println!("== decoding samples to WAV in '{out_dir}' ==");
    let mut written = 0;
    for info in main.samples.iter().take(6) {
        match main.decode_sample(info, &main.pcmd) {
            Some(sample) => {
                let path = format!("{out_dir}/dse_sample_{:03}.wav", info.id);
                write_wav(&path, &sample);
                println!(
                    "   sample {:>3}: {:?}, {} Hz, root {}, {} samples -> {}",
                    info.id,
                    info.format,
                    sample.sample_rate as u32,
                    info.root_key,
                    sample.data.len(),
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

/// Writes a mono 16-bit PCM WAV from a normalized [`Sample`].
fn write_wav(path: &str, sample: &Sample) {
    let rate = sample.sample_rate as u32;
    let n = sample.data.len();
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
    for &s in &sample.data {
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(path, out).expect("write wav");
}

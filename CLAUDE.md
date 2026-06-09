# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Optime Player emulates the Nintendo DS sound system: it loads `.nds` ROMs or standalone `.sdat` sound archives, parses the SDAT container, and software-synthesizes the SSEQ (sequenced music) format in real time. Much of the sound-engine logic comes from the reverse-engineered pokediamond project (https://github.com/pret/pokediamond).

The project is a **Rust workspace** with an `egui`/`eframe` UI that runs both natively and on the web (wasm). The synthesis core began as a faithful port of the original JavaScript engine but is now driven toward **pokediamond accuracy** (the DS sound ROM decomp) where the two disagree — verified by Rust unit tests transcribed from the pokediamond C, not by JS parity. The original browser app is preserved verbatim under `legacy-js/` (see "Legacy app" below) but is no longer the primary codebase or the accuracy oracle.

## Commands

- `cargo run -p optime-app` — run the native desktop app (loads a demo from `demos/` on start)
- `cargo test --workspace` — run all tests (unit + SDAT integration + pokediamond-reference tests)
- `cargo clippy --workspace --all-targets` — lint (kept warning-clean)
- `cargo fmt --all` — format
- `cargo clippy -p optime-app --target wasm32-unknown-unknown` — type-check the web build
- Web dev server: `cd crates/optime-app && trunk serve` (needs `rustup target add wasm32-unknown-unknown` and `cargo install trunk`)
- Web release build: `cd crates/optime-app && trunk build --release`

The web app deploys to GitHub Pages via `.github/workflows/deploy.yml` (Trunk release build with `--public-url "/<repo>/"` → Pages artifact). It triggers on pushes to `rust-rewrite`/`master`.

## Architecture

Two crates:

### `crates/optime-core` — the platform-independent engine (no browser/UI/audio deps)

Pure-`std` port of the JS engine, organized one concept per module:

- `tables.rs` — DS BIOS / pokediamond lookup tables (reproduced bit-identically; correctness depends on this), plus `snd_sin_idx`.
- `util.rs` — `read_u8/u16/u32` (return 0 on OOB), `search_for_sequence`, generic `CircularBuffer<T>`.
- `dsp.rs` — `BiquadFilter` (cascaded biquads).
- `tuning.rs` — `midi_note_to_hz` for equal temperament and Pythagorean "pure" tuning.
- `sample.rs` — `Sample` + `decode_pcm8`/`decode_pcm16`/`decode_adpcm`/`decode_wav`.
- `bank.rs` — `InstrumentBank`/`InstrumentRecord`/`InstrumentRegion`, ADSR coefficient helpers.
- `sdat.rs` — the `Sdat` parser (SYMB/INFO/FAT/SBNK), `*Info` records.
- `sequence.rs` — `Sequence` + `SequenceTrack`, the SSEQ bytecode interpreter, emitting `Message`s.
- `synth.rs` — `SampleInstrument`, `SampleSynthesizer` (16 round-robin voices), `DelayLine` (Haas stereo).
- `controller.rs` — `Controller` (note lifecycle, ADSR/LFO, the DS master clock) and `FsVisController` (look-ahead visualizer driver).

Data flow: `Sdat::load_all(&[u8])` → `Controller::new(sample_rate, &sdat, sseq_id)` decodes the linked sample archives up front → `Controller::next_sample(&config)` advances the master clock and returns one stereo frame (`fill` does buffers). The `Sequence` interpreter emits `Message`s (`PlayNote`, `VolumeChange`, `PitchBend`, `Jump`, `TrackEnded`, …) that the `Controller` consumes in `handle_message`.

Runtime options that were JS globals (`g_enableStereoSeparation`, `g_usePureTuning`, `g_trackEnables`, …) are now a `SynthConfig` struct threaded into the synthesis calls — there is no global state.

**The master clock** lives in one place: `Controller::next_sample` accumulates `DS_CLOCK_RATE` (33,513,982) per sample and ticks the sequence every `CYCLES_PER_TICK * sample_rate` cycles (`CYCLES_PER_TICK = 64 * 2728`), with the BPM timer `+= bpm` / `while >= 240`. The offline WAV renderer (`optime-app/src/player.rs`) drives this same method.

### `crates/optime-app` — the eframe/egui front-end (native + web)

- `app.rs` — `OptimeApp` (the `eframe::App`): song list, transport, settings, WAV export, live keyboard input. Demo loading is platform-split: native reads `demos/`, web fetches the `.sdat` at runtime (`web::fetch_bytes`).
- `audio.rs` — `AudioEngine`: a `cpal` f32 output stream (native + WebAudio) pulling from shared state.
- `player.rs` — `AudioState` shared between the UI and the audio callback via `Arc<Mutex<…>>`, plus `render_to_samples` (offline loop+fadeout render for WAV export).
- `visualizer.rs` — procedural 16-track × 88-key piano roll drawn with `egui::Painter` (no image assets; original PNGs remain in `legacy-js`).
- `wav.rs` — minimal 16-bit PCM WAV encoder.
- `web.rs` (wasm only) — browser download + demo fetch helpers.
- `main.rs` — native window entry; on wasm mounts onto `#the_canvas_id` (see `index.html`, built by Trunk).

`SynthConfig` is rebuilt each frame from UI mirrors and pushed into the audio thread under the lock; the same lock is used to snapshot `notes_on` for the visualizer.

## Testing

- `crates/optime-core/tests/sdat.rs` — parses the real `demos/*.sdat` archives.
- **pokediamond-reference unit tests** — the accuracy anchor. Synthesis math is validated by transcribing the relevant `pret/pokediamond` C function directly into a Rust test oracle and asserting the engine matches it across a parameter grid (e.g. `lfo_tick_matches_pokediamond_reference` and `calc_channel_volume_matches_pokediamond` in `controller.rs`). When porting a new accuracy fix, add a transcription test rather than a recorded golden fixture. (The former Node/`gen_fixture.js` golden-parity fixture against the legacy JS was removed — the legacy JS is no longer the oracle.)
- Each `optime-core` module has inline `#[cfg(test)]` unit tests (decoders, ADSR helpers, tuning, biquad, circular buffer, opcode interpreter).

## Conventions

- Idiomatic, zero-extra-dependency `std` in `optime-core`; keep it free of UI/audio/browser deps so both the app and the headless tests reuse it.
- Binary parsing uses `from_le_bytes` via the `util::read_*` helpers (return 0 on out-of-bounds rather than panicking).
- Keep `cargo clippy --workspace --all-targets` and `cargo fmt --all --check` clean.
- When touching synthesis math, check it against `pret/pokediamond` and add/adjust a transcription unit test that pins the behavior to the C reference.

## Legacy app

`legacy-js/` holds the original browser app verbatim: `OptimePlayer/OptimePlayer.js` (engine + parser), `OptimePlayer/dsp.js`, `index.js` (DOM glue), `index.html`, plus the ffmpeg/canvas exporters `video-exporter.js` / `playlist-exporter.js` (intentionally **not** ported — the Rust app provides in-app WAV export instead). It is kept only as a historical reference (no longer the accuracy oracle). `demos/`, `assets/`, and `fonts/` remain at the repo root and are shared.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Optime Player emulates retro console sound systems and plays their sequenced music in real time:

- **Nintendo DS** — loads `.nds` ROMs or standalone `.sdat` sound archives, parses the SDAT container, and synthesizes the SSEQ format. The engine logic comes from the reverse-engineered pokediamond project (https://github.com/pret/pokediamond).
- **Game Boy Advance** — loads `.gba` ROMs running the MP2K ("Sappy" / `m4a`) engine, locating the song table by code signature or heuristic scan. The engine logic comes from the pokeemerald decompilation (https://github.com/pret/pokeemerald), cloned locally at `~/Git/pokeemerald` for reference.

The project is a **Rust workspace** with an `egui`/`eframe` UI that runs both natively and on the web (wasm). Synthesis accuracy is anchored to the pret decompilations (pokediamond for DS, pokeemerald for GBA), verified by Rust unit tests transcribed from the C — not by parity with the original JavaScript app, which is preserved verbatim under `legacy-js/` (see "Legacy app" below) purely as a historical reference.

## Commands

The workspace pins **nightly Rust** (`rust-toolchain.toml`): the default `simd` cargo feature uses `std::simd` (portable SIMD) in the sinc resampler. `--no-default-features` on `optime-core` builds the scalar gather (works on stable). `.cargo/config.toml` sets `-C target-cpu=native` for native x86-64/aarch64 hosts — the hot path relies on AVX2/FMA codegen.

- `cargo run -p optime-app` — run the native desktop app (loads a demo from `demos/` on start)
- `cargo test --workspace` — run all tests (unit + integration + pret-reference transcription tests)
- `cargo clippy --workspace --all-targets` — lint (kept warning-clean)
- `cargo fmt --all` — format
- `cargo clippy -p optime-app --target wasm32-unknown-unknown` — type-check the web build
- Web dev server: `cd crates/optime-app && trunk serve` (needs `rustup target add wasm32-unknown-unknown` and `cargo install trunk`)
- Web release build: `cd crates/optime-app && trunk build --release`

The web app deploys to GitHub Pages via `.github/workflows/deploy.yml` (Trunk release build with `--public-url "/<repo>/"` → Pages artifact). It triggers on pushes to `rust-rewrite`/`master`.

## Architecture

Two crates. The core's data flow is a strict pipeline — each device parses its own formats and emits one **standardized event stream**; the synthesis layer knows nothing about any console:

```text
ROM / archive bytes ─► devices::SoundData          (per-console parsing: songs, instruments)
SoundData + song id ─► devices::DevicePlayer       (per-console sequencer + envelope model)
DevicePlayer::tick  ─► devices::SynthEvent stream  (the standardized message set)
SynthEvent stream   ─► SynthController             (voice pools, master clock, mixing)
```

### `crates/optime-core` — the platform-independent engine (no browser/UI/audio deps)

Pure-`std`. Top-level modules are the console-agnostic synthesis layer; everything console-specific lives in a folder under `devices/`.

**`devices/`** — one folder per console, plus the shared interface:

- `devices/mod.rs` — `SoundData` (enum over parsed archives: `load_all`, `song_ids`, `song_name`, `make_player`) and `DevicePlayer` (enum over running players: `clock_rate`, `cycles_per_tick`, `steps_elapsed`, `step_rate`, keyboard input, `tick`). Both enums box their variants.
- `devices/messages.rs` — the standardized stream: `SynthEvent` (`NoteStarted`, `VoiceVolume`, `VoicePitch`, `VoiceDetune`, `VoiceStopped`, `NoteReleased`, `TrackPan`, `TrackDetune`, `Looped`, `Ended`), `VoicePitch` (`Midi {note, sample_pitch_hz}` for repitched samples vs `DataRateHz` for register-driven rates), `VoiceId` (monotonic `u64`), and `TickFeedback` (synth-side voice endings reported back to the device each tick).
- `devices/nintendo_ds/` — `sdat.rs` (the `Sdat` parser: SYMB/INFO/FAT/SBNK), `bank.rs` (`InstrumentBank`, ADSR coefficient helpers), `sequence/` (the SSEQ bytecode interpreter: `track.rs`, `message.rs`, `interpreter.rs`, `mod.rs` with the 16 tracks / message queue / `SND_CalcRandom`), `tables.rs` (DS BIOS / pokediamond lookup tables, bit-identical), `volume.rs` / `lfo.rs` (decibel-domain channel volume and LFO math, pokediamond ports), and `player.rs` (`NdsPlayer`: decodes the linked SWAR sample archives up front, runs the sequencer behind the BPM timer, owns per-note ADSR/LFO state, emits `SynthEvent`s).
- `devices/gba/` — `rom.rs` (`GbaRom`: song-table location via the `m4aSongNumStart` signature with a brute-scan fallback, `SongHeader` parsing), `voice.rs` (`ToneData` / voicegroup resolution incl. key-split and rhythm voices, `WaveData`), `sequencer.rs` (`Mp2kSequencer`: the MPlayMain track interpreter — runs per VBlank frame, `tempo_c += tempo_i; while >= 150` steps, emits `Mp2kOp`s), `tables.rs` (`gClockTable`/`gScaleTable`/`gFreqTable`/CGB/noise tables + `MidiKeyToFreq`/`MidiKeyToCgbFreq`, bit-identical), and `player.rs` (`GbaPlayer`: DirectSound + 4 CGB channel allocation/stealing, per-frame envelopes incl. pseudo-echo, generated square/wave/noise sample data, emits `SynthEvent`s).

**Synthesis layer** (console-agnostic):

- `synth_controller/` — `mod.rs` (`SynthController`: per-track voice pools via `slot_owner` bookkeeping, the master clock, `next_sample`/`fill`/`tick`, applies `SynthEvent`s and feeds `TickFeedback` — voice steals and one-shot sample endings — back to the device), `config.rs` (`SynthConfig`), `vis.rs` (`FsVisController`: a parallel device-sequencer runner emitting `VisNote`s for look-ahead visualizers, no audio).
- `synth/` — `instrument.rs` (`SampleInstrument`, one voice: `VoicePitch`-driven frequency, resampling, block rendering), `synthesizer.rs` (`SampleSynthesizer`: round-robin voice pool, pan, Haas widening, bass-mono crossover), `delay.rs` (`DelayLine`), `gather.rs` (SIMD/scalar sample gather).
- `resample/` — windowed-sinc resampling: `kernels.rs` (FIR kernel/response, `ResampleTables`), `mod.rs` (public API).
- `sample.rs` — `Sample`, `ResampleMode`, + `decode_pcm8`/`decode_pcm16`/`decode_adpcm`/`decode_wav`.
- `dsp.rs` — `BiquadFilter` (cascaded biquads). `tuning.rs` — `midi_note_to_hz` (equal temperament / Pythagorean `TuningSystem`). `util.rs` — `read_u8/u16/u32` (return 0 on OOB), `search_for_sequence`, `CircularBuffer<T>`.

Runtime options are a `SynthConfig` struct threaded into the synthesis calls — there is no global state.

**The master clock** lives in one place: `SynthController::next_sample` accumulates the device's `clock_rate()` per output sample and ticks the player every `cycles_per_tick() * sample_rate` cycles. For the DS that is `DS_CLOCK_RATE` (33,513,982) / `CYCLES_PER_TICK` (64 × 2728) ≈ 192 Hz with the BPM timer `+= bpm` / `while >= 240` inside the player; for the GBA it is `GBA_CLOCK_RATE` (16,777,216) / `CYCLES_PER_FRAME` (280,896) ≈ 59.73 Hz VBlanks with the `tempo_c` accumulator inside the sequencer. `SynthController::fill` renders in tick-aligned blocks, bit-identical to the per-sample path (pinned by `tests/block_render.rs`). The visualizer timeline is *sequencer steps* (DS: SSEQ ticks; GBA: tempo steps), converted to wall time via `step_rate()`.

### `crates/optime-app` — the eframe/egui front-end (native + web)

- `app.rs` — `OptimeApp` (the `eframe::App`): song list/queue, transport, settings, WAV export, live keyboard input. Loads any `SoundData` source (`.nds`/`.sdat`/`.gba`). Demo loading is platform-split: native reads `demos/`, web fetches the `.sdat` at runtime (`web::fetch_bytes`). Shareable demo tracks via the URL query string on web.
- `library.rs` — persistent user library (`TrackRef`, `Playlist`, `Library`, `RepeatMode`) and the `Persisted` bundle saved to eframe storage. (`TrackRef::sseq_id` is the song id for any device; the field name is kept for storage compatibility.)
- `audio.rs` — `AudioEngine`: a `cpal` f32 output stream (native + WebAudio) pulling from shared state.
- `player.rs` — `AudioState` shared between the UI and the audio callback via `Arc<Mutex<…>>` (pause/fade/volume ramps, DSP-load stats), plus `render_to_samples` (offline loop+fadeout render for WAV export).
- `visualizer.rs` / `piano_roll.rs` — procedural 16-track × 88-key piano roll drawn with `egui::Painter` (no image assets; original PNGs remain in `legacy-js`). Driven by `VisSnapshot` (`steps` + `step_rate`) and the `FsVisController` look-ahead.
- `theme.rs` — egui style/typography (Inter/SF Pro).
- `filter_plot.rs` — frequency-response plot for the resampler/crossover settings UI.
- `wav.rs` — minimal 16-bit PCM WAV encoder.
- `web.rs` (wasm only) — browser download + demo fetch helpers.
- `main.rs` / `lib.rs` — native window entry; on wasm mounts onto `#the_canvas_id` (see `index.html`, built by Trunk).

`SynthConfig` is rebuilt each frame from UI mirrors and pushed into the audio thread under the lock; the same lock is used to snapshot `notes_on` for the visualizer.

## Testing

- `crates/optime-core/tests/sdat.rs` — parses the real `demos/*.sdat` archives.
- `crates/optime-core/tests/gba.rs` — end-to-end MP2K test against a synthetic in-memory GBA ROM (table scan, sequencer, player, audio output, look-ahead).
- `crates/optime-core/tests/block_render.rs` — pins `SynthController::fill` bit-identical to the per-sample path.
- **pret-reference unit tests** — the accuracy anchor. Synthesis math is validated by transcribing the relevant C function from `pret/pokediamond` (DS) or `pret/pokeemerald` (GBA) directly into a Rust test oracle and asserting the engine matches it across a parameter grid (e.g. `calc_channel_volume_matches_pokediamond` in `devices/nintendo_ds/volume.rs`, `midi_key_to_freq_matches_pokeemerald` in `devices/gba/tables.rs`). When porting a new accuracy fix, add a transcription test rather than a recorded golden fixture.
- Each `optime-core` module has inline `#[cfg(test)]` unit tests (decoders, ADSR helpers, tuning, biquad, circular buffer, opcode interpreter).

## Conventions

- Idiomatic, zero-extra-dependency `std` in `optime-core`; keep it free of UI/audio/browser deps so both the app and the headless tests reuse it.
- Console-specific code belongs under `devices/<console>/`; it talks to the synthesis layer only through `SynthEvent`/`TickFeedback`. The synthesis layer never reaches into a device.
- Prefer making invalid states unrepresentable: enums over flag/int combinations (`SoundData`, `DevicePlayer`, `VoicePitch`, `ToneKind`, `Mp2kOp`, ADSR phases), `Option` over sentinels.
- Binary parsing uses `from_le_bytes` via the `util::read_*` helpers (return 0 on out-of-bounds rather than panicking).
- Keep `cargo clippy --workspace --all-targets` and `cargo fmt --all --check` clean.
- When touching synthesis math, check it against the matching pret decomp (pokediamond for DS, pokeemerald for GBA) and add/adjust a transcription unit test that pins the behavior to the C reference.

## Legacy app

`legacy-js/` holds the original browser app verbatim: `OptimePlayer/OptimePlayer.js` (engine + parser), `OptimePlayer/dsp.js`, `index.js` (DOM glue), `index.html`, plus the ffmpeg/canvas exporters `video-exporter.js` / `playlist-exporter.js` (intentionally **not** ported — the Rust app provides in-app WAV export instead). It is kept only as a historical reference (no longer the accuracy oracle). `demos/`, `assets/`, and `fonts/` remain at the repo root and are shared.

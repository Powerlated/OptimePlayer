# CLAUDE.md

Guidance for Claude Code in this repo. Keep it current with the code.

**Write style — mandatory for all agents editing this file.** Terse, telegraphic, high-density. Drop filler, hedging, and connective prose. Fragments over sentences. Keep every technical fact (names, paths, constants, behaviors); cut only the words around them. Example: "New object ref each render. Inline object prop = new ref = re-render. Wrap in useMemo." — not "The reason your component re-renders is likely that you create a new object reference each render cycle..." Contribute new facts in this same style.

When using Claude Haiku, follow these literal rules:
* DO NOT run builds after completing a task
* DO NOT run tests after completing a task
* DO NOT modify files unless explicitly named in the request
* DO NOT modify functions unless explicitly named in the request
* DO NOT make assumptions about scope beyond what is explicitly stated
* DO NOT add extra features, refactoring, or cleanup code
* DO NOT change code outside the requested target
* DO NOT run any command at the end unless explicitly asked to do so
* ONLY perform exactly what was requested, nothing more

## Overview

Optime Player: emulates retro console sound systems, plays sequenced music real-time. Rust workspace, `egui`/`eframe` UI, native + web (wasm).

Three backends, each anchored to a pret decomp (accuracy via Rust unit tests transcribed from the C, not JS parity):

- **Nintendo DS** — `.nds` ROMs / standalone `.sdat`. Parses SDAT, synthesizes SSEQ. From pokediamond (https://github.com/pret/pokediamond).
- **GBA** — `.gba` ROMs, MP2K ("Sappy"/`m4a`). Song table via code signature or heuristic scan. From pokeemerald (https://github.com/pret/pokeemerald), local `~/Git/pokeemerald`.
- **DSE (Procyon)** — SMDL (seq) + SWDL (bank), *PMD: Explorers of Sky*. From pmd-sky (https://github.com/pret/pmd-sky), local `D:\Git\pmd-sky` (ships `files/SOUND/BGM/*.smd`/`*.swd`). Full `SoundData`/`DevicePlayer` backend (`devices/dse/`): parse + sample decode, SMDL sequencer, volume envelope, ROM-exact note→freq + square-law volume from voice-update asm. PMD `.nds` (no SDAT) found via `swdl`/`smdl` scan, same pipeline.

## Commands

Nightly Rust (`rust-toolchain.toml`). Default `simd` cargo feature = `std::simd` in sinc resampler; `--no-default-features` on `optime-core` = scalar gather (stable). `.cargo/config.toml`: `-C target-cpu=native` (hot path needs AVX2/FMA) + `--cfg=web_sys_unstable_apis` for wasm (Media Session API).

- `cargo run -p optime-app` — native app (loads a `demos/` demo on start)
- `cargo test --workspace` — all tests (unit + integration + pret transcription)
- `cargo clippy --workspace --all-targets` — lint (kept clean)
- `cargo fmt --all` — format
- `cargo clippy -p optime-app --target wasm32-unknown-unknown` — type-check web build
- `cd crates/optime-app && trunk serve` — web dev (needs `rustup target add wasm32-unknown-unknown` + `cargo install trunk`)
- `cd crates/optime-app && trunk build --release` — web release

Deploy: GitHub Pages via `.github/workflows/deploy.yml` (Trunk `--public-url "/<repo>/"` → Pages artifact), on push to `rust-rewrite`/`master`. Also stages legacy app into `dist/legacy-js/` served at `/<repo>/legacy-js/`: copies `legacy-js/` verbatim (now contains its own `fonts/`+`assets/`) plus the six `demos/*.sdat` its buttons reference (not the whole `demos/` — has large `.zip`/`.gz`/`.gbaaudio` the legacy app can't use).

### Local-only extras

`crates/optime-app/build.rs` generates local-only extras from gitignored `local_extras.txt`: song-name tables by filename (`JSONS_BY_GAME_FILENAME`) + demo entries (`LOCAL_DEMOS`) → `OUT_DIR`, `include!`d by `song_names/mod.rs` + `app.rs`. Lets a personal checkout carry ROM-hack metadata (gitignored `src/song_names/local/`, `demos/*.gbaaudio.gz`), never committed. No manifest = empty tables = byte-identical to clean clone. Manifest lines (`#`/blank ignored), `|`-separated: `table|<filename-key>|<json-under-src/song_names>`, `demo|<label>|<demo-stem>`.

Offline render examples (both take a decompressed archive — gunzip `*.gbaaudio.gz` first):
- `render_songs -- <archive> <out_dir> [seconds]` — every playable song → mono WAV + `manifest.json`.
- `export_album -- <archive> <names.json> <out.flac> [--max-silence S] [--limit N]` — songs named by curated JSON, album order, each console's HQ preset (`LoopAndTransitionOptions::export`: fade after one loop, 90s after-loop cap for long loops, then 3s fade) → one stereo FLAC, album-wide −16 LUFS (EBU R128). Leading/trailing near-silence trimmed, `--max-silence` gap (default 0.8s) between songs. 3 parallel phases: render fans across CPUs → per-track temp PCM (progress bar/worker); analyze per-track `EbuR128` via rayon combined by `loudness_global_multiple` (integrated loudness gates silence, per-track == whole-album); encode serial-in, channels parallel via flac-codec `rayon`. Bounded memory. Dev-deps only: `flac-codec`+`ebur128`+`clap`+`indicatif`+`rayon` (lib stays lean). `--benchmark [PCT]`: renders deterministic evenly-spread PCT (no FLAC/temp), median of 3 passes; shares `parallel_render` with FLAC path. **THE perf benchmark**: `export_album --benchmark 3%` on Mother 3 (`demos/mother-3.gbaaudio` + `mother_3.json`), 4 threads — `docs/sample-width-benchmark.md` (found `Sample` f64→f32 = no speedup).

## Architecture

Two crates. Core data flow = strict pipeline; synth layer knows no console:

```text
ROM/archive bytes ─► devices::SoundData         (per-console parse: songs, instruments)
SoundData + id    ─► devices::DevicePlayer      (per-console sequencer + envelope)
DevicePlayer::tick─► devices::SynthEvent stream (standardized message set)
SynthEvent stream ─► SynthController            (voice pools, master clock, mixing)
```

### `crates/optime-core` — platform-independent engine (no browser/UI/audio deps)

Pure-`std`. Top-level modules = console-agnostic synth layer; console-specific → `devices/`.

**`devices/`** — one folder per console + shared interface:

- `mod.rs` — interface = two object-safe traits (dyn dispatch, no console methods leak). `SoundData`: `song_ids` (only playable), `make_player`→`Option<Box<dyn DevicePlayer>>`, provided `song_name`→`None`, `waveform_dc_stats`→empty, `as_any`, provided `song_length_seconds` (runs player headless). `load_all(bytes)->Vec<Box<dyn SoundData>>` free fn (probes NDS→DSE→GBA). `WaveformDcStat` (`label`/`dc_shift`/`length`/`sample_rate`). `DevicePlayer`: `clock_rate`, `cycles_per_tick`, `steps_elapsed`, `step_rate`, `steps_per_beat`, `tick`, provided `tick_rate`. Console-specific (GBA `game_code()`/`extract_audio()`) via `as_any().downcast_ref::<GbaRom>()`, never a trait method.
- `messages.rs` — the stream. `SynthEvent`: `NoteStarted`, `VoiceVolume`, `VoicePitch`, `VoiceDetune`, `VoiceStopped`, `NoteReleased`, `TrackPan`, `TrackDetune`, `Looped`, `Ended`. `VoicePitch`: `Midi{note, sample_pitch_hz}` (repitched samples) vs `DataRateHz` (register rates). `VoiceId` (monotonic u64). `TickFeedback` (synth voice endings → device each tick).
- `nintendo_ds/` — `sdat.rs` (`Sdat`: SYMB/INFO/FAT/SBNK), `bank.rs` (`InstrumentBank`, ADSR coeff helpers), `sequence/` (SSEQ interpreter: `track.rs`/`message.rs`/`interpreter.rs`/`mod.rs` — 16 tracks, message queue, `SND_CalcRandom`), `tables.rs` (DS BIOS/pokediamond LUTs, bit-identical), `volume.rs`+`lfo.rs` (dB-domain channel volume + LFO, pokediamond ports), `player.rs` (`NdsPlayer`: decodes SWAR archives up front, runs sequencer behind BPM timer, owns per-note ADSR/LFO, emits `SynthEvent`s).
- `gba/` — `rom.rs` (`GbaRom`: song-table via `m4aSongNumStart` sig + brute-scan fallback, `SongHeader`), `voice.rs` (`ToneData`/voicegroup incl. key-split + rhythm, `WaveData`), `sequencer.rs` (`Mp2kSequencer`: MPlayMain per VBlank, `tempo_c += tempo_i; while >= 150`, emits `Mp2kOp`), `tables.rs` (`gClockTable`/`gScaleTable`/`gFreqTable`/CGB/noise + `MidiKeyToFreq`/`MidiKeyToCgbFreq`, bit-identical), `player.rs` (`GbaPlayer`: DirectSound + 4 CGB alloc/steal, per-frame envelopes incl. pseudo-echo, generated square/wave/noise, emits `SynthEvent`s; pan-law pinned by `pan_law_composition_matches_per_side_envelopes`; `dc_center` always centers PSG, and when per-device `remove_sample_dc_offset` on (default off; cache keyed by flag) also subtracts each DirectSound PCM's DC to match AC-coupled output + kill on/off thump), `extract.rs` (`extract_audio`: audio-only ROM image — static walk of track bytecode/voicegroups/wave data zeroes the unreachable, keeps offsets so playback bit-identical; app "Export GBA audio data" button + `extract_mp2k` example; also `waveform_dc_stats` — walks voicegroup (DirectSound only, sub-groups followed), decodes each, reports DC shift most-shifted first).
- `dse/` — full backend (`DseSoundData`/`DsePlayer`): `swdl.rs` (`Swdl`: 0x50 header + 0x10-aligned `wavi`/`prgi`/`kgrp`/`pcmd`; WAVI→`Waveform` via shared PCM8/PCM16/IMA-ADPCM decoders; main `bgm.swd` holds `pcmd`, per-song `bgm####.swd` hold `prgi` referencing it), `smdl.rs` (`Smdl`: `song` chunk TPQN + 4-byte-aligned `trk ` chunks), `events.rs` (SMDL bytecode: `decode_track` + opcode/operand table from pmd-sky asm — `SMD_EVENTS_FUN_TABLE`, each `DseTrackEvent_*` advance in `asm/main_0206C9BC.s`/`lib/DSE`, `ParseDseEvent` PlayNote bit layout, `_020B0B7C` pause LUT), `sequencer.rs` (`DseSequencer`: multi-track — pauses, sub/main loops, tempo → flat `SeqOp`), `envelope.rs` (`SoundEnvelope`: `dc_envelope.c` slide), `pitch.rs` (`note_key_to_hz`: ROM-exact — split `key_base`/`note_delta`/key → 8.8 `note_key` → bit-identical `_020B1310`/`_020B1394` tables → absolute PCM rate Hz, WAVI `sample_rate` unused at runtime; `0x00FFB0FF` = NDS sound clock), `volume.rs` (`DseVoice_*` integer chain: `velocity*program*split/127²` note vol, `track*expr/127` channel vol, `(env*vol*note/8032)²>>9` square law, magic-divs bit-for-bit), `lfo.rs` (`Lfo`: vibrato/tremolo/auto-pan from `dc_lfo_1.s`/`dc_lfo_2.c` — 8 `SoundLfoWave_*`, `SoundLfoBank_Set` (depth/period/fade-in), `SoundLfoBank_Tick`, routed as `DseVoice_UpdateParameters`), `player.rs` (`DsePlayer`: decode on demand, sequencer behind ~100 Hz driver tick, one `SoundEnvelope`/voice, pitch-wheel key bend (`SetKeyBend`/`SetKeyBendRange`→`TrackDetune` `range*value/8192` semitones, dominant Explorers effect), per-note pitch/vol LFOs + track auto-pan LFO, emits via `VoicePitch::DataRateHz`). Proven by `dump_dse` example vs `D:\Git\pmd-sky\files\SOUND\BGM`. Not modelled: `SongVolumeFade`, per-note pan, tuning jitter.

**Synth layer** (console-agnostic):

- `synth_controller/mod.rs` — `SynthController`: owns two per-track synth sets (output-rate + mixer-rate) + `slot_owner` bookkeeping (set-agnostic free fns `render_set`/`find_slot`/`cut_finished`), master clock, `next_sample`/`fill`/`tick`, applies `SynthEvent`s, feeds `TickFeedback` (voice steals + one-shot endings) back to device. **Sole home of end-of-song fade/loop policy**: `set_loop_and_transition(LoopAndTransitionOptions)` (loops-before-fade / too-long-after-loop-threshold / fade-on-end / grace / fade seconds; presets `none`/`export`/`live`) or immediate `request_transition(fade_seconds)`; counts loops, applies one linear fade gain to `next_sample`/`fill`, pumps `PlaybackEvent`s (`Looped`/`TransitionStarted`/`Finished`) drained via `take_messages`. Shared by live (`audio.rs`) + both exporters; default `none` never fades (plain renders/tests unaffected). `use_mixer` set: sampled (non-PSG) voices on mixer set @ `mixer_sample_rate`, PSG on output set; routes mixer stereo bus into `Bank` (owns only a `StreamResampler`) to upsample, output set sums straight; off = everything on output set, bit-identical to single-set.
- `synth_controller/config.rs` — option types: `PopSmoothing{psg, sample, slew_seconds}` (per-kind de-click gain slew, `DEFAULT_POP_SLEW_SECONDS`=2ms, all resample modes), `DelaySmoothing` (stereo-expander delay-length changes), `HighShelf`. Config struct itself = `PerDeviceSettings` (`device_settings.rs`), consumed by synth via resolvers: `resample()`/`mixer_resample_mode()`→`InstrumentResampleMode`, `tuning()`→`TuningSystem`, `pop_smoothing()`, `delay_smoothing()`; plus fields `smooth_pan` (slew L/R pan split, orthogonal to `DelaySmoothing`), `use_mixer`/`mixer_sample_rate`/`track_enables`, `psg_crunch_compensation` (per-rate RBJ low-pass cascade in `mod.rs`: order/cutoff/Q fit by `scripts/fit_compensation.m` to measured nearest→crunch HF loss on real DirectSound, rebuilt via `BiquadFilter::low_pass` on output-rate change so knee stays fixed; colors PSG/output bus with same HF rolloff crunch gives DirectSound; only when `use_mixer` + `mixer_resample`=output-Nyquist crunch).
- `synth_controller/vis.rs` — `FsVisController`: parallel look-ahead for visualizers, no audio. Runs a second headless `Box<dyn DevicePlayer>`, extracts `VisNote`s from standard `SynthEvent` stream (`NoteStarted`/`NoteReleased`/`Looped`/`Ended`) — device-agnostic, never touches a sequencer. Live `tick` + whole-song `overview` share note/duration bookkeeping.
- `synth/` — `instrument.rs` (`WaveformInstrument`, one voice: `VoicePitch` frequency, resampling, pop-smoothing slew via `Slewer`, block render; `advance`/`advance_block` share `fold_pos`), `synthesizer.rs` (`WaveformSynthesizer`: round-robin pool, pan (L/R split optionally `Slewer`-slewed under `smooth_pan`), Haas widening w/ optional held delay changes, bass-mono crossover), `delay_line.rs` (`DelayLine`). Resampling via `crate::dsp::resample::{gather_sinc, …}`.
- `resample/` — windowed-sinc: `kernels.rs` (FIR kernel/response, `ResampleTables`), `gather.rs` (SIMD/scalar gather), `source.rs` (loop-aware `gather_sinc`/`GatherSource` feeding a voice), `stream.rs` (`StreamResampler`: continuous fixed-ratio stereo for mixer-bus→output, applies `InstrumentResampleMode` reusing `resample_sinc`), `plan.rs` (`effective_gather`/`sinc_fc`/`mode_half_taps`: resolve `InstrumentResampleMode`→gather+cutoff, shared by voice gather + stream), `mod.rs` (public API).
- `waveform.rs` — `Waveform` (decoded + playback metadata), aliases `Sample`(`= f64`, one amplitude) + `Frame`(`= (Sample, Sample)`), `InstrumentResampleMode` (`NearestNeighbor`/`Linear`/clean `SincSampleNyquist`/crunchy `SincOutputNyquist` w/ per-kind PSG/sampler cutoff sliders), `decode_pcm8`/`decode_pcm16`/`decode_adpcm`/`decode_wav`.
- `dsp/` — `biquad_filter.rs` (`BiquadFilter`, cascaded), `slewer.rs` (`Slewer`: linear slew toward target by bounded per-sample step, lands exact; behind pop-smoothing + pan slew), `resample/` (above). `tuning.rs` (`midi_note_to_hz`, equal-temp / Pythagorean `TuningSystem`). `util.rs` (`read_u8/u16/u32` 0-on-OOB, `search_for_sequence`, `CircularBuffer<T>`).

Runtime options = `PerDeviceSettings` (`device_settings.rs`) threaded into synth calls — no global state. Both app's serializable per-console settings AND engine runtime config: settings choices (enums, indices, sliders) → concrete engine values via resolver methods; no separate "resolved config" type.

**Master clock** in one place: `SynthController::next_sample` accumulates `clock_rate()` per output sample, ticks player every `cycles_per_tick() * sample_rate` cycles. DS: `DS_CLOCK_RATE`(33,513,982)/`CYCLES_PER_TICK`(64×2728)≈192 Hz, BPM timer `+= bpm`/`while >= 240` in player. GBA: `GBA_CLOCK_RATE`(16,777,216)/`CYCLES_PER_FRAME`(280,896)≈59.73 Hz VBlanks, `tempo_c` in sequencer. `fill` renders tick-aligned blocks, bit-identical to per-sample (`tests/block_render.rs`). Visualizer timeline = sequencer steps (DS: SSEQ ticks; GBA: tempo steps) → wall time via `step_rate()`.

### `crates/optime-app` — eframe/egui front-end (native + web)

- `app.rs` — `OptimeApp`: song list, transport, settings, WAV export, audio-only GBA ROM export. "Stats for Nerds" button (settings-panel bottom) → floating `stats_ui`: decoded samples + DC shift each got (% full scale), most-shifted first, from `waveform_dc_stats`, cached per `(archive, song_id)` in `stats_cache`. Loads any `SoundData` (`.nds`/`.sdat`/`.gba`); `load_bytes` gunzips gzip-magic first (`flate2`) so `*.sdat.gz`/`*.gbaaudio.gz` load directly. **Playlist owned by audio thread, not UI** (fade+advance survive frozen repaint, e.g. hidden tab): UI builds ordered `PlaylistEntry`s (`Arc<dyn SoundData>` + `TrackRef`), *sends* via `PlaybackCommand` (`SetPlaylist`/`PlayAt`/`Next`/`Prev`/`Reorder`), *reflects* audio thread's authoritative `index` into visuals in `reconcile_playback` on `status_gen` change. Shuffle materialized once at toggle (`materialize_shuffle`→`Reorder` keeping current playing); audio thread walks fixed list. `self.playlist` = natural order to re-materialize from; `current_song` = derived highlight. Settings **per device**: `Persisted` holds one `PerDeviceSettings` each for `nds`/`gba`; `device_settings`/`device_settings_mut` pick current console's (default DS when nothing loaded; only active mode's sliders shown). Demos platform-split: native reads `demos/`, web fetches `.sdat` at runtime (`web::fetch_bytes`). Shareable demo via URL query on web. Narrow (<600px) → Spotify-style mobile (`MobileTab`: Now Playing / Library / Playlists / Settings, floating mini-player, swipe-to-change on Now Playing). Battery: `update` repaints fast only on Now Playing (animated roll); other tabs idle 1 Hz; mini-player EQ bars static.
- `song_names/` — curated titles + OST ordering for ROMs with no song names (app-side cosmetic metadata, not engine). Per-game JSON (`[{"songId","title"}]`) via `include_str!` + `serde_json`; array index = listing order. `mod.rs::lookup(filename, game_code, song_id)`: by source filename first (`JSONS_BY_GAME_FILENAME`, DS/code-less), then GBA game code (`JSONS_BY_GBA_GAME_CODE`) — `pokemon_emerald.json` (`BPEE`, Super Music Collection titles vs pokeemerald `songs.h`, album-first) + `mother_3.json` (`A3UJ`, MOTHER 3 fan-translation Sound Player names from `musicnames.txt`, paired to real MP2K song id via ROM slot→id table; player order ≠ song-id order). Curated title **wins over** ROM name. Drives **Default** sort: curated first in that order, rest native (`Song::ost_order`/`apply_sort`). Native dev-tool "Edit song names" toggle (settings bottom) → inline editor (`song_edit_list_ui`: rename, ▲/▼, drag); "Save to JSON" writes source-tree file (`source_json_dir` = `CARGO_MANIFEST_DIR` + `target_filename`), baked next build. Only curated written (loaded-table + renamed/reordered this session, `Song::curated`); untouched ROM-named stay out. New game writes derived filename that must be hand-wired into `JSONS_BY_*` consts.
- `persisted.rs` — user library (`TrackRef`, `Playlist`, `Library`, `RepeatMode`), per-device `PerDeviceSettings` (re-exported from core `device_settings.rs`), `Persisted` bundle → eframe storage. All out-of-box defaults = engine presets `PerDeviceSettings::high_quality_nintendo_ds()`/`high_quality_gba()` which `Persisted::default` calls (offline tools share exact app defaults); core `PerDeviceSettings::neutral()` = separate every-effect-off baseline for tests/examples. Sub-structs have no `Default` impls. `TrackRef::song_id` = any-device song id; serializes as `sseq_id` via `#[serde(rename)]` for storage/URL compat.
- `audio.rs` — `AudioEngine`: `cpal` f32 output stream (native + WebAudio) from shared state. Callback `step_playback` owns advancement: drains `PlaybackCommand`s, asks controller for quick fade before manual switch (`request_transition`), installs `live()` policy per freshly decoded song, watches controller `Finished` `PlaybackEvent` (loop-count/FINE/requested fade → silence) and once silent **advances index + decodes next right there** (stall inaudible, output already silent). Fade ramp lives in controller; `GainRamp` only does master volume + fade-in + pause ramp.
- `media_controls.rs` — media keys (BT/keyboard/lock-screen — AirPods double-tap, phone lock-screen). Native `souvlaki` (Win SMTC / mac Now Playing / Linux MPRIS), `HWND` from `eframe::Frame` (`raw-window-handle`), `MediaControlEvent`s via channel. Web = Media Session API (`navigator.mediaSession`): per-action JS closures push `MediaAction` into `Rc<RefCell<…>>` + `ctx.request_repaint()`. Both: `MediaControls::new(ctx, frame)`/`poll`/`set_now_playing`; drained each frame in `handle_media_controls`. Media Session gated behind web-sys `web_sys_unstable_apis` cfg.
- `player.rs` — `AudioState` (UI↔callback via `Arc<Mutex<…>>`: pause/fade/volume ramps, DSP-load stats), audio-thread-owned `Playback` (`PlaylistEntry` list + `index` + `repeat` + command queue + `status_gen`, `manual_step`/`auto_advance` unit-tested), `PlaybackCommand` set, `render_to_samples` (offline WAV export: sets controller `export()` policy, stops on `Finished`, applies only 0.5 export headroom — loop/fade is controller's).
- `visualizer.rs`/`piano_roll.rs` — procedural 16-track × 88-key roll via `egui::Painter` (no image assets; PNGs in `legacy-js`). Driven by `VisSnapshot` (`steps` + `step_rate`) + `FsVisController`.
- `theme.rs` — egui style/typography (Inter/SF Pro). `filter_plot.rs` — freq-response plot for resampler/crossover UI. `wav.rs` — 16-bit PCM WAV encoder. `web.rs` (wasm) — browser download + demo fetch. `main.rs`/`lib.rs` — native window; wasm mounts `#the_canvas_id` (`index.html`, Trunk).

Config (`PerDeviceSettings` for current console + live track mutes as `track_enables`) cloned each frame, pushed to audio thread under lock; same lock snapshots `notes_on` for visualizer.

## Testing

- `tests/sdat.rs` — parses real `demos/*.sdat`.
- `tests/gba.rs` — end-to-end MP2K vs synthetic in-memory ROM (scan, sequencer, player, audio, look-ahead).
- `tests/block_render.rs` — pins `fill` bit-identical to per-sample.
- **pret-reference unit tests = accuracy anchor.** Transcribe the C fn from pokediamond (DS)/pokeemerald (GBA) into a Rust oracle, assert engine matches across a param grid (`calc_channel_volume_matches_pokediamond` in `nintendo_ds/volume.rs`, `midi_key_to_freq_matches_pokeemerald` in `gba/tables.rs`). New accuracy fix → add a transcription test, not a golden fixture.
- Each core module has inline `#[cfg(test)]` tests (decoders, ADSR, tuning, biquad, circular buffer, interpreter).

## Conventions

- Idiomatic zero-extra-dep `std` in `optime-core`; no UI/audio/browser deps (app + headless tests reuse it).
- Console code → `devices/<console>/`; talks to synth only via `SynthEvent`/`TickFeedback`. Synth never reaches into a device.
- Make invalid states unrepresentable: enums over flag/int combos (`SoundData`, `DevicePlayer`, `VoicePitch`, `ToneKind`, `Mp2kOp`, ADSR phases), `Option` over sentinels.
- Binary parse = `from_le_bytes` via `util::read_*` (0 on OOB, no panic).
- **Naming = signal theory first.** "sample" = one amplitude value at an instant (sampling-theorem sense) = `Sample`(`= f64`); `Frame` = stereo `(Sample, Sample)`; whole decoded clip = `Waveform` (never "sample"). "sample" derivatives only for genuine sampling (`sample_rate`, per-sample loop, PCM sample data, "sampler"/"sampled" voice vs PSG). Renaming on these grounds → update surrounding comments same pass.
- Keep `cargo clippy --workspace --all-targets` + `cargo fmt --all --check` clean.
- Touching synth math → check vs matching pret decomp, add/adjust a transcription test pinning to the C.
- **Commit messages: never mention Claude/AI.** No `Co-Authored-By: Claude`, no "Generated with Claude", no attribution of any kind. Applies to commit messages and PR bodies.

## Legacy app

`legacy-js/` = original browser app verbatim: `OptimePlayer/OptimePlayer.js` (engine+parser), `OptimePlayer/dsp.js`, `index.js` (DOM glue), `index.html`, ffmpeg/canvas exporters `video-exporter.js`/`playlist-exporter.js` (not ported — Rust app does in-app WAV export). Historical reference only (no longer accuracy oracle). Its `assets/` + `fonts/` now live inside `legacy-js/`; only `demos/` stays a shared repo-root dir.

## Refactor / Rearchitecting

In progress: replace core's frontends with algorithmically translated / statically recompiled reverse-engineered sound engine code. Nintendo DS first.

# Block-chain refactor — cleanup options

Follow-up to `d5b241a` ("synth: render the whole signal chain in blocks"). That commit is correct,
verified (`optime-cli golden` unchanged, 1.64× faster, chunk-invariance tested down to one sample),
and shipped. This file catalogues what it left behind that is worth tidying, and the ways each
could go.

**How to use this:** each item is independent. Read the summary table, pick the items worth doing,
then read that item's options. Nothing here is a bug that affects released audio output unless
marked otherwise.

## Summary

| # | Issue | Severity | Recommended |
|---|---|---|---|
| 1 | Per-sample entry points ~5–11× slower (measured) | **High** — live annotation path | Opt 1c + 1d |
| 2 | `next_sample` / `next_frame` API shape encourages the slow path | High | Opt 2b |
| 3 | Four copies of the "was_active edge → reset_state" pattern | Medium | Opt 3a |
| 4 | The two high-band-compressor stage wrappers are near-identical | Medium | Opt 4a |
| 5 | `advance_block` sets `stopped_at` in three places | Medium — correctness risk | Opt 5a |
| 6 | `last_input_needed` duplicates `advance`'s float recurrence | Medium — correctness risk | Opt 6a |
| 7 | `StreamResampler::set` silently resets position when the ring grows | Medium | Opt 7b |
| 8 | `render_block` takes both `n` and slices that must be ≥ `n` | Medium | Opt 8a |
| 9 | `ChainScratch` allocates 7 buffers, several usually unused | Low | Opt 9b |
| 10 | Five hand-rolled planar→interleaved loops | Low | Opt 10a |
| 11 | `fill`'s odd-trailing-`f32` handling is inside the chunk loop | Low | Opt 11b |
| 12 | `quiet_at`'s `try_fold` is opaque | Low | Opt 12a |
| 13 | `Transition::advance` / `Reverb::process` exist only for tests | Low | Opt 13a |
| 14 | `dsp::block` exports test-only items | Low | Opt 14a |
| 15 | `MAX_BLOCK` reachable by two paths | Low | Opt 15a |
| 16 | `HighBandCompressorStage::process_block` takes caller scratch | Low | Opt 16a |
| 17 | `HighBandCompressor::params()` puts a DSP type in a settings module | Low | Opt 17c |
| 18 | `apply_stereo_block` fills pan-gain arrays even when pan is constant | Low | Opt 18c |
| 19 | `tests/block_render.rs` names describe a structure that no longer exists | Low | Opt 19a |
| 20 | `Bank::route_block` is a one-line passthrough | Low | Opt 20b |
| 21 | Annotation mixer still runs per-frame | Low | Opt 21b |
| 22 | `render` relies on zip-shortest to bound its writes | Low | Opt 22a |
| 23 | Stereo-length invariant asserted 3 ways; `Reverb::process_block` not at all | Low–Medium | Opt 23a |
| 24 | `CLAUDE.md` documents `resample/` as a folder; it is one file | Low — pre-existing | Opt 24a |

---

## 1. Per-sample entry points are now much slower (measured)

The single-sample wrappers build the *whole* block's scratch and then use one element of it.
Measured on this machine, release, Mother 3 song 5, Enhanced GBA preset, 200k samples:

```
controller: render(200000) = 179 ms   next_sample ×200000 = 877 ms   ratio =  4.90×
synth:      render_block   =  2.3 ms   next_sample ×200000 =  26 ms   ratio = 10.92×
```

Per call, `SynthController::next_sample` zeroes a 7 KB `ChainScratch`; `WaveformSynthesizer::
next_sample` zeroes `mono` + `out_l` + `out_r` (3 KB) plus `gl` + `gr` in `apply_stereo_block`
(2 KB, and `high_l` too under `bass_mono`). All of it to produce one sample.

This is not hypothetical: **`ChordVoicer::next_frame` (`optime-app/src/annotation/chord_voice.rs`)
calls `WaveformSynthesizer::next_sample` once per output frame** inside the audio callback, via
`mix_annotation`. Annotation mode is a maintainer tool, so this is not shipping-critical, but it is
a 10× regression on a real path.

**Options**

- **1a. Leave it.** The hot paths are all blocked; only the annotation tool pays. Cheapest, and the
  measured absolute cost (26 ms per 200k frames ≈ 6 s of audio) is still far under real time.
- **1b. Size the scratch to `n`.** Replace `[Sample; MAX_BLOCK]` locals with `MaybeUninit` +
  initialise only `[..n]`, or keep the array but track that only `[..n]` is ever read. Removes the
  zeroing without changing any API. `unsafe` unless done with a `Vec`/`ArrayVec` crate, which
  `optime-core` will not take (zero-dep policy).
- **1c. Move the scratch into the structs.** `ChainScratch` becomes a `SynthController` field and
  `mono`/`out_l`/`out_r`/`gl`/`gr` become `WaveformSynthesizer` fields, each `fill(0.0)`-ed over
  `[..n]` only. No `unsafe`, no API change, and it removes the per-call zeroing entirely. Costs
  ~7 KB in `SynthController` and ~5 KB × 32 synths (2 sets × 16 tracks) ≈ 160 KB, and reintroduces
  the borrow-checker friction that pushed the scratch onto the stack in the first place (solvable
  by destructuring `self`, which `synth_controller/mod.rs` already does elsewhere).
- **1d. Give `ChordVoicer` a block API.** `next_block(&mut [f32], &mut [f32])` over
  `WaveformSynthesizer::render_block`, called from `mix_annotation`. Fixes the only live caller
  directly and is a small, local change. Pairs well with 1c or on its own.
- **1e. Delete the per-sample entry points entirely.** Forces every caller to block. Largest churn;
  breaks the "a caller that needs one sample can have one" property the refactor advertises.

**Recommended: 1c + 1d.** 1c removes the cost at the root without `unsafe`; 1d fixes the one caller
that actually runs per frame. Measure with the harness sketched above before and after.

---

## 2. The one-sample API shape invites the slow path

`WaveformSynthesizer::next_sample` returns nothing — it renders into throwaway one-length buffers
with `mix: false` and the caller reads the public `val_l`/`val_r` fields. Two exits from one call,
and `val_l`/`val_r` now mean "the last sample of the block", which is a per-sample-world concept
that survived the refactor. Only `chord_voice.rs` reads them.

**Options**

- **2a. Keep as is.** Zero churn.
- **2b. Return the frame.** `pub fn next_sample(&mut self, config) -> (Sample, Sample)` and drop the
  public `val_l`/`val_r` fields (or make them private). One exit, and the "last sample of a block"
  oddity disappears. Touches `chord_voice.rs` only.
- **2c. Delete `next_sample`, expose only `render_block`.** Cleanest conceptually; makes item 1d
  mandatory rather than optional.

**Recommended: 2b**, and 2c if 1d is done anyway.

---

## 3. Four copies of the "was_active edge → reset_state" pattern

`psg_comp_was_active`, `high_comp_psg_was_active`, `high_comp_sampled_was_active` (and
`Bank::was_active`) are each a `bool` field plus the same six-line if/else in the stage method:
bypass sets the flag false and returns; the inactive→active edge clears the stage's state.

**Options**

- **3a. A tiny `EdgeGate` helper** in `dsp/block.rs` or `dsp/mod.rs`: `fn arm(&mut self, active:
  bool) -> Gate` returning `Gate::{Bypass, FirstActive, Active}`, so each stage becomes a `match`.
  Removes three copies of the logic and makes the "clear on fresh enable" rule findable in one
  place.
- **3b. A macro.** Terser but harder to read; the codebase does not use macros for this kind of
  thing.
- **3c. Leave it.** The pattern is short and each copy is next to the state it guards.

**Recommended: 3a.**

---

## 4. The two high-band-compressor wrappers are near-identical

`compress_psg_high_band_block` and `compress_sampled_high_band_block` differ only in which stage,
which `was_active` flag, and `is_active_psg()` vs `is_active_sampled()`. ~20 duplicated lines.

**Options**

- **4a. One free function** taking `(&mut HighBandCompressorStage, &mut bool, active: bool, l, r,
  params, high_l, high_r)`, with the two methods as two-line callers (keeping the distinct doc
  comments, which carry real information about *why* each bus is compressed where it is).
- **4b. Fold both into `render`** at the call sites.
- **4c. Leave it.** The duplication is small and the two comments differ meaningfully.

**Recommended: 4a** — combines naturally with 3a, since the shared function is exactly where the
edge gate lives.

---

## 5. `advance_block` sets `stopped_at` in three places

`WaveformInstrument::advance_block` records the stop index in the non-sinc fallback loop, the
fully-attenuated fast path, and the main gather loop. The `HoldDuringNotes` correctness fix (and
therefore chunk-invariance) depends on all three staying in sync. A fourth path added later would
break it silently — `tests/block_render.rs` would catch it, but only for the configs it covers.

**Options**

- **5a. Funnel every stop through one place.** A private `fn stop(&mut self, at: usize)` that sets
  `playing = false` and records the index, so `playing` can only be cleared one way inside
  `advance_block`. Small change, removes the class of bug.
- **5b. Debug-assert the invariant.** `debug_assert!(self.playing || self.stopped_at.is_some())` at
  the end of `advance_block`. Catches it in tests without restructuring.
- **5c. Have `advance_block` return `Option<usize>`** instead of stashing it on the instrument,
  so the value cannot go stale between calls.

**Recommended: 5a**, optionally with 5b.

---

## 6. `last_input_needed` duplicates `advance`'s float recurrence

`StreamResampler::last_input_needed` walks a copy of `(pos_int, pos_frac)` with the same
`pos_frac += step; floor; carry` arithmetic that `advance` uses, because the pull must land on
exactly the positions the gather loop will visit. Two copies of a float recurrence that must agree
bit-for-bit, in the same file, with nothing enforcing it.

**Options**

- **6a. Extract one `#[inline] fn step_pos(pos_int: &mut i64, pos_frac: &mut f32, step: f32)`** used
  by both. Mechanical, removes the risk entirely.
- **6b. Closed form + a test.** `pos_int + floor(pos_frac + k·step)` is not bit-identical to the
  incremental walk in general, so this needs a property test over ratios and would still be a second
  implementation. Not recommended.
- **6c. Leave it, add a test** that runs both and asserts the same final position over a grid of
  ratios and block lengths.

**Recommended: 6a.**

---

## 7. `StreamResampler::set` silently resets the stream when the ring grows

Growing the ring changes the `index & (len - 1)` mapping, so every stored sample moves; the code
therefore resets `pos_int`/`pos_frac`/`loaded` to start clean. That is a mid-stream discontinuity
(a click) with no signal to the caller. In practice it cannot fire for any shipping configuration —
`new()` sizes for step 1.0 (512 slots) and both real ratios need fewer — but a live
`mixer_sample_rate` change to a much larger step would trip it.

**Options**

- **7a. Leave it,** documented as-is.
- **7b. Preserve the contents.** Copy the live window into the new ring at its new offsets before
  swapping. ~10 lines, removes the discontinuity.
- **7c. Size once for the worst case at construction.** Pick the largest step the UI can express
  and never resize. Wastes memory at typical ratios but makes the whole question disappear.
- **7d. Return a `bool`/log when it happens** so a caller can crossfade.

**Recommended: 7b**, or 7c if you would rather delete the code path than fix it.

---

## 8. `render_block` takes both `n` and slices that must be at least `n`

`WaveformSynthesizer::render_block(config, n, acc_l, acc_r, mix)` asserts `acc_l.len() >= n`. Two
sources of truth for one length; a caller can pass `n = 4` with 256-long slices and only the first
4 are touched. The rest of the refactor derives the length from the slice.

**Options**

- **8a. Drop `n`**, use `acc_l.len()`, `debug_assert_eq!(acc_l.len(), acc_r.len())`. Matches every
  other block API in the crate. Touches `render_set_block` and the synthesizer's tests.
- **8b. Keep `n`, assert exact equality** rather than `>=`.
- **8c. Leave it.**

**Recommended: 8a.**

---

## 9. `ChainScratch` allocates seven buffers, several usually unused

`mix_l`/`mix_r` are only touched when `use_mixer`; `high_l`/`high_r` only when a compressor is
active; `gain` is filled with 1.0 and multiplied through on every sample even though no fade is
running for the overwhelming majority of playback.

**Options**

- **9a. Leave it.** 7 KB of stack, once per `render` call.
- **9b. Skip the gain multiply when idle.** Add `Transition::is_idle()`; when true, copy
  `acc_*` straight to `out_*` and skip both the `gain` fill and the per-sample multiply. Removes one
  buffer's traffic from the common path.
- **9c. Split the struct** so the mixer and compressor buffers only exist on the paths that use
  them. More types, marginal gain.

**Recommended: 9b** (and it composes with 1c, which makes the buffers persistent anyway).

---

## 10. Five hand-rolled planar→interleaved loops

`SynthController::fill`, `fill_mixer_bus`, `audio.rs::write_audio` (stereo and general paths), and
`mix_annotation` each write "planar pair → interleaved output, optionally with a per-frame gain".

**Options**

- **10a. One helper in `dsp::block`**: `interleave(l: &[Sample], r: &[Sample], out: &mut [f32])`,
  with the gain-applying variants kept at their call sites (they differ: master volume, pause ramp,
  channel-count fan-out).
- **10b. A helper in the app only**, since the two core cases are trivial.
- **10c. Leave it.** The loops are three lines each and the surrounding logic differs.

**Recommended: 10a** for the two core cases; the app's channel-count mapping is genuinely different
and should stay.

---

## 11. `fill`'s odd-trailing-`f32` handling sits inside the chunk loop

`if chunk.len() % 2 == 1` is evaluated for every chunk, but only the final chunk can ever be odd
(the chunk size `2 * MAX_BLOCK` is even). It also calls `next_sample`, the expensive path from item 1.

**Options**

- **11a. Hoist it after the loop**, restoring the original structure.
- **11b. Assert even lengths.** Every real caller passes interleaved stereo, so `debug_assert!(out
  .len() % 2 == 0)` and deleting the branch is defensible — but it is a behaviour change for any
  caller relying on it. Check `optime-cli`, `player.rs`, `bounce.rs` first.
- **11c. Leave it, add a comment** explaining that only the last chunk can be odd.

**Recommended: 11b** if the audit comes back clean, else **11a**.

---

## 12. `quiet_at`'s `try_fold` is opaque

```rust
.map(|&i| if instr.playing { None } else { instr.stopped_at })
.try_fold(0, |latest, stop| stop.map(|s| latest.max(s)))
```

"`None` if any voice is still playing, else the latest stop index" is correct but takes a minute to
read, and this is the function the `HoldDuringNotes` fix hinges on.

**Options**

- **12a. An explicit loop** with an early `return None`. Longer, immediately obvious.
- **12b. Split in two:** `all_stopped()` and `latest_stop()`. Two trivial functions.
- **12c. Leave it** and lean on the doc comment.

**Recommended: 12a.**

---

## 13. `Transition::advance` and `Reverb::process` exist only for tests

Both are `#[cfg(test)]` one-sample wrappers, kept because their tests step and inspect per sample.
Production code shaped by test convenience.

**Options**

- **13a. Keep them.** `#[cfg(test)]` is honest about what they are, and the tests read better for it.
- **13b. Rewrite the tests in block form** and delete the wrappers. The reverb impulse-response
  tests get noticeably worse to read.
- **13c. Move them into the test modules** as free helpers taking `&mut Reverb`.

**Recommended: 13a** — flagged only so the next reader knows it was deliberate.

---

## 14. `dsp::block` exports test-only items

`TEST_BLOCK_LENGTHS` and `test_signal` are `#[cfg(test)] pub(crate)` in a production module so the
per-stage equivalence tests can share them.

**Options**

- **14a. Move to `#[cfg(test)] pub(crate) mod testutil`** inside `dsp/`, keeping `block.rs` purely
  about the runtime contract.
- **14b. Leave it** — they are `#[cfg(test)]`, so nothing ships.
- **14c. Duplicate the generator** into each test module. Worse.

**Recommended: 14a.**

---

## 15. `MAX_BLOCK` is reachable by two paths

Defined in `dsp::block`, re-exported by `synth` (`pub use crate::dsp::block::MAX_BLOCK;`) so existing
imports kept working. Two import paths for one constant.

**Options**

- **15a. Drop the re-export**, update the handful of `use super::MAX_BLOCK` sites in `synth/`.
- **15b. Keep it** as the ergonomic path for `synth` code.

**Recommended: 15a.**

---

## 16. `HighBandCompressorStage::process_block` takes caller-supplied scratch

Five parameters, and the caller must size `high_l`/`high_r` correctly. The one-sample `process`
allocates its own `[0.0; 1]`, so the stage already knows how to own a buffer.

**Options**

- **16a. Give the stage its own `[Sample; MAX_BLOCK]` fields.** Two extra buffers per stage × two
  stages = 4 KB total, and the signature drops to `(l, r, params)`. Simplest to call.
- **16b. Keep caller scratch** — lets `ChainScratch` own every buffer in one place, which is a real
  virtue if item 1c lands.
- **16c. Pass a `&mut ChainScratch`-shaped struct** so the parameter count stays at three.

**Recommended: 16a** unless 1c lands, in which case **16b** is already the right shape.

---

## 17. `HighBandCompressor::params()` puts a DSP type in a settings module

`synth_controller/config.rs` now imports `dsp::high_band_compressor::HighBandCompressorParams` so
the two controller wrappers stop repeating a six-field literal. Settings types referencing DSP
types is a mild layering smudge, though `PerDeviceSettings` is already explicitly both a settings
struct and the engine's runtime config.

**Options**

- **17a. Leave it.** It removed a genuine duplication and `config.rs` already resolves settings into
  engine values.
- **17b. Move the conversion to the controller** as a private free function.
- **17c. `impl From<&HighBandCompressor> for HighBandCompressorParams`** in the DSP module, so the
  dependency points the other way and settings stays ignorant of DSP.

**Recommended: 17c.**

---

## 18. `apply_stereo_block` fills pan-gain arrays even for constant pan

With `smooth_pan` off, `gl`/`gr` are filled with one repeated value and then read per sample —
2 KB of stores and 2 KB of loads to express two scalars.

**Options**

- **18a. Leave it.** Uniform code, and the fill is cheap next to the gathers.
- **18b. Branch on constant vs slewed** inside each of the three stereo branches. Fastest, triples
  the branch count.
- **18c. A small `PanGains` enum** (`Const(Sample, Sample)` / `Ramped(&[Sample], &[Sample])`) with an
  `at(i)` accessor. One branch, no duplication, and it documents the two cases.

**Recommended: 18c**, or 18a if the profile says it does not matter (it probably does not).

---

## 19. `tests/block_render.rs` names describe a structure that no longer exists

`assert_fill_matches_next_sample` and the seven `fill_matches_next_sample_*` tests were named when
`fill` and `next_sample` were two independent implementations. Both are now thin wrappers over
`render`, so the names describe a comparison that no longer exists as such — although the tests are
still valuable (they pin the wrappers *and* the chunking).

**Options**

- **19a. Rename** to `render_is_chunk_invariant_*` / `assert_chunking_does_not_change_output`, and
  fold the newer `render_is_chunk_invariant_down_to_one_sample` into the same naming family.
- **19b. Leave the names** for `git blame` continuity.

**Recommended: 19a.**

---

## 20. `Bank::route_block` is a one-line passthrough

After the refactor it is `self.resampler.process(out_l, out_r, render)`. `Bank` still earns its
keep for rate bookkeeping and `prepare`/`disable`, but this method no longer does anything.

**Options**

- **20a. Inline it** at the one call site in `route_mixer_block`.
- **20b. Keep it** — it preserves the "the controller talks to the bank, not to the bank's
  resampler" boundary the module doc describes.

**Recommended: 20b**, listed so nobody deletes it thinking it is dead weight.

---

## 21. The annotation mixer still runs per-frame

`mix_annotation`'s pull callback loops over the block calling `bounce.next_frame()` and
`ChordVoicer::next_frame()`. Deliberate and documented — but it means annotation is the one path the
refactor did not actually block, and it is where item 1's 10× cost lands.

**Options**

- **21a. Leave it.** Documented as a maintainer tool.
- **21b. `Bounce::fill_block`** (a slice copy plus loop wrapping) **and `ChordVoicer::next_block`.**
  The bounce side is nearly free to do; the chord side needs the envelope loop to accept a slice.
- **21c. Only do the chord side** (item 1d), which is where the cost actually is.

**Recommended: 21b**, or **21c** as the cheap version.

---

## 22. `render` relies on zip-shortest to bound its writes

```rust
for ((o, &v), &g) in out_l[frame..].iter_mut().zip(acc_l.iter()).zip(gain.iter())
```

Correct — `acc_l` and `gain` are both length `n` — but a reader cannot see that exactly `n` samples
are written without checking two other bindings.

**Options**

- **22a. Slice explicitly:** `out_l[frame..frame + n]`. One character of intent per line.
- **22b. Leave it.**

**Recommended: 22a.**

---

## 23. The stereo-length invariant is asserted three different ways, and once not at all

`dsp::block::stereo_len` exists to centralise "both channels same length, ≤ `MAX_BLOCK`". Actual
usage:

| Site | How |
|---|---|
| `render_set_block`, `apply_stereo_block`, `HighBandCompressorStage::process_block` | `block::stereo_len` |
| `SimpleCompressor::process_block`, `StreamResampler::process`, `SynthController::render` | inline `debug_assert_eq!` |
| **`Reverb::process_block`** | **nothing** — it zips `l` and `r`, so a length mismatch silently processes the shorter one and leaves the rest of the longer channel undelayed |

The last row is the only one with teeth. It cannot happen today (the one caller passes two slices of
the same `ChainScratch`), but it is the one stage where getting it wrong would produce plausible-
sounding wrong output rather than a panic.

`StreamResampler::process` and `SynthController::render` legitimately cannot use `stereo_len` as it
stands, because both accept blocks longer than `MAX_BLOCK` and chunk internally.

**Options**

- **23a. Add the assert to `Reverb::process_block`** and use `stereo_len` wherever the `MAX_BLOCK`
  cap genuinely applies; add a `stereo_len_unbounded` (or a `max: Option<usize>` argument) for the
  two chunking entry points.
- **23b. Delete the helper**, inline `debug_assert_eq!` everywhere including the reverb. Fewer
  indirections; loses the single place the invariant is written down.
- **23c. Just fix the reverb** and leave the inconsistency.

**Recommended: 23a**, or **23c** if you want the one-line version.

---

## 24. `CLAUDE.md` describes `resample/` as a folder

Pre-existing drift, not from this refactor, but adjacent to it and noticed while updating the file.
`CLAUDE.md` documents `resample/` with `kernels.rs`, `gather.rs`, `source.rs`, `stream.rs`,
`plan.rs`, `mod.rs`; the code is a single `crates/optime-core/src/dsp/resample.rs` (~1250 lines)
with those as comment-delimited sections.

**Options**

- **24a. Fix the doc** to describe the single file and its sections.
- **24b. Split the file** to match the doc. `resample.rs` is large enough that this is defensible on
  its own merits, and the section banners already mark the seams.
- **24c. Leave it.**

**Recommended: 24a** now; **24b** is a reasonable separate task.

---

## Suggested batching

If you want to do this in passes rather than all at once:

1. **Correctness-risk pass** (items 5, 6): single-source the stop index and the position recurrence.
   Small, mechanical, removes two ways for a future edit to silently break chunk-invariance.
2. **Duplication pass** (items 3, 4, 10, 17): the edge gate, the shared compressor wrapper, the
   interleave helper, the `From` impl.
3. **Per-sample cost pass** (items 1, 2, 9, 21): the one with a measured payoff, and the largest.
4. **Polish pass** (items 8, 11, 12, 14, 15, 19, 22, 23, 24): naming, asserts, slicing, docs.

Items 7, 13, 16, 18, 20 are judgement calls with no clear default; decide them individually.

## Verification for any of this

Nothing here should change output. After each pass:

```sh
cargo test --workspace
cargo clippy --workspace --all-targets
cargo fmt --all --check
cargo clippy -p optime-app --target wasm32-unknown-unknown
cargo run -p optime-cli --release -- golden        # must match d5b241a exactly
```

Golden covers DS with the mixer off only. For the GBA mixer path, compare against a pristine
worktree (`git worktree add --detach ../optime-old <ref>`) with a throwaway test that hashes
`SynthController::fill` output under `PerDeviceSettings::enhanced_gba()` for ≥40 s of a real song —
4 s is not long enough to catch divergence, as this refactor found the hard way.

Performance, for item 1 or 9:

```sh
RAYON_NUM_THREADS=4 cargo run -p optime-cli --release -- export-album --benchmark 3% \
    demos/mother-3.gbaaudio crates/optime-app/src/song_names/mother_3.json /tmp/bench.flac
```

Baseline after `d5b241a`: 3.299 s median, 82.3× realtime, 5.39 Msamp/s (4 threads).

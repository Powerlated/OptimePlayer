# optime-ml run — 2026-07-16 m02-cuda

**Date:** 2026-07-16 · **Wall time:** ~1:20 (pretrain 70 min + fine-tune 8 min) · **Machine:** Intel Core Ultra 7 265HX, 18 threads + **NVIDIA RTX PRO 2000 Blackwell (8 GB)** · **Backend:** **CUDA (cubecl/nvrtc 13.2), f32, 1-way DP**
**Goal:** First **`m02_hier`** (set-transformer) full run, and the first run on **CUDA** — new `--features cuda` swaps `backend::{Inner, MlDevice}`, every bin follows. Two-stage: AR pretrain on real songs → supervised fine-tune on synthetic.

## Dataset

| set | source | songs | windows | split |
|---|---|---|---|---|
| synthetic (labeled) | `generate_data` (seq 256, seed 0xC0FFEE) | 4000 / 500 | 4000 / 500 | independent RNG |
| real (SSL) | `harvest ../demos` (all archives) | 5984 | **15736 / 943** | **song-level** (4× random-offset train, tiled val) |
| is-music labels | — | — | — | probe is generation-00 only; not run |

## Config

- **Model:** d_model 128, d_ff 512, **sub_d_ff 256**, heads 4, layers 4, **n_sub_layers 1**, dropout 0.1, n_frames **256**, ~1.72 GFLOP/window (set encoder ~1.18 G dominates trunk ~537 M)
- **Optim:** Adam, lr 3e-4, key_loss_weight 0.5, chord_smoothness 0.1 · **Epochs:** pretrain 20, fine-tune 12 · **Batch:** 32 / 32
- **Pretext:** AR next-frame (pitch-class + channel multi-hot, BCE) · **Augment (transpose):** on · **DP shards:** **1** (`default_shards()` returns 1 under `cuda` — sharding only fills CPU cores)

## Results

| stage | metric | value |
|---|---|---|
| pretrain (SSL, AR) | loss train / val | 0.5072→**0.1383** / 0.8972→**0.3720** (best **0.3518** @ ep19) |
| fine-tune | synthetic val key / chord acc | **99.6%** / **99.7%** |
| fine-tune | changes/seq (flicker) | 35.2 → **16.8** |
| is-music probe | — | n/a (generation 00 only) |
| SSL recon gap (real vs synth) | — | n/a (masked-frame MSE is m00's pretext; AR loss is not on that scale) |
| **real-song chord quality** | — | **pending hand evaluation in-app** (`.ocd` exported, see below) |

**Deliberately absent: `eval_real` chord agreement.** See Caveats — it is not an accuracy number and is no longer reported.

## Observations

- **CUDA works and is the right call for m02.** Sustained ~3.7 batch/s at 82% GPU util, 6 GB / 8 GB, 132 s/epoch pretrain and a flat 38.8 s/epoch fine-tune. The 2026-07-15 CPU runs thermally throttled from 176 s → 487 s per epoch; this GPU stayed at 51–60 °C with **no throttle flags** and full 2430 MHz SM clock.
- **Synthetic is now effectively solved — and that is a statement about the generator, not the model.** m02 reaches 99.7% chord / 99.6% key vs m00's 81.1% / 94.6%. Plausible: m02 consumes the *exact note set* per frame, so a synthetically-voiced chord is nearly readable by construction. A ceiling this high means synthetic val has stopped being an informative signal.
- **Warm-start transferred cleanly across backends and across the int-element fix.** The AR-pretrained trunk (CUDA) loaded into the fine-tune with no loss spike — epoch 1 already hit 83.1% chord, above m00's *final* 81.1%. `CompactRecorder` output stayed backend-agnostic as documented.
- **Smoothness penalty did its job**: predicted transitions per 256-frame window fell 35.2 → 16.8 and plateaued.
- **Unexplained pretrain slowdown**: epochs held 132 s through ep11, then degraded monotonically to 487 s by ep20. Not GPU thermal (no throttle reasons; 51 °C idle after). My own concurrent `clippy`/`cargo test --features cuda` overlapped part of that window but finished well before the rise continued, so they don't explain it. The fine-tune (13 min, same backend) showed **no** degradation, so it correlates with long runs / large window sets — cubecl allocator growth is a candidate. Unresolved.

## Caveats

- **The `eval_real` chord-agreement metric is discarded as of this run, per standing instruction.** For the record it read 4.0% for m02 vs m00's 19.4% (2026-07-15), and that comparison is worthless in both directions:
  1. **The reference is not ground truth.** `estimate.rs` is a chroma-template + Viterbi heuristic. On these real windows its top three outputs were *all Sus2* (~31% of frames) — an artifact of template-matching sparse game voicings.
  2. **It is structurally biased toward m00**, whose *input representation is chroma* — the very thing the reference matches on. Agreement partly measures "how chroma-template-like is this backbone", not correctness.
  3. **m02 was not collapsing**: 85 distinct predicted labels, 27.3% root-only agreement vs 4% joint — i.e. it mostly disputes *quality*, diversely, against a reference that is itself quality-biased. Agreement was also flat (~4–5%) across *every* fine-tune epoch including ep1 at 83% synthetic, which rules out progressive overfitting as the story.
- **No same-data m00 baseline exists.** This box was fresh: data was regenerated at seq 256 and only `hier` was trained. The 2026-07-15 m00 numbers are seq **128** on a different harvest — not comparable. Any m00-vs-m02 claim needs an m00 run on *this* dataset.
- Real-song chord quality is therefore **unmeasured**, not "4%". The `.ocd` for hand evaluation is exported and installed.
- Training is not reproducible (burn's param init/dropout draw from an unseeded global RNG); don't gate on exact losses.

## Changes landed with this run

- `--features cuda` (`backend.rs`): swaps `Inner`/`MlDevice`; default CPU build unchanged and still green (clippy + 41 tests pass on both).
- **`Cuda<f32, i64>`, not `Cuda<f32>`** — the default `IntElem = i32` panics `TypeMismatch(expected I32, got I64)` in `shared::eval_counts`, which reads `Int` tensors as a concrete `Vec<i64>` (matching ndarray). Cost the first fine-tune attempt at end of epoch 1. Pinning i64 keeps one element type across both builds.
- `parallel::default_shards()` → 1 under `cuda`.

## Next

- **Hand-evaluate the exported `.ocd` in-app** (Emerald/BPEE, m02) — the only trustworthy read on real music. Previous m00 table restorable via `git checkout crates/optime-app/src/chord_data/pokemon_emerald.ocd` to A/B by ear.
- **Retire synthetic val as a progress metric** at 99.7% — it no longer discriminates. Either enrich the generator (modal mixture, chromatic harmony, in-song modulation, denser real-like voicings) or move to a small hand-labeled real eval set.
- A **hand-labeled real chord set** is now the top blocker: it would replace both the discarded heuristic and the saturated synthetic val.
- If m02 is kept: train m00 on this same seq-256 dataset for an honest backbone comparison.
- Chase the pretrain epoch-time degradation if long runs become routine.

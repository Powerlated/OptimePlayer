# optime-ml run — contrived-sft

**Date:** 2026-07-17 · **Wall time:** ~2 min per SFT · **Machine:** WSL2, CUDA (RTX, `/usr/local/cuda-13.2`) · **Backend:** cuda, 1-way DP
**Goal:** Prove the stage-3 hand-label SFT loop runs end-to-end on real labels, deliberately overfit on one Emerald song and score another, then memorize both for a by-ear check in the app. Not a capability result — a plumbing + sanity experiment.

## What this run actually fixed

Before it could run at all, the hand-label path produced **zero** windows at `seq_len` 256:

1. **Windowing was anchored to frame 0.** `build::songs_from_annotations` / `eval_labeled` cut windows on an absolute grid, but every song opens with a pickup so its labels start a frame or two in (Route 110 at frame 2, Petalburg at 16). Window `[0,256)` always caught an unlabeled frame and was dropped — discarding runs the annotator had made exactly one window long. Worked at the old 128 window; the 128→256 bump on 2026-07-16 silently emptied the set. Fixed with `annotations::complete_windows`, which cuts from each labelled *run's* own start; both consumers share it now.
2. **No song had a `key`,** and the app has no key picker (CLAUDE.md's "set the key" was aspirational). SFT drops keyless songs. Hand-wrote keys for the two songs used here, derived from the diatonic fit over their own labelled spans (not heard): Route 110 → F# major, Petalburg → Eb major.
3. **`eval_labeled` / `chord_export` could only load `model`,** never `model_sft` — so "re-score after sft" scored the pre-SFT checkpoint. Added `--model <stem>` (via `Args::model_prefix`).

New flags: `--train-songs` / `--val-songs` (`Split::Songs`, a contrived hand-picked split that bypasses the hash holdout, checks its own disjointness, and shouts that its numbers aren't reportable).

## Dataset

| set | source | songs | windows | split |
|---|---|---|---|---|
| hand labels | `ml/annotations/pokemon_emerald.json` (BPEE) | Route 110 (360), Petalburg (362) | 1 + 1 | hand-picked, contrived |

Warm-start: synthetic fine-tune `models/02-hier/model` (the real, committed checkpoint — untouched; all runs wrote to throwaway `--out-dir`s under scratchpad).

## Config

- **Model:** m02 hier, default cfg, seq_len 256, ~1.9 MB checkpoint
- **Optim:** Adam, lr 1e-4, smoothness 0.1, augment on · **Batch:** 1 (run A), 2 (run B)

## Results

| run | train → val | epochs | outcome |
|---|---|---|---|
| pre-SFT baseline | — → Petalburg | — | root **53.5%**, quality 34.4%, joint 25.8% (`model`) |
| A: 360 → 362 | Route 110 → Petalburg | 20 | root **59.0%**, quality 59.4%, joint 41.0% (`model_sft`); train loss 1.29 → 0.62 |
| B: 360,362 → (none) | both → — | 1000 | train loss 6.0 → **0.039** (memorized); no val song left; `.ocd` baked for by-ear check |

Run B's `.ocd` (209 songs, 829 labels) validates against the parser and, on the two training songs, reproduces the memorized labels: Route 110 → `I (F#) / V (C#)`, Petalburg → `I (D#) / i (D#m)`, both matching the annotations.

## Observations

- The loop works: warm-start, factored heads, save-as-`model_sft`, re-score all functioned.
- Run A moved root +5.5 pts training on one song and scoring a genuinely different one — a signal the trunk generalizes a little, on n=1 window. Quality jumped more (34→59%) but that's one window of noise.
- Run B drove loss to 0.039 — clean memorization of 2 windows, exactly as the `<32 windows` warning predicts. The bar-to-bar roman numerals under the app's chord lane are now the real check.

## Caveats

- **n=1 window per side (run A); no held-out song at all (run B).** Neither is the real-music metric. The `eval_labeled --val-songs` warning says so.
- **Keys were inferred from the labels, not heard.** They should be confirmed by ear; a wrong key rotates every roman numeral.
- **The `.ocd` for run B was trained on both songs it displays** — Route 110 and Petalburg in the app are training data. The other ~200 Emerald songs in the table are unseen; those are the honest by-ear test.
- `git checkout crates/optime-app/src/chord_data/pokemon_emerald.ocd` restores the prior committed table.

## Next

- Confirm the two keys by ear; if wrong, re-annotate and re-fit.
- Build the annotation-tool key picker so keys stop being hand-edited JSON.
- Grow the label set past ~32 windows before any SFT number is worth reporting.

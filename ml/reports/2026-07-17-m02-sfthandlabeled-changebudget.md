# optime-ml run — change-budget-sft

**Date:** 2026-07-17 · **Wall time:** ~0:05 (8 epochs, 0.6s each) · **Machine:** WSL2, ndarray CPU · **Backend:** ndarray + 18-way CPU DP
**Goal:** Add a differentiable *sliding-window change budget* to the fine-tune loss that heavily penalises more than 2 chord changes per 4-beat window, and run it on the m02 (hier) real-label SFT stage.

## Dataset

| set | source | songs | windows | split |
|---|---|---|---|---|
| real (hand-labelled) | annotations/pokemon_emerald.json + demos | 5 (3 usable) | 2 / 1 | song-level 25% holdout |

Only 3 of 5 annotated songs carry a key; 2 were dropped (`songs_missing_key`), leaving **2 train / 1 val** windows. This is a smoke-scale run, not a real measurement of the penalty's effect.

## Config

- **Model:** m02 hier, seq_len 256 (= 64 beats), ~0.9M params, warm-started from `models/02-hier/model`
- **Optim:** Adam, lr 1e-4, key_loss_weight 0.5 · **Epochs:** fine-tune 8 · **Batch:** 16
- **Regularisers:** `chord_smoothness_weight` 0.1 (beat-aware TV), **`chord_budget_weight` 1.0** (new; window 4 beats, budget 2, `CHANGE_BUDGET_*`) · **Augment:** on · **DP shards:** 18

## Results

| stage | metric | value |
|---|---|---|
| fine-tune | synthetic-free real val key acc | 100% (1 window) |
| fine-tune | real val chord acc | 33% → 44% |
| fine-tune | **changes/seq (flicker, 256-frame val window)** | **126 → 96** |

## Observations

- The new penalty acts as intended: predicted transitions per window fell from 126 to ~96 over 8 epochs while chord accuracy rose — the model is producing fewer, longer-held chords, not collapsing to one.
- The soft change indicator is `1 − (1−tv_root)(1−tv_quality)` (either-head total variation), so an uncertain near-uniform frame is **not** miscounted as a change — this is why the first (overlap-based) formulation was rejected (it read low self-overlap of a uniform distribution as a constant chord change).
- Cost is O(seq): a cumulative sum over per-transition change turns every 16-frame window's soft change count into one subtraction. Unit tests pin within-budget≈0, 5-change window>2, short-sequence=0.
- Default `chord_budget_weight` is **0.0** in `TrainConfig`, so the synthetic `train` baseline and all existing reports/tests are unchanged; `sft` sets it to 1.0 (overridable via `CHORD_BUDGET`).

## Caveats

- **2 training windows.** Every number here is provisional to the point of anecdote; treat the flicker drop as "the gradient flows and points the right way", not as evidence of generalisation. The bottleneck is annotation coverage + missing song keys, not the penalty.
- `changes/seq` is measured on a single held-out window.
- A 96-change window over 64 beats is still ~1.5 changes/beat, above the 2-per-4-beats budget — the penalty is still pushing at epoch 8; more data (and/or a higher weight) would let it settle further.

## Next

- Set keys on the 2 dropped songs and label more contiguous 16-bar stretches so the SFT set is large enough to measure the penalty's real effect.
- Sweep `chord_budget_weight` (`CHORD_BUDGET`) once the set is larger; compare `eval-labeled` and by-ear `.ocd` A/B against the no-budget control.

# optime-ml run — 2026-07-15 augmented

**Date:** 2026-07-15 · **Wall time:** ~1:52 · **Machine:** AMD Ryzen 8840U, 8c/16t · **Backend:** ndarray + 16-way CPU DP
**Goal:** Add on-the-fly **transposition augmentation** + fix the **song-level train/val split**, vs. the [baseline](2026-07-15-m00-sftsynthetic-baseline.md). Everything else identical.

## Dataset

| set | source | songs | windows | split |
|---|---|---|---|---|
| synthetic (labeled) | generate_data | — | 4000 / 500 | independent RNG |
| real (SSL) | harvest all `demos/` (7 games) | 1433 | 10099 / 393 | **song-level** (clean) |
| is-music labels | song_names (emerald BPEE, mother3 A3UJ) | — | 1388 train / 126 val (1043/345 music/sfx) | — |

## Config

- **Model:** d_model 128, d_ff 512, heads 4, layers 4, seq_len 128, ~0.89M params
- **Optim:** Adam, lr 3e-4, key_loss_weight 0.5 · **Epochs:** pretrain 20, fine-tune 12 · **Batch:** pretrain 256, fine-tune 128
- **SSL mask:** 15% · **Augment (transpose):** **on** · **DP shards:** 16

## Results

| stage | metric | value |
|---|---|---|
| pretrain (SSL) | recon loss train / val | 0.191→**0.0448** / 0.078→**0.0512** |
| fine-tune | synthetic val key / chord acc | **94.6%** / **81.1%** |
| is-music probe | real val acc (n) | **85.7%** (126: 72/54 music/sfx) |
| **eval_real** | SSL recon real / synth (gap) | 0.0507 / 0.0509 (**−0.0001**) |
| **eval_real** | chord agreement all / chord-frames | **19.3%** / **19.4%** |

## Baseline comparison

| metric | baseline | augmented | Δ |
|---|---|---|---|
| real chord agreement (chord-frames) | 16.8% | **19.4%** | **+2.6** |
| real chord agreement (all frames) | 19.1% | 19.3% | +0.2 |
| SSL recon gap (real−synth) | −0.013 | −0.0001 | → ~0 |
| real val recon | 0.039 *(leaky)* | 0.051 *(honest)* | — |
| synthetic val key / chord | 94.8% / 83.0% | 94.6% / 81.1% | ~flat |
| is-music probe val | 93.4% *(leaky, 25% neg)* | 85.7% *(clean, 43% neg)* | — |

## Observations

- **The evaluation is now honest.** Song-level split makes val recon *higher* than train (0.051 > 0.045) — a normal generalization gap — instead of the baseline's leaky inversion (val < train). The is-music val is also balanced (43% negatives vs 25%), so its 85.7% is a harder, truer number than the baseline's 93.4%, not a regression.
- **Augmentation removed synthetic's "easy-key" advantage.** The real−synthetic recon gap collapsed from −0.013 to ~0: with all 12 transpositions seen, synthetic is no longer artificially easier, and the encoder generalizes equally to both.
- **Small real-chord gain.** Chord-frame agreement rose 16.8% → 19.4% (+2.6). Real, but modest — expected, because transposition fixes *key-invariance*, not *harmonic vocabulary*, which is the bigger gap.
- **In-distribution unchanged** (synthetic 94.6%/81.1% ≈ baseline), so augmentation didn't cost accuracy; it regularized (train recon 0.045 slightly above baseline's 0.041).
- Thermal throttling again: pretrain epochs 176s → 487s over ~96 min.

## Caveats

- Same heuristic-reference caveat: 19.4% is agreement with a template/Viterbi estimator, not verified accuracy.
- Real val is small (393 windows / ~143 songs) after the clean split — noisier estimate, but trustworthy.

## Next

- **Harmonic vocabulary is now the bottleneck**, not key-invariance. Richer synthetic generator: modal mixture, borrowed/chromatic chords, secondary dominants beyond current templates, pedal points, in-song modulation.
- More harvested games (SSL diversity is still ~7 titles).
- A small hand-labeled real chord eval set → replace the heuristic reference with a trustworthy accuracy number.

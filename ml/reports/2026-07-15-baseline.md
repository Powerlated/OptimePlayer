# optime-ml run — 2026-07-15 baseline

**Date:** 2026-07-15 · **Wall time:** ~1:44 · **Machine:** AMD Ryzen 8840U, 8c/16t · **Backend:** ndarray + 16-way CPU DP
**Goal:** First full-scale run of the SSL-pretrain → synthetic fine-tune → frozen is-music probe pipeline, with CPU data parallelism. No augmentation (baseline).

## Dataset

| set | source | songs | windows | split |
|---|---|---|---|---|
| synthetic (labeled) | generate_data | — | 4000 / 500 | independent RNG |
| real (SSL) | harvest all `demos/` (7 games: ace-attorney, NSMB, black2, emerald, heartgold, platinum, SM64DS, mother3) | — | 9443 / 1049 | **window-level** (leaky) |
| is-music labels | song_names (emerald BPEE, mother3 A3UJ) | — | 1362 train / 152 val (1001/361 music/sfx) | — |

## Config

- **Model:** d_model 128, d_ff 512, heads 4, layers 4, seq_len 128, ~0.89M params
- **Optim:** Adam, lr 3e-4, key_loss_weight 0.5 · **Epochs:** pretrain 20, fine-tune 12 · **Batch:** pretrain 256, fine-tune 128
- **SSL mask:** 15% · **Augment (transpose):** off · **DP shards:** 16

## Results

| stage | metric | value |
|---|---|---|
| pretrain (SSL) | recon loss train / val | 0.193→**0.041** / 0.067→**0.039** |
| fine-tune | synthetic val key / chord acc | **94.8%** / **83.0%** |
| is-music probe | real val acc (n) | **93.4%** (152) |
| **eval_real** | SSL recon real / synth (gap) | 0.0389 / 0.0522 (**−0.013**) |
| **eval_real** | chord agreement all / chord-frames | **19.1%** / **16.8%** |

## Observations

- **SSL transfer is real.** Warm-started fine-tune hit 52.6% key acc at epoch 1 (vs ~30% cold), and the pretrained encoder reconstructs real songs *better* than synthetic (negative recon gap). The frozen is-music probe reaching 93.4% confirms the latent separates music from SFX without touching the encoder.
- **Strong in-distribution, weak out-of-distribution.** Synthetic val is excellent (95%/83%); real-song chord agreement is only ~17% — the synthetic→real domain gap.
- **Thermal throttling** stretched pretrain epochs 173s → 413s over ~90 min of sustained all-core load (28W APU). Loss kept falling regardless; total run ~2× longer than an unthrottled estimate.
- **CPU DP:** ~2× wall-clock speedup, ~7–8 busy cores; ceiling is memory bandwidth, not shard count.

## Caveats

- **Chord agreement ≠ accuracy.** The reference is a heuristic (chroma-template + Viterbi), not ground truth, so 17% is *disagreement between two imperfect estimators* on hard polyphonic real music, not "83% wrong." No trustworthy real chord labels exist yet.
- **Real val split is window-level** → windows of one song can straddle train/val, so the SSL val recon (0.039) is optimistic.
- **is-music labels are lopsided:** emerald yielded only music ids, so nearly all sfx negatives come from mother3.

## Next

- **Transposition augmentation + song-level split** (built, tested — this is the next run).
- More games harvested; richer synthetic generator (modal mixture, chromatic mediants, pedal points, in-song modulation).
- A small hand-labeled real chord eval set to replace the heuristic reference.

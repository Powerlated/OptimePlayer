# optime-ml run — <short-id>

**Date:** YYYY-MM-DD · **Wall time:** <h:mm> · **Machine:** <cpu, cores> · **Backend:** ndarray + <N>-way CPU DP
**Goal:** <one sentence — what this run changes or tests vs. the previous one>

## Dataset

| set | source | songs | windows | split |
|---|---|---|---|---|
| synthetic (labeled) | generate_data | — | <n_train>/<n_val> | independent RNG |
| real (SSL) | harvest <archives> | <n_songs> | <n_train>/<n_val> | <song-level \| window> |
| is-music labels | song_names <games> | — | <n> (<music>/<sfx>) | — |

## Config

- **Model:** d_model <>, d_ff <>, heads <>, layers <>, seq_len <>, ~<>M params
- **Optim:** Adam, lr <>, key_loss_weight <> · **Epochs:** pretrain <>, fine-tune <> · **Batch:** pretrain <>, fine-tune <>
- **SSL mask:** <>% · **Augment (transpose):** on/off · **DP shards:** <>

## Results

| stage | metric | value |
|---|---|---|
| pretrain (SSL) | recon loss train / val | <start>→<end> / <start>→<end> |
| fine-tune | synthetic val key / chord acc | <>% / <>% |
| is-music probe | real val acc (n) | <>% (<n>) |
| **eval_real** | SSL recon real / synth (gap) | <> / <> (<±>) |
| **eval_real** | chord agreement all / chord-frames | <>% / <>% |

## Observations

- <e.g. warm-start transfer, throttling, convergence shape>

## Caveats

- <known limits: heuristic reference, leakage, label imbalance, etc.>

## Next

- <highest-leverage follow-ups>

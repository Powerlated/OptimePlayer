# optime-ml — key & chord recognition transformer

A transformer that reads a stream of **synthesizer note events** (the same data
OptimePlayer's `SynthEvent` stream carries) and predicts:

- the **global key** of the excerpt (24 classes: 12 major + 12 minor), and
- the **chord at every frame** (121 classes: 12 roots × 10 qualities + no-chord).

Intended to eventually drive a live circle-of-fifths display and annotated chord
readout over playing music. Built with the [Burn](https://burn.dev) Rust ML
framework (pure-Rust, CPU `ndarray` backend, multi-core).

## Why note events, not audio

The input is **not** chroma extracted from rendered audio. It is the note stream
itself — exact pitch + velocity + instrument role + stereo pan + onset — because
OptimePlayer already synthesizes music from sequenced note data, so those events
are available losslessly at runtime. Training and inference share one extractor
([`features.rs`](src/features.rs)), so the model sees an identical representation
in both settings.

## Pipeline

```text
progression + voicing ─► NoteEvent{start,end,pitch,velocity,instrument,pan}   (retained raw data)
NoteEvent stream       ─► per-frame feature grid  [n_frames, 57]
frame grid             ─► Transformer encoder (learned pos-emb, pre-norm, 4 layers)
                          ├─► per-frame CHORD head  → 121 classes
                          └─► pooled (mean) KEY head → 24 classes
```

### Per-frame feature vector (57 dims)

| block | dims | meaning |
|-------|------|---------|
| chroma | 12 | velocity-weighted active pitch classes |
| bass | 12 | lowest sounding pitch class (root/inversion cue) |
| melody | 12 | highest sounding pitch class |
| onset | 12 | pitch classes attacking this frame |
| scalars | 9 | polyphony, total velocity, bass/mean MIDI, pitch spread, pan mean/spread, percussion energy, onset flag |

Percussion notes contribute energy/metadata only — never a harmonic pitch class.
The four pitch-class blocks are L2-normalised so harmonic *shape* drives the
signal rather than absolute loudness.

## Synthetic data

No real audio is needed. [`theory.rs`](src/theory.rs) encodes 24 keys, 10 chord
qualities and diatonic harmony; [`progression.rs`](src/progression.rs) generates
chord progressions two ways:

- **Curated templates** — I–V–vi–IV, ii–V–I, 50s doo-wop, Pachelbel, 12-bar
  blues, circle-of-fifths, Andalusian cadence, minor ii–V–i, rhythm-changes, …
  with random seventh upgrades, **secondary dominants** and **borrowed chords**.
- **Functional-harmony Markov walk** — tonic → predominant → dominant → tonic
  transition bias over the diatonic degrees, with occasional secondary dominants.

[`notes.rs`](src/notes.rs) arranges each progression into note events across three
styles (block pads, rhythmic comping, arpeggios), with a separate bass voice,
optional melody (chord tones on strong beats, scale passing tones between),
optional percussion, inversions, dynamics, panning, and the occasional rest
(→ genuine no-chord frames). Every frame is labelled with its ground-truth chord;
every song with its key.

The **raw note-event songs are the retained dataset** (`data/*.bin`); per-frame
features are derived deterministically on load, so the exact same code produces
the training tensors and the live-inference tensors.

## Real songs + self-supervised pretraining

Synthetic progressions give free chord/key *labels* but don't match real
game-music note statistics (pedal points, ostinatos, non-functional harmony). So
training is a **two-stage hybrid**:

1. **Self-supervised pretraining on real songs.** Behind the `harvest` feature,
   [`harvest.rs`](src/harvest.rs) runs each console's device sequencer *headlessly*
   (the same trick the engine's visualiser look-ahead uses) and turns its
   `SynthEvent` stream into unlabeled `NoteEvent` windows on the same
   4-frames-per-beat grid the synthetic data uses. [`pretrain.rs`](src/pretrain.rs)
   then trains the encoder with a **masked-frame** objective (hide ~15% of frames,
   reconstruct their pitch-class content) — no labels, just the real note-event
   distribution.
2. **Supervised fine-tune with synthetic labels.** `train --pretrained` warm-starts
   the encoder from stage 1 and trains the chord/key heads on the synthetic labeled
   set exactly as before. **SSL closes the domain gap; the labels still come from
   theory** — "Cmaj7" is a human-defined output space you can't read off raw songs.

Because real songs are unlabeled, [`estimate.rs`](src/estimate.rs) provides a
**training-free** chord reference (chroma-template matching + Viterbi smoothing);
`eval_real` scores the model's chord agreement against it and reports the
masked-reconstruction loss gap (real vs. synthetic) as the domain-gap number.

### Augmentation & splits

Both training stages apply **on-the-fly transposition** ([`Song::transpose`](src/notes.rs),
enabled by `TrainConfig`/`PretrainConfig::augment`, default on): each song is
shifted by a random semitone (all 12 pitch-class rotations, register kept in range)
per shard per epoch, with the key and per-frame chord labels shifted to match. It's
a correct, free augmentation that teaches key-invariance and multiplies effective
data. The real-song train/val split is done at the **song level** (`harvest` groups
windows by source song, then splits songs) so windows of one song never straddle
the split — otherwise the SSL val loss is optimistic.

### "Is-music" probe

[`probe.rs`](src/probe.rs) trains a small binary head on the **frozen** SSL
encoder to tell real music from SFX/jingles. The weak labels come from the app's
`song_names/{pokemon_emerald,mother_3}.json`: a GBA song whose id is curated is
music, any other song-table entry is not (`harvest --annotate <CODE>=<json>`
stamps them via `GbaRom::game_code()`). The encoder is run once and its pooled
features cached — only the decoder trains, so the shared representation never
moves. It doubles as an SFX filter for the pretraining corpus.

## Usage

```sh
# 1. Generate + retain the SYNTHETIC labeled dataset  (n_train n_val seq_len [seed])
cargo run --release --bin generate_data -- 4000 500 128
#   → data/train.bin, data/val.bin, data/gen_config.json

# 2. (optional) Harvest REAL game songs for pretraining  (rom_dir [seq_len] [val_fraction])
#    --annotate <GAME_CODE>=<song_names.json> (repeatable) adds weak is-music labels
cargo run --release --features harvest --bin harvest -- ../demos 128 \
    --annotate BPEE=../crates/optime-app/src/song_names/pokemon_emerald.json \
    --annotate A3UJ=../crates/optime-app/src/song_names/mother_3.json
#   → data/real_train.bin, data/real_val.bin

# 3. (optional) Self-supervised pretrain on real songs  (epochs batch [lr])
cargo run --release --bin pretrain -- 20 32
#   → models/pretrained(.mpk) + models/pretrained.json

# 4. Train the chord/key heads  (epochs batch [lr] [--pretrained <prefix>])
cargo run --release --bin train -- 12 32 --pretrained models/pretrained
#   without --pretrained: from-scratch synthetic-only baseline (identical to before)
#   → models/model.mpk (weights) + models/model.json (architecture)

# 5. (optional) Train the frozen "is-music" probe on the annotated real windows
cargo run --release --bin probe -- models/model
#   → models/probe(.mpk) + models/probe.json

# 6. Inspect a prediction vs ground truth  (optional val index)
cargo run --release --bin infer -- 0

# 7. Measure the real-music gap: recon-loss gap + chord agreement + is-music accuracy
cargo run --release --bin eval_real -- data/real_val.bin models
```

To compare the synthetic-only baseline against the SSL+fine-tune model, train each
into its own directory (e.g. copy `models/` aside) and point `eval_real` at each.

### Multi-core training (data parallelism)

The CPU backend is `Autodiff<NdArray<f32>>`. This model is **small**, so backend
intra-op parallelism (threaded matmul) can't fill many cores — a single training
stream tops out at ~2 cores regardless of batch size, and threaded-CPU (candle)
and GPU (wgpu/ROCm) backends were measured to be no better at this size. Instead
each optimizer step is **data-parallel** ([`src/parallel.rs`](src/parallel.rs)):
the minibatch is split into shards that differentiate the shared weights
concurrently (rayon, via per-shard `fork`), their gradients are summed, and one
Adam step is taken — exact synchronous data-parallel SGD.

Shard count defaults to the logical-core count; override with `DP_SHARDS=N`
(e.g. `DP_SHARDS=8` to match physical cores). On an 8-core/16-thread Ryzen this
gives ~2× wall-clock speedup and ~7–8 busy cores; the ceiling is memory bandwidth
(the parallel matmuls saturate DRAM before the ALUs), not shard count. Per-epoch
wall time is printed so you can tune `DP_SHARDS` for your machine.

## Layout

| file | role |
|------|------|
| `src/theory.rs` | pitch classes, chords, keys, diatonic harmony, label space (unit-tested) |
| `src/progression.rs` | progression generation (templates + Markov) |
| `src/notes.rs` | arrangement/voicing → note events + per-frame labels |
| `src/features.rs` | note events → per-frame feature grid (the model input) |
| `src/data.rs` | dataset generation, on-disk retention, example shape |
| `src/model.rs` | Burn transformer (chord + key heads + SSL reconstruction head) |
| `src/train.rs` | multi-task training loop, evaluation, model save/load, pretrained warm-start |
| `src/parallel.rs` | CPU data-parallel optimizer step (shard → parallel grads → sum → one step) |
| `src/pretrain.rs` | self-supervised masked-frame pretraining on real songs |
| `src/harvest.rs` | *(feature `harvest`)* headless `SynthEvent` → real songs + weak is-music labels |
| `src/probe.rs` | frozen-encoder "is-music" linear probe (weak song-name labels) |
| `src/estimate.rs` | training-free chroma-template + Viterbi chord reference |
| `src/infer.rs` | run a trained model → key + merged chord timeline |
| `src/bin/*` | `generate_data`, `train`, `infer`, `pretrain`, `harvest`, `probe`, `eval_real` CLIs |
| `data/` | retained datasets: synthetic (`train`/`val`) + harvested real (`real_train`/`real_val`) |
| `models/` | trained weights + architecture config (`model`, `pretrained`) |

## Integrating with OptimePlayer (future)

Adapt the live `SynthEvent` stream (`NoteStarted` / `VoiceVolume` / `VoicePitch`
/ `TrackPan` / `NoteReleased`) into `NoteEvent`s on the same frame grid, call
`features::extract`, then `infer::predict` on a trailing window for a rolling
key/chord read-out. The model and label space are console-agnostic — the note
representation is the only contract.

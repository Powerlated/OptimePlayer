# Tuning the top-end exciter against a reference soundtrack

Measured 2026-08-08. Reproduce with the two commands at the bottom.

## The question

The engine's top end is a side effect. In crunch mode (`SincOutputNyquist`) the resampler
reconstructs a *stepped* source, and a step is a zero-order hold: the harmonics above the source's
own band are whatever the staircase happened to contain, at whatever level the staircase put them.
Nothing chose them.

The alternative tested here is to reconstruct the source cleanly (`SincSampleNyquist`, no ZOH
excitation at all) and generate the top end deliberately — a saturating waveshaper on a high band,
with the corner, the drive and the blend as free parameters. `dsp/exciter.rs` is that stage, and it
is antialiased by the antiderivative method so the harmonics it invents do not fold back down the
band the way a naive waveshaper's would.

## Reference corpus

DELTARUNE's soundtrack (`steamapps/common/DELTARUNE/mus`), 339 `.ogg` files, first 90 s of each,
reduced by `timbre-profile` to a per-element mean and standard deviation:

| descriptor | mean | sd |
| --- | --- | --- |
| crest | 15.7 dB | 2.4 |
| dynamic range (p95 − p10 of 400 ms RMS) | 9.7 dB | 9.0 |
| spectral flux | 2.96 dB/frame | 1.25 |

The spectrum is 28 third-octave bands from 30 Hz to 16 kHz, in dB about each track's own mean band.
It peaks at band 6 (≈130 Hz) at +13.6 dB and falls monotonically to −37.4 dB at band 27 (≈14 kHz).

The corpus standard deviation is what weights the loss. That is deliberate — a band where 339
finished tracks disagree by ±18 dB says little about what any one track *should* be — but it has a
consequence measured below.

## Result

Pokémon Emerald, 193 tuning songs and 24 holdout, 25 s each rendered at 48 kHz. 12 parameters
(exciter ×3, high-band compressor ×6, master shelf ×3), 300 SPSA steps, minibatch 6.

| configuration | tuning set | holdout |
| --- | --- | --- |
| zero-order-hold excitation (shipped `enhanced_gba`) | **0.9717** | **0.7032** |
| shaper excitation, untuned | 1.0405 | 0.7102 |
| shaper excitation, tuned | 0.9981 | 0.7099 |
| …same, exciter blend forced to zero | 1.0352 | 0.7132 |

**The tuned exciter does not beat the zero-order hold on this metric.** It closes about 60% of the gap it
started with on the tuning set and none of it on the holdout, and the shipped crunch resampler stays
ahead of both.

Two things are nonetheless clear from the ablation. The exciter, not the EQ, is doing the work:
silencing its blend and keeping every other tuned value gives back 0.037 of the 0.042 the tuning
won, so the compressor and shelf contributed almost nothing. And the exciter does what a top-end
exciter is supposed to do, in the bands where it operates:

| band | ≈centre | target | zero-order hold | tuned exciter |
| --- | --- | --- | --- | --- |
| 24 | 7.3 kHz | −21.4 | −19.0 | −22.9 |
| 25 | 9.1 kHz | −25.9 | −19.7 | −26.1 |
| 26 | 11.4 kHz | −31.2 | −21.7 | −29.2 |

The zero-order hold is **6–9 dB too bright** at 9–11 kHz against the reference; the tuned exciter
lands within 0.2–2 dB. That is a large, real correction, and it is invisible in the scalar loss
because those are exactly the bands where the corpus deviation is ±10–16 dB, so inverse-deviation
weighting all but ignores them.

## Why the loss barely moves

The loss is dominated by the mid bands, where the corpus is tight (±7–9 dB) and Emerald already
agrees with it. What Emerald actually gets wrong against DELTARUNE is the bottom: bands 0–2 (34–53
Hz) are 6.3, 4.8 and 2.8 dB short of the reference. No exciter can fix a bass deficit, so twelve
parameters aimed at the top end are being scored mostly on a mismatch none of them can reach.

This is a limitation of the objective, not of the search — the SPSA gradient is sound (both
evaluations of a step share a minibatch, so song variance cancels out of the difference), and the
descent is visible in the trace. The metric is simply insensitive to the change being made.

Worth trying next, roughly in order of expected value: weight the loss by band rather than by corpus
deviation alone, so the top octaves can actually register; give the search something with authority
over the low end (a low shelf, or the bass-mono crossover) so the dominant error term is reachable;
and score against DELTARUNE tracks matched for instrumentation rather than the whole soundtrack,
since 339 tracks include ambience and drones whose spectra are nothing like a Game Boy Advance
sequence.

## Caveats

- The 12 parameters are not identifiable from one scalar loss; the ablation above separates the
  exciter from the rest, but not the compressor from the shelf.
- The holdout is strided across album order, so it is representative, but 24 songs is a small test
  set and the tuning-versus-holdout loss levels are not comparable to each other — only within a
  column.
- Renders are 25 s from the start of each song, which over-weights intros.
- Tuning ran at 48 kHz, matching live playback. `export-album` renders at 32768 Hz, where the
  exciter's own harmonics have less room above them.

## Reproducing

```sh
cargo build --release -p optime-cli

./target/release/optime-cli timbre-profile \
    "/c/Program Files (x86)/Steam/steamapps/common/DELTARUNE/mus" \
    scripts/deltarune-timbre.json --seconds 90

./target/release/optime-cli tune-exciter \
    demos/pokemon-emerald.gbaaudio \
    crates/optime-app/src/song_names/pokemon_emerald.json \
    scripts/deltarune-timbre.json \
    --out scripts/exciter-tuned.json \
    --steps 300 --batch 6 --seconds 25 --eval-songs 16 --eval-every 25 --holdout 24
```

About 23 minutes for the tuning run. `--load scripts/exciter-tuned.json --steps 0` re-scores a saved
parameter set without repeating the search.

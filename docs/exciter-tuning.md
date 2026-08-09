# Replacing the zero-order hold's excitation with a saturating exciter

Measured 2026-08-08. Reproduce with the commands at the bottom.

## The question

The engine's top end is a side effect. In crunch mode (`SincOutputNyquist`) the resampler
reconstructs a *stepped* intermediate mixer bus, and a step is a zero-order hold. Two things follow
from that, and they are inseparable: the images the staircase throws above the bus's Nyquist are
what give the sampled voices their air, and the hold's `sinc(f / 13379)` envelope puts an exact null
at 13379 Hz. You cannot keep the air and drop the null, because they are the same artifact.

So: reconstruct cleanly, losing both, and put the air back on purpose with a saturating waveshaper —
`dsp/exciter.rs`, antialiased by the antiderivative method so the harmonics it invents do not fold
back down the band.

## What the notch measurement says

`notch-sweep` renders songs, averages a 16384-point spectrum (2.9 Hz per bin at 48 kHz), and reports
how far a narrow window around a probe sits below the shoulders either side. Positive is a dip. It
also scans 2.5–20 kHz for the deepest dip anywhere, which is how the second finding below turned up
at all.

| configuration | notch @13379 | deepest notch anywhere | timbre |
| --- | --- | --- | --- |
| Enhanced (crunch mixer @13379) | **7.68 dB** | 13350 Hz +8.2 | 0.2693 |
| Original | **8.07 dB** | 13350 Hz +8.1 | 0.3750 |
| Enhanced+, *first attempt* (clean mixer @13379) | 0.38 dB | **7500 Hz +6.3** | 0.3381 |
| Enhanced+, *shipped* (clean mixer @48000) | **0.31 dB** | 3950 Hz +5.3 | 0.2714 |

Two findings, both from measurement rather than reasoning:

**The first Enhanced+ did kill the 13379 Hz null — and replaced it with a worse one at 7500 Hz.**
Clean-reconstructing a 13379 Hz bus band-limits everything to 6689.5 Hz, so real content stops dead
there and only exciter harmonics continue above. That cliff reads as a 6.3 dB dip in the presence
region, which is a more audible place to have a hole than 13.4 kHz.

**Raising the mixer bus to 48 kHz removes the cliff too**, because there is no intermediate
band-limit left: sampled voices are resampled once, straight to the output rate, instead of twice
via 13379. The residual 3950 Hz dip is *not* an artifact — every configuration shows it, including
one with no mixer bus at all (3950 Hz +5.0 dB), so it is Emerald's own average spectrum and it is
the floor this measurement can reach.

## The exciter runs on the sampled bus only

PSG voices bypass the intermediate mixer entirely, so they never had staircase images and there is
nothing to replace. Exciting them adds harshness instead of trading it away. `excite_block`
therefore runs inside the mixer path — after `route_mixer_block`, before the high-band compressor
that tames it — and is gated on `use_mixer` exactly like the sampled compressor. The sweep confirms
the gating: with no mixer bus, exciter on and off score identically to four decimal places.

## Tuning result

193 tuning songs, 24 strided holdout, 25 s each at 48 kHz, 12 parameters, 300 SPSA steps.

| configuration | tuning set | holdout |
| --- | --- | --- |
| zero-order-hold excitation (Enhanced) | 0.9717 | 0.7032 |
| shaper excitation, untuned | 0.9507 | 0.6715 |
| **shaper excitation, tuned** | **0.9430** | **0.6656** |
| …same, exciter blend forced to zero | 0.9506 | 0.6712 |

The tuned shaper now beats the zero-order hold on both the tuning set and the holdout, so the win
generalises rather than being fitted to the songs it was tuned on. The two structural corrections
did the heavy lifting — restricting the exciter to the sampled bus and lifting the mixer to 48 kHz
moved the untuned baseline from 1.0405/0.7102 (an earlier run, where it *lost* to the hold) to
0.9507/0.6715. Continuous tuning added 0.008 on top, of which the exciter itself accounts for
essentially all: silencing its blend gives back 0.0076 of it.

This is worth stating plainly, because it is the lesson of the whole exercise. Twelve continuous
parameters searched for 23 minutes bought less than two discrete choices about the shape of the
chain, and no setting of any of them could have removed a null — a zero is put there by structure,
and only structure moves it. That is why `notch-sweep` enumerates rather than descends.

## Caveats

- The tuning and holdout loss levels are not comparable to each other, only within a column.
- Renders are 25 s from the start of each song, which over-weights intros.
- `drive` (22.1 of a 24 ceiling) and the compressor threshold (−59.97 of a −60 floor) sit at their
  bounds, so the search wanted to go further than the ranges allow.
- Enhanced+ mixes at 48 kHz, which is **not** what a Game Boy Advance does. It is the "sounds good"
  preset; Enhanced and Original remain the hardware-shaped ones.
- The timbre metric is dominated by mid bands where the corpus is tight, and it is largely blind to
  the top two octaves where the corpus disagrees with itself by ±10–16 dB. It also cannot see a
  narrow notch at all — that is why the separate `notch-sweep` measurement exists.

## Reproducing

```sh
cargo build --release -p optime-cli

./target/release/optime-cli timbre-profile \
    "/c/Program Files (x86)/Steam/steamapps/common/DELTARUNE/mus" \
    scripts/deltarune-timbre.json --seconds 90

./target/release/optime-cli notch-sweep \
    demos/pokemon-emerald.gbaaudio \
    crates/optime-app/src/song_names/pokemon_emerald.json \
    scripts/deltarune-timbre.json --songs 10 --seconds 20

./target/release/optime-cli tune-exciter \
    demos/pokemon-emerald.gbaaudio \
    crates/optime-app/src/song_names/pokemon_emerald.json \
    scripts/deltarune-timbre.json \
    --out scripts/exciter-tuned.json \
    --steps 300 --batch 6 --seconds 25 --eval-songs 16 --eval-every 30 --holdout 24
```

`notch-sweep` takes about 20 s, `tune-exciter` about 23 minutes. `--load scripts/exciter-tuned.json
--steps 0` re-scores a saved parameter set without repeating the search.

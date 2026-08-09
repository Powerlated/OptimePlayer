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
| Enhanced+, *shipped* (clean mixer @48000) | **0.15 dB** | 3950 Hz +5.3 | 0.2780 |

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

## Odd harmonics are the harsh ones

`tanh` is an odd function, and an odd nonlinearity produces only odd harmonics. From a 3 kHz
fundamental those land at 9 kHz and 15 kHz — a twelfth and two octaves plus a third above, intervals
that read as grit rather than as the note. That is the entire palette a symmetric shaper has, so a
tuner told to add air with nothing but `crossover`, `drive` and `amount` can only add the harsh
kind, and turning it down trades harshness for the clarity it was there to provide.

`bias` displaces the signal along the curve — `tanh(drive·(x + bias))`, offset to keep the origin
fixed — which makes it asymmetric, and asymmetry is what produces *even* harmonics. The second
harmonic is an octave, which the ear reads as loudness and warmth rather than as distortion. So the
bias knob is the axis along which harshness and brightness can be traded independently, which is
what "the same clarity with less harshness" actually requires. `bias_is_what_creates_even_harmonics`
pins the claim: at bias 0 the second harmonic is below 1e-3, and off zero it is more than twenty
times that.

## Making the search actually search

Four findings about the optimiser itself, all negative results worth keeping, because each one cost
a full run to establish.

**The metric was nearly blind above 3 kHz.** Bands are weighted by the inverse of the corpus's own
standard deviation, and DELTARUNE disagrees with itself by ±10–18 dB in the top octaves, so those
bands were discounted into irrelevance — which is precisely where an exciter works. `--focus-hz` /
`--focus-weight` add a multiplier for bands centred above a frequency; at the default 3000 Hz and
6×, the top seven bands carry 56% of the loss instead of 11%. Without that, the search had no reason
not to saturate the presence region, and indeed the first tuned result put the exciter crossover at
1351 Hz with a drive of 22 — a "top-end" exciter chewing on everything above 1.3 kHz.

**Widening the bounds alone changed nothing, and could not have.** A run with every range widened
(drive to 200, threshold to −100 dB, and so on) finished with parameters bit-identical to its
starting point: it never once beat where it began. Parameters are searched in a squashed coordinate,
so holding the perturbation size fixed while widening a range means each probe jumps a much larger
distance in the actual parameter — the search got noisier, not freer. Widening a bound is only
useful together with a smaller perturbation.

**A tilt is free, so the shelf must not be allowed to become one.** With the shelf's corner free to
fall anywhere, the search put it at 1423 Hz, −15.2 dB, Q 0.115 — not a shelf but a broadband tilt.
Against a mean-removed spectrum a tilt costs nothing to apply and buys shape everywhere, and with
the top bands weighted 6× it bought them by spending the bottom: 60–130 Hz went from within 0.6 dB
of target to +2.0…+3.6 dB over it. The aggregate loss *improved* while the result got audibly worse.
The fix is a bound, not a penalty: the master shelf is a top-end tool, so its corner is now confined
to 3 kHz and up, where it cannot tilt the whole spectrum. A `Below focus_hz` line is now reported
next to the `Above` one, because a number that must not regress has to be on screen.

**The minibatch was the real obstacle.** Song-to-song variation in this objective is far larger than
the differences between parameter settings, so a gradient estimated on 6–8 random songs pointed
somewhere useful for those songs and nowhere useful for the measured set; two runs wandered above
their starting point and never recovered. `--deterministic` estimates the gradient on the same fixed
evaluation songs every step, which makes the objective a deterministic function and removes that
noise entirely. It is the first configuration that descends — and the holdout immediately caught what it cost:
train fell 0.4970 → 0.4878 while the holdout *rose* 0.3292 → 0.3702, and above 3 kHz the holdout
went from 0.1484 to 0.1976. A deterministic objective over fourteen songs is a fourteen-song
objective. The configuration that finally both descends and generalises is the middle one — a
stochastic minibatch of 10 for the gradient, a 24-song set for selecting the best iterate, and the
shelf bounded — which is what produced the shipped numbers below.

## Where it ended up

Re-tuned against the ≥3 kHz-weighted metric with the shelf bounded and a stochastic minibatch, 20 s
renders, 24-song selection set, 24-song strided holdout. Only compare within a column: each row set
below was measured under one configuration, and the absolute levels move when the render length or
song count does.

| configuration | tuning set | holdout |
| --- | --- | --- |
| **≥3 kHz only** — zero-order hold | 0.2046 | 0.3316 |
| **≥3 kHz only** — Enhanced+ before this round | 0.1517 | 0.2516 |
| **≥3 kHz only** — Enhanced+ shipped now | **0.1416** | **0.2400** |
| whole spectrum — zero-order hold | 0.4590 | 0.5794 |
| whole spectrum — Enhanced+ before | 0.4236 | 0.5403 |
| whole spectrum — Enhanced+ now | 0.4204 | 0.5384 |
| …exciter blend forced to zero | 0.4623 | 0.5548 |

Above 3 kHz — the region this round was aimed at — Enhanced+ is now **28% closer to the reference
than the zero-order hold on held-out songs**, and 4.6% closer than it was before the round. The
whole-spectrum number improves too, and the exciter is carrying it: silencing its blend costs more
than the entire tuning gain.

The honest cost is in the last column of the run: below 3 kHz the holdout moved 0.5245 → 0.5329, a
1.6% regression. That is the price of weighting the top six-fold, it is two orders of magnitude
smaller than the tilt disaster that bound removed, and it is reported every run so it cannot drift
unnoticed.

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

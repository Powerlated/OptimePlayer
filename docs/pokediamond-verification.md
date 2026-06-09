# Audio-engine verification against `pret/pokediamond`

This document records a source-level audit of the Optime Player synthesis core against the
reverse-engineered NitroSDK sound library in [`pret/pokediamond`](https://github.com/pret/pokediamond)
(`arm7/lib/src/SND_*.c`). The Rust engine was originally ported from the legacy JS engine
(`legacy-js/`), and the golden-parity test only proves **Rust ↔ JS** equivalence — it does *not*
prove equivalence with pokediamond. This audit closes that gap for the numeric core and lists the
remaining structural differences.

Oracle: `pret/pokediamond` @ `master`, files `SND_util.c`, `SND_exChannel.c`, `SND_seq.c`.

The numeric DSP core and the sequence interpreter have now both been reconciled against pokediamond.
The golden render still matches sample-for-sample (the fixture song disables note-wait and uses only
small pitch bends, so the corrected interpreter produces identical output for it).

## Verified bit-exact / behaviourally-equivalent

| Component | Our code | pokediamond | Result |
|---|---|---|---|
| LFO sine table lookup | `tables::snd_sin_idx` | `SND_SinIdx` (`SND_util.c`) | **Exact** — same quadrant folding and `s8` sign-extension on the negative quadrants. |
| Velocity/ADSR → linear volume | `controller::calc_channel_volume` | `SND_CalcChannelVolume` (`SND_util.c`) | **Equivalent.** pokediamond returns the volume in the low byte and a hardware divider field in bits 8–9 (`div ∈ {0,1,2,3}`). DS hardware maps that field to ÷1, ÷2, ÷4, **÷16** (the GBATEK quirk where `3` is ÷16, not ÷8). Our `÷1/÷2/÷4/÷16` ladder bakes that in correctly. `SND_VOL_DB_MIN = -723`. |
| Decay/release coefficient | `bank::calc_decay_coeff` | `CalcDecayCoeff` (`SND_exChannel.c`) | **Exact** — `127→0xFFFF`, `126→0x3C00`, `<50→2v+1`, else `0x1E00/(126−v)`. |
| Attack coefficient | `bank::get_effective_attack` | `SND_SetExChannelAttack` | **Exact** — `<109 → 255−a`, else `sAttackCoeffTable[127−a]` (our `ATTACK_COEFF_TABLE`). |
| Sustain level | `bank::get_sustain_level` | `SND_SetExChannelSustain` + decay step | **Exact** — `DecibelSquareTable[sustain] << 7`. |
| ADSR state machine | `controller::apply_adsr` | `SND_UpdateExChannelEnvelope` | **Exact.** Attack `att = −((−att·k)>>8)`; decay subtracts and clamps to the sustain floor; release subtracts. Release-end at `att <= −92544` equals pokediamond's `(att>>7) <= −723` (`SND_VOL_DB_MIN`), since `−723<<7 = −92544`. |
| LFO value + phase | `controller::lfo_tick` | `SND_GetLfoValue` + `SND_UpdateLfo` | **Exact** after the fix below. Verified tick-for-tick against a transcribed reference over a parameter grid (`controller::tests::lfo_tick_matches_pokediamond_reference`). |

## Bug found and fixed

**Delayed LFO never engaged.** pokediamond uses a *single* `SNDLfo::delayCounter` to gate both the
LFO value (`SND_GetLfoValue`) and the phase advance (`SND_UpdateLfo`): the value is suppressed and
the phase frozen until the counter reaches `delay`, after which it engages. The legacy JS engine
(and the faithful Rust port) split this into two fields — `lfoDelayCounter`, used to gate the value,
and `delayCounter`, used to advance the phase — but **only `delayCounter` was ever incremented**.
`lfoDelayCounter` was stuck at `0`, so for any track with a non-zero LFO delay (opcode `0xE0`) the
gate `0 < delay` was always true and the LFO value was forced to `0` forever. Result: **delayed
vibrato / tremolo / auto-pan was silently disabled** (LFOs with zero delay were unaffected, which is
why basic modulation still worked).

Fix: gate the value on the same counter that is incremented, matching pokediamond's single
`delayCounter` (`crates/optime-core/src/controller.rs`, `lfo_tick`). The golden-parity test still
passes (the fixture song uses no delayed LFO), so this is a strict correctness gain over the JS
oracle. Covered by `controller::tests::delayed_lfo_engages_after_delay`.

## Sequence interpreter — reconciled

`Sequence::execute_track` was rewritten to follow `TrackStepTicks` and `TrackInit`. Each item below
was a divergence from pokediamond that is now fixed and covered by unit tests in `sequence.rs`:

- **Note timing / `noteWait`.** `flags.noteWait` (default **true**, per `TrackInit`) is now modelled:
  a note in note-wait mode advances the track clock by its duration (`resting_for = length`), and a
  zero-duration note sets `note_finish_wait`, stalling the track until its channels finish (the
  controller reports per-track channel activity into `Sequence::tick`). Tests:
  `note_wait_on_advances_by_note_duration`, `note_wait_off_fires_notes_back_to_back`.
- **Opcode `0xC7`** now sets `note_wait` (was: mono/poly).
- **Track-init defaults** corrected to `TrackInit`: `volume/expression = 127`, `priority = 64`,
  `bendRange = 2`, **`lfo_range = 1`** (was 0, which silently disabled any LFO that set only depth),
  `portamentoKey = 60`.
- **Pitch bend (`0xC4`)** is stored as a signed byte (`par._s8`) and used directly. The previous
  code sign-extended only the low 7 bits, halving the bend range and corrupting the sign for
  magnitudes ≥ 64.
- **Transpose (`0xC3`)** is applied to note keys (with `0..127` clamp). Test:
  `transpose_shifts_note_keys`.
- **Loops (`0xD4`/`0xFC`)** implemented over the shared call stack with per-frame counts (count 0 =
  infinite). Test: `loop_start_end_repeats_the_body`.
- **Variable / random / conditional family** implemented: the `0xA0` (random), `0xA1` (variable),
  `0xA2` (conditional) value prefixes; `0xB0`–`0xBD` set/arith/compare ops over 16 player variables;
  `SND_CalcRandom`'s LCG; and per-command operand encodings via `TrackParseValue` (`U8`/`U16`/`VLV`).
  Test: `conditional_prefix_gates_on_compare_flag`.
- **`0xC8` tie, `0xC9` portamento-key, `0xE3` sweep-pitch** now parse and store their operands.

### Intentional, documented design choice (not a bug)

- **Pan representation.** We keep an internal `0..128` pan (centre 64, with the `127 → 128`
  hard-right nudge) rather than pokediamond's signed `pan − 0x40`. The DS pan field feeds the
  hardware mixer's pan table; Optime's stereo stage is a deliberately different Haas-delay +
  bass-mono crossover design (see `synth.rs`), so the pan value is reparameterised for it. Both are
  centre-symmetric; this is a synthesis design difference, not an interpreter inaccuracy.

### Parsed but not synthesised

These commands are now decoded faithfully (operands consumed, PC kept aligned, track state stored),
but their *audio* effect is a synthesis feature the engine has never rendered (nor did the JS
original): portamento glide (`0xC9`/`0xCE`/`0xCF`), tie channel-reuse (`0xC8`), pitch sweep
(`0xE3`), and cross-track mute (`0xD7`). These are pitch/voice-management behaviours layered on top
of the (now-verified) note/ADSR/LFO core, not interpreter inaccuracies.

## How to re-run

- `cargo test -p optime-core` — includes the ADSR/coefficient unit tests, the new `lfo_tick`
  pokediamond-reference tests, and the golden-parity anchor.

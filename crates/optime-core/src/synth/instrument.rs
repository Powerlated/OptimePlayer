//! [`SampleInstrument`]: a single playing voice — pitch-shifted sample playback with the
//! resampling modes (nearest / linear / the two windowed-sinc gathers).

use std::sync::Arc;

use crate::devices::VoicePitch;
use crate::dsp::resample::{
    effective_gather, gather_sinc, sinc_fc, EffectiveGather, GatherSource, ResampleTables,
};
use crate::sample::{InstrumentResampleMode, Sample};
use crate::tuning::{midi_note_to_hz, TuningSystem};

/// Seconds a pop-smoothed PSG voice takes to slew across the full gain range. Short enough to
/// be inaudible as an envelope, long enough to turn the hard on/off steps into clicks-free ramps.
const POP_SLEW_SECONDS: f64 = 0.002;

/// A single playing voice.
#[derive(Clone)]
pub struct SampleInstrument {
    inv_sample_rate: f64,
    /// The sample this voice is playing.
    pub sample: Arc<Sample>,
    /// The voice's base pitch (a MIDI note relative to the sample's pitch, or an absolute
    /// data rate — see [`VoicePitch`]).
    pub pitch: VoicePitch,
    /// Current playback gain.
    pub volume: f64,
    /// Whether this voice is sounding.
    pub playing: bool,
    /// The gain actually applied last sample. Tracks [`Self::volume`] exactly, except for
    /// pop-smoothed PSG voices, where it slews toward it (see [`Self::advance`]'s `smooth_pops`).
    gain: f64,
    /// Set by [`Self::begin_fade_out`]: the voice slews to silence and then stops itself.
    fading_out: bool,
    /// Fractional sample position.
    pub sample_t: f64,
    /// Whether a looping voice has wrapped past the sample end at least once. Once it has, the
    /// signal under the gather window is fully periodic in the loop, so every tap may be read
    /// through the periodic mapping (before the first wrap, taps left of the loop end must still
    /// read the one-shot pre-loop data directly).
    pub(super) wrapped: bool,
    finetune: f64,
    finetune_lfo: f64,
    freq_ratio: f64,
    /// Last computed output sample.
    pub output: f64,
}

impl SampleInstrument {
    /// Creates an idle voice bound to `sample_rate` playing `sample`.
    pub fn new(sample_rate: f64, sample: Arc<Sample>) -> Self {
        let pitch = VoicePitch::Midi {
            note: 0.0,
            sample_pitch_hz: sample.frequency,
        };
        Self {
            inv_sample_rate: 1.0 / sample_rate,
            sample,
            pitch,
            volume: 1.0,
            playing: false,
            gain: 1.0,
            fading_out: false,
            sample_t: 0.0,
            wrapped: false,
            finetune: 0.0,
            finetune_lfo: 0.0,
            freq_ratio: 0.0,
            output: 0.0,
        }
    }

    /// Begins a note: sets the envelope volume and primes the applied gain. A pop-smoothed PSG
    /// voice starts from silence and slews up; everything else starts at `volume` exactly.
    pub fn begin_note(&mut self, volume: f64, smooth_pops: bool) {
        self.volume = volume;
        self.gain = if smooth_pops && self.sample.is_psg_square {
            0.0
        } else {
            volume
        };
        self.fading_out = false;
    }

    /// Starts a short fade to silence; the voice flips [`Self::playing`] off once it lands.
    pub fn begin_fade_out(&mut self) {
        self.fading_out = true;
    }

    /// Re-targets the voice to a new output sample rate. The per-sample step
    /// `r = freq_ratio * sample.sample_rate * inv_sample_rate` is recomputed from this on each
    /// [`Self::advance`], so nothing else needs to change.
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.inv_sample_rate = 1.0 / sample_rate;
    }

    /// Advances playback by one output sample, updating [`Self::output`].
    ///
    /// `mode` is the global resampling choice from [`SynthConfig`](crate::SynthConfig).  `tables`
    /// is required for the two sinc modes and may be `None` otherwise (falls back to
    /// nearest-neighbour).  With `smooth_pops` set, PSG voices slew their gain toward the
    /// envelope volume (and toward silence after [`Self::begin_fade_out`]) instead of stepping
    /// it, turning the abrupt hardware on/off transitions into click-free ramps.
    pub fn advance(
        &mut self,
        mode: InstrumentResampleMode,
        tables: Option<&ResampleTables>,
        smooth_pops: bool,
    ) {
        // r = source samples advanced per output sample (pitch-shifted playback speed).
        let r = self.freq_ratio * self.sample.sample_rate * self.inv_sample_rate;
        self.sample_t += r;

        let data = &self.sample.data;
        let looping = self.sample.looping;
        let loop_point = self.sample.loop_point;
        let data_len = data.len() as i64;
        let loop_len = data_len - loop_point;

        // Fold the read position back into the loop body once playback wraps.
        let fold = looping && loop_len > 0;
        let (folded, wrapped) = fold_pos(
            self.sample_t,
            fold,
            data_len as f64,
            loop_point as f64,
            loop_len as f64,
        );
        self.sample_t = folded;
        self.wrapped |= wrapped;
        let pos = self.sample_t;

        // Resolve this sample's applied gain: pop-smoothed PSG voices slew toward the target,
        // everything else applies the envelope volume exactly.
        let target = if self.fading_out { 0.0 } else { self.volume };
        self.gain = if smooth_pops && self.sample.is_psg_square {
            slew_toward(self.gain, target, self.inv_sample_rate / POP_SLEW_SECONDS)
        } else {
            target
        };
        if self.fading_out && self.gain == 0.0 {
            self.playing = false;
        }

        // A fully attenuated voice (release floor / silent track) contributes exactly 0 — skip
        // the gather. The position keeps advancing so re-opening the envelope stays seamless.
        if self.gain == 0.0 {
            self.output = 0.0;
            return;
        }

        let effective = effective_gather(mode, self.sample.is_psg_square);

        // Loop-aware sample accessor for the cheap 1–2 tap modes (and the no-tables fallback).
        let get = |mut t: i64| -> f64 {
            if t >= data_len && looping {
                if loop_len <= 0 {
                    return 0.0;
                }
                t = (t - loop_point).rem_euclid(loop_len) + loop_point;
            }
            if t >= 0 && t < data_len {
                f64::from(data[t as usize])
            } else {
                0.0
            }
        };

        let result = match effective {
            EffectiveGather::Nearest => get(pos.floor() as i64),
            EffectiveGather::Linear => {
                let i = pos.floor() as i64;
                let frac = pos - i as f64;
                let a = get(i);
                let b = get(i + 1);
                a + (b - a) * frac
            }
            EffectiveGather::Sinc {
                step_mode,
                cutoff_hz,
            } => {
                if let Some(tbl) = tables {
                    let fc = sinc_fc(r, self.inv_sample_rate, step_mode, cutoff_hz);
                    let src = GatherSource {
                        data,
                        looping,
                        loop_point,
                        loop_len,
                        wrapped: self.wrapped,
                    };
                    gather_sinc(&src, tbl, pos, fc, step_mode)
                } else {
                    // Tables not yet built; fall back to nearest.
                    get(pos.floor() as i64)
                }
            }
        };

        self.output = result * self.gain;
    }

    /// Advances playback by `out.len()` output samples, adding each `volume`-scaled sample into
    /// the corresponding `out` slot. Equivalent to calling [`Self::advance`] once per slot and
    /// summing [`Self::output`], but hoists the per-sample setup (playback speed, mode
    /// resolution, cutoff) out of the loop. Voice parameters (pitch, volume) are constant within
    /// a block — the controller only changes them on sequencer ticks, and blocks never span one.
    pub fn advance_block(
        &mut self,
        mode: InstrumentResampleMode,
        tables: Option<&ResampleTables>,
        smooth_pops: bool,
        out: &mut [f64],
    ) {
        // Only the sinc modes are worth hoisting; the 1–2 tap modes (and the missing-tables
        // fallback) just take the per-sample path.
        let effective = effective_gather(mode, self.sample.is_psg_square);
        let (
            EffectiveGather::Sinc {
                step_mode,
                cutoff_hz,
            },
            Some(tbl),
        ) = (effective, tables)
        else {
            for slot in out.iter_mut() {
                self.advance(mode, tables, smooth_pops);
                *slot += self.output;
            }
            return;
        };

        let r = self.freq_ratio * self.sample.sample_rate * self.inv_sample_rate;
        let data = &self.sample.data;
        let looping = self.sample.looping;
        let loop_point = self.sample.loop_point;
        let data_len = data.len() as i64;
        let data_len_f = data_len as f64;
        let loop_len = data_len - loop_point;
        let fold = looping && loop_len > 0;
        let (lp_f, loop_len_f) = (loop_point as f64, loop_len as f64);

        let mut pos = self.sample_t;
        let mut wrapped = self.wrapped;
        // The same per-sample gain resolution as `advance` (kept bit-identical): the target is
        // constant within a block, but a pop-smoothed gain still slews sample by sample.
        let smooth = smooth_pops && self.sample.is_psg_square;
        let slew = self.inv_sample_rate / POP_SLEW_SECONDS;
        let target = if self.fading_out { 0.0 } else { self.volume };
        let mut gain = self.gain;

        if target == 0.0 && (gain == 0.0 || !smooth) {
            // Fully attenuated for the whole block: advance the position (with identical
            // per-sample folding) only.
            for _ in 0..out.len() {
                pos += r;
                let (p, w) = fold_pos(pos, fold, data_len_f, lp_f, loop_len_f);
                pos = p;
                wrapped |= w;
            }
            if self.fading_out && !out.is_empty() {
                self.playing = false;
            }
            self.sample_t = pos;
            self.wrapped = wrapped;
            self.gain = 0.0;
            self.output = 0.0;
            return;
        }

        // See `sinc_fc` for the cutoff rationale (clean reconstruction vs BLEP output-Nyquist).
        let fc = sinc_fc(r, self.inv_sample_rate, step_mode, cutoff_hz);

        let mut last = 0.0;
        for slot in out.iter_mut() {
            pos += r;
            let (p, w) = fold_pos(pos, fold, data_len_f, lp_f, loop_len_f);
            pos = p;
            wrapped |= w;
            gain = if smooth {
                slew_toward(gain, target, slew)
            } else {
                target
            };
            if self.fading_out && gain == 0.0 {
                self.playing = false;
            }
            if gain == 0.0 {
                last = 0.0;
                continue;
            }
            let src = GatherSource {
                data,
                looping,
                loop_point,
                loop_len,
                wrapped,
            };
            let result = gather_sinc(&src, tbl, pos, fc, step_mode);
            last = result * gain;
            *slot += last;
        }
        self.sample_t = pos;
        self.wrapped = wrapped;
        self.gain = gain;
        self.output = last;
    }

    fn recompute_freq(&mut self, tuning: TuningSystem) {
        let tune = self.finetune_lfo + self.finetune;
        self.freq_ratio = match self.pitch {
            VoicePitch::Midi {
                note,
                sample_pitch_hz,
            } => midi_note_to_hz(note + tune, tuning) / sample_pitch_hz,
            // Absolute data rate: the ratio that makes the data step at `hz` samples/second,
            // with any (rare) detune applied as an equal-tempered factor.
            VoicePitch::DataRateHz(hz) => hz / self.sample.sample_rate * (tune / 12.0).exp2(),
        };
    }

    /// Sets the voice's base pitch.
    pub fn set_pitch(&mut self, pitch: VoicePitch, tuning: TuningSystem) {
        self.pitch = pitch;
        self.recompute_freq(tuning);
    }

    /// Sets the LFO pitch offset, in semitones.
    pub fn set_finetune_lfo(&mut self, semitones: f64, tuning: TuningSystem) {
        self.finetune_lfo = semitones;
        self.recompute_freq(tuning);
    }

    /// Sets the static finetune offset, in semitones.
    pub fn set_finetune(&mut self, semitones: f64, tuning: TuningSystem) {
        self.finetune = semitones;
        self.recompute_freq(tuning);
    }
}

/// Folds the read position back into the loop body once it passes the sample end, keeping the
/// position (and its fractional precision) bounded over arbitrarily long notes and letting the
/// sinc gather read taps without a per-tap loop-mapping division. Returns the (possibly folded)
/// position and whether it wrapped. `fold` must be `looping && loop_len > 0`; all lengths are in
/// source samples. Shared by [`SampleInstrument::advance`] and [`SampleInstrument::advance_block`]
/// so the two paths fold bit-identically.
#[inline]
fn fold_pos(pos: f64, fold: bool, data_len: f64, loop_point: f64, loop_len: f64) -> (f64, bool) {
    if fold && pos >= data_len {
        ((pos - loop_point) % loop_len + loop_point, true)
    } else {
        (pos, false)
    }
}

/// Moves `gain` toward `target` by at most `max_step`, landing exactly on the target.
fn slew_toward(gain: f64, target: f64, max_step: f64) -> f64 {
    let d = target - gain;
    if d.abs() <= max_step {
        target
    } else {
        gain + max_step.copysign(d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    // ── Resampling fixtures ───────────────────────────────────────────────────────────────
    //
    // The resampling tests exercise [`SampleInstrument::advance`] directly (rather than the
    // full synthesizer) so the pitch ratio `r` is controlled exactly:
    //
    //   r = freq_ratio · sample.sample_rate / out_rate,  freq_ratio = midi_note_to_hz(note) / sample_frequency
    //
    // By playing MIDI note 69 (440 Hz) with `sample_frequency = 440` we pin `freq_ratio = 1`, so
    //   r = sample.sample_rate / out_rate
    // is set purely by the two sample rates: `r < 1` upsamples (imaging), `r > 1` downsamples
    // (aliasing).

    /// The crunchy mode with both cutoff sliders parked at "off".
    fn crunch(half_taps: usize) -> InstrumentResampleMode {
        InstrumentResampleMode::SincOutputNyquist {
            half_taps,
            psg_cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
            sampler_cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
        }
    }

    /// A looping source sample holding a pure sine of `period` samples per cycle (frequency
    /// `1/period` cycles per source-sample), recorded at `src_rate`.
    fn sine_sample(period: usize, periods: usize, src_rate: f64) -> Arc<Sample> {
        let len = period * periods;
        let data: Vec<f32> = (0..len)
            .map(|k| (2.0 * PI * k as f64 / period as f64).sin() as f32)
            .collect();
        Arc::new(Sample::new(data, 440.0, src_rate, true, 0))
    }

    #[test]
    fn set_sample_rate_rescales_playback_step() {
        // A `DataRateHz` voice steps the source at `r = data_rate / out_rate` per output sample.
        // A long non-looping sample avoids any loop fold so the step is read straight off sample_t.
        let sample = Arc::new(Sample::new(vec![0.0; 4096], 440.0, 22_050.0, false, 0));
        let mut instr = SampleInstrument::new(44_100.0, sample);
        instr.set_pitch(VoicePitch::DataRateHz(22_050.0), TuningSystem::Equal);

        // 22050 / 44100 = 0.5 source samples per output sample.
        instr.advance(InstrumentResampleMode::NearestNeighbor, None, false);
        assert!(
            (instr.sample_t - 0.5).abs() < 1e-9,
            "got {}",
            instr.sample_t
        );

        // Halving the output rate doubles the step (now 1.0 per sample).
        instr.set_sample_rate(22_050.0);
        instr.advance(InstrumentResampleMode::NearestNeighbor, None, false);
        assert!(
            (instr.sample_t - 1.5).abs() < 1e-9,
            "got {}",
            instr.sample_t
        );
    }

    /// Renders `n` output samples of `sample` played at `out_rate` with `freq_ratio == 1`
    /// (so the playback speed `r = sample.sample_rate / out_rate`).
    fn render(
        out_rate: f64,
        sample: Arc<Sample>,
        mode: InstrumentResampleMode,
        tables: Option<&ResampleTables>,
        n: usize,
    ) -> Vec<f64> {
        let mut instr = SampleInstrument::new(out_rate, sample);
        // 440 Hz over a 440 Hz sample pitch ⇒ freq_ratio = 1
        instr.set_pitch(
            VoicePitch::Midi {
                note: 69.0,
                sample_pitch_hz: 440.0,
            },
            TuningSystem::Equal,
        );
        instr.begin_note(1.0, false);
        instr.playing = true;
        (0..n)
            .map(|_| {
                instr.advance(mode, tables, false);
                instr.output
            })
            .collect()
    }

    /// Single-bin DFT magnitude expressed as the peak amplitude of a sinusoid at `freq_hz`.
    /// Accurate when `freq_hz` lands on a DFT bin (an integer number of cycles over the window).
    fn amp_at(signal: &[f64], freq_hz: f64, rate: f64) -> f64 {
        let w = 2.0 * PI * freq_hz / rate;
        let (mut re, mut im) = (0.0, 0.0);
        for (n, &x) in signal.iter().enumerate() {
            let p = w * n as f64;
            re += x * p.cos();
            im -= x * p.sin();
        }
        2.0 * (re * re + im * im).sqrt() / signal.len() as f64
    }

    #[test]
    fn crunch_mode_preserves_imaging_clean_mode_removes_it() {
        // Upsample 4× (8192 → 32768): r = 0.25. A 512 Hz source sine produces zero-order-hold
        // images at n·8192 ± 512 Hz. The first image sits at 7680 Hz, above the source Nyquist
        // (4096 Hz) but below the output Nyquist (16384 Hz).
        let out_rate = 32768.0;
        let src_rate = 8192.0;
        let sample = sine_sample(16, 8, src_rate); // f0 = 8192/16 = 512 Hz
        let fund_hz = 512.0;
        let image_hz = 7680.0; // 8192 − 512
        let warmup = 256;
        let n = 2048; // 512, 7680 both land on bins (rate/n = 16 Hz)
        let tables = ResampleTables::new(16);

        // OutputNyquist ("crunch"): the BLEP step gather deliberately keeps the staircase ZOH
        // images that sit below output Nyquist (7680 < 16384) — the crunchy colour.
        let crunch = render(
            out_rate,
            sample.clone(),
            crunch(16),
            Some(&tables),
            warmup + n,
        );
        // SampleNyquist ("clean"): low-passes at the source Nyquist, removing the images.
        let clean = render(
            out_rate,
            sample,
            InstrumentResampleMode::SincSampleNyquist { half_taps: 16 },
            Some(&tables),
            warmup + n,
        );

        let crunch_fund = amp_at(&crunch[warmup..], fund_hz, out_rate);
        let crunch_image = amp_at(&crunch[warmup..], image_hz, out_rate);
        let clean_fund = amp_at(&clean[warmup..], fund_hz, out_rate);
        let clean_image = amp_at(&clean[warmup..], image_hz, out_rate);

        // Both modes pass the fundamental essentially untouched.
        assert!(crunch_fund > 0.9, "crunch fundamental = {crunch_fund}");
        assert!(clean_fund > 0.9, "clean fundamental = {clean_fund}");

        // The defining contrast: imaging is clearly present in crunch mode and gone in clean mode.
        // (ZOH theory predicts the 7680 Hz image at ≈ sinc(7680/8192) ≈ 0.066 of the input.)
        assert!(
            crunch_image > 0.03,
            "crunch should retain imaging, image amp = {crunch_image}"
        );
        assert!(
            clean_image < 0.005,
            "clean should suppress imaging, image amp = {clean_image}"
        );
        assert!(
            crunch_image > 5.0 * clean_image,
            "crunch imaging ({crunch_image}) should dwarf clean imaging ({clean_image})"
        );
    }

    #[test]
    fn upsampled_crunch_bandlimits_stairstep_edges() {
        // The core property: on an *upsampled* voice the crunch mode must band-limit the ZOH
        // stairstep edges to the output Nyquist, rather than point-sampling a hard staircase
        // (plain nearest-neighbour), which jitters/aliases at non-integer ratios.
        //
        // Upsample 2.5× (8192 → 20480): r = 0.4 (a non-integer 1/r, so nearest-neighbour edge
        // timing is quantized → audible aliasing). A 512 Hz source sine has ZOH images at
        // n·8192 ± 512 Hz. Output Nyquist is 10240 Hz:
        //   • the first image (7680 Hz) is *below* it → kept as crunch colour;
        //   • the second image (15872 Hz) is *above* it → nearest folds it down to an alias at
        //     20480 − 15872 = 4608 Hz; band-limited stairsteps must remove it before it folds.
        let out_rate = 20480.0;
        let src_rate = 8192.0; // r = 0.4
        let sample = sine_sample(16, 8, src_rate);
        let image_hz = 7680.0; // 8192 − 512 (below output Nyquist → crunch image)
        let alias_hz = 4608.0; // fold of the 15872 Hz image (above output Nyquist)
        let warmup = 256;
        let n = 2560; // 7680, 4608 both land on bins (rate/n = 8 Hz)
        let tables = ResampleTables::new(16);

        let nearest = render(
            out_rate,
            sample.clone(),
            InstrumentResampleMode::NearestNeighbor,
            None,
            warmup + n,
        );
        let crunch = render(out_rate, sample, crunch(16), Some(&tables), warmup + n);

        let nearest_image = amp_at(&nearest[warmup..], image_hz, out_rate);
        let nearest_alias = amp_at(&nearest[warmup..], alias_hz, out_rate);
        let crunch_image = amp_at(&crunch[warmup..], image_hz, out_rate);
        let crunch_alias = amp_at(&crunch[warmup..], alias_hz, out_rate);

        // The in-band crunch image is preserved (band-limiting only touches energy above output
        // Nyquist), so crunch keeps essentially the same colour as the raw staircase.
        assert!(
            crunch_image > 0.03,
            "crunch should keep the in-band ZOH image, got {crunch_image}"
        );
        assert!(
            crunch_image > 0.7 * nearest_image,
            "crunch image ({crunch_image}) should match the raw staircase ({nearest_image})"
        );

        // The defining win: nearest-neighbour point-samples a hard staircase, folding the
        // above-Nyquist image into an audible alias; band-limiting the stairstep edges cuts that
        // fold sharply. (It cannot vanish entirely — keeping the near-Nyquist crunch images is the
        // whole point of the mode — but it must drop well below the nearest-neighbour alias and
        // stay far beneath the legitimate in-band image.)
        assert!(
            nearest_alias > 0.02,
            "sanity: nearest-neighbour should alias here, got {nearest_alias}"
        );
        assert!(
            crunch_alias < 0.5 * nearest_alias,
            "band-limited stairsteps should suppress the alias: crunch={crunch_alias}, \
             nearest={nearest_alias}"
        );
        assert!(
            crunch_alias < 0.25 * crunch_image,
            "alias ({crunch_alias}) should sit well below the kept crunch image ({crunch_image})"
        );
    }

    #[test]
    fn psg_square_uses_blep_step_under_sample_nyquist() {
        // PSG square waves are special-cased to the BLEP step gather even in clean
        // (SampleNyquist) mode, preserving their hard edges band-limited at the output Nyquist.
        // Verify the effective-mode override fires: clean mode on a PSG voice must match the
        // crunch mode with the cutoff sliders parked at "off".
        let out_rate = 32768.0;
        let mut sample = Sample::new(vec![1.0, 1.0, -1.0, -1.0], 440.0, 16384.0, true, 0);
        sample.is_psg_square = true;
        let sample = Arc::new(sample);
        let tables = ResampleTables::new(16);
        let n = 512;
        let crunch_mode = render(out_rate, sample.clone(), crunch(16), Some(&tables), n);
        let clean_mode = render(
            out_rate,
            sample,
            InstrumentResampleMode::SincSampleNyquist { half_taps: 16 },
            Some(&tables),
            n,
        );
        for (i, (a, b)) in crunch_mode.iter().zip(&clean_mode).enumerate() {
            assert!((a - b).abs() < 1e-12, "sample {i}: crunch={a}, clean={b}");
        }
    }

    #[test]
    fn crunchy_cutoff_attenuates_highs_per_voice_kind() {
        // The crunchy-mode cutoff sliders must low-pass the matching voice kind only: a
        // 4096 Hz tone passes untouched with the cutoff above it and is strongly attenuated
        // with the cutoff far below it, independently for PSG and sampler voices.
        let out_rate = 32768.0;
        let src_rate = 16384.0;
        let tone_hz = 4096.0;
        let warmup = 256;
        let n = 2048;
        let tables = ResampleTables::new(32);
        for is_psg in [false, true] {
            let mut s = sine_sample(4, 64, src_rate); // 16384/4 = 4096 Hz
            if is_psg {
                Arc::get_mut(&mut s).unwrap().is_psg_square = true;
            }
            let mode_with = |this_kind_hz: u32| {
                let other = InstrumentResampleMode::CUTOFF_OFF_HZ;
                InstrumentResampleMode::SincOutputNyquist {
                    half_taps: 32,
                    psg_cutoff_hz: if is_psg { this_kind_hz } else { other },
                    sampler_cutoff_hz: if is_psg { other } else { this_kind_hz },
                }
            };
            let open = render(
                out_rate,
                s.clone(),
                mode_with(20_000),
                Some(&tables),
                warmup + n,
            );
            let cut = render(
                out_rate,
                s.clone(),
                mode_with(1_000),
                Some(&tables),
                warmup + n,
            );
            // The *other* kind's slider must not touch this voice.
            let other_cut = {
                let other = InstrumentResampleMode::SincOutputNyquist {
                    half_taps: 32,
                    psg_cutoff_hz: if is_psg {
                        InstrumentResampleMode::CUTOFF_OFF_HZ
                    } else {
                        1_000
                    },
                    sampler_cutoff_hz: if is_psg {
                        1_000
                    } else {
                        InstrumentResampleMode::CUTOFF_OFF_HZ
                    },
                };
                render(out_rate, s, other, Some(&tables), warmup + n)
            };
            let a_open = amp_at(&open[warmup..], tone_hz, out_rate);
            let a_cut = amp_at(&cut[warmup..], tone_hz, out_rate);
            let a_other = amp_at(&other_cut[warmup..], tone_hz, out_rate);
            // (The step gather keeps the ZOH sinc rolloff: ≈0.90 at a quarter of the source
            // rate — that colouring is the point of crunch mode.)
            assert!(
                a_open > 0.85,
                "is_psg={is_psg}: open cutoff passes tone, got {a_open}"
            );
            assert!(
                a_cut < 0.05 * a_open,
                "is_psg={is_psg}: 1 kHz cutoff should crush a 4096 Hz tone: {a_cut} vs {a_open}"
            );
            assert!(
                (a_other - a_open).abs() < 0.05 * a_open,
                "is_psg={is_psg}: the other kind's slider must not apply: {a_other} vs {a_open}"
            );
        }
    }

    #[test]
    fn pop_smoothing_slews_psg_gain() {
        // A pop-smoothed PSG voice must ramp from silence instead of stepping on, and a
        // fade-out must land at exactly zero and stop the voice. With smoothing off the gain
        // steps instantly (the preserved hardware pop).
        let out_rate = 48_000.0;
        let mut sample = Sample::new(vec![1.0; 64], 440.0, out_rate, true, 0); // DC source
        sample.is_psg_square = true;
        let sample = Arc::new(sample);
        let pitch = VoicePitch::Midi {
            note: 69.0,
            sample_pitch_hz: 440.0,
        };

        let mut instr = SampleInstrument::new(out_rate, sample.clone());
        instr.set_pitch(pitch, TuningSystem::Equal);
        instr.playing = true;
        instr.begin_note(1.0, true);
        instr.advance(InstrumentResampleMode::NearestNeighbor, None, true);
        assert!(
            instr.output < 0.05,
            "smoothed start must ramp from silence, got {}",
            instr.output
        );
        let mut prev = instr.output;
        let mut reached = false;
        for _ in 0..1024 {
            instr.advance(InstrumentResampleMode::NearestNeighbor, None, true);
            // DC source ⇒ output equals the applied gain.
            let delta = instr.output - prev;
            assert!((-1e-12..0.02).contains(&delta), "ramp step {delta}");
            prev = instr.output;
            if (instr.output - 1.0).abs() < 1e-12 {
                reached = true;
                break;
            }
        }
        assert!(reached, "gain must land exactly on the target");

        instr.begin_fade_out();
        for _ in 0..1024 {
            if !instr.playing {
                break;
            }
            instr.advance(InstrumentResampleMode::NearestNeighbor, None, true);
        }
        assert!(!instr.playing, "fade-out must stop the voice");
        assert_eq!(instr.output, 0.0);

        let mut hard = SampleInstrument::new(out_rate, sample);
        hard.set_pitch(pitch, TuningSystem::Equal);
        hard.playing = true;
        hard.begin_note(1.0, false);
        hard.advance(InstrumentResampleMode::NearestNeighbor, None, false);
        assert!(
            (hard.output - 1.0).abs() < 1e-12,
            "unsmoothed start must step to full gain, got {}",
            hard.output
        );
    }

    /// Renders a stopband source sine downsampled by 2.5× through the clean (SampleNyquist)
    /// low-pass and returns the residual (aliased) amplitude that survives in the output band
    /// for a sinc kernel of `half_taps` zero-crossings.
    fn alias_residual(half_taps: usize) -> f64 {
        // Downsample 2.5× (20480 → 8192): r = 2.5, anti-alias cutoff fc = 0.5/r = 0.2
        // cycles/source-sample. A period-4 source sine sits at f0 = 0.25 (1.25× fc, in the
        // stopband). Its energy folds to |0.25·2.5 − 1|·8192 = 3072 Hz in the output.
        let out_rate = 8192.0;
        let src_rate = 20480.0; // r = 2.5
        let alias_hz = 3072.0;
        let sample = sine_sample(4, 64, src_rate);
        let tables = ResampleTables::new(half_taps);
        let warmup = 256;
        let n = 2048; // 3072 Hz lands on a bin (rate/n = 4 Hz)
        let out = render(
            out_rate,
            sample,
            InstrumentResampleMode::SincSampleNyquist { half_taps },
            Some(&tables),
            warmup + n,
        );
        amp_at(&out[warmup..], alias_hz, out_rate)
    }

    #[test]
    fn alias_suppression_increases_with_kernel_size() {
        // A bigger kernel ⇒ a sharper anti-alias low-pass ⇒ a stopband tone is pushed deeper
        // below the noise floor. The residual must shrink monotonically as `half_taps` grows.
        //
        // This is a property of the clean (SampleNyquist) anti-alias low-pass specifically.
        // The crunch (OutputNyquist) mode is *not* tested here: its BLEP step-difference gather
        // deliberately retains the ZOH staircase colour, so a stopband tone does not fall
        // monotonically with tap count — that intentional grit is the point of the mode.
        let taps = [2usize, 4, 8, 16, 32];
        let residuals: Vec<f64> = taps.iter().map(|&t| alias_residual(t)).collect();
        for w in residuals.windows(2) {
            assert!(
                w[1] < w[0],
                "alias residual should fall with more taps, got {residuals:?} for {taps:?}"
            );
        }
        // The improvement should be substantial across the range, not a marginal wobble.
        assert!(
            residuals[0] > residuals[residuals.len() - 1] * 3.0,
            "expected a large suppression gain across the tap range, got {residuals:?}"
        );
    }
}

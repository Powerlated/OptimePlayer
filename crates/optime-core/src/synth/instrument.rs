//! [`SampleInstrument`]: a single playing voice — pitch-shifted sample playback with the
//! resampling modes (nearest / linear / the two windowed-sinc gathers).

use std::sync::Arc;

use super::gather::{gather_sinc, GatherSource};
use crate::devices::VoicePitch;
use crate::resample::ResampleTables;
use crate::sample::{ResampleMode, Sample};
use crate::tuning::{midi_note_to_hz, TuningSystem};

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
            sample_t: 0.0,
            wrapped: false,
            finetune: 0.0,
            finetune_lfo: 0.0,
            freq_ratio: 0.0,
            output: 0.0,
        }
    }

    /// Advances playback by one output sample, updating [`Self::output`].
    ///
    /// `mode` is the global resampling choice from [`SynthConfig`](crate::SynthConfig).  `tables`
    /// is required for the two sinc modes and may be `None` otherwise (falls back to
    /// nearest-neighbour).
    pub fn advance(&mut self, mode: ResampleMode, tables: Option<&ResampleTables>) {
        // r = source samples advanced per output sample (pitch-shifted playback speed).
        let r = self.freq_ratio * self.sample.sample_rate * self.inv_sample_rate;
        self.sample_t += r;

        let data = &self.sample.data;
        let looping = self.sample.looping;
        let loop_point = self.sample.loop_point;
        let data_len = data.len() as i64;
        let loop_len = data_len - loop_point;

        // Fold the read position back into the loop body once playback wraps. This keeps the
        // position (and its fractional precision) bounded over arbitrarily long notes, and lets
        // the sinc gather below read source taps without a per-tap loop-mapping division.
        if looping && loop_len > 0 && self.sample_t >= data_len as f64 {
            let lp = loop_point as f64;
            self.sample_t = (self.sample_t - lp) % loop_len as f64 + lp;
            self.wrapped = true;
        }
        let pos = self.sample_t;

        // A fully attenuated voice (release floor / silent track) contributes exactly 0 — skip
        // the gather. The position keeps advancing so re-opening the envelope stays seamless.
        if self.volume == 0.0 {
            self.output = 0.0;
            return;
        }

        let is_psg = self.sample.is_psg_square;

        // Resolve the effective mode: PSG squares under SampleNyquist → nearest (per design).
        // OutputNyquist stays in BLEP step-mode at every ratio — including upsampling — so the
        // ZOH stairstep edges are band-limited to the output Nyquist instead of being
        // point-sampled as hard discontinuities (which jitter/alias at non-integer ratios).
        let effective = match mode {
            ResampleMode::SincSampleNyquist { .. } if is_psg => ResampleMode::NearestNeighbor,
            other => other,
        };

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
            ResampleMode::NearestNeighbor => get(pos.floor() as i64),
            ResampleMode::Linear => {
                let i = pos.floor() as i64;
                let frac = pos - i as f64;
                let a = get(i);
                let b = get(i + 1);
                a + (b - a) * frac
            }
            ResampleMode::SincSampleNyquist { .. } | ResampleMode::SincOutputNyquist { .. } => {
                if let Some(tbl) = tables {
                    let step_mode = matches!(effective, ResampleMode::SincOutputNyquist { .. });
                    // fc in cycles/source-sample (source Nyquist = 0.5).
                    // SampleNyquist (impulse): clean reconstruction low-pass at min(0.5, 0.5/r)
                    //   — removes ZOH images when upsampling, anti-aliases when downsampling.
                    // OutputNyquist (BLEP step): fc = 0.5/r, the *output* Nyquist, so the stairstep
                    //   edges are band-limited to the output rate. For r > 1 (downsampling) this is
                    //   < 0.5 and anti-aliases; for r ≤ 1 (upsampling) it is ≥ 0.5, keeping the
                    //   crunch images that sit below output Nyquist while still band-limiting the
                    //   hard edges (no nearest-neighbour jitter).
                    let fc = if step_mode || r > 1.0 { 0.5 / r } else { 0.5 };
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

        self.output = result * self.volume;
    }

    /// Advances playback by `out.len()` output samples, adding each `volume`-scaled sample into
    /// the corresponding `out` slot. Equivalent to calling [`Self::advance`] once per slot and
    /// summing [`Self::output`], but hoists the per-sample setup (playback speed, mode
    /// resolution, cutoff) out of the loop. Voice parameters (pitch, volume) are constant within
    /// a block — the controller only changes them on sequencer ticks, and blocks never span one.
    pub fn advance_block(
        &mut self,
        mode: ResampleMode,
        tables: Option<&ResampleTables>,
        out: &mut [f64],
    ) {
        // Only the sinc modes are worth hoisting; the 1–2 tap modes (and the PSG-square nearest
        // override / missing-tables fallback) just take the per-sample path.
        let hoisted_sinc = tables.is_some()
            && match mode {
                ResampleMode::SincSampleNyquist { .. } => !self.sample.is_psg_square,
                ResampleMode::SincOutputNyquist { .. } => true,
                _ => false,
            };
        if !hoisted_sinc {
            for slot in out.iter_mut() {
                self.advance(mode, tables);
                *slot += self.output;
            }
            return;
        }

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
        let vol = self.volume;

        if vol == 0.0 {
            // Fully attenuated: advance the position (with identical per-sample folding) only.
            for _ in 0..out.len() {
                pos += r;
                if fold && pos >= data_len_f {
                    pos = (pos - lp_f) % loop_len_f + lp_f;
                    wrapped = true;
                }
            }
            self.sample_t = pos;
            self.wrapped = wrapped;
            self.output = 0.0;
            return;
        }

        let tbl = tables.expect("hoisted_sinc implies tables");
        let step_mode = matches!(mode, ResampleMode::SincOutputNyquist { .. });
        // See `advance` for the cutoff rationale (clean reconstruction vs BLEP output-Nyquist).
        let fc = if step_mode || r > 1.0 { 0.5 / r } else { 0.5 };

        let mut last = 0.0;
        for slot in out.iter_mut() {
            pos += r;
            if fold && pos >= data_len_f {
                pos = (pos - lp_f) % loop_len_f + lp_f;
                wrapped = true;
            }
            let src = GatherSource {
                data,
                looping,
                loop_point,
                loop_len,
                wrapped,
            };
            let result = gather_sinc(&src, tbl, pos, fc, step_mode);
            last = result * vol;
            *slot += last;
        }
        self.sample_t = pos;
        self.wrapped = wrapped;
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

    /// A looping source sample holding a pure sine of `period` samples per cycle (frequency
    /// `1/period` cycles per source-sample), recorded at `src_rate`.
    fn sine_sample(period: usize, periods: usize, src_rate: f64) -> Arc<Sample> {
        let len = period * periods;
        let data: Vec<f32> = (0..len)
            .map(|k| (2.0 * PI * k as f64 / period as f64).sin() as f32)
            .collect();
        Arc::new(Sample::new(data, 440.0, src_rate, true, 0))
    }

    /// Renders `n` output samples of `sample` played at `out_rate` with `freq_ratio == 1`
    /// (so the playback speed `r = sample.sample_rate / out_rate`).
    fn render(
        out_rate: f64,
        sample: Arc<Sample>,
        mode: ResampleMode,
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
        instr.volume = 1.0;
        instr.playing = true;
        (0..n)
            .map(|_| {
                instr.advance(mode, tables);
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
            ResampleMode::SincOutputNyquist { half_taps: 16 },
            Some(&tables),
            warmup + n,
        );
        // SampleNyquist ("clean"): low-passes at the source Nyquist, removing the images.
        let clean = render(
            out_rate,
            sample,
            ResampleMode::SincSampleNyquist { half_taps: 16 },
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
            ResampleMode::NearestNeighbor,
            None,
            warmup + n,
        );
        let crunch = render(
            out_rate,
            sample,
            ResampleMode::SincOutputNyquist { half_taps: 16 },
            Some(&tables),
            warmup + n,
        );

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
    fn psg_square_uses_nearest_under_sample_nyquist() {
        // PSG square waves are special-cased to nearest-neighbour even in clean (SampleNyquist)
        // mode, preserving their hard edges. Verify the effective-mode override fires.
        let out_rate = 32768.0;
        let mut sample = Sample::new(vec![1.0, 1.0, -1.0, -1.0], 440.0, 16384.0, true, 0);
        sample.is_psg_square = true;
        let sample = Arc::new(sample);
        let tables = ResampleTables::new(16);
        let n = 512;
        let nearest = render(
            out_rate,
            sample.clone(),
            ResampleMode::NearestNeighbor,
            None,
            n,
        );
        let sinc = render(
            out_rate,
            sample,
            ResampleMode::SincSampleNyquist { half_taps: 16 },
            Some(&tables),
            n,
        );
        for (i, (a, b)) in nearest.iter().zip(&sinc).enumerate() {
            assert!((a - b).abs() < 1e-12, "sample {i}: nearest={a}, sinc={b}");
        }
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
            ResampleMode::SincSampleNyquist { half_taps },
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

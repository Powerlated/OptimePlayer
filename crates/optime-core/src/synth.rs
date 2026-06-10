//! Sample playback: per-voice [`SampleInstrument`]s, a polyphonic [`SampleSynthesizer`] per
//! track, and the stereo-separation [`DelayLine`].

use std::sync::Arc;

use crate::controller::SynthConfig;
use crate::dsp::BiquadFilter;
use crate::resample::{resample_sinc, tap_window, ResampleTables, MAX_HALF_TAPS};
use crate::sample::{ResampleMode, Sample};
use crate::tuning::{midi_note_to_hz, TuningSystem};

/// Q for the bass-mono crossover low-pass (Butterworth). `pub` so the app can reconstruct the
/// filters for the analysis popup without duplicating the constant.
pub const CROSSOVER_Q: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// A fixed-length delay line used to widen the stereo image (Haas effect).
#[derive(Debug, Clone)]
pub struct DelayLine {
    buffer: Vec<f64>,
    pos_out: usize,
    delay: usize,
    /// Output gain.
    pub gain: f64,
}

impl DelayLine {
    /// Creates a delay line able to hold up to `max_length` samples.
    pub fn new(max_length: usize) -> Self {
        Self {
            buffer: vec![0.0; max_length.max(1)],
            pos_out: 0,
            delay: 0,
            gain: 1.0,
        }
    }

    /// Pushes `val` and returns the delayed (and gain-scaled) output sample.
    pub fn process(&mut self, val: f64) -> f64 {
        let len = self.buffer.len();
        self.buffer[(self.pos_out + self.delay) % len] = val;
        let out_val = self.buffer[self.pos_out];
        self.pos_out += 1;
        if self.pos_out >= len {
            self.pos_out = 0;
        }
        out_val * self.gain
    }

    /// Sets the delay length in samples (clamped to the buffer capacity).
    pub fn set_delay(&mut self, length: usize) {
        self.delay = length.min(self.buffer.len());
    }
}

/// A single playing voice.
#[derive(Clone)]
pub struct SampleInstrument {
    inv_sample_rate: f64,
    /// The sample this voice is playing.
    pub sample: Arc<Sample>,
    /// The pitch (Hz) the current sample represents (may differ from `sample.frequency`).
    pub sample_frequency: f64,
    /// Current playback gain.
    pub volume: f64,
    /// Whether this voice is sounding.
    pub playing: bool,
    /// The tick the note started (used to detect voice reuse).
    pub start_time: u32,
    /// MIDI note (may be fractional once finetune is applied).
    pub midi_note: f64,
    /// Fractional sample position.
    pub sample_t: f64,
    /// Whether a looping voice has wrapped past the sample end at least once. Once it has, the
    /// signal under the gather window is fully periodic in the loop, so every tap may be read
    /// through the periodic mapping (before the first wrap, taps left of the loop end must still
    /// read the one-shot pre-loop data directly).
    wrapped: bool,
    finetune: f64,
    finetune_lfo: f64,
    freq_ratio: f64,
    /// Last computed output sample.
    pub output: f64,
}

impl SampleInstrument {
    /// Creates an idle voice bound to `sample_rate` playing `sample`.
    pub fn new(sample_rate: f64, sample: Arc<Sample>) -> Self {
        let sample_frequency = sample.frequency;
        Self {
            inv_sample_rate: 1.0 / sample_rate,
            sample,
            sample_frequency,
            volume: 1.0,
            playing: false,
            start_time: 0,
            midi_note: 0.0,
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
    /// `mode` is the global resampling choice from [`SynthConfig`].  `tables` is required for
    /// the two sinc modes and may be `None` otherwise (falls back to nearest-neighbour).
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

                    // Pre-stage the exact tap window the gather will read so its accessor is a
                    // branch-free slice index instead of a per-tap loop-mapping with a division
                    // (the gather is the synth's hot loop: ~2P taps per voice per output sample).
                    let (k_lo, k_hi) = tap_window(tbl, pos);
                    let periodic = looping && self.wrapped && loop_len > 0;
                    if !periodic && k_lo >= 0 && k_hi < data_len {
                        // Fast path: the whole window is in-bounds one-shot data.
                        let src = &data[k_lo as usize..=k_hi as usize];
                        resample_sinc(
                            tbl,
                            |t| f64::from(src[(t - k_lo) as usize]),
                            pos,
                            fc,
                            step_mode,
                        )
                    } else {
                        // Edge path: stage the window into a stack buffer so the gather still
                        // reads a plain slice.
                        let n = (k_hi - k_lo + 1) as usize;
                        let mut buf = [0.0f32; 2 * MAX_HALF_TAPS + 2];
                        if periodic {
                            // The voice has wrapped: the signal is periodic in the loop, so
                            // every tap maps into the loop body. One division to place the
                            // first tap, then an increment-and-wrap walk.
                            let mut idx = (k_lo - loop_point).rem_euclid(loop_len) + loop_point;
                            for slot in buf[..n].iter_mut() {
                                *slot = data[idx as usize];
                                idx += 1;
                                if idx == data_len {
                                    idx = loop_point;
                                }
                            }
                        } else {
                            // Window crosses the sample start/end before any wrap: zeros
                            // outside, direct reads inside, and (for looping voices) the right
                            // tail peeks into the first loop pass.
                            for (j, slot) in buf[..n].iter_mut().enumerate() {
                                let t = k_lo + j as i64;
                                *slot = if t < 0 {
                                    0.0
                                } else if t < data_len {
                                    data[t as usize]
                                } else if looping && loop_len > 0 {
                                    data[((t - loop_point).rem_euclid(loop_len) + loop_point)
                                        as usize]
                                } else {
                                    0.0
                                };
                            }
                        }
                        let src = &buf[..n];
                        resample_sinc(
                            tbl,
                            |t| f64::from(src[(t - k_lo) as usize]),
                            pos,
                            fc,
                            step_mode,
                        )
                    }
                } else {
                    // Tables not yet built; fall back to nearest.
                    get(pos.floor() as i64)
                }
            }
        };

        self.output = result * self.volume;
    }

    fn recompute_freq(&mut self, tuning: TuningSystem) {
        self.freq_ratio =
            midi_note_to_hz(self.midi_note + self.finetune_lfo + self.finetune, tuning)
                / self.sample_frequency;
    }

    /// Sets the base MIDI note.
    pub fn set_note(&mut self, midi_note: f64, tuning: TuningSystem) {
        self.midi_note = midi_note;
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

/// A polyphonic synthesizer for one sequence track. Holds a fixed pool of voices and mixes the
/// active ones into a stereo sample, optionally widened by per-channel delay lines.
pub struct SampleSynthesizer {
    sample_rate: f64,
    instrs: Vec<SampleInstrument>,
    active_instrs: Vec<usize>,
    playing_index: usize,
    /// Last mixed left output.
    pub val_l: f64,
    /// Last mixed right output.
    pub val_r: f64,
    /// Track volume (0..1).
    pub volume: f64,
    pan: f64,
    delay_line_l: DelayLine,
    delay_line_r: DelayLine,
    /// Filters that form a Fourth-order Linkwitz-Riley crossover filter
    crossover_lp: BiquadFilter,
    crossover_hp: BiquadFilter,
    /// The cutoff the crossover is currently configured for (so we only recompute on change).
    crossover_freq: f64,
    finetune: f64,
    /// Cached sinc tables; rebuilt whenever `half_taps` changes.
    resample_tables: Option<ResampleTables>,
    /// The `half_taps` value the cached tables were built for.
    resample_half_taps: usize,
}

impl SampleSynthesizer {
    /// Creates a synthesizer with `instrs_available` voices at `sample_rate`.
    pub fn new(sample_rate: f64, instrs_available: usize) -> Self {
        let empty = Arc::new(Sample::new(vec![0.0], 440.0, sample_rate, false, 0));
        let instrs = (0..instrs_available)
            .map(|_| SampleInstrument::new(sample_rate, empty.clone()))
            .collect();
        let delay_len = (sample_rate * 0.1).round() as usize;
        let crossover_freq = 200.0;
        Self {
            sample_rate,
            instrs,
            active_instrs: Vec::new(),
            playing_index: 0,
            val_l: 0.0,
            val_r: 0.0,
            volume: 1.0,
            pan: 0.5,
            delay_line_l: DelayLine::new(delay_len),
            delay_line_r: DelayLine::new(delay_len),
            crossover_lp: BiquadFilter::low_pass(4, sample_rate, crossover_freq, CROSSOVER_Q),
            crossover_hp: BiquadFilter::high_pass(4, sample_rate, crossover_freq, CROSSOVER_Q),
            crossover_freq,
            finetune: 0.0,
            resample_tables: None,
            resample_half_taps: 0,
        }
    }

    /// Number of voices in the pool.
    #[inline]
    pub fn voice_count(&self) -> usize {
        self.instrs.len()
    }

    /// Immutable access to a voice.
    #[inline]
    pub fn instr(&self, index: usize) -> &SampleInstrument {
        &self.instrs[index]
    }

    /// Mutable access to a voice (used by the controller for ADSR/LFO updates).
    #[inline]
    pub fn instr_mut(&mut self, index: usize) -> &mut SampleInstrument {
        &mut self.instrs[index]
    }

    /// Starts `sample` on the next round-robin voice and returns its index.
    pub fn play(
        &mut self,
        sample: Arc<Sample>,
        midi_note: f64,
        sample_frequency: f64,
        volume: f64,
        meta: u32,
        tuning: TuningSystem,
    ) -> usize {
        let index = self.playing_index;
        if self.instrs[index].playing {
            self.cut_instrument(index);
        }

        {
            let instr = &mut self.instrs[index];
            instr.sample = sample;
            instr.sample_frequency = sample_frequency;
            instr.set_note(midi_note, tuning);
            instr.set_finetune_lfo(0.0, tuning);
            instr.set_finetune(self.finetune, tuning);
            instr.volume = volume;
            instr.start_time = meta;
            instr.sample_t = 0.0;
            instr.wrapped = false;
            instr.playing = true;
        }

        self.playing_index = (self.playing_index + 1) % self.instrs.len();
        self.active_instrs.push(index);
        index
    }

    /// Stops the voice at `index` if it is active.
    pub fn cut_instrument(&mut self, index: usize) {
        if let Some(pos) = self.active_instrs.iter().position(|&i| i == index) {
            self.instrs[index].playing = false;
            self.active_instrs.remove(pos);
        }
    }

    /// Advances all active voices by one sample and mixes them into `val_l`/`val_r`.
    pub fn next_sample(&mut self, config: &SynthConfig) {
        // Rebuild the sinc tables if the mode switched to a sinc variant or the tap count changed.
        let needed_taps = match config.resample {
            ResampleMode::SincSampleNyquist { half_taps }
            | ResampleMode::SincOutputNyquist { half_taps } => Some(half_taps),
            _ => None,
        };
        if let Some(ht) = needed_taps {
            if self.resample_tables.is_none() || self.resample_half_taps != ht {
                self.resample_tables = Some(ResampleTables::new(ht));
                self.resample_half_taps = ht;
            }
        }
        let tables = self.resample_tables.as_ref();

        let mut mono = 0.0;
        for &i in &self.active_instrs {
            self.instrs[i].advance(config.resample, tables);
            mono += self.instrs[i].output;
        }

        if !config.stereo_separation {
            self.val_l = mono * (1.0 - self.pan) * self.volume;
            self.val_r = mono * self.pan * self.volume;
            return;
        }

        if config.bass_mono {
            // Split into a centered low band and a widened high band via crossover.
            self.ensure_crossover(config.bass_mono_freq);
            let lo = self.crossover_lp.transform(mono);
            let hi = self.crossover_hp.transform(mono);
            let center = lo * 0.5;
            let hi_l = self.delay_line_l.process(hi * (1.0 - self.pan));
            let hi_r = self.delay_line_r.process(hi * self.pan);
            self.val_l = (hi_l + center) * self.volume;
            self.val_r = (hi_r + center) * self.volume;
        } else {
            self.val_l = self.delay_line_l.process(mono * (1.0 - self.pan)) * self.volume;
            self.val_r = self.delay_line_r.process(mono * self.pan) * self.volume;
        }
    }

    /// Reconfigures the crossover low-pass if the cutoff changed.
    fn ensure_crossover(&mut self, cutoff: f64) {
        if cutoff != self.crossover_freq {
            self.crossover_lp
                .set_low_pass(self.sample_rate, cutoff, CROSSOVER_Q);
            self.crossover_hp
                .set_high_pass(self.sample_rate, cutoff, CROSSOVER_Q);
            self.crossover_freq = cutoff;
        }
    }

    /// Sets the static finetune (in semitones) applied to every voice.
    pub fn set_finetune(&mut self, semitones: f64, tuning: TuningSystem) {
        self.finetune = semitones;
        for instr in &mut self.instrs {
            instr.set_finetune(semitones, tuning);
        }
    }

    /// Sets the stereo pan (0 = left, 1 = right), recomputing the Haas delay lines.
    pub fn set_pan(&mut self, pan: f64, config: &SynthConfig) {
        const SPEED_OF_SOUND: f64 = 343.0;
        let r = 3.0;
        let ear_x = 0.20;
        let mut x = pan * 2.0 - 1.0;
        let gain_r = 1.0;
        if config.force_stereo_separation && x > -0.2 && x < 0.2 {
            x = 0.2 * x.signum();
        }
        let y = (r * r - x * x).sqrt();
        let mut dist_l = ((ear_x + x).powi(2) + y * y).sqrt();
        let mut dist_r = ((-ear_x + x).powi(2) + y * y).sqrt();
        let min_dist = dist_l.min(dist_r);
        dist_l -= min_dist;
        dist_r -= min_dist;
        let delay_l = (dist_l / SPEED_OF_SOUND * 50.0 * self.sample_rate).round() as usize;
        let delay_r = (dist_r / SPEED_OF_SOUND * 50.0 * self.sample_rate).round() as usize;
        self.delay_line_l.set_delay(delay_l);
        self.delay_line_r.set_delay(delay_r);
        self.delay_line_r.gain = gain_r;
        self.pan = pan;
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
        instr.sample_frequency = 440.0;
        instr.set_note(69.0, TuningSystem::Equal); // 440 Hz ⇒ freq_ratio = 1
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

    /// Plays a constant-amplitude looping sample hard-left, settles, and returns `(val_l, val_r)`.
    fn run_dc(config: &SynthConfig) -> (f64, f64) {
        let sample_rate = 32768.0;
        let mut synth = SampleSynthesizer::new(sample_rate, 16);
        // DC sample so essentially all energy is in the low (bass) band.
        let sample = Arc::new(Sample::new(vec![1.0; 64], 440.0, sample_rate, true, 0));
        synth.play(sample, 69.0, 440.0, 1.0, 0, TuningSystem::Equal);
        // Pan hard left.
        synth.set_pan(0.0, config);
        let mut last = (0.0, 0.0);
        for _ in 0..8000 {
            synth.next_sample(config);
            last = (synth.val_l, synth.val_r);
        }
        last
    }

    #[test]
    fn bass_mono_centers_low_frequencies() {
        // With plain separation, a hard-left DC tone barely reaches the right channel.
        let separated = SynthConfig {
            stereo_separation: true,
            bass_mono: false,
            ..SynthConfig::default()
        };
        let (l, r) = run_dc(&separated);
        assert!(
            l.abs() > 0.1,
            "left should carry the panned signal, got {l}"
        );
        assert!(r.abs() < 1e-3, "right should be nearly silent, got {r}");

        // With bass-mono on, the (low-frequency) DC tone is glued to the center: equal L/R.
        let glued = SynthConfig {
            stereo_separation: true,
            bass_mono: true,
            bass_mono_freq: 200.0,
            ..SynthConfig::default()
        };
        let (l, r) = run_dc(&glued);
        assert!(
            r.abs() > 0.1,
            "right should now carry centered bass, got {r}"
        );
        assert!((l - r).abs() < 1e-6, "bass should be centered: {l} vs {r}");
    }
}

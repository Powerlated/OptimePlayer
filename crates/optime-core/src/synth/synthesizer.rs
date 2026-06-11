//! [`SampleSynthesizer`]: a polyphonic synthesizer for one sequence track — a round-robin voice
//! pool mixed to stereo through pan, optional Haas widening, and the bass-mono crossover.

use std::sync::Arc;

use super::delay::DelayLine;
use super::instrument::SampleInstrument;
use super::{CROSSOVER_Q, MAX_BLOCK};
use crate::controller::SynthConfig;
use crate::dsp::BiquadFilter;
use crate::resample::ResampleTables;
use crate::sample::{ResampleMode, Sample};
use crate::tuning::TuningSystem;

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

    /// Rebuilds the sinc tables if the mode switched to a sinc variant or the tap count changed.
    fn ensure_tables(&mut self, config: &SynthConfig) {
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
    }

    /// Advances all active voices by one sample and mixes them into `val_l`/`val_r`.
    pub fn next_sample(&mut self, config: &SynthConfig) {
        self.ensure_tables(config);
        let tables = self.resample_tables.as_ref();

        let mut mono = 0.0;
        for &i in &self.active_instrs {
            self.instrs[i].advance(config.resample, tables);
            mono += self.instrs[i].output;
        }

        self.apply_stereo(mono, config);
    }

    /// Renders `n` samples in one block, adding the track's stereo output into
    /// `acc_l[..n]`/`acc_r[..n]` when `mix` is true. State (voice positions, delay lines,
    /// crossover filters) advances identically either way — `mix: false` matches how
    /// [`Controller::next_sample`] keeps disabled tracks running without mixing them.
    ///
    /// Equivalent to `n` calls of [`Self::next_sample`], but voices render the whole block in one
    /// pass (see [`SampleInstrument::advance_block`]). `n` must be at most [`MAX_BLOCK`].
    ///
    /// [`Controller::next_sample`]: crate::controller::Controller::next_sample
    pub fn render_block(
        &mut self,
        config: &SynthConfig,
        n: usize,
        acc_l: &mut [f64],
        acc_r: &mut [f64],
        mix: bool,
    ) {
        assert!(n <= MAX_BLOCK && acc_l.len() >= n && acc_r.len() >= n);
        self.ensure_tables(config);
        let tables = self.resample_tables.as_ref();

        let mut mono = [0.0f64; MAX_BLOCK];
        for &i in &self.active_instrs {
            self.instrs[i].advance_block(config.resample, tables, &mut mono[..n]);
        }

        for ((&m, l), r) in mono[..n].iter().zip(&mut acc_l[..n]).zip(&mut acc_r[..n]) {
            self.apply_stereo(m, config);
            if mix {
                *l += self.val_l;
                *r += self.val_r;
            }
        }
    }

    /// The per-sample stereo stage: pan, optional Haas widening, and the bass-mono crossover.
    /// Consumes the voice-mixed `mono` sample and sets `val_l`/`val_r`.
    fn apply_stereo(&mut self, mono: f64, config: &SynthConfig) {
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

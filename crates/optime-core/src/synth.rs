//! Sample playback: per-voice [`SampleInstrument`]s, a polyphonic [`SampleSynthesizer`] per
//! track, and the stereo-separation [`DelayLine`].

use std::sync::Arc;

use crate::controller::SynthConfig;
use crate::sample::Sample;
use crate::tuning::{midi_note_to_hz, TuningSystem};

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
            finetune: 0.0,
            finetune_lfo: 0.0,
            freq_ratio: 0.0,
            output: 0.0,
        }
    }

    /// Advances playback by one output sample, updating [`Self::output`].
    pub fn advance(&mut self) {
        let converted_sample_rate = self.freq_ratio * self.sample.sample_rate;
        self.sample_t += self.inv_sample_rate * converted_sample_rate;
        self.output = self.sample_data_at(self.sample_t.floor() as i64) * self.volume;
    }

    /// Returns the (loop-aware) sample value at integer position `t`.
    fn sample_data_at(&self, mut t: i64) -> f64 {
        let len = self.sample.data.len() as i64;
        if t >= len && self.sample.looping {
            let loop_point = self.sample.loop_point;
            let loop_length = len - loop_point;
            if loop_length <= 0 {
                return 0.0;
            }
            let t_no_intro = (t - loop_point).rem_euclid(loop_length);
            t = t_no_intro + loop_point;
        }
        if t >= 0 && t < len {
            f64::from(self.sample.data[t as usize])
        } else {
            0.0
        }
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
    finetune: f64,
}

impl SampleSynthesizer {
    /// Creates a synthesizer with `instrs_available` voices at `sample_rate`.
    pub fn new(sample_rate: f64, instrs_available: usize) -> Self {
        let empty = Arc::new(Sample::new(vec![0.0], 440.0, sample_rate, false, 0));
        let instrs = (0..instrs_available)
            .map(|_| SampleInstrument::new(sample_rate, empty.clone()))
            .collect();
        let delay_len = (sample_rate * 0.1).round() as usize;
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
            finetune: 0.0,
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
        let mut val_l = 0.0;
        let mut val_r = 0.0;
        for &i in &self.active_instrs {
            self.instrs[i].advance();
            let output = self.instrs[i].output;
            val_l += output * (1.0 - self.pan);
            val_r += output * self.pan;
        }

        if config.stereo_separation {
            self.val_l = self.delay_line_l.process(val_l) * self.volume;
            self.val_r = self.delay_line_r.process(val_r) * self.volume;
        } else {
            self.val_l = val_l * self.volume;
            self.val_r = val_r * self.volume;
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

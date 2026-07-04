//! [`WaveformSynthesizer`]: a polyphonic synthesizer for one sequence track — a round-robin voice
//! pool mixed to stereo through pan, optional Haas widening, and the bass-mono crossover.

use std::sync::Arc;

use super::delay_line::DelayLine;
use super::instrument::WaveformInstrument;
use super::{CROSSOVER_Q, MAX_BLOCK};
use crate::devices::VoicePitch;
use crate::dsp::biquad_filter::BiquadFilter;
use crate::dsp::resample::{mode_half_taps, ResampleTables};
use crate::dsp::slewer::Slewer;
use crate::waveform::{Sample, Waveform};
use crate::synth_controller::{DelaySmoothing, PopSmoothing};
use crate::tuning::TuningSystem;
use crate::PerDeviceSettings;

/// Seconds the panning slew takes to cross the full 0..1 pan range when pan smoothing is on. Short
/// enough to track quick auto-pans, long enough to turn a hard pan jump into a click-free ramp.
const PAN_SLEW_SECONDS: f64 = 0.01;

/// A polyphonic synthesizer for one sequence track. Holds a fixed pool of voices and mixes the
/// active ones into a stereo sample, optionally widened by per-channel delay lines.
pub struct WaveformSynthesizer {
    sample_rate: f64,
    instrs: Vec<WaveformInstrument>,
    active_instrs: Vec<usize>,
    playing_index: usize,
    /// Last mixed left output.
    pub val_l: Sample,
    /// Last mixed right output.
    pub val_r: Sample,
    /// Track volume (0..1).
    pub volume: f64,
    /// The pan target most recently requested (0 = left, 1 = right). The applied pan in
    /// [`Self::apply_stereo`] either jumps to this or slews toward it (see [`PerDeviceSettings::smooth_pan`]).
    pan_target: f64,
    /// The pan gain-split value actually applied last sample. Slews toward `pan_target` when pan
    /// smoothing is on; otherwise tracks it exactly.
    pan: Slewer,
    delay_line_l: DelayLine,
    delay_line_r: DelayLine,
    /// Delay lengths waiting for a note-free moment ([`DelaySmoothing::HoldDuringNotes`]).
    pending_delays: Option<(usize, usize)>,
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

impl WaveformSynthesizer {
    /// Creates a synthesizer with `instrs_available` voices at `sample_rate`.
    pub fn new(sample_rate: f64, instrs_available: usize) -> Self {
        let empty = Arc::new(Waveform::new(vec![0.0], 440.0, sample_rate, false, 0));
        let instrs = (0..instrs_available)
            .map(|_| WaveformInstrument::new(sample_rate, empty.clone()))
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
            pan_target: 0.5,
            pan: Slewer::from_time(0.5, PAN_SLEW_SECONDS, sample_rate),
            delay_line_l: DelayLine::new(delay_len),
            delay_line_r: DelayLine::new(delay_len),
            pending_delays: None,
            crossover_lp: BiquadFilter::low_pass(4, sample_rate, crossover_freq, CROSSOVER_Q),
            crossover_hp: BiquadFilter::high_pass(4, sample_rate, crossover_freq, CROSSOVER_Q),
            crossover_freq,
            finetune: 0.0,
            resample_tables: None,
            resample_half_taps: 0,
        }
    }

    /// Re-targets every voice and the rate-dependent stereo/crossover state to a new output
    /// sample rate. A no-op when the rate is unchanged.
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        if sample_rate == self.sample_rate {
            return;
        }
        let ratio = sample_rate / self.sample_rate;
        self.sample_rate = sample_rate;
        for instr in &mut self.instrs {
            instr.set_sample_rate(sample_rate);
        }
        // Keep the pan slew the same wall-clock duration at the new rate.
        self.pan.set_step(1.0 / (PAN_SLEW_SECONDS * sample_rate));

        // Resize the Haas delay lines to the new 100 ms capacity, rescaling the current (and any
        // pending) delay *length* by the rate ratio so the physical widening time is preserved.
        let capacity = (sample_rate * 0.1).round() as usize;
        let rescale = |len: usize| (len as f64 * ratio).round() as usize;
        let (delay_l, delay_r) = (
            rescale(self.delay_line_l.delay()),
            rescale(self.delay_line_r.delay()),
        );
        self.delay_line_l.set_capacity(capacity);
        self.delay_line_r.set_capacity(capacity);
        self.delay_line_l.set_delay(delay_l);
        self.delay_line_r.set_delay(delay_r);
        self.pending_delays = self.pending_delays.map(|(l, r)| (rescale(l), rescale(r)));

        // Rebuild the crossover for the new rate at its current cutoff.
        self.crossover_lp
            .set_low_pass(sample_rate, self.crossover_freq, CROSSOVER_Q);
        self.crossover_hp
            .set_high_pass(sample_rate, self.crossover_freq, CROSSOVER_Q);
    }

    /// Number of voices in the pool.
    #[inline]
    pub fn voice_count(&self) -> usize {
        self.instrs.len()
    }

    /// Number of voices currently sounding.
    #[inline]
    pub fn active_voice_count(&self) -> usize {
        self.active_instrs.len()
    }

    /// Immutable access to a voice.
    #[inline]
    pub fn instr(&self, index: usize) -> &WaveformInstrument {
        &self.instrs[index]
    }

    /// Mutable access to a voice (used by the controller for ADSR/LFO updates).
    #[inline]
    pub fn instr_mut(&mut self, index: usize) -> &mut WaveformInstrument {
        &mut self.instrs[index]
    }

    /// Starts `waveform` at `pitch` on the next round-robin voice and returns its index.
    pub fn play(
        &mut self,
        waveform: Arc<Waveform>,
        pitch: VoicePitch,
        volume: f64,
        config: &PerDeviceSettings,
    ) -> usize {
        let tuning = config.tuning();
        let index = self.playing_index;
        if self.instrs[index].playing {
            self.cut_instrument(index);
        }

        {
            let instr = &mut self.instrs[index];
            instr.waveform = waveform;
            instr.set_finetune_lfo(0.0, tuning);
            instr.set_finetune(self.finetune, tuning);
            instr.set_pitch(pitch, tuning);
            instr.begin_note(volume, config.pop_smoothing());
            instr.sample_t = 0.0;
            instr.wrapped = false;
            instr.playing = true;
        }

        self.playing_index = (self.playing_index + 1) % self.instrs.len();
        self.active_instrs.push(index);
        index
    }

    /// Stops the voice at `index` if it is active: immediately, or — when `pops` enables smoothing
    /// for this voice's kind — via a short fade-out after which the voice stops itself.
    pub fn stop_instrument(&mut self, index: usize, pops: PopSmoothing) {
        let instr = &mut self.instrs[index];
        if instr.playing && pops.enabled_for(instr.waveform.is_psg_square) {
            instr.begin_fade_out();
        } else {
            self.cut_instrument(index);
        }
    }

    /// Stops the voice at `index` immediately if it is active.
    pub fn cut_instrument(&mut self, index: usize) {
        if let Some(pos) = self.active_instrs.iter().position(|&i| i == index) {
            self.instrs[index].playing = false;
            self.active_instrs.remove(pos);
        }
    }

    /// Drops voices that stopped themselves (a landed fade-out) from the active pool.
    fn prune_stopped(&mut self) {
        let instrs = &self.instrs;
        self.active_instrs.retain(|&i| instrs[i].playing);
    }

    /// Rebuilds the sinc tables if the mode switched to a sinc variant or the tap count changed.
    fn ensure_tables(&mut self, config: &PerDeviceSettings) {
        let needed_taps = mode_half_taps(config.resample());
        if let Some(ht) = needed_taps {
            if self.resample_tables.is_none() || self.resample_half_taps != ht {
                self.resample_tables = Some(ResampleTables::new(ht));
                self.resample_half_taps = ht;
            }
        }
    }

    /// Advances all active voices by one sample and mixes them into `val_l`/`val_r`.
    pub fn next_sample(&mut self, config: &PerDeviceSettings) {
        self.ensure_tables(config);
        let tables = self.resample_tables.as_ref();
        let resample = config.resample();
        let pop_smoothing = config.pop_smoothing();

        let mut mono = 0.0;
        for &i in &self.active_instrs {
            self.instrs[i].advance(resample, tables, pop_smoothing);
            mono += self.instrs[i].output;
        }
        self.prune_stopped();
        self.apply_pending_delays();

        self.apply_stereo(mono, config);
    }

    /// Renders `n` samples in one block, adding the track's stereo output into
    /// `acc_l[..n]`/`acc_r[..n]` when `mix` is true. State (voice positions, delay lines,
    /// crossover filters) advances identically either way — `mix: false` matches how
    /// [`SynthController::next_sample`] keeps disabled tracks running without mixing them.
    ///
    /// Equivalent to `n` calls of [`Self::next_sample`], but voices render the whole block in one
    /// pass (see [`WaveformInstrument::advance_block`]). `n` must be at most [`MAX_BLOCK`].
    ///
    /// [`SynthController::next_sample`]: crate::synth_controller::SynthController::next_sample
    pub fn render_block(
        &mut self,
        config: &PerDeviceSettings,
        n: usize,
        acc_l: &mut [Sample],
        acc_r: &mut [Sample],
        mix: bool,
    ) {
        assert!(n <= MAX_BLOCK && acc_l.len() >= n && acc_r.len() >= n);
        self.ensure_tables(config);
        let tables = self.resample_tables.as_ref();
        let resample = config.resample();
        let pop_smoothing = config.pop_smoothing();

        let mut mono: [Sample; MAX_BLOCK] = [0.0; MAX_BLOCK];
        for &i in &self.active_instrs {
            self.instrs[i].advance_block(resample, tables, pop_smoothing, &mut mono[..n]);
        }
        self.prune_stopped();
        self.apply_pending_delays();

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
    fn apply_stereo(&mut self, mono: Sample, config: &PerDeviceSettings) {
        // Resolve this sample's pan gain split: slew toward the target when smoothing is on,
        // otherwise jump straight to it (keeping the slewer in sync for a later toggle).
        let pan = if config.smooth_pan {
            self.pan.advance(self.pan_target)
        } else {
            self.pan.set(self.pan_target);
            self.pan_target
        };

        // Narrow the f64 pan/volume gains to the sample width once, so the mix arithmetic is done
        // entirely in `Sample` (a no-op when `Sample = f64`).
        let vol = self.volume as Sample;
        let gl = (1.0 - pan) as Sample;
        let gr = pan as Sample;

        if !config.stereo_separation {
            self.val_l = mono * gl * vol;
            self.val_r = mono * gr * vol;
            return;
        }

        if config.bass_mono {
            // Split into a centered low band and a widened high band via crossover.
            self.ensure_crossover(f64::from(config.bass_mono_freq));
            let lo = self.crossover_lp.transform(mono);
            let hi = self.crossover_hp.transform(mono);
            let center = lo * 0.5;
            let hi_l = self.delay_line_l.process(hi * gl);
            let hi_r = self.delay_line_r.process(hi * gr);
            self.val_l = (hi_l + center) * vol;
            self.val_r = (hi_r + center) * vol;
        } else {
            self.val_l = self.delay_line_l.process(mono * gl) * vol;
            self.val_r = self.delay_line_r.process(mono * gr) * vol;
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
    ///
    /// The pan gains apply immediately unless [`PerDeviceSettings::smooth_pan`] is set, in which
    /// case [`Self::apply_stereo`] slews them toward the new target over [`PAN_SLEW_SECONDS`].
    /// Independently, under [`DelaySmoothing::HoldDuringNotes`] a delay-*length* change (which
    /// would click in the middle of flowing audio) is deferred until the track has no notes
    /// playing.
    pub fn set_pan(&mut self, pan: f64, config: &PerDeviceSettings) {
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
        match config.delay_smoothing() {
            DelaySmoothing::HoldDuringNotes if !self.active_instrs.is_empty() => {
                self.pending_delays = Some((delay_l, delay_r));
            }
            _ => {
                self.pending_delays = None;
                self.delay_line_l.set_delay(delay_l);
                self.delay_line_r.set_delay(delay_r);
            }
        }
        self.delay_line_r.gain = gain_r;
        self.pan_target = pan;
    }

    /// Applies a deferred delay-length change once the track is note-free.
    fn apply_pending_delays(&mut self) {
        if self.active_instrs.is_empty() {
            if let Some((delay_l, delay_r)) = self.pending_delays.take() {
                self.delay_line_l.set_delay(delay_l);
                self.delay_line_r.set_delay(delay_r);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::waveform::Frame;

    #[test]
    fn set_sample_rate_resizes_delays_and_preserves_pool() {
        let mut synth = WaveformSynthesizer::new(44_100.0, 8);
        let config = PerDeviceSettings {
            stereo_separation: true,
            ..PerDeviceSettings::neutral()
        };
        // Hard right gives the left delay line a non-zero length to rescale.
        synth.set_pan(1.0, &config);
        let delay_44k = synth.delay_line_l.delay();
        assert!(delay_44k > 0, "expected a non-zero Haas delay to rescale");

        synth.set_sample_rate(96_000.0);
        assert_eq!(synth.voice_count(), 8, "the voice pool must be preserved");
        // Capacity tracks the new 100 ms window, and the delay length is rescaled by the ratio.
        let ratio = 96_000.0 / 44_100.0;
        assert_eq!(
            synth.delay_line_l.delay(),
            (delay_44k as f64 * ratio).round() as usize
        );

        // No-op when the rate is unchanged.
        let before = synth.delay_line_l.delay();
        synth.set_sample_rate(96_000.0);
        assert_eq!(synth.delay_line_l.delay(), before);
    }

    /// Plays a constant-amplitude looping waveform hard-left, settles, and returns `(val_l, val_r)`.
    fn run_dc(config: &PerDeviceSettings) -> Frame {
        let sample_rate = 32768.0;
        let mut synth = WaveformSynthesizer::new(sample_rate, 16);
        // DC waveform so essentially all energy is in the low (bass) band.
        let waveform = Arc::new(Waveform::new(vec![1.0; 64], 440.0, sample_rate, true, 0));
        synth.play(
            waveform,
            VoicePitch::Midi {
                note: 69.0,
                sample_pitch_hz: 440.0,
            },
            1.0,
            config,
        );
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
        let separated = PerDeviceSettings {
            stereo_separation: true,
            bass_mono: false,
            ..PerDeviceSettings::neutral()
        };
        let (l, r) = run_dc(&separated);
        assert!(
            l.abs() > 0.1,
            "left should carry the panned signal, got {l}"
        );
        assert!(r.abs() < 1e-3, "right should be nearly silent, got {r}");

        // With bass-mono on, the (low-frequency) DC tone is glued to the center: equal L/R.
        let glued = PerDeviceSettings {
            stereo_separation: true,
            bass_mono: true,
            bass_mono_freq: 200.0,
            ..PerDeviceSettings::neutral()
        };
        let (l, r) = run_dc(&glued);
        assert!(
            r.abs() > 0.1,
            "right should now carry centered bass, got {r}"
        );
        assert!((l - r).abs() < 1e-6, "bass should be centered: {l} vs {r}");
    }

    #[test]
    fn smooth_pan_slews_the_gain_split() {
        // With pan smoothing on, a hard pan change must ramp the L/R gain split over a few
        // milliseconds instead of stepping it; with it off the same change applies instantly.
        // Stereo separation is off so the split is read straight off `val_l`/`val_r` (no Haas
        // delay/crossover in the way), and the source is DC so `mono` is a constant 1.0.
        let sample_rate = 48_000.0;
        let base = PerDeviceSettings {
            stereo_separation: false,
            ..PerDeviceSettings::neutral()
        };
        let smooth = PerDeviceSettings {
            smooth_pan: true,
            ..base.clone()
        };
        let waveform = Arc::new(Waveform::new(vec![1.0; 64], 440.0, sample_rate, true, 0));
        let pitch = VoicePitch::Midi {
            note: 69.0,
            sample_pitch_hz: 440.0,
        };

        let mut synth = WaveformSynthesizer::new(sample_rate, 4);
        synth.play(waveform.clone(), pitch, 1.0, &smooth);
        synth.next_sample(&smooth);
        // Centered: equal L/R.
        assert!((synth.val_l - synth.val_r).abs() < 1e-9);

        // Pan hard right: the right channel must climb gradually, not jump.
        synth.set_pan(1.0, &smooth);
        synth.next_sample(&smooth);
        assert!(
            synth.val_r < 0.9,
            "pan must ramp, not jump (got {})",
            synth.val_r
        );
        // After enough samples it lands fully right, silencing the left.
        for _ in 0..2000 {
            synth.next_sample(&smooth);
        }
        assert!((synth.val_r - 1.0).abs() < 1e-9, "should settle hard right");
        assert!(
            synth.val_l.abs() < 1e-9,
            "left should be silent once settled"
        );

        // Smoothing off: the pan change is applied on the very next sample.
        let mut synth2 = WaveformSynthesizer::new(sample_rate, 4);
        synth2.play(waveform, pitch, 1.0, &base);
        synth2.next_sample(&base);
        synth2.set_pan(1.0, &base);
        synth2.next_sample(&base);
        assert!(
            (synth2.val_r - 1.0).abs() < 1e-9,
            "no smoothing should step straight to hard right, got {}",
            synth2.val_r
        );
    }

    #[test]
    fn delay_change_held_while_notes_play() {
        // Under HoldDuringNotes, a pan change while a note sounds must not move the widening
        // delay lines (only the gains); the deferred lengths land once the track is silent.
        let sample_rate = 32768.0;
        let config = PerDeviceSettings {
            stereo_separation: true,
            delay_smoothing_choice: 1, // HoldDuringNotes
            ..PerDeviceSettings::neutral()
        };
        let mut synth = WaveformSynthesizer::new(sample_rate, 4);
        synth.set_pan(0.0, &config); // hard left while silent: applies immediately
        let baseline_delay_r = synth.delay_line_r.delay();

        let waveform = Arc::new(Waveform::new(vec![1.0; 64], 440.0, sample_rate, true, 0));
        let slot = synth.play(
            waveform,
            VoicePitch::Midi {
                note: 69.0,
                sample_pitch_hz: 440.0,
            },
            1.0,
            &config,
        );
        // Pan hard right mid-note: the delay change must be held back.
        synth.set_pan(1.0, &config);
        synth.next_sample(&config);
        assert_eq!(
            synth.delay_line_r.delay(),
            baseline_delay_r,
            "delay must not change while a note plays"
        );
        assert!(synth.pending_delays.is_some(), "the change is deferred");

        // Cut the note: the pending lengths land on the next sample.
        synth.cut_instrument(slot);
        synth.next_sample(&config);
        assert!(synth.pending_delays.is_none(), "deferred change applied");
        assert_ne!(
            synth.delay_line_r.delay(),
            baseline_delay_r,
            "hard-right pan should produce different delays than hard-left"
        );

        // With smoothing off the same change applies instantly, even mid-note.
        let immediate = PerDeviceSettings {
            stereo_separation: true,
            delay_smoothing_choice: 0, // None
            ..PerDeviceSettings::neutral()
        };
        synth.play(
            Arc::new(Waveform::new(vec![1.0; 64], 440.0, sample_rate, true, 0)),
            VoicePitch::Midi {
                note: 69.0,
                sample_pitch_hz: 440.0,
            },
            1.0,
            &immediate,
        );
        synth.set_pan(0.0, &immediate);
        assert_eq!(synth.delay_line_r.delay(), baseline_delay_r);
    }
}

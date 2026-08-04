use std::sync::Arc;

use super::delay_line::DelayLine;
use super::instrument::WaveformInstrument;
use super::{CROSSOVER_Q, MAX_BLOCK};
use crate::PerDeviceSettings;
use crate::devices::VoicePitch;
use crate::dsp::biquad_filter::BiquadFilter;
use crate::dsp::block;
use crate::dsp::resample::{ResampleTables, mode_half_taps};
use crate::dsp::slewer::Slewer;
use crate::synth_controller::{DelaySmoothing, PopSmoothing};
use crate::tuning::TuningSystem;
use crate::waveform::{Sample, Waveform};

const PAN_SLEW_SECONDS: f64 = 0.01;

pub struct WaveformSynthesizer {
    sample_rate: f64,
    instrs: Vec<WaveformInstrument>,
    active_instrs: Vec<usize>,
    playing_index: usize,
    pub val_l: Sample,
    pub val_r: Sample,
    pub volume: Sample,
    pan_l_target: Sample,
    pan_r_target: Sample,
    pan_l: Slewer,
    pan_r: Slewer,
    delay_line_l: DelayLine,
    delay_line_r: DelayLine,
    pending_delays: Option<(usize, usize)>,
    crossover_lp: BiquadFilter,
    crossover_hp: BiquadFilter,
    crossover_freq: f64,
    finetune: f64,
    resample_tables: Option<ResampleTables>,
    resample_half_taps: usize,
}

impl WaveformSynthesizer {
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
            pan_l_target: 0.5,
            pan_r_target: 0.5,
            pan_l: Slewer::from_time(0.5, PAN_SLEW_SECONDS, sample_rate),
            pan_r: Slewer::from_time(0.5, PAN_SLEW_SECONDS, sample_rate),
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

    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        if sample_rate == self.sample_rate {
            return;
        }
        let ratio = sample_rate / self.sample_rate;
        self.sample_rate = sample_rate;
        for instr in &mut self.instrs {
            instr.set_sample_rate(sample_rate);
        }
        let pan_step = (1.0 / (PAN_SLEW_SECONDS * sample_rate)) as Sample;
        self.pan_l.set_step(pan_step);
        self.pan_r.set_step(pan_step);

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

        self.crossover_lp
            .set_low_pass(sample_rate, self.crossover_freq, CROSSOVER_Q);
        self.crossover_hp
            .set_high_pass(sample_rate, self.crossover_freq, CROSSOVER_Q);
    }

    #[inline]
    pub fn voice_count(&self) -> usize {
        self.instrs.len()
    }

    #[inline]
    pub fn active_voice_count(&self) -> usize {
        self.active_instrs.len()
    }

    #[inline]
    pub fn instr(&self, index: usize) -> &WaveformInstrument {
        &self.instrs[index]
    }

    #[inline]
    pub fn instr_mut(&mut self, index: usize) -> &mut WaveformInstrument {
        &mut self.instrs[index]
    }

    pub fn play(
        &mut self,
        waveform: Arc<Waveform>,
        pitch: VoicePitch,
        volume: Sample,
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

    pub fn stop_instrument(&mut self, index: usize, pops: PopSmoothing) {
        let instr = &mut self.instrs[index];
        if instr.playing && pops.enabled_for(instr.waveform.is_psg_square) {
            instr.begin_fade_out();
        } else {
            self.cut_instrument(index);
        }
    }

    pub fn cut_instrument(&mut self, index: usize) {
        if let Some(pos) = self.active_instrs.iter().position(|&i| i == index) {
            self.instrs[index].playing = false;
            self.active_instrs.remove(pos);
        }
    }

    fn prune_stopped(&mut self) {
        let instrs = &self.instrs;
        self.active_instrs.retain(|&i| instrs[i].playing);
    }

    fn quiet_at(&self) -> Option<usize> {
        if self.active_instrs.is_empty() {
            return Some(0);
        }
        self.active_instrs
            .iter()
            .map(|&i| {
                let instr = &self.instrs[i];
                if instr.playing {
                    None
                } else {
                    instr.stopped_at
                }
            })
            .try_fold(0, |latest, stop| stop.map(|s| latest.max(s)))
    }

    fn ensure_tables(&mut self, config: &PerDeviceSettings) {
        let needed_taps = mode_half_taps(config.resample());
        if let Some(ht) = needed_taps
            && (self.resample_tables.is_none() || self.resample_half_taps != ht)
        {
            self.resample_tables = Some(ResampleTables::new(ht));
            self.resample_half_taps = ht;
        }
    }

    pub fn next_sample(&mut self, config: &PerDeviceSettings) {
        let (mut acc_l, mut acc_r) = ([0.0], [0.0]);
        self.render_block(config, 1, &mut acc_l, &mut acc_r, false);
    }

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
        let quiet_at = self.quiet_at();
        self.prune_stopped();

        let (mut out_l, mut out_r): ([Sample; MAX_BLOCK], [Sample; MAX_BLOCK]) =
            ([0.0; MAX_BLOCK], [0.0; MAX_BLOCK]);
        let split = quiet_at
            .filter(|_| self.pending_delays.is_some())
            .unwrap_or(n);
        self.apply_stereo_block(
            &mono[..split],
            &mut out_l[..split],
            &mut out_r[..split],
            config,
        );
        self.apply_pending_delays();
        if split < n {
            self.apply_stereo_block(
                &mono[split..n],
                &mut out_l[split..n],
                &mut out_r[split..n],
                config,
            );
        }
        if mix {
            for (&v, acc) in out_l[..n].iter().zip(&mut acc_l[..n]) {
                *acc += v;
            }
            for (&v, acc) in out_r[..n].iter().zip(&mut acc_r[..n]) {
                *acc += v;
            }
        }
    }

    fn apply_stereo_block(
        &mut self,
        mono: &[Sample],
        out_l: &mut [Sample],
        out_r: &mut [Sample],
        config: &PerDeviceSettings,
    ) {
        let n = block::stereo_len(out_l, out_r);
        debug_assert_eq!(mono.len(), n);
        if n == 0 {
            return;
        }

        let (mut gl, mut gr): ([Sample; MAX_BLOCK], [Sample; MAX_BLOCK]) =
            ([0.0; MAX_BLOCK], [0.0; MAX_BLOCK]);
        let (gl, gr) = (&mut gl[..n], &mut gr[..n]);
        if config.smooth_pan {
            self.pan_l.advance_block(gl, self.pan_l_target);
            self.pan_r.advance_block(gr, self.pan_r_target);
        } else {
            self.pan_l.set(self.pan_l_target);
            self.pan_r.set(self.pan_r_target);
            gl.fill(self.pan_l_target);
            gr.fill(self.pan_r_target);
        }

        let vol = self.volume;

        if !config.stereo_separation {
            for ((o, &m), &g) in out_l.iter_mut().zip(mono).zip(gl.iter()) {
                *o = m * g * vol;
            }
            for ((o, &m), &g) in out_r.iter_mut().zip(mono).zip(gr.iter()) {
                *o = m * g * vol;
            }
        } else if config.bass_mono {
            self.ensure_crossover(f64::from(config.bass_mono_freq));
            out_l.copy_from_slice(mono);
            out_r.copy_from_slice(mono);
            self.crossover_lp.transform_block(out_l);
            self.crossover_hp.transform_block(out_r);

            let mut high_l: [Sample; MAX_BLOCK] = [0.0; MAX_BLOCK];
            let high_l = &mut high_l[..n];
            for ((h, &hi), &gl) in high_l.iter_mut().zip(out_r.iter()).zip(gl.iter()) {
                *h = hi * gl;
            }
            for (hi, &gr) in out_r.iter_mut().zip(gr.iter()) {
                *hi *= gr;
            }
            self.delay_line_l.process_block(high_l);
            self.delay_line_r.process_block(out_r);

            for ((l, r), &h) in out_l.iter_mut().zip(out_r.iter_mut()).zip(high_l.iter()) {
                let center = *l * 0.5;
                *l = (h + center) * vol;
                *r = (*r + center) * vol;
            }
        } else {
            for ((o, &m), &gl) in out_l.iter_mut().zip(mono).zip(gl.iter()) {
                *o = m * gl;
            }
            for ((o, &m), &gr) in out_r.iter_mut().zip(mono).zip(gr.iter()) {
                *o = m * gr;
            }
            self.delay_line_l.process_block(out_l);
            self.delay_line_r.process_block(out_r);
            for o in out_l.iter_mut().chain(out_r.iter_mut()) {
                *o *= vol;
            }
        }

        self.val_l = out_l[n - 1];
        self.val_r = out_r[n - 1];
    }

    fn ensure_crossover(&mut self, cutoff: f64) {
        if cutoff != self.crossover_freq {
            self.crossover_lp
                .set_low_pass(self.sample_rate, cutoff, CROSSOVER_Q);
            self.crossover_hp
                .set_high_pass(self.sample_rate, cutoff, CROSSOVER_Q);
            self.crossover_freq = cutoff;
        }
    }

    pub fn set_finetune(&mut self, semitones: f64, tuning: TuningSystem) {
        self.finetune = semitones;
        for instr in &mut self.instrs {
            instr.set_finetune(semitones, tuning);
        }
    }

    pub fn set_pan(&mut self, pan_vol_l: f64, pan_vol_r: f64, config: &PerDeviceSettings) {
        let pan = if pan_vol_l + pan_vol_r > 0.0 {
            pan_vol_r / (pan_vol_l + pan_vol_r)
        } else {
            0.5
        };
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
        self.delay_line_r.gain = gain_r as Sample;
        self.pan_l_target = pan_vol_l as Sample;
        self.pan_r_target = pan_vol_r as Sample;
    }

    fn apply_pending_delays(&mut self) {
        if self.active_instrs.is_empty()
            && let Some((delay_l, delay_r)) = self.pending_delays.take()
        {
            self.delay_line_l.set_delay(delay_l);
            self.delay_line_r.set_delay(delay_r);
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
        synth.set_pan(0.0, 1.0, &config);
        let delay_44k = synth.delay_line_l.delay();
        assert!(delay_44k > 0, "expected a non-zero Haas delay to rescale");

        synth.set_sample_rate(96_000.0);
        assert_eq!(synth.voice_count(), 8, "the voice pool must be preserved");
        let ratio = 96_000.0 / 44_100.0;
        assert_eq!(
            synth.delay_line_l.delay(),
            (delay_44k as f64 * ratio).round() as usize
        );

        let before = synth.delay_line_l.delay();
        synth.set_sample_rate(96_000.0);
        assert_eq!(synth.delay_line_l.delay(), before);
    }

    fn run_dc(config: &PerDeviceSettings) -> Frame {
        let sample_rate = 32768.0;
        let mut synth = WaveformSynthesizer::new(sample_rate, 16);
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
        synth.set_pan(1.0, 0.0, config);
        let mut last = (0.0, 0.0);
        for _ in 0..8000 {
            synth.next_sample(config);
            last = (synth.val_l, synth.val_r);
        }
        last
    }

    #[test]
    fn bass_mono_centers_low_frequencies() {
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
        assert!((synth.val_l - synth.val_r).abs() < 1e-9);

        synth.set_pan(0.0, 1.0, &smooth);
        synth.next_sample(&smooth);
        assert!(
            synth.val_r < 0.9,
            "pan must ramp, not jump (got {})",
            synth.val_r
        );
        for _ in 0..2000 {
            synth.next_sample(&smooth);
        }
        assert!((synth.val_r - 1.0).abs() < 1e-9, "should settle hard right");
        assert!(
            synth.val_l.abs() < 1e-9,
            "left should be silent once settled"
        );

        let mut synth2 = WaveformSynthesizer::new(sample_rate, 4);
        synth2.play(waveform, pitch, 1.0, &base);
        synth2.next_sample(&base);
        synth2.set_pan(0.0, 1.0, &base);
        synth2.next_sample(&base);
        assert!(
            (synth2.val_r - 1.0).abs() < 1e-9,
            "no smoothing should step straight to hard right, got {}",
            synth2.val_r
        );
    }

    #[test]
    fn delay_change_held_while_notes_play() {
        let sample_rate = 32768.0;
        let config = PerDeviceSettings {
            stereo_separation: true,
            delay_smoothing_choice: 1,
            ..PerDeviceSettings::neutral()
        };
        let mut synth = WaveformSynthesizer::new(sample_rate, 4);
        synth.set_pan(1.0, 0.0, &config);
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
        synth.set_pan(0.0, 1.0, &config);
        synth.next_sample(&config);
        assert_eq!(
            synth.delay_line_r.delay(),
            baseline_delay_r,
            "delay must not change while a note plays"
        );
        assert!(synth.pending_delays.is_some(), "the change is deferred");

        synth.cut_instrument(slot);
        synth.next_sample(&config);
        assert!(synth.pending_delays.is_none(), "deferred change applied");
        assert_ne!(
            synth.delay_line_r.delay(),
            baseline_delay_r,
            "hard-right pan should produce different delays than hard-left"
        );

        let immediate = PerDeviceSettings {
            stereo_separation: true,
            delay_smoothing_choice: 0,
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
        synth.set_pan(1.0, 0.0, &immediate);
        assert_eq!(synth.delay_line_r.delay(), baseline_delay_r);
    }
}

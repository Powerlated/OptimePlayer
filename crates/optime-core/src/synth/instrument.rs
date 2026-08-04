use std::sync::Arc;

use crate::devices::VoicePitch;
use crate::dsp::resample::{
    EffectiveGather, GatherSource, ResampleTables, effective_gather, gather_sinc, sinc_fc,
};
use crate::dsp::slewer::{Direction, Slewer};
use crate::synth_controller::{DEFAULT_POP_SLEW_SECONDS, PopSmoothing};
use crate::tuning::{TuningSystem, midi_note_to_hz};
use crate::waveform::{InstrumentResampleMode, Sample, Waveform};

#[derive(Clone)]
pub struct WaveformInstrument {
    inv_sample_rate: f64,
    pub waveform: Arc<Waveform>,
    pub pitch: VoicePitch,
    pub volume: Sample,
    pub playing: bool,
    gain: Slewer,
    pop_slew_seconds: f64,
    fading_out: bool,
    pub sample_t: Sample,
    pub(super) wrapped: bool,
    finetune: f64,
    finetune_lfo: f64,
    freq_ratio: f64,
    pub output: Sample,
    pub(super) stopped_at: Option<usize>,
}

impl WaveformInstrument {
    pub fn new(sample_rate: f64, waveform: Arc<Waveform>) -> Self {
        let pitch = VoicePitch::Midi {
            note: 0.0,
            sample_pitch_hz: waveform.frequency,
        };
        Self {
            inv_sample_rate: 1.0 / sample_rate,
            waveform,
            pitch,
            volume: 1.0,
            playing: false,
            gain: Slewer::from_time(1.0, DEFAULT_POP_SLEW_SECONDS, sample_rate),
            pop_slew_seconds: DEFAULT_POP_SLEW_SECONDS,
            fading_out: false,
            sample_t: 0.0,
            wrapped: false,
            finetune: 0.0,
            finetune_lfo: 0.0,
            freq_ratio: 0.0,
            output: 0.0,
            stopped_at: None,
        }
    }

    pub fn begin_note(&mut self, volume: Sample, pops: PopSmoothing) {
        self.volume = volume;
        self.pop_slew_seconds = pops.slew_seconds;
        self.refresh_pop_step();
        self.gain.set_direction(pops.direction);
        let ramps_attack = pops.enabled_for(self.waveform.is_psg_square)
            && matches!(pops.direction, Direction::UpOnly | Direction::UpAndDown);
        self.gain.set(if ramps_attack { 0.0 } else { volume });
        self.fading_out = false;
    }

    fn refresh_pop_step(&mut self) {
        let step = if self.pop_slew_seconds > 0.0 {
            (self.inv_sample_rate / self.pop_slew_seconds) as Sample
        } else {
            Sample::INFINITY
        };
        self.gain.set_step(step);
    }

    pub fn begin_fade_out(&mut self) {
        self.fading_out = true;
    }

    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.inv_sample_rate = 1.0 / sample_rate;
        self.refresh_pop_step();
    }

    pub fn advance(
        &mut self,
        mode: InstrumentResampleMode,
        tables: Option<&ResampleTables>,
        pops: PopSmoothing,
    ) {
        let r = (self.freq_ratio * self.waveform.sample_rate * self.inv_sample_rate) as Sample;
        self.sample_t += r;

        let data = &self.waveform.data;
        let looping = self.waveform.looping;
        let loop_point = self.waveform.loop_point;
        let data_len = data.len() as i64;
        let loop_len = data_len - loop_point;

        let fold = looping && loop_len > 0;
        let (folded, wrapped) =
            fold_pos(self.sample_t, fold, data_len as Sample, loop_len as Sample);
        self.sample_t = folded;
        self.wrapped |= wrapped;
        let pos = self.sample_t;

        let target = if self.fading_out { 0.0 } else { self.volume };
        let gain = if pops.enabled_for(self.waveform.is_psg_square) {
            self.gain.advance(target)
        } else {
            self.gain.set(target);
            target
        };
        if self.fading_out && gain == 0.0 {
            self.playing = false;
        }

        if gain == 0.0 {
            self.output = 0.0;
            return;
        }

        let effective = effective_gather(mode, self.waveform.is_psg_square);

        let get = |mut t: i64| -> Sample {
            if t >= data_len && looping {
                if loop_len <= 0 {
                    return 0.0;
                }
                t = (t - loop_point).rem_euclid(loop_len) + loop_point;
            }
            if t >= 0 && t < data_len {
                Sample::from(data[t as usize])
            } else {
                0.0
            }
        };

        let result = match effective {
            EffectiveGather::Nearest => get(pos.floor() as i64),
            EffectiveGather::Linear => {
                let i = pos.floor() as i64;
                let frac = pos - i as Sample;
                let a = get(i);
                let b = get(i + 1);
                a + (b - a) * frac
            }
            EffectiveGather::Sinc {
                step_mode,
                cutoff_hz,
            } => {
                if let Some(tbl) = tables {
                    let fc = sinc_fc(r, self.inv_sample_rate as Sample, step_mode, cutoff_hz);
                    let src = GatherSource {
                        data,
                        looping,
                        loop_point,
                        loop_len,
                        wrapped: self.wrapped,
                    };
                    gather_sinc(&src, tbl, pos, fc, step_mode)
                } else {
                    get(pos.floor() as i64)
                }
            }
        };

        self.output = result * gain;
    }

    pub fn advance_block(
        &mut self,
        mode: InstrumentResampleMode,
        tables: Option<&ResampleTables>,
        pops: PopSmoothing,
        out: &mut [Sample],
    ) {
        self.stopped_at = None;
        let effective = effective_gather(mode, self.waveform.is_psg_square);
        let (
            EffectiveGather::Sinc {
                step_mode,
                cutoff_hz,
            },
            Some(tbl),
        ) = (effective, tables)
        else {
            for (i, slot) in out.iter_mut().enumerate() {
                let was_playing = self.playing;
                self.advance(mode, tables, pops);
                if was_playing && !self.playing {
                    self.stopped_at.get_or_insert(i);
                }
                *slot += self.output;
            }
            return;
        };

        let r = (self.freq_ratio * self.waveform.sample_rate * self.inv_sample_rate) as Sample;
        let data = &self.waveform.data;
        let looping = self.waveform.looping;
        let loop_point = self.waveform.loop_point;
        let data_len = data.len() as i64;
        let data_len_f = data_len as Sample;
        let loop_len = data_len - loop_point;
        let fold = looping && loop_len > 0;
        let loop_len_f = loop_len as Sample;

        let mut pos = self.sample_t;
        let mut wrapped = self.wrapped;
        let smooth = pops.enabled_for(self.waveform.is_psg_square);
        let target = if self.fading_out { 0.0 } else { self.volume };
        let mut gain = self.gain;

        if target == 0.0 && (gain.value() == 0.0 || !smooth) {
            for _ in 0..out.len() {
                pos += r;
                let (p, w) = fold_pos(pos, fold, data_len_f, loop_len_f);
                pos = p;
                wrapped |= w;
            }
            if self.fading_out && !out.is_empty() {
                if self.playing {
                    self.stopped_at = Some(0);
                }
                self.playing = false;
            }
            self.sample_t = pos;
            self.wrapped = wrapped;
            self.gain.set(0.0);
            self.output = 0.0;
            return;
        }

        let fc = sinc_fc(r, self.inv_sample_rate as Sample, step_mode, cutoff_hz);

        let mut last = 0.0;
        for (i, slot) in out.iter_mut().enumerate() {
            pos += r;
            let (p, w) = fold_pos(pos, fold, data_len_f, loop_len_f);
            pos = p;
            wrapped |= w;
            let g = if smooth {
                gain.advance(target)
            } else {
                gain.set(target);
                target
            };
            if self.fading_out && g == 0.0 {
                if self.playing {
                    self.stopped_at = Some(i);
                }
                self.playing = false;
            }
            if g == 0.0 {
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
            last = result * g;
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
            VoicePitch::DataRateHz(hz) => hz / self.waveform.sample_rate * (tune / 12.0).exp2(),
        };
    }

    pub fn set_pitch(&mut self, pitch: VoicePitch, tuning: TuningSystem) {
        self.pitch = pitch;
        self.recompute_freq(tuning);
    }

    pub fn set_finetune_lfo(&mut self, semitones: f64, tuning: TuningSystem) {
        self.finetune_lfo = semitones;
        self.recompute_freq(tuning);
    }

    pub fn set_finetune(&mut self, semitones: f64, tuning: TuningSystem) {
        self.finetune = semitones;
        self.recompute_freq(tuning);
    }
}

#[inline]
fn fold_pos(pos: Sample, fold: bool, data_len: Sample, loop_len: Sample) -> (Sample, bool) {
    if fold && pos >= data_len {
        let mut p = pos;
        while p >= data_len {
            p -= loop_len;
        }
        (p, true)
    } else {
        (pos, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    fn crunch(half_taps: usize) -> InstrumentResampleMode {
        InstrumentResampleMode::SincOutputNyquist {
            half_taps,
            psg_cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
            sampler_cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
        }
    }

    fn sine_waveform(period: usize, periods: usize, src_rate: f64) -> Arc<Waveform> {
        let len = period * periods;
        let data: Vec<f32> = (0..len)
            .map(|k| (2.0 * PI * k as f64 / period as f64).sin() as f32)
            .collect();
        Arc::new(Waveform::new(data, 440.0, src_rate, true, 0))
    }

    #[test]
    fn set_sample_rate_rescales_playback_step() {
        let waveform = Arc::new(Waveform::new(vec![0.0; 4096], 440.0, 22_050.0, false, 0));
        let mut instr = WaveformInstrument::new(44_100.0, waveform);
        instr.set_pitch(VoicePitch::DataRateHz(22_050.0), TuningSystem::Equal);

        instr.advance(
            InstrumentResampleMode::NearestNeighbor,
            None,
            PopSmoothing::default(),
        );
        assert!(
            (instr.sample_t - 0.5).abs() < 1e-9,
            "got {}",
            instr.sample_t
        );

        instr.set_sample_rate(22_050.0);
        instr.advance(
            InstrumentResampleMode::NearestNeighbor,
            None,
            PopSmoothing::default(),
        );
        assert!(
            (instr.sample_t - 1.5).abs() < 1e-9,
            "got {}",
            instr.sample_t
        );
    }

    fn render(
        out_rate: f64,
        waveform: Arc<Waveform>,
        mode: InstrumentResampleMode,
        tables: Option<&ResampleTables>,
        n: usize,
    ) -> Vec<Sample> {
        let mut instr = WaveformInstrument::new(out_rate, waveform);
        instr.set_pitch(
            VoicePitch::Midi {
                note: 69.0,
                sample_pitch_hz: 440.0,
            },
            TuningSystem::Equal,
        );
        instr.begin_note(1.0, PopSmoothing::default());
        instr.playing = true;
        (0..n)
            .map(|_| {
                instr.advance(mode, tables, PopSmoothing::default());
                instr.output
            })
            .collect()
    }

    fn amp_at(signal: &[Sample], freq_hz: f64, rate: f64) -> Sample {
        let w = 2.0 * PI * freq_hz / rate;
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (n, &x) in signal.iter().enumerate() {
            let p = w * n as f64;
            re += f64::from(x) * p.cos();
            im -= f64::from(x) * p.sin();
        }
        (2.0 * (re * re + im * im).sqrt() / signal.len() as f64) as Sample
    }

    #[test]
    fn crunch_mode_preserves_imaging_clean_mode_removes_it() {
        let out_rate = 32768.0;
        let src_rate = 8192.0;
        let waveform = sine_waveform(16, 8, src_rate);
        let fund_hz = 512.0;
        let image_hz = 7680.0;
        let warmup = 256;
        let n = 2048;
        let tables = ResampleTables::new(16);

        let crunch = render(
            out_rate,
            waveform.clone(),
            crunch(16),
            Some(&tables),
            warmup + n,
        );
        let clean = render(
            out_rate,
            waveform,
            InstrumentResampleMode::SincSampleNyquist { half_taps: 16 },
            Some(&tables),
            warmup + n,
        );

        let crunch_fund = amp_at(&crunch[warmup..], fund_hz, out_rate);
        let crunch_image = amp_at(&crunch[warmup..], image_hz, out_rate);
        let clean_fund = amp_at(&clean[warmup..], fund_hz, out_rate);
        let clean_image = amp_at(&clean[warmup..], image_hz, out_rate);

        assert!(crunch_fund > 0.9, "crunch fundamental = {crunch_fund}");
        assert!(clean_fund > 0.9, "clean fundamental = {clean_fund}");

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
        let out_rate = 20480.0;
        let src_rate = 8192.0;
        let waveform = sine_waveform(16, 8, src_rate);
        let image_hz = 7680.0;
        let alias_hz = 4608.0;
        let warmup = 256;
        let n = 2560;
        let tables = ResampleTables::new(16);

        let nearest = render(
            out_rate,
            waveform.clone(),
            InstrumentResampleMode::NearestNeighbor,
            None,
            warmup + n,
        );
        let crunch = render(out_rate, waveform, crunch(16), Some(&tables), warmup + n);

        let nearest_image = amp_at(&nearest[warmup..], image_hz, out_rate);
        let nearest_alias = amp_at(&nearest[warmup..], alias_hz, out_rate);
        let crunch_image = amp_at(&crunch[warmup..], image_hz, out_rate);
        let crunch_alias = amp_at(&crunch[warmup..], alias_hz, out_rate);

        assert!(
            crunch_image > 0.03,
            "crunch should keep the in-band ZOH image, got {crunch_image}"
        );
        assert!(
            crunch_image > 0.7 * nearest_image,
            "crunch image ({crunch_image}) should match the raw staircase ({nearest_image})"
        );

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
        let out_rate = 32768.0;
        let mut waveform = Waveform::new(vec![1.0, 1.0, -1.0, -1.0], 440.0, 16384.0, true, 0);
        waveform.is_psg_square = true;
        let waveform = Arc::new(waveform);
        let tables = ResampleTables::new(16);
        let n = 512;
        let crunch_mode = render(out_rate, waveform.clone(), crunch(16), Some(&tables), n);
        let clean_mode = render(
            out_rate,
            waveform,
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
        let out_rate = 32768.0;
        let src_rate = 16384.0;
        let tone_hz = 4096.0;
        let warmup = 256;
        let n = 2048;
        let tables = ResampleTables::new(32);
        for is_psg in [false, true] {
            let mut s = sine_waveform(4, 64, src_rate);
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
        let out_rate = 48_000.0;
        let mut waveform = Waveform::new(vec![1.0; 64], 440.0, out_rate, true, 0);
        waveform.is_psg_square = true;
        let waveform = Arc::new(waveform);
        let pitch = VoicePitch::Midi {
            note: 69.0,
            sample_pitch_hz: 440.0,
        };

        let mut instr = WaveformInstrument::new(out_rate, waveform.clone());
        instr.set_pitch(pitch, TuningSystem::Equal);
        instr.playing = true;
        let psg_on = PopSmoothing {
            psg: true,
            sampled: false,
            ..PopSmoothing::default()
        };
        instr.begin_note(1.0, psg_on);
        instr.advance(InstrumentResampleMode::NearestNeighbor, None, psg_on);
        assert!(
            instr.output < 0.05,
            "smoothed start must ramp from silence, got {}",
            instr.output
        );
        let mut prev = instr.output;
        let mut reached = false;
        for _ in 0..1024 {
            instr.advance(InstrumentResampleMode::NearestNeighbor, None, psg_on);
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
            instr.advance(InstrumentResampleMode::NearestNeighbor, None, psg_on);
        }
        assert!(!instr.playing, "fade-out must stop the voice");
        assert_eq!(instr.output, 0.0);

        let mut hard = WaveformInstrument::new(out_rate, waveform);
        hard.set_pitch(pitch, TuningSystem::Equal);
        hard.playing = true;
        hard.begin_note(1.0, PopSmoothing::default());
        hard.advance(
            InstrumentResampleMode::NearestNeighbor,
            None,
            PopSmoothing::default(),
        );
        assert!(
            (hard.output - 1.0).abs() < 1e-12,
            "unsmoothed start must step to full gain, got {}",
            hard.output
        );
    }

    #[test]
    fn pop_smoothing_edge_selects_which_note_edge_ramps() {
        let out_rate = 48_000.0;
        let mut waveform = Waveform::new(vec![1.0; 64], 440.0, out_rate, true, 0);
        waveform.is_psg_square = true;
        let waveform = Arc::new(waveform);
        let pitch = VoicePitch::Midi {
            note: 69.0,
            sample_pitch_hz: 440.0,
        };
        let mode = InstrumentResampleMode::NearestNeighbor;

        let start = |direction| {
            let pops = PopSmoothing {
                psg: true,
                direction,
                ..PopSmoothing::default()
            };
            let mut instr = WaveformInstrument::new(out_rate, waveform.clone());
            instr.set_pitch(pitch, TuningSystem::Equal);
            instr.playing = true;
            instr.begin_note(1.0, pops);
            instr.advance(mode, None, pops);
            (instr, pops)
        };

        let (mut attack, pops) = start(Direction::UpOnly);
        assert!(
            attack.output < 0.05,
            "attack must ramp, got {}",
            attack.output
        );
        attack.volume = 1.0;
        attack.gain.set(1.0);
        attack.begin_fade_out();
        attack.advance(mode, None, pops);
        assert_eq!(attack.output, 0.0, "an attack-only cut must be instant");
        assert!(!attack.playing, "the instant cut must stop the voice");

        let (mut release, pops) = start(Direction::DownOnly);
        assert!(
            (release.output - 1.0).abs() < 1e-12,
            "a release-only start must step to full gain, got {}",
            release.output
        );
        release.begin_fade_out();
        release.advance(mode, None, pops);
        assert!(
            release.output > 0.9 && release.playing,
            "a release-only fade-out must ramp, got {}",
            release.output
        );
    }

    fn alias_residual(half_taps: usize) -> Sample {
        let out_rate = 8192.0;
        let src_rate = 20480.0;
        let alias_hz = 3072.0;
        let waveform = sine_waveform(4, 64, src_rate);
        let tables = ResampleTables::new(half_taps);
        let warmup = 256;
        let n = 2048;
        let out = render(
            out_rate,
            waveform,
            InstrumentResampleMode::SincSampleNyquist { half_taps },
            Some(&tables),
            warmup + n,
        );
        amp_at(&out[warmup..], alias_hz, out_rate)
    }

    #[test]
    fn alias_suppression_increases_with_kernel_size() {
        let taps = [2usize, 4, 8, 16, 32];
        let residuals: Vec<Sample> = taps.iter().map(|&t| alias_residual(t)).collect();
        for w in residuals.windows(2) {
            assert!(
                w[1] < w[0],
                "alias residual should fall with more taps, got {residuals:?} for {taps:?}"
            );
        }
        assert!(
            residuals[0] > residuals[residuals.len() - 1] * 3.0,
            "expected a large suppression gain across the tap range, got {residuals:?}"
        );
    }
}

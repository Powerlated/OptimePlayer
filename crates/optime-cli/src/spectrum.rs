//! High-resolution long-term average spectrum, and the depth of a notch in it.
//!
//! `timbre.rs` deliberately smears the spectrum into third-octave bands, because it is asking a
//! broad question about tonal balance. A reconstruction artifact is the opposite kind of feature: a
//! zero-order hold's `sinc(f / rate)` envelope puts an exact null at its own sample rate, and a
//! third-octave band 2 kHz wide averages that null away to nothing. So this module keeps linear
//! resolution — a 16384-point transform is 2.9 Hz per bin at 48 kHz — and reports one number: how
//! far the level in a narrow window around a probe frequency sits below the shoulders either side
//! of it. Positive is a dip. A hold at the probe frequency reads tens of dB; a spectrum that merely
//! slopes through it reads near zero, because the shoulders are taken symmetrically and their
//! average cancels a linear tilt.

use rustfft::{FftPlanner, num_complex::Complex32};

const WINDOW: usize = 16_384;
const HOP: usize = 8_192;
const FRAME_GATE_DB: f32 = -60.0;
const POWER_FLOOR: f32 = 1.0e-20;
const NOTCH_HALF_WIDTH_HZ: f64 = 150.0;
const SHOULDER_INNER_HZ: f64 = 500.0;
const SHOULDER_OUTER_HZ: f64 = 1_800.0;

pub struct Spectrum {
    pub power: Vec<f32>,
    pub rate: f64,
}

impl Spectrum {
    pub fn analyze(samples: &[f32], rate: f64) -> Option<Self> {
        if samples.len() < WINDOW * 2 || rate <= 0.0 {
            return None;
        }
        let window: Vec<f32> = (0..WINDOW)
            .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / WINDOW as f32).cos())
            .collect();
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(WINDOW);
        let mut scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
        let mut buf = vec![Complex32::default(); WINDOW];

        let mut frames: Vec<Vec<f32>> = Vec::new();
        let mut energy: Vec<f32> = Vec::new();
        let mut start = 0;
        while start + WINDOW <= samples.len() {
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = Complex32::new(samples[start + i] * window[i], 0.0);
            }
            fft.process_with_scratch(&mut buf, &mut scratch);
            let frame: Vec<f32> = buf[..WINDOW / 2].iter().map(|c| c.norm_sqr()).collect();
            energy.push(frame.iter().sum::<f32>());
            frames.push(frame);
            start += HOP;
        }

        let loudest = energy.iter().copied().fold(f32::MIN, f32::max);
        if loudest <= 0.0 {
            return None;
        }
        let gate = loudest * 10f32.powf(FRAME_GATE_DB / 10.0);
        let kept: Vec<&Vec<f32>> = frames
            .iter()
            .zip(&energy)
            .filter(|(_, e)| **e > gate)
            .map(|(f, _)| f)
            .collect();
        if kept.is_empty() {
            return None;
        }

        let power = (0..WINDOW / 2)
            .map(|k| kept.iter().map(|f| f[k]).sum::<f32>() / kept.len() as f32)
            .collect();
        Some(Self { power, rate })
    }

    fn bin(&self, hz: f64) -> usize {
        ((hz * WINDOW as f64 / self.rate).round() as usize).min(self.power.len() - 1)
    }

    fn mean_db(&self, lo_hz: f64, hi_hz: f64) -> f32 {
        let (lo, hi) = (self.bin(lo_hz), self.bin(hi_hz).max(self.bin(lo_hz) + 1));
        let mean = self.power[lo..hi].iter().sum::<f32>() / (hi - lo) as f32;
        10.0 * (mean + POWER_FLOOR).log10()
    }

    pub fn notch_depth_db(&self, probe_hz: f64) -> f32 {
        let inside = self.mean_db(
            probe_hz - NOTCH_HALF_WIDTH_HZ,
            probe_hz + NOTCH_HALF_WIDTH_HZ,
        );
        let below = self.mean_db(probe_hz - SHOULDER_OUTER_HZ, probe_hz - SHOULDER_INNER_HZ);
        let above = self.mean_db(probe_hz + SHOULDER_INNER_HZ, probe_hz + SHOULDER_OUTER_HZ);
        0.5 * (below + above) - inside
    }

    pub fn level_db(&self, hz: f64) -> f32 {
        self.mean_db(hz - NOTCH_HALF_WIDTH_HZ, hz + NOTCH_HALF_WIDTH_HZ)
    }

    pub fn deepest_notches(
        &self,
        lo_hz: f64,
        hi_hz: f64,
        step_hz: f64,
        count: usize,
    ) -> Vec<(f64, f32)> {
        let mut probes: Vec<(f64, f32)> = Vec::new();
        let mut hz = lo_hz;
        while hz <= hi_hz {
            probes.push((hz, self.notch_depth_db(hz)));
            hz += step_hz;
        }
        let mut peaks: Vec<(f64, f32)> = probes
            .windows(3)
            .filter(|w| w[1].1 >= w[0].1 && w[1].1 >= w[2].1)
            .map(|w| w[1])
            .collect();
        peaks.sort_by(|a, b| b.1.total_cmp(&a.1));
        peaks.truncate(count);
        peaks
    }

    pub fn accumulate(&mut self, other: &Spectrum) {
        for (a, b) in self.power.iter_mut().zip(&other.power) {
            *a += *b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f64 = 48_000.0;

    fn held(samples_per_step: usize, seconds: f64) -> Vec<f32> {
        let mut seed = 1u32;
        let mut held = 0.0f32;
        (0..(RATE * seconds) as usize)
            .map(|n| {
                if n % samples_per_step == 0 {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    held = (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5;
                }
                held
            })
            .collect()
    }

    #[test]
    fn a_zero_order_hold_reads_a_deep_notch_at_its_own_rate() {
        let spectrum = Spectrum::analyze(&held(4, 8.0), RATE).unwrap();
        let depth = spectrum.notch_depth_db(RATE / 4.0);
        assert!(depth > 10.0, "hold notch measured only {depth} dB");
    }

    #[test]
    fn a_jittered_hold_fills_its_own_null_in() {
        let mut seed = 1u32;
        let mut held = 0.0f32;
        let mut phase = 0.0f64;
        let jittered: Vec<f32> = (0..(RATE * 8.0) as usize)
            .map(|_| {
                phase += 13_379.0 / RATE;
                if phase >= 1.0 {
                    phase -= 1.0;
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    held = (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5;
                }
                held
            })
            .collect();
        let depth = Spectrum::analyze(&jittered, RATE)
            .unwrap()
            .notch_depth_db(13_379.0);
        assert!(
            depth < 5.0,
            "a hold whose rate does not divide the output rate still notched {depth} dB"
        );
    }

    #[test]
    fn broadband_noise_reads_no_notch() {
        let mut seed = 7u32;
        let noise: Vec<f32> = (0..(RATE * 8.0) as usize)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5
            })
            .collect();
        let depth = Spectrum::analyze(&noise, RATE)
            .unwrap()
            .notch_depth_db(13_379.0);
        assert!(depth.abs() < 1.0, "flat noise reported a {depth} dB notch");
    }
}

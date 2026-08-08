//! Reduces a piece of music to a small, level-invariant description of how it sits in the spectrum
//! and how it moves, and measures the distance between two such descriptions.
//!
//! Everything here is deliberately blind to absolute level, because the two sides being compared —
//! a corpus of finished commercial recordings and a freshly rendered chiptune — have no shared
//! gain reference and normalising one to the other would just move the disagreement into a number
//! nobody chose. The spectrum is third-octave band energy in dB with the mean over bands removed,
//! so only the tilt and the bumps survive; crest, dynamic range and flux are ratios or differences
//! of dB, so they are invariant already. The signal is still scaled to unit RMS first, because the
//! floors that keep the logarithms finite are absolute, and a quiet input would otherwise meet them
//! in bands a loud one clears — which is level dependence smuggled back in through the floor.
//!
//! A `Target` is a corpus reduced to a per-element mean and standard deviation, and `distance`
//! scores a single profile against it in units of that deviation. That weighting is the point of
//! collecting a corpus rather than a single reference track: a band where the reference music
//! itself disagrees track to track carries little information about what the music *should* be, and
//! inverse-deviation weighting is what stops those bands from dominating a sum of squares.

use std::path::Path;

use rustfft::{FftPlanner, num_complex::Complex32};
use serde::{Deserialize, Serialize};

use crate::decode::decode_mono;

pub const BAND_COUNT: usize = 28;
const BAND_LOW_HZ: f64 = 30.0;
const BAND_HIGH_HZ: f64 = 16_000.0;
const WINDOW: usize = 4096;
const HOP: usize = 1024;
const FRAME_GATE_DB: f32 = -60.0;
const LOUDNESS_WINDOW_SECONDS: f64 = 0.4;
const LOUDNESS_HOP_SECONDS: f64 = 0.1;
const RANGE_LOW_PERCENTILE: f32 = 0.10;
const RANGE_HIGH_PERCENTILE: f32 = 0.95;
const POWER_FLOOR: f32 = 1.0e-12;
const DEVIATION_FLOOR_DB: f32 = 0.75;

const CREST_WEIGHT: f32 = 4.0;
const RANGE_WEIGHT: f32 = 4.0;
const FLUX_WEIGHT: f32 = 4.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    pub spectrum_db: Vec<f32>,
    pub crest_db: f32,
    pub dynamic_range_db: f32,
    pub flux_db: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Target {
    pub sources: usize,
    pub mean: Profile,
    pub deviation: Profile,
}

fn band_edges() -> Vec<f64> {
    let ratio = (BAND_HIGH_HZ / BAND_LOW_HZ).powf(1.0 / BAND_COUNT as f64);
    (0..=BAND_COUNT)
        .map(|i| BAND_LOW_HZ * ratio.powi(i as i32))
        .collect()
}

fn band_bins(rate: f64) -> Vec<Vec<usize>> {
    let edges = band_edges();
    let hz_of = |k: usize| k as f64 * rate / WINDOW as f64;
    let nyquist_bin = (WINDOW / 2).max(1);
    (0..BAND_COUNT)
        .map(|b| {
            let (lo, hi) = (edges[b], edges[b + 1]);
            let bins: Vec<usize> = (1..nyquist_bin)
                .filter(|&k| (lo..hi).contains(&hz_of(k)))
                .collect();
            if !bins.is_empty() {
                return bins;
            }
            let centre = (lo * hi).sqrt();
            let nearest = (1..nyquist_bin)
                .min_by(|&a, &b| {
                    (hz_of(a) - centre)
                        .abs()
                        .total_cmp(&(hz_of(b) - centre).abs())
                })
                .unwrap_or(1);
            vec![nearest]
        })
        .collect()
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f32 * p).round() as usize;
    sorted[idx]
}

fn short_term_range_db(samples: &[f32], rate: f64) -> f32 {
    let window = (LOUDNESS_WINDOW_SECONDS * rate) as usize;
    let hop = (LOUDNESS_HOP_SECONDS * rate).max(1.0) as usize;
    if samples.len() < window || window == 0 {
        return 0.0;
    }
    let mut levels: Vec<f32> = samples
        .windows(window)
        .step_by(hop)
        .map(|w| {
            let power = w.iter().map(|x| x * x).sum::<f32>() / w.len() as f32;
            10.0 * (power + POWER_FLOOR).log10()
        })
        .collect();
    let peak = levels.iter().copied().fold(f32::MIN, f32::max);
    levels.retain(|&l| l > peak + FRAME_GATE_DB);
    levels.sort_by(f32::total_cmp);
    percentile(&levels, RANGE_HIGH_PERCENTILE) - percentile(&levels, RANGE_LOW_PERCENTILE)
}

pub fn analyze(input: &[f32], rate: f64) -> Option<Profile> {
    if input.len() < WINDOW * 4 || rate <= 0.0 {
        return None;
    }
    let rms = (input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32).sqrt();
    if rms <= 0.0 || !rms.is_finite() {
        return None;
    }
    let samples: Vec<f32> = input.iter().map(|x| x / rms).collect();
    let samples = samples.as_slice();
    let bands = band_bins(rate);
    let band_widths: Vec<f32> = bands.iter().map(|b| b.len() as f32).collect();
    let window: Vec<f32> = (0..WINDOW)
        .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / WINDOW as f32).cos())
        .collect();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(WINDOW);
    let mut scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
    let mut buf = vec![Complex32::default(); WINDOW];

    let mut frames: Vec<[f32; BAND_COUNT]> = Vec::new();
    let mut frame_db: Vec<f32> = Vec::new();
    let mut start = 0;
    while start + WINDOW <= samples.len() {
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = Complex32::new(samples[start + i] * window[i], 0.0);
        }
        fft.process_with_scratch(&mut buf, &mut scratch);
        let mut power = [0.0f32; BAND_COUNT];
        for ((slot, bins), width) in power.iter_mut().zip(&bands).zip(&band_widths) {
            *slot = bins.iter().map(|&k| buf[k].norm_sqr()).sum::<f32>() / width;
        }
        frame_db.push(10.0 * (power.iter().sum::<f32>() + POWER_FLOOR).log10());
        frames.push(power);
        start += HOP;
    }

    let loudest = frame_db.iter().copied().fold(f32::MIN, f32::max);
    let gate = loudest + FRAME_GATE_DB;
    let kept: Vec<&[f32; BAND_COUNT]> = frames
        .iter()
        .zip(&frame_db)
        .filter(|(_, db)| **db > gate)
        .map(|(f, _)| f)
        .collect();
    if kept.len() < 4 {
        return None;
    }

    let mut spectrum_db = vec![0.0f32; BAND_COUNT];
    for (b, slot) in spectrum_db.iter_mut().enumerate() {
        let mean = kept.iter().map(|f| f[b]).sum::<f32>() / kept.len() as f32;
        *slot = 10.0 * (mean + POWER_FLOOR).log10();
    }
    let tilt = spectrum_db.iter().sum::<f32>() / BAND_COUNT as f32;
    for v in &mut spectrum_db {
        *v -= tilt;
    }

    let flux_db = {
        let per_frame_db = |f: &[f32; BAND_COUNT]| -> Vec<f32> {
            f.iter().map(|p| 10.0 * (p + POWER_FLOOR).log10()).collect()
        };
        let mut total = 0.0f32;
        for pair in kept.windows(2) {
            let (a, b) = (per_frame_db(pair[0]), per_frame_db(pair[1]));
            total += a.iter().zip(&b).map(|(x, y)| (y - x).abs()).sum::<f32>() / BAND_COUNT as f32;
        }
        total / (kept.len() - 1) as f32
    };

    let peak = samples.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let crest_db = 20.0 * (peak + 1.0e-9).log10();

    Some(Profile {
        spectrum_db,
        crest_db,
        dynamic_range_db: short_term_range_db(samples, rate),
        flux_db,
    })
}

pub fn analyze_file(path: &Path, max_seconds: Option<f64>) -> Result<Profile, String> {
    let (mono, rate) = decode_mono(path)?;
    let limit = max_seconds
        .map(|s| ((s * rate) as usize).min(mono.len()))
        .unwrap_or(mono.len());
    analyze(&mono[..limit], rate).ok_or_else(|| "too short or too quiet to analyse".to_string())
}

impl Target {
    pub fn from_profiles(profiles: &[Profile]) -> Option<Self> {
        if profiles.len() < 2 {
            return None;
        }
        let n = profiles.len() as f32;
        let summarise = |get: &dyn Fn(&Profile) -> f32| -> (f32, f32) {
            let mean = profiles.iter().map(get).sum::<f32>() / n;
            let variance = profiles
                .iter()
                .map(|p| (get(p) - mean).powi(2))
                .sum::<f32>()
                / n;
            (mean, variance.sqrt())
        };

        let mut spectrum_mean = vec![0.0f32; BAND_COUNT];
        let mut spectrum_dev = vec![0.0f32; BAND_COUNT];
        for b in 0..BAND_COUNT {
            let (mean, dev) = summarise(&|p: &Profile| p.spectrum_db[b]);
            spectrum_mean[b] = mean;
            spectrum_dev[b] = dev;
        }
        let (crest_mean, crest_dev) = summarise(&|p: &Profile| p.crest_db);
        let (range_mean, range_dev) = summarise(&|p: &Profile| p.dynamic_range_db);
        let (flux_mean, flux_dev) = summarise(&|p: &Profile| p.flux_db);

        Some(Self {
            sources: profiles.len(),
            mean: Profile {
                spectrum_db: spectrum_mean,
                crest_db: crest_mean,
                dynamic_range_db: range_mean,
                flux_db: flux_mean,
            },
            deviation: Profile {
                spectrum_db: spectrum_dev,
                crest_db: crest_dev,
                dynamic_range_db: range_dev,
                flux_db: flux_dev,
            },
        })
    }

    pub fn distance(&self, profile: &Profile) -> f32 {
        let term = |got: f32, want: f32, dev: f32, weight: f32| -> (f32, f32) {
            let z = (got - want) / dev.max(DEVIATION_FLOOR_DB);
            (weight * z * z, weight)
        };
        let mut total = 0.0;
        let mut weight = 0.0;
        for b in 0..BAND_COUNT {
            let (t, w) = term(
                profile.spectrum_db[b],
                self.mean.spectrum_db[b],
                self.deviation.spectrum_db[b],
                1.0,
            );
            total += t;
            weight += w;
        }
        for (got, want, dev, w) in [
            (
                profile.crest_db,
                self.mean.crest_db,
                self.deviation.crest_db,
                CREST_WEIGHT,
            ),
            (
                profile.dynamic_range_db,
                self.mean.dynamic_range_db,
                self.deviation.dynamic_range_db,
                RANGE_WEIGHT,
            ),
            (
                profile.flux_db,
                self.mean.flux_db,
                self.deviation.flux_db,
                FLUX_WEIGHT,
            ),
        ] {
            let (t, wt) = term(got, want, dev, w);
            total += t;
            weight += wt;
        }
        total / weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hz: f64, rate: f64, seconds: f64, amplitude: f32) -> Vec<f32> {
        let n = (rate * seconds) as usize;
        (0..n)
            .map(|i| (std::f64::consts::TAU * hz * i as f64 / rate).sin() as f32 * amplitude)
            .collect()
    }

    fn noise(rate: f64, seconds: f64, amplitude: f32) -> Vec<f32> {
        let mut seed = 1u32;
        (0..(rate * seconds) as usize)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5) * amplitude
            })
            .collect()
    }

    #[test]
    fn every_band_holds_at_least_one_bin_at_every_rate_in_use() {
        for rate in [13_379.0, 22_050.0, 32_768.0, 44_100.0, 48_000.0, 96_000.0] {
            for (b, bins) in band_bins(rate).iter().enumerate() {
                assert!(!bins.is_empty(), "band {b} is empty at {rate} Hz");
            }
        }
    }

    #[test]
    fn the_spectrum_is_blind_to_level() {
        let rate = 48_000.0;
        let quiet = noise(rate, 3.0, 0.02);
        let loud: Vec<f32> = quiet.iter().map(|x| x * 40.0).collect();
        let a = analyze(&quiet, rate).unwrap();
        let b = analyze(&loud, rate).unwrap();
        for (x, y) in a.spectrum_db.iter().zip(&b.spectrum_db) {
            assert!((x - y).abs() < 1.0e-2, "band moved from {x} to {y}");
        }
        assert!((a.crest_db - b.crest_db).abs() < 1.0e-2);
    }

    #[test]
    fn a_brighter_signal_lands_in_higher_bands() {
        let rate = 48_000.0;
        let low = analyze(&tone(200.0, rate, 3.0, 0.4), rate).unwrap();
        let high = analyze(&tone(6_000.0, rate, 3.0, 0.4), rate).unwrap();
        let top_half = |p: &Profile| p.spectrum_db[BAND_COUNT / 2..].iter().sum::<f32>();
        assert!(
            top_half(&high) > top_half(&low),
            "6 kHz tone did not out-weigh a 200 Hz tone in the upper bands"
        );
    }

    #[test]
    fn distance_is_zero_against_the_corpus_mean() {
        let rate = 48_000.0;
        let profiles: Vec<Profile> = [300.0, 900.0, 2_700.0]
            .iter()
            .map(|&hz| analyze(&tone(hz, rate, 3.0, 0.4), rate).unwrap())
            .collect();
        let target = Target::from_profiles(&profiles).unwrap();
        let mean = target.mean.clone();
        assert!(target.distance(&mean) < 1.0e-6);
        assert!(target.distance(&profiles[0]) > 0.0);
    }
}

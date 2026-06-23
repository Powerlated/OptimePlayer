//! [`StreamResampler`]: a continuous, fixed-ratio resampler for a *stream* of stereo samples
//! (the intermediate mixer's bus → the output rate).
//!
//! Unlike the voice gather in [`source`](super::source) — which reads a finite, loop-mapped
//! [`Sample`](crate::sample::Sample) at a pitch-driven position — this reads an open-ended stream
//! pulled on demand from a callback, keeping just enough recent input in a small ring to feed the
//! windowed-sinc kernel. It applies the same [`InstrumentResampleMode`] set as a voice (resolved
//! against a non-PSG signal — a finished mix has no PSG/sampled distinction) by reusing the shared
//! [`effective_gather`]/[`sinc_fc`] resolution and the one [`resample_sinc`] gather; the only new
//! code is the ring-staging of the tap window, mirroring [`gather_sinc`](super::gather_sinc)'s edge
//! path.

use super::{
    effective_gather, mode_half_taps, resample_sinc, sinc_fc, tap_window, EffectiveGather,
    ResampleTables, GATHER_BUF_LEN, MAX_HALF_TAPS,
};
use crate::sample::InstrumentResampleMode;

/// Recent-input ring length: wider than the widest possible tap window so the oldest tap a gather
/// reads is never overwritten before it is read.
const RING: usize = GATHER_BUF_LEN + 2;

/// A streaming, fixed-ratio stereo resampler driven by an [`InstrumentResampleMode`].
///
/// The read position advances by `step = in_rate / out_rate` input samples per output sample;
/// input is pulled lazily so the caller only synthesizes a mixer sample when one is actually
/// consumed.
pub struct StreamResampler {
    /// The resolved gather for the current mode (signal treated as non-PSG).
    gather: EffectiveGather,
    /// Sinc tables when the gather is a sinc variant; `None` for nearest / linear.
    tables: Option<ResampleTables>,
    /// Anti-image / anti-alias cutoff in cycles/input-sample (only used by the sinc gather).
    fc: f64,
    /// Input samples advanced per output sample.
    step: f64,
    /// Absolute input position of the next output sample.
    pos: f64,
    /// Count of inputs pushed so far (the absolute index of the next push).
    loaded: i64,
    ring_l: [f32; RING],
    ring_r: [f32; RING],
}

impl Default for StreamResampler {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamResampler {
    /// An idle nearest-neighbour resampler at unity ratio. Call [`Self::set`] before use.
    pub fn new() -> Self {
        Self {
            gather: EffectiveGather::Nearest,
            tables: None,
            fc: 0.5,
            step: 1.0,
            pos: 0.0,
            loaded: 0,
            ring_l: [0.0; RING],
            ring_r: [0.0; RING],
        }
    }

    /// Configures the conversion ratio and resampling mode. The mode is resolved exactly like a
    /// (non-PSG) voice's: the sinc cutoff is the input Nyquist when upsampling and the output
    /// Nyquist when downsampling, lowered further by the crunchy mode's cutoff slider (see
    /// [`sinc_fc`]). Cheap to call every block — only rebuilds the sinc tables on a `half_taps`
    /// change and never disturbs the running position.
    pub fn set(&mut self, in_rate: f64, out_rate: f64, mode: InstrumentResampleMode) {
        self.step = if out_rate > 0.0 {
            in_rate / out_rate
        } else {
            1.0
        };
        // A finished stereo bus has no PSG/sampled split, so resolve against a non-PSG signal.
        self.gather = effective_gather(mode, false);
        if let EffectiveGather::Sinc {
            step_mode,
            cutoff_hz,
        } = self.gather
        {
            let inv_out_rate = if out_rate > 0.0 { 1.0 / out_rate } else { 0.0 };
            self.fc = sinc_fc(self.step, inv_out_rate, step_mode, cutoff_hz);
        }
        match mode_half_taps(mode) {
            Some(p) => {
                let p = p.clamp(1, MAX_HALF_TAPS);
                if self.tables.as_ref().map(|t| t.half_taps) != Some(p) {
                    self.tables = Some(ResampleTables::new(p));
                }
            }
            None => self.tables = None,
        }
    }

    /// Clears the ring and read position (used when the mixer is (re)enabled, to start clean).
    pub fn reset(&mut self) {
        self.pos = 0.0;
        self.loaded = 0;
        self.ring_l = [0.0; RING];
        self.ring_r = [0.0; RING];
    }

    #[inline]
    fn push(&mut self, l: f64, r: f64) {
        let slot = (self.loaded as usize) % RING;
        self.ring_l[slot] = l as f32;
        self.ring_r[slot] = r as f32;
        self.loaded += 1;
    }

    /// Reads the input sample at absolute index `k` from the ring (zero before the stream start).
    #[inline]
    fn at(ring: &[f32; RING], k: i64) -> f32 {
        if k < 0 {
            0.0
        } else {
            ring[(k as usize) % RING]
        }
    }

    /// Pulls input from `next_in` until the ring holds the sample at absolute index `k`.
    #[inline]
    fn fill_to(&mut self, k: i64, next_in: &mut impl FnMut() -> (f64, f64)) {
        while self.loaded <= k {
            let (l, r) = next_in();
            self.push(l, r);
        }
    }

    /// Produces one output stereo sample, pulling mixer-rate input from `next_in` as the read
    /// window requires it.
    pub fn next(&mut self, next_in: &mut impl FnMut() -> (f64, f64)) -> (f64, f64) {
        let out = match self.gather {
            EffectiveGather::Nearest => {
                // Zero-order hold: the most recent input at or before `pos`.
                let idx = self.pos.floor() as i64;
                self.fill_to(idx, next_in);
                (
                    f64::from(Self::at(&self.ring_l, idx)),
                    f64::from(Self::at(&self.ring_r, idx)),
                )
            }
            EffectiveGather::Linear => {
                let i = self.pos.floor() as i64;
                let frac = self.pos - i as f64;
                self.fill_to(i + 1, next_in);
                let lerp = |ring: &[f32; RING]| -> f64 {
                    let a = f64::from(Self::at(ring, i));
                    let b = f64::from(Self::at(ring, i + 1));
                    a + (b - a) * frac
                };
                (lerp(&self.ring_l), lerp(&self.ring_r))
            }
            EffectiveGather::Sinc { step_mode, .. } => {
                // Clone the (cheap, half-width-only) table handle so the pull loop can borrow
                // `self` mutably while staging the window.
                let tables = self.tables.clone().expect("sinc gather has tables");
                let (k_lo, k_hi) = tap_window(&tables, self.pos);
                self.fill_to(k_hi, next_in);
                let n = (k_hi - k_lo + 1) as usize;
                let mut buf_l = [0.0f32; GATHER_BUF_LEN];
                let mut buf_r = [0.0f32; GATHER_BUF_LEN];
                for (j, (sl, sr)) in buf_l[..n].iter_mut().zip(&mut buf_r[..n]).enumerate() {
                    let k = k_lo + j as i64;
                    *sl = Self::at(&self.ring_l, k);
                    *sr = Self::at(&self.ring_r, k);
                }
                (
                    resample_sinc(&tables, &buf_l[..n], self.pos, self.fc, step_mode),
                    resample_sinc(&tables, &buf_r[..n], self.pos, self.fc, step_mode),
                )
            }
        };
        self.pos += self.step;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nearest-neighbour (zero-order hold) repeats each input across the upsample ratio.
    #[test]
    fn nearest_holds_input_samples() {
        let mut rs = StreamResampler::new();
        rs.set(1.0, 4.0, InstrumentResampleMode::NearestNeighbor); // 4× upsample, step = 0.25
        let inputs = [1.0f64, 2.0, 3.0];
        let mut it = inputs.into_iter();
        let mut pull = move || {
            let v = it.next().unwrap_or(0.0);
            (v, -v)
        };
        // pos = 0, .25, .5, .75 → floor 0 → input[0]; then 1.0,1.25,.. → input[1]; ...
        let got: Vec<(f64, f64)> = (0..12).map(|_| rs.next(&mut pull)).collect();
        for (i, &(l, r)) in got.iter().enumerate() {
            let expected = inputs[i / 4];
            assert_eq!(l, expected, "sample {i}");
            assert_eq!(r, -expected, "sample {i} right");
        }
    }

    /// Linear interpolation of a ramp stream lands on the expected mid-points.
    #[test]
    fn linear_interpolates_between_inputs() {
        let mut rs = StreamResampler::new();
        rs.set(1.0, 2.0, InstrumentResampleMode::Linear); // 2× upsample, step = 0.5
        let inputs = [0.0f64, 2.0, 4.0];
        let mut it = inputs.into_iter();
        let mut pull = move || {
            let v = it.next().unwrap_or(4.0);
            (v, 0.0)
        };
        // pos = 0, .5, 1, 1.5, 2 → 0, 1, 2, 3, 4 (left channel).
        let got: Vec<f64> = (0..5).map(|_| rs.next(&mut pull).0).collect();
        for (i, &l) in got.iter().enumerate() {
            assert!((l - i as f64).abs() < 1e-9, "sample {i}: got {l}");
        }
    }

    /// Both sinc modes reconstruct a DC stream as flat DC at unity gain once the window fills.
    #[test]
    fn sinc_reconstructs_dc_at_unity_gain() {
        for mode in [
            InstrumentResampleMode::SincSampleNyquist { half_taps: 16 },
            InstrumentResampleMode::SincOutputNyquist {
                half_taps: 16,
                psg_cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
                sampler_cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
            },
        ] {
            let mut rs = StreamResampler::new();
            rs.set(8000.0, 48000.0, mode);
            let mut pull = || (1.0, 0.5);
            let mut last = (0.0, 0.0);
            for _ in 0..2000 {
                last = rs.next(&mut pull);
            }
            assert!((last.0 - 1.0).abs() < 1e-3, "left DC gain off: {}", last.0);
            assert!((last.1 - 0.5).abs() < 1e-3, "right DC gain off: {}", last.1);
        }
    }
}

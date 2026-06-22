//! [`StreamResampler`]: a continuous, fixed-ratio resampler for a *stream* of stereo samples
//! (the intermediate mixer's bus → the output rate).
//!
//! Unlike the voice gather in [`source`](super::source) — which reads a finite, loop-mapped
//! [`Sample`](crate::sample::Sample) at a pitch-driven position — this reads an open-ended stream
//! pulled on demand from a callback, keeping just enough recent input in a small ring to feed the
//! windowed-sinc kernel. Both share the one gather, [`resample_sinc`]: the only new code here is
//! the ring-staging of the tap window, mirroring [`gather_sinc`](super::gather_sinc)'s edge path.

use super::{resample_sinc, tap_window, ResampleTables, GATHER_BUF_LEN, MAX_HALF_TAPS};

/// Recent-input ring length: wider than the widest possible tap window so the oldest tap a gather
/// reads is never overwritten before it is read.
const RING: usize = GATHER_BUF_LEN + 2;

/// A streaming, fixed-ratio stereo resampler.
///
/// `half_taps = None` selects zero-order hold (keep the input stairstep); `Some(p)` selects the
/// windowed-sinc reconstruction with support half-width `p`. The read position advances by
/// `step = in_rate / out_rate` input samples per output sample; input is pulled lazily so the
/// caller only synthesizes a mixer sample when one is actually consumed.
pub struct StreamResampler {
    /// Sinc tables (and the half-width); `None` ⇒ nearest / zero-order hold.
    tables: Option<ResampleTables>,
    /// Anti-image / anti-alias cutoff in cycles/input-sample (source Nyquist = 0.5).
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
    /// An idle resampler at unity ratio. Call [`Self::set`] before use.
    pub fn new() -> Self {
        Self {
            tables: None,
            fc: 0.5,
            step: 1.0,
            pos: 0.0,
            loaded: 0,
            ring_l: [0.0; RING],
            ring_r: [0.0; RING],
        }
    }

    /// Configures the conversion ratio and kernel. `half_taps` `None` is zero-order hold; `Some`
    /// builds (shares) the sinc tables. The cutoff is the input Nyquist when upsampling and the
    /// output Nyquist when downsampling (`fc = 0.5 / max(step, 1)`), matching the voice gather's
    /// impulse-mode reconstruction. Cheap to call every block — only rebuilds tables on a
    /// `half_taps` change and never disturbs the running position.
    pub fn set(&mut self, in_rate: f64, out_rate: f64, half_taps: Option<usize>) {
        self.step = if out_rate > 0.0 {
            in_rate / out_rate
        } else {
            1.0
        };
        self.fc = 0.5 / self.step.max(1.0);
        match half_taps {
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

    /// Produces one output stereo sample, pulling mixer-rate input from `next_in` as the read
    /// window requires it.
    pub fn next(&mut self, next_in: &mut impl FnMut() -> (f64, f64)) -> (f64, f64) {
        // Clone the (cheap, half-width-only) table handle so the pull loop can borrow `self`
        // mutably while staging the window.
        let tables = self.tables.clone();
        let out = match &tables {
            None => {
                // Zero-order hold: the most recent input at or before `pos`.
                let idx = self.pos.floor() as i64;
                while self.loaded <= idx {
                    let (l, r) = next_in();
                    self.push(l, r);
                }
                (
                    f64::from(Self::at(&self.ring_l, idx)),
                    f64::from(Self::at(&self.ring_r, idx)),
                )
            }
            Some(tables) => {
                // Stage the exact tap window from the ring, then run the shared gather per channel.
                let (k_lo, k_hi) = tap_window(tables, self.pos);
                while self.loaded <= k_hi {
                    let (l, r) = next_in();
                    self.push(l, r);
                }
                let n = (k_hi - k_lo + 1) as usize;
                let mut buf_l = [0.0f32; GATHER_BUF_LEN];
                let mut buf_r = [0.0f32; GATHER_BUF_LEN];
                for (j, (sl, sr)) in buf_l[..n].iter_mut().zip(&mut buf_r[..n]).enumerate() {
                    let k = k_lo + j as i64;
                    *sl = Self::at(&self.ring_l, k);
                    *sr = Self::at(&self.ring_r, k);
                }
                (
                    resample_sinc(tables, &buf_l[..n], self.pos, self.fc, false),
                    resample_sinc(tables, &buf_r[..n], self.pos, self.fc, false),
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

    /// A nearest (zero-order-hold) stream just repeats each input across the upsample ratio.
    #[test]
    fn nearest_holds_input_samples() {
        let mut rs = StreamResampler::new();
        rs.set(1.0, 4.0, None); // 4× upsample, step = 0.25
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

    /// Sinc reconstruction of a DC stream is flat DC once the window has filled (unity gain).
    #[test]
    fn sinc_reconstructs_dc_at_unity_gain() {
        let mut rs = StreamResampler::new();
        rs.set(8000.0, 48000.0, Some(16));
        let mut pull = || (1.0, 0.5);
        let mut last = (0.0, 0.0);
        for _ in 0..2000 {
            last = rs.next(&mut pull);
        }
        assert!((last.0 - 1.0).abs() < 1e-3, "left DC gain off: {}", last.0);
        assert!((last.1 - 0.5).abs() < 1e-3, "right DC gain off: {}", last.1);
    }
}

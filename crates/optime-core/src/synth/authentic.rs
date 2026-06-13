//! Per-voice state for [`ResampleMode::Authentic`](crate::sample::ResampleMode::Authentic):
//! reproduces the device's fixed-rate hardware output chain
//! ([`HardwareChain`](crate::devices::HardwareChain)) sample by sample.
//!
//! ```text
//! source ──(linear interp)──► mixer_hz ──(NN hold)──► dac_hz ──(windowed-sinc SRC)──► output
//! PSG    ──────────────(nearest-neighbour)──────────► dac_hz ──(windowed-sinc SRC)──► output
//! ```
//!
//! The DAC-rate stream is synthesized on demand into a small ring buffer, always far enough
//! ahead to cover the final reconstruction kernel's tap window; the chain stages before it are
//! exact integer-rate processes, so no second kernel is needed.

use crate::devices::HardwareChain;
use crate::resample::{resample_sinc, tap_window, ResampleTables, MAX_HALF_TAPS};
use crate::sample::Sample;

/// Ring capacity in DAC-rate samples: one full tap window at the widest supported kernel.
const RING_LEN: usize = 2 * MAX_HALF_TAPS + 2;

/// Per-call chain parameters, resolved by the owning voice.
pub(super) struct ChainParams {
    /// The voice's *resolved* chain: `mixer_hz` must already be `None` for PSG voices and
    /// mixer-less devices (straight nearest-neighbour at the DAC rate).
    pub chain: HardwareChain,
    /// Extra low-pass on the final reconstruction (the Authentic mode's cutoff slider), Hz.
    pub cutoff_hz: u32,
    /// Source samples per source-rate second of playback (the voice's pitch ratio).
    pub freq_ratio: f64,
    /// Reciprocal of the output sample rate.
    pub inv_sample_rate: f64,
}

/// The running chain state of one voice.
#[derive(Clone)]
pub(super) struct AuthenticState {
    /// Fractional read position on the DAC-rate grid, in DAC samples.
    t_dac: f64,
    /// Next DAC-grid index to synthesize into the ring.
    next_n: i64,
    /// Synthesized DAC-rate samples; index `n` lives at `n mod RING_LEN`.
    ring: [f32; RING_LEN],
    /// Source read position, in source samples.
    src_pos: f64,
    /// Mixer-stage accumulator, in mixer samples (a new mixer sample is drawn when it
    /// crosses 1). Starts at 1 so the very first DAC sample draws one.
    mix_acc: f64,
    /// The mixer-stage output currently held by the NN upsampler.
    mix_hold: f64,
}

impl AuthenticState {
    pub fn new() -> Self {
        AuthenticState {
            // One DAC sample behind grid index 0, so the first advance of a ratio-1 output
            // lands exactly on x[0] — the same advance-then-read convention as the standard
            // resample path.
            t_dac: -1.0,
            next_n: 0,
            ring: [0.0; RING_LEN],
            src_pos: 0.0,
            mix_acc: 1.0,
            mix_hold: 0.0,
        }
    }

    /// Whether the chain has not produced anything yet (used to seed the source position when
    /// the user switches an already-sounding voice into Authentic mode).
    pub fn is_fresh(&self) -> bool {
        self.next_n == 0
    }

    /// Starts the source stage at `pos` source samples.
    pub fn seed(&mut self, pos: f64) {
        self.src_pos = pos;
    }

    /// The source read position, mirrored into
    /// [`SampleInstrument::sample_t`](super::SampleInstrument::sample_t) so one-shot-end
    /// detection keeps working.
    pub fn src_pos(&self) -> f64 {
        self.src_pos
    }

    /// Advances the chain by one output sample and returns the reconstructed (unscaled) value.
    ///
    /// With `gather` false the final reconstruction is skipped (returning 0) for fully
    /// attenuated voices while the chain still advances.
    pub fn advance(
        &mut self,
        sample: &Sample,
        params: &ChainParams,
        tables: &ResampleTables,
        gather: bool,
    ) -> f64 {
        let &ChainParams {
            chain: HardwareChain { mixer_hz, dac_hz },
            cutoff_hz,
            freq_ratio,
            inv_sample_rate,
        } = params;
        // DAC samples per output sample.
        let r_dac = dac_hz * inv_sample_rate;
        self.t_dac += r_dac;
        let (k_lo, k_hi) = tap_window(tables, self.t_dac);

        // Synthesize the chain forward to cover the window's right edge.
        while self.next_n <= k_hi {
            let v = match mixer_hz {
                // PSG / mixer-less: nearest-neighbour straight at the DAC rate.
                None => {
                    self.src_pos += freq_ratio * sample.sample_rate / dac_hz;
                    self.fold_src(sample);
                    read_source(sample, self.src_pos.floor() as i64)
                }
                // Sampled voice: linear-interpolate to the mixer rate, NN-hold to the DAC rate.
                Some(mixer_hz) => {
                    self.mix_acc += mixer_hz / dac_hz;
                    while self.mix_acc >= 1.0 {
                        self.mix_acc -= 1.0;
                        self.src_pos += freq_ratio * sample.sample_rate / mixer_hz;
                        self.fold_src(sample);
                        let i = self.src_pos.floor() as i64;
                        let frac = self.src_pos - i as f64;
                        let a = read_source(sample, i);
                        let b = read_source(sample, i + 1);
                        self.mix_hold = a + (b - a) * frac;
                    }
                    self.mix_hold
                }
            };
            self.ring[self.next_n.rem_euclid(RING_LEN as i64) as usize] = v as f32;
            self.next_n += 1;
        }

        if !gather {
            return 0.0;
        }

        // Stage the tap window (zeros before the note started) and reconstruct: a clean
        // band-limited interpolation of the DAC-rate stream ("proper" SRC), anti-aliased when
        // the output rate is below the DAC rate.
        let n = (k_hi - k_lo + 1) as usize;
        let mut buf = [0.0f32; RING_LEN];
        for (slot, k) in buf[..n].iter_mut().zip(k_lo..) {
            *slot = if k < 0 {
                0.0
            } else {
                self.ring[k.rem_euclid(RING_LEN as i64) as usize]
            };
        }
        // fc in cycles per DAC sample: the reconstruction band limit, the output Nyquist when
        // downsampling, and the user's extra cutoff, whichever is lowest.
        let fc = (0.5 / r_dac).min(0.5).min(f64::from(cutoff_hz) / dac_hz);
        resample_sinc(tables, &buf[..n], self.t_dac, fc, false)
    }

    /// Folds the source position back into the loop body once playback wraps (keeps precision
    /// bounded over arbitrarily long notes, mirroring `SampleInstrument::advance`).
    fn fold_src(&mut self, sample: &Sample) {
        let data_len = sample.data.len() as f64;
        let loop_len = data_len - sample.loop_point as f64;
        if sample.looping && loop_len > 0.0 && self.src_pos >= data_len {
            let lp = sample.loop_point as f64;
            self.src_pos = (self.src_pos - lp) % loop_len + lp;
        }
    }
}

/// Loop-aware source read: maps positions past the end into the loop body, zeros outside.
fn read_source(sample: &Sample, mut t: i64) -> f64 {
    let data_len = sample.data.len() as i64;
    let loop_len = data_len - sample.loop_point;
    if t >= data_len && sample.looping {
        if loop_len <= 0 {
            return 0.0;
        }
        t = (t - sample.loop_point).rem_euclid(loop_len) + sample.loop_point;
    }
    if t >= 0 && t < data_len {
        f64::from(sample.data[t as usize])
    } else {
        0.0
    }
}

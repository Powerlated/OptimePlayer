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

use super::gather::{gather_sinc, GatherSource};
use crate::devices::HardwareChain;
use crate::resample::{resample_sinc, tap_window, ResampleTables, MAX_HALF_TAPS};
use crate::sample::Sample;

/// Ring capacity in DAC-rate samples: one full tap window at the widest supported kernel.
const RING_LEN: usize = 2 * MAX_HALF_TAPS + 2;

/// How a sampled voice's source is taken through the software-mixer → DAC stages of the chain.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Reconstruction {
    /// GBA Authentic: linear interpolation to the mixer rate, nearest-neighbour hold to the DAC
    /// rate — exactly what the MP2K software mixer and the DAC do.
    HardwareHold,
    /// GBA Crunchy Authentic: band-limited sinc reconstruction to the mixer rate, then a
    /// band-limited zero-order hold (the BLEP step kernel) to the DAC rate.
    Crunchy,
}

/// Per-call chain parameters, resolved by the owning voice.
pub(super) struct ChainParams {
    /// The voice's *resolved* chain: `mixer_hz` must already be `None` for PSG voices and
    /// mixer-less devices (straight nearest-neighbour at the DAC rate).
    pub chain: HardwareChain,
    /// Extra low-pass on the final reconstruction (the cutoff slider), Hz.
    pub cutoff_hz: u32,
    /// How the sampled-voice source → mixer → DAC stages are reconstructed.
    pub recon: Reconstruction,
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
    /// crosses 1). Starts at 1 so the very first DAC sample draws one. ([`Reconstruction::HardwareHold`].)
    mix_acc: f64,
    /// The mixer-stage output currently held by the NN upsampler. ([`Reconstruction::HardwareHold`].)
    mix_hold: f64,
    /// Whether the source has wrapped its loop at least once (selects the periodic gather mapping
    /// in [`Reconstruction::Crunchy`]).
    wrapped: bool,
    /// Synthesized mixer-rate samples for [`Reconstruction::Crunchy`]; index `m` lives at
    /// `m mod RING_LEN`.
    mixer_ring: [f32; RING_LEN],
    /// Next mixer-grid index to synthesize into [`Self::mixer_ring`].
    next_mixer_n: i64,
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
            wrapped: false,
            mixer_ring: [0.0; RING_LEN],
            next_mixer_n: 0,
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
            recon,
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
                // Sampled voice: hardware hold, or the crunchy reconstruction.
                Some(mixer_hz) => match recon {
                    Reconstruction::HardwareHold => {
                        // Linear-interpolate to the mixer rate, NN-hold to the DAC rate.
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
                    Reconstruction::Crunchy => {
                        self.synth_dac_crunchy(sample, mixer_hz, dac_hz, freq_ratio, tables)
                    }
                },
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

    /// Produces one DAC-rate sample (at grid index `self.next_n`) for the crunchy reconstruction:
    /// the source is band-limited-sinc resampled onto the mixer grid (filling [`Self::mixer_ring`]
    /// as needed), then a band-limited zero-order hold (BLEP step kernel) reads that ring at the
    /// DAC sample's mixer-grid position.
    fn synth_dac_crunchy(
        &mut self,
        sample: &Sample,
        mixer_hz: f64,
        dac_hz: f64,
        freq_ratio: f64,
        tables: &ResampleTables,
    ) -> f64 {
        // Mixer samples per DAC sample (< 1, an upsampling stage) and source samples per mixer
        // sample (the pitch ratio carried onto the mixer grid).
        let r_md = mixer_hz / dac_hz;
        let r_sm = freq_ratio * sample.sample_rate / mixer_hz;
        // This DAC sample sits at mixer-grid position `n · r_md`.
        let t_mixer = self.next_n as f64 * r_md;
        let (m_lo, m_hi) = tap_window(tables, t_mixer);
        // After a mode switch the mixer cursor can be far behind the window; skip the gap so the
        // fill is always bounded to one tap window (a one-window transient on the switch).
        if self.next_mixer_n < m_lo {
            self.next_mixer_n = m_lo;
        }
        // Synthesize mixer samples up to the window's right edge by reconstructing the source.
        let fc_src = (0.5 / r_sm).min(0.5);
        while self.next_mixer_n <= m_hi {
            self.src_pos += r_sm;
            self.fold_src(sample);
            let v = self.gather_source(sample, tables, fc_src);
            self.mixer_ring[self.next_mixer_n.rem_euclid(RING_LEN as i64) as usize] = v as f32;
            self.next_mixer_n += 1;
        }
        // Band-limited zero-order hold: stage the mixer window (zeros before the start) and gather
        // it with the BLEP step kernel band-limited to the DAC Nyquist.
        let n = (m_hi - m_lo + 1) as usize;
        let mut buf = [0.0f32; RING_LEN];
        for (slot, m) in buf[..n].iter_mut().zip(m_lo..) {
            *slot = if m < 0 {
                0.0
            } else {
                self.mixer_ring[m.rem_euclid(RING_LEN as i64) as usize]
            };
        }
        resample_sinc(tables, &buf[..n], t_mixer, 0.5 / r_md, true)
    }

    /// A clean band-limited sinc read of the looping source at the current [`Self::src_pos`].
    fn gather_source(&self, sample: &Sample, tables: &ResampleTables, fc: f64) -> f64 {
        let data_len = sample.data.len() as i64;
        let src = GatherSource {
            data: &sample.data,
            looping: sample.looping,
            loop_point: sample.loop_point,
            loop_len: data_len - sample.loop_point,
            wrapped: self.wrapped,
        };
        gather_sinc(&src, tables, self.src_pos, fc, false)
    }

    /// Folds the source position back into the loop body once playback wraps (keeps precision
    /// bounded over arbitrarily long notes, mirroring `SampleInstrument::advance`).
    fn fold_src(&mut self, sample: &Sample) {
        let data_len = sample.data.len() as f64;
        let loop_len = data_len - sample.loop_point as f64;
        if sample.looping && loop_len > 0.0 && self.src_pos >= data_len {
            let lp = sample.loop_point as f64;
            self.src_pos = (self.src_pos - lp) % loop_len + lp;
            self.wrapped = true;
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

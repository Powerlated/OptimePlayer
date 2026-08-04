//! Continuous fixed-ratio stereo resampling, carrying the mixer bus up to the output rate. Unlike a
//! voice, which reads a finite waveform it already holds, this reads a stream that does not exist
//! yet: input arrives through a pull closure, so the ring is filled on demand to exactly the last
//! input the block's taps will reach, and the same call site can be driven a block or a sample at a
//! time without changing the output.
//!
//! Stream position is an integer plus a fraction rather than one float, so an hour of playback
//! costs no precision. The ring is a power of two sized from the resample ratio — a full block's
//! worth of input, plus the widest tap window — and grows only when a rate change demands it, since
//! growing resets the phase.

use crate::dsp::block::MAX_BLOCK;
use crate::dsp::resample::mode::{EffectiveGather, effective_gather, mode_half_taps, sinc_fc};
use crate::dsp::resample::{DefaultResampler, GATHER_BUF_LEN, MAX_HALF_TAPS, Resampler};
use crate::waveform::{InstrumentResampleMode, Sample};

fn ring_len_for(step: f32) -> usize {
    let per_block = (step.max(0.0) * MAX_BLOCK as f32).ceil() as usize;
    (GATHER_BUF_LEN + per_block + 2).next_power_of_two()
}

pub struct StreamResampler<R: Resampler = DefaultResampler> {
    gather: EffectiveGather,
    tables: Option<R::Tables>,
    fc: f32,
    step: f32,
    pos_int: i64,
    pos_frac: f32,
    loaded: i64,
    ring_l: Vec<f32>,
    ring_r: Vec<f32>,
}

impl Default for StreamResampler<DefaultResampler> {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamResampler<DefaultResampler> {
    pub fn new() -> Self {
        Self::with_resampler()
    }
}

impl<R: Resampler> StreamResampler<R> {
    pub fn with_resampler() -> Self {
        let ring = ring_len_for(1.0);
        Self {
            gather: EffectiveGather::Nearest,
            tables: None,
            fc: 0.5,
            step: 1.0,
            pos_int: 0,
            pos_frac: 0.0,
            loaded: 0,
            ring_l: vec![0.0; ring],
            ring_r: vec![0.0; ring],
        }
    }

    pub fn set(&mut self, in_rate: f32, out_rate: f32, mode: InstrumentResampleMode) {
        self.step = if out_rate > 0.0 {
            in_rate / out_rate
        } else {
            1.0
        };
        let needed = ring_len_for(self.step);
        if needed > self.ring_l.len() {
            self.ring_l = vec![0.0; needed];
            self.ring_r = vec![0.0; needed];
            self.pos_int = 0;
            self.pos_frac = 0.0;
            self.loaded = 0;
        }
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
                if self.tables.as_ref().map(R::half_taps) != Some(p) {
                    self.tables = Some(R::tables(p));
                }
            }
            None => self.tables = None,
        }
    }

    pub fn reset(&mut self) {
        self.pos_int = 0;
        self.pos_frac = 0.0;
        self.loaded = 0;
        self.ring_l.fill(0.0);
        self.ring_r.fill(0.0);
    }

    #[inline]
    fn at(ring: &[f32], k: i64) -> f32 {
        if k < 0 {
            0.0
        } else {
            ring[(k as usize) & (ring.len() - 1)]
        }
    }

    fn fill_to(&mut self, k: i64, fill_in: &mut impl FnMut(&mut [Sample], &mut [Sample])) {
        let wanted = k + 1 - self.loaded;
        if wanted <= 0 {
            return;
        }
        let len = self.ring_l.len();
        let start = self.loaded as usize & (len - 1);
        let n = wanted as usize;
        debug_assert!(n <= len, "pull larger than the ring");
        let first = (len - start).min(n);
        let (head_l, tail_l) = self.ring_l.split_at_mut(start);
        let (head_r, tail_r) = self.ring_r.split_at_mut(start);
        fill_in(&mut tail_l[..first], &mut tail_r[..first]);
        if n > first {
            fill_in(&mut head_l[..n - first], &mut head_r[..n - first]);
        }
        self.loaded += wanted;
    }

    #[inline]
    fn advance(&mut self) {
        self.pos_frac += self.step;
        let carry = self.pos_frac.floor();
        self.pos_int += carry as i64;
        self.pos_frac -= carry;
    }

    fn last_input_needed(&self, n: usize) -> i64 {
        let (mut pos_int, mut pos_frac) = (self.pos_int, self.pos_frac);
        let mut highest = self.pos_int;
        for _ in 0..n {
            let needed = match self.gather {
                EffectiveGather::Nearest => pos_int,
                EffectiveGather::Linear => pos_int + 1,
                EffectiveGather::Sinc { .. } => {
                    let tables = self.tables.as_ref().expect("sinc gather has tables");
                    let half_taps = R::half_taps(tables);
                    let syn_pos = half_taps as f32 + pos_frac;
                    let (syn_lo, syn_hi) = R::tap_window(tables, syn_pos);
                    pos_int - half_taps as i64 + (syn_hi - syn_lo)
                }
            };
            highest = highest.max(needed);
            pos_frac += self.step;
            let carry = pos_frac.floor();
            pos_int += carry as i64;
            pos_frac -= carry;
        }
        highest
    }

    pub fn process(
        &mut self,
        out_l: &mut [Sample],
        out_r: &mut [Sample],
        fill_in: &mut impl FnMut(&mut [Sample], &mut [Sample]),
    ) {
        debug_assert_eq!(out_l.len(), out_r.len());
        for (l, r) in out_l.chunks_mut(MAX_BLOCK).zip(out_r.chunks_mut(MAX_BLOCK)) {
            self.process_block(l, r, fill_in);
        }
    }

    fn process_block(
        &mut self,
        out_l: &mut [Sample],
        out_r: &mut [Sample],
        fill_in: &mut impl FnMut(&mut [Sample], &mut [Sample]),
    ) {
        if out_l.is_empty() {
            return;
        }
        self.fill_to(self.last_input_needed(out_l.len()), fill_in);

        match self.gather {
            EffectiveGather::Nearest => {
                for (l, r) in out_l.iter_mut().zip(out_r.iter_mut()) {
                    let idx = self.pos_int;
                    *l = Self::at(&self.ring_l, idx);
                    *r = Self::at(&self.ring_r, idx);
                    self.advance();
                }
            }
            EffectiveGather::Linear => {
                for (l, r) in out_l.iter_mut().zip(out_r.iter_mut()) {
                    let i = self.pos_int;
                    let frac = self.pos_frac;
                    let lerp = |ring: &[f32]| -> Sample {
                        let a = Self::at(ring, i);
                        let b = Self::at(ring, i + 1);
                        a + (b - a) * frac
                    };
                    *l = lerp(&self.ring_l);
                    *r = lerp(&self.ring_r);
                    self.advance();
                }
            }
            EffectiveGather::Sinc { step_mode, .. } => {
                let tables = self.tables.clone().expect("sinc gather has tables");
                let p = R::half_taps(&tables) as i64;
                let fc = self.fc;
                let mut buf_l = [0.0f32; GATHER_BUF_LEN];
                let mut buf_r = [0.0f32; GATHER_BUF_LEN];
                for (l, r) in out_l.iter_mut().zip(out_r.iter_mut()) {
                    let syn_pos = p as f32 + self.pos_frac;
                    let (syn_lo, syn_hi) = R::tap_window(&tables, syn_pos);
                    debug_assert_eq!(syn_lo, 0);
                    let n = (syn_hi - syn_lo + 1) as usize;
                    let k_lo = self.pos_int - p;
                    for (j, (sl, sr)) in buf_l[..n].iter_mut().zip(&mut buf_r[..n]).enumerate() {
                        let k = k_lo + j as i64;
                        *sl = Self::at(&self.ring_l, k);
                        *sr = Self::at(&self.ring_r, k);
                    }
                    *l = R::resample(&tables, &buf_l[..n], syn_pos, fc, step_mode);
                    *r = R::resample(&tables, &buf_r[..n], syn_pos, fc, step_mode);
                    self.advance();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::resample::MAX_HALF_TAPS;
    use crate::synth::WaveformSynthesizer;

    struct CenterTapOnly;

    impl Resampler for CenterTapOnly {
        type Tables = usize;

        fn tables(half_taps: usize) -> usize {
            half_taps.clamp(1, MAX_HALF_TAPS)
        }

        fn half_taps(tables: &usize) -> usize {
            *tables
        }

        fn tap_window(tables: &usize, pos: f32) -> (i64, i64) {
            let p = *tables as f32;
            ((pos - p).floor() as i64, (pos + p).ceil() as i64)
        }

        fn resample(tables: &usize, src: &[f32], pos: f32, _: f32, _: bool) -> Sample {
            let (k_lo, _) = Self::tap_window(tables, pos);
            src[(pos.round() as i64 - k_lo) as usize]
        }
    }

    #[test]
    fn a_second_implementation_is_selectable_at_compile_time() {
        let mode = InstrumentResampleMode::SincSampleNyquist { half_taps: 8 };
        let mut rs = StreamResampler::<CenterTapOnly>::with_resampler();
        rs.set(1.0, 2.0, mode);
        let inputs = [0.0f32, 2.0, 4.0];
        let mut pull = pull_list(&inputs, 4.0);
        let (mut got_l, mut got_r) = ([0.0f32; 6], [0.0f32; 6]);
        rs.process(&mut got_l, &mut got_r, &mut pull);
        for (i, &l) in got_l.iter().enumerate() {
            let nearest_index = (i as f32 * 0.5).round() as usize;
            let nearest = inputs.get(nearest_index).copied().unwrap_or(4.0);
            assert_eq!(l, nearest, "sample {i}");
        }

        let synth = WaveformSynthesizer::<CenterTapOnly>::with_resampler(44_100.0, 4);
        assert_eq!(synth.voice_count(), 4);
    }

    fn pull_list(inputs: &[f32], tail: f32) -> impl FnMut(&mut [Sample], &mut [Sample]) + use<'_> {
        let mut next = 0usize;
        move |l, r| {
            for (l, r) in l.iter_mut().zip(r.iter_mut()) {
                let v = inputs.get(next).copied().unwrap_or(tail);
                next += 1;
                (*l, *r) = (v, -v);
            }
        }
    }

    #[test]
    fn nearest_holds_input_samples() {
        let mut rs = StreamResampler::new();
        rs.set(1.0, 4.0, InstrumentResampleMode::NearestNeighbor);
        let inputs = [1.0f32, 2.0, 3.0];
        let mut pull = pull_list(&inputs, 0.0);
        let (mut got_l, mut got_r) = ([0.0f32; 12], [0.0f32; 12]);
        rs.process(&mut got_l, &mut got_r, &mut pull);
        for (i, (&l, &r)) in got_l.iter().zip(&got_r).enumerate() {
            let expected = inputs[i / 4];
            assert_eq!(l, expected, "sample {i}");
            assert_eq!(r, -expected, "sample {i} right");
        }
    }

    #[test]
    fn linear_interpolates_between_inputs() {
        let mut rs = StreamResampler::new();
        rs.set(1.0, 2.0, InstrumentResampleMode::Linear);
        let inputs = [0.0f32, 2.0, 4.0];
        let mut pull = pull_list(&inputs, 4.0);
        let (mut got_l, mut got_r) = ([0.0f32; 5], [0.0f32; 5]);
        rs.process(&mut got_l, &mut got_r, &mut pull);
        for (i, &l) in got_l.iter().enumerate() {
            assert!((l - i as f32).abs() < 1e-6, "sample {i}: got {l}");
        }
    }

    #[test]
    fn linear_psg_falls_back_to_nearest() {
        assert_eq!(
            effective_gather(InstrumentResampleMode::Linear, true),
            EffectiveGather::Nearest
        );
        assert_eq!(
            effective_gather(InstrumentResampleMode::Linear, false),
            EffectiveGather::Linear
        );
    }

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
            let mut pull = |l: &mut [Sample], r: &mut [Sample]| {
                l.fill(1.0);
                r.fill(0.5);
            };
            let (mut buf_l, mut buf_r) = ([0.0f32; 2000], [0.0f32; 2000]);
            rs.process(&mut buf_l, &mut buf_r, &mut pull);
            let last = (buf_l[1999], buf_r[1999]);
            assert!((last.0 - 1.0).abs() < 1e-3, "left DC gain off: {}", last.0);
            assert!((last.1 - 0.5).abs() < 1e-3, "right DC gain off: {}", last.1);
        }
    }

    #[test]
    fn process_is_chunk_invariant() {
        let mode = InstrumentResampleMode::SincSampleNyquist { half_taps: 16 };
        let make = || {
            let mut rs = StreamResampler::new();
            rs.set(13379.0, 32768.0, mode);
            rs
        };
        let stream = |seed: &mut u32, l: &mut [Sample], r: &mut [Sample]| {
            for (l, r) in l.iter_mut().zip(r.iter_mut()) {
                *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                *l = (*seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5;
                *r = -*l;
            }
        };

        let (mut whole_l, mut whole_r) = ([0.0f32; 500], [0.0f32; 500]);
        let mut s = 1;
        make().process(&mut whole_l, &mut whole_r, &mut |l, r| stream(&mut s, l, r));

        for size in [1, 7, 37, 256, 500] {
            let (mut chunked_l, mut chunked_r) = ([0.0f32; 500], [0.0f32; 500]);
            let mut s2 = 1;
            let mut rs = make();
            let mut pull = |l: &mut [Sample], r: &mut [Sample]| stream(&mut s2, l, r);
            for (l, r) in chunked_l.chunks_mut(size).zip(chunked_r.chunks_mut(size)) {
                rs.process(l, r, &mut pull);
            }
            assert_eq!(
                (whole_l, whole_r),
                (chunked_l, chunked_r),
                "chunk size {size}"
            );
        }
    }
}

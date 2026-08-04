use core::f32::consts::PI;
use std::simd::prelude::*;
use std::sync::OnceLock;

use crate::dsp::block::MAX_BLOCK;
use crate::waveform::{InstrumentResampleMode, Sample};

const OVERSAMPLE: usize = 512;
const TAU_MAX: usize = 64;
pub const MAX_HALF_TAPS: usize = TAU_MAX;

fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-7 {
        1.0
    } else {
        let px = PI * x;
        px.sin() / px
    }
}

fn blackman(x: f32) -> f32 {
    if x >= 1.0 {
        return 0.0;
    }
    0.42 + 0.5 * (PI * x).cos() + 0.08 * (2.0 * PI * x).cos()
}

#[inline]
fn kernel_weight(d: f32, fc: f32, p: f32) -> f32 {
    sinc(2.0 * fc * d) * blackman(d.abs() / p)
}

struct Kernels {
    sinc_int: Vec<f32>,
}

fn kernels() -> &'static Kernels {
    static K: OnceLock<Kernels> = OnceLock::new();
    K.get_or_init(|| {
        let len = TAU_MAX * OVERSAMPLE;

        let sinc_tab: Vec<f32> = (0..=len)
            .map(|k| sinc(k as f32 / OVERSAMPLE as f32))
            .collect();

        let step = 1.0 / OVERSAMPLE as f32;
        let mut sinc_int = vec![0.0f32; len + 1];
        let mut sum = 0.0f32;
        let mut comp = 0.0f32;
        for k in 1..=len {
            let trap = (sinc_tab[k - 1] + sinc_tab[k]) * 0.5 * step;
            let y = trap - comp;
            let t = sum + y;
            comp = (t - sum) - y;
            sum = t;
            sinc_int[k] = sum;
        }
        let tail = sinc_int[len];
        if tail > 1e-12 {
            let scale = 0.5 / tail;
            for v in &mut sinc_int {
                *v *= scale;
            }
        }

        Kernels { sinc_int }
    })
}

#[inline]
fn lerp(tab: &[f32], idx: f32) -> f32 {
    let i = idx as usize;
    let frac = idx - i as f32;
    let lo = tab[i];
    lo + (tab[i + 1] - lo) * frac
}

#[inline]
fn sinc_int_at(k: &Kernels, idx: f32) -> f32 {
    let mag = idx.abs();
    let v = if mag >= (k.sinc_int.len() - 1) as f32 {
        0.5
    } else {
        lerp(&k.sinc_int, mag)
    };
    if idx < 0.0 { -v } else { v }
}

#[inline]
fn sinc_int_simd(k: &Kernels, idx: Fv) -> Fv {
    let mag = idx.abs();
    let past_end = mag.simd_ge(Fv::splat((k.sinc_int.len() - 1) as f32));
    let mag = past_end.select(Fv::splat(0.0), mag);
    let i = mag.cast::<usize>();
    let frac = mag - i.cast::<f32>();
    let lo = Fv::gather_or_default(&k.sinc_int, i);
    let hi = Fv::gather_or_default(&k.sinc_int, i + Simd::splat(1));
    let v = past_end.select(Fv::splat(0.5), lo + (hi - lo) * frac);
    idx.simd_lt(Fv::splat(0.0)).select(-v, v)
}

#[inline]
fn blackman_from_cos(c: Fv) -> Fv {
    Fv::splat(0.34) + (Fv::splat(0.5) + Fv::splat(0.16) * c) * c
}

fn gather_step(k: &Kernels, src: &[f32], d0: f32, sinc_idx_step: f32, p: f32) -> (f32, f32) {
    let lane_offsets = Fv::from_array([0.0, 1.0, 2.0, 3.0]);
    let mut ph_win = Phasor::new(PI / p, d0 - 0.5);
    let (mut out, mut wsum) = (Fv::splat(0.0), Fv::splat(0.0));
    let mut carry = sinc_int_at(k, sinc_idx_step * d0);
    let mut base = d0 - 1.0;

    for chunk in src.chunks_exact(LANES) {
        let d_lo = Fv::splat(base) - lane_offsets;
        let s_lo = sinc_int_simd(k, d_lo * Fv::splat(sinc_idx_step));
        let mut s_hi = s_lo.rotate_elements_right::<1>();
        s_hi.as_mut_array()[0] = carry;
        carry = s_lo.as_array()[LANES - 1];

        let d_mid = Fv::splat(base + 0.5) - lane_offsets;
        let inside = d_mid.abs().simd_lt(Fv::splat(p));
        let w = inside.select(
            blackman_from_cos(ph_win.cos) * (s_hi - s_lo),
            Fv::splat(0.0),
        );

        out += Fv::from_slice(chunk) * w;
        wsum += w;
        base -= LANES as f32;
        ph_win.rotate();
    }
    let (mut out, mut wsum) = (out.reduce_sum(), wsum.reduce_sum());

    let done = src.len() - src.chunks_exact(LANES).remainder().len();
    let mut si_hi = carry;
    for (j, &s) in src.iter().enumerate().skip(done) {
        let d_hi = d0 - j as f32;
        let si_lo = sinc_int_at(k, sinc_idx_step * (d_hi - 1.0));
        let w = blackman((d_hi - 0.5).abs() / p) * (si_hi - si_lo);
        out += s * w;
        wsum += w;
        si_hi = si_lo;
    }
    (out, wsum)
}

const LANES: usize = 4;
type Fv = Simd<f32, LANES>;

struct Phasor {
    sin: Fv,
    cos: Fv,
    step_sin: f32,
    step_cos: f32,
}

impl Phasor {
    fn new(rate: f32, d0: f32) -> Self {
        let (mut sin, mut cos) = ([0.0; LANES], [0.0; LANES]);
        for i in 0..LANES {
            (sin[i], cos[i]) = f32::sin_cos(rate * (d0 - i as f32));
        }
        let (step_sin, step_cos) = f32::sin_cos(rate * LANES as f32);
        Self {
            sin: Fv::from_array(sin),
            cos: Fv::from_array(cos),
            step_sin,
            step_cos,
        }
    }

    #[inline]
    fn rotate(&mut self) {
        let (s, c) = (self.sin, self.cos);
        let (ss, sc) = (Fv::splat(self.step_sin), Fv::splat(self.step_cos));
        self.sin = s * sc - c * ss;
        self.cos = c * sc + s * ss;
    }
}

fn gather_impulse_simd(src: &[f32], d0: f32, fc: f32, p: f32) -> (f32, f32) {
    let a = PI * 2.0 * fc;
    let b = PI / p;
    let mut ph_sinc = Phasor::new(a, d0);
    let mut ph_win = Phasor::new(b, d0);

    let (mut out, mut wsum) = (Fv::splat(0.0), Fv::splat(0.0));
    let mut d = Fv::splat(d0) - Fv::from_array([0.0, 1.0, 2.0, 3.0]);
    for chunk in src.chunks_exact(LANES) {
        let arg = d * Fv::splat(a);
        let near_zero = arg.abs().simd_lt(Fv::splat(1e-7));
        let sinc = near_zero.select(Fv::splat(1.0), ph_sinc.sin / arg);
        let inside = d.abs().simd_lt(Fv::splat(p));
        let w = inside.select(sinc * blackman_from_cos(ph_win.cos), Fv::splat(0.0));

        out += Fv::from_slice(chunk) * w;
        wsum += w;
        d -= Fv::splat(LANES as f32);
        ph_sinc.rotate();
        ph_win.rotate();
    }
    let (mut out, mut wsum) = (out.reduce_sum(), wsum.reduce_sum());

    let done = src.len() - src.chunks_exact(LANES).remainder().len();
    for (j, &s) in src.iter().enumerate().skip(done) {
        let d = d0 - j as f32;
        let w = kernel_weight(d, fc, p);
        out += s * w;
        wsum += w;
    }
    (out, wsum)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EffectiveGather {
    Nearest,
    Linear,
    Sinc {
        step_mode: bool,
        cutoff_hz: Option<u32>,
    },
}

pub(crate) fn effective_gather(mode: InstrumentResampleMode, is_psg: bool) -> EffectiveGather {
    match mode {
        InstrumentResampleMode::NearestNeighbor => EffectiveGather::Nearest,
        InstrumentResampleMode::Linear if is_psg => EffectiveGather::Nearest,
        InstrumentResampleMode::Linear => EffectiveGather::Linear,
        InstrumentResampleMode::SincSampleNyquist { .. } => EffectiveGather::Sinc {
            step_mode: is_psg,
            cutoff_hz: None,
        },
        InstrumentResampleMode::SincOutputNyquist {
            psg_cutoff_hz,
            sampler_cutoff_hz,
            ..
        } => EffectiveGather::Sinc {
            step_mode: true,
            cutoff_hz: Some(if is_psg {
                psg_cutoff_hz
            } else {
                sampler_cutoff_hz
            }),
        },
    }
}

pub(crate) fn mode_half_taps(mode: InstrumentResampleMode) -> Option<usize> {
    match mode {
        InstrumentResampleMode::SincSampleNyquist { half_taps }
        | InstrumentResampleMode::SincOutputNyquist { half_taps, .. } => Some(half_taps),
        _ => None,
    }
}

pub(crate) fn sinc_fc(
    r: f32,
    inv_sample_rate: f32,
    step_mode: bool,
    cutoff_hz: Option<u32>,
) -> f32 {
    let mut fc = if step_mode || r > 1.0 { 0.5 / r } else { 0.5 };
    if let Some(hz) = cutoff_hz {
        fc = fc.min(hz as f32 * inv_sample_rate / r);
    }
    fc
}

pub(crate) const GATHER_BUF_LEN: usize = 2 * MAX_HALF_TAPS + 2;

#[derive(Clone)]
pub struct ResampleTables {
    pub half_taps: usize,
}

impl ResampleTables {
    pub fn new(half_taps: usize) -> Self {
        let _ = kernels();
        Self {
            half_taps: half_taps.clamp(1, MAX_HALF_TAPS),
        }
    }
}

#[inline]
pub fn tap_window(tables: &ResampleTables, pos: f32) -> (i64, i64) {
    let p = tables.half_taps as f32;
    ((pos - p).floor() as i64, (pos + p).ceil() as i64)
}

pub fn resample_sinc(
    tables: &ResampleTables,
    src: &[f32],
    pos: f32,
    fc: f32,
    step_mode: bool,
) -> Sample {
    let k = kernels();
    let (k_lo, k_hi) = tap_window(tables, pos);
    debug_assert_eq!(
        src.len() as i64,
        k_hi - k_lo + 1,
        "src must cover the tap window"
    );

    let fc = if step_mode {
        fc.max(1e-6)
    } else {
        fc.clamp(1e-6, 0.5)
    };
    let sinc_idx_step = 2.0 * fc * OVERSAMPLE as f32;

    let d0 = pos - k_lo as f32;
    let p = tables.half_taps as f32;
    let (out, wsum): (Sample, Sample) = if step_mode {
        gather_step(k, src, d0, sinc_idx_step, p)
    } else {
        gather_impulse_simd(src, d0, fc, p)
    };
    let wsum = if step_mode { wsum.abs() } else { wsum };

    if wsum > 1e-12 {
        out / wsum
    } else {
        Sample::from(src[(pos.round() as i64 - k_lo) as usize])
    }
}

#[inline]
fn loop_wrap(t: i64, loop_point: i64, loop_len: i64) -> i64 {
    (t - loop_point).rem_euclid(loop_len) + loop_point
}

pub struct GatherSource<'a> {
    pub data: &'a [f32],
    pub looping: bool,
    pub loop_point: i64,
    pub loop_len: i64,
    pub wrapped: bool,
}

#[inline]
pub fn gather_sinc(
    src: &GatherSource,
    tbl: &ResampleTables,
    pos: f32,
    fc: f32,
    step_mode: bool,
) -> Sample {
    let &GatherSource {
        data,
        looping,
        loop_point,
        loop_len,
        wrapped,
    } = src;
    let data_len = data.len() as i64;
    let (k_lo, k_hi) = tap_window(tbl, pos);
    let periodic = looping && wrapped && loop_len > 0;
    if !periodic && k_lo >= 0 && k_hi < data_len {
        let src = &data[k_lo as usize..=k_hi as usize];
        return resample_sinc(tbl, src, pos, fc, step_mode);
    }

    let n = (k_hi - k_lo + 1) as usize;
    let mut buf = [0.0f32; GATHER_BUF_LEN];
    if periodic {
        let mut idx = loop_wrap(k_lo, loop_point, loop_len);
        for slot in &mut buf[..n] {
            *slot = data[idx as usize];
            idx += 1;
            if idx == data_len {
                idx = loop_point;
            }
        }
    } else {
        for (t, slot) in (k_lo..).zip(&mut buf[..n]) {
            *slot = if (0..data_len).contains(&t) {
                data[t as usize]
            } else if t >= data_len && looping && loop_len > 0 {
                data[loop_wrap(t, loop_point, loop_len) as usize]
            } else {
                0.0
            };
        }
    }
    resample_sinc(tbl, &buf[..n], pos, fc, step_mode)
}

fn ring_len_for(step: f32) -> usize {
    let per_block = (step.max(0.0) * MAX_BLOCK as f32).ceil() as usize;
    (GATHER_BUF_LEN + per_block + 2).next_power_of_two()
}

pub struct StreamResampler {
    gather: EffectiveGather,
    tables: Option<ResampleTables>,
    fc: f32,
    step: f32,
    pos_int: i64,
    pos_frac: f32,
    loaded: i64,
    ring_l: Vec<f32>,
    ring_r: Vec<f32>,
}

impl Default for StreamResampler {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamResampler {
    pub fn new() -> Self {
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
                if self.tables.as_ref().map(|t| t.half_taps) != Some(p) {
                    self.tables = Some(ResampleTables::new(p));
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
                    let syn_pos = tables.half_taps as f32 + pos_frac;
                    let (syn_lo, syn_hi) = tap_window(tables, syn_pos);
                    pos_int - tables.half_taps as i64 + (syn_hi - syn_lo)
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
                let p = tables.half_taps as i64;
                let fc = self.fc;
                let mut buf_l = [0.0f32; GATHER_BUF_LEN];
                let mut buf_r = [0.0f32; GATHER_BUF_LEN];
                for (l, r) in out_l.iter_mut().zip(out_r.iter_mut()) {
                    let syn_pos = tables.half_taps as f32 + self.pos_frac;
                    let (syn_lo, syn_hi) = tap_window(&tables, syn_pos);
                    debug_assert_eq!(syn_lo, 0);
                    let n = (syn_hi - syn_lo + 1) as usize;
                    let k_lo = self.pos_int - p;
                    for (j, (sl, sr)) in buf_l[..n].iter_mut().zip(&mut buf_r[..n]).enumerate() {
                        let k = k_lo + j as i64;
                        *sl = Self::at(&self.ring_l, k);
                        *sr = Self::at(&self.ring_r, k);
                    }
                    *l = resample_sinc(&tables, &buf_l[..n], syn_pos, fc, step_mode);
                    *r = resample_sinc(&tables, &buf_r[..n], syn_pos, fc, step_mode);
                    self.advance();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use core::f64::consts::PI;

    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn staged(tables: &ResampleTables, pos: f64, f: impl Fn(i64) -> f64) -> Vec<f32> {
        let (k_lo, k_hi) = tap_window(tables, pos as f32);
        (k_lo..=k_hi).map(|t| f(t) as f32).collect()
    }

    #[test]
    fn sinc_int_boundary_and_symmetry() {
        let k = kernels();
        let si = |tau: f64| f64::from(sinc_int_at(k, (tau * OVERSAMPLE as f64) as f32));
        assert!(close(si(0.0), 0.0, 1e-12));
        assert!(close(si(TAU_MAX as f64 + 5.0), 0.5, 1e-6));
        assert!(close(si(-(TAU_MAX as f64) - 5.0), -0.5, 1e-6));
        for i in 1..=20 {
            let tau = TAU_MAX as f64 * i as f64 / 20.0;
            assert!(close(si(tau) + si(-tau), 0.0, 1e-12), "S({tau}) not odd");
        }
    }

    #[test]
    fn sinc_int_is_bounded() {
        let k = kernels();
        for i in 0..=2000 {
            let tau = -(TAU_MAX as f64) + 2.0 * TAU_MAX as f64 * i as f64 / 2000.0;
            let s = f64::from(sinc_int_at(k, (tau * OVERSAMPLE as f64) as f32));
            assert!((-0.6..=0.6).contains(&s), "S({tau}) = {s} out of band");
        }
    }

    #[test]
    fn blackman_folding_matches_the_direct_window() {
        for i in 0..=200 {
            let x = i as f64 / 200.0;
            let folded = f64::from(blackman_from_cos(Fv::splat((PI * x).cos() as f32))[0]);
            assert!(
                close(folded, f64::from(blackman(x as f32)), 1e-6),
                "blackman({x}): folded={folded}"
            );
        }
    }

    #[test]
    fn step_gather_matches_a_scalar_oracle() {
        let k = kernels();
        let oracle = |src: &[f32], d0: f32, sinc_idx_step: f32, p: f32| -> (f64, f64) {
            let (mut out, mut wsum) = (0.0f64, 0.0f64);
            for (j, &s) in src.iter().enumerate() {
                let d_hi = d0 - j as f32;
                let rise = sinc_int_at(k, sinc_idx_step * d_hi)
                    - sinc_int_at(k, sinc_idx_step * (d_hi - 1.0));
                let w = f64::from(blackman((d_hi - 0.5).abs() / p) * rise);
                out += f64::from(s) * w;
                wsum += w;
            }
            (out, wsum)
        };

        let mut seed = 0x1234_5678u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5
        };

        for p in [1usize, 3, 16, 32, 64] {
            for n in [2 * p, 2 * p + 1, 2 * p + 2] {
                for fc in [0.5, 0.25, 0.5 / 8.0, 1.5] {
                    for frac in [0.0, 0.13, 0.5, 0.87] {
                        let src: Vec<f32> = (0..n).map(|_| next()).collect();
                        let d0 = p as f32 + frac;
                        let step = 2.0 * fc * OVERSAMPLE as f32;
                        let (got_o, got_w) = gather_step(k, &src, d0, step, p as f32);
                        let (want_o, want_w) = oracle(&src, d0, step, p as f32);
                        let scale = want_w.abs().max(1e-3);
                        assert!(
                            close(f64::from(got_o), want_o, 1e-5 * scale),
                            "out at p={p} n={n} fc={fc} frac={frac}: {got_o} vs {want_o}"
                        );
                        assert!(
                            close(f64::from(got_w), want_w, 1e-5 * scale),
                            "wsum at p={p} n={n} fc={fc} frac={frac}: {got_w} vs {want_w}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn step_mode_preserves_dc() {
        let tables = ResampleTables::new(16);
        for fc in [0.1, 0.25, 0.5, 1.5] {
            for pos in [3.0, 7.35, 20.7] {
                let src = staged(&tables, pos, |_| 1.0);
                let out = f64::from(resample_sinc(&tables, &src, pos as f32, fc as f32, true));
                assert!(close(out, 1.0, 1e-9), "DC at fc={fc}, pos={pos}: {out}");
            }
        }
    }

    #[test]
    fn step_mode_is_a_bandlimited_step() {
        let tables = ResampleTables::new(32);
        let fc = 0.5 / 4.0;
        let step = |k: i64| if k >= 0 { 1.0_f64 } else { 0.0 };
        let at = |pos: f64| {
            f64::from(resample_sinc(
                &tables,
                &staged(&tables, pos, step),
                pos as f32,
                fc as f32,
                true,
            ))
        };

        assert!(close(at(0.0), 0.5, 0.02));
        let half_width = tables.half_taps as f64 / (2.0 * fc);
        assert!(close(at(-(half_width + 10.0)), 0.0, 1e-6));
        assert!(close(at(half_width + 10.0), 1.0, 1e-6));
        let mut prev = -1.0;
        for i in 0..=200 {
            let pos = -40.0 + 80.0 * i as f64 / 200.0;
            let v = at(pos);
            assert!(
                v > prev - 0.05,
                "non-monotone at pos={pos}: {v} after {prev}"
            );
            prev = v;
        }
    }

    #[test]
    fn impulse_mode_dc_gain() {
        let tables = ResampleTables::new(16);
        let pos = 12.37;
        let src = staged(&tables, pos, |_| 1.0);
        let out = f64::from(resample_sinc(&tables, &src, pos as f32, 0.4, false));
        assert!(close(out, 1.0, 1e-6), "DC gain = {out}");
    }

    #[test]
    fn impulse_mode_passband_signal_reconstructed() {
        let tables = ResampleTables::new(16);
        let fc = 0.45;
        let f0 = 0.05;
        let get = |k: i64| (2.0 * PI * f0 * k as f64).cos();
        for frac in [0.0, 0.25, 0.5, 0.75] {
            let pos = 32.0 + frac;
            let ideal = (2.0 * PI * f0 * pos).cos();
            let out = f64::from(resample_sinc(
                &tables,
                &staged(&tables, pos, get),
                pos as f32,
                fc as f32,
                false,
            ));
            assert!(
                close(out, ideal, 1e-3),
                "at pos={pos}: reconstructed={out}, ideal={ideal}"
            );
        }
    }

    #[test]
    fn fixed_tap_count_independent_of_ratio() {
        let p = 16usize;
        for fc in [0.5, 0.5 / 4.0, 0.5 / 8.0] {
            let pos = 100.37_f64;
            let k_lo = (pos - p as f64).floor() as i64;
            let k_hi = (pos + p as f64).ceil() as i64;
            let n = (k_lo..=k_hi).count();
            assert!(
                (2 * p..=2 * p + 2).contains(&n),
                "fc={fc}: {n} taps (expected ≈{})",
                2 * p
            );
        }
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

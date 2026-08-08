//! Precomputes the kernel instead of evaluating it, for the one cutoff that is worth precomputing.
//! Every other implementation recomputes a windowed sinc from scratch at every output sample,
//! because the cutoff moves with the resample ratio and a kernel that depends on the cutoff cannot
//! be tabulated by phase. But the cutoff only moves when the source runs *faster* than the output:
//! `sinc_fc` returns exactly 0.5 whenever a voice is being upsampled, which is nearly every voice,
//! since console sources sit at 8–32 kHz and the output at 48. At that one cutoff the kernel is a
//! function of the fractional read position alone, and the tap window always starts at
//! `floor(pos) - half_taps`, so the weights repeat.
//!
//! So this builds a filter bank: `PHASES + 1` rows of weights, row `i` being the kernel sampled at
//! fraction `i / PHASES`, each row pre-normalised to sum to one. Resampling is then two contiguous
//! dot products and a blend between neighbouring rows — no transcendentals, no gather, no phasor,
//! and the normalising division falls out because a blend of two rows that each sum to one also
//! sums to one. The bank is interpolated rather than merely quantised because the phase step alone
//! would cost tens of dB; interpolating trades a second row's worth of loads for the SNR contract.
//!
//! The cases it cannot table — a cutoff below Nyquist, and step mode, whose kernel is stretched by
//! the ratio — it hands to `ResampleImplSimd`, which is what this is a fast path *on top of* rather
//! than a replacement for. The two therefore share a tap window by construction. `LANES` is the
//! bank's dot-product width only; the delegated kernel keeps its own, because the two want opposite
//! answers — a contiguous dot product gets faster with wider vectors and a gather-bound one does
//! not, and measurement says so in both directions.

use std::simd::prelude::*;
use std::sync::OnceLock;

use super::resample_impl_00_simd::{self, ResampleImplSimd};
use super::{DEFAULT_LANES, Fv, load_partial, sinc};
use crate::dsp::resample::{MAX_HALF_TAPS, Resampler};
use crate::waveform::Sample;

const PHASES: usize = 128;
const ROW_ALIGN: usize = 8;

type Tabulated = ResampleImplSimd<DEFAULT_LANES>;

pub struct ResampleImplPolyphase<const LANES: usize = DEFAULT_LANES>;

#[derive(Clone)]
pub struct Tables {
    pub half_taps: usize,
    bank: &'static Bank,
    tabulated: resample_impl_00_simd::Tables,
}

struct Bank {
    rows: Vec<f32>,
    stride: usize,
    phases: usize,
}

fn bank(half_taps: usize) -> &'static Bank {
    static BANKS: [OnceLock<Bank>; MAX_HALF_TAPS + 1] =
        [const { OnceLock::new() }; MAX_HALF_TAPS + 1];
    BANKS[half_taps].get_or_init(|| build_bank(half_taps, PHASES))
}

fn build_bank(half_taps: usize, phases: usize) -> Bank {
    let p = half_taps as f32;
    let taps = 2 * half_taps + 2;
    let stride = taps.next_multiple_of(ROW_ALIGN);
    let mut rows = vec![0.0f32; (phases + 1) * stride];
    for phase in 0..=phases {
        let fraction = phase as f32 / phases as f32;
        let row = &mut rows[phase * stride..phase * stride + taps];
        for (j, weight) in row.iter_mut().enumerate() {
            let d = p + fraction - j as f32;
            *weight = sinc(d) * blackman_f64(f64::from(d.abs()) / f64::from(p));
        }
        let total: f32 = row.iter().sum();
        if total.abs() > 1e-12 {
            for weight in row.iter_mut() {
                *weight /= total;
            }
        }
    }
    Bank {
        rows,
        stride,
        phases,
    }
}

fn blackman_f64(x: f64) -> f32 {
    if x >= 1.0 {
        return 0.0;
    }
    let c = (core::f64::consts::PI * x).cos();
    (0.34 + (0.5 + 0.16 * c) * c) as f32
}

impl<const LANES: usize> Resampler for ResampleImplPolyphase<LANES> {
    type Tables = Tables;
    type State = ();

    fn tables(half_taps: usize) -> Tables {
        let half_taps = half_taps.clamp(1, MAX_HALF_TAPS);
        Tables {
            half_taps,
            bank: bank(half_taps),
            tabulated: Tabulated::tables(half_taps),
        }
    }

    #[inline]
    fn half_taps(tables: &Tables) -> usize {
        tables.half_taps
    }

    #[inline]
    fn tap_window(tables: &Tables, pos: f32) -> (i64, i64) {
        Tabulated::tap_window(&tables.tabulated, pos)
    }

    fn resample(
        tables: &Tables,
        state: &mut (),
        src: &[f32],
        pos: f32,
        fc: f32,
        step_mode: bool,
    ) -> Sample {
        if step_mode || fc < 0.5 {
            return Tabulated::resample(&tables.tabulated, state, src, pos, fc, step_mode);
        }
        let bank = tables.bank;
        let phase = (pos - pos.floor()) * bank.phases as f32;
        let index = phase as usize;
        let blend = phase - index as f32;
        let lower = &bank.rows[index * bank.stride..(index + 1) * bank.stride];
        let upper = &bank.rows[(index + 1) * bank.stride..(index + 2) * bank.stride];
        blend_rows::<LANES>(src, lower, upper, blend)
    }
}

fn blend_rows<const N: usize>(src: &[f32], lower: &[f32], upper: &[f32], blend: f32) -> Sample {
    let (mut low, mut high) = (Fv::<N>::splat(0.0), Fv::<N>::splat(0.0));
    let mut offset = 0;
    while offset < src.len() {
        let x = load_partial::<N>(&src[offset..]);
        low += x * Fv::<N>::from_slice(&lower[offset..]);
        high += x * Fv::<N>::from_slice(&upper[offset..]);
        offset += N;
    }
    let (low, high) = (low.reduce_sum(), high.reduce_sum());
    low + (high - low) * blend
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_row_sums_to_one() {
        for half_taps in [1usize, 16, 64] {
            let bank = bank(half_taps);
            for phase in 0..=PHASES {
                let row = &bank.rows[phase * bank.stride..phase * bank.stride + 2 * half_taps + 2];
                let total: f64 = row.iter().map(|&w| f64::from(w)).sum();
                assert!(
                    (total - 1.0).abs() < 1e-5,
                    "half_taps={half_taps} phase={phase} sums to {total}"
                );
            }
        }
    }

    #[test]
    fn the_last_row_is_the_first_row_shifted_by_one_tap() {
        let bank = bank(16);
        let first = &bank.rows[..2 * 16 + 2];
        let last = &bank.rows[PHASES * bank.stride..PHASES * bank.stride + 2 * 16 + 2];
        for (j, &w) in last[1..].iter().enumerate() {
            assert!((w - first[j]).abs() < 1e-6, "tap {j}: {w} vs {}", first[j]);
        }
    }

    fn worst_blend_error(half_taps: usize, phases: usize) -> f64 {
        let bank = build_bank(half_taps, phases);
        let taps = 2 * half_taps + 2;
        let mut worst = 0.0f64;
        for step in 0..97 {
            let fraction = step as f64 / 97.0;
            let exact: Vec<f64> = (0..taps)
                .map(|j| {
                    let d = half_taps as f64 + fraction - j as f64;
                    let x = std::f64::consts::PI * d;
                    let lobe = if x.abs() < 1e-12 { 1.0 } else { x.sin() / x };
                    lobe * f64::from(blackman_f64(d.abs() / half_taps as f64))
                })
                .collect();
            let total: f64 = exact.iter().sum();

            let scaled = fraction * phases as f64;
            let index = scaled as usize;
            let blend = scaled - index as f64;
            for (j, want) in exact.iter().enumerate() {
                let low = f64::from(bank.rows[index * bank.stride + j]);
                let high = f64::from(bank.rows[(index + 1) * bank.stride + j]);
                worst = worst.max((low + (high - low) * blend - want / total).abs());
            }
        }
        worst
    }

    #[test]
    fn the_bank_resolves_the_kernel_it_tabulates() {
        for half_taps in [16usize, 64] {
            let resolved = worst_blend_error(half_taps, PHASES);
            let coarse = worst_blend_error(half_taps, 16);
            assert!(
                resolved < 5e-5,
                "half_taps={half_taps}: {PHASES} phases leave {resolved:.2e}"
            );
            assert!(
                coarse > 10.0 * resolved,
                "half_taps={half_taps}: 16 phases leave only {coarse:.2e}"
            );
        }
    }

    #[test]
    fn a_cutoff_below_nyquist_defers_to_the_tabulated_kernel() {
        let data: Vec<f32> = (0..256)
            .map(|k| (0.7 * (k as f32) + 0.3 * (k as f32).sin()).sin())
            .collect();
        let banked = ResampleImplPolyphase::<4>::tables(16);
        let tabulated = ResampleImplSimd::<4>::tables(16);
        for fc in [0.2f32, 0.37, 0.49] {
            for i in 0..32 {
                let pos = 16.0 + i as f32 * 0.31;
                let (lo, hi) = ResampleImplPolyphase::<4>::tap_window(&banked, pos);
                let src: Vec<f32> = (lo..=hi)
                    .map(|k| data[k.rem_euclid(data.len() as i64) as usize])
                    .collect();
                assert_eq!(
                    ResampleImplPolyphase::<4>::resample(&banked, &mut (), &src, pos, fc, false),
                    ResampleImplSimd::<4>::resample(&tabulated, &mut (), &src, pos, fc, false)
                );
            }
        }
    }
}

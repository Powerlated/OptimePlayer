//! Step mode as a recursive filter instead of a tap window, and the default for it. Every other
//! implementation answers "what is the band-limited staircase worth at this instant" by summing a
//! windowed kernel over the source samples around it, which costs one pass over the window per
//! output sample. This one never builds a window: it keeps a filter running.
//!
//! What makes that possible is that step mode's cutoff is not really ratio-dependent. `sinc_fc`
//! returns `0.5/r` in source-normalised units, and multiplying by `r` to reach output-normalised
//! units leaves `min(0.5, cutoff_hz/out_rate)` — the same corner for every voice, whatever it is
//! playing. So the whole of step mode is one fixed low-pass applied to a staircase and sampled at
//! the output rate, and the only thing that varies per voice is *when* the staircase steps.
//!
//! A continuous-time filter driven by a piecewise-constant input has an exact solution: with a
//! state `s' = p s + u`, holding `u` for a span `t` gives `s ← e^{pt} s + u (e^{pt} - 1)/p`. So the
//! filter is advanced in one or two hops per output sample — one per source-sample boundary that
//! falls inside it — and `e^{pt}` and `(e^{pt} - 1)/p` come from a small table of spans. Because
//! this integrates the staircase in continuous time before sampling it, it anti-aliases correctly
//! at any ratio rather than only when upsampling.
//!
//! The prototype is a Chebyshev type I of `ORDER`, in modal (partial-fraction) form so the poles
//! are independent and the whole state is one SIMD register of real parts and one of imaginary.
//! Modal form is ill-conditioned when poles cluster; these sit spread around an ellipse, residues
//! stay of order one, and f32 tracks f64 to −135 dB, which is why it is safe here. Phase is not
//! linear and deliberately so: that is what buys the order-of-magnitude, and step mode's contract is
//! the amplitude of the harmonics it passes and the energy it rejects, not the shape of an edge.
//!
//! Only one table is ever built, for a corner at the output Nyquist, because changing the corner is
//! exactly a change of time scale: advancing by `t` under a corner `c` is advancing by `2ct` under
//! `0.5`, and the residue and drive scalings that go with it cancel between the state and the
//! output. Impulse mode is not this file's business and goes to `ResampleImplPolyphase`.

use std::simd::prelude::*;
use std::sync::OnceLock;

use super::resample_impl_02_polyphase::{self, ResampleImplPolyphase};
use super::{DEFAULT_LANES, Fv};
use crate::dsp::resample::{GatherSource, MAX_HALF_TAPS, Resampler};
use crate::waveform::Sample;

const ORDER: usize = 10;
const POLE_LANES: usize = 8;
const SPANS: usize = 128;
const PASSBAND_RIPPLE_DB: f64 = 0.05;
const PASSBAND_EDGE: f64 = 0.92;

pub struct ResampleImplIir<const LANES: usize = DEFAULT_LANES>;

#[derive(Clone)]
pub struct Tables {
    pub half_taps: usize,
    impulse: resample_impl_02_polyphase::Tables,
}

#[derive(Clone, Default)]
pub struct State {
    real: Fv<POLE_LANES>,
    imaginary: Fv<POLE_LANES>,
    last_pos: f32,
    last_span: f32,
    running: bool,
}

struct Prototype {
    step_real: Vec<f32>,
    step_imaginary: Vec<f32>,
    drive_real: Vec<f32>,
    drive_imaginary: Vec<f32>,
    residue_real: Fv<POLE_LANES>,
    residue_imaginary: Fv<POLE_LANES>,
}

fn complex_exp(re: f64, im: f64) -> (f64, f64) {
    let magnitude = re.exp();
    (magnitude * im.cos(), magnitude * im.sin())
}

fn complex_div(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}

fn complex_mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn chebyshev_poles() -> Vec<(f64, f64)> {
    let corner = std::f64::consts::PI * PASSBAND_EDGE;
    let epsilon = (10f64.powf(PASSBAND_RIPPLE_DB / 10.0) - 1.0).sqrt();
    let a = (1.0 / epsilon).asinh() / ORDER as f64;
    (0..ORDER)
        .map(|k| {
            let theta = std::f64::consts::PI * (2 * k + 1) as f64 / (2 * ORDER) as f64;
            (
                corner * -a.sinh() * theta.sin(),
                corner * a.cosh() * theta.cos(),
            )
        })
        .collect()
}

fn prototype() -> &'static Prototype {
    static P: OnceLock<Prototype> = OnceLock::new();
    P.get_or_init(|| {
        let poles = chebyshev_poles();

        let mut gain = (1.0, 0.0);
        for &(re, im) in &poles {
            gain = complex_mul(gain, (-re, -im));
        }
        let gain = gain.0;

        let mut residues = Vec::new();
        let mut upper = Vec::new();
        for (k, &pole) in poles.iter().enumerate() {
            if pole.1 <= 0.0 {
                continue;
            }
            let mut denominator = (1.0, 0.0);
            for (j, &other) in poles.iter().enumerate() {
                if j != k {
                    denominator = complex_mul(denominator, (pole.0 - other.0, pole.1 - other.1));
                }
            }
            residues.push(complex_div((gain, 0.0), denominator));
            upper.push(pole);
        }

        let mut step_real = vec![0.0f32; (SPANS + 1) * POLE_LANES];
        let mut step_imaginary = vec![0.0f32; (SPANS + 1) * POLE_LANES];
        let mut drive_real = vec![0.0f32; (SPANS + 1) * POLE_LANES];
        let mut drive_imaginary = vec![0.0f32; (SPANS + 1) * POLE_LANES];
        for slot in 0..=SPANS {
            let span = slot as f64 / SPANS as f64;
            for (k, &pole) in upper.iter().enumerate() {
                let e = complex_exp(pole.0 * span, pole.1 * span);
                let g = complex_div((e.0 - 1.0, e.1), pole);
                step_real[slot * POLE_LANES + k] = e.0 as f32;
                step_imaginary[slot * POLE_LANES + k] = e.1 as f32;
                drive_real[slot * POLE_LANES + k] = g.0 as f32;
                drive_imaginary[slot * POLE_LANES + k] = g.1 as f32;
            }
        }

        let (mut residue_real, mut residue_imaginary) =
            ([0.0f32; POLE_LANES], [0.0f32; POLE_LANES]);
        for (k, &(re, im)) in residues.iter().enumerate() {
            residue_real[k] = re as f32;
            residue_imaginary[k] = im as f32;
        }

        Prototype {
            step_real,
            step_imaginary,
            drive_real,
            drive_imaginary,
            residue_real: Simd::from_array(residue_real),
            residue_imaginary: Simd::from_array(residue_imaginary),
        }
    })
}

impl<const LANES: usize> Resampler for ResampleImplIir<LANES> {
    type Tables = Tables;
    type State = State;

    fn tables(half_taps: usize) -> Tables {
        let half_taps = half_taps.clamp(1, MAX_HALF_TAPS);
        let _ = prototype();
        Tables {
            half_taps,
            impulse: ResampleImplPolyphase::<LANES>::tables(half_taps),
        }
    }

    #[inline]
    fn half_taps(tables: &Tables) -> usize {
        tables.half_taps
    }

    #[inline]
    fn tap_window(tables: &Tables, pos: f32) -> (i64, i64) {
        ResampleImplPolyphase::<LANES>::tap_window(&tables.impulse, pos)
    }

    fn resample(
        tables: &Tables,
        state: &mut State,
        src: &[f32],
        pos: f32,
        fc: f32,
        step_mode: bool,
    ) -> Sample {
        if !step_mode {
            return ResampleImplPolyphase::<LANES>::resample(
                &tables.impulse,
                &mut (),
                src,
                pos,
                fc,
                step_mode,
            );
        }
        let window_start = (pos - tables.half_taps as f32).floor() as i64;
        run(prototype(), state, pos, fc, tables.half_taps, |index| {
            let offset = (index - window_start).clamp(0, src.len() as i64 - 1);
            src[offset as usize]
        })
    }

    fn gather(
        source: &GatherSource,
        tables: &Tables,
        state: &mut State,
        pos: f32,
        fc: f32,
        step_mode: bool,
    ) -> Sample {
        if !step_mode {
            return ResampleImplPolyphase::<LANES>::gather(
                source,
                &tables.impulse,
                &mut (),
                pos,
                fc,
                step_mode,
            );
        }
        let &GatherSource {
            data,
            looping,
            loop_point,
            loop_len,
            ..
        } = source;
        let len = data.len() as i64;
        run(prototype(), state, pos, fc, tables.half_taps, |index| {
            if index >= 0 && index < len {
                data[index as usize]
            } else if index >= len && looping && loop_len > 0 {
                data[((index - loop_point).rem_euclid(loop_len) + loop_point) as usize]
            } else {
                0.0
            }
        })
    }
}

fn run(
    prototype: &Prototype,
    state: &mut State,
    pos: f32,
    fc: f32,
    half_taps: usize,
    sample_at: impl Fn(i64) -> f32,
) -> Sample {
    let newest = pos.floor() as i64;

    if !state.running {
        state.running = true;
        state.last_pos = pos;
        state.last_span = 0.0;
        return 0.0;
    }

    let mut span = pos - state.last_pos;
    if span.is_nan() || span <= 0.0 || span > half_taps as f32 {
        span = state.last_span.clamp(f32::MIN_POSITIVE, half_taps as f32);
    }
    state.last_pos = pos;
    state.last_span = span;

    let scale = 2.0 * fc * span;
    let first_boundary = (pos - span).floor() as i64 + 1;

    let mut walked = pos - span;
    let mut boundary = first_boundary;
    while (boundary as f32) < pos && boundary <= newest {
        advance(
            prototype,
            state,
            (boundary as f32 - walked) / span * scale,
            sample_at(boundary - 1),
        );
        walked = boundary as f32;
        boundary += 1;
    }
    advance(
        prototype,
        state,
        (pos - walked) / span * scale,
        sample_at(walked.floor() as i64),
    );

    let out = prototype.residue_real * state.real - prototype.residue_imaginary * state.imaginary;
    Sample::from(2.0 * out.reduce_sum())
}

#[inline]
fn advance(prototype: &Prototype, state: &mut State, span: f32, drive: f32) {
    let slot = ((span.clamp(0.0, 1.0) * SPANS as f32) as usize).min(SPANS) * POLE_LANES;
    let step_real = Fv::<POLE_LANES>::from_slice(&prototype.step_real[slot..]);
    let step_imaginary = Fv::<POLE_LANES>::from_slice(&prototype.step_imaginary[slot..]);
    let drive_real = Fv::<POLE_LANES>::from_slice(&prototype.drive_real[slot..]);
    let drive_imaginary = Fv::<POLE_LANES>::from_slice(&prototype.drive_imaginary[slot..]);
    let u = Fv::<POLE_LANES>::splat(drive);

    let real = step_real * state.real - step_imaginary * state.imaginary + drive_real * u;
    let imaginary = step_real * state.imaginary + step_imaginary * state.real + drive_imaginary * u;
    state.real = real;
    state.imaginary = imaginary;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prototype_has_the_pole_count_its_order_implies() {
        let p = prototype();
        let used = (0..POLE_LANES)
            .filter(|&k| p.residue_real[k] != 0.0 || p.residue_imaginary[k] != 0.0)
            .count();
        assert_eq!(used, ORDER / 2);
    }

    #[test]
    fn modal_residues_stay_of_order_one() {
        let p = prototype();
        for k in 0..ORDER / 2 {
            let magnitude = (p.residue_real[k].powi(2) + p.residue_imaginary[k].powi(2)).sqrt();
            assert!(magnitude < 4.0, "residue {k} has magnitude {magnitude}");
        }
    }

    #[test]
    fn a_held_input_settles_to_it() {
        let p = prototype();
        let mut state = State::default();
        for _ in 0..4000 {
            advance(p, &mut state, 1.0 / 64.0, 1.0);
        }
        let out = 2.0
            * (p.residue_real * state.real - p.residue_imaginary * state.imaginary).reduce_sum();
        assert!((out - 1.0).abs() < 1e-4, "settled to {out}");
    }
}

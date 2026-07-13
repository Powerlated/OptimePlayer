//! A reusable linear slew: moves a held value toward a target by a bounded step per sample.
//!
//! Stepping a control value (a gain, a pan position, a filter cutoff) straight to its new target
//! produces an audible click or zipper. [`Slewer`] turns that hard step into a short linear ramp:
//! each [`advance`](Slewer::advance) moves the held value toward the supplied target by at most a
//! fixed per-sample step, landing exactly on the target once it is within reach. It is the de-click
//! ramp behind pop smoothing and the building block for smoothing panning changes and similar
//! per-sample control slews.

/// A linearly-slewed value.
///
/// Holds a current value and a maximum change per [`advance`](Self::advance) call. The slew rate is
/// expressed in value units per sample, so the same `Slewer` works for any quantity (gain, pan,
/// cutoff) as long as the target is in the same units.
#[derive(Debug, Clone, Copy)]
pub struct Slewer {
    current: f32,
    /// Maximum change applied per [`advance`](Self::advance), always non-negative.
    step: f32,
}

impl Slewer {
    /// Creates a slewer starting at `initial`, moving at most `step_per_sample` value units toward
    /// the target on each [`advance`](Self::advance). A `step_per_sample` of `0.0` freezes the
    /// value; a very large step makes `advance` jump straight to the target.
    pub fn new(initial: f32, step_per_sample: f32) -> Self {
        Self {
            current: initial,
            step: step_per_sample.abs(),
        }
    }

    /// Creates a slewer whose step crosses one unit of value (e.g. the full `0..1` gain range) in
    /// `seconds` at `sample_rate` Hz. This is the de-click ramp used for gain/pan smoothing, where
    /// the controlled value lives in a normalized `0..1` range. `seconds`/`sample_rate` are wall-
    /// clock timing (kept `f64`); only the resulting per-sample step is narrowed to the slew width.
    pub fn from_time(initial: f32, seconds: f64, sample_rate: f64) -> Self {
        let step = if seconds > 0.0 && sample_rate > 0.0 {
            (1.0 / (seconds * sample_rate)) as f32
        } else {
            f32::INFINITY
        };
        Self::new(initial, step)
    }

    /// The current (last produced) value.
    #[inline]
    pub fn value(&self) -> f32 {
        self.current
    }

    /// Jumps immediately to `value`, bypassing the ramp. Use when a discontinuity is intended
    /// (e.g. priming a voice's gain at note start).
    #[inline]
    pub fn set(&mut self, value: f32) {
        self.current = value;
    }

    /// Replaces the per-sample step (value units per [`advance`](Self::advance)).
    #[inline]
    pub fn set_step(&mut self, step_per_sample: f32) {
        self.step = step_per_sample.abs();
    }

    /// Moves the held value toward `target` by at most one step, landing exactly on `target` once
    /// within reach, and returns the new value.
    #[inline]
    pub fn advance(&mut self, target: f32) -> f32 {
        let d = target - self.current;
        self.current = if d.abs() <= self.step {
            target
        } else {
            self.current + self.step.copysign(d)
        };
        self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advances_in_bounded_steps_and_lands_exactly() {
        let mut s = Slewer::new(0.0, 0.25);
        assert_eq!(s.advance(1.0), 0.25);
        assert_eq!(s.advance(1.0), 0.5);
        assert_eq!(s.advance(1.0), 0.75);
        // Final partial step lands exactly on the target rather than overshooting.
        assert_eq!(s.advance(1.0), 1.0);
        assert_eq!(s.advance(1.0), 1.0);
    }

    #[test]
    fn slews_downward_too() {
        let mut s = Slewer::new(1.0, 0.4);
        // `0.4` is not exactly representable in `f32`, so compare within the sample width's epsilon
        // rather than the old `f64` 1e-12.
        assert!((s.advance(0.0) - 0.6).abs() < 1e-6);
        assert!((s.advance(0.0) - 0.2).abs() < 1e-6);
        assert_eq!(s.advance(0.0), 0.0);
    }

    #[test]
    fn from_time_crosses_unit_range_in_the_given_seconds() {
        let sample_rate = 48_000.0;
        let seconds = 0.002;
        let mut s = Slewer::from_time(0.0, seconds, sample_rate);
        let steps = (seconds * sample_rate) as usize; // 96 samples to cross 0..1
        for _ in 0..steps {
            s.advance(1.0);
        }
        assert!((s.value() - 1.0).abs() < 1e-9, "got {}", s.value());
        // One step before the end it must not have arrived yet.
        let mut s2 = Slewer::from_time(0.0, seconds, sample_rate);
        for _ in 0..steps - 1 {
            s2.advance(1.0);
        }
        assert!(s2.value() < 1.0);
    }

    #[test]
    fn set_jumps_without_ramping() {
        let mut s = Slewer::new(0.0, 0.01);
        s.set(0.7);
        assert_eq!(s.value(), 0.7);
    }

    #[test]
    fn zero_seconds_jumps_immediately() {
        let mut s = Slewer::from_time(0.0, 0.0, 48_000.0);
        assert_eq!(s.advance(1.0), 1.0);
    }
}

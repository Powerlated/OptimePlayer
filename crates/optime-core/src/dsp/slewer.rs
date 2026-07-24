//! Stepping a control value such as a gain, a pan position, or a filter cutoff straight to a new
//! target produces an audible click, so [`Slewer`] spreads that change over a short linear ramp
//! instead, which is the de-click mechanism behind pop smoothing and pan smoothing.

/// Which direction of movement the ramp applies to, the opposite direction being taken in one jump.
// The `dsp` module is private to the crate, so this stays dead code until a caller selects a
// direction.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Only a rise is ramped, and a fall happens at once.
    UpOnly,
    /// Only a fall is ramped, and a rise happens at once.
    DownOnly,
    /// Both a rise and a fall are ramped.
    UpAndDown,
}

/// A value that moves toward whatever target it is given by at most a fixed amount per sample,
/// measured in the same units as the target, so one `Slewer` serves a gain, a pan position, or a
/// cutoff equally well.
#[derive(Debug, Clone, Copy)]
pub struct Slewer {
    current: f32,
    /// The furthest the value may move in one call to [`advance`](Slewer::advance), never negative.
    step: f32,
}

impl Slewer {
    /// Starts at `initial` and moves at most `step_per_sample` value units toward the target per
    /// call to [`advance`](Self::advance), so a step of zero freezes the value where it is and a
    /// very large step arrives on the target in a single call.
    pub fn new(initial: f32, step_per_sample: f32) -> Self {
        Self {
            current: initial,
            step: step_per_sample.abs(),
        }
    }

    /// Starts at `initial` and picks the step that takes `seconds` to cross one whole unit of
    /// value at `sample_rate` Hz, such as the full `0..1` gain range used by the de-click ramp,
    /// with both arguments kept as `f64` because they are wall-clock timings rather than audio
    /// values.
    pub fn from_time(initial: f32, seconds: f64, sample_rate: f64) -> Self {
        let step = if seconds > 0.0 && sample_rate > 0.0 {
            (1.0 / (seconds * sample_rate)) as f32
        } else {
            f32::INFINITY
        };
        Self::new(initial, step)
    }

    /// The value last produced by [`advance`](Self::advance), or the initial value if it has not
    /// run yet.
    #[inline]
    pub fn value(&self) -> f32 {
        self.current
    }

    /// Jumps straight to `value` without ramping, for the cases where a discontinuity is intended,
    /// such as priming a voice's gain at the start of a note.
    #[inline]
    pub fn set(&mut self, value: f32) {
        self.current = value;
    }

    /// Replaces the step with `step_per_sample` value units per call to
    /// [`advance`](Self::advance).
    #[inline]
    pub fn set_step(&mut self, step_per_sample: f32) {
        self.step = step_per_sample.abs();
    }

    /// Moves the held value one step toward `target`, landing exactly on it once it is within a
    /// single step rather than overshooting, and returns the new value.
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
        // The last step is only 0.25 away from the target, so it lands exactly on it rather than
        // overshooting to 1.25.
        assert_eq!(s.advance(1.0), 1.0);
        assert_eq!(s.advance(1.0), 1.0);
    }

    #[test]
    fn slews_downward_too() {
        let mut s = Slewer::new(1.0, 0.4);
        // `0.4` cannot be stored exactly in an `f32`, so compare within a small margin instead of
        // for equality.
        assert!((s.advance(0.0) - 0.6).abs() < 1e-6);
        assert!((s.advance(0.0) - 0.2).abs() < 1e-6);
        assert_eq!(s.advance(0.0), 0.0);
    }

    #[test]
    fn from_time_crosses_unit_range_in_the_given_seconds() {
        let sample_rate = 48_000.0;
        let seconds = 0.002;
        let mut s = Slewer::from_time(0.0, seconds, sample_rate);
        // A 2 ms ramp at 48 kHz is 96 samples, which is how many steps crossing the whole `0..1`
        // range should take.
        let steps = (seconds * sample_rate) as usize;
        for _ in 0..steps {
            s.advance(1.0);
        }
        assert!((s.value() - 1.0).abs() < 1e-9, "got {}", s.value());
        // One step short of the end the value must still be below the target, or the ramp is
        // finishing early.
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

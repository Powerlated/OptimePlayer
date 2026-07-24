//! Stepping a control value such as a gain, a pan position, or a filter cutoff straight to a new
//! target produces an audible click. [`Slewer`] spreads that change over a short linear ramp
//! instead. It is the de-click mechanism behind pop smoothing and pan smoothing.

/// Which direction of movement gets the ramp. A move the other way arrives in one jump.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Direction {
    /// Ramp a rise, drop instantly.
    UpOnly,
    /// Ramp a fall, rise instantly.
    DownOnly,
    /// Ramp both, which is what de-clicking a gain or a pan position needs.
    #[default]
    UpAndDown,
}

/// A value that moves toward whatever target it is given, by at most a fixed amount per sample.
/// The step is in the same units as the target, so one `Slewer` serves a gain, a pan position or a
/// cutoff equally well.
#[derive(Debug, Clone, Copy)]
pub struct Slewer {
    current: f32,
    /// The furthest the value may move in one call to [`advance`](Slewer::advance). Never negative.
    step: f32,
    /// The direction that gets the ramp; the other one is jumped in a single call to
    /// [`advance`](Slewer::advance).
    direction: Direction,
}

impl Slewer {
    /// Starts at `initial` and moves at most `step_per_sample` value units toward the target on
    /// each call to [`advance`](Self::advance). A step of zero freezes the value where it is; a
    /// very large one arrives on the target in a single call.
    pub fn new(initial: f32, step_per_sample: f32) -> Self {
        Self {
            current: initial,
            step: step_per_sample.abs(),
            direction: Direction::default(),
        }
    }

    /// Restricts the ramp to `direction`, so a move the other way completes in one call to
    /// [`advance`](Self::advance). Suits a control that must fall instantly but may rise gently, or
    /// the reverse.
    pub fn with_direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }

    /// Starts at `initial` and picks the step that takes `seconds` to cross one whole unit of
    /// value at `sample_rate` Hz, such as the full `0..1` gain range the de-click ramp works in.
    /// Both arguments stay `f64` because they are wall-clock timings rather than audio values.
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

    /// Jumps straight to `value` without ramping. Use it where a discontinuity is intended, such as
    /// priming a voice's gain at the start of a note.
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

    /// Replaces the ramped direction on an existing slewer, for a control whose smoothing setting
    /// can change while it is in use.
    #[inline]
    pub fn set_direction(&mut self, direction: Direction) {
        self.direction = direction;
    }

    /// Moves the held value one step toward `target` and returns the new value. Once the target is
    /// within a single step it lands exactly on it rather than overshooting, and a move that runs
    /// against the ramped direction arrives in this one call.
    #[inline]
    pub fn advance(&mut self, target: f32) -> f32 {
        let d = target - self.current;
        let ramped = match self.direction {
            Direction::UpOnly => d > 0.0,
            Direction::DownOnly => d < 0.0,
            Direction::UpAndDown => true,
        };
        self.current = if !ramped || d.abs() <= self.step {
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
        // A 2 ms ramp at 48 kHz is 96 samples. That is how many steps crossing the whole `0..1`
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
    fn up_only_ramps_a_rise_and_drops_at_once() {
        let mut s = Slewer::new(0.0, 0.25).with_direction(Direction::UpOnly);
        assert_eq!(s.advance(1.0), 0.25);
        assert_eq!(s.advance(1.0), 0.5);
        // The fall runs against the ramped direction, so it arrives in this one call.
        assert_eq!(s.advance(0.0), 0.0);
    }

    #[test]
    fn down_only_ramps_a_fall_and_rises_at_once() {
        let mut s = Slewer::new(1.0, 0.25).with_direction(Direction::DownOnly);
        assert_eq!(s.advance(0.0), 0.75);
        assert_eq!(s.advance(0.0), 0.5);
        // The rise runs against the ramped direction, so it arrives in this one call.
        assert_eq!(s.advance(1.0), 1.0);
    }

    #[test]
    fn a_new_slewer_ramps_both_ways() {
        let mut s = Slewer::new(0.0, 0.25);
        assert_eq!(s.advance(1.0), 0.25);
        assert_eq!(s.advance(0.0), 0.0);
        assert_eq!(Direction::default(), Direction::UpAndDown);
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

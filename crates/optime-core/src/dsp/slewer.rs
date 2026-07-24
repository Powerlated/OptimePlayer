//! Eases a value toward a target a little at a time, so that a change of gain, pan, or cutoff
//! ramps over a few milliseconds instead of jumping and clicking.

/// A value that moves toward whatever target it is given, by at most a fixed amount each sample,
/// in whatever units the caller works in.
#[derive(Debug, Clone, Copy)]
pub struct Slewer {
    current: f32,
    /// The furthest the value may move in one step, never negative.
    step: f32,
}

impl Slewer {
    /// Starts at `initial` and moves at most `step_per_sample` toward the target on each step, so a
    /// step of zero freezes the value and a huge one jumps straight there.
    pub fn new(initial: f32, step_per_sample: f32) -> Self {
        Self {
            current: initial,
            step: step_per_sample.abs(),
        }
    }

    /// Starts at `initial` and picks a step that takes `seconds` to cross one whole unit of value
    /// at `sample_rate` Hz, with both of those kept as `f64` because they are wall-clock timings
    /// rather than audio values.
    pub fn from_time(initial: f32, seconds: f64, sample_rate: f64) -> Self {
        let step = if seconds > 0.0 && sample_rate > 0.0 {
            (1.0 / (seconds * sample_rate)) as f32
        } else {
            f32::INFINITY
        };
        Self::new(initial, step)
    }

    /// The value as it stands right now.
    #[inline]
    pub fn value(&self) -> f32 {
        self.current
    }

    /// Jumps straight to `value` without ramping, for when a sudden change is what is wanted, such
    /// as setting a voice's gain as a note starts.
    #[inline]
    pub fn set(&mut self, value: f32) {
        self.current = value;
    }

    /// Changes how far the value is allowed to move in one step.
    #[inline]
    pub fn set_step(&mut self, step_per_sample: f32) {
        self.step = step_per_sample.abs();
    }

    /// Moves one step toward `target`, settling exactly on it once it is within a step's reach, and
    /// returns the new value.
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
        // The last part-step lands exactly on the target instead of overshooting it.
        assert_eq!(s.advance(1.0), 1.0);
        assert_eq!(s.advance(1.0), 1.0);
    }

    #[test]
    fn slews_downward_too() {
        let mut s = Slewer::new(1.0, 0.4);
        // `0.4` cannot be stored exactly in an `f32`, so allow a small margin when comparing.
        assert!((s.advance(0.0) - 0.6).abs() < 1e-6);
        assert!((s.advance(0.0) - 0.2).abs() < 1e-6);
        assert_eq!(s.advance(0.0), 0.0);
    }

    #[test]
    fn from_time_crosses_unit_range_in_the_given_seconds() {
        let sample_rate = 48_000.0;
        let seconds = 0.002;
        let mut s = Slewer::from_time(0.0, seconds, sample_rate);
        // Crossing the whole range should take 96 samples here.
        let steps = (seconds * sample_rate) as usize;
        for _ in 0..steps {
            s.advance(1.0);
        }
        assert!((s.value() - 1.0).abs() < 1e-9, "got {}", s.value());
        // One step short of the end it must not have arrived yet.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Direction {
    UpOnly,
    DownOnly,
    #[default]
    UpAndDown,
}

#[derive(Debug, Clone, Copy)]
pub struct Slewer {
    current: f32,
    step: f32,
    direction: Direction,
}

impl Slewer {
    pub fn new(initial: f32, step_per_sample: f32) -> Self {
        Self {
            current: initial,
            step: step_per_sample.abs(),
            direction: Direction::default(),
        }
    }

    pub fn with_direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }

    pub fn from_time(initial: f32, seconds: f64, sample_rate: f64) -> Self {
        let step = if seconds > 0.0 && sample_rate > 0.0 {
            (1.0 / (seconds * sample_rate)) as f32
        } else {
            f32::INFINITY
        };
        Self::new(initial, step)
    }

    #[inline]
    pub fn value(&self) -> f32 {
        self.current
    }

    #[inline]
    pub fn set(&mut self, value: f32) {
        self.current = value;
    }

    #[inline]
    pub fn set_step(&mut self, step_per_sample: f32) {
        self.step = step_per_sample.abs();
    }

    #[inline]
    pub fn set_direction(&mut self, direction: Direction) {
        self.direction = direction;
    }

    pub fn advance_block(&mut self, out: &mut [f32], target: f32) {
        let (step, direction) = (self.step, self.direction);
        let mut current = self.current;
        for slot in out.iter_mut() {
            let d = target - current;
            let ramped = match direction {
                Direction::UpOnly => d > 0.0,
                Direction::DownOnly => d < 0.0,
                Direction::UpAndDown => true,
            };
            current = if !ramped || d.abs() <= step {
                target
            } else {
                current + step.copysign(d)
            };
            *slot = current;
        }
        self.current = current;
    }

    #[inline]
    pub fn advance(&mut self, target: f32) -> f32 {
        let mut out = [0.0];
        self.advance_block(&mut out, target);
        out[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advance_block_matches_per_sample() {
        use crate::dsp::block::TEST_BLOCK_LENGTHS;

        for direction in [Direction::UpAndDown, Direction::UpOnly, Direction::DownOnly] {
            for n in TEST_BLOCK_LENGTHS {
                let targets: Vec<f32> = (0..8)
                    .map(|i| if i % 2 == 0 { 1.0 } else { -0.3 })
                    .collect();
                let make = || Slewer::new(0.0, 0.01).with_direction(direction);

                let mut blocked = make();
                let mut got = Vec::new();
                for &target in &targets {
                    let mut chunk = vec![0.0; n];
                    blocked.advance_block(&mut chunk, target);
                    got.extend_from_slice(&chunk);
                }

                let mut per_sample = make();
                let want: Vec<f32> = targets
                    .iter()
                    .flat_map(|&target| (0..n).map(move |_| target))
                    .map(|target| per_sample.advance(target))
                    .collect();

                assert_eq!(got, want, "{direction:?}, block length {n}");
            }
        }
    }

    #[test]
    fn advances_in_bounded_steps_and_lands_exactly() {
        let mut s = Slewer::new(0.0, 0.25);
        assert_eq!(s.advance(1.0), 0.25);
        assert_eq!(s.advance(1.0), 0.5);
        assert_eq!(s.advance(1.0), 0.75);
        assert_eq!(s.advance(1.0), 1.0);
        assert_eq!(s.advance(1.0), 1.0);
    }

    #[test]
    fn slews_downward_too() {
        let mut s = Slewer::new(1.0, 0.4);
        assert!((s.advance(0.0) - 0.6).abs() < 1e-6);
        assert!((s.advance(0.0) - 0.2).abs() < 1e-6);
        assert_eq!(s.advance(0.0), 0.0);
    }

    #[test]
    fn from_time_crosses_unit_range_in_the_given_seconds() {
        let sample_rate = 48_000.0;
        let seconds = 0.002;
        let mut s = Slewer::from_time(0.0, seconds, sample_rate);
        let steps = (seconds * sample_rate) as usize;
        for _ in 0..steps {
            s.advance(1.0);
        }
        assert!((s.value() - 1.0).abs() < 1e-9, "got {}", s.value());
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
        assert_eq!(s.advance(0.0), 0.0);
    }

    #[test]
    fn down_only_ramps_a_fall_and_rises_at_once() {
        let mut s = Slewer::new(1.0, 0.25).with_direction(Direction::DownOnly);
        assert_eq!(s.advance(0.0), 0.75);
        assert_eq!(s.advance(0.0), 0.5);
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

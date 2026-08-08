//! Simultaneous-perturbation stochastic gradient descent over bounded parameters.
//!
//! The thing being minimised here is a full render of a song through the engine, which is not
//! differentiable and never will be — there is no autodiff tape under a sequencer, a voice pool and
//! a chain of recursive filters. SPSA is the honest stochastic gradient method for that case: it
//! perturbs *every* parameter at once by a random sign vector, evaluates the objective twice, and
//! reads the whole gradient off that single difference. The estimate is noisy and, per step, biased
//! along the perturbation, but it is unbiased in expectation over the sign draws, and its cost is
//! two evaluations regardless of how many parameters there are — which is what makes tuning a
//! dozen knobs against a several-second render affordable at all. Adam then does the averaging that
//! turns those noisy estimates into a descent direction.
//!
//! The objective is handed a batch seed alongside the parameters, and it must select the same
//! minibatch for both evaluations of a step. Two evaluations drawn from different songs would
//! differ mostly by which songs they were, and the perturbation — the only thing the difference is
//! supposed to be measuring — would be lost in that.
//!
//! Parameters are searched in an unconstrained coordinate and squashed back into their range, so a
//! step can never leave the range and no clamping is needed anywhere. A knob declared on a `Log`
//! scale is squashed geometrically, which is what a frequency or a time constant wants: the step
//! that moves 100 Hz to 200 Hz should move 4 kHz to 8 kHz.

const ADAM_BETA1: f64 = 0.9;
const ADAM_BETA2: f64 = 0.999;
const ADAM_EPSILON: f64 = 1.0e-8;
const PERTURBATION_DECAY: f64 = 0.101;
const FREE_LIMIT: f64 = 8.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scale {
    Linear,
    Log,
}

#[derive(Debug, Clone, Copy)]
pub struct Knob {
    pub name: &'static str,
    pub lo: f64,
    pub hi: f64,
    pub scale: Scale,
}

impl Knob {
    pub fn value(&self, free: f64) -> f64 {
        let unit = 1.0 / (1.0 + (-free).exp());
        let value = match self.scale {
            Scale::Linear => self.lo + (self.hi - self.lo) * unit,
            Scale::Log => (self.lo.ln() + (self.hi.ln() - self.lo.ln()) * unit).exp(),
        };
        value.clamp(self.lo, self.hi)
    }

    pub fn free(&self, value: f64) -> f64 {
        let unit = match self.scale {
            Scale::Linear => (value - self.lo) / (self.hi - self.lo),
            Scale::Log => (value.ln() - self.lo.ln()) / (self.hi.ln() - self.lo.ln()),
        };
        let unit = unit.clamp(1.0e-6, 1.0 - 1.0e-6);
        (unit / (1.0 - unit)).ln().clamp(-FREE_LIMIT, FREE_LIMIT)
    }
}

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn sign(&mut self) -> f64 {
        if self.next_u64() & 1 == 0 { -1.0 } else { 1.0 }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StepReport {
    pub loss: f64,
    pub perturbation: f64,
}

pub struct Spsa {
    knobs: &'static [Knob],
    free: Vec<f64>,
    moment: Vec<f64>,
    velocity: Vec<f64>,
    learning_rate: f64,
    perturbation: f64,
    step: usize,
    rng: Rng,
}

impl Spsa {
    pub fn new(
        knobs: &'static [Knob],
        start: &[f64],
        learning_rate: f64,
        perturbation: f64,
        seed: u64,
    ) -> Self {
        let free = knobs
            .iter()
            .zip(start)
            .map(|(k, &v)| k.free(v))
            .collect::<Vec<_>>();
        Self {
            knobs,
            moment: vec![0.0; free.len()],
            velocity: vec![0.0; free.len()],
            free,
            learning_rate,
            perturbation,
            step: 0,
            rng: Rng::new(seed),
        }
    }

    pub fn values(&self) -> Vec<f64> {
        self.knobs
            .iter()
            .zip(&self.free)
            .map(|(k, &f)| k.value(f))
            .collect()
    }

    fn values_at(&self, free: &[f64]) -> Vec<f64> {
        self.knobs
            .iter()
            .zip(free)
            .map(|(k, &f)| k.value(f))
            .collect()
    }

    pub fn step(&mut self, objective: &mut impl FnMut(&[f64], u64) -> f64) -> StepReport {
        self.step += 1;
        let size = self.perturbation / (self.step as f64).powf(PERTURBATION_DECAY);
        let signs: Vec<f64> = (0..self.free.len()).map(|_| self.rng.sign()).collect();
        let batch = self.rng.next_u64();

        let shifted = |scale: f64, free: &[f64], signs: &[f64]| -> Vec<f64> {
            free.iter()
                .zip(signs)
                .map(|(&f, &s)| f + scale * s)
                .collect()
        };
        let plus = shifted(size, &self.free, &signs);
        let minus = shifted(-size, &self.free, &signs);
        let loss_plus = objective(&self.values_at(&plus), batch);
        let loss_minus = objective(&self.values_at(&minus), batch);

        let difference = (loss_plus - loss_minus) / (2.0 * size);
        let bias1 = 1.0 - ADAM_BETA1.powi(self.step as i32);
        let bias2 = 1.0 - ADAM_BETA2.powi(self.step as i32);
        for (((free, moment), velocity), &sign) in self
            .free
            .iter_mut()
            .zip(&mut self.moment)
            .zip(&mut self.velocity)
            .zip(&signs)
        {
            let gradient = difference / sign;
            *moment = ADAM_BETA1 * *moment + (1.0 - ADAM_BETA1) * gradient;
            *velocity = ADAM_BETA2 * *velocity + (1.0 - ADAM_BETA2) * gradient * gradient;
            let corrected = *moment / bias1;
            let scale = (*velocity / bias2).sqrt() + ADAM_EPSILON;
            *free = (*free - self.learning_rate * corrected / scale).clamp(-FREE_LIMIT, FREE_LIMIT);
        }

        StepReport {
            loss: 0.5 * (loss_plus + loss_minus),
            perturbation: size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static KNOBS: [Knob; 2] = [
        Knob {
            name: "linear",
            lo: -10.0,
            hi: 10.0,
            scale: Scale::Linear,
        },
        Knob {
            name: "log",
            lo: 100.0,
            hi: 16_000.0,
            scale: Scale::Log,
        },
    ];

    #[test]
    fn a_knob_roundtrips_through_its_free_coordinate() {
        for (knob, value) in [(KNOBS[0], 3.5), (KNOBS[1], 4_200.0)] {
            let got = knob.value(knob.free(value));
            assert!(
                (got - value).abs() < 1.0e-6 * value.abs().max(1.0),
                "{} roundtripped {value} to {got}",
                knob.name
            );
        }
    }

    #[test]
    fn a_knob_never_leaves_its_range() {
        for knob in KNOBS {
            for free in [-1.0e3, -FREE_LIMIT, 0.0, FREE_LIMIT, 1.0e3] {
                let v = knob.value(free);
                assert!(v >= knob.lo && v <= knob.hi, "{} produced {v}", knob.name);
            }
        }
    }

    #[test]
    fn descent_finds_the_minimum_of_a_noisy_quadratic() {
        let target = [2.0, 5_000.0];
        let mut spsa = Spsa::new(&KNOBS, &[-6.0, 400.0], 0.15, 0.4, 7);
        let mut noise = Rng::new(99);
        let mut objective = |v: &[f64], _batch: u64| {
            let a = (v[0] - target[0]) / 10.0;
            let b = (v[1].ln() - target[1].ln()) / 5.0;
            let jitter = (noise.next_u64() % 1000) as f64 / 1.0e6;
            a * a + b * b + jitter
        };
        let before = {
            let v = spsa.values();
            objective(&v, 0)
        };
        for _ in 0..600 {
            spsa.step(&mut objective);
        }
        let after = {
            let v = spsa.values();
            objective(&v, 0)
        };
        assert!(
            after < 0.05 * before,
            "loss only fell from {before} to {after}"
        );
    }
}

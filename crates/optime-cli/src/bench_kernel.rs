//! Times the resampler kernels on their own, with no engine around them. `bench-resample` measures
//! what a song costs, which is the number that matters but also the number that moves least: voices
//! come and go, the mixer and filters take their share, and a kernel that got twice as fast shows up
//! as a few percent. This drives `Resampler::resample` directly over a synthetic source at a fixed
//! ratio, so a change to the kernel is the only thing the stopwatch can see.
//!
//! The scenario grid is the one the cutoff rule actually produces: clean mode upsampling sits at
//! exactly `fc = 0.5` (the common case — every GBA and most DS voices), clean mode downsampling at
//! `0.5/r`, and step mode at `0.5/r` either way. Contenders are interleaved round by round with a
//! rotating start order for the same reason `bench-resample` does it.

use std::process::ExitCode;
use std::time::Instant;

use clap::Args as ClapArgs;
use optime_core::Resampler;

use crate::resampler_roster::{self, ResamplerVisitor};

const SOURCE_LEN: usize = 8192;

#[derive(ClapArgs)]
#[command(about = "Time every resampler kernel directly, with no engine around it.")]
pub struct Args {
    #[arg(default_value_t = 64)]
    half_taps: usize,
    #[arg(long, default_value_t = 5)]
    rounds: usize,
    #[arg(long, default_value_t = 200_000)]
    samples: usize,
}

struct Scenario {
    name: &'static str,
    ratio: f32,
    step_mode: bool,
}

const SCENARIOS: [Scenario; 4] = [
    Scenario {
        name: "clean up",
        ratio: 0.35,
        step_mode: false,
    },
    Scenario {
        name: "clean down",
        ratio: 1.4,
        step_mode: false,
    },
    Scenario {
        name: "step up",
        ratio: 0.35,
        step_mode: true,
    },
    Scenario {
        name: "step down",
        ratio: 1.4,
        step_mode: true,
    },
];

fn cutoff(scenario: &Scenario) -> f32 {
    if scenario.step_mode || scenario.ratio > 1.0 {
        0.5 / scenario.ratio
    } else {
        0.5
    }
}

fn source() -> Vec<f32> {
    let mut seed = 0x2545_F491u32;
    (0..SOURCE_LEN)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5
        })
        .collect()
}

trait Contender {
    fn implementation(&self) -> &'static str;
    fn run(&self, scenario: &Scenario, samples: usize) -> f64;
}

struct Kernel<R: Resampler> {
    implementation: &'static str,
    tables: R::Tables,
    source: Vec<f32>,
}

impl<R: Resampler> Contender for Kernel<R> {
    fn implementation(&self) -> &'static str {
        self.implementation
    }

    fn run(&self, scenario: &Scenario, samples: usize) -> f64 {
        let fc = cutoff(scenario);
        let half_taps = R::half_taps(&self.tables) as f32;
        let span = self.source.len() as f32 - 2.0 * half_taps - 4.0;
        let mut pos = half_taps + 1.0;
        let mut state = R::State::default();
        let mut acc = 0.0f64;
        for _ in 0..samples {
            let (lo, hi) = R::tap_window(&self.tables, pos);
            let window = &self.source[lo as usize..=hi as usize];
            acc += f64::from(R::resample(
                &self.tables,
                &mut state,
                window,
                pos,
                fc,
                scenario.step_mode,
            ));
            pos += scenario.ratio;
            if pos - half_taps - 1.0 >= span {
                pos -= span;
            }
        }
        acc
    }
}

struct Build {
    half_taps: usize,
}

impl ResamplerVisitor for Build {
    type Output = Box<dyn Contender>;

    fn visit<R: Resampler + 'static>(&mut self, name: &'static str) -> Box<dyn Contender> {
        Box::new(Kernel::<R> {
            implementation: name,
            tables: R::tables(self.half_taps),
            source: source(),
        })
    }
}

pub fn run(args: Args) -> ExitCode {
    let Args {
        half_taps,
        rounds,
        samples,
    } = args;

    let contenders = resampler_roster::walk(&mut Build { half_taps });

    let mut sink = 0.0;
    for c in &contenders {
        for scenario in &SCENARIOS {
            sink += c.run(scenario, samples / 8);
        }
    }

    let mut walls = vec![vec![Vec::with_capacity(rounds); SCENARIOS.len()]; contenders.len()];
    for round in 0..rounds {
        for (s, scenario) in SCENARIOS.iter().enumerate() {
            for offset in 0..contenders.len() {
                let i = (round + offset) % contenders.len();
                let start = Instant::now();
                sink += contenders[i].run(scenario, samples);
                walls[i][s].push(start.elapsed().as_secs_f64());
            }
        }
    }

    println!("kernel only  taps={half_taps}  {rounds} rounds × {samples} samples, round-robin");
    print!("  {:<12}", "");
    for scenario in &SCENARIOS {
        print!("{:>14}", scenario.name);
    }
    println!();
    for (c, wall) in contenders.iter().zip(&mut walls) {
        print!("  {:<12}", c.implementation());
        for times in wall.iter_mut() {
            times.sort_by(f64::total_cmp);
            print!("{:>11.1} ns", times[0] / samples as f64 * 1e9);
        }
        println!();
    }
    println!("  (best round of each; sink {sink:.1})");
    ExitCode::SUCCESS
}

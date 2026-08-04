//! Times every `Resampler` implementation against the same song in one process. Each implementation
//! and resample mode is a contender rendering its own copy of the song, and the run interleaves them
//! round by round with a rotating start order, so a clock that drops partway through the run —
//! thermal throttling, a boost budget expiring, another process arriving — lands on all of them
//! alike instead of on whichever one happened to be measured last. Reported per contender: the
//! median round, and the best round, which is the least heat-contaminated sample of the set.

use std::process::ExitCode;
use std::time::Instant;

use clap::Args as ClapArgs;
use optime_core::{
    InstrumentResampleChoice, InstrumentResampleMode, InstrumentResampleSettings,
    PerDeviceSettings, PopSmoothingEdge, ResampleImplSimd, ResampleImplSimdClosedForm, Resampler,
    SoundData, SynthController, load_all,
};

const SAMPLE_RATE: f64 = 48_000.0;
const CHUNK_FRAMES: u64 = 1024;

#[derive(ClapArgs)]
#[command(about = "Time every resampler implementation on one song, round-robin.")]
pub struct Args {
    #[arg(default_value_t = 1025)]
    sseq_id: u32,
    #[arg(default_value_t = 64)]
    half_taps: usize,
    #[arg(long, default_value_t = 7)]
    rounds: usize,
    #[arg(long, default_value_t = 3.0)]
    round_seconds: f64,
}

trait Contender {
    fn implementation(&self) -> &'static str;
    fn mode(&self) -> &'static str;
    fn render(&mut self, chunks: u64) -> f64;
}

struct Rendered<R: Resampler> {
    implementation: &'static str,
    mode: &'static str,
    controller: SynthController<R>,
    config: PerDeviceSettings,
    buf: Vec<f32>,
}

impl<R: Resampler> Contender for Rendered<R> {
    fn implementation(&self) -> &'static str {
        self.implementation
    }

    fn mode(&self) -> &'static str {
        self.mode
    }

    fn render(&mut self, chunks: u64) -> f64 {
        let mut acc = 0.0;
        for _ in 0..chunks {
            self.controller.fill(&mut self.buf, &self.config);
            acc += self.buf.iter().map(|&v| f64::from(v)).sum::<f64>();
        }
        acc
    }
}

fn contender<R: Resampler + 'static>(
    implementation: &'static str,
    mode: &'static str,
    instrument_resample: InstrumentResampleSettings,
    data: &dyn SoundData,
    sseq_id: u32,
) -> Option<Box<dyn Contender>> {
    let controller = SynthController::<R>::with_resampler(SAMPLE_RATE, data, sseq_id)?;
    Some(Box::new(Rendered {
        implementation,
        mode,
        controller,
        config: PerDeviceSettings {
            instrument_resample,
            ..PerDeviceSettings::neutral()
        },
        buf: vec![0.0f32; 2 * CHUNK_FRAMES as usize],
    }))
}

fn settings(choice: InstrumentResampleChoice, sinc_taps: usize) -> InstrumentResampleSettings {
    InstrumentResampleSettings {
        choice,
        sinc_taps,
        psg_cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
        sampler_cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
        smooth_psg_pops: false,
        smooth_sample_pops: false,
        pop_slew_ms: 2.0,
        pop_smooth_edge: PopSmoothingEdge::Both,
    }
}

pub fn run(args: Args) -> ExitCode {
    let Args {
        sseq_id,
        half_taps,
        rounds,
        round_seconds,
    } = args;

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../demos/pokemon-platinum.sdat"
    );
    let bytes = std::fs::read(path).expect("read pokemon-platinum.sdat");
    let archives = load_all(&bytes);
    let Some(data) = archives
        .iter()
        .find(|s| SynthController::new(SAMPLE_RATE, &***s, sseq_id).is_some())
    else {
        eprintln!("SSEQ {sseq_id} not found");
        return ExitCode::FAILURE;
    };
    let data = &**data;

    let sinc_taps = half_taps * 2;
    let clean = settings(InstrumentResampleChoice::SincSampleNyquist, sinc_taps);
    let crunch = settings(InstrumentResampleChoice::SincOutputNyquist, sinc_taps);
    let mut contenders: Vec<Box<dyn Contender>> = [
        contender::<ResampleImplSimd>("simd", "clean", clean.clone(), data, sseq_id),
        contender::<ResampleImplSimdClosedForm>("simd/closed", "clean", clean, data, sseq_id),
        contender::<ResampleImplSimd>("simd", "crunch", crunch.clone(), data, sseq_id),
        contender::<ResampleImplSimdClosedForm>("simd/closed", "crunch", crunch, data, sseq_id),
    ]
    .into_iter()
    .flatten()
    .collect();

    let chunks_per_round = ((SAMPLE_RATE * round_seconds) as u64).div_ceil(CHUNK_FRAMES);
    let rendered_seconds = (chunks_per_round * CHUNK_FRAMES) as f64 / SAMPLE_RATE;
    let mut sink = 0.0;
    for c in &mut contenders {
        sink += c.render(chunks_per_round);
    }

    let mut walls = vec![Vec::with_capacity(rounds); contenders.len()];
    for round in 0..rounds {
        for offset in 0..contenders.len() {
            let i = (round + offset) % contenders.len();
            let start = Instant::now();
            sink += contenders[i].render(chunks_per_round);
            walls[i].push(start.elapsed().as_secs_f64());
        }
    }

    println!(
        "SSEQ {sseq_id}  taps={half_taps}  {rounds} rounds × {rendered_seconds:.1}s, round-robin"
    );
    for (c, wall) in contenders.iter().zip(&mut walls) {
        wall.sort_by(f64::total_cmp);
        let median = rendered_seconds / wall[wall.len() / 2];
        let best = rendered_seconds / wall[0];
        println!(
            "  {:<12} {:<7}  median {median:7.2}× real-time   best {best:7.2}×",
            c.implementation(),
            c.mode()
        );
    }
    println!("  (sink {sink:.1})");
    ExitCode::SUCCESS
}

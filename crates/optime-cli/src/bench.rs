//! Offline timing harness for the sinc resampler hot path.
//!
//! Renders a fixed wall of audio for a given SSEQ at a chosen sinc tap count and reports the
//! real-time factor (rendered seconds / wall seconds). A factor well above 1× means the synth
//! has real-time headroom; below 1× means it cannot keep up and the audio callback will underrun.
//!
//! Defaults to SSEQ 1025 from `demos/pokemon-platinum.sdat` at 64 taps (per side). Run it
//! `--release` or the numbers are meaningless.
//!
//! The default build uses the nightly portable-SIMD gather plus `target-cpu=native` (see
//! `.cargo/config.toml`); add `--no-default-features` to time the scalar gather instead.

use std::process::ExitCode;
use std::time::Instant;

use clap::Args as ClapArgs;
use optime_core::{
    InstrumentResampleChoice, InstrumentResampleMode, InstrumentResampleSettings,
    PerDeviceSettings, PopSmoothingEdge, SynthController, load_all,
};

#[derive(ClapArgs)]
#[command(about = "Time the sinc resampler hot path and report the real-time factor.")]
pub struct Args {
    /// The SSEQ to render.
    #[arg(default_value_t = 1025)]
    sseq_id: u32,
    /// Sinc taps per side.
    #[arg(default_value_t = 64)]
    half_taps: usize,
}

pub fn run(args: Args) -> ExitCode {
    let Args { sseq_id, half_taps } = args;

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../demos/pokemon-platinum.sdat"
    );
    let bytes = std::fs::read(path).expect("read pokemon-platinum.sdat");
    let archives = load_all(&bytes);
    // Pick the first archive that actually contains the requested SSEQ.
    let data = archives
        .iter()
        .find(|s| SynthController::new(48_000.0, &***s, sseq_id).is_some())
        .expect("an archive containing the requested SSEQ");

    let sr = 48_000.0;
    let render_secs = 20.0;
    let total = (sr * render_secs) as u64;

    // `half_taps` maps to the settings-level `sinc_taps` (total taps = 2 × half-taps).
    let sinc_taps = half_taps * 2;
    for (name, instrument_resample) in [
        (
            "clean (SampleNyquist)",
            InstrumentResampleSettings {
                choice: InstrumentResampleChoice::SincSampleNyquist,
                sinc_taps,
                psg_cutoff_hz: 0,
                sampler_cutoff_hz: 0,
                smooth_psg_pops: false,
                smooth_sample_pops: false,
                pop_slew_ms: 2.0,
                pop_smooth_edge: PopSmoothingEdge::Both,
            },
        ),
        (
            "crunch (OutputNyquist)",
            InstrumentResampleSettings {
                choice: InstrumentResampleChoice::SincOutputNyquist,
                sinc_taps,
                psg_cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
                sampler_cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
                smooth_psg_pops: false,
                smooth_sample_pops: false,
                pop_slew_ms: 2.0,
                pop_smooth_edge: PopSmoothingEdge::Both,
            },
        ),
    ] {
        let config = PerDeviceSettings {
            instrument_resample,
            ..PerDeviceSettings::neutral()
        };
        let Some(mut controller) = SynthController::new(sr, &**data, sseq_id) else {
            eprintln!("SSEQ {sseq_id} not found");
            return ExitCode::FAILURE;
        };
        // Render through `fill` (the block path the app's audio callback uses), in chunks the
        // size of a typical device buffer.
        let mut buf = vec![0.0f32; 2 * 1024];
        let chunks = |samples: u64| samples.div_ceil(1024);

        // Warm up the tables / steady-state polyphony before timing.
        for _ in 0..chunks(sr as u64) {
            controller.fill(&mut buf, &config);
        }

        let start = Instant::now();
        let mut acc = 0.0f64;
        for _ in 0..chunks(total) {
            controller.fill(&mut buf, &config);
            acc += buf.iter().map(|&v| f64::from(v)).sum::<f64>();
        }
        let wall = start.elapsed().as_secs_f64();
        let rtf = render_secs / wall;
        println!(
            "SSEQ {sseq_id}  {name:<24}  taps={half_taps:<3}  \
             {render_secs:.0}s in {wall:.3}s  →  {rtf:.2}× real-time  (sink {acc:.1})"
        );
    }
    ExitCode::SUCCESS
}

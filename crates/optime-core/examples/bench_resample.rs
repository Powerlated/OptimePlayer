//! Offline timing harness for the sinc resampler hot path.
//!
//! Renders a fixed wall of audio for a given SSEQ at a chosen sinc tap count and reports the
//! real-time factor (rendered seconds / wall seconds). A factor well above 1× means the synth
//! has real-time headroom; below 1× means it cannot keep up and the audio callback will underrun.
//!
//! Usage: `cargo run --release -p optime-core --example bench_resample [sseq_id] [half_taps]`
//! Defaults to SSEQ 1025 from `demos/pokemon-platinum.sdat` at 64 taps (per side).
//!
//! The default build uses the nightly portable-SIMD gather plus `target-cpu=native` (see
//! `.cargo/config.toml`); add `--no-default-features` to time the scalar gather instead.

use std::time::Instant;

use optime_core::{ResampleMode, SoundData, SynthConfig, SynthController};

fn main() {
    let mut args = std::env::args().skip(1);
    let sseq_id: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1025);
    let half_taps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../demos/pokemon-platinum.sdat"
    );
    let bytes = std::fs::read(path).expect("read pokemon-platinum.sdat");
    let archives = SoundData::load_all(&bytes);
    // Pick the first archive that actually contains the requested SSEQ.
    let data = archives
        .iter()
        .find(|s| SynthController::new(48_000.0, s, sseq_id).is_some())
        .expect("an archive containing the requested SSEQ");

    let sr = 48_000.0;
    let render_secs = 20.0;
    let total = (sr * render_secs) as u64;

    for (name, mode) in [
        (
            "clean (SampleNyquist)",
            ResampleMode::SincSampleNyquist { half_taps },
        ),
        (
            "crunch (OutputNyquist)",
            ResampleMode::SincOutputNyquist {
                half_taps,
                psg_cutoff_hz: ResampleMode::CUTOFF_OFF_HZ,
                sampler_cutoff_hz: ResampleMode::CUTOFF_OFF_HZ,
            },
        ),
    ] {
        let config = SynthConfig {
            resample: mode,
            ..SynthConfig::default()
        };
        let Some(mut controller) = SynthController::new(sr, data, sseq_id) else {
            eprintln!("SSEQ {sseq_id} not found");
            return;
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
}

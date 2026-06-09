//! Offline timing harness for the sinc resampler hot path.
//!
//! Renders a fixed wall of audio for a given SSEQ at a chosen sinc tap count and reports the
//! real-time factor (rendered seconds / wall seconds). A factor well above 1× means the synth
//! has real-time headroom; below 1× means it cannot keep up and the audio callback will underrun.
//!
//! Usage: `cargo run --release -p optime-core --example bench_resample [sseq_id] [half_taps]`
//! Defaults to SSEQ 1025 from `demos/pokemon-platinum.sdat` at 64 taps.

use std::time::Instant;

use optime_core::{Controller, ResampleMode, Sdat, SynthConfig};

fn main() {
    let mut args = std::env::args().skip(1);
    let sseq_id: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1025);
    let half_taps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../demos/pokemon-platinum.sdat"
    );
    let bytes = std::fs::read(path).expect("read pokemon-platinum.sdat");
    let sdats = Sdat::load_all(&bytes);
    // Pick the first archive that actually contains the requested SSEQ.
    let sdat = sdats
        .iter()
        .find(|s| Controller::new(48_000.0, s, sseq_id).is_some())
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
            ResampleMode::SincOutputNyquist { half_taps },
        ),
    ] {
        let config = SynthConfig {
            resample: mode,
            ..SynthConfig::default()
        };
        let Some(mut controller) = Controller::new(sr, sdat, sseq_id) else {
            eprintln!("SSEQ {sseq_id} not found");
            return;
        };
        // Warm up the tables / steady-state polyphony before timing.
        for _ in 0..(sr as u64) {
            controller.next_sample(&config);
        }

        let start = Instant::now();
        let mut acc = 0.0f64;
        for _ in 0..total {
            let (l, r) = controller.next_sample(&config);
            acc += f64::from(l) + f64::from(r);
        }
        let wall = start.elapsed().as_secs_f64();
        let rtf = render_secs / wall;
        println!(
            "SSEQ {sseq_id}  {name:<24}  taps={half_taps:<3}  \
             {render_secs:.0}s in {wall:.3}s  →  {rtf:.2}× real-time  (sink {acc:.1})"
        );
    }
}

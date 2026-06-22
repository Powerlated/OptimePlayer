//! Golden-output harness: renders a fixed slice of audio from every demo SDAT under a few
//! `SynthConfig` variants and prints an FNV-1a hash of the rendered f32 bits. Used to verify that
//! refactors are behavior-preserving (the hashes must not change). Not a correctness oracle —
//! just a bit-for-bit baseline of the current engine.
//!
//! Run with `cargo run -p optime-core --example golden` (add `--no-default-features` to capture a
//! baseline for the scalar gather; the SIMD and scalar builds hash differently by design).

use optime_core::{InstrumentResampleMode, SoundData, SynthConfig, SynthController, TuningSystem};

const SAMPLE_RATE: f64 = 32768.0;
const FRAMES: usize = 32768 * 4; // ~4 seconds of stereo audio per config.

fn fnv1a(hash: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *hash ^= u64::from(b);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
}

/// Renders `FRAMES` stereo frames of song 0 from `data` under `config`, folding every output
/// f32's raw bits into `hash`.
fn render_into(hash: &mut u64, data: &SoundData, config: &SynthConfig) {
    let Some(mut ctrl) = SynthController::new(SAMPLE_RATE, data, 0) else {
        fnv1a(hash, b"<no-controller>");
        return;
    };
    let mut out = vec![0.0f32; FRAMES * 2];
    ctrl.fill(&mut out, config);
    for s in &out {
        fnv1a(hash, &s.to_bits().to_le_bytes());
    }
}

fn configs() -> Vec<(&'static str, SynthConfig)> {
    vec![
        ("default", SynthConfig::default()),
        (
            "sinc+stereo",
            SynthConfig {
                stereo_separation: true,
                bass_mono: true,
                tuning: TuningSystem::Pure { tonic: 0 },
                resample: InstrumentResampleMode::SincSampleNyquist { half_taps: 16 },
                ..SynthConfig::default()
            },
        ),
        (
            "crunch",
            SynthConfig {
                resample: InstrumentResampleMode::SincOutputNyquist {
                    half_taps: 8,
                    psg_cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
                    sampler_cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
                },
                ..SynthConfig::default()
            },
        ),
    ]
}

fn main() {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let demos_dir = manifest.join("../../demos");
    let mut demos: Vec<_> = std::fs::read_dir(&demos_dir)
        .expect("read demos dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "sdat"))
        .collect();
    demos.sort();

    for path in demos {
        let bytes = std::fs::read(&path).expect("read demo");
        let archives = SoundData::load_all(&bytes);
        let name = path.file_name().unwrap().to_string_lossy();
        for (cfg_name, cfg) in configs() {
            let mut hash = 0xcbf2_9ce4_8422_2325u64;
            for data in &archives {
                render_into(&mut hash, data, &cfg);
            }
            println!("{name:32} {cfg_name:14} {hash:016x}");
        }
    }
}

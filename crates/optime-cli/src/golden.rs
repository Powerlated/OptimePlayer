use clap::Args as ClapArgs;
use optime_core::{
    InstrumentResampleChoice, InstrumentResampleMode, InstrumentResampleSettings,
    PerDeviceSettings, PopSmoothingEdge, SoundData, SynthController, load_all,
};

const SAMPLE_RATE: f64 = 32768.0;
const FRAMES: usize = 32768 * 4;

fn fnv1a(hash: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *hash ^= u64::from(b);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
}

fn render_into(hash: &mut u64, data: &dyn SoundData, config: &PerDeviceSettings) {
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

fn configs() -> Vec<(&'static str, PerDeviceSettings)> {
    vec![
        ("default", PerDeviceSettings::neutral()),
        (
            "sinc+stereo",
            PerDeviceSettings {
                stereo_separation: true,
                bass_mono: true,
                tuning_choice: 1,
                pure_tonic: 0,
                instrument_resample: InstrumentResampleSettings {
                    choice: InstrumentResampleChoice::SincSampleNyquist,
                    sinc_taps: 32,
                    psg_cutoff_hz: 0,
                    sampler_cutoff_hz: 0,
                    smooth_psg_pops: false,
                    smooth_sample_pops: false,
                    pop_slew_ms: 2.0,
                    pop_smooth_edge: PopSmoothingEdge::Both,
                },
                ..PerDeviceSettings::neutral()
            },
        ),
        (
            "crunch",
            PerDeviceSettings {
                instrument_resample: InstrumentResampleSettings {
                    choice: InstrumentResampleChoice::SincOutputNyquist,
                    sinc_taps: 16,
                    psg_cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
                    sampler_cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
                    smooth_psg_pops: false,
                    smooth_sample_pops: false,
                    pop_slew_ms: 2.0,
                    pop_smooth_edge: PopSmoothingEdge::Both,
                },
                ..PerDeviceSettings::neutral()
            },
        ),
    ]
}

#[derive(ClapArgs)]
#[command(
    about = "Hash a fixed render of every demo SDAT under several settings (refactor guard)."
)]
pub struct Args {
    #[arg(long)]
    demos: Option<std::path::PathBuf>,
}

pub fn run(args: Args) -> std::process::ExitCode {
    let demos_dir = args.demos.unwrap_or_else(|| {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../demos")
    });
    let mut demos: Vec<_> = std::fs::read_dir(&demos_dir)
        .expect("read demos dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "sdat"))
        .collect();
    demos.sort();

    for path in demos {
        let bytes = std::fs::read(&path).expect("read demo");
        let archives = load_all(&bytes);
        let name = path.file_name().unwrap().to_string_lossy();
        for (cfg_name, cfg) in configs() {
            let mut hash = 0xcbf2_9ce4_8422_2325u64;
            for data in &archives {
                render_into(&mut hash, &**data, &cfg);
            }
            println!("{name:32} {cfg_name:14} {hash:016x}");
        }
    }
    std::process::ExitCode::SUCCESS
}

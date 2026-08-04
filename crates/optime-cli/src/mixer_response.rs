//! Writes mixer-bus renders at two resample settings, the measurement the compensation filter is fit to.

use std::io::Write as _;
use std::process::ExitCode;

use clap::Args as ClapArgs;
use optime_core::{
    InstrumentResampleMode, PerDeviceSettings, StreamResampler, SynthController, load_all,
};

const MIXER_RATE: f64 = 32_768.0;
const OUT_RATE: f64 = 48_000.0;
const HALF_TAPS: usize = 32;
const CRUNCH_CUTOFF_HZ: u32 = 15_000;
const SECS_PER_SONG: f64 = 8.0;

const SONGS: &[u32] = &[
    548, 413, 465, 403, 429, 444, 374, 539, 398, 474, 538, 479, 524,
];

fn mixer_config() -> PerDeviceSettings {
    PerDeviceSettings {
        use_mixer: true,
        mixer_sample_rate: MIXER_RATE as u32,
        stereo_separation: false,
        bass_mono: false,
        ..PerDeviceSettings::neutral()
    }
}

fn resample_bus(bus: &[(f32, f32)], mode: InstrumentResampleMode) -> Vec<f32> {
    let mut rs = StreamResampler::new();
    rs.set(MIXER_RATE as f32, OUT_RATE as f32, mode);
    let n_out = ((bus.len() as f64) * OUT_RATE / MIXER_RATE).floor() as usize;
    let mut idx = 0usize;
    let mut pull = |l: &mut [f32], r: &mut [f32]| {
        for (l, r) in l.iter_mut().zip(r.iter_mut()) {
            (*l, *r) = bus.get(idx).copied().unwrap_or((0.0, 0.0));
            idx += 1;
        }
    };
    let (mut out_l, mut out_r) = (vec![0.0f32; n_out], vec![0.0f32; n_out]);
    rs.process(&mut out_l, &mut out_r, &mut pull);
    out_l
        .iter()
        .zip(&out_r)
        .map(|(&l, &r)| (l + r) * 0.5)
        .collect()
}

#[derive(ClapArgs)]
#[command(about = "Capture the mixer-to-output resampler's effect on real DirectSound content.")]
pub struct Args {
    #[arg(default_value = ".")]
    out_dir: String,
}

pub fn run(args: Args) -> ExitCode {
    let out_dir = args.out_dir;

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../demos/pokemon-emerald.gbaaudio"
    );
    let bytes = std::fs::read(path).expect("read pokemon-emerald.gbaaudio");
    let archives = load_all(&bytes);

    let crunch = InstrumentResampleMode::SincOutputNyquist {
        half_taps: HALF_TAPS,
        psg_cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
        sampler_cutoff_hz: CRUNCH_CUTOFF_HZ,
    };
    let nearest = InstrumentResampleMode::NearestNeighbor;

    let frames = (MIXER_RATE * SECS_PER_SONG) as usize;
    let mut near_all: Vec<f32> = Vec::new();
    let mut crunch_all: Vec<f32> = Vec::new();

    for &song in SONGS {
        let Some(data) = archives
            .iter()
            .find(|s| SynthController::new(OUT_RATE, &***s, song).is_some())
        else {
            eprintln!("song {song}: not found, skipping");
            continue;
        };
        let mut controller =
            SynthController::new(OUT_RATE, &**data, song).expect("controller for song");

        let cfg = mixer_config();
        let mut buf = vec![0.0f32; 2 * frames];
        controller.fill_mixer_bus(&mut buf, &cfg);
        let bus: Vec<(f32, f32)> = buf.chunks_exact(2).map(|c| (c[0], c[1])).collect();

        let rms = (bus
            .iter()
            .map(|&(l, r)| f64::from(l * l + r * r))
            .sum::<f64>()
            / (bus.len().max(1) as f64))
            .sqrt();
        println!(
            "song {song}: captured {} frames, bus rms {rms:.5}",
            bus.len()
        );

        near_all.extend(resample_bus(&bus, nearest));
        crunch_all.extend(resample_bus(&bus, crunch));
    }

    let write_f32 = |name: &str, data: &[f32]| {
        let p = format!("{out_dir}/{name}");
        let mut f = std::fs::File::create(&p).expect("create output");
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        f.write_all(&bytes).expect("write output");
        println!("wrote {p} ({} samples)", data.len());
    };
    write_f32("near.f32", &near_all);
    write_f32("crunch.f32", &crunch_all);

    let meta = format!(
        "mixer_rate={MIXER_RATE}\nout_rate={OUT_RATE}\nhalf_taps={HALF_TAPS}\n\
         crunch_cutoff_hz={CRUNCH_CUTOFF_HZ}\nsecs_per_song={SECS_PER_SONG}\n\
         songs={SONGS:?}\nsamples={}\n",
        near_all.len()
    );
    let meta_path = format!("{out_dir}/meta.txt");
    std::fs::write(&meta_path, meta).expect("write meta");
    println!("wrote {meta_path}");
    ExitCode::SUCCESS
}

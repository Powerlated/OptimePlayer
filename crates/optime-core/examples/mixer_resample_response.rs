//! Captures the mixer-to-output resampler's effect on **real DirectSound content** so MATLAB can
//! measure the spectral power loss of switching the mixer-to-output stage from "Nearest neighbor"
//! to "Sinc - output Nyquist (crunch)".
//!
//! For each Emerald song it:
//!   1. renders the isolated sampled (DirectSound) mixer bus at `MIXER_RATE`
//!      ([`SynthController::fill_mixer_bus`] — no PSG, no resampling), then
//!   2. feeds that exact bus through a [`StreamResampler`] twice — once nearest, once crunch — at
//!      `MIXER_RATE → OUT_RATE`, exactly as the engine's mixer-to-output stage does.
//!
//! The two output-rate streams (concatenated across all songs, mono = (L+R)/2) are written as raw
//! little-endian `f32` to the scratchpad, alongside a small meta file. Because the resampler is
//! linear its transfer function is input-independent; real songs only supply the *spectral
//! weighting* that makes the "average" power-loss number reflect actual musical content.
//!
//! Usage: `cargo run --release -p optime-core --example mixer_resample_response -- <out_dir>`

use std::io::Write as _;

use optime_core::{
    InstrumentResampleMode, PerDeviceSettings, StreamResampler, SynthController, load_all,
};

/// Mixer-bus (DirectSound) rate — deliberately below the output rate so nearest-neighbour ZOH
/// upsampling produces real imaging for the crunch low-pass to remove.
const MIXER_RATE: f64 = 32_768.0;
/// Output (audio device) rate.
const OUT_RATE: f64 = 48_000.0;
/// Sinc tap count (per side) for the crunch mode — the app's mixer default.
const HALF_TAPS: usize = 32;
/// Crunch low-pass cutoff (Hz) for the sampled bus — the app's mixer default (`MIXER_CUTOFF_HZ`).
const CRUNCH_CUTOFF_HZ: u32 = 15_000;
/// Seconds of DirectSound captured per song.
const SECS_PER_SONG: f64 = 8.0;

/// The Emerald song ids the user picked as test material.
const SONGS: &[u32] = &[
    548, 413, 465, 403, 429, 444, 374, 539, 398, 474, 538, 479, 524,
];

/// A config that captures the isolated mixer (DirectSound) bus at [`MIXER_RATE`]. The captured bus
/// is independent of the mixer→output resampler (`fill_mixer_bus` reads the bus directly), so the
/// neutral mixer_resample is fine; the per-voice resampling stays at neutral's nearest-neighbour.
fn mixer_config() -> PerDeviceSettings {
    PerDeviceSettings {
        use_mixer: true,
        mixer_sample_rate: MIXER_RATE as u32,
        // Mono-summed analysis: keep the stereo expander out of the captured bus.
        stereo_separation: false,
        bass_mono: false,
        ..PerDeviceSettings::neutral()
    }
}

/// Resamples a captured mixer-rate stereo bus to the output rate in `mode`, returning the mono
/// `(L+R)/2` output stream.
fn resample_bus(bus: &[(f32, f32)], mode: InstrumentResampleMode) -> Vec<f32> {
    let mut rs = StreamResampler::new();
    rs.set(MIXER_RATE as f32, OUT_RATE as f32, mode);
    let n_out = ((bus.len() as f64) * OUT_RATE / MIXER_RATE).floor() as usize;
    let mut idx = 0usize;
    let mut pull = || {
        let s = bus.get(idx).copied().unwrap_or((0.0, 0.0));
        idx += 1;
        s
    };
    let mut out = vec![(0.0f32, 0.0f32); n_out];
    rs.process(&mut out, &mut pull);
    out.iter().map(|&(l, r)| (l + r) * 0.5).collect()
}

fn main() {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());

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

        // Capture the isolated DirectSound bus at the mixer rate.
        let cfg = mixer_config(); // capture is mode-independent; reuse one config
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
}

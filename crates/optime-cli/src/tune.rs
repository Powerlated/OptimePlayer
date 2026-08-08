//! Tunes the exciter and the stages around it against a reference timbre profile, and reports what
//! that buys over the excitation the engine shipped with.
//!
//! The experiment this command exists to run: the crunch resampler generates the engine's top end
//! as a side effect of reconstructing a stepped source, which is a zero-order hold — the harmonics
//! are whatever the staircase happened to contain, at whatever level the staircase put them. The
//! alternative is to reconstruct the source cleanly and generate the top end on purpose, with a
//! saturating waveshaper whose amount, corner and drive are free parameters. Those two are set up
//! here as `Excitation::ZeroOrderHold` and `Excitation::Shaper`, both are scored untuned, and the
//! second is then fitted.
//!
//! Songs are split into a tuning set and a holdout the search never sees, because a dozen free
//! parameters against a handful of songs can fit the songs rather than the console. Only the
//! holdout number says whether the fit generalises, and both are reported side by side.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args as ClapArgs, ValueEnum};
use optime_core::devices::gba::GbaRom;
use optime_core::{
    LoopAndTransitionOptions, PerDeviceSettings, PlaybackEvent, SoundData, SynthController,
    load_all,
};
use rayon::prelude::*;

use crate::album::album_order;
use crate::search::{Knob, Rng, Scale, Spsa};
use crate::timbre::{self, Profile, Target};

static KNOBS: [Knob; 12] = [
    Knob {
        name: "exciter.crossover_hz",
        lo: 1_000.0,
        hi: 12_000.0,
        scale: Scale::Log,
    },
    Knob {
        name: "exciter.drive",
        lo: 0.5,
        hi: 24.0,
        scale: Scale::Log,
    },
    Knob {
        name: "exciter.amount",
        lo: 0.0,
        hi: 2.0,
        scale: Scale::Linear,
    },
    Knob {
        name: "high_band.cutoff_hz",
        lo: 2_000.0,
        hi: 16_000.0,
        scale: Scale::Log,
    },
    Knob {
        name: "high_band.threshold_db",
        lo: -60.0,
        hi: 0.0,
        scale: Scale::Linear,
    },
    Knob {
        name: "high_band.ratio",
        lo: 1.0,
        hi: 8.0,
        scale: Scale::Linear,
    },
    Knob {
        name: "high_band.attack_ms",
        lo: 0.5,
        hi: 50.0,
        scale: Scale::Log,
    },
    Knob {
        name: "high_band.release_ms",
        lo: 10.0,
        hi: 500.0,
        scale: Scale::Log,
    },
    Knob {
        name: "high_band.makeup_db",
        lo: -6.0,
        hi: 12.0,
        scale: Scale::Linear,
    },
    Knob {
        name: "shelf.cutoff_hz",
        lo: 1_000.0,
        hi: 14_000.0,
        scale: Scale::Log,
    },
    Knob {
        name: "shelf.gain_db",
        lo: -12.0,
        hi: 12.0,
        scale: Scale::Linear,
    },
    Knob {
        name: "shelf.q",
        lo: 0.3,
        hi: 2.0,
        scale: Scale::Log,
    },
];

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
pub enum Excitation {
    ZeroOrderHold,
    Shaper,
}

#[derive(ClapArgs)]
#[command(about = "Fit the exciter and its compressor/EQ to a reference timbre profile by SPSA.")]
pub struct Args {
    archive: PathBuf,
    names_json: PathBuf,
    reference: PathBuf,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, default_value_t = 150)]
    steps: usize,
    #[arg(long, default_value_t = 4)]
    batch: usize,
    #[arg(long, default_value_t = 30.0)]
    seconds: f64,
    #[arg(long, default_value_t = 48_000)]
    rate: u32,
    #[arg(long, default_value_t = 8)]
    holdout: usize,
    #[arg(long, default_value_t = 12)]
    eval_songs: usize,
    #[arg(long, default_value_t = 25)]
    eval_every: usize,
    #[arg(long, default_value_t = 0.1)]
    learning_rate: f64,
    #[arg(long, default_value_t = 0.3)]
    perturbation: f64,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[arg(long, value_enum, default_value_t = Excitation::Shaper)]
    start_from: Excitation,
    #[arg(long)]
    load: Option<PathBuf>,
}

fn base_settings(data: &dyn SoundData, excitation: Excitation) -> PerDeviceSettings {
    let is_gba = data.as_any().downcast_ref::<GbaRom>().is_some();
    let mut config = if is_gba {
        PerDeviceSettings::enhanced_gba()
    } else {
        PerDeviceSettings::high_quality_nintendo_ds()
    };
    match excitation {
        Excitation::ZeroOrderHold => config.exciter.enabled = false,
        Excitation::Shaper => {
            config.mixer_resample.choice = optime_core::InstrumentResampleChoice::SincSampleNyquist;
            config.psg_crunch_compensation = false;
            config.exciter.enabled = true;
        }
    }
    config
}

fn read_knobs(config: &PerDeviceSettings) -> Vec<f64> {
    vec![
        config.exciter.crossover_hz,
        f64::from(config.exciter.drive),
        f64::from(config.exciter.amount),
        config.high_band_compress.cutoff_hz,
        config.high_band_compress.threshold_db,
        config.high_band_compress.ratio,
        config.high_band_compress.attack_ms,
        config.high_band_compress.release_ms,
        config.high_band_compress.makeup_db,
        config.shelf.cutoff_hz,
        config.shelf.gain_db,
        config.shelf.q,
    ]
}

fn apply_knobs(config: &PerDeviceSettings, values: &[f64]) -> PerDeviceSettings {
    let mut config = config.clone();
    config.exciter.enabled = true;
    config.exciter.crossover_hz = values[0];
    config.exciter.drive = values[1] as f32;
    config.exciter.amount = values[2] as f32;
    config.high_band_compress.enabled_psg = true;
    config.high_band_compress.enabled_sampled = true;
    config.high_band_compress.cutoff_hz = values[3];
    config.high_band_compress.threshold_db = values[4];
    config.high_band_compress.ratio = values[5];
    config.high_band_compress.attack_ms = values[6];
    config.high_band_compress.release_ms = values[7];
    config.high_band_compress.makeup_db = values[8];
    config.shelf.enabled = true;
    config.shelf.cutoff_hz = values[9];
    config.shelf.gain_db = values[10];
    config.shelf.q = values[11];
    config
}

fn load_parameters(path: &std::path::Path) -> Result<Vec<f64>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    KNOBS
        .iter()
        .map(|k| {
            json["parameters"][k.name]
                .as_f64()
                .ok_or_else(|| format!("missing parameter '{}'", k.name))
        })
        .collect()
}

fn render_mono(
    data: &dyn SoundData,
    song_id: u32,
    config: &PerDeviceSettings,
    rate: u32,
    seconds: f64,
) -> Vec<f32> {
    let Some(mut controller) = SynthController::new(f64::from(rate), data, song_id) else {
        return Vec::new();
    };
    controller.set_loop_and_transition(LoopAndTransitionOptions::none());
    let wanted = (f64::from(rate) * seconds) as usize;
    const CHUNK_FRAMES: usize = 1024;
    let mut buf = vec![0.0f32; 2 * CHUNK_FRAMES];
    let mut mono = Vec::with_capacity(wanted);
    while mono.len() < wanted {
        let n = CHUNK_FRAMES.min(wanted - mono.len());
        let chunk = &mut buf[..2 * n];
        controller.fill(chunk, config);
        for frame in chunk.chunks_exact(2) {
            mono.push(0.5 * (frame[0] + frame[1]));
        }
        if controller
            .take_messages()
            .any(|m| m == PlaybackEvent::Finished)
        {
            break;
        }
    }
    mono
}

struct Objective<'a> {
    data: &'a dyn SoundData,
    target: &'a Target,
    rate: u32,
    seconds: f64,
}

impl Objective<'_> {
    fn profile(&self, song_id: u32, config: &PerDeviceSettings) -> Option<Profile> {
        let mono = render_mono(self.data, song_id, config, self.rate, self.seconds);
        timbre::analyze(&mono, f64::from(self.rate))
    }

    fn mean_spectrum(&self, songs: &[u32], config: &PerDeviceSettings) -> Vec<f32> {
        let profiles: Vec<Profile> = songs
            .par_iter()
            .filter_map(|&id| self.profile(id, config))
            .collect();
        (0..timbre::BAND_COUNT)
            .map(|b| {
                profiles.iter().map(|p| p.spectrum_db[b]).sum::<f32>()
                    / profiles.len().max(1) as f32
            })
            .collect()
    }

    fn score(&self, songs: &[u32], config: &PerDeviceSettings) -> f64 {
        let scored: Vec<f64> = songs
            .par_iter()
            .filter_map(|&id| {
                self.profile(id, config)
                    .map(|p| f64::from(self.target.distance(&p)))
            })
            .collect();
        if scored.is_empty() {
            return f64::MAX;
        }
        scored.iter().sum::<f64>() / scored.len() as f64
    }
}

fn spread(ids: &[u32], count: usize) -> Vec<u32> {
    if ids.len() <= count {
        return ids.to_vec();
    }
    (0..count)
        .map(|k| ids[((k as f64 + 0.5) * ids.len() as f64 / count as f64) as usize])
        .collect()
}

fn minibatch(ids: &[u32], size: usize, seed: u64) -> Vec<u32> {
    if ids.len() <= size {
        return ids.to_vec();
    }
    let mut rng = Rng::new(seed);
    let mut pool = ids.to_vec();
    for i in 0..size {
        let j = i + (rng.next_u64() as usize) % (pool.len() - i);
        pool.swap(i, j);
    }
    pool.truncate(size);
    pool
}

fn report_line(label: &str, train: f64, holdout: f64) {
    println!("  {label:<34} train {train:7.4}   holdout {holdout:7.4}");
}

pub fn run(args: Args) -> ExitCode {
    let bytes = match std::fs::read(&args.archive) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to read '{}': {e}", args.archive.display());
            return ExitCode::FAILURE;
        }
    };
    let Some(data) = load_all(&bytes).into_iter().next() else {
        eprintln!("No songs found in '{}'.", args.archive.display());
        return ExitCode::FAILURE;
    };
    let target: Target = match std::fs::read_to_string(&args.reference)
        .map_err(|e| e.to_string())
        .and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string()))
    {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to read '{}': {e}", args.reference.display());
            return ExitCode::FAILURE;
        }
    };
    let album = match album_order(&*data, &args.names_json, None) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let ids: Vec<u32> = album.iter().map(|(id, _)| *id).collect();
    let holdout_count = args.holdout.min(ids.len().saturating_sub(1));
    let holdout_ids = spread(&ids, holdout_count);
    let train_ids: Vec<u32> = ids
        .iter()
        .copied()
        .filter(|id| !holdout_ids.contains(id))
        .collect();
    let train_eval = spread(&train_ids, args.eval_songs);
    let holdout_eval = spread(&holdout_ids, args.eval_songs);

    println!(
        "Reference: {} recordings. Songs: {} tuning, {} holdout. Rendering {:.0}s each at {} Hz.",
        target.sources,
        train_ids.len(),
        holdout_ids.len(),
        args.seconds,
        args.rate
    );

    let objective = Objective {
        data: &*data,
        target: &target,
        rate: args.rate,
        seconds: args.seconds,
    };

    let zoh = base_settings(&*data, Excitation::ZeroOrderHold);
    let shaper_base = base_settings(&*data, Excitation::Shaper);
    let untuned = apply_knobs(&shaper_base, &read_knobs(&shaper_base));

    println!("\nBaselines:");
    report_line(
        "zero-order-hold excitation",
        objective.score(&train_eval, &zoh),
        objective.score(&holdout_eval, &zoh),
    );
    report_line(
        "shaper excitation, untuned",
        objective.score(&train_eval, &untuned),
        objective.score(&holdout_eval, &untuned),
    );

    let start = match &args.load {
        Some(path) => match load_parameters(path) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Failed to read '{}': {e}", path.display());
                return ExitCode::FAILURE;
            }
        },
        None => read_knobs(&base_settings(&*data, args.start_from)),
    };
    let mut spsa = Spsa::new(
        &KNOBS,
        &start,
        args.learning_rate,
        args.perturbation,
        args.seed,
    );

    println!(
        "\nTuning {} parameters over {} steps:",
        KNOBS.len(),
        args.steps
    );
    let mut best = {
        let values = spsa.values();
        (
            objective.score(&train_eval, &apply_knobs(&shaper_base, &values)),
            values,
        )
    };
    for step in 1..=args.steps {
        let report = {
            let mut evaluate = |values: &[f64], batch: u64| {
                let songs = minibatch(&train_ids, args.batch, batch);
                objective.score(&songs, &apply_knobs(&shaper_base, values))
            };
            spsa.step(&mut evaluate)
        };
        if step % args.eval_every == 0 || step == args.steps {
            let values = spsa.values();
            let train = objective.score(&train_eval, &apply_knobs(&shaper_base, &values));
            if train < best.0 {
                best = (train, values);
            }
            println!(
                "  step {step:4}  batch {:7.4}  eval {train:7.4}  (c {:.3})",
                report.loss, report.perturbation
            );
        }
    }

    let tuned = apply_knobs(&shaper_base, &best.1);
    let tuned_train = objective.score(&train_eval, &tuned);
    let tuned_holdout = objective.score(&holdout_eval, &tuned);

    let mut silenced = tuned.clone();
    silenced.exciter.amount = 0.0;

    println!("\nResult:");
    report_line("shaper excitation, tuned", tuned_train, tuned_holdout);
    report_line(
        "  same, exciter amount at zero",
        objective.score(&train_eval, &silenced),
        objective.score(&holdout_eval, &silenced),
    );

    println!("\nSpectrum on the holdout, dB about each render's own mean band:");
    let zoh_bands = objective.mean_spectrum(&holdout_eval, &zoh);
    let tuned_bands = objective.mean_spectrum(&holdout_eval, &tuned);
    println!("    band    target      zoh    tuned");
    for b in 0..timbre::BAND_COUNT {
        println!(
            "    {b:4}  {:+7.1}  {:+7.1}  {:+7.1}",
            target.mean.spectrum_db[b], zoh_bands[b], tuned_bands[b]
        );
    }

    println!("\nTuned parameters:");
    for (knob, value) in KNOBS.iter().zip(&best.1) {
        println!(
            "  {:<24} {value:10.3}   [{} .. {}]",
            knob.name, knob.lo, knob.hi
        );
    }

    if let Some(path) = &args.out {
        let json = serde_json::json!({
            "reference": args.reference.display().to_string(),
            "archive": args.archive.display().to_string(),
            "rate": args.rate,
            "seconds": args.seconds,
            "loss": { "train": tuned_train, "holdout": tuned_holdout },
            "parameters": KNOBS
                .iter()
                .zip(&best.1)
                .map(|(k, v)| (k.name.to_string(), serde_json::json!(v)))
                .collect::<serde_json::Map<String, serde_json::Value>>(),
            "settings": serde_json::to_value(&tuned).unwrap_or(serde_json::Value::Null),
        });
        match serde_json::to_string_pretty(&json) {
            Ok(text) => {
                if let Err(e) = std::fs::write(path, text) {
                    eprintln!("Failed to write '{}': {e}", path.display());
                    return ExitCode::FAILURE;
                }
                println!("\nWrote {}", path.display());
            }
            Err(e) => {
                eprintln!("Failed to serialise the result: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_knob_reads_back_what_it_wrote() {
        let base = PerDeviceSettings::enhanced_gba();
        let values: Vec<f64> = KNOBS.iter().map(|k| 0.5 * (k.lo + k.hi)).collect();
        let applied = apply_knobs(&base, &values);
        let read = read_knobs(&applied);
        for ((knob, want), got) in KNOBS.iter().zip(&values).zip(&read) {
            assert!(
                (want - got).abs() < 1.0e-3 * want.abs().max(1.0),
                "{} wrote {want} and read back {got}",
                knob.name
            );
        }
    }

    #[test]
    fn the_knob_table_covers_every_free_parameter() {
        assert_eq!(
            KNOBS.len(),
            read_knobs(&PerDeviceSettings::enhanced_gba()).len()
        );
    }
}

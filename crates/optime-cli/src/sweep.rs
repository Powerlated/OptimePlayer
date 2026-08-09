//! Sweeps the structural choices that decide whether a reconstruction notch exists at all, scoring
//! each one for both notch depth and timbre.
//!
//! The parameters `tune-exciter` searches are all continuous, and none of them can remove a null:
//! the null is put there by the *shape* of the chain — an intermediate mixer bus held at 13379 Hz
//! reconstructs as a staircase whose `sinc(f / 13379)` envelope is exactly zero at 13379 Hz, and no
//! setting of a drive or a shelf gain moves a zero. What moves it is the mixer rate, the
//! reconstruction mode, or whether the intermediate bus exists. Those are a handful of discrete
//! choices, so they are enumerated rather than descended, and each combination is reported with the
//! notch it leaves and the timbre distance it costs.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Args as ClapArgs;
use optime_core::{InstrumentResampleChoice, PerDeviceSettings, SoundData, load_all};
use rayon::prelude::*;

use crate::album::album_order;
use crate::spectrum::Spectrum;
use crate::timbre::{self, Target};
use crate::tune::{
    Excitation, apply_knobs, base_settings, load_parameters, read_knobs, render_mono,
};

#[derive(ClapArgs)]
#[command(about = "Sweep mixer rate / reconstruction / exciter for reconstruction notches.")]
pub struct Args {
    archive: PathBuf,
    names_json: PathBuf,
    reference: PathBuf,
    #[arg(long, default_value_t = 13_379.0)]
    probe_hz: f64,
    #[arg(long, default_value_t = 10)]
    songs: usize,
    #[arg(long, default_value_t = 20.0)]
    seconds: f64,
    #[arg(long, default_value_t = 48_000)]
    rate: u32,
    #[arg(long)]
    load: Option<PathBuf>,
}

type Row = (String, f32, f32, Vec<(f64, f32)>);

struct Variant {
    label: String,
    config: PerDeviceSettings,
}

fn variants(base: &PerDeviceSettings, tuned: &[f64]) -> Vec<Variant> {
    let mut out = Vec::new();
    for &use_mixer in &[true, false] {
        let rates: &[u32] = if use_mixer {
            &[13_379, 26_758, 32_768, 48_000]
        } else {
            &[0]
        };
        for &mixer_rate in rates {
            let modes: &[InstrumentResampleChoice] = if use_mixer {
                &[
                    InstrumentResampleChoice::SincOutputNyquist,
                    InstrumentResampleChoice::SincSampleNyquist,
                ]
            } else {
                &[InstrumentResampleChoice::SincSampleNyquist]
            };
            for mode in modes {
                for &exciter in &[false, true] {
                    let mut config = apply_knobs(base, tuned);
                    config.use_mixer = use_mixer;
                    if use_mixer {
                        config.mixer_sample_rate = mixer_rate;
                        config.mixer_resample.choice = mode.clone();
                        config.mixer_resample.cutoff_hz = mixer_rate;
                    }
                    config.psg_crunch_compensation = false;
                    config.exciter.enabled = exciter;
                    let where_ = if use_mixer {
                        format!(
                            "mixer {mixer_rate:>5} {}",
                            match mode {
                                InstrumentResampleChoice::SincOutputNyquist => "crunch",
                                _ => "clean ",
                            }
                        )
                    } else {
                        "no mixer bus     ".to_string()
                    };
                    out.push(Variant {
                        label: format!("{where_}  exciter {}", if exciter { "on " } else { "off" }),
                        config,
                    });
                }
            }
        }
    }
    out
}

struct Measured {
    notch_db: f32,
    level_db: f32,
    timbre: f32,
    worst: Vec<(f64, f32)>,
}

fn measure(
    data: &dyn SoundData,
    songs: &[u32],
    config: &PerDeviceSettings,
    target: &Target,
    rate: u32,
    seconds: f64,
    probe_hz: f64,
) -> Measured {
    let rendered: Vec<(Option<Spectrum>, Option<f32>)> = songs
        .par_iter()
        .map(|&id| {
            let mono = render_mono(data, id, config, rate, seconds);
            let spectrum = Spectrum::analyze(&mono, f64::from(rate));
            let distance = timbre::analyze(&mono, f64::from(rate)).map(|p| target.distance(&p));
            (spectrum, distance)
        })
        .collect();

    let distances: Vec<f32> = rendered.iter().filter_map(|(_, d)| *d).collect();
    let timbre = if distances.is_empty() {
        f32::NAN
    } else {
        distances.iter().sum::<f32>() / distances.len() as f32
    };

    let mut summed: Option<Spectrum> = None;
    for (spectrum, _) in &rendered {
        let Some(s) = spectrum else { continue };
        match &mut summed {
            Some(acc) => acc.accumulate(s),
            None => {
                summed = Some(Spectrum {
                    power: s.power.clone(),
                    rate: s.rate,
                })
            }
        }
    }
    match summed {
        Some(s) => Measured {
            notch_db: s.notch_depth_db(probe_hz),
            level_db: s.level_db(probe_hz),
            timbre,
            worst: s.deepest_notches(2_500.0, 20_000.0, 50.0, 3),
        },
        None => Measured {
            notch_db: f32::NAN,
            level_db: f32::NAN,
            timbre,
            worst: Vec::new(),
        },
    }
}

fn worst_text(worst: &[(f64, f32)]) -> String {
    worst
        .iter()
        .map(|(hz, db)| format!("{hz:.0}Hz{db:+.1}"))
        .collect::<Vec<_>>()
        .join(" ")
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
    let songs: Vec<u32> = (0..args.songs.min(ids.len()))
        .map(|k| ids[((k as f64 + 0.5) * ids.len() as f64 / args.songs as f64) as usize])
        .collect();

    let base = base_settings(&*data, Excitation::Shaper);
    let tuned = match &args.load {
        Some(path) => match load_parameters(path) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Failed to read '{}': {e}", path.display());
                return ExitCode::FAILURE;
            }
        },
        None => read_knobs(&PerDeviceSettings::enhanced_plus_gba()),
    };

    println!(
        "Probing {:.0} Hz over {} songs, {:.0}s each at {} Hz. Notch is shoulders minus centre, so positive is a dip.\n",
        args.probe_hz,
        songs.len(),
        args.seconds,
        args.rate
    );

    let shipped = [
        ("Enhanced   (shipped)", PerDeviceSettings::enhanced_gba()),
        (
            "Enhanced+  (shipped)",
            PerDeviceSettings::enhanced_plus_gba(),
        ),
        ("Original   (shipped)", PerDeviceSettings::original_gba()),
    ];
    println!(
        "{:<36}  {:>8}  {:>8}  {:>7}  deepest notches anywhere",
        "configuration", "notch dB", "level dB", "timbre"
    );
    for (label, config) in &shipped {
        let m = measure(
            &*data,
            &songs,
            config,
            &target,
            args.rate,
            args.seconds,
            args.probe_hz,
        );
        println!(
            "{label:<36}  {:8.2}  {:8.1}  {:7.4}  {}",
            m.notch_db,
            m.level_db,
            m.timbre,
            worst_text(&m.worst)
        );
    }

    println!();
    let mut rows: Vec<Row> = variants(&base, &tuned)
        .into_iter()
        .map(|v| {
            let m = measure(
                &*data,
                &songs,
                &v.config,
                &target,
                args.rate,
                args.seconds,
                args.probe_hz,
            );
            println!(
                "{:<36}  {:8.2}  {:8.1}  {:7.4}  {}",
                v.label,
                m.notch_db,
                m.level_db,
                m.timbre,
                worst_text(&m.worst)
            );
            (v.label, m.notch_db, m.timbre, m.worst)
        })
        .collect();

    rows.sort_by(|a, b| {
        let flattest = |r: &Row| r.3.first().map(|(_, db)| *db).unwrap_or(f32::MAX);
        flattest(a).total_cmp(&flattest(b))
    });
    println!("\nFlattest overall (ranked by the deepest notch anywhere, not by the probe):");
    for (label, notch, timbre, worst) in rows.iter().take(6) {
        println!(
            "  {label:<36}  probe {notch:6.2}  timbre {timbre:7.4}  worst {}",
            worst_text(worst)
        );
    }
    ExitCode::SUCCESS
}

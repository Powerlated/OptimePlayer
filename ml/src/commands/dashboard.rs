//! Serve the training dashboard with **dummy data** — for visually testing the dashboard itself
//! (the architecture diagram, tiles, charts, tables) without waiting on a real training run.
//!
//! It builds a faithful [`RunMeta`] for the chosen backbone — real params, FLOPs, and serialized
//! config, so the Architecture card and Model facts show true numbers — then streams synthetic
//! epochs/batches at a configurable pace so the live charts animate, and finally holds the finished
//! view open (Ctrl-C to exit). `--stage` picks which dashboard mode to exercise: `finetune`
//! (accuracy plot) or `pretrain` (held-out-loss plot).

use super::opts::Backbone;
use clap::{Args, ValueEnum};
use optime_ml::backend::{precision, Back, MlDevice};
use optime_ml::dashboard::{self, ContextWindow, DataStats, EpochPoint, RunMeta};
use optime_ml::m00_frame::FrameModel;
use optime_ml::m01_event::EventModel;
use optime_ml::m02_hier::HierModel;
use optime_ml::notes::N_TRANSPOSITIONS;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::path::PathBuf;
use std::time::Duration;

/// Which dashboard layout to populate — the two stages report different held-out metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum DemoStage {
    /// Held-out loss only (the SSL pretext view).
    Pretrain,
    /// Key + chord accuracy (the supervised view).
    #[default]
    Finetune,
}

#[derive(Args, Debug)]
pub struct DashboardArgs {
    /// Backbone whose architecture + real params/FLOPs the demo reports.
    #[arg(long, value_enum, default_value_t = Backbone::Hier)]
    pub backbone: Backbone,
    /// Which dashboard mode to exercise.
    #[arg(long, value_enum, default_value_t = DemoStage::Finetune)]
    pub stage: DemoStage,
    /// Epochs to simulate.
    #[arg(long, default_value_t = 12)]
    pub epochs: usize,
    /// Simulated wall-time per epoch, in seconds (the animation pace). Small values approximate an
    /// instant fill while still spreading points along the time axis.
    #[arg(long, default_value_t = 1.5)]
    pub epoch_secs: f64,
}

pub fn run(args: DashboardArgs) {
    match args.backbone {
        Backbone::Frame => go::<FrameModel<Back>>(&args),
        Backbone::Event => go::<EventModel<Back>>(&args),
        Backbone::Hier => go::<HierModel<Back>>(&args),
    }
}

fn go<M: optime_ml::backbone::Backbone<Back>>(args: &DashboardArgs) {
    let device = MlDevice::default();
    let cfg = M::default_cfg();
    let model = M::init(&cfg, &device);

    // Real params/FLOPs/config for the diagram + facts; the window length is read out of the
    // serialized config (m00 calls it `max_seq_len`, m01/m02 `n_frames`).
    let cfg_json = serde_json::to_value(&cfg).unwrap_or(serde_json::Value::Null);
    let seq = cfg_json
        .get("n_frames")
        .or_else(|| cfg_json.get("max_seq_len"))
        .and_then(|v| v.as_u64())
        .unwrap_or(256) as usize;
    let notes_per_window = 405.0;

    // Plausible dataset shape, kept internally consistent (beats = windows × seq/4).
    let train_windows = 16_000usize;
    let train_beats = train_windows as f64 * (seq as f64 / 4.0);
    let train_hours = train_beats / dashboard::REFERENCE_BPM / 60.0;
    let data = DataStats {
        train_windows,
        val_windows: 1_800,
        notes_per_window,
        train_beats,
        train_hours,
        transpositions: N_TRANSPOSITIONS,
        augmented_hours: train_hours * N_TRANSPOSITIONS as f64,
    };

    let pretrain = args.stage == DemoStage::Pretrain;
    // Stage string drives the diagram's causal/bidirectional + heads-idle switch — it keys on
    // "pretrain", so keep that word for the pretext view.
    let stage = if pretrain {
        "AR pretrain (dummy data)".to_string()
    } else {
        "supervised fine-tune (dummy data)".to_string()
    };
    // The Hyperparameters + FLOPs/batch tiles read `train_config`; a plain TrainConfig-shaped blob
    // covers both stages for a visual test.
    let train_config = serde_json::json!({
        "epochs": args.epochs,
        "batch_size": 32,
        "lr": 3.0e-4,
    });

    dashboard::start(RunMeta {
        stage,
        backbone: M::NAME.to_string(),
        backend: format!(
            "{} (dummy data)",
            dashboard::backend_label(std::any::type_name::<Back>())
        ),
        precision: precision::<Back>(),
        epochs: args.epochs,
        context: ContextWindow::from_frames(seq),
        data,
        params: model.num_params(),
        flops_per_window: M::flops_per_window(&cfg, notes_per_window as usize),
        model_config: cfg_json,
        train_config,
    });
    // `start` prints the URL; None means disabled (ML_DASHBOARD=0) or the port was taken.
    if std::env::var("ML_DASHBOARD").as_deref() == Ok("0") {
        eprintln!("ML_DASHBOARD=0 disables the server — nothing to view. Unset it and re-run.");
        return;
    }
    println!(
        "serving DEMO dashboard for `{}` with dummy data — open the URL above. Ctrl-C to exit.",
        M::NAME
    );

    simulate(args, pretrain);

    // Hold the finished view open regardless of ML_DASHBOARD_HOLD (this command exists to be looked
    // at), by opting into `finish`'s park loop.
    std::env::set_var("ML_DASHBOARD_HOLD", "1");
    dashboard::finish(&PathBuf::from("(dummy run — nothing written)"));
}

/// Stream synthetic per-batch and per-epoch points, sleeping so the live charts animate.
fn simulate(args: &DashboardArgs, pretrain: bool) {
    let mut rng = StdRng::seed_from_u64(0xDA5B0A2D);
    let epochs = args.epochs.max(1);
    let of = 30usize; // batches/epoch
    let per_batch = Duration::from_secs_f64((args.epoch_secs / of as f64).max(0.0));

    // A smooth exponential decay from a high starting loss to a low floor across the whole run.
    let loss_at = |frac: f64| 0.35 + 2.25 * (-3.0 * frac).exp();

    for epoch in 1..=epochs {
        for b in 1..=of {
            let frac = ((epoch - 1) as f64 + b as f64 / of as f64) / epochs as f64;
            let loss = loss_at(frac) * (1.0 + rng.gen_range(-0.04..0.04));
            let rate = 8.0 * (1.0 + rng.gen_range(-0.08..0.08));
            dashboard::record_batch(epoch, b, of, loss, rate);
            if !per_batch.is_zero() {
                std::thread::sleep(per_batch);
            }
        }

        let frac = epoch as f64 / epochs as f64;
        let train_loss = loss_at(frac);
        if pretrain {
            // Held-out pretext loss tracks train, a touch higher.
            let val_loss = train_loss * 1.08 + 0.02;
            dashboard::record_epoch(EpochPoint::pretext(
                epoch,
                train_loss,
                val_loss,
                args.epoch_secs,
            ));
        } else {
            // Accuracies climb to a ceiling; chord flicker (changes/seq) settles down.
            let key_acc = (0.30 + 0.68 * frac).min(0.99);
            let chord_acc = (0.10 + 0.85 * frac).min(0.97);
            let changes = 40.0 - 30.0 * frac;
            dashboard::record_epoch(EpochPoint::supervised(
                epoch,
                train_loss,
                key_acc,
                chord_acc,
                changes,
                args.epoch_secs,
            ));
        }
    }
}

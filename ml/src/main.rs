//! `optime-ml` — the whole ML pipeline behind one binary.
//!
//! ```sh
//! cargo run --release -- <subcommand> [args]      # add --features harvest / gpu / cuda
//! cargo run --release -- --help                   # list subcommands
//! ```
//!
//! Subcommands map 1:1 to the former `src/bin/*` binaries. The harvest-only ones
//! (`harvest`/`sft`/`eval-labeled`/`chord-export`) and the GPU one (`pretrain-gpu`) appear only
//! when their cargo feature is on. Shared model/path flags live in [`commands::opts::ModelOpts`].

mod commands;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "optime-ml",
    about = "OptimePlayer chord/key ML pipeline",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate the synthetic labeled dataset → data/{train,val}.bin.
    GenerateData(commands::generate_data::GenerateDataArgs),
    /// SSL-pretrain a backbone on harvested real songs → <out>/<NN>/pretrained.
    Pretrain(commands::pretrain::PretrainArgs),
    /// Supervised fine-tune on the synthetic dataset → <out>/<NN>/model.
    Train(commands::train::TrainArgs),
    /// Real-music gap report (recon / chord-agreement / is-music). NOT accuracy.
    EvalReal(commands::eval_real::EvalRealArgs),
    /// Run a trained model on one song, print prediction vs ground truth.
    Infer(commands::infer::InferArgs),
    /// Train the frozen is-music linear probe → <out>/probe (frame only).
    Probe(commands::probe::ProbeArgs),
    /// Size the token/polyphony caps from the real corpus.
    TokenStats,
    /// Serve the dashboard with dummy data, for visual testing (no training).
    Dashboard(commands::dashboard::DashboardArgs),

    /// Harvest real game songs into unlabeled note-event windows (needs `harvest`).
    #[cfg(feature = "harvest")]
    Harvest(commands::harvest::HarvestArgs),
    /// Stage-3 fine-tune on hand labels → <out>/<NN>/model_sft (needs `harvest`).
    #[cfg(feature = "harvest")]
    Sft(commands::sft::SftArgs),
    /// Score against hand labels — THE real-music metric (needs `harvest`).
    #[cfg(feature = "harvest")]
    EvalLabeled(commands::eval_labeled::EvalLabeledArgs),
    /// Bake chord predictions into the app's `.ocd` chord lane (needs `harvest`).
    #[cfg(feature = "harvest")]
    ChordExport(commands::chord_export::ChordExportArgs),

    /// AR-pretrain the hierarchical backbone on the WGPU GPU (needs `gpu`).
    #[cfg(feature = "gpu")]
    PretrainGpu(commands::pretrain_gpu::PretrainGpuArgs),
}

fn main() {
    match Cli::parse().command {
        Command::GenerateData(a) => commands::generate_data::run(a),
        Command::Pretrain(a) => commands::pretrain::run(a),
        Command::Train(a) => commands::train::run(a),
        Command::EvalReal(a) => commands::eval_real::run(a),
        Command::Infer(a) => commands::infer::run(a),
        Command::Probe(a) => commands::probe::run(a),
        Command::TokenStats => commands::token_stats::run(),
        Command::Dashboard(a) => commands::dashboard::run(a),

        #[cfg(feature = "harvest")]
        Command::Harvest(a) => commands::harvest::run(a),
        #[cfg(feature = "harvest")]
        Command::Sft(a) => commands::sft::run(a),
        #[cfg(feature = "harvest")]
        Command::EvalLabeled(a) => commands::eval_labeled::run(a),
        #[cfg(feature = "harvest")]
        Command::ChordExport(a) => commands::chord_export::run(a),

        #[cfg(feature = "gpu")]
        Command::PretrainGpu(a) => commands::pretrain_gpu::run(a),
    }
}

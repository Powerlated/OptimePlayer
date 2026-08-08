//! Command-line entry point: parses the subcommand and dispatches to its module.

use std::process::ExitCode;

use clap::{Parser, Subcommand};
use optime_cli::{
    album, bench, bench_kernel, dse, extract, golden, match_ost, mixer_response, render,
};

#[derive(Parser)]
#[command(
    name = "optime-cli",
    about = "Offline tools for the Optime Player engine.",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    RenderSongs(render::Args),
    ExportAlbum(album::Args),
    MatchOst(match_ost::Args),
    ExtractMp2k(extract::Args),
    DumpDse(dse::Args),
    Golden(golden::Args),
    BenchResample(bench::Args),
    BenchKernel(bench_kernel::Args),
    MixerResponse(mixer_response::Args),
}

fn main() -> ExitCode {
    match Cli::parse().command {
        Command::RenderSongs(a) => render::run(a),
        Command::ExportAlbum(a) => album::run(a),
        Command::MatchOst(a) => match_ost::run(a),
        Command::ExtractMp2k(a) => extract::run(a),
        Command::DumpDse(a) => dse::run(a),
        Command::Golden(a) => golden::run(a),
        Command::BenchResample(a) => bench::run(a),
        Command::BenchKernel(a) => bench_kernel::run(a),
        Command::MixerResponse(a) => mixer_response::run(a),
    }
}

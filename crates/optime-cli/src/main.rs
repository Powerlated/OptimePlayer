//! Offline command-line tools for the Optime Player engine.
//!
//! Every subcommand is an operation on the engine that has no place in the desktop app: bulk
//! renders, soundtrack exports, ROM surgery, format dumps and performance harnesses. They live in
//! one binary so `optime-core` itself keeps a single dependency (`serde`) and pays nothing for the
//! FLAC encoder, loudness meter and audio decoders these tools need.
//!
//! Run `cargo run -p optime-cli -- --help` for the list, or `-- <subcommand> --help` for one
//! tool's arguments. Anything measuring performance wants `--release`.

mod album;
mod bench;
mod dse;
mod extract;
mod golden;
mod match_ost;
mod mixer_response;
mod render;
mod wav;

use std::process::ExitCode;

use clap::{Parser, Subcommand};

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
        Command::MixerResponse(a) => mixer_response::run(a),
    }
}

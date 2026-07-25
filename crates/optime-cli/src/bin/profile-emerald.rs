//! Single-threaded soundtrack render for sampling profilers (AMD uProf, Intel VTune).
//!
//! Its own executable rather than an `optime-cli` subcommand: a profiler is pointed at a binary and
//! runs it, and both uProf and VTune are far easier to set up when the target needs no arguments.
//! `target/release/profile-emerald` with an empty argument list renders the whole Pokémon Emerald
//! soundtrack. The implementation lives in `optime_cli::profile`.

use std::process::ExitCode;

use clap::Parser;
use optime_cli::profile;

fn main() -> ExitCode {
    profile::run(profile::Args::parse())
}

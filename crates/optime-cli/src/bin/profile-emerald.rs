//! Profiler target: runs the single-threaded soundtrack render, taking no arguments so a profiler can launch it directly.

use std::process::ExitCode;

use clap::Parser;
use optime_cli::profile;

fn main() -> ExitCode {
    profile::run(profile::Args::parse())
}

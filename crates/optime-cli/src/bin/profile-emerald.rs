use std::process::ExitCode;

use clap::Parser;
use optime_cli::profile;

fn main() -> ExitCode {
    profile::run(profile::Args::parse())
}

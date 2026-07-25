//! Implementation of the offline Optime Player tools, exposed as a library so the crate's
//! executables can share it.
//!
//! Most tools are subcommands of the `optime-cli` binary (`main.rs`). The exception is
//! `profile-emerald`, a separate executable that a sampling profiler can launch with no arguments;
//! it reuses the album exporter's render path from here rather than copying it.

pub mod album;
pub mod bench;
pub mod dse;
pub mod extract;
pub mod golden;
pub mod match_ost;
pub mod mixer_response;
pub mod profile;
pub mod render;
pub mod wav;

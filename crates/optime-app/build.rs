//! Build script for the app: stamps the git revision and build time into the binary, and generates the local-only song-name tables and demo entries listed in `local_extras.txt`.

use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, fs};

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!s.is_empty()).then_some(s)
}

fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (y + i64::from(m <= 2), m, d)
}

fn build_time_utc() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    let rem = secs.rem_euclid(86_400);
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02} UTC",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

fn emit_build_info() {
    let hash = git(&["rev-parse", "--short=9", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let commit_date = git(&["log", "-1", "--format=%cd", "--date=format:%Y-%m-%d %H:%M"])
        .unwrap_or_else(|| "unknown".into());
    if let Some(git_dir) = git(&["rev-parse", "--git-dir"]) {
        println!("cargo:rerun-if-changed={git_dir}/HEAD");
        println!("cargo:rerun-if-changed={git_dir}/logs/HEAD");
    }
    println!("cargo:rustc-env=OPTIME_GIT_HASH={hash}");
    println!("cargo:rustc-env=OPTIME_COMMIT_DATE={commit_date}");
    println!("cargo:rustc-env=OPTIME_BUILD_TIME={}", build_time_utc());
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed=local_extras.txt");
    emit_build_info();

    let mut tables = String::new();
    let mut demos = String::new();
    if let Ok(text) = fs::read_to_string(manifest_dir.join("local_extras.txt")) {
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            match line.split('|').collect::<Vec<_>>().as_slice() {
                ["table", key, json] => {
                    let path = manifest_dir.join("src/song_names").join(json);
                    if path.exists() {
                        println!("cargo:rerun-if-changed=src/song_names/{json}");
                        let lit = path.to_string_lossy();
                        tables.push_str(&format!(
                            "    ({key:?}, {json:?}, include_str!({lit:?})),\n"
                        ));
                    } else {
                        println!(
                            "cargo:warning=local_extras: missing table json {}",
                            path.display()
                        );
                    }
                }
                ["demo", label, stem] => demos.push_str(&format!("    ({label:?}, {stem:?}),\n")),
                _ => println!("cargo:warning=local_extras: unrecognized line {line:?}"),
            }
        }
    }

    fs::write(
        out_dir.join("local_filename_tables.rs"),
        format!(
            "/// Song-name tables keyed by loaded source filename, generated from the optional,\n\
             /// gitignored `local_extras.txt` (see build.rs). Empty in a clean clone.\n\
             const JSONS_BY_GAME_FILENAME: &[(&str, &str, &str)] = &[\n{tables}];\n"
        ),
    )
    .unwrap();
    fs::write(
        out_dir.join("local_demos.rs"),
        format!(
            "/// Extra demo entries from the optional, gitignored `local_extras.txt` (see build.rs).\n\
             const LOCAL_DEMOS: &[(&str, &str)] = &[\n{demos}];\n"
        ),
    )
    .unwrap();
}

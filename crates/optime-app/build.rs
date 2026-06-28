//! Generates optional, **local-only** "extras" from a gitignored `local_extras.txt` manifest:
//! curated song-name tables keyed by source filename, and extra demo entries. This lets a personal
//! checkout carry ROM-hack metadata (titles + a demo) that is never committed. With no manifest the
//! generated tables are empty and the app builds byte-for-byte like a clean clone.
//!
//! Manifest lines (`#` comments and blanks ignored), `|`-separated:
//!   table|<source-filename-key>|<json-file-under-src/song_names>
//!   demo|<label>|<demo-stem-under-demos>

use std::path::PathBuf;
use std::{env, fs};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed=local_extras.txt");

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

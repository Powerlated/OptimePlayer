//! Golden-output parity test.
//!
//! `test-fixtures/golden.bin` was rendered from the ORIGINAL JS engine (see
//! `test-fixtures/gen_fixture.js`). This test renders the same SSEQ with the Rust port using the
//! identical clock/mix and asserts the output matches sample-for-sample, which proves the
//! emulation port is faithful end to end.

use std::path::PathBuf;

use optime_core::{Controller, Sdat, SynthConfig};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-fixtures")
}

/// Extracts a numeric field like `"frames": 12345` from the simple golden.json.
fn json_number(json: &str, key: &str) -> i64 {
    let needle = format!("\"{key}\"");
    let start = json.find(&needle).expect("key present") + needle.len();
    let rest = &json[start..];
    let colon = rest.find(':').unwrap() + 1;
    rest[colon..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-')
        .collect::<String>()
        .parse()
        .unwrap()
}

/// Extracts a string field like `"demo": "foo.sdat"`.
fn json_string(json: &str, key: &str) -> String {
    let needle = format!("\"{key}\"");
    let start = json.find(&needle).unwrap() + needle.len();
    let rest = &json[start..];
    let first_quote = rest[rest.find(':').unwrap()..].find('"').unwrap() + rest.find(':').unwrap();
    let after = &rest[first_quote + 1..];
    after[..after.find('"').unwrap()].to_string()
}

#[test]
fn matches_legacy_engine_sample_for_sample() {
    let meta = std::fs::read_to_string(fixtures_dir().join("golden.json"))
        .expect("run test-fixtures/gen_fixture.js to generate the golden fixture");
    let demo = json_string(&meta, "demo");
    let sseq_id = json_number(&meta, "sseqId") as u32;
    let sample_rate = json_number(&meta, "sampleRate") as f64;
    let frames = json_number(&meta, "frames") as usize;

    // Load the golden interleaved-f32 stereo samples.
    let golden_bytes = std::fs::read(fixtures_dir().join("golden.bin")).unwrap();
    assert_eq!(golden_bytes.len(), frames * 2 * 4, "fixture size mismatch");
    let golden: Vec<f32> = golden_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // Render the same sequence with the Rust engine, default config (equal tuning, all tracks,
    // no stereo separation) — exactly how the fixture was produced.
    let rom = std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../demos")
            .join(&demo),
    )
    .unwrap();
    let sdats = Sdat::load_all(&rom);
    let sdat = &sdats[0];
    let mut controller = Controller::new(sample_rate, sdat, sseq_id).expect("controller");
    let config = SynthConfig::default();

    let mut max_abs_diff = 0.0f32;
    let mut sum_sq_diff = 0.0f64;
    let mut peak = 0.0f32;
    for frame in 0..frames {
        let (l, r) = controller.next_sample(&config);
        for (got, &want) in [l, r].iter().zip(&golden[frame * 2..frame * 2 + 2]) {
            let diff = (got - want).abs();
            max_abs_diff = max_abs_diff.max(diff);
            sum_sq_diff += f64::from(diff) * f64::from(diff);
            peak = peak.max(want.abs());
        }
    }

    let rms = (sum_sq_diff / (frames * 2) as f64).sqrt();
    eprintln!("golden parity: peak={peak:.4} max_abs_diff={max_abs_diff:.3e} rms={rms:.3e}");

    // The port is bit-exact against the legacy engine on this platform (max_abs_diff == 0). The
    // tiny tolerances below only guard against last-ULP differences in transcendental functions
    // (pow/sqrt) that could appear on other libm implementations.
    assert!(peak > 0.05, "golden fixture appears silent (peak {peak})");
    assert!(
        max_abs_diff < 1e-5,
        "max abs diff too large: {max_abs_diff}"
    );
    assert!(rms < 1e-6, "rms error too large: {rms}");
}

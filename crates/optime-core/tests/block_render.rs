//! The block renderer (`SynthController::fill`) must be bit-identical to the per-sample path
//! (`SynthController::next_sample` in a loop): same clock evolution, same voice math, same mixing
//! order. Rendered against a real demo SDAT so the full engine (sequencer, ADSR/LFO ticks,
//! looping samples, stereo stage) is exercised.

use std::path::PathBuf;

use optime_core::{InstrumentResampleMode, PopSmoothing, SoundData, SynthConfig, SynthController};

fn load_demo() -> SoundData {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../demos/super-mario-64-ds.sdat");
    let bytes = std::fs::read(path).expect("demo file should exist");
    SoundData::load_all(&bytes).remove(0)
}

fn assert_fill_matches_next_sample(config: &SynthConfig, label: &str) {
    let data = load_demo();
    let sr = 32768.0;
    let mut per_sample = SynthController::new(sr, &data, 0).expect("SSEQ 0");
    let mut blocked = SynthController::new(sr, &data, 0).expect("SSEQ 0");

    // Uneven chunk sizes so blocks split at chunk boundaries as well as tick boundaries,
    // including a chunk with an odd f32 count (a trailing half frame).
    let chunk_sizes = [2048usize, 510, 64, 2, 333, 4096];
    let mut buf = vec![0.0f32; *chunk_sizes.iter().max().unwrap()];

    let mut rendered = 0usize;
    for &size in chunk_sizes.iter().cycle().take(40) {
        let chunk = &mut buf[..size];
        blocked.fill(chunk, config);

        let mut expect = vec![0.0f32; size];
        for frame in expect.chunks_mut(2) {
            let (l, r) = per_sample.next_sample(config);
            frame[0] = l;
            if frame.len() > 1 {
                frame[1] = r;
            }
        }

        for (i, (&got, &want)) in chunk.iter().zip(&expect).enumerate() {
            assert!(
                got == want,
                "{label}: sample {} differs: fill={got}, next_sample={want}",
                rendered + i
            );
        }
        rendered += size;
    }
    assert!(rendered as f64 > sr, "should cover more than a second");
}

#[test]
fn fill_matches_next_sample_sinc_stereo() {
    let config = SynthConfig {
        resample: InstrumentResampleMode::SincOutputNyquist {
            half_taps: 4,
            psg_cutoff_hz: 8_000,
            sampler_cutoff_hz: 12_000,
        },
        stereo_separation: true,
        bass_mono: true,
        ..SynthConfig::default()
    };
    assert_fill_matches_next_sample(&config, "sinc crunch + stereo + bass mono");
}

#[test]
fn fill_matches_next_sample_clean_plain() {
    let config = SynthConfig {
        resample: InstrumentResampleMode::SincSampleNyquist { half_taps: 4 },
        stereo_separation: false,
        ..SynthConfig::default()
    };
    assert_fill_matches_next_sample(&config, "sinc clean, no separation");
}

#[test]
fn fill_matches_next_sample_smoothed_pops() {
    // Both pop-smoothing kinds (PSG + sampled) slew gains sample-by-sample, so the block path
    // must reproduce the per-sample path exactly.
    let config = SynthConfig {
        resample: InstrumentResampleMode::SincOutputNyquist {
            half_taps: 4,
            psg_cutoff_hz: 8_000,
            sampler_cutoff_hz: 12_000,
        },
        pop_smoothing: PopSmoothing {
            psg: true,
            sample: true,
        },
        ..SynthConfig::default()
    };
    assert_fill_matches_next_sample(&config, "crunch + smoothed PSG & sample pops");
}

#[test]
fn fill_matches_next_sample_nearest() {
    let config = SynthConfig {
        resample: InstrumentResampleMode::NearestNeighbor,
        stereo_separation: true,
        ..SynthConfig::default()
    };
    assert_fill_matches_next_sample(&config, "nearest + stereo");
}

#[test]
fn fill_matches_next_sample_intermediate_mixer_sinc() {
    // The mixer bus is pulled per output sample, so the block path must still match the
    // per-sample path with sampled voices routed through the upsampling mixer.
    let config = SynthConfig {
        resample: InstrumentResampleMode::SincOutputNyquist {
            half_taps: 4,
            psg_cutoff_hz: 8_000,
            sampler_cutoff_hz: 12_000,
        },
        stereo_separation: true,
        bass_mono: true,
        use_mixer: true,
        mixer_sample_rate: 18_000.0,
        mixer_resample: InstrumentResampleMode::SincOutputNyquist {
            half_taps: 8,
            psg_cutoff_hz: InstrumentResampleMode::CUTOFF_OFF_HZ,
            sampler_cutoff_hz: 12_000,
        },
        ..SynthConfig::default()
    };
    assert_fill_matches_next_sample(&config, "intermediate mixer (sinc crunch)");
}

#[test]
fn fill_matches_next_sample_intermediate_mixer_nearest() {
    let config = SynthConfig {
        resample: InstrumentResampleMode::NearestNeighbor,
        stereo_separation: true,
        use_mixer: true,
        mixer_sample_rate: 13_379.0,
        mixer_resample: InstrumentResampleMode::NearestNeighbor,
        ..SynthConfig::default()
    };
    assert_fill_matches_next_sample(&config, "intermediate mixer (nearest/ZOH)");
}

#[test]
fn fill_matches_next_sample_intermediate_mixer_linear() {
    let config = SynthConfig {
        resample: InstrumentResampleMode::SincSampleNyquist { half_taps: 4 },
        stereo_separation: true,
        use_mixer: true,
        mixer_sample_rate: 24_000.0,
        mixer_resample: InstrumentResampleMode::Linear,
        ..SynthConfig::default()
    };
    assert_fill_matches_next_sample(&config, "intermediate mixer (linear)");
}

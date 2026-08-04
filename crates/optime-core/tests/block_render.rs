use std::path::PathBuf;

use optime_core::{
    InstrumentResampleChoice, InstrumentResampleSettings, MixerResampleSettings, PerDeviceSettings,
    PopSmoothingEdge, SoundData, SynthController, load_all,
};

fn load_demo() -> Box<dyn SoundData> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../demos/super-mario-64-ds.sdat");
    let bytes = std::fs::read(path).expect("demo file should exist");
    load_all(&bytes).remove(0)
}

fn instrument(
    choice: InstrumentResampleChoice,
    sinc_taps: usize,
    psg_cutoff_hz: u32,
    sampler_cutoff_hz: u32,
    smooth_psg_pops: bool,
    smooth_sample_pops: bool,
) -> InstrumentResampleSettings {
    InstrumentResampleSettings {
        choice,
        sinc_taps,
        psg_cutoff_hz,
        sampler_cutoff_hz,
        smooth_psg_pops,
        smooth_sample_pops,
        pop_slew_ms: 2.0,
        pop_smooth_edge: PopSmoothingEdge::Both,
    }
}

fn assert_fill_matches_next_sample(config: &PerDeviceSettings, label: &str) {
    let data = load_demo();
    let sr = 32768.0;
    let mut per_sample = SynthController::new(sr, &*data, 0).expect("SSEQ 0");
    let mut blocked = SynthController::new(sr, &*data, 0).expect("SSEQ 0");

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
    let config = PerDeviceSettings {
        instrument_resample: instrument(
            InstrumentResampleChoice::SincOutputNyquist,
            8,
            8_000,
            12_000,
            false,
            false,
        ),
        stereo_separation: true,
        bass_mono: true,
        ..PerDeviceSettings::neutral()
    };
    assert_fill_matches_next_sample(&config, "sinc crunch + stereo + bass mono");
}

#[test]
fn fill_matches_next_sample_clean_plain() {
    let config = PerDeviceSettings {
        instrument_resample: instrument(
            InstrumentResampleChoice::SincSampleNyquist,
            8,
            0,
            0,
            false,
            false,
        ),
        stereo_separation: false,
        ..PerDeviceSettings::neutral()
    };
    assert_fill_matches_next_sample(&config, "sinc clean, no separation");
}

#[test]
fn fill_matches_next_sample_smoothed_pops() {
    let config = PerDeviceSettings {
        instrument_resample: instrument(
            InstrumentResampleChoice::SincOutputNyquist,
            8,
            8_000,
            12_000,
            true,
            true,
        ),
        ..PerDeviceSettings::neutral()
    };
    assert_fill_matches_next_sample(&config, "crunch + smoothed PSG & sample pops");
}

#[test]
fn fill_matches_next_sample_nearest() {
    let config = PerDeviceSettings {
        instrument_resample: instrument(InstrumentResampleChoice::Nearest, 32, 0, 0, false, false),
        stereo_separation: true,
        ..PerDeviceSettings::neutral()
    };
    assert_fill_matches_next_sample(&config, "nearest + stereo");
}

#[test]
fn fill_matches_next_sample_intermediate_mixer_sinc() {
    let config = PerDeviceSettings {
        instrument_resample: instrument(
            InstrumentResampleChoice::SincOutputNyquist,
            8,
            8_000,
            12_000,
            false,
            false,
        ),
        stereo_separation: true,
        bass_mono: true,
        use_mixer: true,
        mixer_sample_rate: 18_000,
        mixer_resample: MixerResampleSettings {
            choice: InstrumentResampleChoice::SincOutputNyquist,
            sinc_taps: 16,
            cutoff_hz: 12_000,
        },
        ..PerDeviceSettings::neutral()
    };
    assert_fill_matches_next_sample(&config, "intermediate mixer (sinc crunch)");
}

#[test]
fn fill_matches_next_sample_intermediate_mixer_nearest() {
    let config = PerDeviceSettings {
        instrument_resample: instrument(InstrumentResampleChoice::Nearest, 32, 0, 0, false, false),
        stereo_separation: true,
        use_mixer: true,
        mixer_sample_rate: 13_379,
        mixer_resample: MixerResampleSettings {
            choice: InstrumentResampleChoice::Nearest,
            sinc_taps: 32,
            cutoff_hz: 0,
        },
        ..PerDeviceSettings::neutral()
    };
    assert_fill_matches_next_sample(&config, "intermediate mixer (nearest/ZOH)");
}

#[test]
fn fill_matches_next_sample_intermediate_mixer_linear() {
    let config = PerDeviceSettings {
        instrument_resample: instrument(
            InstrumentResampleChoice::SincSampleNyquist,
            8,
            0,
            0,
            false,
            false,
        ),
        stereo_separation: true,
        use_mixer: true,
        mixer_sample_rate: 24_000,
        mixer_resample: MixerResampleSettings {
            choice: InstrumentResampleChoice::Linear,
            sinc_taps: 32,
            cutoff_hz: 0,
        },
        ..PerDeviceSettings::neutral()
    };
    assert_fill_matches_next_sample(&config, "intermediate mixer (linear)");
}

#[test]
fn render_is_chunk_invariant_down_to_one_sample() {
    let config = PerDeviceSettings {
        instrument_resample: instrument(
            InstrumentResampleChoice::SincOutputNyquist,
            16,
            10_000,
            12_000,
            true,
            true,
        ),
        stereo_separation: true,
        bass_mono: true,
        smooth_pan: true,
        delay_smoothing_choice: 1,
        use_mixer: true,
        mixer_sample_rate: 13_379,
        mixer_resample: MixerResampleSettings {
            choice: InstrumentResampleChoice::SincOutputNyquist,
            sinc_taps: 16,
            cutoff_hz: 12_000,
        },
        ..PerDeviceSettings::neutral()
    };
    let data = load_demo();
    let sr = 32768.0;
    const FRAMES: usize = 20_000;

    let mut whole = SynthController::new(sr, &*data, 0).expect("SSEQ 0");
    let (mut want_l, mut want_r) = (vec![0.0f32; FRAMES], vec![0.0f32; FRAMES]);
    whole.render(&mut want_l, &mut want_r, &config);

    for size in [1usize, 3, 97, 171, 256, 257, 1000] {
        let mut chunked = SynthController::new(sr, &*data, 0).expect("SSEQ 0");
        let (mut got_l, mut got_r) = (vec![0.0f32; FRAMES], vec![0.0f32; FRAMES]);
        for (l, r) in got_l.chunks_mut(size).zip(got_r.chunks_mut(size)) {
            chunked.render(l, r, &config);
        }
        for (i, (((&gl, &gr), &wl), &wr)) in got_l
            .iter()
            .zip(&got_r)
            .zip(&want_l)
            .zip(&want_r)
            .enumerate()
        {
            assert!(
                gl == wl && gr == wr,
                "chunk size {size}: frame {i} differs: ({gl}, {gr}) vs ({wl}, {wr})"
            );
        }
    }
}

#[test]
fn sampled_high_band_compressor_survives_cutoff_above_mixer_nyquist() {
    use optime_core::synth_controller::HighBandCompressor;
    let data = load_demo();
    let sr = 32768.0;
    let config = PerDeviceSettings {
        use_mixer: true,
        mixer_sample_rate: 13_379,
        high_band_compress: HighBandCompressor {
            enabled_psg: false,
            enabled_sampled: true,
            cutoff_hz: 10_000.0,
            threshold_db: -18.0,
            ratio: 4.0,
            attack_ms: 2.0,
            release_ms: 85.53,
            makeup_db: 0.0,
        },
        ..PerDeviceSettings::neutral()
    };
    let mut controller = SynthController::new(sr, &*data, 0).expect("SSEQ 0");

    let mut peak = 0.0f32;
    for _ in 0..(sr as usize * 2) {
        let (l, r) = controller.next_sample(&config);
        assert!(l.is_finite() && r.is_finite(), "non-finite output");
        peak = peak.max(l.abs()).max(r.abs());
    }
    assert!(peak > 0.01, "output too quiet: peak {peak}");
    assert!(peak <= 2.0, "output exploded: peak {peak}");
}

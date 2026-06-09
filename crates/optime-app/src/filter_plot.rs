//! Analysis popup windows: Bode (magnitude + phase) plots for the bass-mono crossover (IIR)
//! and the sinc-resampler kernel (FIR), drawn with `egui_plot`.

use std::f64::consts::PI;

use egui::Color32;
use egui_plot::{AxisHints, Line, Plot, PlotPoints, Points};
use optime_core::{fir_kernel, fir_response, BiquadFilter, ResampleMode, CROSSOVER_Q};

/// Shows the Bode + pole-zero analysis popup for the bass-mono crossover filter.
///
/// `open` is toggled by the ❌ button; pass `&mut self.crossover_plot_open`.
pub fn show_crossover_window(
    ctx: &egui::Context,
    open: &mut bool,
    sample_rate: f64,
    bass_mono_freq: f64,
) {
    egui::Window::new("Crossover filter analysis")
        .open(open)
        .resizable(true)
        .default_width(640.0)
        .default_height(560.0)
        .show(ctx, |ui| {
            let nyquist = sample_rate / 2.0;
            let lp = BiquadFilter::low_pass(4, sample_rate, bass_mono_freq, CROSSOVER_Q);
            let hp = BiquadFilter::high_pass(4, sample_rate, bass_mono_freq, CROSSOVER_Q);

            // X axis: log₁₀(f).  We use this as the plot coordinate so the x-spacing is
            // logarithmic, and a custom formatter displays the true Hz value.
            let log_freqs: Vec<f64> = (0..512)
                .map(|i| {
                    let t = i as f64 / 511.0;
                    let log_lo = (20.0f64).log10();
                    let log_hi = nyquist.log10();
                    log_lo + (log_hi - log_lo) * t
                })
                .collect();
            let hz_formatter =
                |mark: egui_plot::GridMark, _range: &std::ops::RangeInclusive<f64>| {
                    let hz = (10.0f64).powf(mark.value);
                    if hz >= 1000.0 {
                        format!("{:.0}k", hz / 1000.0)
                    } else {
                        format!("{:.0}", hz)
                    }
                };

            // Magnitude plot (dB).
            ui.label("Magnitude");
            let mag_plot = Plot::new("xover_mag")
                .height(160.0)
                .y_axis_label("dB")
                .custom_x_axes(vec![AxisHints::new_x()
                    .label("Frequency (Hz)")
                    .formatter(hz_formatter)])
                .include_y(-60.0)
                .include_y(3.0)
                .allow_drag(false)
                .allow_zoom(false)
                .allow_scroll(false);
            mag_plot.show(ui, |plot_ui| {
                let lp_mag: PlotPoints = log_freqs
                    .iter()
                    .map(|&lf| {
                        let f = (10.0f64).powf(lf);
                        let w = 2.0 * PI * f / sample_rate;
                        let (mag, _) = lp.frequency_response(w);
                        let db = 20.0 * mag.max(1e-10).log10();
                        [lf, db]
                    })
                    .collect();
                let hp_mag: PlotPoints = log_freqs
                    .iter()
                    .map(|&lf| {
                        let f = (10.0f64).powf(lf);
                        let w = 2.0 * PI * f / sample_rate;
                        let (mag, _) = hp.frequency_response(w);
                        let db = 20.0 * mag.max(1e-10).log10();
                        [lf, db]
                    })
                    .collect();
                plot_ui.line(
                    Line::new(lp_mag)
                        .color(Color32::from_rgb(100, 180, 255))
                        .name("Low-pass")
                        .fill(-60.0),
                );
                plot_ui.line(
                    Line::new(hp_mag)
                        .color(Color32::from_rgb(255, 160, 80))
                        .name("High-pass")
                        .fill(-60.0),
                );
            });

            ui.add_space(4.0);

            // Phase plot (degrees).
            ui.label("Phase");
            let hz_formatter2 =
                |mark: egui_plot::GridMark, _range: &std::ops::RangeInclusive<f64>| {
                    let hz = (10.0f64).powf(mark.value);
                    if hz >= 1000.0 {
                        format!("{:.0}k", hz / 1000.0)
                    } else {
                        format!("{:.0}", hz)
                    }
                };
            let phase_plot = Plot::new("xover_phase")
                .height(130.0)
                .y_axis_label("°")
                .custom_x_axes(vec![AxisHints::new_x()
                    .label("Frequency (Hz)")
                    .formatter(hz_formatter2)])
                .include_y(-360.0)
                .include_y(360.0)
                .allow_drag(false)
                .allow_zoom(false)
                .allow_scroll(false);
            phase_plot.show(ui, |plot_ui| {
                let lp_phase: PlotPoints = log_freqs
                    .iter()
                    .map(|&lf| {
                        let f = (10.0f64).powf(lf);
                        let w = 2.0 * PI * f / sample_rate;
                        let (_, ph) = lp.frequency_response(w);
                        [lf, ph.to_degrees()]
                    })
                    .collect();
                let hp_phase: PlotPoints = log_freqs
                    .iter()
                    .map(|&lf| {
                        let f = (10.0f64).powf(lf);
                        let w = 2.0 * PI * f / sample_rate;
                        let (_, ph) = hp.frequency_response(w);
                        [lf, ph.to_degrees()]
                    })
                    .collect();
                plot_ui.line(
                    Line::new(lp_phase)
                        .color(Color32::from_rgb(100, 180, 255))
                        .fill(0.0),
                );
                plot_ui.line(
                    Line::new(hp_phase)
                        .color(Color32::from_rgb(255, 160, 80))
                        .fill(0.0),
                );
            });

            ui.add_space(4.0);

            // Pole-zero plot (one biquad section; multiplicity = num_cascade shown by label).
            ui.label(format!(
                "Pole-zero (order {}; each marker has multiplicity {})",
                lp.num_cascade() * 2,
                lp.num_cascade()
            ));
            let pz_plot = Plot::new("xover_pz")
                .height(200.0)
                .x_axis_label("Re")
                .y_axis_label("Im")
                .data_aspect(1.0)
                .include_x(-1.5)
                .include_x(1.5)
                .include_y(-1.5)
                .include_y(1.5)
                .allow_drag(false)
                .allow_zoom(false)
                .allow_scroll(false);
            pz_plot.show(ui, |plot_ui| {
                // Unit circle.
                let circle: PlotPoints = (0..=256)
                    .map(|i| {
                        let a = 2.0 * PI * i as f64 / 256.0;
                        [a.cos(), a.sin()]
                    })
                    .collect();
                plot_ui.line(Line::new(circle).color(Color32::from_gray(100)));

                // LP poles and zeros (use LP; HP would mirror them).
                let poles = lp.poles();
                let zeros = lp.zeros();

                let pole_pts: PlotPoints = poles.iter().map(|&(re, im)| [re, im]).collect();
                let zero_pts: PlotPoints = zeros.iter().map(|&(re, im)| [re, im]).collect();

                plot_ui.points(
                    Points::new(pole_pts)
                        .color(Color32::from_rgb(255, 100, 100))
                        .shape(egui_plot::MarkerShape::Cross)
                        .radius(8.0)
                        .name("Poles (LP)"),
                );
                plot_ui.points(
                    Points::new(zero_pts)
                        .color(Color32::from_rgb(100, 200, 100))
                        .shape(egui_plot::MarkerShape::Circle)
                        .radius(6.0)
                        .name("Zeros (LP)"),
                );

                let hp_poles = hp.poles();
                let hp_zeros = hp.zeros();
                let hp_pole_pts: PlotPoints = hp_poles.iter().map(|&(re, im)| [re, im]).collect();
                let hp_zero_pts: PlotPoints = hp_zeros.iter().map(|&(re, im)| [re, im]).collect();

                plot_ui.points(
                    Points::new(hp_pole_pts)
                        .color(Color32::from_rgb(255, 180, 60))
                        .shape(egui_plot::MarkerShape::Cross)
                        .radius(8.0)
                        .name("Poles (HP)"),
                );
                plot_ui.points(
                    Points::new(hp_zero_pts)
                        .color(Color32::from_rgb(100, 200, 200))
                        .shape(egui_plot::MarkerShape::Circle)
                        .radius(6.0)
                        .name("Zeros (HP)"),
                );
            });
        });
}

/// Shows the Bode + impulse-response analysis popup for the sinc-resampler kernel.
///
/// `open` is toggled by the ❌ button; pass `&mut self.sinc_plot_open`.
pub fn show_sinc_window(ctx: &egui::Context, open: &mut bool, mode: ResampleMode) {
    let (half_taps, label) = match mode {
        ResampleMode::SincSampleNyquist { half_taps } => {
            (half_taps, "Sinc – sample Nyquist (clean)")
        }
        ResampleMode::SincOutputNyquist { half_taps } => {
            (half_taps, "Sinc – output Nyquist (crunch)")
        }
        _ => return, // no popup for non-sinc modes
    };

    egui::Window::new(format!("{label} analysis"))
        .open(open)
        .resizable(true)
        .default_width(600.0)
        .default_height(500.0)
        .show(ctx, |ui| {
            // Use a representative fc = 0.45 for the plot (close to Nyquist, shows
            // the cutoff behaviour clearly regardless of live playback ratio).
            let fc = 0.45;

            // Frequency axis: normalised [0, 0.5] cycles/source-sample.
            let freqs_norm: Vec<f64> = (0..512).map(|i| i as f64 * 0.5 / 511.0).collect();

            // Magnitude plot (dB).
            ui.label(format!("Magnitude  ({} taps, fc = {fc})", half_taps * 2));
            let mag_plot = Plot::new("sinc_mag")
                .height(150.0)
                .x_axis_label("Frequency (cycles/source-sample)")
                .y_axis_label("dB")
                .include_y(-80.0)
                .include_y(3.0)
                .allow_drag(false)
                .allow_zoom(false)
                .allow_scroll(false);
            mag_plot.show(ui, |plot_ui| {
                let mag_pts: PlotPoints = freqs_norm
                    .iter()
                    .map(|&f| {
                        let w = f * 2.0 * PI;
                        let (mag, _) = fir_response(half_taps, fc, w);
                        let db = 20.0 * mag.max(1e-10).log10();
                        [f, db]
                    })
                    .collect();
                plot_ui.line(
                    Line::new(mag_pts)
                        .color(Color32::from_rgb(120, 210, 160))
                        .fill(-80.0),
                );
            });

            ui.add_space(4.0);

            // Phase plot (degrees).
            ui.label("Phase");
            let phase_plot = Plot::new("sinc_phase")
                .height(120.0)
                .x_axis_label("Frequency (cycles/source-sample)")
                .y_axis_label("°")
                .allow_drag(false)
                .allow_zoom(false)
                .allow_scroll(false);
            phase_plot.show(ui, |plot_ui| {
                let phase_pts: PlotPoints = freqs_norm
                    .iter()
                    .map(|&f| {
                        let w = f * 2.0 * PI;
                        let (_, ph) = fir_response(half_taps, fc, w);
                        [f, ph.to_degrees()]
                    })
                    .collect();
                plot_ui.line(
                    Line::new(phase_pts)
                        .color(Color32::from_rgb(120, 210, 160))
                        .fill(0.0),
                );
            });

            ui.add_space(4.0);

            // Impulse-response stem plot.
            ui.label("Impulse response (kernel taps)");
            let kernel = fir_kernel(half_taps, fc);
            let n_half = half_taps as i64 - 1;
            let stem_plot = Plot::new("sinc_ir")
                .height(150.0)
                .x_axis_label("Sample offset")
                .y_axis_label("Amplitude")
                .allow_drag(false)
                .allow_zoom(false)
                .allow_scroll(false);
            stem_plot.show(ui, |plot_ui| {
                for (idx, &val) in kernel.iter().enumerate() {
                    let x = (idx as i64 - n_half) as f64;
                    // Vertical stem from y=0 to y=val.
                    let stem: PlotPoints = vec![[x, 0.0], [x, val]].into();
                    plot_ui.line(
                        Line::new(stem)
                            .color(Color32::from_rgb(200, 200, 80))
                            .width(1.0),
                    );
                }
                // Dot at the tip of each tap.
                let tips: PlotPoints = kernel
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| [(i as i64 - n_half) as f64, v])
                    .collect();
                plot_ui.points(
                    Points::new(tips)
                        .color(Color32::from_rgb(240, 220, 60))
                        .radius(3.5),
                );
            });
        });
}

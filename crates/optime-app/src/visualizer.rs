//! Procedural 16-track × 88-key piano-roll visualizer drawn with [`egui::Painter`].
//!
//! Replaces the original PNG-based renderer: white/black key rectangles are lit per the
//! controller's `notes_on` state, with per-track enable toggles and an active-keyboard-track
//! highlight. No image assets required.

use egui::{Color32, Rect, Sense, Stroke, Vec2};

use crate::TRACK_COUNT;

/// A frame's worth of note state copied out of the controller under lock.
pub struct VisSnapshot {
    /// Whether a controller is loaded (dims the view when not).
    pub active: bool,
    /// `notes_on[track][midi]` — sequence-driven notes.
    pub notes_on: [[bool; 128]; TRACK_COUNT],
    /// `notes_kbd[track][midi]` — live keyboard notes.
    pub notes_kbd: [[bool; 128]; TRACK_COUNT],
    /// The track receiving live keyboard input, if any.
    pub active_track: Option<usize>,
    /// Sequencer steps elapsed (drives the piano-roll playhead).
    pub steps: u32,
    /// Current sequencer step rate in steps/second (tempo-dependent), for the piano roll's
    /// smoothed scroll clock.
    pub step_rate: f64,
}

impl Default for VisSnapshot {
    fn default() -> Self {
        Self {
            active: false,
            notes_on: [[false; 128]; TRACK_COUNT],
            notes_kbd: [[false; 128]; TRACK_COUNT],
            active_track: None,
            steps: 0,
            step_rate: 0.0,
        }
    }
}

// Layout constants (in points).
const SECTION_H: f32 = 24.0;
const WHITE_W: f32 = 9.0;
const WHITE_H: f32 = 18.0;
const BLACK_W: f32 = 6.0;
const BLACK_H: f32 = 11.0;
const TOGGLE_W: f32 = 18.0;
const PAD: f32 = 4.0;

// Indexed by `midi % 12` with A as base (j=0 -> MIDI 21 = A0).
const IS_BLACK: [bool; 12] = [
    false, true, false, false, true, false, true, false, false, true, false, true,
];
const KEY_NUM: [u32; 12] = [0, 0, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6];

/// Draws the visualizer and handles clicks: toggling track enables and selecting the active
/// keyboard track.
// The track index drives per-row geometry (`row_y`), so a range loop reads clearer than `enumerate`.
#[allow(clippy::needless_range_loop)]
pub fn draw(
    ui: &mut egui::Ui,
    snap: &VisSnapshot,
    track_enables: &mut [bool; TRACK_COUNT],
    active_track: &mut Option<usize>,
) {
    // 7 white keys per octave * ~7.5 octaves; computed conservatively wide.
    let keys_w = 53.0 * WHITE_W;
    let total_w = TOGGLE_W + PAD + keys_w + PAD * 2.0;
    let total_h = SECTION_H * TRACK_COUNT as f32 + PAD * 2.0;

    let (rect, response) = ui.allocate_exact_size(Vec2::new(total_w, total_h), Sense::click());
    let painter = ui.painter().with_clip_rect(rect);
    let origin = rect.min;

    let dim = if snap.active { 1.0 } else { 0.35 };
    let alpha = |c: Color32| {
        Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), (c.a() as f32 * dim) as u8)
    };

    let keys_x = origin.x + TOGGLE_W + PAD;

    for track in 0..TRACK_COUNT {
        let row_y = origin.y + PAD + track as f32 * SECTION_H;

        // Track-enable toggle.
        let toggle_rect = Rect::from_min_size(
            egui::pos2(origin.x, row_y),
            Vec2::new(TOGGLE_W, SECTION_H - 2.0),
        );
        let toggle_color = if track_enables[track] {
            Color32::from_rgb(0, 0xCC, 0)
        } else {
            Color32::from_rgb(0xCC, 0, 0)
        };
        painter.rect_filled(toggle_rect, 2.0, alpha(toggle_color));
        painter.rect_stroke(
            toggle_rect,
            2.0,
            Stroke::new(1.0_f32, alpha(Color32::BLACK)),
        );

        // White keys (background + lit).
        for j in 0..88usize {
            let midi = j + 21;
            let octave = (j / 12) as u32;
            let kio = j % 12;
            if IS_BLACK[kio] {
                continue;
            }
            let white_num = octave * 7 + KEY_NUM[kio];
            let x = keys_x + white_num as f32 * WHITE_W;
            let key_rect = Rect::from_min_size(
                egui::pos2(x, row_y + 2.0),
                Vec2::new(WHITE_W - 1.0, WHITE_H),
            );
            painter.rect_filled(key_rect, 0.0, alpha(Color32::WHITE));
            painter.rect_stroke(key_rect, 0.0, Stroke::new(0.5_f32, alpha(Color32::GRAY)));
            if snap.notes_on[track][midi] {
                let c = if snap.notes_kbd[track][midi] {
                    Color32::RED
                } else {
                    Color32::BLACK
                };
                painter.rect_filled(key_rect, 0.0, alpha(c));
            }
        }

        // Black keys on top.
        for j in 0..88usize {
            let midi = j + 21;
            let octave = (j / 12) as u32;
            let kio = j % 12;
            if !IS_BLACK[kio] {
                continue;
            }
            let white_num = octave * 7 + KEY_NUM[kio];
            let x = keys_x + white_num as f32 * WHITE_W + (WHITE_W - BLACK_W / 2.0);
            let key_rect =
                Rect::from_min_size(egui::pos2(x, row_y + 2.0), Vec2::new(BLACK_W, BLACK_H));
            let base = if snap.notes_on[track][midi] {
                if snap.notes_kbd[track][midi] {
                    Color32::RED
                } else {
                    Color32::BLACK
                }
            } else {
                Color32::from_rgb(0x33, 0x33, 0x33)
            };
            painter.rect_filled(key_rect, 0.0, alpha(base));
        }

        // Active-keyboard-track outline.
        if *active_track == Some(track) {
            let outline = Rect::from_min_size(
                egui::pos2(keys_x - 1.0, row_y),
                Vec2::new(keys_w, SECTION_H - 2.0),
            );
            painter.rect_stroke(
                outline,
                0.0,
                Stroke::new(1.5_f32, Color32::from_rgb(0, 0x66, 0xFF)),
            );
        }
    }

    // Click handling against the computed layout.
    if let Some(pos) = response.interact_pointer_pos() {
        if response.clicked() {
            for track in 0..TRACK_COUNT {
                let row_y = origin.y + PAD + track as f32 * SECTION_H;
                let toggle_rect = Rect::from_min_size(
                    egui::pos2(origin.x, row_y),
                    Vec2::new(TOGGLE_W, SECTION_H - 2.0),
                );
                if toggle_rect.contains(pos) {
                    track_enables[track] = !track_enables[track];
                    return;
                }
                let row_rect = Rect::from_min_size(
                    egui::pos2(keys_x, row_y),
                    Vec2::new(keys_w, SECTION_H - 2.0),
                );
                if row_rect.contains(pos) {
                    *active_track = if *active_track == Some(track) {
                        None
                    } else {
                        Some(track)
                    };
                    return;
                }
            }
        }
    }
}

//! The keyboard strip above the piano roll and the note snapshot it draws from.

use egui::{Color32, Rect, Sense, Stroke, Vec2};

use crate::TRACK_COUNT;

pub struct VisSnapshot {
    pub active: bool,
    pub notes_on: [[bool; 128]; TRACK_COUNT],
    pub steps: u32,
    pub step_rate: f64,
    pub bpm: f64,
}

impl Default for VisSnapshot {
    fn default() -> Self {
        Self {
            active: false,
            notes_on: [[false; 128]; TRACK_COUNT],
            steps: 0,
            step_rate: 0.0,
            bpm: 0.0,
        }
    }
}

const SECTION_H: f32 = 24.0;
const WHITE_W: f32 = 9.0;
const WHITE_H: f32 = 18.0;
const BLACK_W: f32 = 6.0;
const BLACK_H: f32 = 11.0;
const TOGGLE_W: f32 = 18.0;
const PAD: f32 = 4.0;

const IS_BLACK: [bool; 12] = [
    false, true, false, false, true, false, true, false, false, true, false, true,
];
const KEY_NUM: [u32; 12] = [0, 0, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6];

#[allow(clippy::needless_range_loop)]
pub fn draw(ui: &mut egui::Ui, snap: &VisSnapshot, track_enables: &mut [bool; TRACK_COUNT]) {
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
                painter.rect_filled(key_rect, 0.0, alpha(Color32::BLACK));
            }
        }

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
                Color32::BLACK
            } else {
                Color32::from_rgb(0x33, 0x33, 0x33)
            };
            painter.rect_filled(key_rect, 0.0, alpha(base));
        }
    }

    if let Some(pos) = response.interact_pointer_pos()
        && response.clicked()
    {
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
        }
    }
}

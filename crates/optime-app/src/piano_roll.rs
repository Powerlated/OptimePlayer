use egui::{Align2, Color32, ColorImage, FontId, Pos2, Rect, Sense, Stroke};

use optime_core::{FsVisController, SongOverview};

use crate::TRACK_COUNT;
use crate::visualizer::VisSnapshot;

const MIDI_LO: u8 = 21;
const MIDI_HI: u8 = 108;
const LANES: usize = (MIDI_HI - MIDI_LO + 1) as usize;

const WINDOW_TICKS: f64 = 640.0;
const CURSOR_FRAC: f32 = 0.22;
pub const RUN_AHEAD_TICKS: u32 = 560;

const MIN_STEPS_PER_POINT: f64 = 0.02;
const MAX_STEPS_PER_POINT: f64 = 64.0;
const ZOOM_PER_SCROLL: f64 = 0.0025;
const VIEW_EASE_TAU: f64 = 0.07;
const MIN_NOTE_PX: f32 = 2.0;
const KEYBOARD_W: f32 = 40.0;
const CHORD_LANE_H: f32 = 20.0;

const IS_BLACK: [bool; 12] = [
    false, true, false, true, false, false, true, false, true, false, true, false,
];

struct NoteEvent {
    track: usize,
    pitch: u8,
    start: f64,
    end: f64,
    open: bool,
}

pub struct LaneSpan {
    pub start: f64,
    pub end: f64,
    pub label: String,
    pub fill: Option<Color32>,
}

#[derive(Default, Clone, Copy)]
pub struct RollInput {
    pub hover_step: Option<f64>,
    pub pointer_pos: Option<(f32, f32)>,
    pub over_lane: bool,
    pub drag_start_step: Option<f64>,
    pub drag_step: Option<f64>,
    pub drag_started: bool,
    pub dragging: bool,
    pub drag_stopped: bool,
    pub scrub_step: Option<f64>,
    pub secondary_clicked: bool,
}

#[derive(Clone, Copy)]
struct View {
    origin_step: f64,
    steps_per_point: f64,
    target_origin: f64,
    target_spp: f64,
}

impl Default for View {
    fn default() -> Self {
        View {
            origin_step: 0.0,
            steps_per_point: 1.0,
            target_origin: 0.0,
            target_spp: 1.0,
        }
    }
}

impl View {
    fn x_of_step(&self, roll_min_x: f32, step: f64) -> f32 {
        roll_min_x + ((step - self.origin_step) / self.steps_per_point) as f32
    }

    fn step_of_x(&self, roll_min_x: f32, x: f32) -> f64 {
        self.origin_step + (x - roll_min_x) as f64 * self.steps_per_point
    }

    fn snap(&mut self) {
        self.origin_step = self.target_origin;
        self.steps_per_point = self.target_spp;
    }

    fn ease(&mut self, dt: f64) {
        let a = 1.0 - (-dt.max(0.0) / VIEW_EASE_TAU).exp();
        self.origin_step += (self.target_origin - self.origin_step) * a;
        self.steps_per_point += (self.target_spp - self.steps_per_point) * a;
    }

    fn zoom_about(&mut self, roll_min_x: f32, anchor_x: f32, factor: f64) {
        let dx = (anchor_x - roll_min_x) as f64;
        let step_at_anchor = self.target_origin + dx * self.target_spp;
        self.target_spp =
            (self.target_spp * factor).clamp(MIN_STEPS_PER_POINT, MAX_STEPS_PER_POINT);
        self.target_origin = step_at_anchor - dx * self.target_spp;
    }

    fn pan_points(&mut self, dx_points: f64) {
        self.target_origin += dx_points * self.target_spp;
    }
}

#[derive(Default)]
pub struct PianoRoll {
    notes: Vec<NoteEvent>,
    chords: Vec<LaneSpan>,
    display_tick: f64,
    primed: bool,
    view: View,
    edit: bool,
    grid: Grid,
    last_roll_width: f32,
    selection: Option<(f64, f64)>,
    drag_start_step: Option<f64>,
}

#[derive(Clone, Copy)]
pub struct Grid {
    pub steps_per_beat: f64,
    pub beats_per_bar: u32,
    pub offset_steps: f64,
}

impl Default for Grid {
    fn default() -> Self {
        Grid {
            steps_per_beat: 0.0,
            beats_per_bar: 4,
            offset_steps: 0.0,
        }
    }
}

impl Grid {
    pub fn bar_steps(&self) -> Option<f64> {
        let b = self.steps_per_beat * self.beats_per_bar.max(1) as f64;
        (b > 0.0).then_some(b)
    }

    pub fn bar_beat_at(&self, step: f64) -> Option<(i64, u32)> {
        let bar_steps = self.bar_steps()?;
        let rel = step - self.offset_steps;
        let bar = (rel / bar_steps).floor();
        let within = rel - bar * bar_steps;
        let beat = (within / self.steps_per_beat).floor() as u32;
        Some((
            bar as i64 + 1,
            beat.min(self.beats_per_bar.saturating_sub(1)) + 1,
        ))
    }
}

impl PianoRoll {
    pub fn clear(&mut self) {
        self.notes.clear();
        self.chords.clear();
        self.display_tick = 0.0;
        self.primed = false;
    }

    pub fn set_chords(&mut self, spans: impl IntoIterator<Item = LaneSpan>) {
        self.chords = spans.into_iter().collect();
    }

    pub fn display_tick(&self) -> f64 {
        self.display_tick
    }

    pub fn set_display_tick(&mut self, tick: f64) {
        self.display_tick = tick;
        self.primed = true;
    }

    pub fn set_edit(&mut self, edit: bool) {
        self.edit = edit;
    }

    pub fn set_grid(&mut self, grid: Grid) {
        self.grid = grid;
    }

    pub fn set_selection(&mut self, sel: Option<(f64, f64)>) {
        self.selection = sel;
    }

    pub fn grid(&self) -> Grid {
        self.grid
    }

    pub fn visible_range(&self) -> (f64, f64) {
        if self.edit {
            let span = self.view.steps_per_point * self.last_roll_width.max(1.0) as f64;
            return (self.view.origin_step, self.view.origin_step + span);
        }
        let start = self.display_tick - WINDOW_TICKS * CURSOR_FRAC as f64;
        (start, start + WINDOW_TICKS)
    }

    pub fn advance(&mut self, snap: &VisSnapshot, dt: f64, playing: bool) {
        self.view.ease(dt);
        if !playing {
            return;
        }
        if !self.primed {
            self.display_tick = snap.steps as f64;
            self.primed = true;
            return;
        }
        self.display_tick += dt.max(0.0) * snap.step_rate.max(0.0);

        let target = snap.steps as f64;
        let err = target - self.display_tick;
        if err.abs() > 96.0 {
            self.display_tick = target;
        } else {
            self.display_tick += err * 0.10;
        }
    }

    pub fn ingest_overview(&mut self, overview: &SongOverview) {
        self.notes.clear();
        self.notes.extend(
            overview
                .notes
                .iter()
                .filter(|n| (MIDI_LO..=MIDI_HI).contains(&n.key))
                .map(|n| NoteEvent {
                    track: n.track,
                    pitch: n.key,
                    start: n.timestamp as f64,
                    end: n.timestamp as f64 + n.duration as f64,
                    open: false,
                }),
        );
    }

    pub fn zoom_to_fit(&mut self, total_steps: u32) {
        let w = self.last_roll_width.max(1.0) as f64;
        self.view.target_origin = 0.0;
        self.view.target_spp =
            (total_steps.max(1) as f64 / w).clamp(MIN_STEPS_PER_POINT, MAX_STEPS_PER_POINT);
        self.view.snap();
    }

    pub fn ingest(&mut self, look: &FsVisController) {
        self.notes.clear();
        let buf = &look.notes;
        for i in 0..buf.entries() {
            let Some(n) = buf.peek(i) else { continue };
            if !(MIDI_LO..=MIDI_HI).contains(&n.key) {
                continue;
            }
            let start = n.timestamp as f64;
            let open = n.duration == 0;
            let end = start + n.duration as f64;
            self.notes.push(NoteEvent {
                track: n.track,
                pitch: n.key,
                start,
                end,
                open,
            });
        }
    }

    pub fn draw(&mut self, ui: &mut egui::Ui, active: bool) -> RollInput {
        let size = ui.available_size_before_wrap();
        let sense = if self.edit {
            Sense::click_and_drag()
        } else {
            Sense::hover()
        };
        let (rect, resp) = ui.allocate_exact_size(size, sense);
        let painter = ui.painter_at(rect);

        painter.rect_filled(rect, 0.0, Color32::from_rgb(0x0c, 0x0e, 0x14));

        let lane_top = if self.chords.is_empty() && !self.edit {
            0.0
        } else {
            CHORD_LANE_H
        };
        let content = Rect::from_min_max(Pos2::new(rect.min.x, rect.min.y + lane_top), rect.max);

        let roll = Rect::from_min_max(
            Pos2::new(content.min.x + KEYBOARD_W, content.min.y),
            content.max,
        );
        let lane_h = roll.height() / LANES as f32;
        let dim = if active { 1.0 } else { 0.4 };
        self.last_roll_width = roll.width();

        if self.edit {
            self.handle_view_input(ui, &resp, roll);
        } else {
            let spp = WINDOW_TICKS / roll.width().max(1.0) as f64;
            let origin = self.display_tick - WINDOW_TICKS * CURSOR_FRAC as f64;
            self.view = View {
                origin_step: origin,
                steps_per_point: spp,
                target_origin: origin,
                target_spp: spp,
            };
        }
        let view = self.view;
        let xt = |tick: f64| view.x_of_step(roll.min.x, tick);
        let cursor_x = xt(self.display_tick);

        self.draw_lanes(&painter, roll, lane_h);
        self.draw_grid(&painter, roll, &xt);
        if lane_top > 0.0 {
            let strip = Rect::from_min_max(
                Pos2::new(roll.min.x, rect.min.y),
                Pos2::new(roll.max.x, rect.min.y + CHORD_LANE_H),
            );
            self.draw_chords(&painter, strip, &xt, dim);
        }

        let mut lit: [Option<usize>; 128] = [None; 128];
        for n in &self.notes {
            if n.start <= self.display_tick && (n.open || self.display_tick <= n.end) {
                lit[n.pitch as usize] = Some(n.track);
            }
        }
        self.draw_notes(&painter, roll, lane_h, cursor_x, &xt, dim);
        self.draw_cursor(&painter, roll, cursor_x);
        self.draw_keyboard(&painter, content, roll, lane_h, &lit, dim);

        if !self.edit {
            return RollInput::default();
        }
        self.collect_input(ui, &resp, rect, roll)
    }

    fn collect_input(
        &mut self,
        ui: &egui::Ui,
        resp: &egui::Response,
        rect: Rect,
        roll: Rect,
    ) -> RollInput {
        let view = self.view;
        let step_at = |x: f32| view.step_of_x(roll.min.x, x);
        let mut input = RollInput::default();

        if let Some(p) = ui.input(|i| i.pointer.hover_pos())
            && rect.contains(p)
            && p.x >= roll.min.x
        {
            input.hover_step = Some(step_at(p.x));
            input.over_lane = p.y <= rect.min.y + CHORD_LANE_H;
            input.pointer_pos = Some((p.x, p.y));
        }
        input.drag_started = resp.drag_started_by(egui::PointerButton::Secondary);
        input.dragging = resp.dragged_by(egui::PointerButton::Secondary);
        input.drag_stopped = resp.drag_stopped_by(egui::PointerButton::Secondary);
        input.secondary_clicked = resp.clicked_by(egui::PointerButton::Secondary);

        let drag_pos = resp.interact_pointer_pos();
        if (resp.clicked() || resp.dragged_by(egui::PointerButton::Primary))
            && let Some(p) = drag_pos
        {
            input.scrub_step = Some(step_at(p.x));
        }
        if input.dragging || input.drag_stopped {
            input.drag_step = drag_pos.map(|p| step_at(p.x));
            if let Some(p) = drag_pos {
                input.pointer_pos = Some((p.x, p.y));
            }
        }
        if input.drag_started {
            self.drag_start_step = drag_pos.map(|p| step_at(p.x));
        }
        input.drag_start_step = self.drag_start_step;
        if input.drag_stopped {
            self.drag_start_step = None;
        }
        input
    }

    fn draw_chords(
        &self,
        painter: &egui::Painter,
        strip: Rect,
        xt: &impl Fn(f64) -> f32,
        dim: f32,
    ) {
        let painter = painter.with_clip_rect(strip);
        let mid_y = (strip.min.y + strip.max.y) * 0.5;

        for c in &self.chords {
            let x0 = xt(c.start);
            let x1 = xt(c.end);
            if x1 < strip.min.x || x0 > strip.max.x {
                continue;
            }
            let now = c.start <= self.display_tick && self.display_tick < c.end;

            if let Some(fill) = c.fill {
                let block = Rect::from_min_max(
                    Pos2::new(x0.max(strip.min.x), strip.min.y + 1.0),
                    Pos2::new(x1.min(strip.max.x), strip.max.y - 1.0),
                );
                if block.width() > 0.5 {
                    painter.rect_filled(block, 2.0, scale_alpha(fill, dim));
                    painter.rect_stroke(
                        block,
                        2.0,
                        Stroke::new(
                            1.0_f32,
                            scale_alpha(lighten(fill, if now { 0.55 } else { 0.25 }), dim),
                        ),
                    );
                }
            } else if x0 >= strip.min.x {
                painter.line_segment(
                    [Pos2::new(x0, strip.min.y), Pos2::new(x0, strip.max.y)],
                    Stroke::new(
                        1.0_f32,
                        Color32::from_rgba_unmultiplied(0x3a, 0x42, 0x58, 90),
                    ),
                );
            }

            let color = match (c.fill.is_some(), now) {
                (true, _) => scale_alpha(Color32::from_rgb(0xf6, 0xf8, 0xff), dim),
                (false, true) => scale_alpha(Color32::from_rgb(0xf2, 0xf4, 0xff), dim),
                (false, false) => scale_alpha(Color32::from_rgb(0x9a, 0xa2, 0xb8), dim),
            };
            let text_clip = Rect::from_min_max(
                Pos2::new(x0.max(strip.min.x), strip.min.y),
                Pos2::new(x1.min(strip.max.x), strip.max.y),
            );
            let painter = if c.fill.is_some() {
                painter.with_clip_rect(text_clip.intersect(strip))
            } else {
                painter.clone()
            };
            let x = (x0 + 3.0).max(strip.min.x + 3.0);
            painter.text(
                Pos2::new(x, mid_y),
                Align2::LEFT_CENTER,
                &c.label,
                FontId::proportional(12.0),
                color,
            );
        }

        if let Some((s, e)) = self.selection {
            let (x0, x1) = (xt(s.min(e)), xt(s.max(e)));
            let r = Rect::from_min_max(Pos2::new(x0, strip.min.y), Pos2::new(x1, strip.max.y));
            painter.rect_filled(
                r,
                0.0,
                Color32::from_rgba_unmultiplied(0x8a, 0xb8, 0xff, 52),
            );
            painter.rect_stroke(
                r,
                0.0,
                Stroke::new(
                    1.5_f32,
                    Color32::from_rgba_unmultiplied(0xbf, 0xd8, 0xff, 235),
                ),
            );
        }
    }

    fn lane_y(roll: Rect, lane_h: f32, midi: u8) -> (f32, f32) {
        let i = (midi - MIDI_LO) as f32;
        let bottom = roll.max.y - i * lane_h;
        (bottom - lane_h, bottom)
    }

    fn draw_lanes(&self, painter: &egui::Painter, roll: Rect, lane_h: f32) {
        for midi in MIDI_LO..=MIDI_HI {
            let (top, bot) = Self::lane_y(roll, lane_h, midi);
            let pc = (midi % 12) as usize;
            if IS_BLACK[pc] {
                let r = Rect::from_min_max(Pos2::new(roll.min.x, top), Pos2::new(roll.max.x, bot));
                painter.rect_filled(r, 0.0, Color32::from_rgb(0x14, 0x16, 0x1f));
            }
            if pc == 0 {
                painter.line_segment(
                    [Pos2::new(roll.min.x, bot), Pos2::new(roll.max.x, bot)],
                    Stroke::new(1.0_f32, Color32::from_rgb(0x22, 0x26, 0x33)),
                );
            }
        }
    }

    fn handle_view_input(&mut self, ui: &egui::Ui, resp: &egui::Response, roll: Rect) {
        let hovered = resp.hovered();
        let (scroll, zoom_gesture, shift, pointer) = ui.input(|i| {
            (
                i.raw_scroll_delta,
                i.zoom_delta(),
                i.modifiers.shift,
                i.pointer.hover_pos(),
            )
        });
        let anchor = pointer.map(|p| p.x).unwrap_or(roll.center().x);

        if hovered {
            if shift && scroll.y != 0.0 {
                self.view.pan_points(-scroll.y as f64);
            } else if scroll.y != 0.0 {
                self.view.zoom_about(
                    roll.min.x,
                    anchor,
                    (-scroll.y as f64 * ZOOM_PER_SCROLL).exp(),
                );
            }
            if scroll.x != 0.0 {
                self.view.pan_points(-scroll.x as f64);
            }
            if zoom_gesture != 1.0 {
                self.view
                    .zoom_about(roll.min.x, anchor, 1.0 / zoom_gesture as f64);
            }
        }
        if resp.dragged_by(egui::PointerButton::Middle) {
            let dx = resp.drag_delta().x as f64;
            self.view.pan_points(-dx);
            self.view.origin_step = self.view.target_origin;
        }
    }

    fn draw_grid(&self, painter: &egui::Painter, roll: Rect, xt: &impl Fn(f64) -> f32) {
        const MIN_BEAT_PX: f32 = 6.0;
        const MIN_BAR_LABEL_PX: f32 = 28.0;

        let Some(bar_steps) = self.grid.bar_steps() else {
            return;
        };
        let (left, right) = (
            self.view.step_of_x(roll.min.x, roll.min.x),
            self.view.step_of_x(roll.min.x, roll.max.x),
        );
        let bar_px = (bar_steps / self.view.steps_per_point) as f32;
        let beat_px = bar_px / self.grid.beats_per_bar.max(1) as f32;
        let bar_stride = if bar_px >= 4.0 {
            1
        } else {
            ((4.0 / bar_px.max(0.01)).ceil() as i64).max(1)
        };

        let first_bar = ((left - self.grid.offset_steps) / bar_steps).floor() as i64;
        let last_bar = ((right - self.grid.offset_steps) / bar_steps).ceil() as i64;
        for bar in first_bar..=last_bar {
            if bar.rem_euclid(bar_stride) != 0 {
                continue;
            }
            let step = self.grid.offset_steps + bar as f64 * bar_steps;
            let x = xt(step);
            if x < roll.min.x || x > roll.max.x {
                continue;
            }
            painter.line_segment(
                [Pos2::new(x, roll.min.y), Pos2::new(x, roll.max.y)],
                Stroke::new(
                    1.0_f32,
                    Color32::from_rgba_unmultiplied(0x4a, 0x54, 0x70, 170),
                ),
            );
            if bar_px >= MIN_BAR_LABEL_PX {
                painter.text(
                    Pos2::new(x + 3.0, roll.min.y + 2.0),
                    Align2::LEFT_TOP,
                    format!("{}", bar + 1),
                    FontId::monospace(9.0),
                    Color32::from_rgba_unmultiplied(0x7a, 0x86, 0xa8, 200),
                );
            }
            if beat_px >= MIN_BEAT_PX {
                for beat in 1..self.grid.beats_per_bar.max(1) {
                    let bx = xt(step + beat as f64 * self.grid.steps_per_beat);
                    if bx < roll.min.x || bx > roll.max.x {
                        continue;
                    }
                    painter.line_segment(
                        [Pos2::new(bx, roll.min.y), Pos2::new(bx, roll.max.y)],
                        Stroke::new(
                            1.0_f32,
                            Color32::from_rgba_unmultiplied(0x30, 0x36, 0x48, 80),
                        ),
                    );
                }
            }
        }
    }

    fn draw_notes(
        &self,
        painter: &egui::Painter,
        roll: Rect,
        lane_h: f32,
        cursor_x: f32,
        xt: &impl Fn(f64) -> f32,
        dim: f32,
    ) {
        for n in &self.notes {
            let x_start = xt(n.start);
            let x_end = if n.open {
                roll.max.x
            } else {
                xt(n.end).max(x_start + MIN_NOTE_PX)
            };
            if x_end < roll.min.x || x_start > roll.max.x {
                continue;
            }
            let x0 = x_start.max(roll.min.x);
            let x1 = x_end.min(roll.max.x);
            if x1 <= x0 {
                continue;
            }

            let (top, bot) = Self::lane_y(roll, lane_h, n.pitch);
            let pad = (lane_h * 0.14).clamp(0.5, 2.0);
            let bar = Rect::from_min_max(Pos2::new(x0, top + pad), Pos2::new(x1, bot - pad));

            let playing = n.start <= self.display_tick && (n.open || self.display_tick <= n.end);
            let base = track_color(n.track);
            let a = if x_start > cursor_x { 0.6 } else { 1.0 } * dim;
            let rounding = (bar.height() * 0.35).min(3.0);

            let core = if playing { lighten(base, 0.25) } else { base };
            painter.rect_filled(bar, rounding, scale_alpha(core, a));
            if playing {
                painter.rect_stroke(
                    bar,
                    rounding,
                    Stroke::new(1.0_f32, scale_alpha(Color32::WHITE, 0.35 * dim)),
                );
            }
        }
    }

    fn draw_cursor(&self, painter: &egui::Painter, roll: Rect, cursor_x: f32) {
        painter.line_segment(
            [
                Pos2::new(cursor_x, roll.min.y),
                Pos2::new(cursor_x, roll.max.y),
            ],
            Stroke::new(
                1.0_f32,
                Color32::from_rgba_unmultiplied(0xff, 0xff, 0xff, 70),
            ),
        );
    }

    fn white_key_bounds(roll: Rect, lane_h: f32, midi: u8) -> (f32, f32) {
        let (mut top, mut bot) = Self::lane_y(roll, lane_h, midi);
        if midi > MIDI_LO && IS_BLACK[((midi - 1) % 12) as usize] {
            bot += lane_h / 2.0;
        }
        if midi < MIDI_HI && IS_BLACK[((midi + 1) % 12) as usize] {
            top -= lane_h / 2.0;
        }
        (top, bot)
    }

    fn draw_keyboard(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        roll: Rect,
        lane_h: f32,
        lit: &[Option<usize>; 128],
        dim: f32,
    ) {
        let kb = Rect::from_min_max(rect.min, Pos2::new(rect.min.x + KEYBOARD_W, rect.max.y));

        for midi in MIDI_LO..=MIDI_HI {
            if IS_BLACK[(midi % 12) as usize] {
                continue;
            }
            let (top, bot) = Self::white_key_bounds(roll, lane_h, midi);
            let key = Rect::from_min_max(
                Pos2::new(kb.min.x, top.max(kb.min.y)),
                Pos2::new(kb.max.x, bot.min(kb.max.y)),
            );
            let fill = match lit[midi as usize].map(track_color) {
                Some(c) => lighten(c, 0.35),
                None => Color32::from_rgb(0xd9, 0xdb, 0xe1),
            };
            painter.rect_filled(key, 0.0, scale_alpha(fill, dim));
            painter.line_segment(
                [Pos2::new(kb.min.x, bot), Pos2::new(kb.max.x, bot)],
                Stroke::new(0.6_f32, Color32::from_rgba_unmultiplied(0, 0, 0, 90)),
            );
        }

        for midi in MIDI_LO..=MIDI_HI {
            if !IS_BLACK[(midi % 12) as usize] {
                continue;
            }
            let (top, bot) = Self::lane_y(roll, lane_h, midi);
            let inset = (lane_h * 0.08).clamp(0.3, 1.0);
            let key = Rect::from_min_max(
                Pos2::new(kb.max.x - KEYBOARD_W * 0.62, top + inset),
                Pos2::new(kb.max.x, bot - inset),
            );
            let fill = match lit[midi as usize].map(track_color) {
                Some(c) => lighten(c, 0.1),
                None => Color32::from_rgb(0x16, 0x17, 0x1c),
            };
            painter.rect_filled(key, 1.5, scale_alpha(fill, dim));
        }

        painter.line_segment(
            [Pos2::new(kb.max.x, kb.min.y), Pos2::new(kb.max.x, kb.max.y)],
            Stroke::new(1.0_f32, Color32::from_rgb(0x05, 0x06, 0x0a)),
        );
    }
}

pub fn overview_image(overview: &SongOverview, width: usize, height: usize) -> ColorImage {
    let bg = Color32::from_rgb(0x0c, 0x0e, 0x14);
    let mut pixels = vec![bg; width.max(1) * height.max(1)];
    let (w, h) = (width.max(1), height.max(1));
    let total = overview.total_steps.max(1) as f64;

    let mut plot = |x: usize, y: usize, c: Color32| {
        if x < w && y < h {
            pixels[y * w + x] = c;
        }
    };

    for n in &overview.notes {
        if !(MIDI_LO..=MIDI_HI).contains(&n.key) {
            continue;
        }
        let start = n.timestamp as f64;
        let end = start + n.duration.max(1) as f64;
        let x0 = ((start / total) * w as f64).floor() as usize;
        let x1 = (((end / total) * w as f64).ceil() as usize).max(x0 + 1);
        let lane = (n.key - MIDI_LO) as f64 / (LANES - 1) as f64;
        let row = ((h - 1) as f64 * (1.0 - lane)).round() as usize;
        let color = track_color(n.track);
        for x in x0..x1.min(w) {
            plot(x, row, color);
        }
    }
    ColorImage {
        size: [w, h],
        pixels,
    }
}

fn track_color(track: usize) -> Color32 {
    let hue = (track as f32 * 360.0 / TRACK_COUNT as f32) / 360.0;
    hsv(hue, 0.72, 1.0)
}

pub fn chord_color(root_pc: Option<u8>, darker: bool) -> Color32 {
    let Some(pc) = root_pc else {
        return Color32::from_rgb(0x4a, 0x50, 0x60);
    };
    let fifth = (pc as u32 * 7) % 12;
    let v = if darker { 0.52 } else { 0.72 };
    hsv(fifth as f32 / 12.0, 0.55, v)
}

fn scale_alpha(c: Color32, f: f32) -> Color32 {
    let a = (c.a() as f32 * f.clamp(0.0, 1.0)) as u8;
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), a)
}

fn lighten(c: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let mix = |v: u8| (v as f32 + (255.0 - v as f32) * t) as u8;
    Color32::from_rgba_unmultiplied(mix(c.r()), mix(c.g()), mix(c.b()), c.a())
}

fn hsv(h: f32, s: f32, v: f32) -> Color32 {
    let h6 = (h.rem_euclid(1.0)) * 6.0;
    let i = h6.floor() as i32;
    let f = h6 - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let (r, g, b) = match i.rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

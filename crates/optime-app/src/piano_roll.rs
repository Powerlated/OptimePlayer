//! FL-Studio-style piano-roll visualizer drawn with [`egui::Painter`].
//!
//! The 88-key piano keyboard is fixed on the **left**; pitch increases upward, one semitone per
//! lane. A stationary **playhead cursor** sits a fixed fraction in from the keyboard. Notes are
//! laid out along a tick timeline and scroll right→left *through* the cursor: notes to the right
//! of the cursor are upcoming, notes to the left have already played. A note's key lights as it
//! crosses the cursor — exactly when it sounds.
//!
//! Each note is drawn as a **ribbon** whose vertical thickness tracks its volume envelope
//! (velocity-scaled attack / sustain / release) and whose centerline bends with pitch-bend over
//! time. Both the note timeline and the pitch-bend timeline come from the engine's look-ahead
//! [`FsVisController`]. The horizontal scale is in sequence *ticks*, so scroll speed tracks tempo.
//!
//! All rendering is isolated here so the draw layer can later be swapped for a `wgpu` paint
//! callback without touching the app.

use egui::epaint::{Vertex, WHITE_UV};
use egui::{Color32, Mesh, Pos2, Rect, Sense, Shape, Stroke};

use optime_core::FsVisController;

use crate::visualizer::VisSnapshot;
use crate::TRACK_COUNT;

/// Lowest/highest MIDI notes shown (A0..=C8, the 88-key piano range).
const MIDI_LO: u8 = 21;
const MIDI_HI: u8 = 108;
const LANES: usize = (MIDI_HI - MIDI_LO + 1) as usize; // 88

/// Total ticks spanned by the roll width.
const WINDOW_TICKS: f64 = 640.0;
/// Cursor position as a fraction of the roll width from the keyboard edge.
const CURSOR_FRAC: f32 = 0.22;
/// Ticks of look-ahead to keep buffered (must exceed the visible future span).
pub const RUN_AHEAD_TICKS: u32 = 560;
/// Spacing of the scrolling vertical time grid, in ticks.
const GRID_TICKS: f64 = 96.0;
/// Minimum gate length so zero/short-duration notes are still visible.
const MIN_NOTE_TICKS: f64 = 24.0;
/// Width of the vertical keyboard, in points.
const KEYBOARD_W: f32 = 40.0;

/// Stylized volume-envelope shape, in ticks. (Per-instrument ADSR isn't available to the
/// look-ahead, so this is a generic illustrative envelope scaled by note velocity.)
const ATTACK_TICKS: f64 = 5.0;
const RELEASE_TICKS: f64 = 40.0;

/// Pixel step when tessellating a note ribbon along its length.
const RIBBON_STEP: f32 = 4.0;

/// Sequence ticks per second per unit BPM: `(33_513_982 / (64 * 2728)) / 240`.
const TICK_RATE_PER_BPM: f64 = 0.799_837;

/// `midi % 12` is a black key. Index 0 == C.
const IS_BLACK: [bool; 12] = [
    false, true, false, true, false, false, true, false, true, false, true, false,
];

/// One note on the timeline (start/end in sequence ticks; velocity 0..1).
struct NoteEvent {
    track: usize,
    pitch: u8,
    start: f64,
    /// Gate end (note-off); the ribbon extends a release tail past this.
    end: f64,
    velocity: f32,
}

/// Piano-roll renderer: owns the visible note + pitch-bend timelines and a smoothed playhead.
#[derive(Default)]
pub struct PianoRoll {
    notes: Vec<NoteEvent>,
    /// Per-track pitch-bend timeline, `(tick, semitones)`, chronological.
    bends: Vec<Vec<(f64, f32)>>,
    /// Wall-clock-smoothed playhead position, in ticks (drives the scroll).
    display_tick: f64,
    /// Whether anything has been ingested yet (first frame snaps to the audio position).
    primed: bool,
}

impl PianoRoll {
    /// Resets the roll for a new song.
    pub fn clear(&mut self) {
        self.notes.clear();
        self.bends.clear();
        self.display_tick = 0.0;
        self.primed = false;
    }

    /// The smoothed playhead position in ticks (used to drive the look-ahead target).
    pub fn display_tick(&self) -> f64 {
        self.display_tick
    }

    /// Advances the smoothed playhead clock by `dt` seconds toward the audio position.
    ///
    /// The audio thread advances ticks in coarse bursts (one full output buffer at a time), so we
    /// integrate at the current tempo for smooth motion and gently correct toward the reported
    /// position to stay locked, snapping on large jumps (loop / restart / seek).
    pub fn advance(&mut self, snap: &VisSnapshot, dt: f64, playing: bool) {
        if !playing {
            return;
        }
        if !self.primed {
            self.display_tick = snap.ticks as f64;
            self.primed = true;
            return;
        }
        let rate = TICK_RATE_PER_BPM * snap.bpm.max(1) as f64;
        self.display_tick += dt.max(0.0) * rate;

        let target = snap.ticks as f64;
        let err = target - self.display_tick;
        if err.abs() > 96.0 {
            self.display_tick = target; // loop / restart — snap.
        } else {
            self.display_tick += err * 0.10; // gentle phase lock.
        }
    }

    /// Rebuilds the note and pitch-bend timelines from the look-ahead controller's buffers.
    pub fn ingest(&mut self, look: &FsVisController) {
        self.notes.clear();
        let buf = &look.active_notes;
        for i in 0..buf.entries() {
            let Some(m) = buf.peek(i) else { continue };
            let pitch = m.param0;
            if !(MIDI_LO as i32..=MIDI_HI as i32).contains(&pitch) {
                continue;
            }
            let start = m.timestamp as f64;
            let end = start + (m.param2.max(0) as f64).max(MIN_NOTE_TICKS);
            self.notes.push(NoteEvent {
                track: m.track_num,
                pitch: pitch as u8,
                start,
                end,
                velocity: (m.param1.clamp(0, 127) as f32) / 127.0,
            });
        }

        // Pitch-bend timeline, bucketed per track (events arrive chronologically).
        self.bends.clear();
        self.bends.resize_with(TRACK_COUNT, Vec::new);
        let pb = &look.pitch_bends;
        for i in 0..pb.entries() {
            let Some(e) = pb.peek(i) else { continue };
            if let Some(v) = self.bends.get_mut(e.track) {
                v.push((e.timestamp as f64, e.semitones));
            }
        }
    }

    /// Bend in semitones for `track` at `tick` (the most recent event at or before `tick`).
    fn bend_at(&self, track: usize, tick: f64) -> f32 {
        let Some(v) = self.bends.get(track) else {
            return 0.0;
        };
        if v.is_empty() {
            return 0.0;
        }
        let idx = v.partition_point(|&(t, _)| t <= tick);
        if idx == 0 {
            0.0
        } else {
            v[idx - 1].1
        }
    }

    /// Paints the roll, filling the available space.
    pub fn draw(&self, ui: &mut egui::Ui, active: bool) {
        let size = ui.available_size_before_wrap();
        let (rect, _resp) = ui.allocate_exact_size(size, Sense::hover());
        let painter = ui.painter_at(rect);

        painter.rect_filled(rect, 0.0, Color32::from_rgb(0x0c, 0x0e, 0x14));

        let roll = Rect::from_min_max(Pos2::new(rect.min.x + KEYBOARD_W, rect.min.y), rect.max);
        let lane_h = roll.height() / LANES as f32;
        let cursor_x = roll.min.x + roll.width() * CURSOR_FRAC;
        let ppt = roll.width() as f64 / WINDOW_TICKS; // points per tick
        let dim = if active { 1.0 } else { 0.4 };

        // tick → x within the roll.
        let xt = |tick: f64| cursor_x + ((tick - self.display_tick) * ppt) as f32;

        self.draw_lanes(&painter, roll, lane_h);
        self.draw_grid(&painter, roll, &xt);

        // Which pitches are under the cursor (for key lighting).
        let mut lit: [Option<usize>; 128] = [None; 128];
        for n in &self.notes {
            if n.start <= self.display_tick && self.display_tick <= n.end {
                lit[n.pitch as usize] = Some(n.track);
            }
        }

        self.draw_notes(&painter, roll, lane_h, cursor_x, ppt, dim);
        self.draw_cursor(&painter, roll, cursor_x);
        self.draw_keyboard(&painter, rect, roll, lane_h, &lit, dim);
    }

    /// y-range (top, bottom) of a pitch's lane within `roll`.
    fn lane_y(roll: Rect, lane_h: f32, midi: u8) -> (f32, f32) {
        let i = (midi - MIDI_LO) as f32; // 0 == lowest
        let bottom = roll.max.y - i * lane_h;
        (bottom - lane_h, bottom)
    }

    /// Alternating lane shading: black-key rows darker, octave dividers marked.
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
                    Stroke::new(1.0, Color32::from_rgb(0x22, 0x26, 0x33)),
                );
            }
        }
    }

    /// Vertical time grid, scrolling with the playhead.
    fn draw_grid(&self, painter: &egui::Painter, roll: Rect, xt: &impl Fn(f64) -> f32) {
        let first = ((self.display_tick - WINDOW_TICKS) / GRID_TICKS).floor() * GRID_TICKS;
        let mut t = first;
        while xt(t) <= roll.max.x {
            let x = xt(t);
            if x >= roll.min.x {
                painter.line_segment(
                    [Pos2::new(x, roll.min.y), Pos2::new(x, roll.max.y)],
                    Stroke::new(1.0, Color32::from_rgba_unmultiplied(0x30, 0x36, 0x48, 80)),
                );
            }
            t += GRID_TICKS;
        }
    }

    /// The note ribbons: thickness from the volume envelope, centerline bent by pitch-bend.
    fn draw_notes(
        &self,
        painter: &egui::Painter,
        roll: Rect,
        lane_h: f32,
        cursor_x: f32,
        ppt: f64,
        dim: f32,
    ) {
        let xt = |tick: f64| cursor_x + ((tick - self.display_tick) * ppt) as f32;
        let half_max = lane_h * 0.5; // a full-velocity sustain spans ~one lane.
        let half_min = (lane_h * 0.12).max(0.6);

        let mut mesh = Mesh::default();
        for n in &self.notes {
            let gate = n.end - n.start;
            let tail = n.end + RELEASE_TICKS;
            let x_start = xt(n.start);
            let x_end = xt(tail);
            if x_end < roll.min.x || x_start > roll.max.x {
                continue;
            }
            let lo = x_start.max(roll.min.x);
            let hi = x_end.min(roll.max.x);
            if hi <= lo {
                continue;
            }

            let base = track_color(n.track);
            let (_, bot0) = Self::lane_y(roll, lane_h, n.pitch);
            let lane_center = bot0 - lane_h * 0.5;

            // Sample the ribbon's centerline + half-thickness across its visible span.
            let mut pts: Vec<(f32, f32, f32, bool)> = Vec::new(); // (x, center_y, half, playing)
            let mut x = lo;
            loop {
                let sx = x.min(hi);
                let tick = self.display_tick + (sx - cursor_x) as f64 / ppt;
                let progress = tick - n.start;
                let env = envelope(progress, gate);
                let half = (half_min + (half_max - half_min) * n.velocity * env).max(0.4);
                let bend = self.bend_at(n.track, tick);
                let center = lane_center - bend * lane_h;
                let playing = tick >= n.start && tick <= n.end;
                pts.push((sx, center, half, playing));
                if sx >= hi {
                    break;
                }
                x += RIBBON_STEP;
            }

            // Emit quads between consecutive samples.
            for w in pts.windows(2) {
                let (x0, c0, h0, p0) = w[0];
                let (x1, c1, h1, p1) = w[1];
                let alpha = if (x0 + x1) * 0.5 > cursor_x { 0.72 } else { 1.0 } * dim;
                let col = if p0 || p1 { lighten(base, 0.25) } else { base };
                let col = scale_alpha(col, alpha);
                push_quad(
                    &mut mesh,
                    Pos2::new(x0, c0 - h0),
                    Pos2::new(x1, c1 - h1),
                    Pos2::new(x1, c1 + h1),
                    Pos2::new(x0, c0 + h0),
                    col,
                );
            }
        }
        if !mesh.is_empty() {
            painter.add(Shape::mesh(mesh));
        }
    }

    /// The stationary playhead cursor.
    fn draw_cursor(&self, painter: &egui::Painter, roll: Rect, cursor_x: f32) {
        painter.line_segment(
            [Pos2::new(cursor_x, roll.min.y), Pos2::new(cursor_x, roll.max.y)],
            Stroke::new(1.0, Color32::from_rgb(0x8a, 0x96, 0xff)),
        );
    }

    /// The vertical keyboard down the left edge: a solid white column with black keys overlaid,
    /// lit per the cursor crossing.
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
        // Solid white column (the white keys), then per-key tint/black-key overlay.
        painter.rect_filled(kb, 0.0, scale_alpha(Color32::from_rgb(0xd2, 0xd7, 0xe2), dim));

        let black_w = KEYBOARD_W * 0.62;
        for midi in MIDI_LO..=MIDI_HI {
            let (top, bot) = Self::lane_y(roll, lane_h, midi);
            let pc = (midi % 12) as usize;
            let black = IS_BLACK[pc];
            let on = lit[midi as usize].map(track_color);

            if black {
                // Black key: short bar anchored at the inner (right) edge facing the roll.
                let key = Rect::from_min_max(Pos2::new(kb.max.x - black_w, top), Pos2::new(kb.max.x, bot));
                let fill = match on {
                    Some(c) => lighten(c, 0.1),
                    None => Color32::from_rgb(0x12, 0x14, 0x1c),
                };
                painter.rect_filled(key, 0.0, scale_alpha(fill, dim));
            } else {
                // White key: tint the whole lane if lit; always draw a faint separator line.
                if let Some(c) = on {
                    let key = Rect::from_min_max(Pos2::new(kb.min.x, top), Pos2::new(kb.max.x, bot));
                    painter.rect_filled(key, 0.0, scale_alpha(lighten(c, 0.25), dim));
                }
                painter.line_segment(
                    [Pos2::new(kb.min.x, bot), Pos2::new(kb.max.x, bot)],
                    Stroke::new(0.5, Color32::from_rgb(0x9a, 0x9f, 0xab)),
                );
            }
        }
        // Edge between keyboard and roll.
        painter.line_segment(
            [Pos2::new(kb.max.x, kb.min.y), Pos2::new(kb.max.x, kb.max.y)],
            Stroke::new(1.0, Color32::from_rgb(0x05, 0x06, 0x0a)),
        );
    }
}

/// Stylized volume envelope in 0..1: quick attack, full sustain through the gate, release tail.
fn envelope(progress: f64, gate: f64) -> f32 {
    if progress < 0.0 {
        return 0.0;
    }
    let v = if progress < ATTACK_TICKS {
        progress / ATTACK_TICKS
    } else if progress <= gate {
        1.0
    } else {
        1.0 - (progress - gate) / RELEASE_TICKS
    };
    v.clamp(0.0, 1.0) as f32
}

/// Appends a colored quad (p0→p1→p2→p3) to `mesh`.
fn push_quad(mesh: &mut Mesh, p0: Pos2, p1: Pos2, p2: Pos2, p3: Pos2, color: Color32) {
    let base = mesh.vertices.len() as u32;
    for p in [p0, p1, p2, p3] {
        mesh.vertices.push(Vertex {
            pos: p,
            uv: WHITE_UV,
            color,
        });
    }
    mesh.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// A vivid per-track colour spread around the hue wheel.
fn track_color(track: usize) -> Color32 {
    let hue = (track as f32 * 360.0 / TRACK_COUNT as f32) / 360.0;
    hsv(hue, 0.72, 1.0)
}

/// Multiplies a colour's alpha by `f` (0..1), preserving RGB.
fn scale_alpha(c: Color32, f: f32) -> Color32 {
    let a = (c.a() as f32 * f.clamp(0.0, 1.0)) as u8;
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), a)
}

/// Blends a colour toward white by `t` (0..1).
fn lighten(c: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let mix = |v: u8| (v as f32 + (255.0 - v as f32) * t) as u8;
    Color32::from_rgba_unmultiplied(mix(c.r()), mix(c.g()), mix(c.b()), c.a())
}

/// HSV→`Color32` with full opacity. `h`, `s`, `v` in 0..1.
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

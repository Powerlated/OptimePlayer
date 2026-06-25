//! FL-Studio-style piano-roll visualizer drawn with [`egui::Painter`].
//!
//! The 88-key piano keyboard is fixed on the **left**; pitch increases upward, one semitone per
//! lane. A stationary **playhead cursor** sits a fixed fraction in from the keyboard. Notes are
//! laid out along a tick timeline and scroll right→left *through* the cursor: notes to the right
//! of the cursor are upcoming, notes to the left have already played. A note's key lights and the
//! bar flares as it crosses the cursor — exactly when it sounds.
//!
//! Future notes come from the engine's look-ahead [`FsVisController`], whose note buffer this
//! module ingests each frame. The horizontal scale is in sequencer *steps* (DS: SSEQ ticks;
//! GBA: MP2K tempo steps), so the scroll speed tracks tempo automatically.
//!
//! All rendering is isolated here so the draw layer can later be swapped for a `wgpu` paint
//! callback (true shader bloom / particles) without touching the app.

use egui::{Color32, ColorImage, Pos2, Rect, Sense, Stroke};

use optime_core::{FsVisController, SongOverview};

use crate::visualizer::VisSnapshot;
use crate::TRACK_COUNT;

/// Lowest/highest MIDI notes shown (A0..=C8, the 88-key piano range).
const MIDI_LO: u8 = 21;
const MIDI_HI: u8 = 108;
const LANES: usize = (MIDI_HI - MIDI_LO + 1) as usize; // 88

/// Total steps spanned by the roll width.
const WINDOW_TICKS: f64 = 640.0;
/// Cursor position as a fraction of the roll width from the keyboard edge.
const CURSOR_FRAC: f32 = 0.22;
/// Steps of look-ahead to keep buffered (must exceed the visible future span).
pub const RUN_AHEAD_TICKS: u32 = 560;
/// Spacing of the scrolling vertical time grid, in steps.
const GRID_TICKS: f64 = 96.0;
/// Minimum drawn note width, in points, so very short (or not-yet-resolved) notes still show as a
/// sliver. Applied at draw time rather than as a tick floor, so the stored lengths stay true to
/// the real note durations.
const MIN_NOTE_PX: f32 = 2.0;
/// Width of the vertical keyboard, in points.
const KEYBOARD_W: f32 = 40.0;

/// `midi % 12` is a black key. Index 0 == C.
const IS_BLACK: [bool; 12] = [
    false, true, false, true, false, false, true, false, true, false, true, false,
];

/// One note on the timeline (start/end in sequence ticks).
struct NoteEvent {
    track: usize,
    pitch: u8,
    start: f64,
    end: f64,
    /// The note's end-command hasn't been received yet (a still-held note / unresolved tie,
    /// signalled by a zero look-ahead duration). `end` is unknown, so it is drawn as still
    /// sounding and extends off the right edge of the roll until its length resolves.
    open: bool,
}

/// Piano-roll renderer: owns the visible note list and a smoothed playhead clock.
#[derive(Default)]
pub struct PianoRoll {
    notes: Vec<NoteEvent>,
    /// Wall-clock-smoothed playhead position, in ticks (drives the scroll).
    display_tick: f64,
    /// Whether anything has been ingested yet (so the first frame snaps to the audio position).
    primed: bool,
}

impl PianoRoll {
    /// Resets the roll for a new song.
    pub fn clear(&mut self) {
        self.notes.clear();
        self.display_tick = 0.0;
        self.primed = false;
    }

    /// The smoothed playhead position in ticks (used to drive the look-ahead target).
    pub fn display_tick(&self) -> f64 {
        self.display_tick
    }

    /// The `[start, end)` step range currently visible in the roll (the cursor sits
    /// [`CURSOR_FRAC`] of the way in). Used to highlight the visible window on the overview bar.
    pub fn visible_range(&self) -> (f64, f64) {
        let start = self.display_tick - WINDOW_TICKS * CURSOR_FRAC as f64;
        (start, start + WINDOW_TICKS)
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
            self.display_tick = snap.steps as f64;
            self.primed = true;
            return;
        }
        self.display_tick += dt.max(0.0) * snap.step_rate.max(0.0);

        let target = snap.steps as f64;
        let err = target - self.display_tick;
        if err.abs() > 96.0 {
            self.display_tick = target; // loop / restart — snap.
        } else {
            self.display_tick += err * 0.10; // gentle phase lock.
        }
    }

    /// Rebuilds the note list from the look-ahead controller's buffer.
    pub fn ingest(&mut self, look: &FsVisController) {
        self.notes.clear();
        let buf = &look.notes;
        for i in 0..buf.entries() {
            let Some(n) = buf.peek(i) else { continue };
            if !(MIDI_LO..=MIDI_HI).contains(&n.key) {
                continue;
            }
            // Real note length: gate time for GBA (ties resolved in the look-ahead), the note's
            // duration in ticks for DS. A minimum *width* is enforced at draw time instead of
            // padding the length here, so the bars match the true note durations. A zero duration
            // means the end hasn't been received yet (a held note / open tie) — flag it so the
            // bar runs off the right edge rather than collapsing to a sliver.
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

        // Which pitches are currently under the cursor (for key lighting).
        let mut lit: [Option<usize>; 128] = [None; 128];
        for n in &self.notes {
            // An open note (end not yet received) stays lit for as long as it is held.
            if n.start <= self.display_tick && (n.open || self.display_tick <= n.end) {
                lit[n.pitch as usize] = Some(n.track);
            }
        }
        self.draw_notes(&painter, roll, lane_h, cursor_x, &xt, dim);
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
                    Stroke::new(1.0_f32, Color32::from_rgb(0x22, 0x26, 0x33)),
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
                    Stroke::new(
                        1.0_f32,
                        Color32::from_rgba_unmultiplied(0x30, 0x36, 0x48, 80),
                    ),
                );
            }
            t += GRID_TICKS;
        }
    }

    /// The note bars: flat rounded capsules, brightening while they cross the cursor.
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
            // An open note (no end command yet) runs off the right edge; otherwise keep at least
            // a sliver visible for very short notes.
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
            // Upcoming notes (fully right of the cursor) are slightly dimmer.
            let a = if x_start > cursor_x { 0.6 } else { 1.0 } * dim;
            let rounding = (bar.height() * 0.35).min(3.0);

            // Flat, clean capsule; gently brighter while sounding.
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

    /// The stationary playhead: a single hairline.
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

    /// The lower / upper bounds of a *white* key: it expands into the halves of any adjacent
    /// black-key lanes, like a real piano (Logic-style vertical keyboard).
    fn white_key_bounds(roll: Rect, lane_h: f32, midi: u8) -> (f32, f32) {
        let (mut top, mut bot) = Self::lane_y(roll, lane_h, midi);
        if midi > MIDI_LO && IS_BLACK[((midi - 1) % 12) as usize] {
            bot += lane_h / 2.0; // expand down over the black lane below
        }
        if midi < MIDI_HI && IS_BLACK[((midi + 1) % 12) as usize] {
            top -= lane_h / 2.0; // expand up over the black lane above
        }
        (top, bot)
    }

    /// The vertical keyboard down the left edge: contiguous white keys with shorter black keys
    /// overlaid (anchored toward the roll), lit per the cursor crossing.
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

        // White keys first: full-width blocks spanning into adjacent black lanes.
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
            // Hairline separator at the key's lower edge.
            painter.line_segment(
                [Pos2::new(kb.min.x, bot), Pos2::new(kb.max.x, bot)],
                Stroke::new(0.6_f32, Color32::from_rgba_unmultiplied(0, 0, 0, 90)),
            );
        }

        // Black keys overlaid: shorter, slightly slimmer, anchored at the roll-side edge.
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

        // Subtle edge between keyboard and roll.
        painter.line_segment(
            [Pos2::new(kb.max.x, kb.min.y), Pos2::new(kb.max.x, kb.max.y)],
            Stroke::new(1.0_f32, Color32::from_rgb(0x05, 0x06, 0x0a)),
        );
    }
}

/// Rasterizes an entire song's note timeline into a small image for the piano-roll overview bar:
/// time spans the width left→right, pitch the height (low at the bottom), one dot per note tinted
/// by track. Background matches the roll so it reads as a zoomed-out mini piano roll.
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

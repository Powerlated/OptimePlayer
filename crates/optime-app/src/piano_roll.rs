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

use egui::{Align2, Color32, ColorImage, FontId, Pos2, Rect, Sense, Stroke};

use optime_core::{FsVisController, SongOverview};

use crate::TRACK_COUNT;
use crate::visualizer::VisSnapshot;

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

/// Zoom limits, in steps per point. The floor is "a handful of steps across the roll" (fine enough
/// to place a chord edge on a sixteenth); the ceiling comfortably fits the longest song.
const MIN_STEPS_PER_POINT: f64 = 0.02;
const MAX_STEPS_PER_POINT: f64 = 64.0;
/// Zoom multiplier per unit of scroll.
const ZOOM_PER_SCROLL: f64 = 0.0025;
/// Exponential time constant (seconds) the viewport eases toward its target with. Small enough to
/// feel immediate, long enough that a wheel notch glides instead of jumping.
const VIEW_EASE_TAU: f64 = 0.07;
/// Minimum drawn note width, in points, so very short (or not-yet-resolved) notes still show as a
/// sliver. Applied at draw time rather than as a tick floor, so the stored lengths stay true to
/// the real note durations.
const MIN_NOTE_PX: f32 = 2.0;
/// Width of the vertical keyboard, in points.
const KEYBOARD_W: f32 = 40.0;
/// Height of the chord-label lane reserved across the top of the roll (only when a
/// song has pre-inferred chords). Chosen to fit one line of the chord font.
const CHORD_LANE_H: f32 = 20.0;

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

/// One chord change on the timeline: a step span and its display label (`"<roman> (<name>)"`, or
/// `"N.C."` for no-chord). Drawn in the chord lane above the roll. Either pre-inferred
/// ([`crate::chord_data`]) or, in annotation mode, hand-authored ([`crate::annotation`]).
///
/// `fill` is what makes an *annotated* span unmistakable: pre-inferred chords are text on bare
/// background, while a hand-authored one is a solid block. Unlabelled stretches stay empty, so
/// "done" and "not done" are readable at a glance across a whole zoomed-out song.
pub struct LaneSpan {
    pub start: f64,
    pub end: f64,
    pub label: String,
    pub fill: Option<Color32>,
}

/// What the pointer did over the roll this frame, in *step* coordinates.
///
/// The roll reports geometry and raw input; it never touches the annotation model. That stays in
/// [`crate::annotation`], so the renderer has no idea what a chord is.
#[derive(Default, Clone, Copy)]
pub struct RollInput {
    /// Step under the pointer, when it's over the roll at all.
    pub hover_step: Option<f64>,
    /// Pointer position in screen space — where a popup should be anchored.
    pub pointer_pos: Option<(f32, f32)>,
    /// Whether the pointer is over the chord lane rather than the note area.
    pub over_lane: bool,
    /// Step the current annotation (right-button) drag began at.
    pub drag_start_step: Option<f64>,
    /// Step under the pointer *during* a drag. Unlike `hover_step` this survives the pointer
    /// leaving the roll, so a drag that ends outside still resolves.
    pub drag_step: Option<f64>,
    /// Right-button drag: selecting a region to annotate.
    pub drag_started: bool,
    pub dragging: bool,
    pub drag_stopped: bool,
    /// Left button: scrubbing. A click *or* a drag — dragging the playhead should scrub
    /// continuously, the way a transport does.
    pub scrub_step: Option<f64>,
    /// Right-click with no drag: annotate this spot.
    pub secondary_clicked: bool,
}

/// The roll's horizontal viewport: which step sits at the roll's left edge, and how many steps one
/// point spans.
///
/// Both values ease toward a target rather than being set directly, so zooming and panning glide
/// (a wheel notch is a target change, not a jump). Live playback drives the target every frame from
/// the playhead, which lands on the same values it would have set outright — so the scrolling roll
/// looks exactly as it did before the viewport existed.
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
    /// Step at the left edge → x, given the roll's left edge.
    fn x_of_step(&self, roll_min_x: f32, step: f64) -> f32 {
        roll_min_x + ((step - self.origin_step) / self.steps_per_point) as f32
    }

    /// The inverse of [`Self::x_of_step`] — what step is under a pixel (for scrub / hit-testing).
    fn step_of_x(&self, roll_min_x: f32, x: f32) -> f64 {
        self.origin_step + (x - roll_min_x) as f64 * self.steps_per_point
    }

    /// Snaps current to target (no glide) — used when the viewport should not animate, e.g. the
    /// first frame or a playhead jump.
    fn snap(&mut self) {
        self.origin_step = self.target_origin;
        self.steps_per_point = self.target_spp;
    }

    /// Eases current toward target over `dt` seconds.
    fn ease(&mut self, dt: f64) {
        let a = 1.0 - (-dt.max(0.0) / VIEW_EASE_TAU).exp();
        self.origin_step += (self.target_origin - self.origin_step) * a;
        self.steps_per_point += (self.target_spp - self.steps_per_point) * a;
    }

    /// Zooms by `factor` about a fixed pixel, so the step under the cursor stays under the cursor —
    /// the behaviour that makes wheel-zoom feel like a DAW rather than a slider.
    ///
    /// Anchored on the *target* viewport, not the eased one: successive notches then compose
    /// predictably instead of chasing a moving anchor.
    fn zoom_about(&mut self, roll_min_x: f32, anchor_x: f32, factor: f64) {
        let dx = (anchor_x - roll_min_x) as f64;
        let step_at_anchor = self.target_origin + dx * self.target_spp;
        self.target_spp =
            (self.target_spp * factor).clamp(MIN_STEPS_PER_POINT, MAX_STEPS_PER_POINT);
        self.target_origin = step_at_anchor - dx * self.target_spp;
    }

    /// Pans by a pixel delta (positive = content moves left, i.e. later in the song).
    fn pan_points(&mut self, dx_points: f64) {
        self.target_origin += dx_points * self.target_spp;
    }
}

/// Piano-roll renderer: owns the visible note list and a smoothed playhead clock.
#[derive(Default)]
pub struct PianoRoll {
    notes: Vec<NoteEvent>,
    /// Pre-inferred chord changes for the whole song (in sequencer steps). Empty for
    /// songs with no chord data; the lane is then omitted entirely.
    chords: Vec<LaneSpan>,
    /// Wall-clock-smoothed playhead position, in ticks (drives the scroll).
    display_tick: f64,
    /// Whether anything has been ingested yet (so the first frame snaps to the audio position).
    primed: bool,
    /// The horizontal viewport. Playhead-driven while live; user-driven in [`Self::edit`].
    view: View,
    /// Annotation mode: the viewport is the user's (scroll = zoom, drag = pan) and the roll shows
    /// the whole song rather than the look-ahead window.
    edit: bool,
    /// The musical grid: steps per beat (from the device) and beats per bar (annotated), plus the
    /// offset of bar 1 for pickup bars.
    grid: Grid,
    /// Width of the roll (excluding the keyboard) at the last draw, in points. The viewport is
    /// stored as steps-per-*point*, so anything that needs a step *span* (the overview highlight,
    /// zoom-to-fit) needs the width the last frame actually used.
    last_roll_width: f32,
    /// Highlighted step range in the chord lane: the selected span, or a drag in progress.
    selection: Option<(f64, f64)>,
    /// Step a primary drag started at, carried across frames.
    drag_start_step: Option<f64>,
}

/// The musical bar grid on the roll's step axis.
///
/// Steps are musical time — `steps_per_beat` is a device constant (DS SSEQ 48, GBA MP2K 24) and a
/// tempo change moves the step *rate*, not this — so bars are uniform in steps and need no tempo
/// map.
#[derive(Clone, Copy)]
pub struct Grid {
    pub steps_per_beat: f64,
    pub beats_per_bar: u32,
    /// Step at which bar 1 begins; non-zero for a pickup (anacrusis).
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
    /// Steps per bar, or `None` when the device reports no beat division (nothing to draw).
    pub fn bar_steps(&self) -> Option<f64> {
        let b = self.steps_per_beat * self.beats_per_bar.max(1) as f64;
        (b > 0.0).then_some(b)
    }

    /// The bar number (1-based, as musicians count) and the beat within it (1-based) at `step`.
    /// Steps before the offset are the pickup: bar 0, counting up to the downbeat.
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
    /// Resets the roll for a new song.
    pub fn clear(&mut self) {
        self.notes.clear();
        self.chords.clear();
        self.display_tick = 0.0;
        self.primed = false;
    }

    /// Installs the song's chord timeline. Cleared by [`Self::clear`]; pass an empty iterator to
    /// hide the lane.
    pub fn set_chords(&mut self, spans: impl IntoIterator<Item = LaneSpan>) {
        self.chords = spans.into_iter().collect();
    }

    /// The smoothed playhead position in ticks (used to drive the look-ahead target).
    pub fn display_tick(&self) -> f64 {
        self.display_tick
    }

    /// Moves the playhead outright (a scrub), snapping the smoothing so the roll doesn't glide
    /// after the cursor.
    pub fn set_display_tick(&mut self, tick: f64) {
        self.display_tick = tick;
        self.primed = true;
    }

    /// Turns annotation mode on/off: the viewport becomes the user's, and [`Self::ingest_overview`]
    /// replaces the live look-ahead as the note source.
    pub fn set_edit(&mut self, edit: bool) {
        self.edit = edit;
    }

    /// Installs the musical grid (device steps-per-beat + the annotated meter/offset).
    pub fn set_grid(&mut self, grid: Grid) {
        self.grid = grid;
    }

    /// Highlights a step range in the chord lane (the selected span, or a drag preview).
    pub fn set_selection(&mut self, sel: Option<(f64, f64)>) {
        self.selection = sel;
    }

    pub fn grid(&self) -> Grid {
        self.grid
    }

    /// The `[start, end)` step range currently visible in the roll. Used to highlight the visible
    /// window on the overview bar. Live, this is the fixed span around the playhead; in annotation
    /// mode it follows the user's zoom.
    pub fn visible_range(&self) -> (f64, f64) {
        if self.edit {
            // `steps_per_point` is only meaningful once drawn at a real width; before that the
            // default span keeps the overview highlight sane.
            let span = self.view.steps_per_point * self.last_roll_width.max(1.0) as f64;
            return (self.view.origin_step, self.view.origin_step + span);
        }
        let start = self.display_tick - WINDOW_TICKS * CURSOR_FRAC as f64;
        (start, start + WINDOW_TICKS)
    }

    /// Advances the smoothed playhead clock by `dt` seconds toward the audio position.
    ///
    /// The audio thread advances ticks in coarse bursts (one full output buffer at a time), so we
    /// integrate at the current tempo for smooth motion and gently correct toward the reported
    /// position to stay locked, snapping on large jumps (loop / restart / seek).
    pub fn advance(&mut self, snap: &VisSnapshot, dt: f64, playing: bool) {
        // The viewport eases whether or not the transport is rolling: in annotation mode the user
        // zooms and pans a stopped song, and that still has to glide.
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
            self.display_tick = target; // loop / restart — snap.
        } else {
            self.display_tick += err * 0.10; // gentle phase lock.
        }
    }

    /// Rebuilds the note list from a whole-song overview — annotation's note source.
    ///
    /// The live path ingests only the look-ahead's rolling window ([`Self::ingest`]), which can't
    /// serve an editor: scrubbing backwards would show nothing. `SongOverview` already holds every
    /// note of the song, so annotation simply reads that once per song instead. Draw clips by
    /// x-range, so the cost is one bounded scan per frame.
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
                    // The overview resolves every note's length at build time, so nothing is open.
                    end: n.timestamp as f64 + n.duration as f64,
                    open: false,
                }),
        );
    }

    /// Frames the whole song in the roll (annotation's initial view).
    pub fn zoom_to_fit(&mut self, total_steps: u32) {
        let w = self.last_roll_width.max(1.0) as f64;
        self.view.target_origin = 0.0;
        self.view.target_spp =
            (total_steps.max(1) as f64 / w).clamp(MIN_STEPS_PER_POINT, MAX_STEPS_PER_POINT);
        self.view.snap();
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
    pub fn draw(&mut self, ui: &mut egui::Ui, active: bool) -> RollInput {
        let size = ui.available_size_before_wrap();
        // Annotation needs the wheel and drags; live playback stays a pure display.
        let sense = if self.edit {
            Sense::click_and_drag()
        } else {
            Sense::hover()
        };
        let (rect, resp) = ui.allocate_exact_size(size, sense);
        let painter = ui.painter_at(rect);

        painter.rect_filled(rect, 0.0, Color32::from_rgb(0x0c, 0x0e, 0x14));

        // Reserve a top strip for the chord lane only when the song has chord data,
        // so DS / uncovered songs look exactly as before.
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
            // Live: the viewport is a pure function of the playhead — same span, same cursor
            // fraction, so the scrolling roll is unchanged by the viewport's existence.
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
        // step → x within the roll.
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
        self.draw_keyboard(&painter, content, roll, lane_h, &lit, dim);

        if !self.edit {
            return RollInput::default();
        }
        self.collect_input(ui, &resp, rect, roll)
    }

    /// Reports what the pointer did, in step coordinates, for the app to interpret. Annotation only.
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
        // The right button is the *only* way to annotate — by drag or by click. The left button is
        // purely the transport, so there is never a question of which gesture a press means.
        input.drag_started = resp.drag_started_by(egui::PointerButton::Secondary);
        input.dragging = resp.dragged_by(egui::PointerButton::Secondary);
        input.drag_stopped = resp.drag_stopped_by(egui::PointerButton::Secondary);
        input.secondary_clicked = resp.clicked_by(egui::PointerButton::Secondary);

        // `interact_pointer_pos` keeps reporting while a drag is held, even once the pointer has
        // left the roll — `hover_pos` does not. A drag that ends outside the widget (easy to do:
        // the chord lane is 20 pt tall) must still commit, and must still have somewhere to put the
        // picker, so the drag's own position comes from here rather than from hover.
        let drag_pos = resp.interact_pointer_pos();
        // Left click or left drag scrubs, and keeps scrubbing while held.
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
        // The press position has to be latched: once a drag is under way egui reports the *current*
        // pointer, and the anchor is what a span is drawn from.
        if input.drag_started {
            self.drag_start_step = drag_pos.map(|p| step_at(p.x));
        }
        input.drag_start_step = self.drag_start_step;
        if input.drag_stopped {
            self.drag_start_step = None;
        }
        input
    }

    /// The chord lane across the top: each change's label, left-anchored at its start
    /// and scrolling with the playhead, brightening while it is the chord sounding now
    /// (its span contains the playhead). A faint divider marks each change boundary.
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

            // An annotated span is a solid block; a pre-inferred one is bare text. The contrast is
            // the point — scanning a zoomed-out song has to show at a glance what is already done.
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
                // Divider at the change boundary (pre-inferred lane only).
                painter.line_segment(
                    [Pos2::new(x0, strip.min.y), Pos2::new(x0, strip.max.y)],
                    Stroke::new(
                        1.0_f32,
                        Color32::from_rgba_unmultiplied(0x3a, 0x42, 0x58, 90),
                    ),
                );
            }

            let color = match (c.fill.is_some(), now) {
                // On a filled block, the label rides on top: near-white reads over every hue.
                (true, _) => scale_alpha(Color32::from_rgb(0xf6, 0xf8, 0xff), dim),
                (false, true) => scale_alpha(Color32::from_rgb(0xf2, 0xf4, 0xff), dim),
                (false, false) => scale_alpha(Color32::from_rgb(0x9a, 0xa2, 0xb8), dim),
            };
            // Text is clipped to its own block so a short span can't spill its label over the next.
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

        // The selection goes on top, as a translucent wash. Under the blocks it would be invisible
        // the moment a region already had a label — which is exactly when you most need to see what
        // you have selected (you are about to overwrite it).
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

    /// DAW-style viewport input (annotation only): wheel zooms about the cursor, shift+wheel and
    /// middle/secondary drag pan.
    fn handle_view_input(&mut self, ui: &egui::Ui, resp: &egui::Response, roll: Rect) {
        let hovered = resp.hovered();
        let (scroll, zoom_gesture, shift, pointer) = ui.input(|i| {
            (
                // Raw, not smoothed: we run our own easing, and stacking egui's smoothing on top of
                // it would read as lag.
                i.raw_scroll_delta,
                i.zoom_delta(),
                i.modifiers.shift,
                i.pointer.hover_pos(),
            )
        });
        let anchor = pointer.map(|p| p.x).unwrap_or(roll.center().x);

        if hovered {
            if shift && scroll.y != 0.0 {
                // Shift+wheel = horizontal pan, the usual DAW binding.
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
            // Trackpad pinch / ctrl+wheel arrives as a zoom gesture, not scroll.
            if zoom_gesture != 1.0 {
                self.view
                    .zoom_about(roll.min.x, anchor, 1.0 / zoom_gesture as f64);
            }
        }
        // Middle-drag pans. Left-drag selects a region; the right button belongs to the chord
        // picker, so it deliberately does *not* pan — right-drag panning would turn any right-click
        // with a few pixels of hand jitter into a pan instead of opening the picker. Panning is
        // still reachable by middle-drag and shift+wheel.
        if resp.dragged_by(egui::PointerButton::Middle) {
            let dx = resp.drag_delta().x as f64;
            self.view.pan_points(-dx);
            // A drag should track the hand exactly, so bypass the glide.
            self.view.origin_step = self.view.target_origin;
        }
    }

    /// The vertical time grid: bar lines (strong, numbered) and beat lines (faint).
    ///
    /// Derived from the device's steps-per-beat and the annotated meter, so a line is a real
    /// musical boundary. Falls back to no grid when the device reports no beat division. Beat lines
    /// and numbers are dropped as they get too dense to read, which is what makes zooming out to a
    /// whole song legible.
    fn draw_grid(&self, painter: &egui::Painter, roll: Rect, xt: &impl Fn(f64) -> f32) {
        /// Minimum spacing (points) before a subdivision stops being drawn.
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
        // Thin the bar lines out when zoomed far out, so a long song doesn't turn into a solid wall.
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
                    // Bar 1 is the first full bar; anything before it is the pickup.
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

/// The fill for an annotated chord block: hue by **root pitch class**, so a repeated progression
/// shows up as a repeated colour pattern down the ribbon — the shape of the harmony becomes visible
/// without reading a single label. Minor-ish qualities sit darker than major-ish ones.
///
/// Deliberately muted (unlike [`track_color`]): the lane sits above the note bars, and two vivid
/// palettes stacked would fight. `None` (no-chord) is grey — an annotation, but not a harmony.
pub fn chord_color(root_pc: Option<u8>, darker: bool) -> Color32 {
    let Some(pc) = root_pc else {
        return Color32::from_rgb(0x4a, 0x50, 0x60);
    };
    // Fifths, not semitones: neighbouring keys in a progression then get *distant* hues, which is
    // what makes I–V–vi–IV legible rather than a smear of adjacent shades.
    let fifth = (pc as u32 * 7) % 12;
    let v = if darker { 0.52 } else { 0.72 };
    hsv(fifth as f32 / 12.0, 0.55, v)
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

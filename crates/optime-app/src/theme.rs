//! The app-wide visual theme: an iOS-dark palette applied to egui's style, plus small
//! Apple-style widget helpers (list rows, section headers, circular icon buttons) shared by
//! the mobile and desktop layouts.

use egui::{Color32, FontId, Pos2, Rect, Sense, Stroke, TextStyle, Vec2};

/// Near-black app background.
pub const BG: Color32 = Color32::from_rgb(0x0b, 0x0b, 0x0f);
/// Card / raised-surface fill.
pub const CARD: Color32 = Color32::from_rgb(0x1a, 0x1a, 0x20);
/// Hovered / pressed surface fill.
pub const CARD_HI: Color32 = Color32::from_rgb(0x26, 0x26, 0x2e);
/// Hairline separators.
pub const HAIRLINE: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x32);
/// The accent (Apple-Music-style red-pink).
pub const ACCENT: Color32 = Color32::from_rgb(0xfc, 0x46, 0x64);
/// Primary text.
pub const TEXT: Color32 = Color32::from_rgb(0xee, 0xee, 0xf2);
/// Secondary / caption text.
pub const TEXT_DIM: Color32 = Color32::from_rgb(0x9a, 0x9a, 0xa5);

/// Applies the theme to the egui context (call once at startup).
pub fn apply(ctx: &egui::Context) {
    // Pin the app to dark mode: egui 0.29 keeps separate dark/light styles and follows the
    // OS theme by default, which would swap in the stock light style on light-mode systems.
    ctx.set_theme(egui::ThemePreference::Dark);
    let mut style = (*ctx.style()).clone();

    style.text_styles = [
        (TextStyle::Heading, FontId::proportional(24.0)),
        (TextStyle::Body, FontId::proportional(14.5)),
        (TextStyle::Button, FontId::proportional(14.5)),
        (TextStyle::Monospace, FontId::monospace(12.0)),
        (TextStyle::Small, FontId::proportional(11.5)),
    ]
    .into();

    style.spacing.item_spacing = egui::vec2(8.0, 8.0);
    style.spacing.button_padding = egui::vec2(12.0, 7.0);
    style.spacing.interact_size = egui::vec2(40.0, 30.0);

    let v = &mut style.visuals;
    *v = egui::Visuals::dark();
    v.panel_fill = BG;
    v.window_fill = CARD;
    v.extreme_bg_color = Color32::from_rgb(0x13, 0x13, 0x18);
    v.faint_bg_color = CARD;
    v.window_rounding = 14.0.into();
    v.menu_rounding = 12.0.into();
    v.window_stroke = Stroke::new(1.0_f32, HAIRLINE);
    v.selection.bg_fill = ACCENT.linear_multiply(0.35);
    v.selection.stroke = Stroke::new(1.0_f32, ACCENT);
    v.hyperlink_color = ACCENT;
    v.slider_trailing_fill = true;

    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, HAIRLINE);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, TEXT);
    v.widgets.inactive.bg_fill = CARD;
    v.widgets.inactive.weak_bg_fill = CARD;
    v.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, TEXT);
    v.widgets.hovered.bg_fill = CARD_HI;
    v.widgets.hovered.weak_bg_fill = CARD_HI;
    v.widgets.hovered.fg_stroke = Stroke::new(1.5_f32, TEXT);
    v.widgets.active.bg_fill = Color32::from_rgb(0x30, 0x30, 0x3a);
    v.widgets.active.weak_bg_fill = Color32::from_rgb(0x30, 0x30, 0x3a);
    v.widgets.active.fg_stroke = Stroke::new(1.5_f32, Color32::WHITE);
    v.widgets.open.bg_fill = CARD_HI;
    v.widgets.open.weak_bg_fill = CARD_HI;
    for w in [
        &mut v.widgets.noninteractive,
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        w.rounding = 10.0.into();
    }

    ctx.set_style(style);
}

/// A small gray uppercase section header, iOS-grouped-list style.
pub fn section_header(ui: &mut egui::Ui, text: &str) {
    ui.add_space(10.0);
    ui.label(
        egui::RichText::new(text.to_uppercase())
            .size(11.5)
            .color(TEXT_DIM),
    );
    ui.add_space(2.0);
}

/// One iOS-style list row: full-width tappable area, optional leading icon, title, optional
/// chevron, hairline separator underneath. `width` lets callers reserve space for trailing
/// widgets (pass `ui.available_width()` otherwise).
pub fn ios_row(
    ui: &mut egui::Ui,
    width: f32,
    icon: Option<&str>,
    title: &str,
    selected: bool,
    chevron: bool,
) -> egui::Response {
    let h = 42.0;
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(width, h), Sense::click());
    let painter = ui.painter_at(rect);

    if resp.is_pointer_button_down_on() || resp.hovered() {
        painter.rect_filled(rect, 10.0, CARD_HI);
    }

    let mut x = rect.left() + 14.0;
    if let Some(icon) = icon {
        painter.text(
            Pos2::new(x, rect.center().y),
            egui::Align2::LEFT_CENTER,
            icon,
            FontId::proportional(16.0),
            if selected { ACCENT } else { TEXT },
        );
        x += 28.0;
    }
    // Clip the title so it never runs under the chevron / trailing widgets.
    let title_right = rect.right() - if chevron { 26.0 } else { 10.0 };
    let title_painter = ui.painter_at(Rect::from_min_max(
        Pos2::new(x, rect.top()),
        Pos2::new(title_right, rect.bottom()),
    ));
    title_painter.text(
        Pos2::new(x, rect.center().y),
        egui::Align2::LEFT_CENTER,
        title,
        FontId::proportional(15.0),
        if selected { ACCENT } else { TEXT },
    );
    if chevron {
        painter.text(
            Pos2::new(rect.right() - 14.0, rect.center().y),
            egui::Align2::CENTER_CENTER,
            "›",
            FontId::proportional(18.0),
            TEXT_DIM,
        );
    }
    painter.line_segment(
        [
            Pos2::new(rect.left() + 14.0, rect.bottom()),
            Pos2::new(rect.right(), rect.bottom()),
        ],
        Stroke::new(0.5_f32, HAIRLINE),
    );
    resp
}

/// A circular icon button (Apple-transport style). `filled` paints a solid accent disc (the
/// big play button); otherwise the glyph floats with a soft circle on hover/press. `active`
/// tints the glyph with the accent (shuffle/repeat/heart states).
pub fn icon_button(
    ui: &mut egui::Ui,
    icon: &str,
    diameter: f32,
    glyph_size: f32,
    filled: bool,
    active: bool,
    enabled: bool,
) -> egui::Response {
    let sense = if enabled {
        Sense::click()
    } else {
        Sense::hover()
    };
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(diameter), sense);
    let painter = ui.painter_at(rect);
    let center = rect.center();
    let r = diameter / 2.0;

    if filled {
        let fill = if enabled { ACCENT } else { CARD_HI };
        painter.circle_filled(center, r, fill);
        if resp.is_pointer_button_down_on() && enabled {
            painter.circle_filled(center, r, Color32::from_black_alpha(60));
        }
        painter.text(
            center,
            egui::Align2::CENTER_CENTER,
            icon,
            FontId::proportional(glyph_size),
            Color32::WHITE,
        );
    } else {
        if (resp.hovered() || resp.is_pointer_button_down_on()) && enabled {
            painter.circle_filled(center, r, CARD_HI);
        }
        let color = if !enabled {
            TEXT_DIM.linear_multiply(0.5)
        } else if active {
            ACCENT
        } else {
            TEXT
        };
        painter.text(
            center,
            egui::Align2::CENTER_CENTER,
            icon,
            FontId::proportional(glyph_size),
            color,
        );
    }
    resp
}

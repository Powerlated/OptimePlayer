use egui::{Color32, FontId, Pos2, Rect, Sense, Stroke, TextStyle, Vec2};

pub const BG: Color32 = Color32::from_rgb(0x0b, 0x0b, 0x0f);
pub const CARD: Color32 = Color32::from_rgb(0x1a, 0x1a, 0x20);
pub const CARD_HI: Color32 = Color32::from_rgb(0x26, 0x26, 0x2e);
pub const HAIRLINE: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x32);
pub const ACCENT: Color32 = Color32::from_rgb(0xfc, 0x46, 0x64);
pub const TEXT: Color32 = Color32::from_rgb(0xee, 0xee, 0xf2);
pub const TEXT_DIM: Color32 = Color32::from_rgb(0x9a, 0x9a, 0xa5);

fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    fonts.font_data.insert(
        "inter".to_owned(),
        egui::FontData::from_static(include_bytes!("../assets/Inter-Regular.ttf")),
    );
    let prop = fonts
        .families
        .get_mut(&egui::FontFamily::Proportional)
        .expect("proportional family exists");
    prop.insert(0, "inter".to_owned());

    #[cfg(not(target_arch = "wasm32"))]
    {
        for candidate in [
            "C:\\Windows\\Fonts\\SF-Pro.ttf",
            "C:\\Windows\\Fonts\\SF-Pro-Display-Regular.otf",
            "C:\\Windows\\Fonts\\SF-Pro-Text-Regular.otf",
            "/System/Library/Fonts/SFNS.ttf",
            "/Library/Fonts/SF-Pro.ttf",
        ] {
            if let Ok(bytes) = std::fs::read(candidate) {
                fonts
                    .font_data
                    .insert("sf-pro".to_owned(), egui::FontData::from_owned(bytes));
                fonts
                    .families
                    .get_mut(&egui::FontFamily::Proportional)
                    .expect("proportional family exists")
                    .insert(0, "sf-pro".to_owned());
                break;
            }
        }
    }

    ctx.set_fonts(fonts);
}

pub fn apply(ctx: &egui::Context) {
    install_fonts(ctx);
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

pub fn section_header(ui: &mut egui::Ui, text: &str) {
    ui.add_space(10.0);
    ui.label(
        egui::RichText::new(text.to_uppercase())
            .size(11.5)
            .color(TEXT_DIM),
    );
    ui.add_space(2.0);
}

pub fn ios_row(
    ui: &mut egui::Ui,
    width: f32,
    icon: Option<&str>,
    title: &str,
    badges: &[(&str, Color32)],
    selected: bool,
    chevron: bool,
) -> egui::Response {
    ios_row_ext(ui, width, icon, title, None, badges, selected, chevron)
}

#[allow(clippy::too_many_arguments)]
pub fn ios_row_ext(
    ui: &mut egui::Ui,
    width: f32,
    icon: Option<&str>,
    title: &str,
    trailing: Option<&str>,
    badges: &[(&str, Color32)],
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
    let mut badge_x = rect.right() - if chevron { 28.0 } else { 12.0 };
    if let Some(trailing) = trailing {
        let galley = painter.text(
            Pos2::new(badge_x, rect.center().y),
            egui::Align2::RIGHT_CENTER,
            trailing,
            FontId::proportional(13.0),
            TEXT_DIM,
        );
        badge_x -= galley.width() + 8.0;
    }
    for (glyph, tint) in badges {
        painter.text(
            Pos2::new(badge_x, rect.center().y),
            egui::Align2::RIGHT_CENTER,
            *glyph,
            FontId::proportional(13.0),
            *tint,
        );
        badge_x -= 20.0;
    }
    let title_right = badge_x - 6.0;
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

pub fn mask_rounded_corners(painter: &egui::Painter, rect: Rect, radius: f32, color: Color32) {
    use std::f32::consts::FRAC_PI_2;
    let r = radius.min(rect.width() * 0.5).min(rect.height() * 0.5);
    if r <= 0.0 {
        return;
    }
    let corners = [
        (
            rect.left_top(),
            Pos2::new(rect.left() + r, rect.top() + r),
            std::f32::consts::PI,
        ),
        (
            rect.right_top(),
            Pos2::new(rect.right() - r, rect.top() + r),
            -FRAC_PI_2,
        ),
        (
            rect.right_bottom(),
            Pos2::new(rect.right() - r, rect.bottom() - r),
            0.0,
        ),
        (
            rect.left_bottom(),
            Pos2::new(rect.left() + r, rect.bottom() - r),
            FRAC_PI_2,
        ),
    ];
    const SEGS: u32 = 16;
    let mut mesh = egui::Mesh::default();
    for (outer, center, start) in corners {
        let base = mesh.vertices.len() as u32;
        mesh.colored_vertex(outer, color);
        for i in 0..=SEGS {
            let a = start + FRAC_PI_2 * (i as f32 / SEGS as f32);
            mesh.colored_vertex(
                Pos2::new(center.x + r * a.cos(), center.y + r * a.sin()),
                color,
            );
        }
        for i in 0..SEGS {
            mesh.add_triangle(base, base + 1 + i, base + 2 + i);
        }
    }
    painter.add(egui::Shape::mesh(mesh));
}

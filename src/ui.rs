// src/ui.rs
//
// The egui side panel and the live editor state it drives. Per design principle
// #1 ("speak plain language, not PBR") the controls use painter words — Color,
// Size, Opacity, Hardness — not material vocabulary.

use egui::Context;

use crate::paint::Brush;

/// All live editor state the UI mutates. The renderer reads `brush` when painting.
#[derive(Default)]
pub struct UiState {
    pub brush: Brush,
}

/// Apply a chunky, dark theme that suits the retro vibe. Called once on startup.
pub fn install_style(ctx: &Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.slider_width = 160.0;
    style.spacing.item_spacing = egui::vec2(8.0, 8.0);
    ctx.set_style(style);
    ctx.set_visuals(egui::Visuals::dark());
}

/// Build the controls panel for this frame. Mutates `state` in place.
pub fn build(ctx: &Context, state: &mut UiState) {
    egui::SidePanel::right("controls")
        .resizable(false)
        .default_width(220.0)
        .show(ctx, |ui| {
            ui.add_space(4.0);
            ui.heading("lowtex");
            ui.label(egui::RichText::new("PSX texture painter").weak());
            ui.separator();

            ui.label("Color");
            ui.color_edit_button_rgb(&mut state.brush.color);
            ui.add_space(6.0);

            ui.add(egui::Slider::new(&mut state.brush.radius, 1.0..=32.0).text("Size"));
            ui.add(egui::Slider::new(&mut state.brush.opacity, 0.0..=1.0).text("Opacity"));
            ui.add(egui::Slider::new(&mut state.brush.hardness, 0.0..=1.0).text("Hardness"));

            ui.add_space(10.0);
            ui.separator();
            ui.label(
                egui::RichText::new("LMB paint · RMB orbit · MMB pan · wheel zoom")
                    .weak()
                    .small(),
            );
        });
}

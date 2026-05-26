// src/ui.rs
//
// The egui side panel and the live editor state it drives. Per design principle
// #1 ("speak plain language, not PBR") the controls use painter words — Color,
// Size, Opacity, Hardness — not material vocabulary.

use egui::Context;

use crate::paint::Brush;
use crate::renderer::PsxSettings;

/// One-shot requests the UI raises this frame, drained by the App after the egui
/// run (file dialogs and texture ops happen outside the egui closure).
#[derive(Default)]
pub struct UiActions {
    pub save_png: bool,
    pub open_texture: bool,
    pub set_resolution: Option<u32>,
}

/// All live editor state the UI mutates. The renderer reads `brush` when painting.
pub struct UiState {
    pub brush: Brush,
    pub psx: PsxSettings,
    /// Mirror of the renderer's current texture resolution, shown in the picker.
    pub resolution: u32,
    pub actions: UiActions,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            brush: Brush::default(),
            psx: PsxSettings::default(),
            resolution: 128,
            actions: UiActions::default(),
        }
    }
}

/// Apply a chunky, dark theme that suits the retro vibe. Called once on startup.
pub fn install_style(ctx: &Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.slider_width = 160.0;
    style.spacing.item_spacing = egui::vec2(8.0, 8.0);
    ctx.set_style(style);
    ctx.set_visuals(egui::Visuals::dark());
}

/// Build the controls panel for this frame. Mutates `state` in place and records
/// one-shot requests in `state.actions` (reset each call).
pub fn build(ctx: &Context, state: &mut UiState) {
    state.actions = UiActions::default();

    egui::SidePanel::right("controls")
        .resizable(false)
        .default_width(220.0)
        .show(ctx, |ui| {
            ui.add_space(4.0);
            ui.heading("lowtex");
            ui.label(egui::RichText::new("PSX texture painter").weak());
            ui.separator();

            ui.label("Brush");
            ui.color_edit_button_rgb(&mut state.brush.color);
            ui.add_space(6.0);
            ui.add(egui::Slider::new(&mut state.brush.radius, 1.0..=32.0).text("Size"));
            ui.add(egui::Slider::new(&mut state.brush.opacity, 0.0..=1.0).text("Opacity"));
            ui.add(egui::Slider::new(&mut state.brush.hardness, 0.0..=1.0).text("Hardness"));

            ui.add_space(10.0);
            ui.separator();
            ui.label("Texture");

            // Resolution picker. A change requests a resample at that size.
            let mut res = state.resolution;
            egui::ComboBox::from_label("Resolution")
                .selected_text(format!("{res}×{res}"))
                .show_ui(ui, |ui| {
                    for size in [64u32, 128, 256] {
                        ui.selectable_value(&mut res, size, format!("{size}×{size}"));
                    }
                });
            if res != state.resolution {
                state.resolution = res;
                state.actions.set_resolution = Some(res);
            }

            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui.button("Open…").clicked() {
                    state.actions.open_texture = true;
                }
                if ui.button("Save PNG…").clicked() {
                    state.actions.save_png = true;
                }
            });

            ui.add_space(10.0);
            ui.separator();
            ui.checkbox(&mut state.psx.enabled, "PSX mode");
            ui.add_enabled_ui(state.psx.enabled, |ui| {
                ui.checkbox(&mut state.psx.affine, "Texture warp");
                ui.checkbox(&mut state.psx.snap, "Vertex wobble");
                ui.add(egui::Slider::new(&mut state.psx.grid, 8.0..=256.0).text("Wobble grid"));
                ui.checkbox(&mut state.psx.flat, "Flat shading");
                ui.checkbox(&mut state.psx.fog, "Fog");
                ui.add_enabled_ui(state.psx.fog, |ui| {
                    ui.color_edit_button_rgb(&mut state.psx.fog_color);
                    ui.add(
                        egui::Slider::new(&mut state.psx.fog_start, 0.0..=10.0).text("Fog near"),
                    );
                    ui.add(egui::Slider::new(&mut state.psx.fog_end, 0.0..=20.0).text("Fog far"));
                });
            });

            ui.add_space(10.0);
            ui.separator();
            ui.label(
                egui::RichText::new("LMB paint · RMB orbit · MMB pan · wheel zoom")
                    .weak()
                    .small(),
            );
        });
}

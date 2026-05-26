// src/ui.rs
//
// The egui side panel and the live editor state it drives. Per design principle
// #1 ("speak plain language, not PBR") the controls use painter words — Color,
// Size, Opacity, Hardness — not material vocabulary.

use egui::Context;

use crate::paint::Brush;
use crate::renderer::PaletteSettings;

/// One-shot requests the UI raises this frame, drained by the App after the egui
/// run (file dialogs and texture ops happen outside the egui closure).
#[derive(Default)]
pub struct UiActions {
    pub save_png: bool,
    pub open_texture: bool,
    pub open_model: bool,
    pub set_resolution: Option<u32>,
    /// Index into `Palette::builtins()` to make active.
    pub select_builtin_palette: Option<usize>,
    /// Generate a palette from a chosen image with this many colors.
    pub generate_palette: Option<usize>,
}

/// All live editor state the UI mutates. The renderer reads `brush` when painting.
pub struct UiState {
    pub brush: Brush,
    pub palette: PaletteSettings,
    /// Colors of the active palette, synced from the renderer for the swatch row.
    pub palette_swatches: Vec<[f32; 3]>,
    /// Color count requested when generating a palette from an image.
    pub palette_size: u32,
    /// Mirror of the renderer's current texture resolution, shown in the picker.
    pub resolution: u32,
    pub actions: UiActions,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            brush: Brush::default(),
            palette: PaletteSettings::default(),
            palette_swatches: Vec::new(),
            palette_size: 16,
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
            ui.label(egui::RichText::new("low-poly texture painter").weak());
            ui.separator();

            if ui.button("Open model…").clicked() {
                state.actions.open_model = true;
            }
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
            ui.label("Palette");
            ui.checkbox(&mut state.palette.enabled, "Quantize");
            ui.add_enabled_ui(state.palette.enabled, |ui| {
                ui.checkbox(&mut state.palette.dither, "Dither");
                ui.add(
                    egui::Slider::new(&mut state.palette.dither_strength, 0.0..=0.3)
                        .text("Dither amt"),
                );
            });

            // Swatch row of the active palette.
            swatches(ui, &state.palette_swatches);

            ui.horizontal_wrapped(|ui| {
                for (i, p) in crate::palette::Palette::builtins().iter().enumerate() {
                    if ui.button(&p.name).clicked() {
                        state.actions.select_builtin_palette = Some(i);
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.add(egui::Slider::new(&mut state.palette_size, 2..=64).text("colors"));
                if ui.button("From image…").clicked() {
                    state.actions.generate_palette = Some(state.palette_size as usize);
                }
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

/// Draw a wrapping row of small color swatches.
fn swatches(ui: &mut egui::Ui, colors: &[[f32; 3]]) {
    let size = egui::vec2(16.0, 16.0);
    ui.horizontal_wrapped(|ui| {
        for c in colors {
            let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
            let color = egui::Color32::from_rgb(
                (c[0] * 255.0) as u8,
                (c[1] * 255.0) as u8,
                (c[2] * 255.0) as u8,
            );
            ui.painter().rect_filled(rect, 2.0, color);
        }
    });
}

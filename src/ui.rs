// src/ui.rs
//
// The egui side panel and the live editor state it drives. Per design principle
// #1 ("speak plain language, not PBR") the controls use painter words — Color,
// Size, Opacity, Hardness — not material vocabulary.

use egui::Context;

use crate::layers::BlendMode;
use crate::paint::Brush;
use crate::renderer::PaletteSettings;

/// A read-only snapshot of one layer, synced from the renderer for the UI list.
#[derive(Clone)]
pub struct LayerInfo {
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub blend: BlendMode,
}

/// One-shot requests the UI raises this frame, drained by the App after the egui
/// run (file dialogs and texture ops happen outside the egui closure).
#[derive(Default)]
pub struct UiActions {
    pub save_png: bool,
    pub open_texture: bool,
    pub open_model: bool,
    // Undo/redo (history over the layer stack).
    pub undo: bool,
    pub redo: bool,
    /// Snapshot the layer stack before a continuous edit (an opacity-slider drag),
    /// so the whole drag collapses to one undo step.
    pub checkpoint: bool,
    pub set_resolution: Option<u32>,
    /// Index into `Palette::builtins()` to make active.
    pub select_builtin_palette: Option<usize>,
    /// Generate a palette from a chosen image with this many colors.
    pub generate_palette: Option<usize>,
    // Layer ops (G10).
    pub add_layer: bool,
    pub remove_layer: bool,
    pub move_layer_up: bool,
    pub move_layer_down: bool,
    pub select_layer: Option<usize>,
    pub set_layer_visible: Option<(usize, bool)>,
    pub set_layer_opacity: Option<(usize, f32)>,
    pub set_layer_blend: Option<(usize, BlendMode)>,
    // AO suite: add a baked AO / edge-highlight layer at the given strength.
    pub apply_ao: Option<f32>,
    pub apply_highlight: Option<f32>,
}

/// All live editor state the UI mutates. The renderer reads `brush` when painting.
pub struct UiState {
    pub brush: Brush,
    pub palette: PaletteSettings,
    /// Colors of the active palette, synced from the renderer for the swatch row.
    pub palette_swatches: Vec<[f32; 3]>,
    /// Color count requested when generating a palette from an image.
    pub palette_size: u32,
    /// Layer stack snapshot (bottom-first), synced from the renderer.
    pub layers: Vec<LayerInfo>,
    pub active_layer: usize,
    /// Strength for the AO-suite bakes.
    pub ao_strength: f32,
    /// Mirror of the renderer's current texture resolution, shown in the picker.
    pub resolution: u32,
    /// Mirrors of the history state, to enable/disable the Undo/Redo buttons.
    pub can_undo: bool,
    pub can_redo: bool,
    pub actions: UiActions,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            brush: Brush::default(),
            palette: PaletteSettings::default(),
            palette_swatches: Vec::new(),
            palette_size: 16,
            layers: Vec::new(),
            active_layer: 0,
            ao_strength: 0.75,
            resolution: 128,
            can_undo: false,
            can_redo: false,
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
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(state.can_undo, egui::Button::new("↶ Undo"))
                    .on_hover_text("Ctrl/⌘+Z")
                    .clicked()
                {
                    state.actions.undo = true;
                }
                if ui
                    .add_enabled(state.can_redo, egui::Button::new("↷ Redo"))
                    .on_hover_text("Ctrl/⌘+Shift+Z")
                    .clicked()
                {
                    state.actions.redo = true;
                }
            });
            ui.separator();

            ui.label("Brush");
            ui.color_edit_button_rgb(&mut state.brush.color);
            ui.add_space(6.0);
            ui.add(egui::Slider::new(&mut state.brush.radius, 1.0..=32.0).text("Size"));
            ui.add(egui::Slider::new(&mut state.brush.opacity, 0.0..=1.0).text("Opacity"));
            ui.add(egui::Slider::new(&mut state.brush.hardness, 0.0..=1.0).text("Hardness"));

            ui.add_space(10.0);
            ui.separator();
            layers_section(ui, state);

            ui.add_space(10.0);
            ui.separator();
            ui.label("Ambient occlusion");
            ui.label(
                egui::RichText::new("Bake shadow/highlights from the mesh into a layer")
                    .weak()
                    .small(),
            );
            ui.add(egui::Slider::new(&mut state.ao_strength, 0.0..=1.0).text("Strength"));
            ui.horizontal(|ui| {
                if ui
                    .button("Darken (AO)")
                    .on_hover_text("Shadow in crevices")
                    .clicked()
                {
                    state.actions.apply_ao = Some(state.ao_strength);
                }
                if ui
                    .button("Highlights")
                    .on_hover_text("Brighten exposed edges")
                    .clicked()
                {
                    state.actions.apply_highlight = Some(state.ao_strength);
                }
            });

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
            ui.label(
                egui::RichText::new("Undo Ctrl/⌘+Z · Redo Ctrl/⌘+Shift+Z")
                    .weak()
                    .small(),
            );
        });
}

/// The layer stack panel: add/remove/reorder, per-layer visibility, the active
/// layer's blend mode + opacity. Emits actions on interaction (no per-frame push).
fn layers_section(ui: &mut egui::Ui, state: &mut UiState) {
    ui.horizontal(|ui| {
        ui.label("Layers");
        if ui.small_button("+").on_hover_text("Add layer").clicked() {
            state.actions.add_layer = true;
        }
        if ui.small_button("−").on_hover_text("Delete layer").clicked() {
            state.actions.remove_layer = true;
        }
        if ui.small_button("↑").on_hover_text("Move up").clicked() {
            state.actions.move_layer_up = true;
        }
        if ui.small_button("↓").on_hover_text("Move down").clicked() {
            state.actions.move_layer_down = true;
        }
    });

    // Top layer first in the list (reverse of the bottom-up storage order).
    let count = state.layers.len();
    for ui_idx in (0..count).rev() {
        let layer = state.layers[ui_idx].clone();
        ui.horizontal(|ui| {
            let mut vis = layer.visible;
            if ui.checkbox(&mut vis, "").changed() {
                state.actions.set_layer_visible = Some((ui_idx, vis));
            }
            let selected = ui_idx == state.active_layer;
            if ui.selectable_label(selected, &layer.name).clicked() {
                state.actions.select_layer = Some(ui_idx);
            }
        });
    }

    // Active layer's blend + opacity.
    if let Some(active) = state.layers.get(state.active_layer).cloned() {
        let i = state.active_layer;
        egui::ComboBox::from_label("Blend")
            .selected_text(active.blend.name())
            .show_ui(ui, |ui| {
                for mode in BlendMode::ALL {
                    let mut sel = active.blend;
                    if ui.selectable_value(&mut sel, mode, mode.name()).clicked() {
                        state.actions.set_layer_blend = Some((i, mode));
                    }
                }
            });
        let mut op = active.opacity;
        let resp = ui.add(egui::Slider::new(&mut op, 0.0..=1.0).text("Layer opacity"));
        // Snapshot once when the drag begins so the whole adjustment is one undo
        // step, not one per frame.
        if resp.drag_started() {
            state.actions.checkpoint = true;
        }
        if resp.changed() {
            state.actions.set_layer_opacity = Some((i, op));
        }
    }
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

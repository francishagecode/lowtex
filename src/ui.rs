// src/ui.rs
//
// The egui side panel and the live editor state it drives. Per design principle
// #1 ("speak plain language, not PBR") the controls use painter words — Color,
// Size, Opacity, Hardness — not material vocabulary.

use egui::Context;

use crate::bake::{Levels, MapSource};
use crate::export::ExportPreset;
use crate::layers::BlendMode;
use crate::paint::Brush;
use crate::renderer::PaletteSettings;

/// The active editing tool. The brush drags a stroke; the fills are one-shot
/// "paint bucket" clicks (solid color, ignoring what's already there), scoped from
/// smallest to largest region: a flat face, a UV island, the whole object.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    /// Freehand brush — drag to paint a stroke.
    Brush,
    /// Fill the flat face (coplanar facet) of the model under the cursor.
    FillFace,
    /// Fill the UV island under the cursor.
    FillIsland,
    /// Fill the whole object (every texel its UVs cover).
    FillObject,
}

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
    // Mesh effects (AO + curvature → layers and masks), all sharing the Levels
    // remap from the controls. Presets fix the source/color/blend; the two generic
    // actions route the chosen `effect_source` into a brush-colored tint layer or
    // into the active layer's reveal mask.
    pub apply_ao: Option<Levels>,
    pub apply_highlight: Option<Levels>,
    pub apply_dirt: Option<Levels>,
    pub apply_edge_wear: Option<Levels>,
    pub apply_tint: Option<(MapSource, Levels, [f32; 3])>,
    pub mask_from_map: Option<(MapSource, Levels)>,
    /// Export the texture (G23): `true` = true indexed PNG, `false` = RGBA8.
    pub export_png: Option<bool>,
    /// Fill the active layer with a chosen material image, tiled this many times.
    pub fill_material: Option<f32>,
    /// Re-unwrap the mesh's UVs (G14–G17).
    pub unwrap: Option<crate::unwrap::UnwrapMode>,
}

/// All live editor state the UI mutates. The renderer reads `brush` when painting.
pub struct UiState {
    pub brush: Brush,
    /// Active tool — what a left-click on the mesh does.
    pub tool: Tool,
    pub palette: PaletteSettings,
    /// Colors of the active palette, synced from the renderer for the swatch row.
    pub palette_swatches: Vec<[f32; 3]>,
    /// Color count requested when generating a palette from an image.
    pub palette_size: u32,
    /// Layer stack snapshot (bottom-first), synced from the renderer.
    pub layers: Vec<LayerInfo>,
    pub active_layer: usize,
    /// Mask painting (G11): paint the active layer's reveal mask instead of its
    /// color, and (when masking) whether the brush reveals (white) or hides (black).
    pub paint_mask: bool,
    pub mask_reveal: bool,
    /// Target engine for export naming/hints (G23).
    pub export_preset: ExportPreset,
    /// How many times a material image tiles across UV when filling a layer.
    pub material_tile: f32,
    /// Levels controls shared by every mesh effect: overall strength, mid-tone
    /// contrast, and invert. `effect_source` is which baked channel the generic
    /// "Tint layer" / "Mask layer" actions read from.
    pub ao_strength: f32,
    pub effect_contrast: f32,
    pub effect_invert: bool,
    pub effect_source: MapSource,
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
            tool: Tool::Brush,
            palette: PaletteSettings::default(),
            palette_swatches: Vec::new(),
            palette_size: 16,
            layers: Vec::new(),
            active_layer: 0,
            paint_mask: false,
            mask_reveal: false,
            export_preset: ExportPreset::Plain,
            material_tile: 4.0,
            ao_strength: 0.75,
            effect_contrast: 0.0,
            effect_invert: false,
            effect_source: MapSource::Cavities,
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
            ui.horizontal(|ui| {
                ui.label("Unwrap:")
                    .on_hover_text("Reassign the model's UVs");
                for mode in crate::unwrap::UnwrapMode::ALL {
                    if ui.small_button(mode.name()).clicked() {
                        state.actions.unwrap = Some(mode);
                    }
                }
            });
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

            // Tool picker. The fills are one-shot bucket clicks; size/hardness only
            // affect the brush, so they're disabled while a fill tool is active.
            ui.label("Tool");
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(&mut state.tool, Tool::Brush, "🖌 Brush");
                ui.selectable_value(&mut state.tool, Tool::FillFace, "🪣 Face")
                    .on_hover_text("Fill the flat face you click");
                ui.selectable_value(&mut state.tool, Tool::FillIsland, "🪣 Island")
                    .on_hover_text("Fill the UV island you click");
                ui.selectable_value(&mut state.tool, Tool::FillObject, "🪣 All")
                    .on_hover_text("Fill the whole object");
            });
            ui.add_space(6.0);

            ui.label("Color");
            ui.color_edit_button_rgb(&mut state.brush.color);
            ui.add_space(6.0);
            let brush_active = state.tool == Tool::Brush;
            ui.add_enabled_ui(brush_active, |ui| {
                ui.add(egui::Slider::new(&mut state.brush.radius, 1.0..=32.0).text("Size"));
                ui.add(egui::Slider::new(&mut state.brush.opacity, 0.0..=1.0).text("Opacity"));
                ui.add(egui::Slider::new(&mut state.brush.hardness, 0.0..=1.0).text("Hardness"));
            });

            ui.add_space(10.0);
            ui.separator();
            layers_section(ui, state);

            ui.add_space(10.0);
            ui.separator();
            mesh_effects_section(ui, state);

            ui.add_space(10.0);
            ui.separator();
            ui.label("Material");
            ui.label(
                egui::RichText::new("Fill the active layer with an image (brick, moss…). Mask it for crevice/edge detail.")
                    .weak()
                    .small(),
            );
            ui.add(egui::Slider::new(&mut state.material_tile, 1.0..=16.0).text("Tile"));
            if ui.button("Fill with image…").clicked() {
                state.actions.fill_material = Some(state.material_tile);
            }

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
            egui::ComboBox::from_label("Engine")
                .selected_text(state.export_preset.name())
                .show_ui(ui, |ui| {
                    for p in ExportPreset::ALL {
                        ui.selectable_value(&mut state.export_preset, p, p.name());
                    }
                });
            ui.label(
                egui::RichText::new(state.export_preset.import_hint())
                    .weak()
                    .small(),
            );
            ui.horizontal(|ui| {
                if ui
                    .button("Export indexed…")
                    .on_hover_text("True paletted PNG for retro pipelines (needs Quantize on)")
                    .clicked()
                {
                    state.actions.export_png = Some(true);
                }
                if ui.button("Export RGBA…").clicked() {
                    state.actions.export_png = Some(false);
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

    // Mask painting (G11): paint into the active layer's reveal mask.
    ui.add_space(4.0);
    ui.checkbox(&mut state.paint_mask, "Paint mask");
    ui.add_enabled_ui(state.paint_mask, |ui| {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut state.mask_reveal, false, "Hide");
            ui.selectable_value(&mut state.mask_reveal, true, "Reveal");
        });
    });
}

/// Mesh effects: the baked AO + curvature maps driving layers and masks, all
/// through one shared Levels remap (Strength / Contrast / Invert). Presets are
/// one-click; the Source combo plus Tint/Mask buttons are the generic route.
fn mesh_effects_section(ui: &mut egui::Ui, state: &mut UiState) {
    ui.label("Mesh effects");
    ui.label(
        egui::RichText::new("Drive layers & masks from baked AO and edges")
            .weak()
            .small(),
    );

    // Shared Levels controls.
    ui.add(egui::Slider::new(&mut state.ao_strength, 0.0..=1.0).text("Strength"));
    ui.add(egui::Slider::new(&mut state.effect_contrast, 0.0..=1.0).text("Contrast"));
    ui.checkbox(&mut state.effect_invert, "Invert");
    let levels = Levels {
        invert: state.effect_invert,
        contrast: state.effect_contrast,
        strength: state.ao_strength,
    };

    // Presets — fixed source + color + blend, honoring the Levels above.
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        if ui
            .button("Darken (AO)")
            .on_hover_text("Shadow in crevices")
            .clicked()
        {
            state.actions.apply_ao = Some(levels);
        }
        if ui
            .button("Highlights")
            .on_hover_text("Brighten convex edges")
            .clicked()
        {
            state.actions.apply_highlight = Some(levels);
        }
    });
    ui.horizontal(|ui| {
        if ui
            .button("Dirt")
            .on_hover_text("Dark grime settling into cavities")
            .clicked()
        {
            state.actions.apply_dirt = Some(levels);
        }
        if ui
            .button("Edge wear")
            .on_hover_text("Worn, lightened convex edges")
            .clicked()
        {
            state.actions.apply_edge_wear = Some(levels);
        }
    });

    // Generic route: pick a source, then drive a brush-colored tint layer or the
    // active layer's reveal mask from it (the Substance-style mask workflow).
    ui.add_space(6.0);
    egui::ComboBox::from_label("Source")
        .selected_text(state.effect_source.name())
        .show_ui(ui, |ui| {
            for src in MapSource::ALL {
                ui.selectable_value(&mut state.effect_source, src, src.name());
            }
        });
    ui.horizontal(|ui| {
        if ui
            .button("Tint layer")
            .on_hover_text("New layer: the Color above, masked to the source")
            .clicked()
        {
            state.actions.apply_tint = Some((state.effect_source, levels, state.brush.color));
        }
        if ui
            .button("Mask layer")
            .on_hover_text("Set the active layer's mask from the source")
            .clicked()
        {
            state.actions.mask_from_map = Some((state.effect_source, levels));
        }
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

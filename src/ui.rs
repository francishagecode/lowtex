// src/ui.rs
//
// The egui side panel and the live editor state it drives. Per design principle
// #1 ("speak plain language, not PBR") the controls use painter words — Color,
// Size, Opacity, Hardness — not material vocabulary.
//
// Layout philosophy (the "dead simple" pass): the panel opens showing only the
// painting essentials — Tool, Color, brush settings, and the Layers stack. Every
// occasional or advanced area (mesh effects, material fill, export, palette,
// viewport) lives behind a collapsed `CollapsingHeader`, so a newcomer sees a
// short, calm panel instead of a wall of buttons. Layers are large, full-width
// clickable rows; the active layer's blend + opacity sit inline directly beneath
// it. Controls that don't apply to the current tool are hidden, not greyed out.

use egui::Context;

use crate::bake::{Levels, MapSource, NoiseMod};
use crate::effects::{Effect, EffectKind};
use crate::export::ExportPreset;
use crate::layers::BlendMode;
use crate::noise::NoiseKind;
use crate::paint::Brush;
use crate::renderer::{PaletteSettings, SymmetryAxis, TextureFilter};

/// The active editing tool. The brush drags a stroke; the fills are one-shot
/// "paint bucket" clicks (solid color, ignoring what's already there), scoped from
/// smallest to largest region: a flat face, a UV island, the whole object.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    /// Freehand brush — drag to paint a stroke. Paints the current color, or, if a
    /// brush image is loaded, the UV-tiled image (revealing the same content a material
    /// fill would, only where you drag). Clear the image to go back to solid color.
    Brush,
    /// Fill the flat face (coplanar facet) of the model under the cursor.
    FillFace,
    /// Fill the UV island under the cursor.
    FillIsland,
    /// Fill the whole object (every texel its UVs cover).
    FillObject,
}

/// A pointer event inside the 2D UV editor, in UV space (`[0,1]²`). The panel emits a
/// sequence per stroke — `Begin`, one `Segment` per drag step (a single click is one
/// zero-length segment), then `End` — which the App replays onto the renderer's UV
/// paint path so a UV stroke is one undo step, exactly like a 3D stroke.
#[derive(Clone, Copy)]
pub enum UvEvent {
    Begin,
    Segment { from: glam::Vec2, to: glam::Vec2 },
    End,
}

/// A read-only snapshot of one layer, synced from the renderer for the UI list.
#[derive(Clone)]
pub struct LayerInfo {
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub blend: BlendMode,
    /// The layer's non-destructive effect stack (G28), bottom-to-top in apply order.
    pub effects: Vec<Effect>,
}

/// One entry in the texture-folder brush/stamp browser: a source image found in the
/// user's chosen folder, plus its lazily-uploaded thumbnail for the grid swatch.
pub struct BrushEntry {
    pub path: std::path::PathBuf,
    pub name: String,
    /// A freshly-downsampled `(w, h, RGBA8)` thumbnail staged by the App on scan; the
    /// UI uploads it to `tex` once, then takes it back to `None`.
    pub thumb: Option<(u32, u32, Vec<u8>)>,
    /// The GPU handle for this entry's swatch, held across frames after the first upload.
    pub tex: Option<egui::TextureHandle>,
}

/// One-shot requests the UI raises this frame, drained by the App after the egui
/// run (file dialogs and texture ops happen outside the egui closure).
#[derive(Default)]
pub struct UiActions {
    pub save_png: bool,
    pub open_texture: bool,
    pub open_model: bool,
    /// Project save/load (G24). `save_project` is always "Save As" (a dialog);
    /// `quicksave` (G31) writes straight to the open file's path, falling back to a
    /// dialog only when the project hasn't been saved anywhere yet.
    pub save_project: bool,
    pub quicksave: bool,
    pub open_project: bool,
    // Undo/redo (history over the layer stack).
    pub undo: bool,
    pub redo: bool,
    /// Snapshot the layer stack before a continuous edit (an opacity-slider drag),
    /// so the whole drag collapses to one undo step.
    pub checkpoint: bool,
    /// Index into `Palette::builtins()` to make active.
    pub select_builtin_palette: Option<usize>,
    /// Generate a palette from a chosen image with this many colors.
    pub generate_palette: Option<usize>,
    /// Region-aware one-time seam cleanup: fill island-rim "teeth" from same-facet paint.
    pub clean_seams: bool,
    // Layer ops (G10).
    pub add_layer: bool,
    pub remove_layer: bool,
    pub move_layer_up: bool,
    pub move_layer_down: bool,
    pub merge_layer_down: bool,
    pub select_layer: Option<usize>,
    pub set_layer_visible: Option<(usize, bool)>,
    pub set_layer_opacity: Option<(usize, f32)>,
    pub set_layer_blend: Option<(usize, BlendMode)>,
    /// Rename a layer by hand (locks its auto-name). Carries (index, new name).
    pub set_layer_name: Option<(usize, String)>,
    // Per-layer effects (G28), all targeting the active layer. `set_effect` carries
    // the effect's index plus its new value (a parameter-slider edit).
    pub add_effect: Option<EffectKind>,
    pub remove_effect: Option<usize>,
    pub move_effect_up: Option<usize>,
    pub move_effect_down: Option<usize>,
    pub set_effect: Option<(usize, Effect)>,
    // Mesh effects (AO + curvature → layers and masks), all sharing the Levels
    // remap and the optional noise breakup from the controls. Presets fix the
    // source/color/blend; the two generic actions route the chosen `effect_source`
    // into a brush-colored tint layer or into the active layer's reveal mask.
    pub apply_ao: Option<(Levels, Option<NoiseMod>)>,
    pub apply_highlight: Option<(Levels, Option<NoiseMod>)>,
    pub apply_dirt: Option<(Levels, Option<NoiseMod>)>,
    pub apply_edge_wear: Option<(Levels, Option<NoiseMod>)>,
    pub apply_tint: Option<(MapSource, Levels, [f32; 3], Option<NoiseMod>)>,
    pub mask_from_map: Option<(MapSource, Levels, Option<NoiseMod>)>,
    /// Directional-light ("sun") effects, sharing the same Levels/noise controls.
    /// `apply_sun` is a warm highlight on the lit faces; `apply_top_dust` is the
    /// top-down dust look (the UI aims the sun straight up before requesting it).
    pub apply_sun: Option<(Levels, Option<NoiseMod>)>,
    pub apply_top_dust: Option<(Levels, Option<NoiseMod>)>,
    /// Gradient map: color the chosen source's value through a low→high ramp.
    /// Carries (source, levels, low_rgb, high_rgb, noise) — colors are 0..1 sRGB.
    pub apply_gradient: Option<(MapSource, Levels, [f32; 3], [f32; 3], Option<NoiseMod>)>,
    /// Export the texture (G23): `true` = true indexed PNG, `false` = RGBA8.
    pub export_png: Option<bool>,
    /// Export the unwrapped mesh (positions + the new UVs) as a Wavefront OBJ, so the
    /// painted texture has geometry to map onto outside lowtex.
    pub export_obj: bool,
    /// Fill the active layer with a chosen material image, tiled this many times.
    pub fill_material: Option<f32>,
    /// Load an image for the brush to paint with (UV-tiled) instead of solid color.
    pub load_brush_image: bool,
    /// Drop the loaded brush image, reverting the brush to painting solid color.
    pub clear_brush_image: bool,
    /// Open a native folder picker; the App scans it for images and stages their
    /// thumbnails into `brush_folder_entries` for the brush/stamp browser.
    pub open_brush_folder: bool,
    /// Use a texture from the folder browser as the brush image (carries its path).
    pub use_brush_entry: Option<std::path::PathBuf>,
    /// Open a native folder picker for brush alpha tips; the App scans it for images and
    /// stages their thumbnails into `alpha_folder_entries` for the tip browser.
    pub open_alpha_folder: bool,
    /// Use an image from the tip browser as the brush alpha tip (carries its path).
    pub use_alpha_entry: Option<std::path::PathBuf>,
    /// Drop the loaded alpha tip, reverting the brush to the circular falloff.
    pub clear_alpha: bool,
    /// Re-unwrap the mesh's UVs at the chosen texel density (the atlas size is
    /// derived from it). Carries the density the user picked.
    pub unwrap: Option<crate::unwrap::Density>,
    /// Set the paint texture resolution directly, resampling the existing paint into
    /// it. Carries the square size in texels. Overrides the unwrap-derived size until
    /// the next unwrap (a manual choice the user made in the Texture section).
    pub set_resolution: Option<u32>,
    /// Re-unwrap at an exact texel density (carries texels-per-world-unit, i.e. per
    /// meter). Unlike `unwrap`, the atlas is sized to hold the mesh at this density
    /// rather than filling a preset-sized atlas.
    pub unwrap_at_density: Option<f32>,
    /// Paint events from the 2D UV editor this frame, in order. Replayed onto the
    /// renderer's `paint_uv_*` path by the App. Empty when the panel is closed or idle.
    pub uv_strokes: Vec<UvEvent>,
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
    /// How many times the texture-brush image tiles across UV (independent of fill).
    pub brush_tile: f32,
    /// Whether an image has been loaded for the texture brush (mirrors the renderer),
    /// so the panel can hint when the brush has nothing to paint with yet.
    pub brush_image_loaded: bool,
    /// A freshly-downsampled `(w, h, RGBA8)` brush-image preview the App stages on
    /// load; `build` uploads it to `brush_thumb_tex` once, then takes it back to None.
    pub brush_thumb: Option<(u32, u32, Vec<u8>)>,
    /// The GPU texture handle for the brush-image swatch, held across frames so the
    /// preview isn't re-uploaded every frame. Cleared with the brush image.
    pub brush_thumb_tex: Option<egui::TextureHandle>,
    /// How a loaded brush image is applied: `false` = Brush (tiled, consistent material
    /// field), `true` = Stamp (oriented decal). For Stamp, `stamp_angle_deg` rotates it
    /// and `stamp_tint` recolours a grayscale alpha to the brush swatch.
    pub brush_stamp: bool,
    pub stamp_angle_deg: f32,
    pub stamp_tint: bool,
    /// The user-chosen texture folder (for the panel header) and the images found in
    /// it, each usable as a tiled brush or a stamp. Empty until a folder is opened.
    pub brush_folder: Option<std::path::PathBuf>,
    pub brush_folder_entries: Vec<BrushEntry>,
    /// Brush alpha tips: an optional grayscale image that shapes the dab in place of the
    /// circle. `alpha_folder`/`alpha_folder_entries` are the chosen folder + its swatches
    /// (same browser as the texture folder); `alpha_loaded` mirrors whether the renderer
    /// has a tip set; `alpha_invert` flips dark/light for black-on-white tip packs. The
    /// tip's rotation reuses `stamp_angle_deg`.
    pub alpha_folder: Option<std::path::PathBuf>,
    pub alpha_folder_entries: Vec<BrushEntry>,
    pub alpha_loaded: bool,
    pub alpha_invert: bool,
    /// Staged thumbnail of the active tip (uploaded to `alpha_thumb_tex` once, then taken
    /// back to None), mirroring `brush_thumb`/`brush_thumb_tex` for the brush image.
    pub alpha_thumb: Option<(u32, u32, Vec<u8>)>,
    pub alpha_thumb_tex: Option<egui::TextureHandle>,
    /// Mirror painting: when on, every dab is also painted at its reflection across
    /// the model-symmetry plane for `symmetry_axis` (through the mesh center).
    pub symmetry_on: bool,
    pub symmetry_axis: SymmetryAxis,
    /// Face lock: when on, each dab stays inside the flat face it lands on instead
    /// of wrapping across an edge onto a neighbouring face.
    pub lock_face: bool,
    /// Levels controls shared by every mesh effect: overall strength, mid-tone
    /// contrast, and invert. `effect_source` is which baked channel the generic
    /// "Tint layer" / "Mask layer" actions read from.
    pub ao_strength: f32,
    pub effect_contrast: f32,
    pub effect_invert: bool,
    pub effect_source: MapSource,
    /// "Break up" controls: a procedural-noise modifier multiplied into any mesh
    /// effect. `noise_amount` 0 = off; the rest pick the noise's character.
    pub noise_kind: NoiseKind,
    pub noise_scale: f32,
    pub noise_contrast: f32,
    pub noise_amount: f32,
    /// Directional-light ("sun") controls for the Light map source: elevation above
    /// the horizon and azimuth around the model (both degrees), and whether the sun
    /// casts shadows. `sun_dir` turns these into a direction *toward* the light.
    pub sun_elevation: f32,
    pub sun_azimuth: f32,
    pub sun_shadow: bool,
    /// Gradient-map endpoints (0..1 sRGB): the color at the map's low and high ends.
    pub grad_low: [f32; 3],
    pub grad_high: [f32; 3],
    /// Viewport display (G29): background color (sRGB) and the grid toggle. Pushed
    /// to the renderer every frame (cheap — they just set fields). The compass is
    /// always shown (its axes are clickable), so it has no toggle.
    pub bg_color: [f32; 3],
    pub show_grid: bool,
    /// How the painted texture is filtered onto the model (G30): nearest (crisp,
    /// the PSX default) or linear (smoothed). Pushed to the renderer every frame.
    pub texture_filter: TextureFilter,
    /// Mirror of the renderer's current texture resolution (now *derived* by the
    /// unwrap, shown read-only).
    pub resolution: u32,
    /// Texel density requested for the next "Unwrap UVs" (remembers the last choice).
    pub unwrap_density: crate::unwrap::Density,
    /// Stack congruent charts (identical/mirrored parts) onto shared UV space on the next
    /// "Unwrap UVs". Remembers the last choice.
    pub unwrap_overlap: bool,
    /// Texels-per-world-unit (per meter) for the "Unwrap at density" action — an exact
    /// density the user types, as an alternative to the Low/Medium/High presets.
    pub unwrap_texels_per_m: f32,
    /// Achieved texels-per-world-unit from the last unwrap (0 before any), shown in the
    /// Texture readout so the user can confirm a requested density took effect.
    pub last_density_d: f32,
    /// Whether the last unwrap had to reduce density to fit the GPU texture limit,
    /// so the resolution readout can flag it.
    pub last_atlas_clamped: bool,
    /// Mirrors of the history state, to enable/disable the Undo/Redo buttons.
    pub can_undo: bool,
    pub can_redo: bool,
    /// The panel's current width in logical points, written each frame after the
    /// panel lays out (it's user-draggable). The App scales this by the DPI factor
    /// and hands it to the renderer so the viewport object clears the panel.
    pub panel_width: f32,
    /// Edit buffer for the active layer's name field. Mirrors the synced (derived or
    /// hand-set) name whenever the field isn't focused, so it tracks auto-renames and
    /// layer switches; while focused it holds the user's in-progress typing.
    pub layer_name_edit: String,
    /// 2D UV editor (paint directly on the unwrapped atlas instead of the model).
    /// `show_uv_panel` toggles the right-side split panel. The rest is state the panel
    /// needs but the egui closure can't pull from the renderer directly, so the App
    /// stages it in before `build`:
    ///   - `uv_image` is the latest atlas `(size, RGBA8)`, restaged only when the
    ///     renderer's paint version moves (`uv_image_version`); `build` uploads it to
    ///     `uv_tex` once per change, mirroring the brush-thumbnail flow.
    ///   - `uv_edges` is the island wireframe (`[u0,v0,u1,v1]` in `[0,1]`), restaged
    ///     only when the mesh topology changes (`uv_edges_version`).
    /// `uv_drag_last` is the previous UV during an in-panel drag, so each step emits a
    /// `Segment { from, to }`.
    pub show_uv_panel: bool,
    pub uv_image: Option<(u32, Vec<u8>)>,
    pub uv_image_version: u64,
    pub uv_tex: Option<egui::TextureHandle>,
    pub uv_edges: Vec<[f32; 4]>,
    pub uv_edges_version: u64,
    pub uv_drag_last: Option<glam::Vec2>,
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
            brush_tile: 4.0,
            brush_image_loaded: false,
            brush_thumb: None,
            brush_thumb_tex: None,
            brush_stamp: false,
            stamp_angle_deg: 0.0,
            stamp_tint: false,
            brush_folder: None,
            brush_folder_entries: Vec::new(),
            alpha_folder: None,
            alpha_folder_entries: Vec::new(),
            alpha_loaded: false,
            alpha_invert: false,
            alpha_thumb: None,
            alpha_thumb_tex: None,
            symmetry_on: false,
            lock_face: false,
            symmetry_axis: SymmetryAxis::X,
            ao_strength: 0.75,
            effect_contrast: 0.0,
            effect_invert: false,
            effect_source: MapSource::Cavities,
            noise_kind: NoiseKind::Perlin,
            noise_scale: 6.0,
            noise_contrast: 0.3,
            noise_amount: 0.0,
            sun_elevation: 50.0, // high, warm key — matches the shader's fixed light
            sun_azimuth: 45.0,
            sun_shadow: true,
            grad_low: [0.10, 0.09, 0.12],    // deep shadow → ...
            grad_high: [0.85, 0.82, 0.74],   // ... pale highlight
            bg_color: [0.221, 0.272, 0.313], // sRGB of the default dark teal
            show_grid: true,
            texture_filter: TextureFilter::default(),
            resolution: 128,
            unwrap_density: crate::unwrap::Density::default(),
            unwrap_overlap: false,
            unwrap_texels_per_m: 128.0,
            last_density_d: 0.0,
            last_atlas_clamped: false,
            can_undo: false,
            can_redo: false,
            panel_width: 248.0,
            layer_name_edit: String::new(),
            show_uv_panel: false,
            uv_image: None,
            uv_image_version: u64::MAX, // force the first stage to differ from the renderer's 0
            uv_tex: None,
            uv_edges: Vec::new(),
            uv_edges_version: u64::MAX,
            uv_drag_last: None,
            actions: UiActions::default(),
        }
    }
}

impl UiState {
    /// The procedural-noise breakup the controls currently describe, or `None` when
    /// the amount is zero (so callers cheaply skip noise entirely).
    fn noise_mod(&self) -> Option<NoiseMod> {
        (self.noise_amount > 0.0).then_some(NoiseMod {
            kind: self.noise_kind,
            scale: self.noise_scale,
            contrast: self.noise_contrast,
            amount: self.noise_amount,
        })
    }

    /// The Levels remap the mesh-effect controls currently describe. Computed from
    /// stored fields so the one-click effects work even while the "Fine-tune"
    /// sliders are collapsed out of view.
    fn levels(&self) -> Levels {
        Levels {
            invert: self.effect_invert,
            contrast: self.effect_contrast,
            strength: self.ao_strength,
        }
    }

    /// The sun direction (pointing *toward* the light) the elevation/azimuth sliders
    /// describe. Elevation 90° is straight up (the top-down look); azimuth orbits it
    /// around the model. Returned as a plain `[f32; 3]` so the renderer owns glam.
    pub fn sun_dir(&self) -> [f32; 3] {
        let el = self.sun_elevation.to_radians();
        let az = self.sun_azimuth.to_radians();
        [el.cos() * az.cos(), el.sin(), el.cos() * az.sin()]
    }
}

/// Apply the Catppuccin Mocha theme. Called once on startup (and in the headless
/// `--ui` screenshot path so captures match). This sets colors/visuals only; egui's
/// stock spacing, proportional fonts, and rounding are left at their defaults.
pub fn install_style(ctx: &Context) {
    catppuccin_egui::set_theme(ctx, catppuccin_egui::MOCHA);
}

/// A small, quiet section heading for the always-visible primary zones. Title-case
/// and strong (not the old shouty ALL-CAPS), so the eye can scan the panel.
fn heading(ui: &mut egui::Ui, text: &str) {
    ui.add_space(4.0);
    ui.label(egui::RichText::new(text).strong());
}

/// A segmented control: equal-width toggle buttons spanning the full panel width,
/// one per `(value, label, hover)`. The button matching `*current` reads selected;
/// clicking one writes it back and returns its index, so callers that need a side
/// effect on selection (e.g. resetting a dependent field) can react.
fn segmented<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    current: &mut T,
    options: &[(T, &str, &str)],
) -> Option<usize> {
    let mut clicked = None;
    ui.columns(options.len(), |cols| {
        for (i, (value, label, hover)) in options.iter().enumerate() {
            let size = egui::vec2(cols[i].available_width(), cols[i].spacing().interact_size.y);
            let mut resp =
                cols[i].add_sized(size, egui::SelectableLabel::new(*current == *value, *label));
            if !hover.is_empty() {
                resp = resp.on_hover_text(*hover);
            }
            if resp.clicked() {
                *current = *value;
                clicked = Some(i);
            }
        }
    });
    clicked
}

/// A row of equal-width action buttons spanning the full panel width, one per
/// `(label, hover)`. Returns the index of the button clicked this frame, if any.
fn button_group(ui: &mut egui::Ui, buttons: &[(&str, &str)]) -> Option<usize> {
    let mut clicked = None;
    ui.columns(buttons.len(), |cols| {
        for (i, (label, hover)) in buttons.iter().enumerate() {
            let size = egui::vec2(cols[i].available_width(), cols[i].spacing().interact_size.y);
            let mut resp = cols[i].add_sized(size, egui::Button::new(*label));
            if !hover.is_empty() {
                resp = resp.on_hover_text(*hover);
            }
            if resp.clicked() {
                clicked = Some(i);
            }
        }
    });
    clicked
}

/// The top menu-bar header. The common document operations (open/save/export) and
/// undo/redo are lifted out of the side panel into a conventional File/Edit/Mesh
/// menu strip, so the panel itself holds only painting controls. Drawn full-width
/// above the side panel and viewport; egui claims clicks on it, so the scene
/// (which still renders full-height behind this thin strip) never mis-picks.
fn menu_bar(ctx: &Context, state: &mut UiState) {
    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
        egui::menu::bar(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Open model…").clicked() {
                    state.actions.open_model = true;
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("Open image…").clicked() {
                    state.actions.open_texture = true;
                    ui.close_menu();
                }
                if ui.button("Save PNG…").clicked() {
                    state.actions.save_png = true;
                    ui.close_menu();
                }
                ui.separator();
                if ui
                    .button("Export indexed PNG…")
                    .on_hover_text("True paletted PNG for retro pipelines (needs Quantize on)")
                    .clicked()
                {
                    state.actions.export_png = Some(true);
                    ui.close_menu();
                }
                if ui.button("Export RGBA PNG…").clicked() {
                    state.actions.export_png = Some(false);
                    ui.close_menu();
                }
                if ui
                    .button("Export mesh (.obj)…")
                    .on_hover_text(
                        "The unwrapped geometry + new UVs — pair it with the exported texture",
                    )
                    .clicked()
                {
                    state.actions.export_obj = true;
                    ui.close_menu();
                }
                ui.separator();
                if ui
                    .button("Save project")
                    .on_hover_text("Ctrl/⌘+S — write to the open file (asks where the first time)")
                    .clicked()
                {
                    state.actions.quicksave = true;
                    ui.close_menu();
                }
                if ui
                    .button("Save project as…")
                    .on_hover_text("Ctrl/⌘+Shift+S")
                    .clicked()
                {
                    state.actions.save_project = true;
                    ui.close_menu();
                }
                if ui.button("Open project (.lowtex)…").clicked() {
                    state.actions.open_project = true;
                    ui.close_menu();
                }
            });
            ui.menu_button("Edit", |ui| {
                if ui
                    .add_enabled(state.can_undo, egui::Button::new("Undo"))
                    .on_hover_text("Ctrl/⌘+Z")
                    .clicked()
                {
                    state.actions.undo = true;
                    ui.close_menu();
                }
                if ui
                    .add_enabled(state.can_redo, egui::Button::new("Redo"))
                    .on_hover_text("Ctrl/⌘+Shift+Z")
                    .clicked()
                {
                    state.actions.redo = true;
                    ui.close_menu();
                }
                ui.separator();
                if ui
                    .button("Clean Seams")
                    .on_hover_text(
                        "Fill dark island-edge 'teeth' left by older GPU paint, from each \
                         island's own colour. Undoable; only repairs unpainted edge texels.",
                    )
                    .clicked()
                {
                    state.actions.clean_seams = true;
                    ui.close_menu();
                }
            });
            ui.menu_button("Mesh", |ui| {
                ui.menu_button("Unwrap UVs", |ui| {
                    ui.label(
                        egui::RichText::new("Connectivity charts, constant texel size")
                            .weak()
                            .small(),
                    );
                    // Preset density only scales the texels-per-world-unit (and the
                    // derived texture size); the constant-density invariant holds either
                    // way. The atlas is then filled for max sharpness.
                    for d in crate::unwrap::Density::ALL {
                        ui.radio_value(&mut state.unwrap_density, d, d.name());
                    }
                    ui.separator();
                    ui.checkbox(&mut state.unwrap_overlap, "Overlap identical UVs")
                        .on_hover_text(
                            "Stack identical (and mirrored) islands onto the same texture \
                             region: paint one, paint all. Shrinks the atlas on meshes with \
                             repeated or symmetric parts.",
                        );
                    if ui.button("Unwrap (preset)").clicked() {
                        state.actions.unwrap = Some(state.unwrap_density);
                        ui.close_menu();
                    }

                    // Exact density: pin texels-per-world-unit numerically instead of a
                    // preset. The atlas is sized to hold the mesh at this density rather
                    // than filled, so e.g. 128 means 128 texels span one world unit.
                    ui.separator();
                    ui.label(
                        egui::RichText::new("Or set an exact density")
                            .weak()
                            .small(),
                    );
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut state.unwrap_texels_per_m)
                                .speed(1.0)
                                .range(1.0..=4096.0),
                        )
                        .on_hover_text(
                            "Texels per world unit (per meter): how many texels span one \
                             unit of the model. The atlas is sized to fit the mesh at this \
                             density.",
                        );
                        ui.label("texels / m");
                    });
                    if ui.button("Unwrap at this density").clicked() {
                        state.actions.unwrap_at_density = Some(state.unwrap_texels_per_m);
                        ui.close_menu();
                    }
                });
            });

            // Viewport display controls (G29/G30) live inline in the top bar so the
            // common view toggles stay one click away instead of behind a collapsed
            // side-panel header. They only set fields; the App pushes them each frame.
            ui.separator();
            ui.color_edit_button_rgb(&mut state.bg_color)
                .on_hover_text("Background color");
            ui.checkbox(&mut state.show_grid, "Grid");
            ui.separator();
            ui.label("Filter").on_hover_text(
                "How the painted texture samples onto the model: nearest is crisp \
                 (PSX), linear smooths it",
            );
            egui::ComboBox::from_id_salt("texture_filter")
                .selected_text(state.texture_filter.label())
                .show_ui(ui, |ui| {
                    for mode in [TextureFilter::Nearest, TextureFilter::Linear] {
                        ui.selectable_value(&mut state.texture_filter, mode, mode.label());
                    }
                });
        });
    });
}

/// Build the controls for this frame: the top menu bar, the left controls panel,
/// and the floating Layers palette. Mutates `state` in place and records one-shot
/// requests in `state.actions` (reset each call).
pub fn build(ctx: &Context, state: &mut UiState) {
    state.actions = UiActions::default();

    menu_bar(ctx, state);

    let panel = egui::SidePanel::left("controls")
        .resizable(true)
        .default_width(248.0)
        .width_range(200.0..=480.0)
        .show(ctx, |ui| {
            // Wrap the whole panel so controls stay reachable when the content is
            // taller than the window — otherwise the bottom sections get clipped.
            egui::ScrollArea::vertical().show(ui, |ui| {
                // ---- Brand ----
                ui.add_space(2.0);
                ui.heading("LOWTEX");
                ui.label(
                    egui::RichText::new("low-poly texture painter")
                        .weak()
                        .small(),
                );

                ui.separator();

                // ---- Tool: paint tools on top, the three fills grouped below ----
                heading(ui, "Tool");
                segmented(
                    ui,
                    &mut state.tool,
                    &[(
                        Tool::Brush,
                        "Brush",
                        "Drag to paint color, or a loaded image",
                    )],
                );
                ui.label(egui::RichText::new("Fill").weak());
                segmented(
                    ui,
                    &mut state.tool,
                    &[
                        (Tool::FillFace, "Face", "Fill the flat face you click"),
                        (Tool::FillIsland, "Island", "Fill the UV island you click"),
                        (Tool::FillObject, "All", "Fill the whole object"),
                    ],
                );

                // Open the 2D UV editor (a split panel on the right): paint straight on
                // the unwrapped atlas with the same brush, seeing both views update live.
                ui.add_space(2.0);
                ui.checkbox(&mut state.show_uv_panel, "UV editor")
                    .on_hover_text(
                        "Paint directly on the unwrapped texture in a panel beside the model",
                    );

                // ---- Color (used by both the brush and the fills) ----
                heading(ui, "Color");
                ui.color_edit_button_rgb(&mut state.brush.color);

                // Brush shape sliders apply only to the tools that drag a brush; for
                // the one-shot fills they're irrelevant, so we hide them entirely
                // rather than show a row of dead, greyed-out controls.
                let brush_active = matches!(state.tool, Tool::Brush);
                if brush_active {
                    ui.add_space(2.0);
                    // Radius is in texels, so the useful max scales with the texture: at
                    // least 256 (enough for the PSX sizes), and up to the full resolution
                    // for the larger atlases the texture/density controls now allow. A
                    // dab bigger than the texture just clamps to its bounds. Logarithmic
                    // so the small radii you paint with most keep fine control.
                    let brush_max = (state.resolution as f32).max(256.0);
                    ui.add(
                        egui::Slider::new(&mut state.brush.radius, 1.0..=brush_max)
                            .logarithmic(true)
                            .text("Size"),
                    );
                    ui.add(egui::Slider::new(&mut state.brush.opacity, 0.0..=1.0).text("Opacity"));
                    ui.add(
                        egui::Slider::new(&mut state.brush.hardness, 0.0..=1.0).text("Hardness"),
                    );

                    // Eraser: the same brush, but it removes paint (lowers alpha toward
                    // transparent) instead of laying the swatch color, revealing whatever
                    // sits below. Shape/opacity/symmetry/snap all still apply.
                    ui.checkbox(&mut state.brush.erase, "Eraser").on_hover_text(
                        "Remove paint instead of laying color, revealing the layers below",
                    );

                    // Mirror painting: stamp every dab on the symmetric side of the
                    // model too. The axis picker is live only while symmetry is on.
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut state.symmetry_on, "Symmetry")
                            .on_hover_text("Mirror each dab across the model's center plane");
                        ui.add_enabled_ui(state.symmetry_on, |ui| {
                            egui::ComboBox::from_id_salt("symmetry_axis")
                                .selected_text(state.symmetry_axis.name())
                                .width(44.0)
                                .show_ui(ui, |ui| {
                                    for a in SymmetryAxis::ALL {
                                        ui.selectable_value(&mut state.symmetry_axis, a, a.name());
                                    }
                                });
                        });
                    });

                    // Face lock: keep each dab inside the flat face it lands on, so
                    // strokes near an edge don't wrap onto the neighbouring face.
                    ui.checkbox(&mut state.lock_face, "Lock to face")
                        .on_hover_text("Confine each dab to the flat face under the cursor");

                    // Grid snap: round each dab's center to a coarse texel cell, so
                    // strokes quantize to the grid for crisp, blocky PSX-style edges.
                    ui.checkbox(&mut state.brush.snap_to_texel, "Snap to grid")
                        .on_hover_text(
                        "Round each dab to a texel-grid cell for crisp, blocky PSX-style strokes",
                    );
                    if state.brush.snap_to_texel {
                        ui.add(
                            egui::Slider::new(&mut state.brush.snap_grid, 2.0..=16.0)
                                .step_by(1.0)
                                .text("Grid"),
                        )
                        .on_hover_text("Grid cell size in texels — bigger is chunkier");
                    }
                }

                // Brush image: load one to paint it (UV-tiled) instead of solid color;
                // clear it to go back to color. The Tile slider only matters once an
                // image is loaded, so it's hidden until then.
                if state.tool == Tool::Brush {
                    ui.add_space(2.0);
                    if state.brush_image_loaded {
                        // Upload a newly-staged preview to a GPU texture once; the handle
                        // then persists across frames until the image is cleared/replaced.
                        if let Some((w, h, px)) = state.brush_thumb.take() {
                            let img = egui::ColorImage::from_rgba_unmultiplied(
                                [w as usize, h as usize],
                                &px,
                            );
                            state.brush_thumb_tex = Some(ui.ctx().load_texture(
                                "brush-thumb",
                                img,
                                egui::TextureOptions::LINEAR,
                            ));
                        }
                        // Show the swatch (fit within 48pt, keeping aspect) beside the
                        // load/clear buttons.
                        let swatch = state.brush_thumb_tex.as_ref().map(|tex| {
                            let [tw, th] = tex.size();
                            let scale = 48.0 / tw.max(th) as f32;
                            let size = egui::vec2(tw as f32 * scale, th as f32 * scale);
                            egui::load::SizedTexture::new(tex.id(), size)
                        });
                        ui.horizontal(|ui| {
                            if let Some(swatch) = swatch {
                                ui.add(egui::Image::new(swatch));
                            }
                            ui.vertical(|ui| {
                                if ui.button("Brush image…").clicked() {
                                    state.actions.load_brush_image = true;
                                }
                                if ui.button("Clear").clicked() {
                                    state.actions.clear_brush_image = true;
                                }
                            });
                        });
                        // Brush = the image anchored & tiled in texture space (a
                        // consistent material field); Stamp = an oriented decal you can
                        // overdraw, shift, and rotate.
                        ui.horizontal(|ui| {
                            ui.selectable_value(&mut state.brush_stamp, false, "Brush")
                                .on_hover_text("Tile the image as a consistent material");
                            ui.selectable_value(&mut state.brush_stamp, true, "Stamp")
                                .on_hover_text("Place the image as an oriented decal");
                        });
                        if state.brush_stamp {
                            ui.add(
                                egui::Slider::new(&mut state.stamp_angle_deg, 0.0..=360.0)
                                    .text("Angle")
                                    .suffix("°"),
                            );
                            // Same `brush_tile` knob, but here it scales the decal down
                            // within the brush footprint (1 = full, larger = smaller),
                            // so the stamp size isn't tied to brush radius alone.
                            ui.add(
                                egui::Slider::new(&mut state.brush_tile, 1.0..=16.0).text("Scale"),
                            )
                            .on_hover_text("Shrink the stamp within the brush footprint");
                            ui.checkbox(&mut state.stamp_tint, "Tint to color")
                                .on_hover_text(
                                    "Recolor a grayscale stamp to the brush color \
                                 (off = keep the image's own colors)",
                                );
                        } else {
                            ui.add(
                                egui::Slider::new(&mut state.brush_tile, 1.0..=16.0).text("Tile"),
                            );
                        }
                    } else {
                        if ui.button("Brush image…").clicked() {
                            state.actions.load_brush_image = true;
                        }
                        ui.label(
                            egui::RichText::new("Load an image to paint it instead of color.")
                                .weak()
                                .small(),
                        );
                    }

                    // Texture folder: point at any folder of images, then click a swatch
                    // to load it as the brush image (it stamps when on the 3D surface).
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        if ui.button("Texture folder…").clicked() {
                            state.actions.open_brush_folder = true;
                        }
                        if let Some(name) = state
                            .brush_folder
                            .as_ref()
                            .and_then(|f| f.file_name())
                            .map(|n| n.to_string_lossy().into_owned())
                        {
                            ui.label(egui::RichText::new(name).weak().small());
                        }
                    });
                    brush_folder_grid(ui, state);

                    // Brush tip (alpha): point at a folder of grayscale tip images, then
                    // click one to shape the dab with it instead of the circle. Brightness
                    // (white) paints; Invert flips that; Angle rotates the tip. Clear goes
                    // back to the round brush.
                    ui.add_space(6.0);
                    ui.separator();
                    ui.label(egui::RichText::new("Brush Tip (Alpha)").strong());
                    if state.alpha_loaded {
                        // Upload a newly-staged tip preview once; the handle persists until
                        // the tip is cleared/replaced (mirrors the brush-image swatch).
                        if let Some((w, h, px)) = state.alpha_thumb.take() {
                            let img = egui::ColorImage::from_rgba_unmultiplied(
                                [w as usize, h as usize],
                                &px,
                            );
                            state.alpha_thumb_tex = Some(ui.ctx().load_texture(
                                "alpha-thumb",
                                img,
                                egui::TextureOptions::LINEAR,
                            ));
                        }
                        let swatch = state.alpha_thumb_tex.as_ref().map(|tex| {
                            let [tw, th] = tex.size();
                            let scale = 48.0 / tw.max(th) as f32;
                            let size = egui::vec2(tw as f32 * scale, th as f32 * scale);
                            egui::load::SizedTexture::new(tex.id(), size)
                        });
                        ui.horizontal(|ui| {
                            if let Some(swatch) = swatch {
                                ui.add(egui::Image::new(swatch));
                            }
                            if ui.button("Clear (circle)").clicked() {
                                state.actions.clear_alpha = true;
                            }
                            ui.checkbox(&mut state.alpha_invert, "Invert")
                                .on_hover_text("Paint dark pixels instead of light");
                        });
                        ui.add(
                            egui::Slider::new(&mut state.stamp_angle_deg, 0.0..=360.0)
                                .text("Angle")
                                .suffix("°"),
                        )
                        .on_hover_text("Rotate the tip in the surface plane");
                    } else {
                        ui.label(
                            egui::RichText::new("Pick a tip below to shape the brush.")
                                .weak()
                                .small(),
                        );
                    }
                    ui.horizontal(|ui| {
                        if ui.button("Alpha tips folder…").clicked() {
                            state.actions.open_alpha_folder = true;
                        }
                        if let Some(name) = state
                            .alpha_folder
                            .as_ref()
                            .and_then(|f| f.file_name())
                            .map(|n| n.to_string_lossy().into_owned())
                        {
                            ui.label(egui::RichText::new(name).weak().small());
                        }
                    });
                    alpha_folder_grid(ui, state);
                }

                ui.add_space(6.0);

                // ---- Everything occasional/advanced collapses away by default ----
                // The active layer's non-destructive effect stack (lives here on the
                // left, not in the Layers palette).
                egui::CollapsingHeader::new("Layer effects")
                    .default_open(false)
                    .show(ui, |ui| effects_section(ui, state));
                egui::CollapsingHeader::new("Mesh effects")
                    .default_open(false)
                    .show(ui, |ui| mesh_effects_section(ui, state));
                egui::CollapsingHeader::new("Material")
                    .default_open(false)
                    .show(ui, |ui| material_section(ui, state));
                egui::CollapsingHeader::new("Canvas")
                    .default_open(false)
                    .show(ui, |ui| export_section(ui, state));
                egui::CollapsingHeader::new("Palette")
                    .default_open(false)
                    .show(ui, |ui| palette_section(ui, state));

                ui.add_space(6.0);
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
        });

    // Report the laid-out width (the user may have dragged the edge) so the App can
    // offset the viewport object by the panel.
    state.panel_width = panel.response.rect.width();

    // ---- 2D UV editor: a split panel on the right, when toggled on ----
    if state.show_uv_panel {
        uv_panel(ctx, state);
    }

    // ---- Layers: a separate paint.net-style palette, floating bottom-right ----
    layers_window(ctx, state);
}

/// The 2D UV editor: a resizable right-side split panel showing the unwrapped atlas
/// with the UV island wireframe overlaid, paintable with the active brush. The egui
/// closure can't touch the renderer, so the App stages the atlas image + edges into
/// `state` (version-gated) before this runs, and this emits `UvEvent`s into
/// `state.actions.uv_strokes` which the App replays onto the renderer's UV paint path.
fn uv_panel(ctx: &Context, state: &mut UiState) {
    // A fixed 50/50 split: the UV editor takes exactly half the area left of it by the
    // controls panel (whose width was just recorded into `panel_width`), so the model
    // viewport and the editor get equal halves. Not user-resizable by design.
    let editor_w = ((ctx.screen_rect().width() - state.panel_width) * 0.5).max(160.0);
    egui::SidePanel::right("uv_editor")
        .resizable(false)
        .exact_width(editor_w)
        .show(ctx, |ui| {
            ui.add_space(2.0);
            ui.heading("UV editor");

            // Upload a freshly-staged atlas to a GPU texture once per change (the App
            // restages `uv_image` only when the paint version moved), mirroring the
            // brush thumb. The handle `uv_tex` persists across frames, so we read the
            // size/guard from it — not from `uv_image`, which is consumed here.
            if let Some((w, px)) = state.uv_image.take() {
                let img = egui::ColorImage::from_rgba_unmultiplied([w as usize, w as usize], &px);
                state.uv_tex = Some(ui.ctx().load_texture(
                    "uv-atlas",
                    img,
                    // Nearest keeps the crisp, pixelated texel feel of the painter.
                    egui::TextureOptions::NEAREST,
                ));
            }

            // Nothing staged yet (a fresh mesh awaiting an unwrap): there's nothing to
            // paint on, so show a hint instead of a blank atlas.
            let (tex_id, size) = match state.uv_tex.as_ref() {
                Some(tex) => (tex.id(), tex.size()[0] as u32),
                None => {
                    ui.label(
                        egui::RichText::new("Unwrap the mesh to paint its texture here.").weak(),
                    );
                    return;
                }
            };

            // Fit the square atlas to the panel (leave a little breathing room).
            let side = ui.available_width().min(ui.available_height()).max(64.0) - 4.0;
            let sized = egui::load::SizedTexture::new(tex_id, egui::vec2(side, side));
            let resp = ui.add(egui::Image::new(sized).sense(egui::Sense::click_and_drag()));
            let rect = resp.rect;

            // Overlay the UV island wireframe. The atlas is drawn row-major top-left
            // (same indexing the brush writes texels with), so UV maps straight to the
            // rect with no V flip — the wireframe lands exactly on the painted content.
            let painter = ui.painter_at(rect);
            let edge_color = egui::Color32::from_rgba_unmultiplied(120, 200, 255, 90);
            let stroke = egui::Stroke::new(1.0, edge_color);
            let to_screen = |u: f32, v: f32| {
                egui::pos2(
                    rect.min.x + u * rect.width(),
                    rect.min.y + v * rect.height(),
                )
            };
            for &[u0, v0, u1, v1] in &state.uv_edges {
                painter.line_segment([to_screen(u0, v0), to_screen(u1, v1)], stroke);
            }

            // Map an in-panel pointer position to UV, clamped to the atlas.
            let to_uv = |p: egui::Pos2| {
                glam::Vec2::new(
                    ((p.x - rect.min.x) / rect.width()).clamp(0.0, 1.0),
                    ((p.y - rect.min.y) / rect.height()).clamp(0.0, 1.0),
                )
            };

            // Brush footprint ring at the cursor (radius in texels → panel pixels).
            if let Some(p) = resp.hover_pos() {
                let r_px = state.brush.radius / size as f32 * rect.width();
                painter.circle_stroke(
                    p,
                    r_px.max(1.0),
                    egui::Stroke::new(1.0, egui::Color32::from_white_alpha(160)),
                );
            }

            // Turn the drag into a Begin / Segment* / End sequence (one undo step).
            if resp.drag_started() {
                let uv = resp
                    .interact_pointer_pos()
                    .map(to_uv)
                    .unwrap_or(glam::Vec2::ZERO);
                state.actions.uv_strokes.push(UvEvent::Begin);
                state.uv_drag_last = Some(uv);
            }
            if resp.dragged() {
                if let Some(p) = resp.interact_pointer_pos() {
                    let uv = to_uv(p);
                    let from = state.uv_drag_last.unwrap_or(uv);
                    state
                        .actions
                        .uv_strokes
                        .push(UvEvent::Segment { from, to: uv });
                    state.uv_drag_last = Some(uv);
                }
            }
            if resp.drag_stopped() {
                state.actions.uv_strokes.push(UvEvent::End);
                state.uv_drag_last = None;
            }
            // A plain click (press+release with no drag) still lays one dab down.
            if resp.clicked() {
                if let Some(p) = resp.interact_pointer_pos().or_else(|| resp.hover_pos()) {
                    let uv = to_uv(p);
                    state.actions.uv_strokes.push(UvEvent::Begin);
                    state
                        .actions
                        .uv_strokes
                        .push(UvEvent::Segment { from: uv, to: uv });
                    state.actions.uv_strokes.push(UvEvent::End);
                }
            }
        });
}

/// Layers as a separate, paint.net-style palette floating in the bottom-right
/// corner over the viewport. The layer list is a column of large single-click
/// rows; the active layer's options (blend, opacity, mask) live in a fixed footer
/// pinned to the bottom of the palette — never inline with the rows, so the
/// controls don't jump around as you click between layers.
fn layers_window(ctx: &Context, state: &mut UiState) {
    egui::Window::new("Layers")
        .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-8.0, -8.0))
        .resizable(false)
        .collapsible(true)
        .show(ctx, |ui| {
            // Fixed width so the palette stays a tidy box; the height grows to fit.
            // (Panels-inside-a-Window don't lay out, so this is a plain top-down
            // flow: header, then a scroll-capped list, then the options footer.)
            ui.set_width(216.0);
            layers_header(ui, state);
            ui.separator();
            // Shrink to the rows when there are few; cap + scroll when there are many.
            egui::ScrollArea::vertical()
                .max_height(220.0)
                .auto_shrink([false, true])
                .show(ui, |ui| layers_list(ui, state));
            ui.separator();
            layer_options(ui, state);
        });
}

/// The palette header: the add / delete / reorder buttons. The "Layers" title is
/// the window's own title bar, so it isn't repeated here.
fn layers_header(ui: &mut egui::Ui, state: &mut UiState) {
    match button_group(
        ui,
        &[
            ("+", "Add layer"),
            ("−", "Delete layer"),
            ("↑", "Move up"),
            ("↓", "Move down"),
            ("⇊", "Merge down"),
        ],
    ) {
        Some(0) => state.actions.add_layer = true,
        Some(1) => state.actions.remove_layer = true,
        Some(2) => state.actions.move_layer_up = true,
        Some(3) => state.actions.move_layer_down = true,
        Some(4) => state.actions.merge_layer_down = true,
        _ => {}
    }
}

/// The layer rows: top layer first, each a tall full-width clickable row with a
/// visibility checkbox. No per-layer controls live here — they're in the footer.
fn layers_list(ui: &mut egui::Ui, state: &mut UiState) {
    let count = state.layers.len();
    if count == 0 {
        ui.label(
            egui::RichText::new("No layers yet — press + to add one.")
                .weak()
                .small(),
        );
        return;
    }
    // Reverse of the bottom-up storage order. The active row paints amber (the theme
    // selection colour) so it's unmistakable.
    for ui_idx in (0..count).rev() {
        let layer = state.layers[ui_idx].clone();
        let selected = ui_idx == state.active_layer;
        ui.horizontal(|ui| {
            let mut vis = layer.visible;
            if ui
                .checkbox(&mut vis, "")
                .on_hover_text("Show / hide layer")
                .changed()
            {
                state.actions.set_layer_visible = Some((ui_idx, vis));
            }
            // Dim the name of a hidden layer so visibility reads at a glance.
            let name = egui::RichText::new(&layer.name);
            let name = if layer.visible { name } else { name.weak() };
            let w = ui.available_width();
            let h = ui.spacing().interact_size.y;
            if ui
                .add_sized([w, h], egui::SelectableLabel::new(selected, name))
                .clicked()
            {
                state.actions.select_layer = Some(ui_idx);
            }
        });
    }
}

/// The active layer's options, pinned to the bottom of the palette: blend mode,
/// opacity, and mask painting. (The effect stack lives in the left panel.)
fn layer_options(ui: &mut egui::Ui, state: &mut UiState) {
    let Some(active) = state.layers.get(state.active_layer).cloned() else {
        ui.add_space(2.0);
        ui.label(
            egui::RichText::new("Select a layer to edit its options.")
                .weak()
                .small(),
        );
        return;
    };
    let i = state.active_layer;

    ui.add_space(2.0);
    // Editable name. Layers auto-name from the ops applied to them; typing here sets
    // a name by hand and locks the auto-naming. The buffer mirrors the synced name
    // while unfocused (so it tracks auto-renames and layer switches) and commits on
    // blur/Enter — committing per keystroke would fight the live re-sync.
    let resp = ui.add(
        egui::TextEdit::singleline(&mut state.layer_name_edit)
            .hint_text("Layer name")
            .desired_width(f32::INFINITY),
    );
    if resp.lost_focus() {
        let new = state.layer_name_edit.trim();
        if !new.is_empty() && new != active.name {
            state.actions.set_layer_name = Some((i, new.to_string()));
        }
    }
    if !resp.has_focus() {
        state.layer_name_edit = active.name.clone();
    }
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
    let resp = ui.add(egui::Slider::new(&mut op, 0.0..=1.0).text("Opacity"));
    // Snapshot once when the drag begins so the whole adjustment is one undo step.
    if resp.drag_started() {
        state.actions.checkpoint = true;
    }
    if resp.changed() {
        state.actions.set_layer_opacity = Some((i, op));
    }

    // Mask painting (G11): the Hide/Reveal choice only matters while masking.
    ui.checkbox(&mut state.paint_mask, "Paint mask")
        .on_hover_text("Brush into the active layer's reveal mask instead of its color");
    if state.paint_mask {
        segmented(
            ui,
            &mut state.mask_reveal,
            &[(false, "Hide", ""), (true, "Reveal", "")],
        );
    }
}

/// The active layer's non-destructive effect stack: an "Add effect" menu plus a
/// row per effect (reorder, remove, and its parameter sliders). Edits emit actions
/// targeting the active layer; the renderer re-runs the stack each composite, so
/// the painted pixels are never touched.
fn effects_section(ui: &mut egui::Ui, state: &mut UiState) {
    ui.menu_button("Add effect +", |ui| {
        for kind in EffectKind::ALL {
            if ui.button(kind.name()).clicked() {
                state.actions.add_effect = Some(kind);
                ui.close_menu();
            }
        }
    });

    // Clone the active layer's stack so we can read it while writing into
    // `state.actions` (only one slider/button changes per frame).
    let effects = state
        .layers
        .get(state.active_layer)
        .map(|l| l.effects.clone())
        .unwrap_or_default();
    if effects.is_empty() {
        ui.label(
            egui::RichText::new("No effects — adjustments are live and re-orderable")
                .weak()
                .small(),
        );
        return;
    }

    let count = effects.len();
    // Top of the stack (last to apply) shown first.
    for i in (0..count).rev() {
        let mut fx = effects[i];
        ui.group(|ui| {
            ui.label(egui::RichText::new(fx.name()).strong());
            match button_group(ui, &[("↑", "Move up"), ("↓", "Move down"), ("X", "Remove")]) {
                Some(0) => state.actions.move_effect_up = Some(i),
                Some(1) => state.actions.move_effect_down = Some(i),
                Some(2) => state.actions.remove_effect = Some(i),
                _ => {}
            }

            let mut changed = false;
            let mut drag_started = false;
            let mut slider = |ui: &mut egui::Ui, v: &mut f32, range, text| {
                let resp = ui.add(egui::Slider::new(v, range).text(text));
                changed |= resp.changed();
                drag_started |= resp.drag_started();
            };
            match &mut fx {
                Effect::HueSatLight { hue, sat, light } => {
                    slider(ui, hue, -180.0..=180.0, "Hue");
                    slider(ui, sat, -1.0..=1.0, "Saturation");
                    slider(ui, light, -1.0..=1.0, "Lightness");
                }
                Effect::BrightnessContrast {
                    brightness,
                    contrast,
                } => {
                    slider(ui, brightness, -1.0..=1.0, "Brightness");
                    slider(ui, contrast, -1.0..=1.0, "Contrast");
                }
                Effect::Blur { radius } => {
                    slider(ui, radius, 0.0..=16.0, "Radius");
                }
                Effect::Warp { amount, scale } => {
                    slider(ui, amount, 0.0..=16.0, "Amount");
                    slider(ui, scale, 1.0..=24.0, "Scale");
                }
            }
            // One checkpoint at drag start collapses the whole drag into a single
            // undo step (same pattern as layer opacity).
            if drag_started {
                state.actions.checkpoint = true;
            }
            if changed {
                state.actions.set_effect = Some((i, fx));
            }
        });
    }
}

/// Mesh effects: baked AO + curvature + sun light driving layers and masks. Grouped
/// into three subsections — Looks (one-click presets), Tune (the shared Levels/noise
/// that shape every result), and Custom (route a Source channel into a tint, mask, or
/// gradient). The occasional controls — sun direction, noise character, the gradient
/// ramp — sit behind nested disclosures so the column stays scannable.
fn mesh_effects_section(ui: &mut egui::Ui, state: &mut UiState) {
    ui.label(
        egui::RichText::new("Wear, dirt & shadow from the mesh's own geometry.")
            .weak()
            .small(),
    );

    // ---- Looks: one-click effects that each drop a new layer, honoring the Tune
    // controls below. Levels/noise are captured here (start-of-frame state) so the
    // presets fire correctly even while Tune is mid-edit or collapsed.
    let levels = state.levels();
    let noise = state.noise_mod();
    heading(ui, "Looks");
    match button_group(
        ui,
        &[
            ("Darken (AO)", "Shadow in crevices"),
            ("Highlights", "Brighten convex edges"),
        ],
    ) {
        Some(0) => state.actions.apply_ao = Some((levels, noise)),
        Some(1) => state.actions.apply_highlight = Some((levels, noise)),
        _ => {}
    }
    match button_group(
        ui,
        &[
            ("Dirt", "Dark grime settling into cavities"),
            ("Edge wear", "Worn, lightened convex edges"),
        ],
    ) {
        Some(0) => state.actions.apply_dirt = Some((levels, noise)),
        Some(1) => state.actions.apply_edge_wear = Some((levels, noise)),
        _ => {}
    }
    // Directional light ("sun"): "Sunlight" lights the faces turned toward the sun;
    // "Top-down" aims it straight up and drops pale dust on the upward faces.
    match button_group(
        ui,
        &[
            ("Sunlight", "Warm highlight on the lit faces"),
            ("Top-down", "Pale dust on upward-facing surfaces"),
        ],
    ) {
        Some(0) => state.actions.apply_sun = Some((levels, noise)),
        Some(1) => {
            // Aim the sun straight up; app.rs pushes this before applying the dust.
            state.sun_elevation = 90.0;
            state.actions.apply_top_dust = Some((levels, noise));
        }
        _ => {}
    }
    // Sun direction shapes the two light looks above (and the Light source in Custom);
    // tucked behind a disclosure since it's irrelevant to the wear/dirt presets.
    egui::CollapsingHeader::new("Sun direction")
        .default_open(false)
        .show(ui, |ui| {
            ui.add(
                egui::Slider::new(&mut state.sun_elevation, 0.0..=90.0)
                    .text("Sun height")
                    .suffix("°"),
            );
            ui.add(
                egui::Slider::new(&mut state.sun_azimuth, 0.0..=360.0)
                    .text("Sun angle")
                    .suffix("°"),
            );
            ui.checkbox(&mut state.sun_shadow, "Cast shadows");
        });

    // ---- Tune: the shared Levels remap and noise "break up" multiplied into every
    // Look above and every Custom route below. Strength/Contrast/Invert/Break up stay
    // inline (the most-touched knobs); the noise character sits behind a disclosure.
    heading(ui, "Tune");
    ui.add(egui::Slider::new(&mut state.ao_strength, 0.0..=1.0).text("Strength"));
    ui.add(egui::Slider::new(&mut state.effect_contrast, 0.0..=1.0).text("Contrast"));
    ui.checkbox(&mut state.effect_invert, "Invert");
    // Break up (noise): a procedural modifier multiplied into whatever effect you
    // apply, so wear/dirt lands in patches instead of a perfect ring — and, with the
    // "Surface" source, becomes pure grunge. Amount 0 leaves it clean.
    ui.add(
        egui::Slider::new(&mut state.noise_amount, 0.0..=1.0)
            .text("Break up")
            .custom_formatter(|v, _| {
                if v <= 0.0 {
                    "off".to_string()
                } else {
                    format!("{v:.2}")
                }
            }),
    );
    // Noise character (only bites when Break up > 0): which pattern, how big, how hard.
    egui::CollapsingHeader::new("Noise detail")
        .default_open(false)
        .show(ui, |ui| {
            egui::ComboBox::from_label("Pattern")
                .selected_text(state.noise_kind.name())
                .show_ui(ui, |ui| {
                    for kind in NoiseKind::ALL {
                        ui.selectable_value(&mut state.noise_kind, kind, kind.name());
                    }
                });
            ui.add(egui::Slider::new(&mut state.noise_scale, 1.0..=24.0).text("Noise scale"));
            ui.add(egui::Slider::new(&mut state.noise_contrast, 0.0..=1.0).text("Noise contrast"));
        });

    // ---- Custom: route any baked Source channel into a tint layer, the active
    // layer's mask, or a gradient ramp. Recompute levels/noise here so these honor
    // edits made in Tune this same frame.
    let levels = state.levels();
    let noise = state.noise_mod();
    heading(ui, "Custom");
    egui::ComboBox::from_label("Source")
        .selected_text(state.effect_source.name())
        .show_ui(ui, |ui| {
            for src in MapSource::ALL {
                ui.selectable_value(&mut state.effect_source, src, src.name());
            }
        });
    match button_group(
        ui,
        &[
            (
                "Tint layer",
                "New layer: the Color above, masked to the source",
            ),
            ("Mask layer", "Set the active layer's mask from the source"),
        ],
    ) {
        Some(0) => {
            state.actions.apply_tint = Some((state.effect_source, levels, state.brush.color, noise))
        }
        Some(1) => state.actions.mask_from_map = Some((state.effect_source, levels, noise)),
        _ => {}
    }
    // Gradient map: color the source's value through a low→high ramp, so one channel
    // reads as a full material across the surface — dark crevices → bright tops, or a
    // lit→shaded ramp from the sun.
    egui::CollapsingHeader::new("Gradient")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Colors");
                ui.color_edit_button_rgb(&mut state.grad_low);
                ui.label("→");
                ui.color_edit_button_rgb(&mut state.grad_high);
            });
            if ui
                .button("Gradient layer")
                .on_hover_text("New layer: the source's value mapped through the ramp")
                .clicked()
            {
                state.actions.apply_gradient = Some((
                    state.effect_source,
                    levels,
                    state.grad_low,
                    state.grad_high,
                    noise,
                ));
            }
        });
}

/// A wrapping grid of swatches for the textures found in the chosen brush folder.
/// Clicking a swatch loads that image as the brush (the App reads `use_brush_entry`).
fn brush_folder_grid(ui: &mut egui::Ui, state: &mut UiState) {
    if let Some(p) = folder_grid(ui, &mut state.brush_folder_entries, "brush-folder") {
        state.actions.use_brush_entry = Some(p);
    }
}

/// A wrapping grid of swatches for the alpha tips found in the chosen tip folder.
/// Clicking a swatch loads that image as the brush tip (the App reads `use_alpha_entry`).
fn alpha_folder_grid(ui: &mut egui::Ui, state: &mut UiState) {
    if let Some(p) = folder_grid(ui, &mut state.alpha_folder_entries, "alpha-folder") {
        state.actions.use_alpha_entry = Some(p);
    }
}

/// Shared wrapping grid of folder swatches. Each freshly-staged thumbnail is uploaded to
/// a GPU texture once (`key_prefix` namespaces the handle so the two browsers don't
/// collide); returns the path of the swatch clicked this frame, if any.
fn folder_grid(
    ui: &mut egui::Ui,
    entries: &mut [BrushEntry],
    key_prefix: &str,
) -> Option<std::path::PathBuf> {
    if entries.is_empty() {
        return None;
    }
    let mut clicked: Option<std::path::PathBuf> = None;
    egui::ScrollArea::vertical()
        .max_height(160.0)
        .id_salt(key_prefix)
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                for entry in entries.iter_mut() {
                    // Upload this entry's staged thumbnail to a GPU texture once.
                    if let Some((w, h, px)) = entry.thumb.take() {
                        let img =
                            egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &px);
                        entry.tex = Some(ui.ctx().load_texture(
                            format!("{key_prefix}-{}", entry.name),
                            img,
                            egui::TextureOptions::LINEAR,
                        ));
                    }
                    if let Some(tex) = &entry.tex {
                        let [tw, th] = tex.size();
                        let scale = 40.0 / tw.max(th) as f32;
                        let size = egui::vec2(tw as f32 * scale, th as f32 * scale);
                        let img = egui::load::SizedTexture::new(tex.id(), size);
                        if ui
                            .add(egui::ImageButton::new(img))
                            .on_hover_text(&entry.name)
                            .clicked()
                        {
                            clicked = Some(entry.path.clone());
                        }
                    }
                }
            });
        });
    clicked
}

/// Fill the active layer with a tiled material image (brick, moss…).
fn material_section(ui: &mut egui::Ui, state: &mut UiState) {
    ui.label(
        egui::RichText::new(
            "Fill the active layer with an image (brick, moss…). Mask it for crevice/edge detail.",
        )
        .weak()
        .small(),
    );
    ui.add(egui::Slider::new(&mut state.material_tile, 1.0..=16.0).text("Tile"));
    if ui.button("Fill with image…").clicked() {
        state.actions.fill_material = Some(state.material_tile);
    }
}

/// Canvas resolution and the engine-target hint that shapes PNG export naming. The
/// open/save/export *actions* themselves live in the File menu; this section holds
/// only the persistent settings they read.
fn export_section(ui: &mut egui::Ui, state: &mut UiState) {
    // The unwrap (Mesh → Unwrap UVs) derives this to hold a constant world-space
    // texel size, but it can be overridden here — handy for a loaded OBJ whose
    // unwrap-chosen size isn't what you want. Picking a size resamples the existing
    // paint into it (undoable); the next unwrap reasserts the density-derived size.
    const SIZES: [u32; 9] = [32, 64, 128, 256, 512, 1024, 2048, 3072, 4096];
    egui::ComboBox::from_label("Texture")
        .selected_text(format!("{0}×{0}", state.resolution))
        .show_ui(ui, |ui| {
            for size in SIZES {
                if ui
                    .selectable_label(size == state.resolution, format!("{size}×{size}"))
                    .clicked()
                {
                    state.actions.set_resolution = Some(size);
                }
            }
        });
    let hint = if state.last_atlas_clamped {
        "Unwrap sets this (clamped to GPU max); override above"
    } else {
        "Unwrap sets this; override above"
    };
    ui.label(egui::RichText::new(hint).weak().small());
    // Achieved texel density from the last unwrap, so a requested "N texels/m" can be
    // confirmed (0 before any unwrap this session).
    if state.last_density_d > 0.0 {
        ui.label(
            egui::RichText::new(format!("≈ {:.0} texels / m", state.last_density_d))
                .weak()
                .small(),
        );
    }

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
    ui.label(
        egui::RichText::new("Open / Save PNG / Export live in the File menu.")
            .weak()
            .small(),
    );
}

/// Palette quantization (the PSX look) plus built-in / from-image palettes.
fn palette_section(ui: &mut egui::Ui, state: &mut UiState) {
    ui.checkbox(&mut state.palette.enabled, "Quantize");
    // Dither only does anything while quantizing, so it appears only then. Shown
    // inline (not indented) to keep the section flat — no nested layout.
    if state.palette.enabled {
        ui.checkbox(&mut state.palette.dither, "Dither");
        ui.add(egui::Slider::new(&mut state.palette.dither_strength, 0.0..=0.3).text("Dither amt"));
    }

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
}

/// Draw a wrapping row of small color swatches.
fn swatches(ui: &mut egui::Ui, colors: &[[f32; 3]]) {
    let size = egui::vec2(12.0, 12.0);
    ui.horizontal_wrapped(|ui| {
        for c in colors {
            let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
            let color = egui::Color32::from_rgb(
                (c[0] * 255.0) as u8,
                (c[1] * 255.0) as u8,
                (c[2] * 255.0) as u8,
            );
            ui.painter().rect_filled(rect, 0.0, color);
        }
    });
}

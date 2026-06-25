// src/app.rs
//
// The application shell: owns the window, the renderer, egui state, and input
// state. Implements winit's ApplicationHandler trait (the modern 0.30+ event
// API).
//
// Event routing: every window event is first fed to egui. Pointer presses that
// egui consumes (clicks on the panel) never start a paint/camera drag, so UI
// hover and widgets don't paint through to the mesh. Once a drag has begun on
// the viewport, it continues until release regardless of where the cursor goes.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use egui::ViewportId;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow},
    keyboard::{KeyCode, ModifiersState, PhysicalKey},
    window::{Window, WindowId},
};

use crate::mesh::Mesh;
use crate::renderer::{Renderer, UiPaint};
use crate::ui::{self, UiState};

/// What the current mouse drag is doing. LMB paints; RMB orbits; MMB pans — so
/// camera control and painting never fight over the same button.
#[derive(Clone, Copy, PartialEq)]
enum Drag {
    None,
    Paint,
    Orbit,
    Pan,
}

/// How often a timed autosave fires, and how many rolling version files it keeps
/// (G31). At PSX texture sizes each version is a few hundred KB, so a ring of ten
/// recovery points costs only a handful of MB on disk.
const AUTOSAVE_INTERVAL: Duration = Duration::from_secs(5 * 60);
const AUTOSAVE_VERSIONS: u32 = 10;

pub struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    // The mesh to paint, taken when the renderer is built on resume.
    mesh: Option<Mesh>,

    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    ui: UiState,

    mouse_pos: (f32, f32),
    last_pos: (f32, f32),
    drag: Drag,
    /// Latest keyboard modifier state, tracked for the undo/redo shortcuts.
    modifiers: ModifiersState,

    /// Save/exit/autosave state (G31). `project_path` is the `.lowtex` file the user
    /// is editing — `None` until they save or open one; quicksave writes here.
    /// `last_autosave` paces the timed recovery write; `autosave_counter` cycles the
    /// rolling version files. `window_title` mirrors the last title we set so we
    /// only call `set_title` when the name or the unsaved dot actually changes.
    project_path: Option<PathBuf>,
    last_autosave: Instant,
    autosave_counter: u32,
    window_title: String,
    /// Save shortcuts (⌘/Ctrl+S, ⌘/Ctrl+Shift+S) captured during key events. They
    /// can't go through `ui.actions` because `ui::build` resets that at the top of
    /// each frame, before `handle_ui_actions` reads it; these App-owned flags
    /// survive the reset and are merged in there.
    pending_quicksave: bool,
    pending_save_as: bool,

    /// Persistent app settings (e.g. the last texture folder), saved across launches.
    config: crate::config::Config,

    /// On-demand rendering: the next time egui has asked to repaint itself (a
    /// running animation, a blinking text cursor, a hover fade). `None` means the
    /// UI is static and the loop can sleep until the next input event. We never
    /// redraw on a fixed cadence — that idle full-frame loop pegged the GPU/CPU on
    /// Windows.
    repaint_at: Option<Instant>,
}

impl App {
    pub fn new(mesh: Mesh) -> Self {
        // Restore the last-used texture folder so the brush browser reopens where the
        // user left off. Scan it now (best-effort) so its thumbnails are ready; a
        // folder that has since moved/been deleted is silently dropped.
        let config = crate::config::Config::load();
        let mut ui = UiState::default();
        if let Some(dir) = config.last_texture_folder.as_ref().filter(|d| d.is_dir()) {
            ui.brush_folder_entries = scan_brush_folder(dir);
            ui.brush_folder = Some(dir.clone());
        }

        Self {
            window: None,
            renderer: None,
            mesh: Some(mesh),
            egui_ctx: egui::Context::default(),
            egui_state: None,
            ui,
            mouse_pos: (0.0, 0.0),
            last_pos: (0.0, 0.0),
            drag: Drag::None,
            modifiers: ModifiersState::empty(),
            project_path: None,
            last_autosave: Instant::now(),
            autosave_counter: 0,
            window_title: String::new(),
            pending_quicksave: false,
            pending_save_as: false,
            config,
            repaint_at: None,
        }
    }

    /// Run egui for this frame and render the scene + overlay.
    fn redraw(&mut self) {
        let Some(window) = self.window.clone() else {
            return;
        };
        let Some(state) = self.egui_state.as_mut() else {
            return;
        };
        let raw_input = state.take_egui_input(&window);

        // Stage the 2D UV editor's inputs into UiState before the egui closure runs —
        // the closure only sees UiState, never the renderer. Gated on the renderer's
        // version counters so the ~MBs of atlas pixels are copied only when they
        // actually changed and the panel is open; the wireframe rebuilds only on a
        // topology change.
        if self.ui.show_uv_panel {
            if let Some(renderer) = self.renderer.as_ref() {
                if renderer.mesh_has_uvs() {
                    let pv = renderer.paint_version();
                    if pv != self.ui.uv_image_version {
                        let (size, px) = renderer.atlas_view();
                        self.ui.uv_image = Some((size, px.to_vec()));
                        self.ui.uv_image_version = pv;
                    }
                    let tv = renderer.topo_version();
                    if tv != self.ui.uv_edges_version {
                        self.ui.uv_edges = renderer.build_uv_edges();
                        self.ui.uv_edges_version = tv;
                    }
                } else {
                    // No UVs yet (mesh awaits an unwrap): drop any stale atlas so the
                    // panel shows its hint, and force a restage once UVs exist.
                    self.ui.uv_image = None;
                    self.ui.uv_tex = None;
                    self.ui.uv_image_version = u64::MAX;
                    self.ui.uv_edges_version = u64::MAX;
                }
            }
        }

        let full_output = self
            .egui_ctx
            .run(raw_input, |ctx| ui::build(ctx, &mut self.ui));

        // On-demand rendering: schedule the next frame only when egui asks for one.
        // A zero delay means "keep animating" (draw again next turn); a finite delay
        // (e.g. the text-cursor blink) becomes a timed wake; the steady-state
        // Duration::MAX leaves the loop asleep until the next input event.
        let repaint_delay = full_output
            .viewport_output
            .get(&ViewportId::ROOT)
            .map_or(Duration::MAX, |v| v.repaint_delay);
        self.repaint_at = if repaint_delay.is_zero() {
            window.request_redraw();
            None
        } else {
            Instant::now().checked_add(repaint_delay)
        };

        self.egui_state
            .as_mut()
            .unwrap()
            .handle_platform_output(&window, full_output.platform_output);
        let jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        self.handle_ui_actions();

        if let Some(renderer) = self.renderer.as_mut() {
            // Coalesce this frame's accumulated paint into one region upload before drawing.
            renderer.flush_paint();
            renderer.render(Some(UiPaint {
                jobs: &jobs,
                textures_delta: &full_output.textures_delta,
                pixels_per_point: full_output.pixels_per_point,
            }));
        }
    }

    /// Apply this frame's UI requests (file dialogs, resolution change). Runs
    /// outside the egui closure so native dialogs can block safely.
    fn handle_ui_actions(&mut self) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        // Push the latest palette settings (cheap; re-quantizes the texture).
        renderer.set_palette_settings(self.ui.palette);

        // Push viewport display settings (G29): background color + grid toggle.
        // The compass is always shown (and its axes are clickable), so no toggle.
        renderer.set_bg_color(self.ui.bg_color);
        renderer.set_show_grid(self.ui.show_grid);

        // Push the texture filter mode (G30); cheap — rebuilds the sampler only on change.
        renderer.set_texture_filter(self.ui.texture_filter);

        // Push the directional-light sun before the mesh-effect actions below, so a
        // Light effect applied this frame bakes against the current sun (the Top-down
        // button, e.g., aims the sun up this same frame).
        renderer.set_sun(self.ui.sun_dir(), self.ui.sun_shadow);

        // Reserve the panel's strip on the left so the scene renders (and picks) in
        // the region beside it. The panel width is in egui points; scale to the
        // physical pixels the renderer's viewport works in.
        let offset_px = self.ui.panel_width * self.egui_ctx.pixels_per_point();
        renderer.set_view_offset(offset_px);

        // Push the paint target (color vs mask) + mask polarity (G11).
        renderer.set_paint_target(if self.ui.paint_mask {
            crate::renderer::PaintTarget::Mask
        } else {
            crate::renderer::PaintTarget::Color
        });
        renderer.set_mask_reveal(self.ui.mask_reveal);

        // Brush-image state: how it's applied (Brush=tiled vs Stamp=decal), the tiling
        // factor, and the stamp's rotation (degrees → radians) + tint-to-swatch.
        renderer.set_brush_image_mode(if self.ui.brush_stamp {
            crate::renderer::BrushImageMode::Stamp
        } else {
            crate::renderer::BrushImageMode::Tiled
        });
        renderer.set_brush_tile(self.ui.brush_tile);
        renderer.set_stamp_options(self.ui.stamp_angle_deg.to_radians(), self.ui.stamp_tint);

        // Push mirror-painting state: the chosen axis when enabled, else off.
        renderer.set_symmetry(self.ui.symmetry_on.then_some(self.ui.symmetry_axis));

        // Push face-lock state: confine each dab to the face it lands on.
        renderer.set_lock_face(self.ui.lock_face);

        // Outline the active face while a painting tool has the lock on, so the user
        // sees exactly where the brush is confined. Cleared (None) otherwise.
        let painting_tool = matches!(self.ui.tool, crate::ui::Tool::Brush);
        let outline_cursor = (self.ui.lock_face && painting_tool).then_some(self.mouse_pos);
        renderer.set_face_outline(outline_cursor);

        // Show the brush footprint ring at the cursor while a painting tool is active.
        renderer.set_brush_cursor(painting_tool.then_some(self.mouse_pos), &self.ui.brush);

        // Ghost-preview the stamp under the cursor for the solid/image brush, so the
        // painter sees what a click lays down before committing.
        let preview_cursor = (self.ui.tool == crate::ui::Tool::Brush).then_some(self.mouse_pos);
        renderer.set_brush_preview(preview_cursor, &self.ui.brush);

        let actions = std::mem::take(&mut self.ui.actions);

        // Replay this frame's 2D UV-editor strokes onto the renderer's UV paint path.
        // Begin/End bracket each stroke into one undo step (exactly like a 3D stroke);
        // each Segment paints a disc-interpolated line directly in texel space.
        for ev in &actions.uv_strokes {
            match *ev {
                crate::ui::UvEvent::Begin => renderer.begin_stroke(),
                crate::ui::UvEvent::Segment { from, to } => {
                    renderer.paint_uv_segment(from, to, &self.ui.brush)
                }
                crate::ui::UvEvent::End => renderer.end_stroke(),
            }
        }

        if actions.open_model {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Mesh", &["gltf", "glb", "obj"])
                .pick_file()
            {
                match renderer.load_model(&path.to_string_lossy()) {
                    Ok(()) => log::info!("loaded model {}", path.display()),
                    Err(e) => log::error!("{e}"),
                }
            }
        }
        // Project save (G31): quicksave writes straight to the open file; Save As
        // always asks. A quicksave with no file chosen yet falls back to Save As.
        // Merge the menu's one-shots with the keyboard shortcuts stashed on App
        // (those are raised outside the egui frame, so they bypass `ui.actions`).
        let want_quicksave = actions.quicksave || std::mem::take(&mut self.pending_quicksave);
        let want_save_as = actions.save_project
            || std::mem::take(&mut self.pending_save_as)
            || (want_quicksave && self.project_path.is_none());
        if want_quicksave {
            if let Some(path) = self.project_path.clone() {
                match renderer.save_project(&path.to_string_lossy()) {
                    Ok(()) => log::info!("saved project {}", path.display()),
                    Err(e) => log::error!("{e}"),
                }
            }
        }
        if want_save_as {
            let suggested = self
                .project_path
                .as_ref()
                .and_then(|p| p.file_name())
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "untitled.lowtex".to_string());
            if let Some(path) = rfd::FileDialog::new()
                .set_file_name(suggested)
                .add_filter("lowtex project", &["lowtex"])
                .save_file()
            {
                match renderer.save_project(&path.to_string_lossy()) {
                    Ok(()) => {
                        log::info!("saved project {}", path.display());
                        self.project_path = Some(path);
                    }
                    Err(e) => log::error!("{e}"),
                }
            }
        }
        if actions.open_project {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("lowtex project", &["lowtex"])
                .pick_file()
            {
                match renderer.load_project(&path.to_string_lossy()) {
                    Ok(()) => {
                        log::info!("opened project {}", path.display());
                        self.project_path = Some(path);
                        // Reopen the brush browser on the folder the project recorded.
                        // A folder that no longer exists keeps its path (shown in the
                        // header) but loads no thumbnails.
                        if let Some(dir) = renderer.texture_folder().map(PathBuf::from) {
                            if dir.is_dir() {
                                self.ui.brush_folder_entries = scan_brush_folder(&dir);
                                self.config.last_texture_folder = Some(dir.clone());
                                self.config.save();
                            } else {
                                self.ui.brush_folder_entries.clear();
                                log::warn!("texture folder {} not found", dir.display());
                            }
                            self.ui.brush_folder = Some(dir);
                        }
                    }
                    Err(e) => log::error!("{e}"),
                }
            }
        }
        if let Some(density) = actions.unwrap {
            let (atlas, clamped, d) = renderer.apply_unwrap(density, self.ui.unwrap_overlap);
            self.ui.last_atlas_clamped = clamped;
            self.ui.last_density_d = d;
            log::info!(
                "unwrapped → {atlas}×{atlas} at {d:.1} texels/unit{}",
                if clamped {
                    " (density clamped to GPU max)"
                } else {
                    ""
                }
            );
        }
        if let Some(texels_per_m) = actions.unwrap_at_density {
            let (atlas, clamped, d) =
                renderer.apply_unwrap_at_density(texels_per_m, self.ui.unwrap_overlap);
            self.ui.last_atlas_clamped = clamped;
            self.ui.last_density_d = d;
            log::info!(
                "unwrapped → {atlas}×{atlas} at {d:.1} texels/unit{}",
                if clamped {
                    " (density clamped to GPU max)"
                } else {
                    ""
                }
            );
        }
        if let Some(size) = actions.set_resolution {
            // Manual override of the unwrap-derived size; resamples the paint into it.
            // UVs are unchanged, so texels-per-unit scales with the atlas size — keep
            // the readout honest by scaling it (using the actually-applied, clamped size).
            let old = renderer.texture_resolution();
            renderer.set_texture_resolution(size);
            let new = renderer.texture_resolution();
            if old > 0 {
                self.ui.last_density_d *= new as f32 / old as f32;
            }
            log::info!("texture resolution → {new}×{new}");
        }
        if let Some(tile) = actions.fill_material {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Image", &["png", "jpg", "jpeg"])
                .pick_file()
            {
                match renderer.fill_active_with_material(&path.to_string_lossy(), tile) {
                    Ok(()) => log::info!("filled layer with material {}", path.display()),
                    Err(e) => log::error!("{e}"),
                }
            }
        }
        if actions.load_brush_image {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Image", &["png", "jpg", "jpeg"])
                .pick_file()
            {
                match renderer.load_brush_material(&path.to_string_lossy()) {
                    Ok(()) => {
                        self.ui.brush_image_loaded = true;
                        // Stage a preview for the panel swatch (uploaded by the UI).
                        self.ui.brush_thumb = renderer.brush_thumbnail(64);
                        log::info!("loaded brush image {}", path.display());
                    }
                    Err(e) => log::error!("{e}"),
                }
            }
        }
        if actions.clear_brush_image {
            renderer.clear_brush_material();
            self.ui.brush_image_loaded = false;
            self.ui.brush_thumb = None;
            self.ui.brush_thumb_tex = None;
        }
        // Open a folder of textures and stage a thumbnail for each image in it, for the
        // brush/stamp browser. Capped so a huge folder can't stall the load.
        if actions.open_brush_folder {
            // Reopen the dialog on the last folder so picking a sibling is one step.
            let mut dialog = rfd::FileDialog::new();
            if let Some(prev) = self.ui.brush_folder.as_ref().filter(|d| d.is_dir()) {
                dialog = dialog.set_directory(prev);
            }
            if let Some(dir) = dialog.pick_folder() {
                self.ui.brush_folder_entries = scan_brush_folder(&dir);
                log::info!(
                    "brush folder {} — {} textures",
                    dir.display(),
                    self.ui.brush_folder_entries.len()
                );
                // Record it on the document (for the next save) and persist it as the
                // last-used folder so it's restored on the next launch.
                renderer.set_texture_folder(Some(dir.to_string_lossy().into_owned()));
                self.config.last_texture_folder = Some(dir.clone());
                self.config.save();
                self.ui.brush_folder = Some(dir);
            }
        }
        // Use a folder texture as the brush image (keeps the current Tile/Stamp mode).
        if let Some(path) = actions.use_brush_entry {
            match renderer.load_brush_material(&path.to_string_lossy()) {
                Ok(()) => {
                    self.ui.brush_image_loaded = true;
                    self.ui.brush_thumb = renderer.brush_thumbnail(64);
                    self.ui.brush_thumb_tex = None;
                    log::info!("brush image {}", path.display());
                }
                Err(e) => log::error!("{e}"),
            }
        }
        if let Some(indexed) = actions.export_png {
            let preset = self.ui.export_preset;
            if let Some(path) = rfd::FileDialog::new()
                .set_file_name(preset.suggested_filename())
                .add_filter("PNG", &["png"])
                .save_file()
            {
                match renderer.export_png(&path.to_string_lossy(), indexed) {
                    Ok(()) => {
                        log::info!("exported {} — {}", path.display(), preset.import_hint())
                    }
                    Err(e) => log::error!("{e}"),
                }
            }
        }
        if let Some(i) = actions.select_builtin_palette {
            if let Some(p) = crate::palette::Palette::builtins().into_iter().nth(i) {
                renderer.set_palette(p);
            }
        }
        if let Some(n) = actions.generate_palette {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Image", &["png", "jpg", "jpeg"])
                .pick_file()
            {
                match renderer.generate_palette_from_image(&path.to_string_lossy(), n) {
                    Ok(()) => log::info!("generated {n}-color palette from {}", path.display()),
                    Err(e) => log::error!("{e}"),
                }
            }
        }

        if actions.save_png {
            if let Some(path) = rfd::FileDialog::new()
                .set_file_name("texture.png")
                .add_filter("PNG", &["png"])
                .save_file()
            {
                match renderer.save_texture_png(&path.to_string_lossy()) {
                    Ok(()) => log::info!("saved {}", path.display()),
                    Err(e) => log::error!("{e}"),
                }
            }
        }
        if actions.open_texture {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Image", &["png", "jpg", "jpeg"])
                .pick_file()
            {
                match renderer.load_texture_png(&path.to_string_lossy()) {
                    Ok(()) => log::info!("loaded {}", path.display()),
                    Err(e) => log::error!("{e}"),
                }
            }
        }

        // Undo / redo. The checkpoint here runs before set_layer_opacity below so
        // it captures the pre-drag opacity (an opacity-slider drag emits checkpoint
        // on its first frame).
        if actions.undo {
            renderer.undo();
        }
        if actions.redo {
            renderer.redo();
        }
        if actions.checkpoint {
            renderer.checkpoint();
        }

        // Layer ops (G10).
        if actions.add_layer {
            renderer.add_layer();
        }
        if actions.remove_layer {
            renderer.remove_active_layer();
        }
        if actions.move_layer_up {
            renderer.move_active_layer(true);
        }
        if actions.move_layer_down {
            renderer.move_active_layer(false);
        }
        if actions.merge_layer_down {
            renderer.merge_active_layer_down();
        }
        if let Some(i) = actions.select_layer {
            renderer.set_active_layer(i);
        }
        if let Some((i, v)) = actions.set_layer_visible {
            renderer.set_layer_visible(i, v);
        }
        if let Some((i, o)) = actions.set_layer_opacity {
            renderer.set_layer_opacity(i, o);
        }
        if let Some((i, b)) = actions.set_layer_blend {
            renderer.set_layer_blend(i, b);
        }
        if let Some((i, name)) = actions.set_layer_name {
            renderer.rename_layer(i, name);
        }

        // Per-layer effects (G28) — all target the active layer.
        if let Some(kind) = actions.add_effect {
            renderer.add_effect(kind);
        }
        if let Some(idx) = actions.remove_effect {
            renderer.remove_effect(idx);
        }
        if let Some(idx) = actions.move_effect_up {
            renderer.move_effect(idx, true);
        }
        if let Some(idx) = actions.move_effect_down {
            renderer.move_effect(idx, false);
        }
        if let Some((idx, fx)) = actions.set_effect {
            renderer.set_effect(idx, fx);
        }

        // Mesh effects — bake mesh maps (cached) then add a generated layer or fill
        // the active layer's mask from a baked map.
        if let Some((lv, noise)) = actions.apply_ao {
            renderer.apply_ao_layer(lv, noise);
        }
        if let Some((lv, noise)) = actions.apply_highlight {
            renderer.apply_highlight_layer(lv, noise);
        }
        if let Some((lv, noise)) = actions.apply_dirt {
            renderer.apply_dirt_layer(lv, noise);
        }
        if let Some((lv, noise)) = actions.apply_edge_wear {
            renderer.apply_edge_wear_layer(lv, noise);
        }
        if let Some((src, lv, color, noise)) = actions.apply_tint {
            let rgb = [
                (color[0] * 255.0).round() as u8,
                (color[1] * 255.0).round() as u8,
                (color[2] * 255.0).round() as u8,
            ];
            renderer.add_map_layer(
                "Tint",
                src,
                lv,
                rgb,
                crate::layers::BlendMode::Normal,
                noise,
            );
        }
        if let Some((src, lv, noise)) = actions.mask_from_map {
            renderer.fill_active_mask_from_map(src, lv, noise);
        }
        // Directional-light effects: a warm highlight on the lit faces, and the
        // top-down dust look (the UI has already aimed the sun straight up).
        if let Some((lv, noise)) = actions.apply_sun {
            renderer.add_map_layer(
                "Sunlight",
                crate::bake::MapSource::Light,
                lv,
                [255, 244, 214],
                crate::layers::BlendMode::Screen,
                noise,
            );
        }
        if let Some((lv, noise)) = actions.apply_top_dust {
            renderer.add_map_layer(
                "Top dust",
                crate::bake::MapSource::Light,
                lv,
                [188, 180, 160],
                crate::layers::BlendMode::Normal,
                noise,
            );
        }
        if let Some((src, lv, low, high, noise)) = actions.apply_gradient {
            let to_u8 = |c: [f32; 3]| {
                [
                    (c[0] * 255.0).round() as u8,
                    (c[1] * 255.0).round() as u8,
                    (c[2] * 255.0).round() as u8,
                ]
            };
            renderer.add_gradient_layer(
                "Gradient",
                src,
                lv,
                crate::bake::Gradient {
                    low: to_u8(low),
                    high: to_u8(high),
                },
                crate::layers::BlendMode::Normal,
                noise,
            );
        }

        // Keep the UI mirrors in sync with the renderer.
        self.ui.resolution = renderer.texture_resolution();
        self.ui.can_undo = renderer.can_undo();
        self.ui.can_redo = renderer.can_redo();
        self.ui.palette_swatches = renderer.palette().colors.clone();
        self.ui.active_layer = renderer.layers().active;
        self.ui.layers = renderer
            .layers()
            .layers
            .iter()
            .map(|l| crate::ui::LayerInfo {
                name: l.name.clone(),
                visible: l.visible,
                opacity: l.opacity,
                blend: l.blend,
                effects: l.effects.clone(),
            })
            .collect();

        // Reflect the open file name + an unsaved-changes dot in the window title
        // (G31). Only call `set_title` when the string actually changes, so we don't
        // hand the OS a fresh title every frame.
        let name = self
            .project_path
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "untitled".to_string());
        let title = format!(
            "{}{} — lowtex",
            if renderer.is_dirty() { "• " } else { "" },
            name
        );
        if title != self.window_title {
            if let Some(window) = self.window.as_ref() {
                window.set_title(&title);
            }
            self.window_title = title;
        }
    }

    /// The destination for the next autosave version (G31): a rolling ring of files
    /// so successive autosaves don't grow without bound. When a project file is
    /// open, versions sit beside it (`sketch.lowtex` → `sketch.autosave3.lowtex`) so
    /// recovery is obvious; for an untitled session they go to a `lowtex-autosave`
    /// folder under the system temp dir.
    fn autosave_path(&self) -> PathBuf {
        let n = self.autosave_counter % AUTOSAVE_VERSIONS;
        match &self.project_path {
            Some(p) => {
                let stem = p
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "untitled".to_string());
                p.with_file_name(format!("{stem}.autosave{n}.lowtex"))
            }
            None => {
                let dir = std::env::temp_dir().join("lowtex-autosave");
                let _ = std::fs::create_dir_all(&dir);
                dir.join(format!("untitled.autosave{n}.lowtex"))
            }
        }
    }

    /// Write a timed recovery version if the interval has elapsed and the document
    /// has changed since the last save/autosave. Called every wake from
    /// `about_to_wait`; cheap (a clock check) until both conditions hold.
    fn maybe_autosave(&mut self) {
        if self.last_autosave.elapsed() < AUTOSAVE_INTERVAL {
            return;
        }
        // Reset the clock whether or not we write, so a clean document is re-checked
        // one interval later rather than on every single frame.
        self.last_autosave = Instant::now();
        let needs = self
            .renderer
            .as_ref()
            .map(|r| r.needs_autosave())
            .unwrap_or(false);
        if !needs {
            return;
        }
        let path = self.autosave_path();
        if let Some(renderer) = self.renderer.as_mut() {
            match renderer.autosave(&path.to_string_lossy()) {
                Ok(()) => log::info!("autosaved {}", path.display()),
                Err(e) => log::error!("autosave failed: {e}"),
            }
        }
        self.autosave_counter = self.autosave_counter.wrapping_add(1);
    }

    /// Decide whether it's safe to close the window (G31). With no unsaved changes,
    /// yes. Otherwise prompt Save / Don't Save / Cancel: Save persists (and aborts
    /// the close if the user backs out of the Save As dialog), Don't Save discards,
    /// Cancel keeps editing.
    fn confirm_close(&mut self) -> bool {
        let dirty = self
            .renderer
            .as_ref()
            .map(|r| r.is_dirty())
            .unwrap_or(false);
        if !dirty {
            return true;
        }
        use rfd::{MessageButtons, MessageDialog, MessageDialogResult, MessageLevel};
        let res = MessageDialog::new()
            .set_level(MessageLevel::Warning)
            .set_title("Unsaved changes")
            .set_description("You have unsaved changes. Save them before closing?")
            .set_buttons(MessageButtons::YesNoCancel)
            .show();
        match res {
            MessageDialogResult::Yes => self.save_before_close(),
            MessageDialogResult::No => true,
            // Cancel, or the dialog dismissed — stay open.
            _ => false,
        }
    }

    /// Save for the close prompt: quicksave to the open file, or ask for a path if
    /// the project was never saved. Returns whether a save actually happened — a
    /// `false` (the user cancelled the Save As dialog, or the write failed) aborts
    /// the close so the work isn't lost.
    fn save_before_close(&mut self) -> bool {
        let path = match self.project_path.clone() {
            Some(p) => p,
            None => match rfd::FileDialog::new()
                .set_file_name("untitled.lowtex")
                .add_filter("lowtex project", &["lowtex"])
                .save_file()
            {
                Some(p) => {
                    self.project_path = Some(p.clone());
                    p
                }
                None => return false,
            },
        };
        match self.renderer.as_mut() {
            Some(renderer) => match renderer.save_project(&path.to_string_lossy()) {
                Ok(()) => {
                    log::info!("saved project {}", path.display());
                    true
                }
                Err(e) => {
                    log::error!("save failed: {e}");
                    false
                }
            },
            None => true,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create the window on resume (correct lifecycle for winit 0.30).
        let window_attrs = Window::default_attributes()
            .with_title("lowtex")
            .with_inner_size(winit::dpi::LogicalSize::new(1024, 768));

        let window = Arc::new(
            event_loop
                .create_window(window_attrs)
                .expect("failed to create window"),
        );

        // Renderer setup is async (wgpu::Instance::request_adapter etc).
        // pollster::block_on runs it synchronously here.
        let mesh = self.mesh.take().unwrap_or_else(Mesh::cube);
        let mut renderer = pollster::block_on(Renderer::new(window.clone(), mesh));
        // Carry the restored texture folder into the document so a save records it
        // even if the user never re-opens the folder this session.
        renderer.set_texture_folder(
            self.ui
                .brush_folder
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned()),
        );

        // egui platform integration.
        ui::install_style(&self.egui_ctx);
        let egui_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        self.window = Some(window);
        self.renderer = Some(renderer);
        self.egui_state = Some(egui_state);

        // Paint the first frame. Under on-demand rendering nothing else requests it
        // until the user interacts, so the window would otherwise come up blank.
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.clone() else {
            return;
        };

        // Feed every event to egui first; remember whether it claimed the event.
        // `repaint` tells us egui itself needs to redraw (a slider drag, a hover
        // fade, a focused widget) — under on-demand rendering nothing else would
        // request that frame, so honor it here.
        let egui_response = self
            .egui_state
            .as_mut()
            .map(|s| s.on_window_event(&window, &event));
        let egui_consumed = egui_response.as_ref().map_or(false, |r| r.consumed);
        if egui_response.map_or(false, |r| r.repaint) {
            window.request_redraw();
        }

        // These need &mut self wholly, so handle them before borrowing renderer.
        match &event {
            WindowEvent::CloseRequested => {
                // Prompt to save unsaved work before quitting (G31); a Cancel (or a
                // backed-out Save dialog) keeps the window open.
                if self.confirm_close() {
                    event_loop.exit();
                }
                return;
            }
            WindowEvent::RedrawRequested => {
                self.redraw();
                return;
            }
            _ => {}
        }

        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        match event {
            WindowEvent::Resized(new_size) => {
                renderer.resize(new_size.width, new_size.height);
            }

            WindowEvent::CursorMoved { position, .. } => {
                let prev = self.last_pos;
                let pos = (position.x as f32, position.y as f32);
                let dx = pos.0 - prev.0;
                let dy = pos.1 - prev.1;
                self.mouse_pos = pos;
                self.last_pos = pos;
                match self.drag {
                    Drag::Paint => {
                        // Interpolate from the previous sample so fast drags stay solid.
                        renderer.paint_segment(prev, pos, &self.ui.brush);
                        window.request_redraw();
                    }
                    Drag::Orbit => {
                        renderer.orbit_camera(dx, dy);
                        window.request_redraw();
                    }
                    Drag::Pan => {
                        renderer.pan_camera(dx, dy);
                        window.request_redraw();
                    }
                    Drag::None => {}
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                self.last_pos = self.mouse_pos;
                // A click egui claimed must not start a viewport drag.
                if pressed && egui_consumed {
                    return;
                }
                match (button, pressed) {
                    (MouseButton::Left, true) => {
                        // A click on the orientation compass snaps the view down that
                        // axis instead of painting the mesh behind the gizmo.
                        if renderer.click_compass(self.mouse_pos) {
                            window.request_redraw();
                            return;
                        }
                        // The brush starts a drag-stroke; the fills are one-shot
                        // bucket clicks that commit their own single undo step.
                        match self.ui.tool {
                            crate::ui::Tool::Brush => {
                                self.drag = Drag::Paint;
                                renderer.begin_stroke();
                                renderer.paint_at(self.mouse_pos, &self.ui.brush);
                            }
                            crate::ui::Tool::FillFace => {
                                renderer.fill_face_at(self.mouse_pos, &self.ui.brush);
                            }
                            crate::ui::Tool::FillIsland => {
                                renderer.fill_island_at(self.mouse_pos, &self.ui.brush);
                            }
                            crate::ui::Tool::FillObject => {
                                renderer.fill_object_at(self.mouse_pos, &self.ui.brush);
                            }
                        }
                        window.request_redraw();
                    }
                    (MouseButton::Right, true) => self.drag = Drag::Orbit,
                    (MouseButton::Middle, true) => self.drag = Drag::Pan,
                    (_, false) => {
                        if self.drag == Drag::Paint {
                            renderer.end_stroke();
                        }
                        self.drag = Drag::None;
                    }
                    _ => {}
                }
            }

            WindowEvent::ModifiersChanged(mods) => {
                self.modifiers = mods.state();
            }

            WindowEvent::KeyboardInput { event: key, .. } => {
                // Let egui claim keys first (e.g. a focused widget); only then do
                // viewport shortcuts fire. Ctrl/⌘+Z undo, Ctrl/⌘+Shift+Z or
                // Ctrl/⌘+Y redo. Super covers macOS's Cmd.
                if egui_consumed || key.state != ElementState::Pressed {
                    return;
                }
                let cmd = self.modifiers.control_key() || self.modifiers.super_key();
                if !cmd {
                    return;
                }
                let shift = self.modifiers.shift_key();
                match key.physical_key {
                    PhysicalKey::Code(KeyCode::KeyZ) => {
                        if shift {
                            renderer.redo();
                        } else {
                            renderer.undo();
                        }
                        window.request_redraw();
                    }
                    PhysicalKey::Code(KeyCode::KeyY) => {
                        renderer.redo();
                        window.request_redraw();
                    }
                    // Quicksave (⌘/Ctrl+S); Shift adds "as…" for a fresh path. The
                    // dialogs/path tracking run in handle_ui_actions next frame, so
                    // stash the request on App (ui.actions would be reset first).
                    PhysicalKey::Code(KeyCode::KeyS) => {
                        if shift {
                            self.pending_save_as = true;
                        } else {
                            self.pending_quicksave = true;
                        }
                        window.request_redraw();
                    }
                    _ => {}
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if egui_consumed {
                    return;
                }
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.05,
                };
                renderer.zoom_camera(amount);
                window.request_redraw();
            }

            WindowEvent::RedrawRequested => self.redraw(),

            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Timed recovery write (G31): cheap clock check most wakes, a save only when
        // the interval has elapsed and the document actually changed.
        self.maybe_autosave();

        let Some(window) = self.window.as_ref() else {
            return;
        };

        // On-demand rendering. The loop sleeps (ControlFlow::WaitUntil) until there
        // is a reason to wake, rather than rendering a full frame every vsync
        // forever — that idle redraw loop is what pegged the GPU/CPU on Windows.
        // Input handlers and `redraw()` request the frames that reflect real
        // changes; here we only fire a due egui repaint and pick the next wake.
        if let Some(deadline) = self.repaint_at {
            if Instant::now() >= deadline {
                self.repaint_at = None;
                window.request_redraw();
            }
        }

        // Wake at worst at the next autosave check, so idle-but-unsaved work still
        // gets a recovery write without a busy loop (AUTOSAVE_INTERVAL is minutes,
        // so this idle wake costs nothing).
        let autosave_at = self.last_autosave + AUTOSAVE_INTERVAL;
        let wake = match self.repaint_at {
            Some(repaint) => repaint.min(autosave_at),
            None => autosave_at,
        };
        event_loop.set_control_flow(ControlFlow::WaitUntil(wake));
    }
}

/// Scan `dir` for image files and stage a small thumbnail for each, for the brush/
/// stamp folder browser. Non-images and unreadable files are skipped, results are
/// sorted by name, and the count is capped so a huge folder can't stall the UI thread.
fn scan_brush_folder(dir: &std::path::Path) -> Vec<crate::ui::BrushEntry> {
    const MAX: usize = 256;
    const EXTS: [&str; 3] = ["png", "jpg", "jpeg"];
    let Ok(read) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut paths: Vec<std::path::PathBuf> = read
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| EXTS.contains(&e.to_ascii_lowercase().as_str()))
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    let mut entries = Vec::new();
    for path in paths.into_iter().take(MAX) {
        let Ok(mat) = crate::material::Material::load(&path.to_string_lossy()) else {
            continue; // unreadable / undecodable — skip it
        };
        let (w, h, px) = mat.thumbnail(48);
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        entries.push(crate::ui::BrushEntry {
            path,
            name,
            thumb: Some((w, h, px)),
            tex: None,
        });
    }
    entries
}

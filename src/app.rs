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

use std::sync::Arc;

use egui::ViewportId;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::ActiveEventLoop,
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
}

impl App {
    pub fn new(mesh: Mesh) -> Self {
        Self {
            window: None,
            renderer: None,
            mesh: Some(mesh),
            egui_ctx: egui::Context::default(),
            egui_state: None,
            ui: UiState::default(),
            mouse_pos: (0.0, 0.0),
            last_pos: (0.0, 0.0),
            drag: Drag::None,
            modifiers: ModifiersState::empty(),
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

        // Push brush-image state: how tightly the loaded image tiles (whether the
        // brush paints an image at all is just whether one is loaded in the renderer).
        renderer.set_brush_tile(self.ui.brush_tile);

        // Push fluid-brush state: whether the active tool is the fluid brush, and its
        // color/viscosity/amount settings.
        renderer.set_fluid(self.ui.tool == crate::ui::Tool::Fluid);
        renderer.set_fluid_spec(self.ui.fluid_spec());

        // Push mirror-painting state: the chosen axis when enabled, else off.
        renderer.set_symmetry(self.ui.symmetry_on.then_some(self.ui.symmetry_axis));

        // Push face-lock state: confine each dab to the face it lands on.
        renderer.set_lock_face(self.ui.lock_face);

        // Outline the active face while a painting tool has the lock on, so the user
        // sees exactly where the brush is confined. Cleared (None) otherwise.
        let painting_tool = matches!(self.ui.tool, crate::ui::Tool::Brush | crate::ui::Tool::Fluid);
        let outline_cursor = (self.ui.lock_face && painting_tool).then_some(self.mouse_pos);
        renderer.set_face_outline(outline_cursor);

        // Show the brush footprint ring at the cursor while a painting tool is active.
        renderer.set_brush_cursor(painting_tool.then_some(self.mouse_pos), &self.ui.brush);

        // Ghost-preview the stamp under the cursor for the solid/image brush, so the
        // painter sees what a click lays down before committing. (Fluid is a dynamic
        // sim with no static footprint, so it only gets the ring.)
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
        if actions.save_project {
            if let Some(path) = rfd::FileDialog::new()
                .set_file_name("untitled.lowtex")
                .add_filter("lowtex project", &["lowtex"])
                .save_file()
            {
                match renderer.save_project(&path.to_string_lossy()) {
                    Ok(()) => log::info!("saved project {}", path.display()),
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
                    Ok(()) => log::info!("opened project {}", path.display()),
                    Err(e) => log::error!("{e}"),
                }
            }
        }
        if let Some(density) = actions.unwrap {
            let (atlas, clamped) = renderer.apply_unwrap(density);
            self.ui.last_atlas_clamped = clamped;
            log::info!(
                "unwrapped → {atlas}×{atlas}{}",
                if clamped { " (density clamped to GPU max)" } else { "" }
            );
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
            renderer.add_map_layer("Tint", src, lv, rgb, crate::layers::BlendMode::Normal, noise);
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
        let renderer = pollster::block_on(Renderer::new(window.clone(), mesh));

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
        let egui_consumed = self
            .egui_state
            .as_mut()
            .map(|s| s.on_window_event(&window, &event).consumed)
            .unwrap_or(false);

        // These need &mut self wholly, so handle them before borrowing renderer.
        match &event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
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
                            crate::ui::Tool::Brush | crate::ui::Tool::Fluid => {
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

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

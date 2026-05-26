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

        // Push the paint target (color vs mask) + mask polarity (G11).
        renderer.set_paint_target(if self.ui.paint_mask {
            crate::renderer::PaintTarget::Mask
        } else {
            crate::renderer::PaintTarget::Color
        });
        renderer.set_mask_reveal(self.ui.mask_reveal);

        let actions = std::mem::take(&mut self.ui.actions);

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
        if let Some(mode) = actions.unwrap {
            renderer.apply_unwrap(mode);
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

        if let Some(size) = actions.set_resolution {
            renderer.set_texture_resolution(size);
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

        // Mesh effects — bake mesh maps (cached) then add a generated layer or fill
        // the active layer's mask from a baked map.
        if let Some(lv) = actions.apply_ao {
            renderer.apply_ao_layer(lv);
        }
        if let Some(lv) = actions.apply_highlight {
            renderer.apply_highlight_layer(lv);
        }
        if let Some(lv) = actions.apply_dirt {
            renderer.apply_dirt_layer(lv);
        }
        if let Some(lv) = actions.apply_edge_wear {
            renderer.apply_edge_wear_layer(lv);
        }
        if let Some((src, lv, color)) = actions.apply_tint {
            let rgb = [
                (color[0] * 255.0).round() as u8,
                (color[1] * 255.0).round() as u8,
                (color[2] * 255.0).round() as u8,
            ];
            renderer.add_map_layer("Tint", src, lv, rgb, crate::layers::BlendMode::Normal);
        }
        if let Some((src, lv)) = actions.mask_from_map {
            renderer.fill_active_mask_from_map(src, lv);
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

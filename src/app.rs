// src/app.rs
//
// The application shell: owns the window, the renderer, and the input state.
// Implements winit's ApplicationHandler trait (the modern 0.30+ event API).

use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

use crate::mesh::Mesh;
use crate::renderer::Renderer;

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
    mouse_pos: (f32, f32),
    last_pos: (f32, f32),
    drag: Drag,
}

impl App {
    pub fn new(mesh: Mesh) -> Self {
        Self {
            window: None,
            renderer: None,
            mesh: Some(mesh),
            mouse_pos: (0.0, 0.0),
            last_pos: (0.0, 0.0),
            drag: Drag::None,
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
        let renderer = pollster::block_on(Renderer::new(window.clone(), mesh));

        self.window = Some(window);
        self.renderer = Some(renderer);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        let Some(window) = self.window.as_ref() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(new_size) => {
                renderer.resize(new_size.width, new_size.height);
            }

            WindowEvent::CursorMoved { position, .. } => {
                let pos = (position.x as f32, position.y as f32);
                let dx = pos.0 - self.last_pos.0;
                let dy = pos.1 - self.last_pos.1;
                self.mouse_pos = pos;
                self.last_pos = pos;
                match self.drag {
                    Drag::Paint => {
                        renderer.paint_at(self.mouse_pos);
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
                match (button, pressed) {
                    (MouseButton::Left, true) => {
                        self.drag = Drag::Paint;
                        renderer.paint_at(self.mouse_pos);
                        window.request_redraw();
                    }
                    (MouseButton::Right, true) => self.drag = Drag::Orbit,
                    (MouseButton::Middle, true) => self.drag = Drag::Pan,
                    (_, false) => self.drag = Drag::None,
                    _ => {}
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.05,
                };
                renderer.zoom_camera(amount);
                window.request_redraw();
            }

            WindowEvent::RedrawRequested => {
                renderer.render();
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

// src/app.rs
//
// The application shell: owns the window, the renderer, and the input state.
// Implements winit's ApplicationHandler trait (the modern 0.30+ event API).

use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

use crate::renderer::Renderer;

#[derive(Default)]
pub struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    mouse_pos: (f32, f32),
    mouse_down: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create the window on resume (correct lifecycle for winit 0.30).
        let window_attrs = Window::default_attributes()
            .with_title("Lowtex — proof of paint")
            .with_inner_size(winit::dpi::LogicalSize::new(1024, 768));

        let window = Arc::new(
            event_loop
                .create_window(window_attrs)
                .expect("failed to create window"),
        );

        // Renderer setup is async (wgpu::Instance::request_adapter etc).
        // pollster::block_on runs it synchronously here.
        let renderer = pollster::block_on(Renderer::new(window.clone()));

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
                self.mouse_pos = (position.x as f32, position.y as f32);
                if self.mouse_down {
                    renderer.paint_at(self.mouse_pos);
                    window.request_redraw();
                }
            }

            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.mouse_down = state == ElementState::Pressed;
                if self.mouse_down {
                    renderer.paint_at(self.mouse_pos);
                    window.request_redraw();
                }
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

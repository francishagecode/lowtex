// src/main.rs
//
// Lowtex — PSX-style 3D texture painter.
// Opens a window, renders a textured mesh, lets you orbit and paint on it.
//
// CLI:
//   lowtex [MESH]                       open a mesh (glTF/glb/OBJ) or the cube
//   lowtex --screenshot OUT [MESH]      render one frame headless to a PNG
//                                       (--width N --height N to set size)
//
// The --screenshot path exists for headless verification: it renders the same
// scene the window would, to an offscreen texture, and writes it to disk.

mod app;
mod camera;
mod mesh;
mod paint;
mod renderer;

use winit::event_loop::{ControlFlow, EventLoop};

use crate::mesh::Mesh;
use crate::renderer::Renderer;

struct Args {
    screenshot: Option<String>,
    mesh: Option<String>,
    width: u32,
    height: u32,
}

fn parse_args() -> Args {
    let mut args = Args {
        screenshot: None,
        mesh: None,
        width: 1024,
        height: 768,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--screenshot" => args.screenshot = it.next(),
            "--width" => {
                if let Some(v) = it.next().and_then(|s| s.parse().ok()) {
                    args.width = v;
                }
            }
            "--height" => {
                if let Some(v) = it.next().and_then(|s| s.parse().ok()) {
                    args.height = v;
                }
            }
            other if !other.starts_with("--") => args.mesh = Some(other.to_string()),
            other => log::warn!("ignoring unknown argument: {other}"),
        }
    }
    args
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = parse_args();

    // TODO(G1): load `args.mesh` once the model loader exists. For now the cube.
    let mesh = Mesh::cube();

    if let Some(out) = args.screenshot {
        run_screenshot(&out, args.width, args.height, mesh);
        return;
    }

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = app::App::default();
    event_loop.run_app(&mut app).expect("event loop error");
}

/// Headless: render one frame to `out` as a PNG and exit.
fn run_screenshot(out: &str, width: u32, height: u32, mesh: Mesh) {
    let mut renderer = pollster::block_on(Renderer::new_headless(width, height, mesh));
    let (pixels, w, h) = renderer.capture();
    image::save_buffer(out, &pixels, w, h, image::ColorType::Rgba8)
        .expect("failed to write screenshot PNG");
    log::info!("wrote screenshot {out} ({w}x{h})");
}

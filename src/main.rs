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
mod model;
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
    /// Headless verification: stamp a brush at screen center before capture.
    paint: bool,
    /// Headless verification: orbit horizontally by this many degrees before capture.
    orbit_deg: f32,
}

fn parse_args() -> Args {
    let mut args = Args {
        screenshot: None,
        mesh: None,
        width: 1024,
        height: 768,
        paint: false,
        orbit_deg: 0.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--screenshot" => args.screenshot = it.next(),
            "--paint" => args.paint = true,
            "--orbit" => {
                if let Some(v) = it.next().and_then(|s| s.parse().ok()) {
                    args.orbit_deg = v;
                }
            }
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

    // Load the requested mesh, or fall back to the sample cube.
    let mesh = match &args.mesh {
        Some(path) => match model::load(path) {
            Ok(m) => m,
            Err(e) => {
                log::error!("{e} — falling back to the sample cube");
                Mesh::cube()
            }
        },
        None => Mesh::cube(),
    };

    if let Some(out) = args.screenshot {
        run_screenshot(
            &out,
            args.width,
            args.height,
            mesh,
            args.paint,
            args.orbit_deg,
        );
        return;
    }

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = app::App::new(mesh);
    event_loop.run_app(&mut app).expect("event loop error");
}

/// Headless: render one frame to `out` as a PNG and exit. With `paint`, stamp a
/// few brush dabs around screen center first; with `orbit_deg`, paint, orbit,
/// then paint again — verifying both that the camera moved and that paint stays
/// fixed on the surface (so you can orbit to the back and paint there).
fn run_screenshot(out: &str, width: u32, height: u32, mesh: Mesh, paint: bool, orbit_deg: f32) {
    let mut renderer = pollster::block_on(Renderer::new_headless(width, height, mesh));
    let (cx, cy) = (width as f32 / 2.0, height as f32 / 2.0);
    let dab = |r: &mut Renderer| {
        for (dx, dy) in [
            (0.0, 0.0),
            (-30.0, 0.0),
            (30.0, 0.0),
            (0.0, -30.0),
            (0.0, 30.0),
        ] {
            r.paint_at((cx + dx, cy + dy));
        }
    };
    if paint {
        dab(&mut renderer);
    }
    if orbit_deg != 0.0 {
        renderer.orbit_view_radians(orbit_deg.to_radians(), 0.0);
        if paint {
            dab(&mut renderer);
        }
    }
    let (pixels, w, h) = renderer.capture();
    image::save_buffer(out, &pixels, w, h, image::ColorType::Rgba8)
        .expect("failed to write screenshot PNG");
    log::info!("wrote screenshot {out} ({w}x{h})");
}

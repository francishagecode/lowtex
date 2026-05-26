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
mod bvh;
mod camera;
mod mesh;
mod model;
mod paint;
mod renderer;
mod ui;

use winit::event_loop::{ControlFlow, EventLoop};

use crate::mesh::Mesh;
use crate::paint::Brush;
use crate::renderer::Renderer;

struct Args {
    screenshot: Option<String>,
    mesh: Option<String>,
    width: u32,
    height: u32,
    /// Headless verification: stamp a brush at screen center before capture.
    paint: bool,
    /// Headless verification: paint one fast drag (a single stroke) before capture.
    stroke: bool,
    /// Headless verification: orbit horizontally by this many degrees before capture.
    orbit_deg: f32,
    /// Headless verification: draw the egui panel into the screenshot.
    ui: bool,
    /// Headless verification: override the brush (color r,g,b in 0..1 and size).
    brush_color: Option<[f32; 3]>,
    brush_size: Option<f32>,
    brush_opacity: Option<f32>,
    /// Headless verification: load a starting texture / save the result / set res.
    load_texture: Option<String>,
    save_texture: Option<String>,
    res: Option<u32>,
    /// Headless verification: disable PSX mode (clean view) / enable fog / flat
    /// shading / override the wobble grid.
    psx_off: bool,
    fog: bool,
    flat: bool,
    psx_grid: Option<f32>,
}

fn parse_args() -> Args {
    let mut args = Args {
        screenshot: None,
        mesh: None,
        width: 1024,
        height: 768,
        paint: false,
        stroke: false,
        orbit_deg: 0.0,
        ui: false,
        brush_color: None,
        brush_size: None,
        brush_opacity: None,
        load_texture: None,
        save_texture: None,
        res: None,
        psx_off: false,
        fog: false,
        flat: false,
        psx_grid: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--screenshot" => args.screenshot = it.next(),
            "--paint" => args.paint = true,
            "--stroke" => args.stroke = true,
            "--ui" => args.ui = true,
            "--orbit" => {
                if let Some(v) = it.next().and_then(|s| s.parse().ok()) {
                    args.orbit_deg = v;
                }
            }
            "--brush-color" => {
                if let Some(s) = it.next() {
                    let c: Vec<f32> = s.split(',').filter_map(|x| x.parse().ok()).collect();
                    if let [r, g, b] = c[..] {
                        args.brush_color = Some([r, g, b]);
                    }
                }
            }
            "--brush-size" => args.brush_size = it.next().and_then(|s| s.parse().ok()),
            "--brush-opacity" => args.brush_opacity = it.next().and_then(|s| s.parse().ok()),
            "--load-texture" => args.load_texture = it.next(),
            "--save-texture" => args.save_texture = it.next(),
            "--res" => args.res = it.next().and_then(|s| s.parse().ok()),
            "--psx-off" => args.psx_off = true,
            "--fog" => args.fog = true,
            "--flat" => args.flat = true,
            "--psx-grid" => args.psx_grid = it.next().and_then(|s| s.parse().ok()),
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

    if let Some(out) = args.screenshot.clone() {
        run_screenshot(&out, mesh, &args);
        return;
    }

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = app::App::new(mesh);
    event_loop.run_app(&mut app).expect("event loop error");
}

/// Headless: render one frame to `out` as a PNG and exit. With `--paint`, stamp a
/// few brush dabs around screen center first; with `--orbit`, paint, orbit, then
/// paint again; with `--ui`, draw the egui panel on top.
fn run_screenshot(out: &str, mesh: Mesh, args: &Args) {
    let (width, height) = (args.width, args.height);
    let mut renderer = pollster::block_on(Renderer::new_headless(width, height, mesh));

    let defaults = renderer::PsxSettings::default();
    let psx = renderer::PsxSettings {
        enabled: !args.psx_off,
        fog: args.fog,
        flat: args.flat,
        grid: args.psx_grid.unwrap_or(defaults.grid),
        ..defaults
    };
    renderer.set_psx_settings(psx);

    let mut brush = Brush::default();
    if let Some(c) = args.brush_color {
        brush.color = c;
    }
    if let Some(s) = args.brush_size {
        brush.radius = s;
    }
    if let Some(o) = args.brush_opacity {
        brush.opacity = o;
    }

    if let Some(size) = args.res {
        renderer.set_texture_resolution(size);
    }
    if let Some(path) = &args.load_texture {
        match renderer.load_texture_png(path) {
            Ok(()) => log::info!("loaded texture {path}"),
            Err(e) => log::error!("{e}"),
        }
    }

    let (cx, cy) = (width as f32 / 2.0, height as f32 / 2.0);
    let dab = |r: &mut Renderer| {
        for (dx, dy) in [
            (0.0, 0.0),
            (-30.0, 0.0),
            (30.0, 0.0),
            (0.0, -30.0),
            (0.0, 30.0),
        ] {
            r.paint_at((cx + dx, cy + dy), &brush);
        }
    };
    if args.paint {
        dab(&mut renderer);
    }
    if args.stroke {
        // One fast diagonal drag across the front face — interpolation should
        // fill it solid despite the large jump between the two endpoints.
        renderer.begin_stroke();
        renderer.paint_segment((cx - 80.0, cy - 50.0), (cx + 80.0, cy + 50.0), &brush);
        renderer.end_stroke();
    }
    if args.orbit_deg != 0.0 {
        renderer.orbit_view_radians(args.orbit_deg.to_radians(), 0.0);
        if args.paint {
            dab(&mut renderer);
        }
    }

    if let Some(path) = &args.save_texture {
        match renderer.save_texture_png(path) {
            Ok(()) => log::info!("saved texture {path}"),
            Err(e) => log::error!("{e}"),
        }
    }

    // Optionally build one egui frame to draw the panel into the screenshot.
    let ui_paint = if args.ui {
        Some(build_headless_ui(width, height))
    } else {
        None
    };
    let (pixels, w, h) =
        renderer.capture(ui_paint.as_ref().map(|(jobs, td, ppp)| renderer::UiPaint {
            jobs,
            textures_delta: td,
            pixels_per_point: *ppp,
        }));
    image::save_buffer(out, &pixels, w, h, image::ColorType::Rgba8)
        .expect("failed to write screenshot PNG");
    log::info!("wrote screenshot {out} ({w}x{h})");
}

/// Run egui once with no window, returning tessellated jobs + texture deltas so
/// the panel can be composited into a headless screenshot.
fn build_headless_ui(
    width: u32,
    height: u32,
) -> (Vec<egui::ClippedPrimitive>, egui::TexturesDelta, f32) {
    let ctx = egui::Context::default();
    ui::install_style(&ctx);
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::pos2(0.0, 0.0),
            egui::vec2(width as f32, height as f32),
        )),
        ..Default::default()
    };
    let mut state = ui::UiState::default();
    let out = ctx.run(raw_input, |ctx| ui::build(ctx, &mut state));
    let jobs = ctx.tessellate(out.shapes, out.pixels_per_point);
    (jobs, out.textures_delta, out.pixels_per_point)
}

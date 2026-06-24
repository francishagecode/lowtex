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
mod bake;
mod bleed;
mod bvh;
mod camera;
mod config;
mod effects;
mod export;
mod fill;
mod history;
mod layers;
mod material;
mod mesh;
mod model;
mod noise;
mod paint;
mod palette;
mod preset;
mod project;
mod renderer;
mod surface;
mod tablet;
mod ui;
mod unwrap;

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
    /// Headless verification: paint one stroke directly in UV space (the 2D UV editor's
    /// path) before capture — a diagonal across the atlas, no raycast.
    paint_uv: bool,
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
    /// Headless verification: enable palette quantize / pick a built-in / no dither.
    quantize: bool,
    palette_builtin: Option<usize>,
    no_dither: bool,
    /// Headless verification: paint base, add a layer, paint it a 2nd color.
    layer_demo: bool,
    /// Headless verification: paint a layer, then carve a stripe out of its mask.
    mask_demo: bool,
    /// Headless verification: bake an AO / edge-highlight layer at this strength.
    ao: Option<f32>,
    highlight: Option<f32>,
    /// Headless verification: bucket-fill the whole object / the island / the flat
    /// face at screen center.
    fill_object: bool,
    fill_island: bool,
    fill_face: bool,
    /// Headless verification: auto-unwrap the mesh before capture, at the texel
    /// density given by `--density` (low|medium|high; default medium).
    unwrap: bool,
    density: Option<String>,
    /// With `--unwrap`: stack congruent (identical/mirrored) charts onto shared UV space.
    overlap_uvs: bool,
    /// Headless verification: export an indexed PNG to this path (needs --quantize).
    export_indexed: Option<String>,
    /// Headless verification: fill the base layer with a material image (tiled).
    material: Option<String>,
    material_tile: f32,
    /// Headless verification: paint one stroke with the texture brush, using this
    /// image (UV-tiled by `material_tile`) — the brush counterpart to `--material`.
    texture_brush: Option<String>,
    /// Headless verification: material on a NEW layer, masked by AO (Cavities) —
    /// the "moss in the crevices" workflow.
    material_crevice: bool,
    /// Headless verification: open a .lowtex first / save one after edits.
    open_project: Option<String>,
    save_project: Option<String>,
    /// Headless verification: apply a built-in preset look by name (G21).
    preset: Option<String>,
    /// Headless verification: save the applied recipe to / load a preset from a path.
    save_preset: Option<String>,
    load_preset: Option<String>,
}

fn parse_args() -> Args {
    let mut args = Args {
        screenshot: None,
        mesh: None,
        width: 1024,
        height: 768,
        paint: false,
        stroke: false,
        paint_uv: false,
        orbit_deg: 0.0,
        ui: false,
        brush_color: None,
        brush_size: None,
        brush_opacity: None,
        load_texture: None,
        save_texture: None,
        res: None,
        quantize: false,
        palette_builtin: None,
        no_dither: false,
        layer_demo: false,
        mask_demo: false,
        ao: None,
        highlight: None,
        fill_object: false,
        fill_island: false,
        fill_face: false,
        unwrap: false,
        density: None,
        overlap_uvs: false,
        export_indexed: None,
        material: None,
        material_tile: 4.0,
        texture_brush: None,
        material_crevice: false,
        open_project: None,
        save_project: None,
        preset: None,
        save_preset: None,
        load_preset: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--screenshot" => args.screenshot = it.next(),
            "--paint" => args.paint = true,
            "--stroke" => args.stroke = true,
            "--paint-uv" => args.paint_uv = true,
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
            "--layer-demo" => args.layer_demo = true,
            "--mask-demo" => args.mask_demo = true,
            "--ao" => args.ao = it.next().and_then(|s| s.parse().ok()),
            "--highlight" => args.highlight = it.next().and_then(|s| s.parse().ok()),
            "--fill-object" => args.fill_object = true,
            "--fill-island" => args.fill_island = true,
            "--fill-face" => args.fill_face = true,
            "--unwrap" => args.unwrap = true,
            "--density" => args.density = it.next(),
            "--overlap-uvs" => args.overlap_uvs = true,
            "--export-indexed" => args.export_indexed = it.next(),
            "--material" => args.material = it.next(),
            "--material-tile" => {
                if let Some(v) = it.next().and_then(|s| s.parse().ok()) {
                    args.material_tile = v;
                }
            }
            "--texture-brush" => args.texture_brush = it.next(),
            "--material-crevice" => args.material_crevice = true,
            "--open-project" => args.open_project = it.next(),
            "--save-project" => args.save_project = it.next(),
            "--preset" => args.preset = it.next(),
            "--save-preset" => args.save_preset = it.next(),
            "--load-preset" => args.load_preset = it.next(),
            "--quantize" => args.quantize = true,
            "--no-dither" => args.no_dither = true,
            "--palette" => args.palette_builtin = it.next().and_then(|s| s.parse().ok()),
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

    if let Some(path) = &args.open_project {
        match renderer.load_project(path) {
            Ok(()) => log::info!("opened project {path}"),
            Err(e) => log::error!("{e}"),
        }
    }

    if args.unwrap {
        let density = match args.density.as_deref() {
            Some("low") => unwrap::Density::Low,
            Some("high") => unwrap::Density::High,
            _ => unwrap::Density::Medium,
        };
        renderer.apply_unwrap(density, args.overlap_uvs);
    }

    if let Some(path) = &args.material {
        if args.material_crevice {
            // "Moss in the crevices": material on a new layer, masked by AO.
            renderer.add_layer();
            if let Err(e) = renderer.fill_active_with_material(path, args.material_tile) {
                log::error!("{e}");
            }
            renderer.fill_active_mask_from_map(
                bake::MapSource::Cavities,
                bake::Levels::amount(1.0),
                None,
            );
        } else {
            match renderer.fill_active_with_material(path, args.material_tile) {
                Ok(()) => log::info!("filled with material {path}"),
                Err(e) => log::error!("{e}"),
            }
        }
    }

    if args.quantize {
        renderer.set_palette_settings(renderer::PaletteSettings {
            enabled: true,
            dither: !args.no_dither,
            dither_strength: 0.06,
        });
    }
    if let Some(i) = args.palette_builtin {
        if let Some(p) = palette::Palette::builtins().into_iter().nth(i) {
            renderer.set_palette(p);
        }
    }

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
    if args.layer_demo {
        // Paint a red stroke on the base, then a green stroke on a layer above it.
        let mut b = Brush {
            color: [0.85, 0.2, 0.2],
            radius: 7.0,
            ..Brush::default()
        };
        renderer.begin_stroke();
        renderer.paint_segment((cx - 70.0, cy - 40.0), (cx + 70.0, cy - 40.0), &b);
        renderer.end_stroke();
        renderer.add_layer();
        b.color = [0.2, 0.8, 0.3];
        renderer.begin_stroke();
        renderer.paint_segment((cx - 70.0, cy + 40.0), (cx + 70.0, cy + 40.0), &b);
        renderer.end_stroke();
    }
    if args.mask_demo {
        // Cover the front with green on a new layer, then hide a band in its mask
        // so the base checkerboard shows through where the mask is black.
        let green = Brush {
            color: [0.2, 0.8, 0.3],
            radius: 22.0,
            ..Brush::default()
        };
        renderer.add_layer();
        for dy in [-30.0, 0.0, 30.0] {
            renderer.begin_stroke();
            renderer.paint_segment((cx - 70.0, cy + dy), (cx + 70.0, cy + dy), &green);
            renderer.end_stroke();
        }
        renderer.set_paint_target(renderer::PaintTarget::Mask);
        renderer.set_mask_reveal(false); // hide
        let eraser = Brush {
            radius: 10.0,
            ..Brush::default()
        };
        renderer.begin_stroke();
        renderer.paint_segment((cx - 80.0, cy), (cx + 80.0, cy), &eraser);
        renderer.end_stroke();
    }
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
    if args.paint_uv {
        // One diagonal stroke straight in UV space (the 2D UV editor's path): no
        // raycast, a flat disc per step. With an unwrapped cube this paints a band
        // across the atlas that shows on whichever faces those texels map to.
        use glam::Vec2;
        renderer.begin_stroke();
        renderer.paint_uv_segment(Vec2::new(0.15, 0.15), Vec2::new(0.85, 0.85), &brush);
        renderer.end_stroke();
        log::info!("painted one UV-space stroke");
    }
    if let Some(path) = &args.texture_brush {
        // Paint one wide diagonal stroke that reveals the tiled image only where it
        // lands — the rest of the front face keeps the base checkerboard.
        match renderer.load_brush_material(path) {
            Ok(()) => {
                renderer.set_brush_tile(args.material_tile);
                let tb = Brush {
                    radius: args.brush_size.unwrap_or(16.0),
                    ..brush
                };
                renderer.begin_stroke();
                renderer.paint_segment((cx - 90.0, cy - 55.0), (cx + 90.0, cy + 55.0), &tb);
                renderer.end_stroke();
                log::info!("painted texture-brush stroke with {path}");
            }
            Err(e) => log::error!("{e}"),
        }
    }
    if args.orbit_deg != 0.0 {
        renderer.orbit_view_radians(args.orbit_deg.to_radians(), 0.0);
        if args.paint {
            dab(&mut renderer);
        }
    }

    if let Some(s) = args.ao {
        renderer.apply_ao_layer(crate::bake::Levels::amount(s), None);
    }
    if let Some(s) = args.highlight {
        renderer.apply_highlight_layer(crate::bake::Levels::amount(s), None);
    }
    if args.fill_face {
        renderer.fill_face_at((cx, cy), &brush);
    }
    if args.fill_island {
        renderer.fill_island_at((cx, cy), &brush);
    }
    if args.fill_object {
        renderer.fill_object_at((cx, cy), &brush);
    }

    if let Some(name) = &args.preset {
        match renderer.apply_builtin_preset(name) {
            Ok(()) => log::info!("applied preset '{name}'"),
            Err(e) => log::error!("{e}"),
        }
    }
    if let Some(path) = &args.load_preset {
        match renderer.load_and_apply_preset(path) {
            Ok(()) => log::info!("applied preset from {path}"),
            Err(e) => log::error!("{e}"),
        }
    }
    if let Some(path) = &args.save_preset {
        match renderer.save_preset(path, "Custom") {
            Ok(()) => log::info!("saved preset {path}"),
            Err(e) => log::error!("{e}"),
        }
    }

    if let Some(path) = &args.save_texture {
        match renderer.save_texture_png(path) {
            Ok(()) => log::info!("saved texture {path}"),
            Err(e) => log::error!("{e}"),
        }
    }
    if let Some(path) = &args.export_indexed {
        match renderer.export_png(path, true) {
            Ok(()) => log::info!("exported indexed PNG {path}"),
            Err(e) => log::error!("{e}"),
        }
    }
    if let Some(path) = &args.save_project {
        match renderer.save_project(path) {
            Ok(()) => log::info!("saved project {path}"),
            Err(e) => log::error!("{e}"),
        }
    }

    // Optionally build one egui frame to draw the panel into the screenshot. Offset
    // the scene by the panel width so the headless capture matches the live app.
    let ui_paint = if args.ui {
        let built = build_headless_ui(width, height);
        renderer.set_view_offset(built.3 * built.2);
        Some(built)
    } else {
        None
    };
    let (pixels, w, h) =
        renderer.capture(
            ui_paint
                .as_ref()
                .map(|(jobs, td, ppp, _)| renderer::UiPaint {
                    jobs,
                    textures_delta: td,
                    pixels_per_point: *ppp,
                }),
        );
    image::save_buffer(out, &pixels, w, h, image::ColorType::Rgba8)
        .expect("failed to write screenshot PNG");
    log::info!("wrote screenshot {out} ({w}x{h})");
}

/// Run egui once with no window, returning tessellated jobs + texture deltas so
/// the panel can be composited into a headless screenshot.
fn build_headless_ui(
    width: u32,
    height: u32,
) -> (Vec<egui::ClippedPrimitive>, egui::TexturesDelta, f32, f32) {
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
    (
        jobs,
        out.textures_delta,
        out.pixels_per_point,
        state.panel_width,
    )
}

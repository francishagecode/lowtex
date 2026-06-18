// src/renderer.rs
//
// The wgpu renderer. Owns:
//   - device / queue (the GPU connection)
//   - an optional surface (the window swap chain); absent in headless mode
//   - vertex + index buffers (the mesh)
//   - uniform buffer (view-projection matrix)
//   - the paint texture (lives on both CPU and GPU; CPU is source of truth in v0.1)
//   - the render pipeline (vertex shader + fragment shader + state)
//   - the camera
//   - the mesh (kept for ray picking)
//
// The scene-draw path (`draw_into`) is shared between two targets:
//   - the window surface (`render`) — interactive use
//   - an offscreen texture (`capture`) — headless screenshots for verification
// so a screenshot is faithful to what the window shows.
//
// Flow on each paint:
//   mouse pixel → Ray::from_screen → BVH pick → surface::splat (walk the mesh
//   surface across faces) → blend_texel → (palette quantize if enabled) →
//   Queue::write_texture (CPU → GPU)
//
// Flow on each frame:
//   acquire target view → encode draw → submit → present
//
// The PSX/low-poly look comes from the texture (low-res, limited palette, dither)
// plus nearest-neighbor sampling — not from screen-space warp/wobble.

use std::sync::Arc;

use glam::Vec2;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::bake::{Gradient, Levels, MapSource, NoiseMod};
use crate::bvh::{Bvh, Hit};
use crate::camera::Camera;
use crate::history::History;
use crate::layers::Layers;
use crate::mesh::{Mesh, Vertex};
use crate::paint::{Brush, Ray, TexRect, Texture as PaintTexture};
use crate::palette::Palette;

const TEX_SIZE: u32 = 128; // PSX-scale. Bump to 256 if you want.

/// How many undo steps to keep. Each step is a full layer-stack snapshot (a few
/// hundred KB at PSX sizes), so this bounds history memory to a handful of MB.
const HISTORY_CAP: usize = 64;

/// The color format used for offscreen capture. sRGB to match the surface.
const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// How many texels to bleed island colors into the UV gutter (G18). A few px is
/// enough to cover nearest-neighbour sampling slop at seams without smearing.
const BLEED_PAD: u32 = 4;

/// egui paint data for one frame, produced by the App and consumed by `render`.
pub struct UiPaint<'a> {
    pub jobs: &'a [egui::ClippedPrimitive],
    pub textures_delta: &'a egui::TexturesDelta,
    pub pixels_per_point: f32,
}

/// Palette-quantize settings (G8). The active `Palette` lives in the renderer.
/// Quantize is applied to the *texture* on the CPU (non-destructive: the
/// full-color paint buffer is preserved), so the model and the exported PNG show
/// the quantized result.
#[derive(Clone, Copy, PartialEq)]
pub struct PaletteSettings {
    pub enabled: bool,
    pub dither: bool,
    pub dither_strength: f32,
}

impl Default for PaletteSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            dither: true,
            dither_strength: 0.06,
        }
    }
}

/// How the painted texture is filtered when sampled onto the model (G30). The
/// sampler does the work — the shader's `textureSample` respects whichever mode
/// the sampler is built with, so switching is just a sampler + bind-group rebuild.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum TextureFilter {
    /// Nearest-neighbor — crisp texels, the default PSX look.
    #[default]
    Nearest,
    /// Bilinear — smooths the texture; some painters prefer the softened preview.
    Linear,
}

impl TextureFilter {
    fn wgpu(self) -> wgpu::FilterMode {
        match self {
            TextureFilter::Nearest => wgpu::FilterMode::Nearest,
            TextureFilter::Linear => wgpu::FilterMode::Linear,
        }
    }

    /// Short label for the picker.
    pub fn label(self) -> &'static str {
        match self {
            TextureFilter::Nearest => "Nearest (crisp)",
            TextureFilter::Linear => "Linear (smooth)",
        }
    }
}

/// What the brush paints into (G11).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PaintTarget {
    /// The active layer's color.
    Color,
    /// The active layer's reveal mask.
    Mask,
}

/// How a loaded brush image is applied. The same image slot (`brush_material`) feeds
/// both; this only changes the mapping.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum BrushImageMode {
    /// Brush: the image is anchored in UV/texture space and tiled, so painting reveals
    /// a *consistent* material field that doesn't shift or repeat as you stroke over an
    /// area (the original brush-image behaviour — right for brick, fabric, grime).
    #[default]
    Tiled,
    /// Stamp: one oriented placement of the image at the cursor, its own alpha driving
    /// coverage, projected into the hit face's tangent plane. Each dab re-centers it, so
    /// you can overdraw, shift, and rotate a decal (bolts, logos, scratches).
    Stamp,
}

/// A model-symmetry plane for mirrored painting: the plane perpendicular to this
/// axis, through the mesh's bounding-box center. With symmetry on, every dab is
/// also painted at its reflection across the plane, so a stroke down one side of a
/// symmetric model lays the same paint down the other.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SymmetryAxis {
    X,
    Y,
    Z,
}

impl SymmetryAxis {
    pub const ALL: [SymmetryAxis; 3] = [SymmetryAxis::X, SymmetryAxis::Y, SymmetryAxis::Z];

    pub fn name(self) -> &'static str {
        match self {
            SymmetryAxis::X => "X",
            SymmetryAxis::Y => "Y",
            SymmetryAxis::Z => "Z",
        }
    }

    /// Index of the component this axis mirrors (x=0, y=1, z=2).
    fn index(self) -> usize {
        match self {
            SymmetryAxis::X => 0,
            SymmetryAxis::Y => 1,
            SymmetryAxis::Z => 2,
        }
    }
}

/// A colored line-list vertex for the ground grid. Position is in the space the
/// bound view-proj expects; color is linear RGB (see lines.wgsl).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LineVertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl LineVertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// The compass's on-screen footprint, in physical pixels: a square of this side
/// (capped to the scene region) inset from the bottom-left corner by this margin.
const COMPASS_SIZE_PX: f32 = 110.0;
const COMPASS_MARGIN_PX: f32 = 12.0;

/// Number of segments in the brush-cursor ring. 48 reads as a smooth circle at
/// any zoom without being worth more vertices.
const BRUSH_RING_SEGMENTS: u32 = 48;

/// How opaque the brush stamp *preview* ghost is, as a fraction of the dab's own
/// per-texel coverage. Low enough to read as a preview, high enough to see the color.
const BRUSH_PREVIEW_ALPHA: f32 = 0.5;

/// A vertex of the thick-line compass. Each axis segment is a quad (6 of these),
/// so a vertex carries the *whole* segment (`start`, `end`) plus its corner in
/// `param` (`[t, side]`); compass.wgsl projects the endpoints and offsets the
/// corner sideways to give the line width. Color is linear RGB.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CompassVertex {
    start: [f32; 3],
    end: [f32; 3],
    color: [f32; 3],
    param: [f32; 2], // x = t (0 at start, 1 at end); y = side (signed half-width)
}

impl CompassVertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 36,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

/// Everything the hovering brush-stamp ghost depends on. The app pushes the brush
/// state and cursor every frame and the redraw loop runs continuously, so without
/// a guard the preview re-picks + re-splats the whole footprint each idle frame —
/// the same large-brush cost a stroke pays, but for nothing. We recompute only when
/// this key changes; committed-image changes (paint commit, fills, undo, palette,
/// resize, load) reset it to force a fresh ghost. `cam` is the camera generation, so
/// an orbit/pan/zoom (which moves where the cursor lands) invalidates it too.
#[derive(Clone, Copy, PartialEq)]
struct PreviewKey {
    cursor: (f32, f32),
    cam: u64,
    brush: Brush,
    tile: f32,
    material_gen: u64,
    symmetry: Option<SymmetryAxis>,
    lock_face: bool,
    target: PaintTarget,
    image_mode: BrushImageMode,
    stamp_angle: f32,
    stamp_tint: bool,
}

pub struct Renderer {
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,

    // Viewport furniture (G29): the ground grid is unlit 1px lines (depth Less, so
    // the mesh occludes lines behind it); the orientation compass is thick
    // screen-space quads (depth Always) drawn last in its own corner viewport, so
    // it's always on top. The compass is permanent and its axes are clickable.
    line_pipeline: wgpu::RenderPipeline, // grid: line list, depth Less, no write
    compass_pipeline: wgpu::RenderPipeline, // compass: triangle quads, depth Always
    grid_bind_group: wgpu::BindGroup,    // bound to the scene view-proj
    grid_vertex_buffer: wgpu::Buffer,
    grid_vertex_count: u32,
    compass_uniform: wgpu::Buffer, // rotation-only view-proj, refreshed with the camera
    compass_bind_group: wgpu::BindGroup,
    compass_vertex_buffer: wgpu::Buffer,
    compass_vertex_count: u32,
    // Active-face outline: the boundary edges of the facet(s) being painted, drawn
    // with the grid's line pipeline so the painter can see exactly where the face
    // lock confines the brush. Rebuilt only when the outlined facet set changes
    // (`outline_facets` is the cache key); `None`/0 when nothing is highlighted.
    outline_vertex_buffer: Option<wgpu::Buffer>,
    outline_vertex_count: u32,
    outline_facets: Vec<i32>,
    // Brush-cursor ring: a circle laid on the mesh surface at the cursor, sized to
    // the brush's true world-space footprint, so the painter sees how big the brush
    // is before stamping. Fixed-capacity buffer rewritten each frame; count is 0
    // when hidden (no painting tool, or the cursor is off the mesh).
    brush_ring_buffer: wgpu::Buffer,
    brush_ring_count: u32,
    // Inputs the ring last reflected (cursor, camera generation, brush radius). The
    // ring is rebuilt only when these change, so a stationary cursor doesn't re-pick +
    // re-upload the ring every idle frame. `None` = nothing drawn (force a rebuild).
    last_ring: Option<(f32, f32, u64, f32)>,
    // Brush stamp preview: a translucent ghost of the dab under the cursor, written
    // straight into the GPU paint texture (never the layers) so the painter sees
    // what a click would lay down. `preview_rect` is the region currently ghosted;
    // it's reverted from `display_buf` (the committed mirror) before the next ghost
    // or any commit, so nothing about the preview is ever persisted.
    preview_rect: Option<TexRect>,
    // Inputs the ghost last reflected; skip the (large-brush-expensive) re-splat when
    // unchanged. Reset to `None` whenever the committed image changes (so the ghost is
    // recomputed over the new pixels). See `PreviewKey`.
    last_preview: Option<PreviewKey>,
    // Monotonic camera/view generation, bumped in `update_uniforms`. Lets the ring and
    // ghost keys detect an orbit/pan/zoom/resize without diffing the matrix.
    camera_gen: std::cell::Cell<u64>,
    // Bumped whenever the brush image is loaded/cleared, so the ghost key notices a
    // material swap (the image itself isn't cheap to compare).
    brush_material_gen: u64,
    // Background clear color, in sRGB (what the picker shows); converted to linear
    // at clear time. Grid visibility toggle.
    bg_color: [f32; 3],
    show_grid: bool,

    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    // Which filter mode `sampler` was built with, so set_texture_filter can skip
    // the rebuild when nothing changed (the App pushes it every frame).
    texture_filter: TextureFilter,

    paint_texture_gpu: wgpu::Texture,
    // Layer stack (G10); painting targets the active layer, composited bottom-up
    // into the GPU texture (then palette-quantized).
    layers: Layers,
    tex_size: u32,

    // Stroke accumulation (G6): the active-layer snapshot at stroke start +
    // per-texel coverage, so overlapping stamps within one stroke don't double-darken.
    stroke_base: Vec<u8>,
    stroke_coverage: Vec<f32>,

    // Dirty-rectangle paint refresh: a CPU-side mirror of the GPU paint texture (the
    // composited + quantized + gutter-bled display image) plus the texel rect touched
    // since the last upload. Stamps only mark `dirty_rect`; `flush_paint` (once per
    // frame) recomposites/quantizes/bleeds/uploads just that rect, so paint cost is
    // proportional to brush area, not texture size. Non-stroke edits still go through
    // the full `refresh_display_texture`, which keeps `display_buf` in sync.
    display_buf: Vec<u8>,
    dirty_rect: Option<TexRect>,

    // Whether the brush paints the active layer's color or its reveal mask (G11),
    // and (in mask mode) whether it reveals (white) or hides (black).
    paint_target: PaintTarget,
    mask_reveal: bool,

    // Brush image: an optional image painted instead of a solid color. `brush_image_mode`
    // picks how: Tiled ("Brush") samples it UV-anchored and repeated `brush_tile` times,
    // so painting reveals a consistent material field; Stamp projects it as an oriented
    // decal through `stamp_angle` (radians, in the hit face's tangent plane), recoloured
    // to the swatch when `stamp_tint`. The flat 2D UV-editor brush is always tiled (it
    // has no surface frame to orient a stamp against).
    brush_material: Option<crate::material::Material>,
    brush_tile: f32,
    brush_image_mode: BrushImageMode,
    stamp_angle: f32,
    stamp_tint: bool,

    // Mirror painting (symmetry): when `Some`, each dab is also stamped at its
    // reflection across the model-symmetry plane for this axis (through the mesh
    // bbox center). Synced from the UI; `None` = off.
    symmetry: Option<SymmetryAxis>,

    // Face lock: when true, a stroke is confined to the facet(s) under its *first*
    // dab, so holding the mouse down and dragging onto neighbouring faces never
    // paints them. Synced from the UI. Uses the cached `fill_map` facet partition
    // — the same one the Face fill bucket fills.
    lock_face: bool,
    // The facet(s) the in-progress stroke is locked to: `None` until the first dab
    // of a face-locked stroke fixes them (one facet, plus the mirror's when
    // symmetry is on). Reset each `begin_stroke`.
    stroke_lock_facets: Option<Vec<i32>>,

    // Undo/redo over the layer stack. A stroke records one entry on release (via
    // `pending` + `stroke_dirty`); discrete layer ops record one each.
    history: History,
    // The layer stack as it was when the in-progress stroke began; committed to
    // history on `end_stroke` only if the stroke actually painted anything.
    pending: Option<Layers>,
    stroke_dirty: bool,

    // --- Dirty tracking for save / autosave (G31) ---
    // `edit_seq` increments on every document mutation (a committed stroke, a
    // checkpoint-recorded op, an undo/redo). `saved_seq` captures it at the last
    // *explicit* save (and on load), so `is_dirty` drives the title dot and the
    // close prompt. `autosave_seq` captures it at the last save *or* autosave, so
    // `needs_autosave` gates the timed recovery write without re-saving an
    // unchanged document.
    edit_seq: u64,
    saved_seq: u64,
    autosave_seq: u64,

    // The texture folder the project was painted against (brush browser source).
    // Document metadata: saved into the `.lowtex` file and restored on load. The
    // UI mirrors it in `UiState::brush_folder`; the app keeps the two in sync.
    texture_folder: Option<String>,

    depth_view: wgpu::TextureView,

    // Palette quantize (G8) — applied to the texture on the CPU, non-destructively
    // (paint_texture_cpu keeps full color; the GPU texture holds the quantized
    // result when enabled).
    palette: Palette,
    palette_settings: PaletteSettings,

    // egui overlay renderer; None in headless mode.
    egui_renderer: Option<egui_wgpu::Renderer>,

    // Surface/texture dimensions are clamped to this (GPU limit).
    max_texture_dim: u32,

    camera: Camera,
    // Physical pixels reserved on the left of the surface for the UI panel. The
    // scene viewport (and the camera aspect + pick mapping) shift right by this, so
    // the model sits centered in the region the panel doesn't cover, not the whole
    // window. Pushed from the UI each frame; 0 in headless.
    view_offset_x: f32,
    mesh: Mesh,
    bvh: Bvh,
    // Cached baked mesh maps (AO/edge) at the current resolution; invalidated on
    // model load or resolution change.
    mesh_maps: Option<crate::bake::MeshMaps>,
    // Cached UV-island map at the current resolution, driving the fill tools.
    // Invalidated alongside mesh_maps (geometry or resolution change).
    fill_map: Option<crate::fill::FillMap>,
    // Cached position-edge triangle adjacency, driving the cross-face brush
    // (`surface::splat`). Topology-only, so unlike the maps above it is *not*
    // invalidated on a resolution change — only when the geometry itself changes.
    surface_adj: Option<crate::surface::Adjacency>,
    // Reused flood scratch for `surface::splat`, so a stroke's many dabs don't each
    // allocate + zero a per-triangle visited buffer (the cost scaled with mesh size,
    // not brush size). Generation-stamped, so resetting between dabs is a counter bump.
    splat_scratch: crate::surface::SplatScratch,
    // Cached UV-coverage mask for island bleed (G18): which texels a triangle
    // covers, so we can dilate paint into the gutter. Lazily built in the display
    // path; invalidated on geometry change. RefCell so the &self display helper can
    // populate it.
    coverage: std::cell::RefCell<Option<Vec<bool>>>,
    // Ordered recipe of mesh-aware generator layers applied so far (G21). Recorded
    // as each generator runs so the current look can be saved as a reusable,
    // mesh-independent preset. Approximate: manual layer deletes aren't reflected.
    recipe: Vec<crate::preset::PresetLayer>,
    // Sun for the directional-light map (G-light): the direction *toward* the light
    // and whether it casts shadows. Pushed from the UI each frame; the light channel
    // in `mesh_maps` is (re)baked lazily for these whenever a Light effect is applied.
    sun_dir: glam::Vec3,
    sun_shadow: bool,

    // Monotonic counters the 2D UV editor polls to know when to re-copy renderer
    // state into the UI (the egui panel never sees the renderer directly). `paint_version`
    // bumps whenever `display_buf` changes (any composite/flush), so the panel re-uploads
    // the atlas image only on actual change; `topo_version` bumps whenever the mesh is
    // swapped, so the panel rebuilds the UV island wireframe only then.
    paint_version: u64,
    topo_version: u64,
}

impl Renderer {
    /// Window-backed renderer: presents to the window's surface.
    pub async fn new(window: Arc<Window>, mesh: Mesh) -> Self {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create surface");

        let adapter = request_adapter(&instance, Some(&surface)).await;
        let (device, queue) = request_device(&adapter).await;

        // Pick an sRGB surface format so our colors match the offscreen path.
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        // build() clamps config to the GPU's max texture size; configure the
        // surface afterwards with those clamped dimensions.
        let mut r = Self::build(device, queue, config, mesh);
        surface.configure(&r.device, &r.config);
        // egui overlay renders to the surface format, no depth, single-sampled.
        r.egui_renderer = Some(egui_wgpu::Renderer::new(
            &r.device,
            surface_format,
            None,
            1,
            false,
        ));
        r.window = Some(window);
        r.surface = Some(surface);
        r
    }

    /// Headless renderer: no surface, renders to an offscreen texture on `capture`.
    pub async fn new_headless(width: u32, height: u32, mesh: Mesh) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = request_adapter(&instance, None).await;
        let (device, queue) = request_device(&adapter).await;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: OFFSCREEN_FORMAT,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let mut r = Self::build(device, queue, config, mesh);
        // Headless can also draw the egui overlay (for screenshot verification).
        r.egui_renderer = Some(egui_wgpu::Renderer::new(
            &r.device,
            OFFSCREEN_FORMAT,
            None,
            1,
            false,
        ));
        r
    }

    /// Shared construction of all scene resources. The target color format is
    /// taken from `config.format` (surface format for window, OFFSCREEN_FORMAT
    /// for headless).
    fn build(
        device: wgpu::Device,
        queue: wgpu::Queue,
        mut config: wgpu::SurfaceConfiguration,
        mesh: Mesh,
    ) -> Self {
        // Never ask for a surface/depth texture larger than the GPU allows.
        let max_texture_dim = device.limits().max_texture_dimension_2d;
        config.width = config.width.clamp(1, max_texture_dim);
        config.height = config.height.clamp(1, max_texture_dim);
        let width = config.width;
        let height = config.height;
        let target_format = config.format;

        // Mesh → GPU buffers.
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex buffer"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index buffer"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let index_count = mesh.indices.len() as u32;

        // Base layer: faint checkerboard so the UV layout is visible pre-paint.
        let tex_size = TEX_SIZE;
        let base = make_checkerboard(tex_size, tex_size);
        let paint_texture_gpu = make_paint_texture(&device, &base);
        // `display_buf` and the GPU texture are populated by the `refresh_display_texture`
        // call at the end of `build` (which composites + uploads the base layer).
        let layers = Layers::new(base);

        // Default to nearest-neighbor — pure PSX, no filtering. Switchable at runtime
        // via set_texture_filter for painters who prefer a smoothed preview.
        let texture_filter = TextureFilter::default();
        let sampler = make_sampler(&device, texture_filter);

        // Uniform buffer: the view-projection matrix.
        let camera = Camera::new(width as f32 / height as f32);
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniform buffer"),
            contents: bytemuck::cast_slice(&[camera.view_proj().to_cols_array_2d()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = make_paint_bind_group(
            &device,
            &bind_group_layout,
            &uniform_buffer,
            &paint_texture_gpu,
            &sampler,
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("main shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/main.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("main pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let depth_view = make_depth_view(&device, width, height);

        // --- Viewport furniture: grid + compass line pipelines (G29) ---
        // A minimal bind group layout: just the view-proj uniform (no texture).
        let line_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("line bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let line_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lines shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/lines.wgsl").into()),
        });
        let line_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("line pipeline layout"),
            bind_group_layouts: &[&line_bgl],
            push_constant_ranges: &[],
        });
        // The grid: 1px lines, occluded by the mesh (depth Less, no write).
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line pipeline"),
            layout: Some(&line_pl),
            vertex: wgpu::VertexState {
                module: &line_shader,
                entry_point: "vs_main",
                buffers: &[LineVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &line_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // The compass: thick quads expanded in screen space (its own shader),
        // triangle list, always on top (depth Always, no write). Shares the line
        // bind-group layout (just the rotation-only view-proj uniform).
        let compass_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compass shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/compass.wgsl").into()),
        });
        let compass_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("compass pipeline"),
            layout: Some(&line_pl),
            vertex: wgpu::VertexState {
                module: &compass_shader,
                entry_point: "vs_main",
                buffers: &[CompassVertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &compass_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // the quads can wind either way as the camera orbits
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Grid shares the scene's view-proj; compass gets its own (rotation-only).
        let grid_bind_group = make_line_bind_group(&device, &line_bgl, &uniform_buffer);
        let grid_verts = build_grid();
        let grid_vertex_count = grid_verts.len() as u32;
        let grid_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grid vertices"),
            contents: bytemuck::cast_slice(&grid_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let compass_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("compass uniform"),
            contents: bytemuck::cast_slice(&[camera.gizmo_view_proj().to_cols_array_2d()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let compass_bind_group = make_line_bind_group(&device, &line_bgl, &compass_uniform);
        let compass_verts = build_compass();
        let compass_vertex_count = compass_verts.len() as u32;
        let compass_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("compass vertices"),
            contents: bytemuck::cast_slice(&compass_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Brush-cursor ring: a fixed line loop (constant vertex count) refreshed in
        // place each frame via `write_buffer`, so following the cursor never
        // reallocates. Capacity = one segment per `BRUSH_RING_SEGMENTS`, ×2 verts.
        let brush_ring_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brush ring vertices"),
            size: (BRUSH_RING_SEGMENTS as usize * 2 * std::mem::size_of::<LineVertex>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Palette quantize is applied to the texture on the CPU (G8).
        let palette = Palette::builtins().remove(0); // PICO-8 by default
        let palette_settings = PaletteSettings::default();

        // BVH over the mesh triangles for fast ray picking (G5).
        let bvh = Bvh::build(&mesh);

        let stroke_base = layers.active_tex().pixels.clone();
        let stroke_coverage = vec![0.0; (tex_size * tex_size) as usize];

        let mut r = Self {
            window: None,
            surface: None,
            device,
            queue,
            config,
            pipeline,
            vertex_buffer,
            index_buffer,
            index_count,
            line_pipeline,
            compass_pipeline,
            grid_bind_group,
            grid_vertex_buffer,
            grid_vertex_count,
            compass_uniform,
            compass_bind_group,
            compass_vertex_buffer,
            compass_vertex_count,
            outline_vertex_buffer: None,
            outline_vertex_count: 0,
            outline_facets: Vec::new(),
            brush_ring_buffer,
            brush_ring_count: 0,
            last_ring: None,
            preview_rect: None,
            last_preview: None,
            camera_gen: std::cell::Cell::new(0),
            brush_material_gen: 0,
            bg_color: [0.221, 0.272, 0.313], // sRGB of the old dark teal clear
            show_grid: true,
            uniform_buffer,
            bind_group,
            bind_group_layout,
            sampler,
            texture_filter,
            paint_texture_gpu,
            layers,
            tex_size,
            stroke_base,
            stroke_coverage,
            display_buf: Vec::new(),
            dirty_rect: None,
            paint_target: PaintTarget::Color,
            mask_reveal: false,
            brush_material: None,
            brush_tile: 4.0,
            brush_image_mode: BrushImageMode::Tiled,
            stamp_angle: 0.0,
            stamp_tint: false,
            symmetry: None,
            lock_face: false,
            stroke_lock_facets: None,
            history: History::new(HISTORY_CAP),
            pending: None,
            stroke_dirty: false,
            edit_seq: 0,
            saved_seq: 0,
            autosave_seq: 0,
            texture_folder: None,
            depth_view,
            palette,
            palette_settings,
            egui_renderer: None,
            max_texture_dim,
            camera,
            view_offset_x: 0.0,
            mesh,
            bvh,
            mesh_maps: None,
            fill_map: None,
            surface_adj: None,
            splat_scratch: crate::surface::SplatScratch::new(),
            coverage: std::cell::RefCell::new(None),
            recipe: Vec::new(),
            // Default sun matches the shader's fixed key light (warm, high, front-right).
            sun_dir: glam::Vec3::new(0.4, 0.8, 0.5).normalize(),
            sun_shadow: true,
            paint_version: 0,
            topo_version: 0,
        };
        // Composite the base layer into `display_buf` and upload it to the GPU texture.
        r.refresh_display_texture();
        r
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        // Clamp to the GPU's max texture size so a huge/HiDPI window can't make
        // Surface::configure fail (it did, at 2056 vs a 2048 cap).
        let width = width.min(self.max_texture_dim);
        let height = height.min(self.max_texture_dim);
        self.config.width = width;
        self.config.height = height;
        if let Some(surface) = &self.surface {
            surface.configure(&self.device, &self.config);
        }
        self.depth_view = make_depth_view(&self.device, width, height);

        // Aspect follows the scene viewport (the panel-free region), not the window.
        self.view_offset_x = self.view_offset_x.min(width as f32);
        self.update_aspect();
        self.update_uniforms();
    }

    /// Reserve `offset_px` physical pixels on the left of the surface for the UI
    /// panel. Shifts the scene viewport, camera aspect, and pick mapping right to
    /// match, so the model stays centered in the visible region rather than the
    /// whole window. Cheap and idempotent — pushed from the UI every frame.
    pub fn set_view_offset(&mut self, offset_px: f32) {
        let offset = offset_px.clamp(0.0, self.config.width as f32);
        if (offset - self.view_offset_x).abs() > f32::EPSILON {
            self.view_offset_x = offset;
            self.update_aspect();
            self.update_uniforms();
        }
    }

    /// The scene viewport in physical pixels — `(x, y, w, h)`. The left strip of
    /// width `view_offset_x` belongs to the UI panel; the 3D scene fills the rest.
    fn scene_viewport(&self) -> (f32, f32, f32, f32) {
        let w = (self.config.width as f32 - self.view_offset_x).max(1.0);
        (self.view_offset_x, 0.0, w, self.config.height.max(1) as f32)
    }

    /// The compass's square viewport in physical pixels — `(x, y, size)`, top-left
    /// origin (as wgpu viewports and the cursor both use). Tucked into the
    /// bottom-left of the scene region so the panel never covers it. `None` when
    /// the scene region is too small to fit the gizmo. Shared by the renderer (to
    /// place the draw) and the click handler (to hit-test the axes), so the two
    /// can't drift apart.
    fn compass_rect(&self) -> Option<(f32, f32, f32)> {
        let (vx, vy, vw, vh) = self.scene_viewport();
        let size = COMPASS_SIZE_PX.min(vw).min(vh);
        if vw < size + COMPASS_MARGIN_PX || vh < size + COMPASS_MARGIN_PX {
            return None;
        }
        let x = vx + COMPASS_MARGIN_PX;
        let y = vy + vh - size - COMPASS_MARGIN_PX;
        Some((x, y, size))
    }

    /// Hit-test a click (physical pixels, top-left origin) against the compass and,
    /// if it lands on an axis, snap the camera to look down that axis. Returns
    /// whether the click was consumed (so the caller skips painting the mesh
    /// behind the gizmo). The match is directional, not pixel-exact: we compare the
    /// click's direction from the gizmo center against each projected axis, so the
    /// whole arm — not just its tip — is a target.
    pub fn click_compass(&mut self, mouse_px: (f32, f32)) -> bool {
        let Some(dir) = self.compass_axis_at(Vec2::new(mouse_px.0, mouse_px.1)) else {
            return false;
        };
        self.camera.look_from(dir);
        self.update_uniforms();
        true
    }

    /// Which world axis (a unit direction) a click selects on the compass, if any.
    /// Projects each of the six axis directions through the gizmo's rotation-only
    /// view-proj into the square viewport, then picks the one whose screen
    /// direction the click best aligns with. A small dead zone at the center
    /// (where the axes overlap) selects nothing.
    fn compass_axis_at(&self, mouse_px: Vec2) -> Option<glam::Vec3> {
        let (cx, cy, size) = self.compass_rect()?;
        let lx = mouse_px.x - cx;
        let ly = mouse_px.y - cy;
        if lx < 0.0 || ly < 0.0 || lx > size || ly > size {
            return None;
        }
        // Centered coordinates, y up (NDC convention) to match the projection.
        let click = Vec2::new((lx / size) * 2.0 - 1.0, 1.0 - (ly / size) * 2.0);
        if click.length() < 0.18 {
            return None; // center dead zone
        }
        let click_dir = click.normalize();

        let vp = self.camera.gizmo_view_proj();
        let axes = [
            glam::Vec3::X,
            glam::Vec3::NEG_X,
            glam::Vec3::Y,
            glam::Vec3::NEG_Y,
            glam::Vec3::Z,
            glam::Vec3::NEG_Z,
        ];
        let mut best: Option<glam::Vec3> = None;
        let mut best_dot = 0.5_f32; // require within ~60° of an axis to count
        for axis in axes {
            let tip = vp.project_point3(axis); // NDC, y up
            let screen = Vec2::new(tip.x, tip.y);
            if screen.length() < 1e-4 {
                continue; // axis points (near) straight at the camera — ambiguous
            }
            let d = click_dir.dot(screen.normalize());
            if d > best_dot {
                best_dot = d;
                best = Some(axis);
            }
        }
        best
    }

    /// Match the camera's aspect ratio to the scene viewport (not the window).
    fn update_aspect(&mut self) {
        let (_, _, w, h) = self.scene_viewport();
        self.camera.aspect = w / h;
    }

    /// Write the view-projection matrix to the GPU. Also refreshes the compass
    /// gizmo's rotation-only matrix, so it tracks the camera as it orbits.
    fn update_uniforms(&self) {
        // The camera/view moved: invalidate the brush ring + ghost keys (the cursor
        // now lands on a different surface point).
        self.camera_gen.set(self.camera_gen.get().wrapping_add(1));
        let view_proj = self.camera.view_proj();
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[view_proj.to_cols_array_2d()]),
        );
        self.queue.write_buffer(
            &self.compass_uniform,
            0,
            bytemuck::cast_slice(&[self.camera.gizmo_view_proj().to_cols_array_2d()]),
        );
    }

    /// Set the viewport background color (sRGB, as the picker shows it). Stored
    /// and converted to linear at clear time.
    pub fn set_bg_color(&mut self, srgb: [f32; 3]) {
        self.bg_color = srgb;
    }

    /// Toggle the ground grid.
    pub fn set_show_grid(&mut self, on: bool) {
        self.show_grid = on;
    }

    /// How far to bleed island colors into the gutter. `LOWTEX_BLEED_PAD` overrides
    /// the default (`BLEED_PAD`) for tuning/debugging.
    fn bleed_pad(&self) -> u32 {
        std::env::var("LOWTEX_BLEED_PAD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(BLEED_PAD)
    }

    /// Lazily build (and cache) the UV-coverage mask for the current resolution.
    fn ensure_coverage(&self) {
        let mut slot = self.coverage.borrow_mut();
        let need = slot
            .as_ref()
            .is_none_or(|c| c.len() != (self.tex_size * self.tex_size) as usize);
        if need {
            *slot = Some(crate::bleed::coverage(&self.mesh, self.tex_size));
        }
    }

    /// Composite the layers, palette-quantize (if enabled), then bleed island
    /// colors into the UV gutter (G18) so nearest sampling at a seam can't reveal
    /// background. Shared by the upload and export paths so the PNG matches the
    /// model. Non-destructive — the layers keep full color.
    fn composite_display(&self) -> Vec<u8> {
        let mut display = self.layers.composite();
        if self.palette_settings.enabled && !self.palette.colors.is_empty() {
            self.palette.quantize_rgba(
                &mut display,
                self.tex_size,
                self.palette_settings.dither,
                self.palette_settings.dither_strength,
            );
        }
        self.ensure_coverage();
        if let Some(cov) = self.coverage.borrow().as_ref() {
            crate::bleed::dilate(&mut display, cov, self.tex_size, self.bleed_pad());
        }
        display
    }

    /// Composite the whole image into `display_buf` and upload it. The path every
    /// non-stroke edit (fills, layer/effect ops, undo, palette, load, resize) uses,
    /// so `display_buf` stays a faithful mirror of the GPU texture for them too.
    fn refresh_display_texture(&mut self) {
        self.display_buf = self.composite_display();
        self.paint_version = self.paint_version.wrapping_add(1);
        upload_pixels(
            &self.queue,
            &self.paint_texture_gpu,
            &self.display_buf,
            self.tex_size,
            self.tex_size,
        );
        // A full upload supersedes any pending region (and a resize would otherwise
        // leave a stale, possibly out-of-bounds, rect). The same goes for the brush
        // preview's ghost region — the GPU now matches `display_buf` everywhere.
        self.dirty_rect = None;
        self.preview_rect = None;
        // The committed image (and possibly the geometry — load/unwrap/resize all land
        // here) changed under the cursor, so rebuild the ghost and ring next frame.
        self.last_preview = None;
        self.last_ring = None;
    }

    /// Recompute just the texels a stroke touched (the dirty-rectangle path). Produces
    /// a `display_buf` byte-identical to a full `composite_display`, but works only over
    /// `rect` (+ bleed margin) so cost scales with brush area, not texture size.
    ///
    /// `rect` is the union of stamp footprints since the last upload. We process a
    /// region padded by `2*pad` (so the kept texels' gutter-dilation rings stay inside
    /// it), then upload the inner `rect + pad`. The outer `pad` ring of the processed
    /// region is unaffected by the stroke but can be miscomputed by the region dilate
    /// (rings exiting the region), so we snapshot and restore it — keeping `display_buf`
    /// exact.
    fn refresh_display_region(&mut self, rect: TexRect) {
        let size = self.tex_size;
        let pad = self.bleed_pad();
        // A neighbourhood effect (blur, warp) spreads a stamp's change outward, so
        // widen the affected region by the largest active spread. Zero in the common
        // case (no effects), which is then a pure brush-sized region.
        let max_spread = self
            .layers
            .layers
            .iter()
            .filter(|l| l.visible && l.opacity > 0.0)
            .flat_map(|l| &l.effects)
            .map(|fx| fx.display_spread())
            .max()
            .unwrap_or(0);
        // `upload` = texels whose final display value the stamp can change (brush
        // footprint + effect spread + gutter bleed). `proc` extends one more `pad` so
        // the uploaded texels' dilation rings stay inside the processed region.
        let upload = rect.expanded(pad + max_spread, size);
        let proc = rect.expanded(2 * pad + max_spread, size);

        // Snapshot the processed region so the outer margin can be restored after dilate.
        let before = copy_region(&self.display_buf, size, proc);

        self.layers
            .composite_into_region(&mut self.display_buf, proc);
        if self.palette_settings.enabled && !self.palette.colors.is_empty() {
            self.palette.quantize_region(
                &mut self.display_buf,
                size,
                proc,
                self.palette_settings.dither,
                self.palette_settings.dither_strength,
            );
        }
        self.ensure_coverage();
        if let Some(cov) = self.coverage.borrow().as_ref() {
            crate::bleed::dilate_region(&mut self.display_buf, cov, size, pad, proc);
        }
        // Restore the margin (processed but not uploaded): unchanged by the stroke, but
        // possibly miscomputed by the region dilate.
        restore_margin(&mut self.display_buf, &before, size, proc, upload);

        upload_region(
            &self.queue,
            &self.paint_texture_gpu,
            &self.display_buf,
            size,
            upload,
        );
        self.paint_version = self.paint_version.wrapping_add(1);
    }

    /// Apply any pending stroke dirty-rect to the GPU, once per frame. Cheap no-op when
    /// nothing was painted since the last call. Called from the app's redraw and before
    /// headless `capture`.
    pub fn flush_paint(&mut self) {
        if let Some(rect) = self.dirty_rect.take() {
            self.refresh_display_region(rect);
        }
    }

    /// Replace the active palette (live swap) and refresh the display.
    pub fn set_palette(&mut self, palette: Palette) {
        self.palette = palette;
        self.refresh_display_texture();
    }

    /// Update quantize/dither settings and refresh the display. The App pushes
    /// these every frame, so skip the (potentially multi-millisecond) recomposite
    /// when nothing actually changed — otherwise idle frames pay a full composite +
    /// GPU upload for no reason.
    pub fn set_palette_settings(&mut self, settings: PaletteSettings) {
        if settings == self.palette_settings {
            return;
        }
        self.palette_settings = settings;
        self.refresh_display_texture();
    }

    /// Switch how the painted texture is filtered onto the model (G30). Rebuilds the
    /// sampler and the bind group it's bound into; no texture recomposite needed.
    /// Skips the work when the mode is unchanged (the App pushes it every frame).
    pub fn set_texture_filter(&mut self, filter: TextureFilter) {
        if filter == self.texture_filter {
            return;
        }
        self.texture_filter = filter;
        self.sampler = make_sampler(&self.device, filter);
        self.bind_group = make_paint_bind_group(
            &self.device,
            &self.bind_group_layout,
            &self.uniform_buffer,
            &self.paint_texture_gpu,
            &self.sampler,
        );
    }

    /// The composited (quantized + gutter-bled, if enabled) texture as currently
    /// shown — used for export so the PNG matches the model.
    pub fn display_pixels(&self) -> Vec<u8> {
        self.composite_display()
    }

    /// The active palette (for the UI swatch row).
    pub fn palette(&self) -> &Palette {
        &self.palette
    }

    /// Build a palette from an image file via median-cut and make it active.
    pub fn generate_palette_from_image(&mut self, path: &str, n: usize) -> Result<(), String> {
        let img = image::open(path).map_err(|e| format!("failed to open image: {e}"))?;
        let rgba = img.to_rgba8();
        self.set_palette(Palette::from_image_median_cut(&rgba, n));
        Ok(())
    }

    // --- Layer stack (G10) ---

    /// Read-only view of the layer stack for the UI (top-first not assumed; index
    /// 0 is the bottom). Returns the active index too.
    pub fn layers(&self) -> &Layers {
        &self.layers
    }

    // --- Undo / redo (history over the layer stack) ---

    pub fn can_undo(&self) -> bool {
        self.history.can_undo()
    }

    pub fn can_redo(&self) -> bool {
        self.history.can_redo()
    }

    /// Snapshot the current layer stack as an undo point (and drop the redo
    /// future). Discrete edits call this before mutating; continuous edits (an
    /// opacity-slider drag) checkpoint once at drag start via the UI, and strokes
    /// use the `begin_stroke`/`end_stroke` pair instead.
    pub fn checkpoint(&mut self) {
        let before = self.layers.clone();
        self.history.record(before);
        self.mark_edited();
    }

    /// Step back to the previous layer state, if any.
    pub fn undo(&mut self) {
        let current = self.layers.clone();
        if let Some(prev) = self.history.undo(current) {
            self.layers = prev;
            self.restore_after_history();
            self.mark_edited();
        }
    }

    /// Step forward to the next layer state, if any.
    pub fn redo(&mut self) {
        let current = self.layers.clone();
        if let Some(next) = self.history.redo(current) {
            self.layers = next;
            self.restore_after_history();
            self.mark_edited();
        }
    }

    /// Note that the document changed, so the next save/autosave/close knows there
    /// is unsaved work. The single funnel for every committed mutation.
    fn mark_edited(&mut self) {
        self.edit_seq = self.edit_seq.wrapping_add(1);
    }

    /// Whether the document differs from the last *explicit* save (drives the
    /// window title's unsaved-changes dot and the close-confirmation prompt).
    pub fn is_dirty(&self) -> bool {
        self.edit_seq != self.saved_seq
    }

    /// Whether the document changed since the last save *or* autosave, so a timed
    /// autosave has something new to capture.
    pub fn needs_autosave(&self) -> bool {
        self.edit_seq != self.autosave_seq
    }

    /// Re-sync GPU + stroke buffers after the stack is replaced wholesale by
    /// undo/redo. A restored state may be at a different resolution (undoing a
    /// resize) or have a different active index, so handle both.
    fn restore_after_history(&mut self) {
        self.layers.active = self.layers.active.min(self.layers.layers.len() - 1);
        if self.tex_size != self.layers.size() {
            // rebuild_paint_gpu re-snapshots stroke buffers and refreshes for us.
            self.mesh_maps = None; // baked at the previous resolution
            self.fill_map = None; // rasterized at the previous resolution
            self.rebuild_paint_gpu();
        } else {
            self.stroke_base = self.target_pixels().to_vec();
            self.stroke_coverage = vec![0.0; (self.tex_size * self.tex_size) as usize];
            self.refresh_display_texture();
        }
    }

    /// The pixels of the current paint target (active layer's color or mask).
    fn target_pixels(&self) -> &[u8] {
        match self.paint_target {
            PaintTarget::Color => &self.layers.active_tex().pixels,
            PaintTarget::Mask => &self.layers.active_mask().pixels,
        }
    }

    /// Re-snapshot the stroke base from the current paint target (after the active
    /// layer or paint target changes).
    fn resync_stroke_base(&mut self) {
        self.stroke_base = self.target_pixels().to_vec();
    }

    /// Switch between painting the layer's color and its reveal mask.
    pub fn set_paint_target(&mut self, target: PaintTarget) {
        if target != self.paint_target {
            self.paint_target = target;
            self.resync_stroke_base();
        }
    }

    /// In mask mode, whether the brush reveals (white) or hides (black).
    pub fn set_mask_reveal(&mut self, reveal: bool) {
        self.mask_reveal = reveal;
    }

    /// How many times the brush image tiles across the 0–1 UV space.
    pub fn set_brush_tile(&mut self, tile: f32) {
        self.brush_tile = tile;
    }

    /// Mirror-painting axis (`None` = off). Synced from the UI.
    pub fn set_symmetry(&mut self, axis: Option<SymmetryAxis>) {
        self.symmetry = axis;
    }

    /// Lock each dab to the flat face under its hit point (`true` = on). Synced
    /// from the UI.
    pub fn set_lock_face(&mut self, on: bool) {
        self.lock_face = on;
    }

    /// Refresh the active-face outline for this frame. `cursor` is the current
    /// mouse pixel when a painting tool with face-lock is active, else `None` (the
    /// outline is then cleared). Mid-stroke the locked face is outlined; otherwise
    /// the face under the cursor is previewed. Rebuilds GPU geometry only when the
    /// highlighted facet set actually changes, so it's cheap to call every frame.
    pub fn set_face_outline(&mut self, cursor: Option<(f32, f32)>) {
        let facets = self.outline_target_facets(cursor);
        if facets == self.outline_facets {
            return; // same face as last frame — keep the existing buffer
        }
        self.outline_facets = facets.clone();
        self.build_outline(&facets);
    }

    /// Which facet(s) the outline should trace this frame: the stroke's locked
    /// face while painting, else the face under the cursor (preview). Empty when
    /// face-lock is off, there's no cursor, or the ray misses the mesh.
    fn outline_target_facets(&mut self, cursor: Option<(f32, f32)>) -> Vec<i32> {
        if !self.lock_face {
            return Vec::new();
        }
        // Mid-stroke: trace the face the stroke locked onto, even as the cursor
        // wanders off it. (`pending` marks an in-progress stroke.)
        if self.pending.is_some() {
            if let Some(facets) = &self.stroke_lock_facets {
                if !facets.is_empty() {
                    return facets.clone();
                }
            }
        }
        let Some((x, y)) = cursor else {
            return Vec::new();
        };
        let ray = self.pick_ray(Vec2::new(x, y));
        let Some(hit) = self.bvh.pick(&ray) else {
            return Vec::new();
        };
        self.ensure_fill_map();
        match self.fill_map.as_ref().unwrap().facet_for_tri(hit.tri) {
            Some(f) => vec![f as i32],
            None => Vec::new(),
        }
    }

    /// Rebuild the outline vertex buffer: the boundary edges of `facets` — every
    /// triangle edge whose neighbour lies in a different facet (or off the mesh) —
    /// as a line list. Endpoints are nudged out along the face normal by a hair so
    /// the lines sit just in front of the surface and aren't z-fought away.
    fn build_outline(&mut self, facets: &[i32]) {
        if facets.is_empty() {
            self.outline_vertex_buffer = None;
            self.outline_vertex_count = 0;
            return;
        }
        self.ensure_fill_map();
        self.ensure_surface_adj();
        let (mn, mx) = self.mesh.bounds();
        let eps = (mx - mn).length().max(1e-3) * 1.5e-3;
        const COLOR: [f32; 3] = [1.0, 0.78, 0.12]; // amber, linear RGB (see lines.wgsl)

        let verts = {
            let map = self.fill_map.as_ref().unwrap();
            let adj = self.surface_adj.as_ref().unwrap();
            let mut verts: Vec<LineVertex> = Vec::new();
            for ti in 0..self.mesh.indices.len() / 3 {
                let f = map.facet_of_tri[ti] as i32;
                if !facets.contains(&f) {
                    continue;
                }
                let i = ti * 3;
                let idx = [
                    self.mesh.indices[i] as usize,
                    self.mesh.indices[i + 1] as usize,
                    self.mesh.indices[i + 2] as usize,
                ];
                let p = [
                    glam::Vec3::from(self.mesh.vertices[idx[0]].position),
                    glam::Vec3::from(self.mesh.vertices[idx[1]].position),
                    glam::Vec3::from(self.mesh.vertices[idx[2]].position),
                ];
                let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero() * eps;
                // The adjacency edge slot `e` spans triangle vertices e and (e+1)%3.
                for e in 0..3 {
                    let nb = adj.neighbors()[ti][e];
                    let boundary = nb < 0 || map.facet_of_tri[nb as usize] as i32 != f;
                    if boundary {
                        let a = p[e] + n;
                        let b = p[(e + 1) % 3] + n;
                        verts.push(LineVertex {
                            position: a.into(),
                            color: COLOR,
                        });
                        verts.push(LineVertex {
                            position: b.into(),
                            color: COLOR,
                        });
                    }
                }
            }
            verts
        };

        self.outline_vertex_count = verts.len() as u32;
        self.outline_vertex_buffer = (!verts.is_empty()).then(|| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("face outline vertices"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                })
        });
    }

    /// Refresh the brush-cursor ring for this frame. `cursor` is the mouse pixel
    /// when a painting tool is active, else `None` (the ring is then hidden). The
    /// ring is a circle of the brush's true world-space footprint laid on the mesh
    /// surface at the cursor, tilted into the hit face's tangent plane, so its size
    /// tracks both the Size slider and the local texel density. Cheap to call every
    /// frame: it rewrites a fixed buffer in place rather than reallocating.
    pub fn set_brush_cursor(&mut self, cursor: Option<(f32, f32)>, brush: &Brush) {
        let Some((x, y)) = cursor else {
            self.brush_ring_count = 0;
            self.last_ring = None;
            return;
        };
        // Skip the pick + buffer rewrite when nothing the ring depends on changed
        // (cursor, camera, brush radius) — the common case while the cursor sits still.
        let key = (x, y, self.camera_gen.get(), brush.radius);
        if self.last_ring == Some(key) {
            return;
        }
        self.last_ring = Some(key);
        let ray = self.pick_ray(Vec2::new(x, y));
        let Some(hit) = self.bvh.pick(&ray) else {
            self.brush_ring_count = 0;
            return;
        };
        // Brush radius (texels) → world units, the same mapping the surface dab uses.
        let radius = self.world_brush_radius(hit.tri, brush.radius);

        // An orthonormal tangent basis in the hit face's plane: pick any axis not
        // parallel to the normal, then two cross products give perpendicular u, v.
        let n = hit.normal.normalize_or_zero();
        let seed = if n.x.abs() < 0.9 {
            glam::Vec3::X
        } else {
            glam::Vec3::Y
        };
        let u = n.cross(seed).normalize_or_zero();
        let v = n.cross(u);
        // Lift the ring just off the surface so it isn't z-fought by the face it sits on.
        let (mn, mx) = self.mesh.bounds();
        let lift = n * (mx - mn).length().max(1e-3) * 1.5e-3;
        let center = hit.pos + lift;

        const COLOR: [f32; 3] = [0.95, 0.95, 0.98]; // near-white, linear RGB
        let seg = BRUSH_RING_SEGMENTS;
        let mut verts: Vec<LineVertex> = Vec::with_capacity(seg as usize * 2);
        let point = |k: u32| -> [f32; 3] {
            let a = std::f32::consts::TAU * (k % seg) as f32 / seg as f32;
            (center + (u * a.cos() + v * a.sin()) * radius).into()
        };
        // Line list: each segment connects point k to k+1 (wrapping closed).
        for k in 0..seg {
            verts.push(LineVertex {
                position: point(k),
                color: COLOR,
            });
            verts.push(LineVertex {
                position: point(k + 1),
                color: COLOR,
            });
        }
        self.queue
            .write_buffer(&self.brush_ring_buffer, 0, bytemuck::cast_slice(&verts));
        self.brush_ring_count = verts.len() as u32;
    }

    /// Texel brush radius → a world-space sphere radius via the (constant) local
    /// texel density at triangle `tri`. Falls back to a bbox-derived radius when the
    /// triangle is degenerate in UV. Shared by the surface dab, the cursor ring, and
    /// the stamp preview so all three agree on how big the brush is.
    fn world_brush_radius(&self, tri: u32, radius_texels: f32) -> f32 {
        crate::surface::world_radius(&self.mesh, tri, radius_texels, self.tex_size).unwrap_or_else(
            || {
                let (mn, mx) = self.mesh.bounds();
                (mx - mn).length().max(1e-3) * (radius_texels / self.tex_size as f32)
            },
        )
    }

    /// The brush's on-screen radius in physical pixels at `mouse_px`, or `None` if
    /// the cursor isn't over the mesh. Used to space stroke dabs: a large brush's
    /// footprint covers many screen pixels, so dabs can be spaced far apart and
    /// still overlap — re-splatting the whole footprint every 2px (the old fixed
    /// step) is almost entirely redundant work. Projects the hit point and a point
    /// offset by the world brush radius along a screen-horizontal world direction
    /// (from the inverse projection, so a tilted face doesn't foreshorten it).
    fn brush_screen_radius(&self, mouse_px: Vec2, brush: &Brush) -> Option<f32> {
        let ray = self.pick_ray(mouse_px);
        let hit = self.bvh.pick(&ray)?;
        let radius_world = self.world_brush_radius(hit.tri, brush.radius);
        let vp = self.camera.view_proj();
        let inv = vp.inverse();
        // A world direction that runs left-right across the screen at mid-depth.
        let a = inv.project_point3(glam::Vec3::new(0.0, 0.0, 0.5));
        let b = inv.project_point3(glam::Vec3::new(1.0, 0.0, 0.5));
        let right = (b - a).normalize_or_zero();
        let (_, _, vw, vh) = self.scene_viewport();
        let c0 = vp.project_point3(hit.pos);
        let c1 = vp.project_point3(hit.pos + right * radius_world);
        // NDC delta → pixels (NDC spans -1..1 across the viewport, so ×0.5×extent).
        let dx = (c1.x - c0.x) * 0.5 * vw;
        let dy = (c1.y - c0.y) * 0.5 * vh;
        Some((dx * dx + dy * dy).sqrt())
    }

    /// The texels a single dab from `hit` would cover, with coverage — the read-only
    /// half of `surface_dab` (no blending, no dirty rect). `&mut` only to reuse the
    /// flood scratch; `ensure_surface_adj` must have run.
    fn dab_splats(&mut self, hit: &Hit, brush: &Brush) -> Vec<(usize, f32)> {
        let radius_world = self.world_brush_radius(hit.tri, brush.radius);
        let adj = self.surface_adj.as_ref().unwrap();
        crate::surface::splat(
            &self.mesh,
            adj,
            hit,
            radius_world,
            brush.opacity,
            brush.hardness,
            self.tex_size,
            &mut self.splat_scratch,
        )
    }

    /// Refresh the brush stamp preview for this frame: a translucent ghost of the dab
    /// under the cursor, written into the GPU paint texture so the painter can see the
    /// color/image and footprint a click would lay down — without committing it. The
    /// layers and `display_buf` are never touched; the previous ghost is reverted from
    /// `display_buf` first, so the preview leaves no trace. Shown only for the
    /// solid/image brush while hovering (not mid-stroke) and painting color. `cursor`
    /// is `None` (or off the mesh) to clear it.
    pub fn set_brush_preview(&mut self, cursor: Option<(f32, f32)>, brush: &Brush) {
        let size = self.tex_size;
        // Everything the ghost depends on this frame (`None` = no cursor → no ghost).
        let key = cursor.map(|(x, y)| PreviewKey {
            cursor: (x, y),
            cam: self.camera_gen.get(),
            brush: *brush,
            tile: self.brush_tile,
            material_gen: self.brush_material_gen,
            symmetry: self.symmetry,
            lock_face: self.lock_face,
            target: self.paint_target,
            image_mode: self.brush_image_mode,
            stamp_angle: self.stamp_angle,
            stamp_tint: self.stamp_tint,
        });
        // Unchanged since last frame → the GPU already shows the correct thing (ghost
        // or none), so skip the pick + (large-brush-expensive) re-splat + upload. A
        // committed-image change resets `last_preview` to `None`, forcing a recompute.
        if key == self.last_preview {
            return;
        }
        self.last_preview = key;
        // Revert last frame's ghost so the texture matches the committed display again.
        if let Some(rect) = self.preview_rect.take() {
            upload_region(
                &self.queue,
                &self.paint_texture_gpu,
                &self.display_buf,
                size,
                rect,
            );
        }
        let Some((x, y)) = cursor else { return };
        // A real stroke already shows the truth; a mask ghost (white/black) isn't
        // meaningful, and an image brush still previews its sampled color.
        // The eraser removes paint rather than laying color, so a color ghost would
        // mislead; the cursor ring alone marks the footprint until the stroke commits.
        if self.pending.is_some() || self.paint_target != PaintTarget::Color || brush.erase {
            return;
        }
        let ray = self.pick_ray(Vec2::new(x, y));
        let Some(hit) = self.bvh.pick(&ray) else {
            return;
        };
        self.ensure_surface_adj();

        // The dab's footprint, plus the symmetric dab's, exactly as a click would lay
        // it — as `(texel, coverage, rgb)` so the stamp (per-texel decal color) and the
        // tiled/solid brush (one color, sampled or flat) share one ghost render below.
        let mirror = self
            .symmetry
            .and_then(|axis| self.bvh.pick(&self.mirror_ray(&ray, axis)));
        let mut preview: Vec<(usize, f32, [u8; 3])> = if self.stamp_active() {
            let mut s = self.decal_splats(&hit, brush);
            if let Some(m) = &mirror {
                s.extend(self.decal_splats(m, brush));
            }
            s.into_iter().map(|(t, (a, rgb))| (t, a, rgb)).collect()
        } else {
            let mut s = self.dab_splats(&hit, brush);
            if let Some(m) = &mirror {
                s.extend(self.dab_splats(m, brush));
            }
            // Tiled brush: the image sampled UV-anchored; else the flat brush color.
            let tile = self.brush_tile;
            let solid = brush.color_u8();
            let mat = self.brush_material.as_ref();
            s.into_iter()
                .map(|(texel, a)| {
                    let (tx, ty) = (texel as u32 % size, texel as u32 / size);
                    let src = match mat {
                        Some(m) => {
                            let c =
                                m.sample(tx as f32 / size as f32, ty as f32 / size as f32, tile);
                            [c[0], c[1], c[2]]
                        }
                        None => solid,
                    };
                    (texel, a, src)
                })
                .collect()
        };
        // Face lock confines the ghost to the face the stroke would lock onto.
        if self.lock_face {
            self.ensure_fill_map();
            let map = self.fill_map.as_ref().unwrap();
            if let Some(f) = map.facet_for_tri(hit.tri).map(|f| f as i32) {
                let tf = &map.texel_facet;
                preview.retain(|&(t, _, _)| tf.get(t).copied() == Some(f));
            }
        }
        if preview.is_empty() {
            return;
        }

        // Bounding box of the touched texels (can span faces when the dab wraps a seam).
        let (mut x0, mut y0, mut x1, mut y1) = (size, size, 0u32, 0u32);
        for &(texel, _, _) in &preview {
            let (tx, ty) = (texel as u32 % size, texel as u32 / size);
            x0 = x0.min(tx);
            y0 = y0.min(ty);
            x1 = x1.max(tx + 1);
            y1 = y1.max(ty + 1);
        }
        let rect = TexRect { x0, y0, x1, y1 };

        // Blend the ghost over a copy of the committed region, then upload just that box.
        let mut packed = copy_region(&self.display_buf, size, rect);
        let rw = rect.width();
        for &(texel, a, src) in &preview {
            let (tx, ty) = (texel as u32 % size, texel as u32 / size);
            let alpha = (a * BRUSH_PREVIEW_ALPHA).clamp(0.0, 1.0);
            let li = (((ty - rect.y0) * rw + (tx - rect.x0)) * 4) as usize;
            for c in 0..3 {
                let dst = packed[li + c] as f32;
                packed[li + c] = (dst * (1.0 - alpha) + src[c] as f32 * alpha).round() as u8;
            }
        }
        upload_packed(&self.queue, &self.paint_texture_gpu, &packed, rect);
        self.preview_rect = Some(rect);
    }

    /// Load an image for the brush to paint (UV-tiled) instead of solid color.
    pub fn load_brush_material(&mut self, path: &str) -> Result<(), String> {
        self.brush_material = Some(crate::material::Material::load(path)?);
        self.brush_material_gen = self.brush_material_gen.wrapping_add(1);
        Ok(())
    }

    /// Drop the loaded brush image; the brush reverts to painting its solid color.
    pub fn clear_brush_material(&mut self) {
        self.brush_material = None;
        self.brush_material_gen = self.brush_material_gen.wrapping_add(1);
    }

    /// How the loaded brush image is applied — tiled "Brush" paint or an oriented "Stamp".
    pub fn set_brush_image_mode(&mut self, mode: BrushImageMode) {
        self.brush_image_mode = mode;
    }

    /// The stamp's rotation in its tangent plane (radians) and whether it's recoloured
    /// to the brush swatch (true) or keeps the image's own RGB (false).
    pub fn set_stamp_options(&mut self, angle_rad: f32, tint: bool) {
        self.stamp_angle = angle_rad;
        self.stamp_tint = tint;
    }

    /// True when a brush image is loaded *and* set to Stamp mode — the paint path then
    /// routes through the decal projection instead of the tiled splat.
    fn stamp_active(&self) -> bool {
        self.brush_image_mode == BrushImageMode::Stamp && self.brush_material.is_some()
    }

    /// The oriented tangent frame a decal is projected through: the surface point and
    /// two orthonormal in-plane axes (`t` = image-right, `b` = image-down), rotated by
    /// `stamp_angle`. The frame is *world-up aligned*: world-up is projected onto the hit
    /// face's tangent plane so the stamp's top points toward the world sky regardless of
    /// camera or unwrap. On a floor/ceiling (normal ∥ world-up) the projection vanishes,
    /// so we fall back to world-Z for a stable orientation. (The brush-cursor ring still
    /// uses an arbitrary normal-seed basis — fine, since a ring is rotationally symmetric,
    /// but a seed basis spins a decal unpredictably across faces, which is what we avoid.)
    fn decal_frame(&self, hit: &Hit) -> (glam::Vec3, glam::Vec3, glam::Vec3) {
        let n = hit.normal.normalize_or_zero();
        let mut up_p = glam::Vec3::Y - n * glam::Vec3::Y.dot(n);
        if up_p.length_squared() < 1e-6 {
            up_p = glam::Vec3::Z - n * glam::Vec3::Z.dot(n);
        }
        let up_p = up_p.normalize_or_zero();
        let right = up_p.cross(n).normalize_or_zero(); // world-right within the plane
        let down = -up_p; // image rows run downward, so image-down is world-down
                          // Rotate the frame about the normal by the stamp angle.
        let (s, c) = self.stamp_angle.sin_cos();
        let t = right * c + down * s;
        let b = -right * s + down * c;
        (hit.pos, t, b)
    }

    /// The texels one decal placement at `hit` would cover, as `(texel, (coverage,
    /// rgb))`. Floods the surface to the decal's square reach, projects each texel into
    /// the oriented tangent frame, and samples the image's alpha for coverage (×
    /// opacity); RGB is the brush swatch when tinting, else the image's own color.
    /// Texels outside the decal's [0,1]² footprint are dropped. The `(usize, T)` shape
    /// lets the shared `retain_locked` face-lock filter apply unchanged.
    /// `ensure_surface_adj` must run first.
    fn decal_splats(&mut self, hit: &Hit, brush: &Brush) -> Vec<(usize, (f32, [u8; 3]))> {
        let Some(mat) = self.brush_material.as_ref() else {
            return Vec::new();
        };
        let radius = self.world_brush_radius(hit.tri, brush.radius);
        // `brush_tile` doubles as the stamp's scale-down: 1 fills the footprint, larger
        // shrinks the image toward the centre (the rest of the footprint stays
        // transparent). Reusing the Tile control keeps one "how small" knob across modes.
        let scale = self.brush_tile.max(1.0);
        // The image now occupies a square of half-side radius/scale; its corners reach
        // (radius/scale)·√2. Flood that far (plus a hair) so the whole footprint is
        // reachable, then clip to the square.
        let half = radius / scale;
        let reach = half * std::f32::consts::SQRT_2 * 1.02;
        let (center, t, b) = self.decal_frame(hit);
        let inv = 1.0 / (2.0 * half).max(1e-6);
        let solid = brush.color_u8();
        let tint = self.stamp_tint;

        let adj = self.surface_adj.as_ref().unwrap();
        let pts = crate::surface::splat_world(
            &self.mesh,
            adj,
            hit,
            reach,
            self.tex_size,
            &mut self.splat_scratch,
        );
        let mut out = Vec::with_capacity(pts.len());
        for (texel, wp) in pts {
            let d = wp - center;
            // Map onto the decal's UV: centre of the image at the hit, ±radius → 0/1.
            let u = d.dot(t) * inv + 0.5;
            let v = d.dot(b) * inv + 0.5;
            if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
                continue;
            }
            let c = mat.sample(u, v, 1.0); // tile=1: one placement, no repeat
            let a = (c[3] as f32 / 255.0) * brush.opacity;
            if a > 0.0 {
                let rgb = if tint { solid } else { [c[0], c[1], c[2]] };
                out.push((texel, (a, rgb)));
            }
        }
        out
    }

    /// A small antialiased preview (≤`max`×`max`, RGBA8) of the loaded brush image,
    /// or `None` if the brush is painting solid color. For the UI swatch.
    pub fn brush_thumbnail(&self, max: u32) -> Option<(u32, u32, Vec<u8>)> {
        self.brush_material.as_ref().map(|m| m.thumbnail(max))
    }

    pub fn add_layer(&mut self) {
        self.checkpoint();
        self.layers.add_layer();
        self.resync_stroke_base();
        self.refresh_display_texture();
    }

    pub fn remove_active_layer(&mut self) {
        if self.layers.layers.len() <= 1 {
            return; // remove_active is a no-op on the last layer — don't record it
        }
        self.checkpoint();
        self.layers.remove_active();
        self.resync_stroke_base();
        self.refresh_display_texture();
    }

    pub fn merge_active_layer_down(&mut self) {
        if self.layers.active == 0 {
            return; // nothing beneath the bottom layer — don't record an undo step
        }
        self.checkpoint();
        self.layers.merge_active_down();
        self.resync_stroke_base();
        self.refresh_display_texture();
    }

    pub fn move_active_layer(&mut self, up: bool) {
        self.checkpoint();
        self.layers.move_active(up);
        self.refresh_display_texture();
    }

    pub fn set_active_layer(&mut self, index: usize) {
        if index < self.layers.layers.len() {
            self.layers.active = index;
            self.resync_stroke_base();
        }
    }

    pub fn set_layer_visible(&mut self, index: usize, visible: bool) {
        if index >= self.layers.layers.len() {
            return;
        }
        self.checkpoint();
        self.layers.layers[index].visible = visible;
        self.refresh_display_texture();
    }

    pub fn set_layer_opacity(&mut self, index: usize, opacity: f32) {
        if let Some(l) = self.layers.layers.get_mut(index) {
            l.opacity = opacity;
            self.refresh_display_texture();
        }
    }

    pub fn set_layer_blend(&mut self, index: usize, blend: crate::layers::BlendMode) {
        if index >= self.layers.layers.len() {
            return;
        }
        self.checkpoint();
        self.layers.layers[index].blend = blend;
        self.refresh_display_texture();
    }

    /// Rename a layer by hand, which locks its auto-name (later ops stop rewriting
    /// it). Pixels are untouched, so no display refresh — the UI mirror picks up the
    /// new name next frame.
    pub fn rename_layer(&mut self, index: usize, name: String) {
        self.checkpoint();
        self.layers.rename(index, name);
    }

    // --- Per-layer effects (G28). All operate on the active layer, matching the
    // UI panel, which only ever shows the active layer's effect stack. ---

    /// Append a new effect (parameters at their identity values) to the active
    /// layer's stack.
    pub fn add_effect(&mut self, kind: crate::effects::EffectKind) {
        self.checkpoint();
        let active = self.layers.active;
        self.layers.layers[active]
            .effects
            .push(kind.default_effect());
        self.layers.layers[active].invalidate();
        self.layers.record_active_op(kind.token()); // auto-name: "+ Blur" etc.
        self.refresh_display_texture();
    }

    /// Remove the effect at `idx` from the active layer's stack.
    pub fn remove_effect(&mut self, idx: usize) {
        let active = self.layers.active;
        if idx >= self.layers.layers[active].effects.len() {
            return;
        }
        self.checkpoint();
        self.layers.layers[active].effects.remove(idx);
        self.layers.layers[active].invalidate();
        self.refresh_display_texture();
    }

    /// Reorder the active layer's effect at `idx` (toward the end when `up`).
    pub fn move_effect(&mut self, idx: usize, up: bool) {
        let active = self.layers.active;
        let fx = &mut self.layers.layers[active].effects;
        if up && idx + 1 < fx.len() {
            fx.swap(idx, idx + 1);
        } else if !up && idx > 0 {
            fx.swap(idx, idx - 1);
        } else {
            return;
        }
        self.layers.layers[active].invalidate();
        self.checkpoint();
        self.refresh_display_texture();
    }

    /// Replace the active layer's effect at `idx` (a parameter-slider edit). Like
    /// `set_layer_opacity` this doesn't checkpoint per call — the UI emits one
    /// checkpoint when the slider drag begins so the whole drag is a single undo.
    pub fn set_effect(&mut self, idx: usize, fx: crate::effects::Effect) {
        let active = self.layers.active;
        if let Some(slot) = self.layers.layers[active].effects.get_mut(idx) {
            *slot = fx;
            self.layers.layers[active].invalidate();
            self.refresh_display_texture();
        }
    }

    /// Orbit the camera by drag deltas and refresh the uniform.
    pub fn orbit_camera(&mut self, dx: f32, dy: f32) {
        self.camera.orbit(dx, dy);
        self.update_uniforms();
    }

    /// Pan the camera target by drag deltas and refresh the uniform.
    pub fn pan_camera(&mut self, dx: f32, dy: f32) {
        self.camera.pan(dx, dy);
        self.update_uniforms();
    }

    /// Zoom (dolly) the camera and refresh the uniform.
    pub fn zoom_camera(&mut self, delta: f32) {
        self.camera.zoom(delta);
        self.update_uniforms();
    }

    /// Orbit by explicit angles (radians); for programmatic/headless control.
    pub fn orbit_view_radians(&mut self, d_azimuth: f32, d_elevation: f32) {
        self.camera.orbit_radians(d_azimuth, d_elevation);
        self.update_uniforms();
    }

    /// The current paint texture resolution (square).
    pub fn texture_resolution(&self) -> u32 {
        self.tex_size
    }

    /// Recreate the GPU paint texture + bind group sized to the current layers.
    /// Call after the layers are replaced or resized.
    fn rebuild_paint_gpu(&mut self) {
        self.tex_size = self.layers.size();
        // The GPU texture only needs the right dimensions; a 1×1 stand-in carries
        // the size, then refresh_display_texture uploads the composited pixels.
        let placeholder = PaintTexture::new(self.tex_size, self.tex_size, [0, 0, 0, 255]);
        self.paint_texture_gpu = make_paint_texture(&self.device, &placeholder);
        self.bind_group = make_paint_bind_group(
            &self.device,
            &self.bind_group_layout,
            &self.uniform_buffer,
            &self.paint_texture_gpu,
            &self.sampler,
        );
        // The stroke buffers must track the (possibly resized) active layer.
        self.stroke_base = self.target_pixels().to_vec();
        self.stroke_coverage = vec![0.0; (self.tex_size * self.tex_size) as usize];
        self.refresh_display_texture();
    }

    /// Save the displayed texture to a PNG — quantized when quantize is on, so the
    /// exported file matches the model (WYSIWYG). Bytes are sRGB; PNG is sRGB.
    pub fn save_texture_png(&self, path: &str) -> Result<(), String> {
        let pixels = self.display_pixels();
        image::save_buffer(
            path,
            &pixels,
            self.tex_size,
            self.tex_size,
            image::ColorType::Rgba8,
        )
        .map_err(|e| format!("failed to save PNG: {e}"))
    }

    /// Export the displayed texture (G23). When `indexed` and a palette is active,
    /// writes a true indexed PNG; otherwise RGBA8. The displayed pixels are already
    /// quantized to the palette, so the index mapping is exact.
    pub fn export_png(&self, path: &str, indexed: bool) -> Result<(), String> {
        let pixels = self.display_pixels();
        let palette_u8: Vec<[u8; 3]> = self
            .palette
            .colors
            .iter()
            .map(|c| {
                [
                    (c[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (c[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (c[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                ]
            })
            .collect();
        let palette = if indexed && self.palette_settings.enabled && !palette_u8.is_empty() {
            Some(palette_u8.as_slice())
        } else {
            None
        };
        crate::export::export_png(path, &pixels, self.tex_size, self.tex_size, palette)
    }

    /// Write the current mesh — including the UVs produced by the last unwrap — to a
    /// Wavefront OBJ. Pair it with `export_png`: the texture only maps onto these UVs,
    /// so an engine needs both files.
    pub fn export_obj(&self, path: &str) -> Result<(), String> {
        crate::export::export_obj(path, &self.mesh)
    }

    /// Load a PNG into the paint buffer, resampling to the current resolution.
    pub fn load_texture_png(&mut self, path: &str) -> Result<(), String> {
        let img = image::open(path).map_err(|e| format!("failed to open image: {e}"))?;
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        let loaded = PaintTexture {
            width: w,
            height: h,
            pixels: rgba.into_raw(),
        };
        self.checkpoint(); // decode succeeded; the load is about to be applied
        *self.layers.active_tex_mut() = loaded.resampled(self.tex_size, self.tex_size);
        self.refresh_display_texture();
        // Re-snapshot the stroke base since the active layer changed wholesale.
        self.stroke_base = self.target_pixels().to_vec();
        Ok(())
    }

    /// Change the paint texture resolution, resampling every layer into it.
    pub fn set_texture_resolution(&mut self, size: u32) {
        let size = size.clamp(8, self.max_texture_dim);
        if size == self.tex_size {
            return;
        }
        self.checkpoint();
        self.layers.resize(size);
        self.mesh_maps = None; // baked at the old resolution
        self.fill_map = None; // rasterized at the old resolution
        self.rebuild_paint_gpu();
    }

    // --- Ambient-occlusion suite (mesh-aware, drives layers) ---

    /// Set the directional-light sun (the direction *toward* the light) and whether
    /// it casts shadows. Pushed from the UI each frame; the light channel rebakes
    /// lazily on the next Light effect if these changed.
    pub fn set_sun(&mut self, dir: [f32; 3], shadow: bool) {
        self.sun_dir = glam::Vec3::from(dir).normalize_or_zero();
        self.sun_shadow = shadow;
    }

    /// Bake the mesh maps for the current resolution if not already cached.
    fn ensure_mesh_maps(&mut self) {
        let stale = self
            .mesh_maps
            .as_ref()
            .is_none_or(|m| m.size != self.tex_size);
        if stale {
            log::info!("baking mesh maps at {}²…", self.tex_size);
            self.mesh_maps = Some(crate::bake::bake(&self.mesh, &self.bvh, self.tex_size));
        }
    }

    /// Ensure the cross-face brush's position-edge adjacency is built for the
    /// current geometry. Cheap and topology-only, so it survives resolution
    /// changes — rebuilt only after `surface_adj` is invalidated on a mesh change.
    fn ensure_surface_adj(&mut self) {
        if self.surface_adj.is_none() {
            self.surface_adj = Some(crate::surface::Adjacency::build(&self.mesh));
        }
    }

    /// Ensure the baked `light` channel reflects the current sun. Cheap to call
    /// before any Light-sourced effect: it bakes the maps if needed, then recomputes
    /// the directional light only when the sun direction or shadow flag changed.
    fn ensure_light(&mut self) {
        self.ensure_mesh_maps();
        let (dir, shadow) = (self.sun_dir, self.sun_shadow);
        let need = self
            .mesh_maps
            .as_ref()
            .is_none_or(|m| m.light_params != Some((dir, shadow)));
        if need {
            // Disjoint field borrows: &self.bvh while &mut self.mesh_maps.
            let bvh = &self.bvh;
            if let Some(m) = self.mesh_maps.as_mut() {
                m.compute_light(bvh, dir, shadow);
            }
        }
    }

    /// Add a generated tint layer: a flat `color` whose alpha is a baked map
    /// (`src`) read through `levels`, optionally broken up by procedural `noise`.
    /// This is the one path behind every AO/curvature effect — the presets below are
    /// just fixed (source, color, blend) choices.
    pub fn add_map_layer(
        &mut self,
        name: &str,
        src: MapSource,
        levels: Levels,
        color: [u8; 3],
        blend: crate::layers::BlendMode,
        noise: Option<NoiseMod>,
    ) {
        self.checkpoint();
        self.ensure_mesh_maps();
        if src == MapSource::Light {
            self.ensure_light();
        }
        let weights = self
            .mesh_maps
            .as_ref()
            .unwrap()
            .sample(src, &levels, noise.as_ref());
        let mut tex = PaintTexture::new(
            self.tex_size,
            self.tex_size,
            [color[0], color[1], color[2], 0],
        );
        for (i, w) in weights.iter().enumerate() {
            tex.pixels[i * 4 + 3] = (w * 255.0).round() as u8;
        }
        self.layers
            .push_generated(name.to_string(), tex, blend, 1.0);
        // Record the recipe so this look can be saved as a reusable preset (G21).
        self.recipe.push(crate::preset::PresetLayer::from_op(
            name, src, levels, color, blend, noise,
        ));
        self.resync_stroke_base();
        self.refresh_display_texture();
    }

    /// Add a gradient-map layer: instead of one flat color, every covered texel is
    /// colored by looking the baked map's value (read through `levels` + optional
    /// `noise`) up in a low→high color ramp. One map then reads as a full material —
    /// e.g. dark crevices → bright tops from Cavities, or a lit/shaded gradient from
    /// the sun. Alpha is the coverage mask, so the ramp paints the whole surface.
    ///
    /// Not recorded in the preset recipe yet: the recipe stores a single flat color
    /// per layer, not a ramp, so a gradient look must currently be re-applied by hand.
    pub fn add_gradient_layer(
        &mut self,
        name: &str,
        src: MapSource,
        levels: Levels,
        grad: Gradient,
        blend: crate::layers::BlendMode,
        noise: Option<NoiseMod>,
    ) {
        self.checkpoint();
        self.ensure_mesh_maps();
        if src == MapSource::Light {
            self.ensure_light();
        }
        let maps = self.mesh_maps.as_ref().unwrap();
        let weights = maps.sample(src, &levels, noise.as_ref());
        let mut tex = PaintTexture::new(self.tex_size, self.tex_size, [0, 0, 0, 0]);
        for (i, &w) in weights.iter().enumerate() {
            if maps.mask[i] {
                let c = grad.sample(w);
                tex.pixels[i * 4] = c[0];
                tex.pixels[i * 4 + 1] = c[1];
                tex.pixels[i * 4 + 2] = c[2];
                tex.pixels[i * 4 + 3] = 255;
            }
        }
        self.layers
            .push_generated(name.to_string(), tex, blend, 1.0);
        self.resync_stroke_base();
        self.refresh_display_texture();
    }

    /// Fill the *active layer's* reveal mask from a baked map — the Substance-style
    /// move: route AO/curvature into a mask so whatever that layer paints only shows
    /// where the map is high (e.g. paint a flat color, then confine it to cavities).
    pub fn fill_active_mask_from_map(
        &mut self,
        src: MapSource,
        levels: Levels,
        noise: Option<NoiseMod>,
    ) {
        self.checkpoint();
        self.ensure_mesh_maps();
        if src == MapSource::Light {
            self.ensure_light();
        }
        let weights = self
            .mesh_maps
            .as_ref()
            .unwrap()
            .sample(src, &levels, noise.as_ref());
        let mask = self.layers.active_mask_mut();
        for (i, w) in weights.iter().enumerate() {
            let v = (w * 255.0).round() as u8;
            mask.pixels[i * 4] = v; // compositor reads the red channel
            mask.pixels[i * 4 + 1] = v;
            mask.pixels[i * 4 + 2] = v;
            mask.pixels[i * 4 + 3] = 255;
        }
        self.resync_stroke_base();
        self.refresh_display_texture();
    }

    /// Preset: a black Multiply layer in the cavities — shadow sinks into crevices.
    pub fn apply_ao_layer(&mut self, levels: Levels, noise: Option<NoiseMod>) {
        self.add_map_layer(
            "AO",
            MapSource::Cavities,
            levels,
            [0, 0, 0],
            crate::layers::BlendMode::Multiply,
            noise,
        );
    }

    /// Preset: a white layer on convex edges/corners — brightens exposed edges and,
    /// being curvature-driven, never lands on flat lit faces or in concave creases.
    pub fn apply_highlight_layer(&mut self, levels: Levels, noise: Option<NoiseMod>) {
        self.add_map_layer(
            "Highlights",
            MapSource::Edges,
            levels,
            [255, 255, 255],
            crate::layers::BlendMode::Normal,
            noise,
        );
    }

    /// Preset: a dark grime tint settling into the cavities (Substance "Dirt").
    pub fn apply_dirt_layer(&mut self, levels: Levels, noise: Option<NoiseMod>) {
        self.add_map_layer(
            "Dirt",
            MapSource::Cavities,
            levels,
            [54, 42, 30],
            crate::layers::BlendMode::Normal,
            noise,
        );
    }

    /// Preset: a worn, lightened tint on convex edges (Substance "Edge wear").
    pub fn apply_edge_wear_layer(&mut self, levels: Levels, noise: Option<NoiseMod>) {
        self.add_map_layer(
            "Edge wear",
            MapSource::Edges,
            levels,
            [205, 200, 185],
            crate::layers::BlendMode::Normal,
            noise,
        );
    }

    // --- Preset looks (G21) ---

    /// Apply a preset's recipe to the *current* mesh: every generator layer is
    /// re-evaluated against this mesh's freshly-baked maps, so the look follows the
    /// new geometry. Layers are appended on top of the current stack.
    pub fn apply_preset(&mut self, preset: &crate::preset::Preset) {
        for pl in &preset.layers {
            let (name, src, levels, color, blend, noise) = pl.to_op();
            self.add_map_layer(&name, src, levels, color, blend, noise);
        }
        if let Some(pname) = &preset.palette {
            if let Some(p) = Palette::builtins()
                .into_iter()
                .find(|p| p.name.eq_ignore_ascii_case(pname))
            {
                self.set_palette(p);
            }
        }
    }

    /// Apply a built-in preset by name (e.g. "Mossy Stone").
    pub fn apply_builtin_preset(&mut self, name: &str) -> Result<(), String> {
        let preset =
            crate::preset::builtin(name).ok_or_else(|| format!("no built-in preset '{name}'"))?;
        self.apply_preset(&preset);
        Ok(())
    }

    /// Save the generator layers applied so far as a reusable, shareable preset.
    /// Carries the active palette (by name) so the look travels with its colors.
    pub fn save_preset(&self, path: &str, name: &str) -> Result<(), String> {
        if self.recipe.is_empty() {
            return Err("no generator layers to save — apply AO/Dirt/Edge-wear first".into());
        }
        let mut preset = crate::preset::Preset::new(name, self.recipe.clone());
        if self.palette_settings.enabled {
            preset.palette = Some(self.palette.name.clone());
        }
        preset.save(path)
    }

    /// Load a preset from disk and apply it to the current mesh.
    pub fn load_and_apply_preset(&mut self, path: &str) -> Result<(), String> {
        let preset = crate::preset::Preset::load(path)?;
        self.apply_preset(&preset);
        Ok(())
    }

    // --- Fill tools (paint bucket; FillMap-driven) ---

    /// Build the fill map (UV islands + coplanar facets) for the current resolution
    /// if not already cached. Cheap (union-find + a UV-space scan-fill, no
    /// ray-casting), so unlike the AO bake it's fine to do lazily on the first click.
    fn ensure_fill_map(&mut self) {
        let stale = self
            .fill_map
            .as_ref()
            .is_none_or(|m| m.size != self.tex_size);
        if stale {
            self.fill_map = Some(crate::fill::FillMap::build(&self.mesh, self.tex_size));
        }
    }

    /// Object fill: a click on the mesh floods the active paint target with the
    /// brush color across every texel the mesh's UVs cover. Returns whether it
    /// landed on the mesh (a click on the background fills nothing). One undo step.
    pub fn fill_object_at(&mut self, mouse_px: (f32, f32), brush: &Brush) -> bool {
        let ray = self.pick_ray(Vec2::new(mouse_px.0, mouse_px.1));
        if self.bvh.pick(&ray).is_none() {
            return false;
        }
        self.ensure_fill_map();
        let map = self.fill_map.as_ref().unwrap();
        let covered: Vec<bool> = map.texel_island.iter().map(|&i| i >= 0).collect();
        self.checkpoint();
        let keep = covered.clone();
        self.fill_texels(brush, &keep, &covered);
        self.resync_stroke_base();
        self.refresh_display_texture();
        true
    }

    /// UV-island fill: a click floods the one UV island under the cursor with the
    /// brush color (the island the picked triangle belongs to), bounded by the
    /// island edge regardless of existing color. Returns whether it hit the mesh.
    /// One undo step.
    pub fn fill_island_at(&mut self, mouse_px: (f32, f32), brush: &Brush) -> bool {
        let ray = self.pick_ray(Vec2::new(mouse_px.0, mouse_px.1));
        let Some(hit) = self.bvh.pick(&ray) else {
            return false;
        };
        self.ensure_fill_map();
        let map = self.fill_map.as_ref().unwrap();
        let Some(island) = map.island_for_tri(hit.tri) else {
            return false;
        };
        let island = island as i32;
        let covered: Vec<bool> = map.texel_island.iter().map(|&i| i >= 0).collect();
        let keep: Vec<bool> = map.texel_island.iter().map(|&i| i == island).collect();
        self.checkpoint();
        self.fill_texels(brush, &keep, &covered);
        self.resync_stroke_base();
        self.refresh_display_texture();
        true
    }

    /// Face fill: a click floods the one flat facet of the model under the cursor
    /// — the connected, near-coplanar triangles the picked triangle belongs to
    /// (a cube side; a quad; a triangulated flat wall). Unlike island fill this is
    /// geometric, so it stops at hard edges regardless of UV layout, and a facet
    /// split across UV islands still fills as one. Returns whether it hit the mesh.
    /// One undo step.
    pub fn fill_face_at(&mut self, mouse_px: (f32, f32), brush: &Brush) -> bool {
        let ray = self.pick_ray(Vec2::new(mouse_px.0, mouse_px.1));
        let Some(hit) = self.bvh.pick(&ray) else {
            return false;
        };
        self.ensure_fill_map();
        let map = self.fill_map.as_ref().unwrap();
        let Some(facet) = map.facet_for_tri(hit.tri) else {
            return false;
        };
        let facet = facet as i32;
        let covered: Vec<bool> = map.texel_island.iter().map(|&i| i >= 0).collect();
        let keep: Vec<bool> = map.texel_facet.iter().map(|&i| i == facet).collect();
        self.checkpoint();
        self.fill_texels(brush, &keep, &covered);
        self.resync_stroke_base();
        self.refresh_display_texture();
        true
    }

    /// Write the fill color into every `keep` texel of the active paint target,
    /// then dilate the filled region one texel into uncovered neighbours so an
    /// island edge reads solid under nearest sampling (principle #5). Dilation
    /// only expands into `!covered` texels, so it never bleeds across a seam into
    /// a neighbouring island. `keep` and `covered` are `size`×`size`, row-major.
    fn fill_texels(&mut self, brush: &Brush, keep: &[bool], covered: &[bool]) {
        // Mask painting ignores the brush color (reveal=white, hide=black), to
        // match `stamp_screen`; color painting writes opaque brush color.
        let rgba = match self.paint_target {
            PaintTarget::Color => {
                let c = brush.color_u8();
                [c[0], c[1], c[2], 255]
            }
            PaintTarget::Mask => {
                let v = if self.mask_reveal { 255 } else { 0 };
                [v, v, v, 255]
            }
        };

        // Frontier texels: uncovered, but 4-adjacent to a filled texel. Computed
        // from `keep`/`covered` (not the texture) so it doesn't fight the borrow.
        let w = self.tex_size as i32;
        let h = self.tex_size as i32;
        let mut frontier: Vec<usize> = Vec::new();
        for (t, &cov) in covered.iter().enumerate() {
            if cov {
                continue;
            }
            let (x, y) = (t as i32 % w, t as i32 / w);
            let adjacent = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                .into_iter()
                .any(|(nx, ny)| {
                    nx >= 0 && ny >= 0 && nx < w && ny < h && keep[(ny * w + nx) as usize]
                });
            if adjacent {
                frontier.push(t);
            }
        }

        let tex = match self.paint_target {
            PaintTarget::Color => self.layers.active_tex_mut(),
            PaintTarget::Mask => self.layers.active_mask_mut(),
        };
        for (t, &k) in keep.iter().enumerate() {
            if k {
                tex.pixels[t * 4..t * 4 + 4].copy_from_slice(&rgba);
            }
        }
        for t in frontier {
            tex.pixels[t * 4..t * 4 + 4].copy_from_slice(&rgba);
        }
        // Auto-name: a color fill is content; a mask fill only shapes where.
        if self.paint_target == PaintTarget::Color {
            self.layers.record_active_op("Fill");
        }
    }

    /// Load a new mesh from a file, rebuilding the GPU buffers and the pick BVH.
    /// The painted texture is preserved (it re-maps onto the new model's UVs).
    pub fn load_model(&mut self, path: &str) -> Result<(), String> {
        let mesh = crate::model::load(path)?;
        self.vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vertex buffer"),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        self.index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("index buffer"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        self.index_count = mesh.indices.len() as u32;
        self.bvh = Bvh::build(&mesh);
        self.mesh = mesh;
        self.surface_adj = None; // topology changed; adjacency is stale
        self.topo_version = self.topo_version.wrapping_add(1); // UV editor rebuilds the wireframe
        self.mesh_maps = None; // geometry changed; baked maps are stale
        self.fill_map = None; // geometry changed; island map is stale
        *self.coverage.get_mut() = None; // UV coverage is stale
        Ok(())
    }

    /// Re-unwrap the current mesh's UVs into connectivity-based charts at a constant
    /// world-space texel density (the atlas size is *derived* from `density`).
    /// Geometry is unchanged but vertices are split and re-UV'd, so the GPU buffers,
    /// BVH, and cached maps are rebuilt; the paint texture is resampled to the new
    /// atlas size and re-maps onto the new UVs. Returns `(atlas_size, clamped)` where
    /// `clamped` means density was reduced to stay within the GPU texture limit.
    pub fn apply_unwrap(&mut self, density: crate::unwrap::Density) -> (u32, bool) {
        self.checkpoint();
        let opts = crate::unwrap::UnwrapOptions {
            density,
            max_atlas: self.max_texture_dim,
            ..Default::default()
        };
        let result = crate::unwrap::auto_unwrap(&self.mesh, &opts);
        log::debug!(
            "unwrap: {0} tris → {1}×{1} atlas at {2:.2} texels/unit{3}",
            self.mesh.indices.len() / 3,
            result.atlas_size,
            result.density_d,
            if result.clamped { " (clamped)" } else { "" },
        );

        // Resize layers first so the painted pixels resample into the new atlas size;
        // inline rather than via set_texture_resolution (that takes its own checkpoint
        // and early-returns when the size is unchanged).
        if result.atlas_size != self.tex_size {
            self.layers.resize(result.atlas_size);
        }

        let mesh = result.mesh;
        self.vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vertex buffer"),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        self.index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("index buffer"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        self.index_count = mesh.indices.len() as u32;
        self.bvh = Bvh::build(&mesh);
        self.mesh = mesh;
        self.surface_adj = None; // topology changed; adjacency is stale
        self.topo_version = self.topo_version.wrapping_add(1); // UV editor rebuilds the wireframe
                                                               // Pick up the new size and recreate the GPU texture / bind group / stroke
                                                               // buffers (the old fixed-resolution unwrap never needed this).
        self.rebuild_paint_gpu();
        self.mesh_maps = None;
        self.fill_map = None;
        *self.coverage.get_mut() = None;
        (result.atlas_size, result.clamped)
    }

    /// Save the entire editing state to a `.lowtex` file (G24).
    pub fn save_project(&mut self, path: &str) -> Result<(), String> {
        self.write_project(path)?;
        // The on-disk file now matches the live document: clean for both the user
        // (title dot / close prompt) and the autosave timer.
        self.saved_seq = self.edit_seq;
        self.autosave_seq = self.edit_seq;
        Ok(())
    }

    /// Write a timed recovery version (G31). Same bytes as an explicit save, but it
    /// only clears the *autosave* watermark — the document stays "dirty" for the
    /// user until they save to their own file, so the title dot and close prompt
    /// still nudge them to do so.
    pub fn autosave(&mut self, path: &str) -> Result<(), String> {
        self.write_project(path)?;
        self.autosave_seq = self.edit_seq;
        Ok(())
    }

    /// Serialize the whole editing state to a `.lowtex` file. The dirty-tracking
    /// watermarks are left untouched here; `save_project`/`autosave` own them.
    fn write_project(&self, path: &str) -> Result<(), String> {
        let blend_index = |b: crate::layers::BlendMode| {
            crate::layers::BlendMode::ALL
                .iter()
                .position(|&m| m == b)
                .unwrap_or(0) as u8
        };
        let layers = self
            .layers
            .layers
            .iter()
            .map(|l| crate::project::LayerDoc {
                name: l.name.clone(),
                blend: blend_index(l.blend),
                visible: l.visible,
                opacity: l.opacity,
                color: crate::project::encode_pixels(&l.tex.pixels),
                mask: crate::project::encode_pixels(&l.mask.pixels),
                effects: l.effects.iter().map(effect_to_doc).collect(),
            })
            .collect();
        let doc = crate::project::ProjectDoc {
            version: crate::project::FORMAT_VERSION,
            tex_size: self.tex_size,
            active_layer: self.layers.active,
            palette: self.palette.colors.clone(),
            quantize: self.palette_settings.enabled,
            dither: self.palette_settings.dither,
            dither_strength: self.palette_settings.dither_strength,
            positions: self.mesh.vertices.iter().map(|v| v.position).collect(),
            normals: self.mesh.vertices.iter().map(|v| v.normal).collect(),
            uvs: self.mesh.vertices.iter().map(|v| v.uv).collect(),
            indices: self.mesh.indices.clone(),
            layers,
            texture_folder: self.texture_folder.clone(),
        };
        doc.save(path)
    }

    /// The texture folder this project records (brush browser source), if any.
    pub fn texture_folder(&self) -> Option<&str> {
        self.texture_folder.as_deref()
    }

    /// Point this project at a texture folder (or clear it with `None`). Called by
    /// the app when the user opens a folder so a later save records it.
    pub fn set_texture_folder(&mut self, folder: Option<String>) {
        self.texture_folder = folder;
    }

    /// Load editing state from a `.lowtex` file, replacing the current project.
    pub fn load_project(&mut self, path: &str) -> Result<(), String> {
        let doc = crate::project::ProjectDoc::load(path)?;

        // Rebuild the mesh + GPU buffers + BVH.
        let vertices: Vec<Vertex> = (0..doc.positions.len())
            .map(|i| Vertex {
                position: doc.positions[i],
                normal: *doc.normals.get(i).unwrap_or(&[0.0, 1.0, 0.0]),
                uv: *doc.uvs.get(i).unwrap_or(&[0.0, 0.0]),
            })
            .collect();
        let mesh = Mesh {
            vertices,
            indices: doc.indices.clone(),
            needs_normals: false,
            needs_uvs: false,
        };
        self.vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vertex buffer"),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        self.index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("index buffer"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        self.index_count = mesh.indices.len() as u32;
        self.bvh = Bvh::build(&mesh);
        self.mesh = mesh;
        self.surface_adj = None; // topology changed; adjacency is stale
        self.topo_version = self.topo_version.wrapping_add(1); // UV editor rebuilds the wireframe
        self.mesh_maps = None;
        self.fill_map = None;
        *self.coverage.get_mut() = None;

        // Rebuild the layer stack.
        let n = (doc.tex_size * doc.tex_size * 4) as usize;
        let mut layers = Vec::with_capacity(doc.layers.len());
        for d in &doc.layers {
            let mut color = crate::project::decode_pixels(&d.color)?;
            let mut mask = crate::project::decode_pixels(&d.mask)?;
            color.resize(n, 0);
            mask.resize(n, 255);
            layers.push(crate::layers::Layer::from_parts(
                d.name.clone(),
                PaintTexture {
                    width: doc.tex_size,
                    height: doc.tex_size,
                    pixels: color,
                },
                PaintTexture {
                    width: doc.tex_size,
                    height: doc.tex_size,
                    pixels: mask,
                },
                d.visible,
                d.opacity,
                crate::layers::BlendMode::ALL
                    .get(d.blend as usize)
                    .copied()
                    .unwrap_or(crate::layers::BlendMode::Normal),
                d.effects.iter().map(doc_to_effect).collect(),
            ));
        }
        if layers.is_empty() {
            return Err("project has no layers".into());
        }
        let active = doc.active_layer.min(layers.len() - 1);
        self.layers = crate::layers::Layers { layers, active };
        self.tex_size = doc.tex_size;

        self.palette = Palette {
            name: "Loaded".to_string(),
            colors: doc.palette,
        };
        self.palette_settings = PaletteSettings {
            enabled: doc.quantize,
            dither: doc.dither,
            dither_strength: doc.dither_strength,
        };

        self.texture_folder = doc.texture_folder;

        // Recreate the GPU paint texture for the (possibly new) resolution and
        // re-snapshot stroke buffers + refresh the display.
        self.rebuild_paint_gpu();
        // The freshly-loaded document matches its file: start clean.
        self.saved_seq = self.edit_seq;
        self.autosave_seq = self.edit_seq;
        Ok(())
    }

    /// Fill the active layer's color with a loaded material image (brick, moss, …),
    /// UV-tiled `tile` times. Combine with a reveal mask (e.g. mask-from-Cavities)
    /// for "moss in the crevices". Undoable.
    pub fn fill_active_with_material(&mut self, path: &str, tile: f32) -> Result<(), String> {
        let material = crate::material::Material::load(path)?;
        self.checkpoint();
        material.fill(self.layers.active_tex_mut(), tile);
        self.layers.record_active_op("Material"); // auto-name
        self.resync_stroke_base();
        self.refresh_display_texture();
        Ok(())
    }

    /// Begin a new stroke: snapshot the texture and clear stroke coverage, so
    /// overlap within the stroke accumulates by max-coverage (no double-darken).
    pub fn begin_stroke(&mut self) {
        self.stroke_base = self.target_pixels().to_vec();
        self.stroke_coverage.fill(0.0);
        // A fresh stroke hasn't fixed its locked face yet (the first dab will).
        self.stroke_lock_facets = None;
        // Snapshot the pre-stroke stack so the whole stroke is one undo step. Held
        // until release, then committed only if the stroke actually painted.
        self.pending = Some(self.layers.clone());
        self.stroke_dirty = false;
        // The hover ghost must be reverted now the stroke is taking over; force the
        // preview path to run (and clear it) next frame rather than skip on a match.
        self.last_preview = None;
    }

    /// End the current stroke, committing it to history as a single undo step.
    /// A stroke that never hit the mesh (`stroke_dirty` false) records nothing,
    /// so an errant click off the model doesn't leave an empty undo entry.
    pub fn end_stroke(&mut self) {
        if let Some(before) = self.pending.take() {
            if self.stroke_dirty {
                self.history.record(before);
                self.mark_edited();
                // Auto-name the layer from the stroke (color only — mask painting
                // shapes *where* a layer shows, not its content, so it adds no token).
                if self.paint_target == PaintTarget::Color {
                    let token = if self.brush_material.is_some() {
                        "Stamp"
                    } else {
                        "Stroke"
                    };
                    self.layers.record_active_op(token);
                }
            }
        }
        self.stroke_dirty = false;
        // The stroke is committed; re-show the hover ghost over the new pixels.
        self.last_preview = None;
    }

    /// Build a stable world-space pick ray for a screen pixel. The pixel is in
    /// full-window coordinates, so map it into the scene viewport (subtract the
    /// panel offset and unproject against the viewport's size) to match what's drawn.
    fn pick_ray(&self, mouse_px: Vec2) -> Ray {
        let inv_view_proj = self.camera.view_proj().inverse();
        let ray_origin = self.camera.eye();
        let (vx, vy, vw, vh) = self.scene_viewport();
        let local = Vec2::new(mouse_px.x - vx, mouse_px.y - vy);
        let ray = Ray::from_screen(local, Vec2::new(vw, vh), inv_view_proj);
        // Pin the origin to the camera eye for stability — unproject can wobble
        // for near==0 depending on driver.
        Ray {
            origin: ray_origin,
            direction: (ray.origin + ray.direction - ray_origin).normalize(),
        }
    }

    /// Reflect a pick ray across the model-symmetry plane for `axis` (perpendicular
    /// to the axis, through the mesh bbox center). The origin reflects about the
    /// plane; the direction just flips its component on that axis (a vector ignores
    /// the plane offset), so the reflected ray points at the mirror-image surface.
    fn mirror_ray(&self, ray: &Ray, axis: SymmetryAxis) -> Ray {
        let (mn, mx) = self.mesh.bounds();
        let center = (mn + mx) * 0.5;
        let i = axis.index();
        let mut origin = ray.origin;
        origin[i] = 2.0 * center[i] - origin[i];
        let mut direction = ray.direction;
        direction[i] = -direction[i];
        Ray { origin, direction }
    }

    /// Pick + stamp a single dab at a screen pixel (no GPU upload). Returns
    /// whether it hit the mesh.
    ///
    /// The dab is a sphere on the mesh *surface*, not a circle in texture space:
    /// from the picked triangle `surface::splat` walks position-adjacent triangles
    /// out to a world-space radius and weights each covered texel by its 3D
    /// distance to the hit. A stroke that reaches a UV seam therefore wraps onto
    /// the neighbouring face instead of dying at the island edge — the cross-face
    /// painting the flat texture-space stamp couldn't do.
    fn stamp_screen(&mut self, mouse_px: Vec2, brush: &Brush) -> bool {
        let ray = self.pick_ray(mouse_px);
        let Some(hit) = self.bvh.pick(&ray) else {
            return false;
        };
        self.ensure_surface_adj();
        // Mirror painting: reflect the pick ray across the model-symmetry plane and
        // re-pick, so the mirrored dab lands on the symmetric surface point with its
        // own correct triangle/normal. Re-picking (vs. reflecting the hit) keeps the
        // splat valid even where the mirror point sits on a different face.
        let mirror = self
            .symmetry
            .and_then(|axis| self.bvh.pick(&self.mirror_ray(&ray, axis)));
        // The first dab of the stroke fixes the locked face(s) — the hit's facet,
        // plus the mirror's — for the rest of the stroke (face lock only).
        let hits: Vec<&Hit> = std::iter::once(&hit).chain(mirror.as_ref()).collect();
        self.ensure_stroke_lock(&hits);
        let mut painted = self.surface_dab(&hit, brush);
        if let Some(mirror) = mirror {
            painted |= self.surface_dab(&mirror, brush);
        }
        painted
    }

    /// Fix the stroke's locked face(s) from its first dab's `hits` (the picked
    /// triangle, plus the mirror's when symmetry is on). No-op once the lock is set
    /// or when face-lock is off, so later dabs in the stroke can't extend it — the
    /// whole stroke stays on the face painting began on. Builds the fill map on
    /// demand, like the Face fill bucket.
    fn ensure_stroke_lock(&mut self, hits: &[&Hit]) {
        if !self.lock_face || self.stroke_lock_facets.is_some() {
            return;
        }
        self.ensure_fill_map();
        let map = self.fill_map.as_ref().unwrap();
        let facets = hits
            .iter()
            .filter_map(|h| map.facet_for_tri(h.tri).map(|f| f as i32))
            .collect();
        self.stroke_lock_facets = Some(facets);
    }

    /// Drop every `(texel, _)` entry not on one of the stroke's locked facets, so a
    /// face-locked dab stops at the face's edges instead of wrapping onto its
    /// neighbours. No-op when the lock is off or unset.
    fn retain_locked<T>(&self, items: &mut Vec<(usize, T)>) {
        if !self.lock_face {
            return;
        }
        let Some(facets) = self.stroke_lock_facets.as_deref() else {
            return;
        };
        if facets.is_empty() {
            return; // the first dab resolved no facet — don't cull (paint freely)
        }
        let texel_facet = &self.fill_map.as_ref().unwrap().texel_facet;
        items.retain(|&(texel, _)| {
            facets
                .iter()
                .any(|&f| texel_facet.get(texel).copied() == Some(f))
        });
    }

    /// Splat one surface dab from an already-resolved `hit` into the active paint
    /// target, accumulating the touched-texel bounding box into `dirty_rect`.
    /// Returns whether it covered any texels. `ensure_surface_adj` must have run.
    /// Return a copy of `hit` whose center is snapped to the texel grid: the UV is
    /// rounded to the covered texel's center, and the world position is re-derived from
    /// that snapped UV via the hit triangle's barycentric coords (the flood and falloff
    /// run off `pos`, so the world point must move with the UV). Falls back to the
    /// original hit if the triangle's UVs are degenerate (zero area).
    fn snap_hit(&self, hit: &Hit, grid: f32) -> Hit {
        let suv = snap_uv(hit.uv, self.tex_size, grid);
        let (p, uv) = crate::surface::tri_data(&self.mesh, hit.tri);
        match uv_bary(suv, uv[0], uv[1], uv[2]) {
            Some(w) => Hit {
                uv: suv,
                tri: hit.tri,
                pos: p[0] * w.x + p[1] * w.y + p[2] * w.z,
                normal: hit.normal,
            },
            None => *hit,
        }
    }

    fn surface_dab(&mut self, hit: &Hit, brush: &Brush) -> bool {
        // Snap to grid: move the dab's center onto the texel it covers (in UV space)
        // before flooding, so 3D strokes quantize to the texture grid the same way the
        // 2D editor does. The flood radius is unchanged, so this never opens gaps.
        let snapped;
        let hit = if brush.snap_to_texel {
            snapped = self.snap_hit(hit, brush.snap_grid);
            &snapped
        } else {
            hit
        };
        // Stamp mode: project the image's alpha through the oriented tangent frame as a
        // decal. In Brush (tiled) mode this is skipped, so the normal splat + tiled
        // `deposit` runs instead — a consistent, UV-anchored material field. Color target
        // only (a decal into a mask isn't meaningful).
        // The eraser removes coverage, so it always takes the plain splat path below —
        // a per-texel decal color is meaningless when we're lowering alpha, not adding it.
        if self.paint_target == PaintTarget::Color && self.stamp_active() && !brush.erase {
            let mut splats = self.decal_splats(hit, brush);
            self.retain_locked(&mut splats);
            if splats.is_empty() {
                return false;
            }
            self.deposit_rgb(&splats);
            return true;
        }
        let size = self.tex_size;
        let radius_world = self.world_brush_radius(hit.tri, brush.radius);
        // Distinct-field borrows (mesh/surface_adj immutable, splat_scratch mutable)
        // in one call — the reused flood scratch keeps the stroke allocation-free.
        let adj = self.surface_adj.as_ref().unwrap();
        let mut splats = crate::surface::splat(
            &self.mesh,
            adj,
            hit,
            radius_world,
            brush.opacity,
            brush.hardness,
            size,
            &mut self.splat_scratch,
        );
        // Face lock: drop every texel that isn't on the stroke's locked face, so
        // the dab stops at the face's edges instead of wrapping onto its neighbours.
        self.retain_locked(&mut splats);
        if splats.is_empty() {
            return false; // hit the mesh, but the dab covered no texels
        }
        self.deposit(&splats, brush);
        true
    }

    /// Composite a set of `(texel, coverage)` splats into the active paint target
    /// through the per-stroke coverage discipline (max coverage per texel, so
    /// overlapping stamps within one stroke don't double-darken), growing
    /// `dirty_rect` by the touched texels' bounding box. Shared by the cross-face
    /// surface brush (`surface_dab`) and the flat 2D UV brush (`stamp_uv_disc`) so
    /// both composite identically — solid color, brush image, and mask alike.
    fn deposit(&mut self, splats: &[(usize, f32)], brush: &Brush) {
        if splats.is_empty() {
            return;
        }
        let size = self.tex_size;
        let base = &self.stroke_base;
        let coverage = &mut self.stroke_coverage;
        // Touched texels can be scattered across the atlas (one cluster per face a
        // surface dab wrapped onto), so accumulate their bounding box rather than
        // assume a fixed footprint around the cursor.
        let (mut x0, mut y0, mut x1, mut y1) = (size, size, 0u32, 0u32);
        // Eraser: lower alpha toward transparent instead of laying color (color target
        // only — a mask's reveal/hide already covers the "remove" case). Max-coverage
        // discipline still holds: more coverage = more erased, monotonic either way.
        let erase = brush.erase && matches!(self.paint_target, PaintTarget::Color);
        let mut mark =
            |texel: usize, coverage: &mut [f32], a: f32, src: [u8; 3], pixels: &mut [u8]| {
                if a <= coverage[texel] {
                    return; // already at least this covered this stroke
                }
                coverage[texel] = a;
                if erase {
                    crate::paint::erase_texel(pixels, base, texel, a);
                } else {
                    crate::paint::blend_texel(pixels, base, texel, src, a);
                }
                let (tx, ty) = (texel as u32 % size, texel as u32 / size);
                x0 = x0.min(tx);
                y0 = y0.min(ty);
                x1 = x1.max(tx + 1);
                y1 = y1.max(ty + 1);
            };
        match self.paint_target {
            PaintTarget::Color => {
                // A brush image reveals itself (UV-tiled) per texel; otherwise the
                // brush's solid color.
                let mat = self.brush_material.as_ref();
                let tile = self.brush_tile;
                let solid = brush.color_u8();
                let tex = self.layers.active_tex_mut();
                for &(texel, a) in splats {
                    let (tx, ty) = (texel as u32 % size, texel as u32 / size);
                    let src = match mat {
                        Some(m) => {
                            let c =
                                m.sample(tx as f32 / size as f32, ty as f32 / size as f32, tile);
                            [c[0], c[1], c[2]]
                        }
                        None => solid,
                    };
                    mark(texel, coverage, a, src, &mut tex.pixels);
                }
            }
            PaintTarget::Mask => {
                // Mask painting ignores the brush color: reveal=white, hide=black.
                let src = if self.mask_reveal { [255; 3] } else { [0; 3] };
                let tex = self.layers.active_mask_mut();
                for &(texel, a) in splats {
                    mark(texel, coverage, a, src, &mut tex.pixels);
                }
            }
        }

        if x1 > x0 && y1 > y0 {
            let r = TexRect { x0, y0, x1, y1 };
            self.dirty_rect = Some(match self.dirty_rect {
                Some(prev) => prev.union(r),
                None => r,
            });
        }
        self.stroke_dirty = true;
    }

    /// Composite `(texel, (coverage, rgb))` splats into the active *color* layer with a
    /// per-texel color — the decal path, where each texel's color comes from the
    /// sampled stamp rather than one brush color for the whole dab. Shares `deposit`'s
    /// per-stroke coverage discipline (max coverage per texel) and dirty-rect growth.
    fn deposit_rgb(&mut self, splats: &[(usize, (f32, [u8; 3]))]) {
        if splats.is_empty() {
            return;
        }
        let size = self.tex_size;
        let base = &self.stroke_base;
        let coverage = &mut self.stroke_coverage;
        let tex = self.layers.active_tex_mut();
        let (mut x0, mut y0, mut x1, mut y1) = (size, size, 0u32, 0u32);
        for &(texel, (a, rgb)) in splats {
            if a <= coverage[texel] {
                continue; // already at least this covered this stroke
            }
            coverage[texel] = a;
            crate::paint::blend_texel(&mut tex.pixels, base, texel, rgb, a);
            let (tx, ty) = (texel as u32 % size, texel as u32 / size);
            x0 = x0.min(tx);
            y0 = y0.min(ty);
            x1 = x1.max(tx + 1);
            y1 = y1.max(ty + 1);
        }
        if x1 > x0 && y1 > y0 {
            let r = TexRect { x0, y0, x1, y1 };
            self.dirty_rect = Some(match self.dirty_rect {
                Some(prev) => prev.union(r),
                None => r,
            });
        }
        self.stroke_dirty = true;
    }

    /// Stamp a single dab at a screen pixel. Used for the initial click of a stroke and
    /// for headless verification. Only mutates the layer + marks the dirty rect; the
    /// GPU upload is coalesced into the next `flush_paint` (per frame, or before capture).
    pub fn paint_at(&mut self, mouse_px: (f32, f32), brush: &Brush) {
        let p = Vec2::new(mouse_px.0, mouse_px.1);
        self.stamp_screen(p, brush);
    }

    /// Paint a continuous stroke segment from `from` to `to` (screen pixels),
    /// interpolating stamps so a fast drag leaves a solid line, not gappy dots.
    /// Re-picks at each sub-sample so the stroke follows the surface across faces.
    /// Accumulates the dirty rect; the upload happens in the next `flush_paint`.
    pub fn paint_segment(&mut self, from: (f32, f32), to: (f32, f32), brush: &Brush) {
        let from = Vec2::new(from.0, from.1);
        let to = Vec2::new(to.0, to.1);
        // Space dabs at ~half the brush's on-screen radius so consecutive dabs
        // overlap solidly without re-splatting the whole footprint every 2px — the
        // dominant cost at large brush sizes, since each splat floods the mesh and
        // rasterizes the entire footprint. Tiny brushes fall back to a 2px floor so
        // the stroke stays gap-free; off-mesh segments also use the floor.
        const STEP_PX: f32 = 2.0;
        let dist = (to - from).length();
        let spacing = self
            .brush_screen_radius(to, brush)
            .map(|r| (r * 0.5).max(STEP_PX))
            .unwrap_or(STEP_PX);
        let steps = ((dist / spacing).ceil() as u32).clamp(1, 1024);
        for k in 1..=steps {
            let t = k as f32 / steps as f32;
            self.stamp_screen(from.lerp(to, t), brush);
        }
    }

    /// Stamp one flat disc directly in UV space (no raycast, no cross-face walk):
    /// `uv` in `[0,1]²` maps straight to a texel, and the dab is a circle in texture
    /// space — the 2D UV editor's brush. Reuses the same `deposit` discipline (and so
    /// the same undo/dirty-rect/upload path) as the surface brush. Texels outside any
    /// island (gutter / empty atlas) paint fine; they're simply never sampled by the
    /// mesh. Only mutates the layer + marks the dirty rect; upload coalesces into the
    /// next `flush_paint`.
    pub fn paint_uv_at(&mut self, uv: Vec2, brush: &Brush) {
        let uv = if brush.snap_to_texel {
            snap_uv(uv, self.tex_size, brush.snap_grid)
        } else {
            uv
        };
        let splats = uv_disc(uv, brush, self.tex_size);
        self.deposit(&splats, brush);
    }

    /// Paint a continuous stroke segment between two UV points, interpolating discs so
    /// a fast drag in the UV editor leaves a solid line. Spacing is in texels (~half
    /// the brush radius), mirroring the surface `paint_segment`. A zero-length segment
    /// (a single click) stamps exactly once.
    pub fn paint_uv_segment(&mut self, from: Vec2, to: Vec2, brush: &Brush) {
        let size = self.tex_size as f32;
        let dist = ((to - from) * size).length();
        let spacing = (brush.radius * 0.5).max(1.0);
        let steps = ((dist / spacing).ceil() as u32).clamp(1, 4096);
        for k in 1..=steps {
            let t = k as f32 / steps as f32;
            self.paint_uv_at(from.lerp(to, t), brush);
        }
    }

    /// The composited display atlas (CPU mirror) the 2D UV editor draws: `(size, pixels)`,
    /// `pixels` RGBA8 row-major, `size×size`. Pairs with [`paint_version`](Self::paint_version).
    pub fn atlas_view(&self) -> (u32, &[u8]) {
        (self.tex_size, &self.display_buf)
    }

    /// Bumps whenever the display atlas changes; the UV editor re-uploads its image only on change.
    pub fn paint_version(&self) -> u64 {
        self.paint_version
    }

    /// Bumps whenever the mesh is swapped; the UV editor rebuilds its wireframe only then.
    pub fn topo_version(&self) -> u64 {
        self.topo_version
    }

    /// Whether the mesh has usable UVs (false right after a fresh load that still needs
    /// an unwrap). The UV editor shows a hint instead of painting when this is false.
    pub fn mesh_has_uvs(&self) -> bool {
        !self.mesh.needs_uvs
    }

    /// The UV island wireframe as flat `[u0, v0, u1, v1]` segments in `[0,1]²` — every
    /// triangle's three edges, for the 2D editor's overlay. Rebuilt by the caller only
    /// when [`topo_version`](Self::topo_version) changes (meshes are low-poly, so the
    /// raw triangle edges are cheap and need no dedup).
    pub fn build_uv_edges(&self) -> Vec<[f32; 4]> {
        let v = &self.mesh.vertices;
        let mut edges = Vec::with_capacity(self.mesh.indices.len());
        for tri in self.mesh.indices.chunks_exact(3) {
            let p = [
                v[tri[0] as usize].uv,
                v[tri[1] as usize].uv,
                v[tri[2] as usize].uv,
            ];
            for k in 0..3 {
                let a = p[k];
                let b = p[(k + 1) % 3];
                edges.push([a[0], a[1], b[0], b[1]]);
            }
        }
        edges
    }

    /// Record the scene into `target_view`. Shared by window + offscreen paths.
    fn draw_into(&self, encoder: &mut wgpu::CommandEncoder, target_view: &wgpu::TextureView) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Adjustable background. The picker's color is sRGB; the clear
                    // value is linear, so convert (default = the old dark teal).
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: srgb_to_linear(self.bg_color[0]) as f64,
                        g: srgb_to_linear(self.bg_color[1]) as f64,
                        b: srgb_to_linear(self.bg_color[2]) as f64,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Confine the scene (mesh + grid) to the panel-free region. The clear above
        // still covered the whole attachment, so the panel's strip shows the
        // background until egui paints over it.
        let (vx, vy, vw, vh) = self.scene_viewport();
        rpass.set_viewport(vx, vy, vw, vh, 0.0, 1.0);

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..self.index_count, 0, 0..1);

        // Ground grid: drawn after the mesh so the depth test (Less) lets the
        // mesh occlude the lines passing behind it.
        if self.show_grid {
            rpass.set_pipeline(&self.line_pipeline);
            rpass.set_bind_group(0, &self.grid_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.grid_vertex_buffer.slice(..));
            rpass.draw(0..self.grid_vertex_count, 0..1);
        }

        // Active-face outline: the locked/hovered facet's boundary edges, drawn with
        // the grid pipeline (shares the scene view-proj) so it's occluded by nearer
        // geometry but reads on top of the face it traces (its verts are nudged out).
        if let Some(buf) = &self.outline_vertex_buffer {
            rpass.set_pipeline(&self.line_pipeline);
            rpass.set_bind_group(0, &self.grid_bind_group, &[]);
            rpass.set_vertex_buffer(0, buf.slice(..));
            rpass.draw(0..self.outline_vertex_count, 0..1);
        }

        // Brush-cursor ring: the brush footprint on the surface at the cursor. Same
        // pipeline/bind group as the grid; drawn after the outline so it reads on top.
        if self.brush_ring_count > 0 {
            rpass.set_pipeline(&self.line_pipeline);
            rpass.set_bind_group(0, &self.grid_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.brush_ring_buffer.slice(..));
            rpass.draw(0..self.brush_ring_count, 0..1);
        }

        // Orientation compass: always shown, in its own square viewport in the
        // bottom-left of the scene region (so the panel never covers it), drawn last
        // with depth-Always so it's never occluded. Skipped only when the region is
        // too small to fit it. Its axes are clickable — see `click_compass`.
        if let Some((x, y, size)) = self.compass_rect() {
            rpass.set_viewport(x, y, size, size, 0.0, 1.0);
            rpass.set_pipeline(&self.compass_pipeline);
            rpass.set_bind_group(0, &self.compass_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.compass_vertex_buffer.slice(..));
            rpass.draw(0..self.compass_vertex_count, 0..1);
        }
    }

    pub fn render(&mut self, ui: Option<UiPaint>) {
        let Some(surface) = &self.surface else {
            return; // headless renderer has nothing to present
        };
        let frame = match surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                surface.configure(&self.device, &self.config);
                return;
            }
            Err(e) => {
                log::error!("surface error: {e:?}");
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame encoder"),
            });

        // Scene → offscreen, quantize → surface, then egui on top (stays crisp).
        self.draw_into(&mut encoder, &view);
        if let Some(ui) = ui.as_ref() {
            self.encode_egui(&mut encoder, &view, ui);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
    }

    /// Record the egui overlay into `view`, loading (not clearing) the scene
    /// underneath. Shared by the window and offscreen paths.
    fn encode_egui(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        ui: &UiPaint,
    ) {
        let Some(er) = self.egui_renderer.as_mut() else {
            return;
        };
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: ui.pixels_per_point,
        };
        for (id, delta) in &ui.textures_delta.set {
            er.update_texture(&self.device, &self.queue, *id, delta);
        }
        er.update_buffers(&self.device, &self.queue, encoder, ui.jobs, &screen);

        let mut rpass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // keep the scene underneath
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            })
            .forget_lifetime();
        er.render(&mut rpass, ui.jobs, &screen);
        drop(rpass);

        for id in &ui.textures_delta.free {
            er.free_texture(id);
        }
    }

    /// Render one frame to an offscreen texture and read it back as RGBA8.
    /// Returns (pixels, width, height). Used by the `--screenshot` path. With
    /// `ui`, the egui overlay is drawn too, so the panel can be verified headless.
    pub fn capture(&mut self, ui: Option<UiPaint>) -> (Vec<u8>, u32, u32) {
        // Flush any pending stroke so a screenshot taken right after `paint_at` shows it.
        self.flush_paint();
        let width = self.config.width;
        let height = self.config.height;

        let color = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("capture color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OFFSCREEN_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());

        // Readback requires bytes_per_row aligned to 256.
        let unpadded = width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("capture readback"),
            size: (padded * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("capture encoder"),
            });
        self.draw_into(&mut encoder, &color_view);
        if let Some(ui) = ui.as_ref() {
            self.encode_egui(&mut encoder, &color_view, ui);
        }
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and copy out, stripping row padding.
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("failed to map readback buffer");

        let data = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((unpadded * height) as usize);
        for row in 0..height {
            let start = (row * padded) as usize;
            let end = start + unpadded as usize;
            pixels.extend_from_slice(&data[start..end]);
        }
        drop(data);
        buffer.unmap();

        (pixels, width, height)
    }
}

/// Map a live effect to its serializable form (G28). Kept here, like the
/// `BlendMode` ↔ index mapping, so serde stays off the core `effects::Effect`.
fn effect_to_doc(fx: &crate::effects::Effect) -> crate::project::EffectDoc {
    use crate::effects::Effect;
    use crate::project::EffectDoc;
    match *fx {
        Effect::HueSatLight { hue, sat, light } => EffectDoc::HueSatLight { hue, sat, light },
        Effect::BrightnessContrast {
            brightness,
            contrast,
        } => EffectDoc::BrightnessContrast {
            brightness,
            contrast,
        },
        Effect::Blur { radius } => EffectDoc::Blur { radius },
        Effect::Warp { amount, scale } => EffectDoc::Warp { amount, scale },
    }
}

fn doc_to_effect(doc: &crate::project::EffectDoc) -> crate::effects::Effect {
    use crate::effects::Effect;
    use crate::project::EffectDoc;
    match *doc {
        EffectDoc::HueSatLight { hue, sat, light } => Effect::HueSatLight { hue, sat, light },
        EffectDoc::BrightnessContrast {
            brightness,
            contrast,
        } => Effect::BrightnessContrast {
            brightness,
            contrast,
        },
        EffectDoc::Blur { radius } => Effect::Blur { radius },
        EffectDoc::Warp { amount, scale } => Effect::Warp { amount, scale },
    }
}

async fn request_adapter(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'static>>,
) -> wgpu::Adapter {
    instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: surface,
            force_fallback_adapter: false,
        })
        .await
        .expect("no suitable GPU adapter")
}

async fn request_device(adapter: &wgpu::Adapter) -> (wgpu::Device, wgpu::Queue) {
    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("lowtex device"),
                required_features: wgpu::Features::empty(),
                // Use the adapter's real limits (downlevel_defaults caps textures
                // at 2048, which a HiDPI window surface can exceed).
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .expect("failed to request device")
}

/// Create the GPU paint texture sized to match a CPU texture.
fn make_paint_texture(device: &wgpu::Device, cpu: &PaintTexture) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("paint texture"),
        size: wgpu::Extent3d {
            width: cpu.width,
            height: cpu.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// Build the paint-texture sampler for the given filter mode (G30). Address mode
/// stays Repeat so UV-tiled materials wrap; only the min/mag/mipmap filters change.
fn make_sampler(device: &wgpu::Device, filter: TextureFilter) -> wgpu::Sampler {
    let mode = filter.wgpu();
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("paint sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: mode,
        min_filter: mode,
        mipmap_filter: mode,
        ..Default::default()
    })
}

/// Bind group for the scene: view-proj uniform + paint texture + sampler.
fn make_paint_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    paint_texture: &wgpu::Texture,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    let view = paint_texture.create_view(&wgpu::TextureViewDescriptor::default());
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

/// Bind group for a line pipeline: just a view-proj uniform at binding 0.
fn make_line_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("line bind group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }],
    })
}

/// sRGB → linear for one channel. The background picker works in sRGB (what the
/// user sees); a render-pass clear value is linear, so convert at clear time.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// The ground grid: evenly spaced lines on the world XZ plane out to `EXTENT`,
/// with the two lines through the origin (the X and Z axes) a touch brighter so
/// the center reads. Colors are linear (see lines.wgsl).
fn build_grid() -> Vec<LineVertex> {
    const EXTENT: f32 = 4.0;
    const STEP: f32 = 0.5;
    let faint = [0.11, 0.11, 0.13];
    let center = [0.24, 0.24, 0.28];
    let n = (EXTENT / STEP) as i32;
    let mut v = Vec::with_capacity(((2 * n + 1) * 4) as usize);
    for i in -n..=n {
        let t = i as f32 * STEP;
        let c = if i == 0 { center } else { faint };
        // Line parallel to X (fixed z = t).
        v.push(LineVertex {
            position: [-EXTENT, 0.0, t],
            color: c,
        });
        v.push(LineVertex {
            position: [EXTENT, 0.0, t],
            color: c,
        });
        // Line parallel to Z (fixed x = t).
        v.push(LineVertex {
            position: [t, 0.0, -EXTENT],
            color: c,
        });
        v.push(LineVertex {
            position: [t, 0.0, EXTENT],
            color: c,
        });
    }
    v
}

/// Side (the screen-space half-width multiplier, see compass.wgsl) of the bold
/// positive arms and the thinner negative stubs.
const COMPASS_ARM_SIDE: f32 = 1.0;
const COMPASS_STUB_SIDE: f32 = 0.55;

/// Append one thick segment (`a` → `b`) as a quad: two triangles, 6 vertices,
/// each carrying the whole segment plus its corner (`t`, `side`). compass.wgsl
/// expands the corners perpendicular to the segment in screen space.
fn push_compass_segment(
    v: &mut Vec<CompassVertex>,
    a: [f32; 3],
    b: [f32; 3],
    color: [f32; 3],
    half: f32,
) {
    // Corners as (t, signed side): a-left, a-right, b-right / b-right, a-left, b-left.
    let corners = [
        (0.0, half),
        (0.0, -half),
        (1.0, -half),
        (1.0, -half),
        (0.0, half),
        (1.0, half),
    ];
    for (t, side) in corners {
        v.push(CompassVertex {
            start: a,
            end: b,
            color,
            param: [t, side],
        });
    }
}

/// The orientation compass: three bold axis arms from the origin — +X red,
/// +Y green, +Z blue at unit length — each with a thinner, dimmer negative stub
/// so the sign reads. Built as thick screen-space quads (see compass.wgsl) and
/// rendered through the camera's rotation-only `gizmo_view_proj`.
fn build_compass() -> Vec<CompassVertex> {
    let axes = [
        ([1.0_f32, 0.0, 0.0], [0.90_f32, 0.20, 0.20]), // X red
        ([0.0, 1.0, 0.0], [0.40, 0.85, 0.30]),         // Y green
        ([0.0, 0.0, 1.0], [0.30, 0.55, 0.95]),         // Z blue
    ];
    let origin = [0.0, 0.0, 0.0];
    let mut v = Vec::with_capacity(axes.len() * 2 * 6);
    for (dir, color) in axes {
        let dim = [color[0] * 0.35, color[1] * 0.35, color[2] * 0.35];
        // Positive arm: origin → +1, bold.
        push_compass_segment(&mut v, origin, dir, color, COMPASS_ARM_SIDE);
        // Negative stub: origin → -0.4, thinner and dimmed.
        let stub = [-dir[0] * 0.4, -dir[1] * 0.4, -dir[2] * 0.4];
        push_compass_segment(&mut v, origin, stub, dim, COMPASS_STUB_SIDE);
    }
    v
}

fn make_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Copy `rect` out of a full-size `width`-wide RGBA8 buffer into a contiguous
/// `rect.width()*rect.height()*4` buffer (row-major). Used to snapshot the
/// processed region before a region refresh, and to pack a sub-rect for upload.
/// Rasterize a flat brush disc centered at UV `uv` (`[0,1]²`) into `(texel, coverage)`
/// pairs over a `size×size` atlas. `coverage = falloff(d, hardness) * opacity`, sampled
/// at each texel center within `brush.radius` texels of the center. The UV-editor analog
/// of `surface::splat`, but a plain circle in texture space (no mesh walk).
/// Snap a UV to the center of the `grid`×`grid`-texel cell it lands in, at texture
/// resolution `size`. `(floor(t/grid) + 0.5) * grid` puts the point dead-center in its
/// cell, so repeated snapped dabs along a drag land on a clean coarse lattice (the
/// blocky PSX look). `grid` is clamped to at least one texel.
fn snap_uv(uv: Vec2, size: u32, grid: f32) -> Vec2 {
    let sz = size as f32;
    let g = grid.max(1.0);
    let snap = |u: f32| (((u * sz / g).floor() + 0.5) * g) / sz;
    Vec2::new(snap(uv.x), snap(uv.y))
}

/// Barycentric coords of `p` in triangle `(a, b, c)` (all in 2D UV space). Returns
/// `None` for a degenerate (zero-area) triangle. Coords are not clamped: a point just
/// outside the triangle yields weights slightly outside `[0,1]`, which is fine for
/// re-deriving a snapped world position near an edge.
fn uv_bary(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> Option<glam::Vec3> {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let det = v0.x * v1.y - v1.x * v0.y;
    if det.abs() < 1e-12 {
        return None;
    }
    let w1 = (v2.x * v1.y - v1.x * v2.y) / det;
    let w2 = (v0.x * v2.y - v2.x * v0.y) / det;
    Some(glam::Vec3::new(1.0 - w1 - w2, w1, w2))
}

fn uv_disc(uv: Vec2, brush: &Brush, size: u32) -> Vec<(usize, f32)> {
    let sz = size as f32;
    let cx = uv.x * sz;
    let cy = uv.y * sz;
    let r = brush.radius.max(0.5);
    let x0 = (cx - r).floor().clamp(0.0, sz) as u32;
    let x1 = (cx + r).ceil().clamp(0.0, sz) as u32;
    let y0 = (cy - r).floor().clamp(0.0, sz) as u32;
    let y1 = (cy + r).ceil().clamp(0.0, sz) as u32;
    let mut out = Vec::new();
    for ty in y0..y1 {
        for tx in x0..x1 {
            let dx = tx as f32 + 0.5 - cx;
            let dy = ty as f32 + 0.5 - cy;
            let d = (dx * dx + dy * dy).sqrt() / r;
            let a = crate::paint::falloff(d, brush.hardness) * brush.opacity;
            if a > 0.0 {
                out.push(((ty * size + tx) as usize, a));
            }
        }
    }
    out
}

fn copy_region(buf: &[u8], width: u32, rect: TexRect) -> Vec<u8> {
    let (rw, rh) = (rect.width() as usize, rect.height() as usize);
    let mut out = vec![0u8; rw * rh * 4];
    let w = width as usize;
    for ry in 0..rh {
        let src = ((rect.y0 as usize + ry) * w + rect.x0 as usize) * 4;
        let dst = ry * rw * 4;
        out[dst..dst + rw * 4].copy_from_slice(&buf[src..src + rw * 4]);
    }
    out
}

/// Restore the texels of `proc` that are *not* in `upload` from `snapshot` (a
/// `copy_region(.., proc)` taken before the region refresh). These margin texels are
/// unchanged by the stroke but can be miscomputed by the region dilate, so we put
/// their known-correct pre-refresh values back — keeping `display_buf` exact.
fn restore_margin(buf: &mut [u8], snapshot: &[u8], width: u32, proc: TexRect, upload: TexRect) {
    let (pw, ph) = (proc.width() as usize, proc.height() as usize);
    let w = width as usize;
    for py in 0..ph {
        let y = proc.y0 as usize + py;
        for px in 0..pw {
            let x = proc.x0 as usize + px;
            if upload.contains(x as u32, y as u32) {
                continue;
            }
            let d = (y * w + x) * 4;
            let s = (py * pw + px) * 4;
            buf[d..d + 4].copy_from_slice(&snapshot[s..s + 4]);
        }
    }
}

/// Upload only `rect` of a full-size `width`-wide RGBA8 buffer into the GPU texture,
/// at the matching origin. The sub-region counterpart to `upload_pixels`.
fn upload_region(queue: &wgpu::Queue, gpu: &wgpu::Texture, buf: &[u8], width: u32, rect: TexRect) {
    let packed = copy_region(buf, width, rect);
    upload_packed(queue, gpu, &packed, rect);
}

/// Upload an already-packed `rect`-sized RGBA8 buffer (row-major, `rect.width()`
/// texels per row) into `rect` of the GPU texture. The packed-buffer counterpart
/// to `upload_region`, used by the brush preview which builds its region by hand.
fn upload_packed(queue: &wgpu::Queue, gpu: &wgpu::Texture, packed: &[u8], rect: TexRect) {
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: gpu,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: rect.x0,
                y: rect.y0,
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        },
        packed,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(rect.width() * 4),
            rows_per_image: Some(rect.height()),
        },
        wgpu::Extent3d {
            width: rect.width(),
            height: rect.height(),
            depth_or_array_layers: 1,
        },
    );
}

/// Upload a raw RGBA8 pixel slice into a GPU texture.
fn upload_pixels(queue: &wgpu::Queue, gpu: &wgpu::Texture, pixels: &[u8], width: u32, height: u32) {
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: gpu,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        pixels,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

fn make_checkerboard(w: u32, h: u32) -> PaintTexture {
    let mut tex = PaintTexture::new(w, h, [255, 255, 255, 255]);
    for y in 0..h {
        for x in 0..w {
            let on = ((x / 8) + (y / 8)) % 2 == 0;
            let c: u8 = if on { 230 } else { 200 };
            let i = ((y * w + x) * 4) as usize;
            tex.pixels[i] = c;
            tex.pixels[i + 1] = c;
            tex.pixels[i + 2] = c;
            tex.pixels[i + 3] = 255;
        }
    }
    tex
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh;

    fn headless() -> Renderer {
        pollster::block_on(Renderer::new_headless(64, 64, Mesh::cube()))
    }

    /// The save/autosave dirty machine (G31): a mutation dirties the document for
    /// both the user and the autosave timer; an autosave only clears the timer's
    /// watermark (the user still has unsaved work); an explicit save clears both;
    /// and a load starts clean.
    #[test]
    fn dirty_tracking_drives_save_and_autosave() {
        let mut r = headless();
        assert!(!r.is_dirty(), "a fresh document is clean");
        assert!(!r.needs_autosave());

        // A discrete edit (add a layer → checkpoint) dirties both flags.
        r.add_layer();
        assert!(r.is_dirty());
        assert!(r.needs_autosave());

        // An autosave captures the timer's watermark but not the user's: the file
        // they care about still doesn't reflect this work.
        let path = std::env::temp_dir().join("lowtex_dirty_autosave_test.lowtex");
        let p = path.to_string_lossy().to_string();
        r.autosave(&p).unwrap();
        assert!(
            r.is_dirty(),
            "autosave must not clear the user-facing dirty flag"
        );
        assert!(
            !r.needs_autosave(),
            "autosave captures the autosave watermark"
        );

        // An explicit save clears both.
        r.save_project(&p).unwrap();
        assert!(!r.is_dirty());
        assert!(!r.needs_autosave());

        // Undo is itself a document change → dirty again.
        r.undo();
        assert!(r.is_dirty());

        // A load resets to the on-disk state: clean.
        r.load_project(&p).unwrap();
        assert!(!r.is_dirty());
        assert!(!r.needs_autosave());

        let _ = std::fs::remove_file(&path);
    }

    /// The dirty-rectangle invariant: after a brush-sized mutation + region refresh,
    /// `display_buf` must be byte-identical to a full `composite_display()`. This is
    /// what guarantees the live (region-uploaded) view never diverges from the
    /// full-composite export path.
    fn assert_region_matches_full(r: &mut Renderer) {
        // Bake the current layer/palette setup into display_buf via a full refresh.
        r.refresh_display_texture();
        // Simulate a stamp: overwrite a small rect of the active layer, opaque.
        let size = r.tex_size;
        let rect = TexRect {
            x0: 20,
            y0: 22,
            x1: 30,
            y1: 31,
        };
        {
            let tex = r.layers.active_tex_mut();
            for y in rect.y0..rect.y1 {
                for x in rect.x0..rect.x1 {
                    let i = ((y * size + x) * 4) as usize;
                    tex.pixels[i..i + 4].copy_from_slice(&[123, 45, 67, 255]);
                }
            }
        }
        r.refresh_display_region(rect);
        let full = r.composite_display();
        assert_eq!(
            r.display_buf, full,
            "region refresh diverged from full composite"
        );
    }

    #[test]
    fn region_matches_full_no_palette() {
        let mut r = headless();
        r.palette_settings.enabled = false;
        assert_region_matches_full(&mut r);
    }

    #[test]
    fn region_matches_full_with_palette_dither() {
        let mut r = headless();
        r.palette_settings = PaletteSettings {
            enabled: true,
            dither: true,
            dither_strength: 0.06,
        };
        assert_region_matches_full(&mut r);
    }

    #[test]
    fn region_matches_full_multiply_layer_and_mask() {
        let mut r = headless();
        r.layers.add_layer(); // active is now the top layer
        r.layers.layers[1].blend = crate::layers::BlendMode::Multiply;
        for (t, px) in r.layers.layers[1]
            .mask
            .pixels
            .chunks_exact_mut(4)
            .enumerate()
        {
            let v = ((t * 7) % 256) as u8;
            px.copy_from_slice(&[v, v, v, 255]);
        }
        assert_region_matches_full(&mut r);
    }

    #[test]
    fn real_stroke_flush_matches_full() {
        // Exercise the true path: actual picking + many overlapping interpolated stamps
        // accumulated into one dirty rect, flushed via `flush_paint` (the per-frame
        // entry point). The resulting GPU mirror must still equal a full composite.
        // A viewport that frames the cube at screen centre (same as the screenshot path).
        let (w, h) = (256u32, 256u32);
        let mut r = pollster::block_on(Renderer::new_headless(w, h, Mesh::cube()));
        r.palette_settings = PaletteSettings {
            enabled: true,
            dither: true,
            dither_strength: 0.06,
        };
        r.refresh_display_texture(); // bake the palette into display_buf (as set_palette_settings does)
        let brush = Brush {
            color: [0.1, 0.7, 0.3],
            radius: 6.0,
            ..Brush::default()
        };
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        r.begin_stroke();
        r.paint_at((cx, cy), &brush);
        r.paint_segment((cx - 20.0, cy - 14.0), (cx + 20.0, cy + 14.0), &brush);
        r.end_stroke();
        assert!(r.dirty_rect.is_some(), "stroke never hit the mesh");
        r.flush_paint();
        assert!(
            r.dirty_rect.is_none(),
            "flush_paint should consume the dirty rect"
        );
        let full = r.composite_display();
        assert_eq!(
            r.display_buf, full,
            "flushed stroke diverged from full composite"
        );
    }

    #[test]
    fn symmetry_mirrors_the_dab_to_the_other_side() {
        // Painting one off-centre dab with X-symmetry on must paint a second,
        // disjoint cluster on the mirror side: the symmetric changed-texel set is a
        // superset of the plain one and is ~twice as large (the mirror is a distinct
        // UV region, so the two clusters don't overlap).
        let (w, h) = (256u32, 256u32);
        let brush = Brush {
            color: [0.1, 0.7, 0.3],
            radius: 5.0,
            ..Brush::default()
        };
        // Off-centre horizontally so the hit sits well off the x=0 symmetry plane and
        // its mirror lands in a different part of the atlas.
        let (cx, cy) = (w as f32 / 2.0 + 24.0, h as f32 / 2.0);

        // The set of texels a single dab changes, for a given symmetry setting.
        let changed = |axis: Option<SymmetryAxis>| -> std::collections::HashSet<usize> {
            let mut r = pollster::block_on(Renderer::new_headless(w, h, Mesh::cube()));
            r.set_symmetry(axis);
            let before = r.layers.active_tex().pixels.clone();
            r.begin_stroke();
            r.paint_at((cx, cy), &brush);
            r.end_stroke();
            before
                .chunks_exact(4)
                .zip(r.layers.active_tex().pixels.chunks_exact(4))
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(i, _)| i)
                .collect()
        };

        let plain = changed(None);
        let mirrored = changed(Some(SymmetryAxis::X));
        assert!(
            !plain.is_empty(),
            "the dab must paint something with symmetry off"
        );
        assert!(
            plain.is_subset(&mirrored),
            "symmetry must keep the original dab and add to it, not replace it"
        );
        assert!(
            mirrored.len() as f32 > plain.len() as f32 * 1.5,
            "the mirror should roughly double the painted area (plain {}, mirrored {})",
            plain.len(),
            mirrored.len()
        );
    }

    #[test]
    fn lock_face_keeps_the_dab_on_one_face() {
        // A dab at a cube face's centre with a radius wider than the face wraps onto
        // the neighbouring faces (the cross-face surface splat). With the face lock
        // on, the same dab must paint only the one facet it landed on.
        let (w, h) = (256u32, 256u32);
        let brush = Brush {
            color: [0.1, 0.7, 0.3],
            radius: 64.0, // wider than half the face → reaches across the edges
            ..Brush::default()
        };
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);

        // The set of facets a single dab touches, for a given lock setting.
        let facets_touched = |lock: bool| -> std::collections::HashSet<i32> {
            let mut r = pollster::block_on(Renderer::new_headless(w, h, Mesh::cube()));
            r.set_lock_face(lock);
            let before = r.layers.active_tex().pixels.clone();
            r.begin_stroke();
            r.paint_at((cx, cy), &brush);
            r.end_stroke();
            r.ensure_fill_map();
            let map = r.fill_map.as_ref().unwrap();
            before
                .chunks_exact(4)
                .zip(r.layers.active_tex().pixels.chunks_exact(4))
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(i, _)| map.texel_facet[i])
                .filter(|&f| f >= 0)
                .collect()
        };

        let unlocked = facets_touched(false);
        let locked = facets_touched(true);
        assert!(
            unlocked.len() >= 2,
            "a dab wider than the face should wrap onto its neighbours (got {} facets)",
            unlocked.len()
        );
        assert_eq!(
            locked.len(),
            1,
            "with the face lock on, the dab must stay on a single facet (got {})",
            locked.len()
        );
        assert!(
            locked.is_subset(&unlocked),
            "the locked facet must be one the unlocked dab also painted"
        );
    }

    #[test]
    fn face_outline_traces_the_locked_facet_boundary() {
        // With face-lock on, hovering the brush over a cube face must produce an
        // outline of that face: a quad (two coplanar triangles) has four boundary
        // edges — the shared diagonal is interior, so 4 segments = 8 line vertices.
        // With the lock off, nothing is outlined.
        let (w, h) = (256u32, 256u32);
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        let mut r = pollster::block_on(Renderer::new_headless(w, h, Mesh::cube()));

        r.set_lock_face(false);
        r.set_face_outline(Some((cx, cy)));
        assert_eq!(r.outline_vertex_count, 0, "lock off → no outline");

        r.set_lock_face(true);
        r.set_face_outline(Some((cx, cy)));
        assert_eq!(
            r.outline_vertex_count, 8,
            "a cube face is a quad: 4 boundary edges → 8 line vertices"
        );

        // Hovering off the mesh clears the outline again.
        r.set_face_outline(Some((1.0, 1.0)));
        assert_eq!(
            r.outline_vertex_count, 0,
            "ray missing the mesh → no outline"
        );
    }

    #[test]
    fn brush_cursor_ring_shows_over_mesh_and_hides_off_it() {
        // A painting tool over the mesh shows the ring (one full loop of segments);
        // pointing off the mesh, or passing no cursor, hides it.
        let (w, h) = (256u32, 256u32);
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        let brush = Brush {
            radius: 8.0,
            ..Brush::default()
        };
        let mut r = pollster::block_on(Renderer::new_headless(w, h, Mesh::cube()));

        r.set_brush_cursor(Some((cx, cy)), &brush);
        assert_eq!(
            r.brush_ring_count,
            BRUSH_RING_SEGMENTS * 2,
            "the ring over the mesh is a closed loop of line segments"
        );

        r.set_brush_cursor(Some((1.0, 1.0)), &brush);
        assert_eq!(r.brush_ring_count, 0, "off the mesh → no ring");

        r.set_brush_cursor(None, &brush);
        assert_eq!(r.brush_ring_count, 0, "no painting tool → no ring");
    }

    #[test]
    fn brush_preview_ghosts_without_committing() {
        // Hovering the brush over the mesh arms a preview region, but must not alter
        // the layers or the committed `display_buf` (the ghost lives only on the GPU).
        // Pointing off the mesh clears it again.
        let (w, h) = (256u32, 256u32);
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        let brush = Brush {
            color: [0.9, 0.1, 0.1],
            radius: 8.0,
            ..Brush::default()
        };
        let mut r = pollster::block_on(Renderer::new_headless(w, h, Mesh::cube()));
        let layer_before = r.layers.active_tex().pixels.clone();
        let display_before = r.display_buf.clone();

        r.set_brush_preview(Some((cx, cy)), &brush);
        assert!(
            r.preview_rect.is_some(),
            "hovering the mesh should arm a preview"
        );
        assert_eq!(
            r.layers.active_tex().pixels,
            layer_before,
            "preview must not write into the layer"
        );
        assert_eq!(
            r.display_buf, display_before,
            "preview must not touch the committed display mirror"
        );

        r.set_brush_preview(Some((1.0, 1.0)), &brush);
        assert!(
            r.preview_rect.is_none(),
            "pointing off the mesh clears the preview"
        );
        assert_eq!(
            r.display_buf, display_before,
            "reverting leaves the committed mirror intact"
        );
    }

    #[test]
    fn region_matches_full_with_blur_effect() {
        let mut r = headless();
        // A blur on the painted layer leaks the stamp beyond its footprint; the region
        // path widens by the blur radius, so the invariant must still hold.
        r.layers.layers[0]
            .effects
            .push(crate::effects::Effect::Blur { radius: 2.0 });
        assert_region_matches_full(&mut r);
    }

    #[test]
    fn brush_preview_skips_recompute_when_inputs_unchanged() {
        // The hover ghost must recompute only when its inputs change — the guard that
        // keeps an idle frame from re-picking + re-splatting the whole footprint. We
        // observe the skip by clearing `preview_rect` ourselves: a skipped call returns
        // before touching it (stays cleared); a recompute sets it again.
        let (w, h) = (256u32, 256u32);
        let mut r = pollster::block_on(Renderer::new_headless(w, h, Mesh::cube()));
        r.refresh_display_texture();
        let brush = Brush {
            color: [0.1, 0.7, 0.3],
            radius: 6.0,
            ..Brush::default()
        };
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);

        // First hover over the cube: a ghost is laid down.
        r.set_brush_preview(Some((cx, cy)), &brush);
        assert!(
            r.preview_rect.is_some(),
            "preview should ghost over the mesh"
        );
        assert!(r.last_preview.is_some());

        // Identical inputs → skip: the cleared rect stays cleared.
        r.preview_rect = None;
        r.set_brush_preview(Some((cx, cy)), &brush);
        assert!(
            r.preview_rect.is_none(),
            "identical inputs must skip the recompute"
        );

        // A brush change recomputes.
        let brush2 = Brush {
            color: [0.9, 0.1, 0.1],
            ..brush
        };
        r.set_brush_preview(Some((cx, cy)), &brush2);
        assert!(
            r.preview_rect.is_some(),
            "a brush change must recompute the ghost"
        );

        // A camera move recomputes too (the cursor now lands on a different point).
        r.preview_rect = None;
        r.orbit_camera(20.0, 12.0);
        r.set_brush_preview(Some((cx, cy)), &brush2);
        assert!(
            r.preview_rect.is_some(),
            "a camera move must recompute the ghost"
        );

        // Committing a stroke re-shows the ghost over the new pixels (no stale skip).
        r.begin_stroke();
        assert!(
            r.last_preview.is_none(),
            "begin_stroke must drop the hover ghost key"
        );
        r.end_stroke();
        assert!(
            r.last_preview.is_none(),
            "end_stroke must force a fresh ghost"
        );
    }
}

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

/// The pre-stroke colour + mask of the one layer a stroke touches — the resolve/deposit
/// base and the single-layer undo snapshot (see `Renderer::pending`).
struct StrokeBackup {
    active: usize,
    tex: PaintTexture,
    mask: PaintTexture,
}

/// What the brush paints into (G11).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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

/// Skip the per-texel hover ghost once a dab's UV footprint exceeds this many texels.
/// The ghost re-runs the CPU `splat` on every cursor move, so for a large brush it costs
/// as much as a real dab (tens of ms/frame) — the dominant idle-frame cost. Past this the
/// cursor ring alone conveys the brush size; smaller brushes keep the live colour ghost.
const PREVIEW_MAX_TEXELS: usize = 30_000;

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

    // Stroke accumulation (G6): per-texel coverage so overlapping stamps within one
    // stroke don't double-darken, plus `stroke_base` — the active-layer values a
    // texel had *before* this stroke, which the max-coverage blend re-composites
    // against. Both are lazily, per-texel scoped to the current stroke by a version
    // stamp instead of being cleared/cloned wholesale each stroke: `stroke_stamp`
    // records the `stroke_id` a texel was last written under, so `coverage`/`base`
    // count only when the stamp matches the live id. `begin_stroke` (and any edit
    // that changes the target outside a stroke) just bumps the id — O(1) — rather
    // than zero-filling and cloning the whole (up to 2048²) texture every stroke.
    stroke_base: Vec<u8>,
    stroke_coverage: Vec<f32>,
    stroke_stamp: Vec<u32>,
    stroke_id: u32,

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
    // Brush alpha tip: an optional grayscale image that shapes the dab in place of the
    // built-in circular falloff. When set, each dab projects every covered texel into the
    // same oriented tangent frame the decal stamp uses (`decal_frame`, world-up aligned +
    // `stamp_angle`) and reads the image's brightness (× its alpha) as coverage — so a
    // star/square/grain tip paints the brush colour in that shape. `invert` flips it for
    // black-on-white tip packs. The image is an RGBA8 `Material` (same loader/thumbnail).
    brush_alpha: Option<crate::material::Material>,
    brush_alpha_invert: bool,

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
    // The *active layer* as it was when the in-progress stroke began — the immutable base
    // the per-frame resolve/deposit blends over, and the single-layer undo snapshot
    // committed to history on `end_stroke` (only if the stroke painted). Just the touched
    // layer, not the whole stack: a stroke can't change any other layer, and the full-stack
    // clone was a ~quarter-second freeze at every stroke start at 4K.
    pending: Option<StrokeBackup>,
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
    // GPU mesh-map baker (Phase 2.5): ray-traces AO + sun against the BVH on the GPU,
    // so AO scales to 2K/4K and the sun is an interactive slider. The CPU rasterization
    // still feeds it pos/nrm/mask; `bake::bake`/`compute_light` stay the parity oracle.
    gpu_baker: crate::gpu_bake::GpuBaker,

    // GPU dab stamping (Phase 1, LOWTEX_GPU_PAINT). When `gpu_paint` is on, the core
    // solid-colour surface brush stamps its dabs into a GPU coverage texture
    // (`surface::splat_faces` → `gpu_dab`) instead of the CPU `splat` + `deposit`; the
    // accumulated coverage is resolved into the active layer once per frame. Mask / erase
    // / image / decal strokes stay on the CPU path. The CPU `splat`+`deposit` remain the
    // default and the parity oracle (`real_stroke_flush_matches_full`, gpu_dab tests).
    gpu_dab: crate::gpu_dab::GpuDab,
    gpu_paint: bool,
    // GPU display compositing (Phase 1–3, LOWTEX_GPU_DISPLAY). When on, the layer stack is
    // composited + palette-quantized + gutter-bled on the GPU into `gpu_layers`' atlas, and
    // the model samples that atlas (`gpu_bind_group`) instead of the CPU `paint_texture_gpu`.
    // Off by default; the CPU `composite_display` stays the export path + parity oracle.
    gpu_layers: crate::gpu_layers::GpuLayers,
    gpu_display: bool,
    gpu_bind_group: Option<wgpu::BindGroup>,
    // For a resolve stroke painting a tiled material brush: the tile factor (the material
    // texture is mirrored to the GPU on demand, keyed on `gpu_material_gen`). `None` = solid
    // colour / eraser.
    gpu_stroke_material: Option<f32>,
    gpu_material_gen: Option<u64>,
    // True for the current stroke when it takes the fully-GPU resolve path (no in-stroke
    // readback): coverage resolves straight into the active layer's GPU slice, the display
    // composites from the slices, and the CPU `Layers` is reconciled once at `end_stroke`.
    // Only solid-colour / eraser surface strokes with GPU display on; everything else keeps
    // the readback path. Decided at the stroke's first dab (`gpu_surface_dab`).
    gpu_stroke_resolve: bool,
    // The (resolution, topology) the static UV coverage currently on `gpu_layers` was built
    // for. The gutter mask depends on both, so re-push when either changes — a same-resolution
    // mesh swap (different UVs) must NOT keep bleeding with the old gutters.
    gpu_coverage_gen: Option<(u32, u64)>,
    // The texel region stamped since the last resolve (the per-frame dirty rect for the
    // GPU stroke), the stroke's solid colour to blend the coverage with at resolve, and
    // whether the stroke erases (lowers alpha toward transparent) rather than lays colour.
    gpu_stroke_rect: Option<TexRect>,
    // The union of every texel a stroke painted, accumulated across the whole stroke (not
    // consumed per frame like `dirty_rect`/`gpu_stroke_rect`). Under GPU display the in-stroke
    // paint paths leave the CPU display mirror (`display_buf`/`paint_texture_gpu`) stale — the
    // model samples the GPU atlas, but the 2D UV editor, brush preview and export read the CPU
    // mirror — so `end_stroke` re-syncs exactly this region once the layers are reconciled.
    stroke_paint_rect: Option<TexRect>,
    // True while the active stroke is a 2D UV-editor stroke (flat discs painted straight in
    // texel space). The user watches the 2D panel (the CPU `display_buf` mirror) as they paint,
    // so under GPU display this keeps the cheap CPU display path live each frame — otherwise the
    // panel wouldn't refresh until `end_stroke`, looking like the stroke never lands. A 3D model
    // stroke leaves this false: the model samples the GPU atlas live; the panel waits for the
    // end-of-stroke re-sync.
    uv_stroke: bool,
    gpu_stroke_color: [u8; 3],
    gpu_stroke_erase: bool,
    // Which target this GPU stroke resolves into — the layer's colour or its reveal mask
    // (captured per dab; `gpu_stroke_color` carries the solid colour, or white/black for
    // mask reveal/hide).
    gpu_stroke_target: PaintTarget,
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
        Self::build_headless(device, queue, width, height, mesh)
    }

    /// Build a headless renderer onto an already-acquired device/queue (sync). Split
    /// out of `new_headless` so the test suite can share one device across many renderers
    /// — each test creating its own `wgpu::Device` exhausts the driver (OutOfMemory) once
    /// enough run in parallel. Production still calls `new_headless`, which owns its device.
    pub(crate) fn build_headless(
        device: wgpu::Device,
        queue: wgpu::Queue,
        width: u32,
        height: u32,
        mesh: Mesh,
    ) -> Self {
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

        // Stroke buffers sized to the texture; `stroke_base`/`stroke_coverage` start
        // garbage and are validated per-texel by `stroke_stamp` against `stroke_id`
        // (which begins at 1 so the zero-initialized stamps read as "not this stroke").
        let texels = (tex_size * tex_size) as usize;
        let stroke_base = vec![0u8; texels * 4];
        let stroke_coverage = vec![0.0f32; texels];
        let stroke_stamp = vec![0u32; texels];

        // GPU mesh-map baker (built once; uploads its mesh on the first map/AO/sun use).
        let gpu_baker = crate::gpu_bake::GpuBaker::new(&device);
        // GPU dab stamper (built once; its stroke target is created on the first stroke).
        let gpu_dab = crate::gpu_dab::GpuDab::new(&device);
        // GPU layer compositor (built once; residency created on the first display sync).
        let gpu_layers = crate::gpu_layers::GpuLayers::new(&device);

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
            stroke_stamp,
            stroke_id: 1,
            display_buf: Vec::new(),
            dirty_rect: None,
            paint_target: PaintTarget::Color,
            mask_reveal: false,
            brush_material: None,
            brush_tile: 4.0,
            brush_image_mode: BrushImageMode::Tiled,
            stamp_angle: 0.0,
            stamp_tint: false,
            brush_alpha: None,
            brush_alpha_invert: false,
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
            gpu_baker,
            gpu_dab,
            gpu_paint: force_gpu_paint(),
            gpu_layers,
            gpu_display: force_gpu_display(),
            gpu_bind_group: None,
            gpu_stroke_material: None,
            gpu_material_gen: None,
            gpu_stroke_resolve: false,
            gpu_coverage_gen: None,
            gpu_stroke_rect: None,
            stroke_paint_rect: None,
            uv_stroke: false,
            gpu_stroke_color: [0, 0, 0],
            gpu_stroke_erase: false,
            gpu_stroke_target: PaintTarget::Color,
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
        self.update_gpu_display();
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

        crate::perf::time("composite_region", || {
            self.layers
                .composite_into_region(&mut self.display_buf, proc);
        });
        if self.palette_settings.enabled && !self.palette.colors.is_empty() {
            crate::perf::time("quantize_region", || {
                self.palette.quantize_region(
                    &mut self.display_buf,
                    size,
                    proc,
                    self.palette_settings.dither,
                    self.palette_settings.dither_strength,
                );
            });
        }
        self.ensure_coverage();
        if let Some(cov) = self.coverage.borrow().as_ref() {
            crate::perf::time("bleed_region", || {
                crate::bleed::dilate_region(&mut self.display_buf, cov, size, pad, proc);
            });
        }
        // Restore the margin (processed but not uploaded): unchanged by the stroke, but
        // possibly miscomputed by the region dilate.
        restore_margin(&mut self.display_buf, &before, size, proc, upload);

        crate::perf::time("upload_region", || {
            upload_region(
                &self.queue,
                &self.paint_texture_gpu,
                &self.display_buf,
                size,
                upload,
            );
        });
        self.paint_version = self.paint_version.wrapping_add(1);
    }

    /// (Re)build the GPU composite atlas (composite + palette quantize + gutter bleed on
    /// the GPU) from the current CPU layer stack, and point the model at it. No-op unless
    /// `LOWTEX_GPU_DISPLAY` is on. The CPU `display_buf` / `paint_texture_gpu` still feed the
    /// 2D UV editor + brush preview (export composites fresh); they're kept in sync by the
    /// CPU refreshes here and by `end_stroke`'s per-region re-sync after a GPU-display stroke.
    /// The GPU pipeline is the byte-parity twin of `composite_display` (see gpu_layers tests).
    fn update_gpu_display(&mut self) {
        if !self.gpu_display {
            return;
        }
        self.gpu_sync_layers();
        self.gpu_compose();
    }

    /// Mirror the CPU layer stack into the GPU layer arrays. The expensive part of a GPU
    /// display refresh; skipped during a resolve stroke (the active slice runs ahead on the
    /// GPU, so re-uploading the stale CPU pixels would clobber it).
    fn gpu_sync_layers(&mut self) {
        self.gpu_layers.upload(&self.device, &self.queue, &self.layers);
    }

    /// Composite the GPU layer arrays → atlas (palette quantize + gutter bleed folded in)
    /// and point the model at the result. Reads whatever is currently in the layer slices,
    /// so it's correct both after a `gpu_sync_layers` (display refresh) and after a paint
    /// `resolve_active` (resolve stroke).
    fn gpu_compose(&mut self) {
        let ps = self.palette_settings;
        self.gpu_layers.set_quantize(
            &self.queue,
            &self.palette,
            ps.enabled && !self.palette.colors.is_empty(),
            ps.dither,
            ps.dither_strength,
        );
        // Push the static UV coverage for the gutter bleed. Keyed on (resolution, topology):
        // `topo_version` bumps on every mesh load / re-unwrap — exactly when the UV layout (and
        // so the gutter mask) changes — so a same-resolution mesh swap re-pushes instead of
        // bleeding the new atlas with the previous mesh's gutters (jagged seams on the model).
        let cov_gen = (self.tex_size, self.topo_version);
        if self.gpu_coverage_gen != Some(cov_gen) {
            self.ensure_coverage();
            if let Some(cov) = self.coverage.borrow().as_ref() {
                self.gpu_layers.set_coverage(&self.queue, cov);
                self.gpu_coverage_gen = Some(cov_gen);
            }
        }
        self.gpu_layers.composite(&self.device, &self.queue);
        self.gpu_layers.bleed(&self.device, &self.queue, self.bleed_pad());
        // Point the model's bind group at the freshly composited atlas.
        if let Some(view) = self.gpu_layers.atlas_srgb_view() {
            self.gpu_bind_group = Some(make_paint_bind_group_view(
                &self.device,
                &self.bind_group_layout,
                &self.uniform_buffer,
                view,
                &self.sampler,
            ));
        }
    }

    /// Apply any pending stroke dirty-rect to the GPU, once per frame. Cheap no-op when
    /// nothing was painted since the last call. Called from the app's redraw and before
    /// headless `capture`.
    pub fn flush_paint(&mut self) {
        // GPU paint: pick up the previous frame's coverage readback and apply it, issue this
        // frame's (async, no stall), and grow `dirty_rect`. No-op on the CPU path.
        self.pump_gpu_stroke();
        if let Some(rect) = self.dirty_rect.take() {
            if self.gpu_display {
                // A 2D UV-editor stroke paints into the very panel it's drawn on (the CPU
                // `display_buf` mirror), so keep that path live each frame — it's a cheap flat
                // disc, and `refresh_display_region` bumps `paint_version` so the panel
                // re-uploads. (Without this the panel wouldn't refresh until `end_stroke`,
                // looking like the stroke never lands.)
                if self.uv_stroke {
                    self.refresh_display_region(rect);
                }
                // 3D-model stroke (readback path: mask / decal stamp / effected layer): the model
                // samples the atlas, so the CPU composite/quantize/bleed/upload would be wasted
                // work — mirror only the changed active layer and recomposite on the GPU. (The 2D
                // panel re-syncs once at `end_stroke`.) For a UV stroke this also keeps the model
                // in step with the panel.
                self.gpu_layers
                    .upload_active(&self.device, &self.queue, &self.layers);
                self.gpu_compose();
            } else {
                self.refresh_display_region(rect);
            }
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
        // A discrete edit can change the stack structure, so snapshot the whole stack.
        self.history
            .record(crate::history::Snapshot::Stack(self.layers.clone()));
        self.mark_edited();
    }

    /// One-time, opt-in seam cleanup for art painted *before* the conservative-dab fix: fill the
    /// dark under-coverage "teeth" at island edges from neighbouring paint of the **same UV facet**,
    /// only on the island rim. Region-aware (`fill_map.texel_facet`) so it can't smear an adjacent
    /// island's colour across a seam, and rim-bounded so interior transparency is untouched — see
    /// `bleed::fill_island_rim_teeth`. One undo step; a no-op on fully-opaque layers. New painting no
    /// longer needs this (the GPU dab is conservative); it just repairs old files without a repaint.
    pub fn clean_seams(&mut self) {
        self.checkpoint();
        self.ensure_fill_map();
        let size = self.tex_size;
        let pad = self.bleed_pad();
        {
            let tf = &self.fill_map.as_ref().unwrap().texel_facet;
            for layer in &mut self.layers.layers {
                crate::bleed::fill_island_rim_teeth(&mut layer.tex.pixels, tf, size, pad);
            }
        }
        self.refresh_display_texture();
    }

    /// Step back to the previous layer state, if any.
    pub fn undo(&mut self) {
        if self.history.undo(&mut self.layers) {
            self.restore_after_history();
            self.mark_edited();
        }
    }

    /// Step forward to the next layer state, if any.
    pub fn redo(&mut self) {
        if self.history.redo(&mut self.layers) {
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
            self.resync_stroke_base();
            self.refresh_display_texture();
        }
    }

    /// Invalidate the lazily-captured stroke base + coverage without touching the
    /// buffers: bump the stroke generation so every texel's stamp now reads as stale
    /// and is re-captured from the live pixels on its next touch. O(1) — the cheap
    /// replacement for cloning the whole paint target each time the active layer,
    /// paint target, or pixels change outside a stroke.
    fn resync_stroke_base(&mut self) {
        self.bump_stroke_id();
    }

    /// Advance to a fresh stroke generation. On the rare u32 wrap, clear the stamps so
    /// a stale stamp can't masquerade as the new (wrapped-to-1) generation.
    fn bump_stroke_id(&mut self) {
        self.stroke_id = self.stroke_id.wrapping_add(1);
        if self.stroke_id == 0 {
            self.stroke_stamp.fill(0);
            self.stroke_id = 1;
        }
    }

    /// Resize the stroke buffers to the current texture and start a fresh generation.
    /// Called after the paint texture is (re)created at a new resolution. The stamps
    /// reset to 0 (≠ the id of 1), so base/coverage contents are treated as empty.
    fn realloc_stroke_buffers(&mut self) {
        let texels = (self.tex_size * self.tex_size) as usize;
        self.stroke_base.resize(texels * 4, 0);
        self.stroke_coverage.resize(texels, 0.0);
        self.stroke_stamp.clear();
        self.stroke_stamp.resize(texels, 0);
        self.stroke_id = 1;
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
        crate::perf::time("preview_splat", || {
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
        })
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
            // Under GPU display the model samples the atlas, so the ghost also lives there —
            // revert that region too. `display_buf` mirrors the committed atlas (kept in sync
            // by every refresh / `end_stroke`), so it's the correct restore source. During a
            // stroke the per-frame `gpu_compose` rebuilds the whole atlas anyway, so this is at
            // worst redundant, never wrong.
            if self.gpu_display {
                if let Some(atlas) = self.gpu_layers.atlas_texture() {
                    upload_region(&self.queue, atlas, &self.display_buf, size, rect);
                }
            }
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

        // Bound the ghost: the per-texel preview re-runs the CPU `splat` on every cursor
        // move, so for a large brush it costs as much as a real dab (tens of ms/frame —
        // the dominant idle-frame lag). Flood the *cheap* face set first and skip the ghost
        // when its UV footprint is large; the cursor ring still shows the size.
        let radius_world = self.world_brush_radius(hit.tri, brush.radius);
        let faces = {
            let adj = self.surface_adj.as_ref().unwrap();
            crate::surface::splat_faces(&self.mesh, adj, &hit, radius_world, &mut self.splat_scratch)
        };
        if self
            .faces_uv_rect(&faces)
            .is_some_and(|r| r.width() as usize * r.height() as usize > PREVIEW_MAX_TEXELS)
        {
            return; // large brush — ring only, no ghost (the previous ghost is already reverted)
        }

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
            // Coverage comes from the alpha tip (its shape) when one is loaded, else the
            // circular falloff — matching `surface_dab`. The colour laid is the same either
            // way (tiled material or flat swatch), so only the coverage source differs.
            let mut s = if self.alpha_tip_active() {
                let mut s = self.alpha_tip_splats(&hit, brush);
                if let Some(m) = &mirror {
                    s.extend(self.alpha_tip_splats(m, brush));
                }
                s
            } else {
                let mut s = self.dab_splats(&hit, brush);
                if let Some(m) = &mirror {
                    s.extend(self.dab_splats(m, brush));
                }
                s
            };
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
        // Mirror the ghost into the GPU atlas so it shows on the model under GPU display (the
        // model samples the atlas, not `paint_texture_gpu`). Only reached while hovering
        // (`pending` is `None` here — a mid-stroke ghost already returned above), so this never
        // races the in-stroke compose.
        if self.gpu_display {
            if let Some(atlas) = self.gpu_layers.atlas_texture() {
                upload_packed(&self.queue, atlas, &packed, rect);
            }
        }
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

    /// Load a grayscale alpha tip that shapes the brush in place of the circle.
    pub fn load_brush_alpha(&mut self, path: &str) -> Result<(), String> {
        self.brush_alpha = Some(crate::material::Material::load(path)?);
        Ok(())
    }

    /// Drop the loaded alpha tip; the brush reverts to the built-in circular falloff.
    pub fn clear_brush_alpha(&mut self) {
        self.brush_alpha = None;
    }

    /// Invert the alpha tip (treat dark instead of light pixels as paint), for
    /// black-on-white tip packs.
    pub fn set_brush_alpha_invert(&mut self, invert: bool) {
        self.brush_alpha_invert = invert;
    }

    /// True when an alpha tip is loaded — the surface dab then shapes coverage from the
    /// tip image instead of the radial falloff.
    fn alpha_tip_active(&self) -> bool {
        self.brush_alpha.is_some()
    }

    /// A small antialiased preview (≤`max`×`max`, RGBA8) of the loaded alpha tip, for the
    /// UI swatch; `None` when no tip is loaded.
    pub fn brush_alpha_thumbnail(&self, max: u32) -> Option<(u32, u32, Vec<u8>)> {
        self.brush_alpha.as_ref().map(|m| m.thumbnail(max))
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

    /// The texels one alpha-tip dab at `hit` would cover, as `(texel, coverage)`. Like
    /// `decal_splats` it floods the surface to the dab's square reach and projects each
    /// texel into the oriented tangent frame (`decal_frame`: world-up aligned, rotated by
    /// `stamp_angle`), but the sample drives *coverage* — the caller's `deposit` then lays
    /// the brush colour (or erases / tiles a material) in that shape. Coverage is the tip
    /// image's brightness × its alpha (× `brush.opacity`), inverted when `brush_alpha_invert`.
    /// Texels outside the tip's [0,1]² footprint are dropped. `ensure_surface_adj` must run
    /// first.
    fn alpha_tip_splats(&mut self, hit: &Hit, brush: &Brush) -> Vec<(usize, f32)> {
        let Some(tip) = self.brush_alpha.as_ref() else {
            return Vec::new();
        };
        // The tip fills the brush-radius footprint: half-side = the world radius, so the
        // square's corners reach radius·√2. Flood that far (plus a hair) then clip to [0,1]².
        let half = self.world_brush_radius(hit.tri, brush.radius);
        if half <= 0.0 {
            return Vec::new();
        }
        let reach = half * std::f32::consts::SQRT_2 * 1.02;
        let (center, t, b) = self.decal_frame(hit);
        let inv = 1.0 / (2.0 * half);
        let invert = self.brush_alpha_invert;
        let opacity = brush.opacity;

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
            let u = d.dot(t) * inv + 0.5;
            let v = d.dot(b) * inv + 0.5;
            if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
                continue;
            }
            let c = tip.sample(u, v, 1.0); // tile=1: one placement, no repeat
            // Perceptual luma (grayscale tips have r=g=b, so this is just their value),
            // gated by the image's own alpha so cutout PNGs work too.
            let lum = (0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32) / 255.0;
            let lum = if invert { 1.0 - lum } else { lum };
            let a = opacity * lum * (c[3] as f32 / 255.0);
            if a > 0.0 {
                out.push((texel, a));
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
        self.realloc_stroke_buffers();
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
        // Invalidate the stroke base since the active layer changed wholesale.
        self.resync_stroke_base();
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

    /// Bake the mesh maps for the current resolution if not already cached. The CPU
    /// rasterizes the geometry maps (cheap, BVH-free); the GPU ray-traces AO (the heavy
    /// part, which scales to 2K/4K), then uploads the residency the sun bake reuses.
    fn ensure_mesh_maps(&mut self) {
        let stale = self
            .mesh_maps
            .as_ref()
            .is_none_or(|m| m.size != self.tex_size);
        if stale {
            log::info!("baking mesh maps at {}²…", self.tex_size);
            if force_cpu_bake() {
                // CPU reference path (LOWTEX_CPU_BAKE): the full rayon bake, no GPU — the
                // oracle the GPU bake is diffed against, and a field escape hatch.
                self.mesh_maps = Some(crate::bake::bake(&self.mesh, &self.bvh, self.tex_size));
                return;
            }
            let mut maps = crate::bake::bake_geometry(&self.mesh, self.tex_size);
            // Upload the BVH + rasterized pos/nrm/mask, then ray-trace AO on the GPU.
            // Disjoint field borrows: &mut gpu_baker with &device/&bvh, then a read of
            // the just-built `maps` channels.
            self.gpu_baker.upload(
                &self.device,
                &self.bvh,
                &maps.pos,
                &maps.nrm,
                &maps.mask,
                maps.diag,
            );
            maps.ao = self.gpu_baker.ao(&self.device, &self.queue);
            self.mesh_maps = Some(maps);
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
            if force_cpu_bake() {
                // CPU reference path (LOWTEX_CPU_BAKE): no GPU residency was uploaded, so
                // recompute the sun on the CPU. Disjoint borrows: &self.bvh, &mut maps.
                let bvh = &self.bvh;
                if let Some(m) = self.mesh_maps.as_mut() {
                    m.compute_light(bvh, dir, shadow);
                }
                return;
            }
            // GPU sun bake against the residency `ensure_mesh_maps` just uploaded — a
            // params write + dispatch + readback, so dragging the sun re-bakes live even
            // at 2K. The CPU `compute_light` stays the parity oracle (gpu_bake tests).
            let light = self.gpu_baker.sun(&self.device, &self.queue, dir, shadow);
            if let Some(m) = self.mesh_maps.as_mut() {
                m.light = light;
                m.light_params = Some((dir, shadow));
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
    /// atlas size and re-maps onto the new UVs. With `overlap`, congruent charts are
    /// stacked onto shared atlas slots (identical/mirrored parts share texels). Returns
    /// `(atlas_size, clamped, density)` where `clamped` means density was reduced to
    /// stay within the GPU texture limit and `density` is the achieved texels-per-
    /// world-unit (here derived to fill the atlas).
    pub fn apply_unwrap(
        &mut self,
        density: crate::unwrap::Density,
        overlap: bool,
    ) -> (u32, bool, f32) {
        self.apply_unwrap_opts(crate::unwrap::UnwrapOptions {
            density,
            max_atlas: self.max_texture_dim,
            overlap_identical: overlap,
            ..Default::default()
        })
    }

    /// Re-unwrap at an *exact* texels-per-world-unit instead of a preset density: the
    /// atlas is sized to hold the charts at `texels_per_unit` (rounded up to a power of
    /// two) rather than stretched to fill it, so e.g. 128 means 128 texels span one
    /// world unit everywhere. Returns `(atlas_size, clamped, density)`; `density`
    /// equals the request unless it overflowed the GPU limit (then `clamped` is set
    /// and it's trimmed). See `apply_unwrap`.
    pub fn apply_unwrap_at_density(&mut self, texels_per_unit: f32, overlap: bool) -> (u32, bool, f32) {
        self.apply_unwrap_opts(crate::unwrap::UnwrapOptions {
            max_atlas: self.max_texture_dim,
            overlap_identical: overlap,
            target_density: Some(texels_per_unit),
            ..Default::default()
        })
    }

    /// Shared body of the unwrap entry points: re-unwrap with `opts`, resize the paint
    /// layers to the new atlas, and rebuild the GPU buffers / BVH / cached maps.
    fn apply_unwrap_opts(&mut self, opts: crate::unwrap::UnwrapOptions) -> (u32, bool, f32) {
        self.checkpoint();
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
        (result.atlas_size, result.clamped, result.density_d)
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
            source_transform: Some(self.mesh.source_transform),
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
            source_transform: doc.source_transform.unwrap_or_default(),
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

    /// Begin a new stroke: open a fresh stroke generation, so overlap within the
    /// stroke accumulates by max-coverage (no double-darken) while each touched
    /// texel's pre-stroke value is captured lazily on first contact. O(1): no
    /// full-texture clone or zero-fill (see the `stroke_*` fields).
    pub fn begin_stroke(&mut self) {
        self.bump_stroke_id();
        // A fresh stroke hasn't fixed its locked face yet (the first dab will).
        self.stroke_lock_facets = None;
        // Snapshot the pre-stroke stack so the whole stroke is one undo step. Held
        // until release, then committed only if the stroke actually painted.
        self.pending = Some(crate::perf::time("stroke_snapshot", || {
            let a = self.layers.active;
            StrokeBackup {
                active: a,
                tex: self.layers.layers[a].tex.clone(),
                mask: self.layers.layers[a].mask.clone(),
            }
        }));
        self.stroke_dirty = false;
        // The hover ghost must be reverted now the stroke is taking over; force the
        // preview path to run (and clear it) next frame rather than skip on a match.
        self.last_preview = None;
        // GPU paint: open a fresh per-stroke coverage target. Always done when the flag is
        // on (a cheap clear) — even if this stroke turns out to be a CPU-path variant, the
        // unused target is harmless.
        if self.gpu_paint {
            self.gpu_dab
                .begin_stroke(&self.device, &self.queue, self.tex_size);
            // Fully-GPU display: make sure the active layer's slice is current (only it can
            // change during a stroke), then snapshot it as the resolve base. Non-active
            // slices stay current from the last display refresh, so no full re-upload.
            if self.gpu_display {
                self.gpu_layers
                    .upload_active(&self.device, &self.queue, &self.layers);
                self.gpu_layers
                    .begin_stroke_resolve(&self.device, &self.queue, self.layers.active as u32);
                // Mirror the brush material to the GPU for the material resolve path, but
                // only when it changed (keyed on the brush's material generation).
                if let Some(mat) = self.brush_material.as_ref() {
                    if self.gpu_material_gen != Some(self.brush_material_gen) {
                        self.gpu_layers.set_material(&self.device, &self.queue, mat);
                        self.gpu_material_gen = Some(self.brush_material_gen);
                    }
                }
            }
        }
        self.gpu_stroke_rect = None;
        self.gpu_stroke_resolve = false;
        self.gpu_stroke_material = None;
        self.stroke_paint_rect = None;
        self.uv_stroke = false;
    }

    /// Grow the whole-stroke painted region by one dab's texel footprint. Unlike
    /// `dirty_rect` (consumed every frame by `flush_paint`), this survives until
    /// `end_stroke` so the CPU display mirror can be re-synced over exactly the painted area.
    fn grow_stroke_paint_rect(&mut self, r: TexRect) {
        self.stroke_paint_rect = Some(match self.stroke_paint_rect {
            Some(prev) => prev.union(r),
            None => r,
        });
    }

    /// End the current stroke, committing it to history as a single undo step.
    /// A stroke that never hit the mesh (`stroke_dirty` false) records nothing,
    /// so an errant click off the model doesn't leave an empty undo entry.
    pub fn end_stroke(&mut self) {
        // Drain the pipelined GPU coverage into the active layer first — it reads the
        // pre-stroke base from `pending`, which the commit below consumes — so the in-flight
        // and just-stamped regions aren't lost. One blocking sync, on release only.
        self.finish_gpu_stroke();
        if let Some(before) = self.pending.take() {
            if self.stroke_dirty {
                self.history.record(crate::history::Snapshot::Layer {
                    index: before.active,
                    tex: before.tex,
                    mask: before.mask,
                });
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
        // Under GPU display the in-stroke paths recomposited the GPU atlas (which the model
        // samples) but left the CPU display mirror stale, since the per-frame CPU
        // composite/upload would be wasted work off the model's path. The 2D UV editor, brush
        // preview and export still read that mirror, so re-sync it here over just the painted
        // region from the now-reconciled CPU layers — brush-area bounded, once per stroke, off
        // the paint hot path. `refresh_display_region` also bumps `paint_version`, which is how
        // the UV editor learns to re-upload. (The CPU path keeps the mirror live each frame in
        // `flush_paint`, so this is a no-op there.)
        let painted = self.stroke_paint_rect.take();
        if self.gpu_display {
            if let Some(rect) = painted {
                self.refresh_display_region(rect);
            }
        }
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
        // Alpha tip: shape this dab's coverage from the loaded tip image instead of the
        // circular falloff, then deposit the brush colour (or erase / tiled material) in
        // that shape. Works for colour and mask targets and for the eraser, and takes
        // precedence over the circular GPU/CPU paths below.
        if self.alpha_tip_active() {
            self.ensure_surface_adj();
            let mut splats = self.alpha_tip_splats(hit, brush);
            self.retain_locked(&mut splats);
            if splats.is_empty() {
                return false;
            }
            self.deposit(&splats, brush);
            return true;
        }
        let size = self.tex_size;
        let radius_world = self.world_brush_radius(hit.tri, brush.radius);
        // GPU dab path: stamp the face set into the GPU coverage target instead of the
        // CPU splat + deposit. Eligible only for the plain solid-colour surface brush
        // (see `gpu_paint_eligible`); everything else falls through to the CPU path.
        if self.gpu_paint_eligible(brush) {
            return self.gpu_surface_dab(hit, radius_world, brush);
        }
        // Distinct-field borrows (mesh/surface_adj immutable, splat_scratch mutable)
        // in one call — the reused flood scratch keeps the stroke allocation-free.
        let adj = self.surface_adj.as_ref().unwrap();
        let mut splats = crate::perf::time("splat", || {
            crate::surface::splat(
                &self.mesh,
                adj,
                hit,
                radius_world,
                brush.opacity,
                brush.hardness,
                size,
                &mut self.splat_scratch,
            )
        });
        // Face lock: drop every texel that isn't on the stroke's locked face, so
        // the dab stops at the face's edges instead of wrapping onto its neighbours.
        self.retain_locked(&mut splats);
        if splats.is_empty() {
            return false; // hit the mesh, but the dab covered no texels
        }
        crate::perf::time("deposit", || self.deposit(&splats, brush));
        true
    }

    /// Whether this dab takes the GPU stamping path (Phase 1). Covered: the solid-colour
    /// surface brush, the eraser, the *tiled image* brush, and mask painting (reveal/hide)
    /// — all reduce to one coverage map that `resolve_gpu_stroke` applies (blend the solid
    /// colour or the per-texel material sample, subtract alpha, or blend white/black into
    /// the mask). Only the oriented decal *stamp* keeps its CPU `deposit_rgb` path, which a
    /// single-channel coverage map can't model (it needs a per-texel projected colour).
    fn gpu_paint_eligible(&self, _brush: &Brush) -> bool {
        // An alpha tip shapes coverage on the CPU (like the decal stamp), so it rides the
        // readback path — never the GPU circle resolve. `surface_dab` already returns before
        // reaching here when a tip is active; this keeps the intent explicit and safe.
        self.gpu_paint
            && !self.alpha_tip_active()
            && match self.paint_target {
                // Colour: anything but the oriented decal stamp (the tiled material brush
                // is fine — its colour is sampled per texel at resolve). The decal-paint
                // case has already returned via `surface_dab`'s stamp branch above.
                PaintTarget::Color => !self.stamp_active(),
                // Mask: reveal/hide ignores any brush image, so always eligible.
                PaintTarget::Mask => true,
            }
    }

    /// Stamp one solid-colour dab on the GPU: flood the face set (`splat_faces`), confine
    /// it to the locked facet(s), and `Max`-accumulate its coverage into the per-stroke GPU
    /// target. No CPU per-texel work and no readback here — the coverage is resolved into
    /// the layer once per frame by `resolve_gpu_stroke`. Grows `gpu_stroke_rect` by the
    /// faces' UV footprint (the region to resolve). Returns whether it stamped anything.
    fn gpu_surface_dab(&mut self, hit: &Hit, radius_world: f32, brush: &Brush) -> bool {
        self.ensure_surface_adj();
        let mut faces = {
            let adj = self.surface_adj.as_ref().unwrap();
            crate::surface::splat_faces(
                &self.mesh,
                adj,
                hit,
                radius_world,
                &mut self.splat_scratch,
            )
        };
        self.retain_locked_faces(&mut faces);
        let Some(rect) = self.faces_uv_rect(&faces) else {
            return false;
        };
        crate::perf::time("gpu_stamp", || {
            self.gpu_dab.stamp(
                &self.mesh,
                &faces,
                hit.pos,
                radius_world,
                brush.opacity,
                brush.hardness,
                self.tex_size,
            );
        });
        // Capture how this dab resolves: paint the solid colour / erase (colour target), or
        // blend white(reveal)/black(hide) into the mask (mask target).
        self.gpu_stroke_target = self.paint_target;
        match self.paint_target {
            PaintTarget::Color => {
                self.gpu_stroke_color = brush.color_u8();
                self.gpu_stroke_erase = brush.erase;
            }
            PaintTarget::Mask => {
                self.gpu_stroke_color = if self.mask_reveal { [255; 3] } else { [0; 3] };
                self.gpu_stroke_erase = false;
            }
        }
        self.gpu_stroke_rect = Some(match self.gpu_stroke_rect {
            Some(prev) => prev.union(rect),
            None => rect,
        });
        self.grow_stroke_paint_rect(rect);
        // Take the no-readback resolve path for the solid-colour brush, the eraser, AND the
        // tiled material brush under GPU display: the resolve shader samples the material per
        // texel (resolve.wgsl mode 2), so it needs no GPU→CPU readback or CPU per-texel blend —
        // the win for the material brush from the lag report. The mask target and the oriented
        // decal stamp keep the readback path (a single coverage channel can't carry the mask's
        // separate target or the stamp's per-texel projected colour). Excluded too when the
        // active layer carries effects: the GPU slice holds its *effected* pixels, so resolving
        // + reconciling into the stored pixels would bake the (non-destructive) effect in.
        // (The deferred-bleed regression that first forced material onto the readback path is
        // gone — the resolve path now does a full compose+bleed every frame, like the readback
        // path, so seam gutters stay filled live.)
        let active_clean = self.layers.layers[self.layers.active]
            .effects
            .iter()
            .all(crate::effects::Effect::is_identity);
        self.gpu_stroke_resolve = self.gpu_display
            && self.paint_target == PaintTarget::Color
            && !self.stamp_active()
            && active_clean;
        // Carry the tile factor when this resolve stroke paints a material (the shader keys
        // `ResolveKind::Material` off it); `None` falls through to the solid-colour resolve.
        self.gpu_stroke_material = (self.gpu_stroke_resolve && self.brush_material.is_some())
            .then_some(self.brush_tile);
        self.stroke_dirty = true;
        true
    }

    /// Face-lock for the GPU path: drop every face not on one of the stroke's locked
    /// facets, the face-set analogue of `retain_locked` (a facet *is* a set of faces, so
    /// this confines the dab exactly). No-op when the lock is off or unset.
    fn retain_locked_faces(&self, faces: &mut Vec<u32>) {
        if !self.lock_face {
            return;
        }
        let Some(locked) = self.stroke_lock_facets.as_deref() else {
            return;
        };
        if locked.is_empty() {
            return; // the first dab resolved no facet — paint freely
        }
        let Some(map) = self.fill_map.as_ref() else {
            return;
        };
        faces.retain(|&f| {
            map.facet_for_tri(f)
                .is_some_and(|fc| locked.contains(&(fc as i32)))
        });
    }

    /// The texel bounding box of a face set's UV footprint, matching the rasterizer's
    /// `floor(min)..ceil(max)` bounds, clamped to the atlas — the region the GPU dab can
    /// touch and the renderer must resolve. `None` if empty/degenerate.
    fn faces_uv_rect(&self, faces: &[u32]) -> Option<TexRect> {
        if faces.is_empty() {
            return None;
        }
        let size = self.tex_size as f32;
        let (mut minx, mut miny, mut maxx, mut maxy) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for &f in faces {
            let (_, uv) = crate::surface::tri_data(&self.mesh, f);
            for c in uv {
                minx = minx.min(c.x);
                maxx = maxx.max(c.x);
                miny = miny.min(c.y);
                maxy = maxy.max(c.y);
            }
        }
        let x0 = (minx * size).floor().max(0.0) as u32;
        let y0 = (miny * size).floor().max(0.0) as u32;
        let x1 = ((maxx * size).ceil().max(0.0) as u32).min(self.tex_size);
        let y1 = ((maxy * size).ceil().max(0.0) as u32).min(self.tex_size);
        (x1 > x0 && y1 > y0).then_some(TexRect { x0, y0, x1, y1 })
    }

    /// Blend the GPU stroke's accumulated coverage (since the last resolve) into the active
    /// layer, then reset the per-frame rect. The coverage texture holds the per-stroke max
    /// coverage; `pending` holds the pre-stroke pixels (the immutable base), so every
    /// resolve re-blends `base·(1-cov) + colour·cov` for the dirty region — idempotent and
    /// re-entrant across frames. Disjoint field borrows: `base` from `pending`, target from
    /// `layers`. No-op on the CPU path (`gpu_stroke_rect` stays `None`).
    /// Pipelined GPU-stroke resolve (per frame): pick up the *previous* frame's coverage
    /// readback (already finished on the GPU, so no stall) and apply it, then submit an
    /// asynchronous readback for this frame's freshly-stamped region. One frame of latency,
    /// but the CPU never blocks on the GPU mid-stroke — the fix for the `poll(Wait)` stall
    /// that dominated paint cost. A new readback is only issued once the prior one is
    /// consumed, so `gpu_stroke_rect` accumulates while a copy is in flight (no region lost).
    fn pump_gpu_stroke(&mut self) {
        // Rasterize this frame's accumulated dabs in one submit before reading anything back.
        self.gpu_dab.flush_dabs(&self.device, &self.queue);
        // No-readback path: resolve coverage straight into the active GPU slice and
        // recomposite — no GPU→CPU stall, no CPU per-texel blend.
        if self.gpu_stroke_resolve {
            self.resolve_gpu_stroke_frame();
            return;
        }
        if let Some((cov, rect)) =
            crate::perf::time("gpu_resolve", || self.gpu_dab.try_take_readback(&self.device))
        {
            crate::perf::time("gpu_apply", || self.apply_gpu_coverage(&cov, rect));
        }
        if !self.gpu_dab.has_pending() {
            if let Some(rect) = self.gpu_stroke_rect.take() {
                self.gpu_dab.issue_readback(&self.device, &self.queue, rect);
            }
        }
    }

    /// One frame of the no-readback resolve stroke: blend the accumulated coverage into the
    /// active layer's GPU slice over the immutable pre-stroke base (idempotent, so
    /// re-resolving the whole accumulated rect each frame is correct), then recomposite the
    /// display. `gpu_stroke_rect` keeps accumulating (no `take`) — there's no readback to
    /// bound. The CPU `Layers` is left stale until `end_stroke` reconciles it.
    /// How this stroke resolves coverage onto the base: erase, tiled material, or solid colour.
    fn gpu_resolve_kind(&self) -> crate::gpu_layers::ResolveKind {
        use crate::gpu_layers::ResolveKind;
        if self.gpu_stroke_erase {
            ResolveKind::Erase
        } else if let Some(tile) = self.gpu_stroke_material {
            ResolveKind::Material(tile)
        } else {
            ResolveKind::Color([
                self.gpu_stroke_color[0] as f32 / 255.0,
                self.gpu_stroke_color[1] as f32 / 255.0,
                self.gpu_stroke_color[2] as f32 / 255.0,
            ])
        }
    }

    fn resolve_gpu_stroke_frame(&mut self) {
        let active = self.layers.active as u32;
        let kind = self.gpu_resolve_kind();
        if let (Some(cov), Some(rect)) = (self.gpu_dab.coverage_view(), self.gpu_stroke_rect) {
            let sc = Some((rect.x0, rect.y0, rect.width(), rect.height()));
            crate::perf::time("gpu_resolve", || {
                self.gpu_layers
                    .resolve_active(&self.device, &self.queue, cov, active, kind, sc);
            });
        }
        // Full composite + gutter bleed every frame, so UV-seam gutters stay filled live (a
        // deferred bleed left visible un-painted edges at seams mid-stroke). Matches the
        // readback path's per-frame compose; cheap enough at PSX/2K, and the headline win
        // (no GPU→CPU readback, no CPU per-texel blend) is already in the resolve itself.
        self.gpu_compose();
    }

    /// Flush the GPU stroke synchronously at stroke end: drain the in-flight readback, then
    /// issue + drain the remaining accumulated region, applying both. Must run before the
    /// history commit consumes `pending` (the base). One blocking sync, on release only.
    fn finish_gpu_stroke(&mut self) {
        // Flush any dabs stamped since the last frame's pump, so the final reconcile sees them.
        self.gpu_dab.flush_dabs(&self.device, &self.queue);
        if self.gpu_stroke_resolve {
            self.finish_gpu_resolve_stroke();
            return;
        }
        if let Some((cov, rect)) = self.gpu_dab.drain_readback(&self.device) {
            self.apply_gpu_coverage(&cov, rect);
        }
        if let Some(rect) = self.gpu_stroke_rect.take() {
            self.gpu_dab.issue_readback(&self.device, &self.queue, rect);
            if let Some((cov, rect)) = self.gpu_dab.drain_readback(&self.device) {
                self.apply_gpu_coverage(&cov, rect);
            }
        }
    }

    /// Finish a no-readback resolve stroke: do the final resolve + recomposite, then read
    /// the active GPU slice back **once** into the authoritative CPU `Layers` so undo, save and
    /// export see the painted result. A single blocking readback, on release only — off the
    /// paint hot path. `end_stroke` then re-syncs the CPU display mirror (`display_buf` +
    /// `paint_texture_gpu`) over the painted region for the 2D UV editor / brush preview.
    fn finish_gpu_resolve_stroke(&mut self) {
        // Final resolve of the accumulated coverage, then a single full composite + gutter
        // bleed so the displayed atlas is fully correct (seams bled) for the finished stroke.
        let active_u = self.layers.active as u32;
        let kind = self.gpu_resolve_kind();
        if let (Some(cov), Some(rect)) = (self.gpu_dab.coverage_view(), self.gpu_stroke_rect) {
            let sc = Some((rect.x0, rect.y0, rect.width(), rect.height()));
            self.gpu_layers
                .resolve_active(&self.device, &self.queue, cov, active_u, kind, sc);
        }
        self.gpu_compose();
        // Reconcile only the painted region back into the authoritative CPU layer (not the
        // whole slice) — the readback then scales with brush area, not atlas size.
        let active = self.layers.active;
        let size = self.tex_size as usize;
        if let Some(rect) = self.gpu_stroke_rect {
            if let Some(px) =
                self.gpu_layers
                    .read_layer_rect(&self.device, &self.queue, active as u32, rect)
            {
                let (rw, rh) = (rect.width() as usize, rect.height() as usize);
                let tex = self.layers.active_tex_mut();
                for ry in 0..rh {
                    let gy = rect.y0 as usize + ry;
                    let src = ry * rw * 4;
                    let dst = (gy * size + rect.x0 as usize) * 4;
                    tex.pixels[dst..dst + rw * 4].copy_from_slice(&px[src..src + rw * 4]);
                }
            }
        }
        self.gpu_stroke_rect = None;
        // The model already samples the GPU atlas (correct), and export composites fresh from
        // the now-reconciled CPU layers — so no full CPU recomposite here (that was a per-stroke,
        // resolution-scaled hitch). `end_stroke` re-syncs the CPU `display_buf` mirror over just
        // the painted region for the 2D UV editor / brush preview (brush-area bounded).
    }

    /// Blend a coverage region (from a GPU readback) into the active layer/mask against the
    /// immutable pre-stroke base in `pending`. Idempotent (re-applying a region from the
    /// base is a no-op-after-first), so the 1-frame-late, possibly-overlapping regions a
    /// pipelined readback produces compose correctly.
    fn apply_gpu_coverage(&mut self, cov: &[f32], rect: TexRect) {
        if self.pending.is_none() {
            return;
        }
        let color = self.gpu_stroke_color;
        let erase = self.gpu_stroke_erase;
        let size = self.tex_size;
        // Tiled image brush: the colour is the material sampled per texel (UV-anchored,
        // tiled), matching the CPU `deposit`. Only for colour paint (not erase / mask).
        // Disjoint field borrow (`brush_material`) vs. the `pending`/`layers` borrows below.
        let material = (self.gpu_stroke_target == PaintTarget::Color && !erase)
            .then(|| self.brush_material.as_ref().map(|m| (m, self.brush_tile)))
            .flatten();
        match self.gpu_stroke_target {
            PaintTarget::Color => {
                let base = &self.pending.as_ref().unwrap().tex.pixels;
                let tex = self.layers.active_tex_mut();
                apply_coverage(&mut tex.pixels, base, cov, rect, size, color, erase, material);
            }
            PaintTarget::Mask => {
                let base = &self.pending.as_ref().unwrap().mask.pixels;
                let tex = self.layers.active_mask_mut();
                apply_coverage(&mut tex.pixels, base, cov, rect, size, color, false, None);
            }
        }
        self.dirty_rect = Some(match self.dirty_rect {
            Some(prev) => prev.union(rect),
            None => rect,
        });
        self.grow_stroke_paint_rect(rect);
        self.stroke_dirty = true;
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
        // Version-stamped per-stroke state: a texel counts toward this stroke only when
        // its stamp matches the live id; the first touch captures its pre-stroke value
        // into `base` (lazy copy-on-write) so the blend re-composites against it.
        let id = self.stroke_id;
        let base = &mut self.stroke_base;
        let stamp = &mut self.stroke_stamp;
        let coverage = &mut self.stroke_coverage;
        // Touched texels can be scattered across the atlas (one cluster per face a
        // surface dab wrapped onto), so accumulate their bounding box rather than
        // assume a fixed footprint around the cursor.
        let (mut x0, mut y0, mut x1, mut y1) = (size, size, 0u32, 0u32);
        // Eraser: lower alpha toward transparent instead of laying color (color target
        // only — a mask's reveal/hide already covers the "remove" case). Max-coverage
        // discipline still holds: more coverage = more erased, monotonic either way.
        let erase = brush.erase && matches!(self.paint_target, PaintTarget::Color);
        let mut mark = |texel: usize, a: f32, src: [u8; 3], pixels: &mut [u8]| {
            let first = stamp[texel] != id;
            let cur = if first { 0.0 } else { coverage[texel] };
            if a <= cur {
                return; // already at least this covered this stroke
            }
            if first {
                // First touch this stroke: snapshot the pre-stroke value before we
                // overwrite it, and adopt this stroke's stamp.
                stamp[texel] = id;
                let i = texel * 4;
                base[i..i + 4].copy_from_slice(&pixels[i..i + 4]);
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
                    mark(texel, a, src, &mut tex.pixels);
                }
            }
            PaintTarget::Mask => {
                // Mask painting ignores the brush color: reveal=white, hide=black.
                let src = if self.mask_reveal { [255; 3] } else { [0; 3] };
                let tex = self.layers.active_mask_mut();
                for &(texel, a) in splats {
                    mark(texel, a, src, &mut tex.pixels);
                }
            }
        }

        if x1 > x0 && y1 > y0 {
            let r = TexRect { x0, y0, x1, y1 };
            self.dirty_rect = Some(match self.dirty_rect {
                Some(prev) => prev.union(r),
                None => r,
            });
            self.grow_stroke_paint_rect(r);
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
        let id = self.stroke_id;
        let base = &mut self.stroke_base;
        let stamp = &mut self.stroke_stamp;
        let coverage = &mut self.stroke_coverage;
        let tex = self.layers.active_tex_mut();
        let (mut x0, mut y0, mut x1, mut y1) = (size, size, 0u32, 0u32);
        for &(texel, (a, rgb)) in splats {
            let first = stamp[texel] != id;
            let cur = if first { 0.0 } else { coverage[texel] };
            if a <= cur {
                continue; // already at least this covered this stroke
            }
            if first {
                stamp[texel] = id;
                let i = texel * 4;
                base[i..i + 4].copy_from_slice(&tex.pixels[i..i + 4]);
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
            self.grow_stroke_paint_rect(r);
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
        // Mark this as a 2D-editor stroke so `flush_paint` keeps the CPU display mirror live
        // (the panel the user is painting on reads it). Set per stamp — `begin_stroke` can't
        // know whether the stroke will be 3D or UV (both share the same begin/end bracket).
        self.uv_stroke = true;
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
        // When GPU display is on, sample the GPU composite atlas instead of the CPU paint
        // texture (falls back to the CPU bind group until the first composite).
        let model_bind = self
            .gpu_bind_group
            .as_ref()
            .filter(|_| self.gpu_display)
            .unwrap_or(&self.bind_group);
        rpass.set_bind_group(0, model_bind, &[]);
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

/// A fresh GPU device/queue for a test (`wgpu::Device` isn't `Clone` in wgpu 22, so the
/// suite can't share one). Routes through the retrying `request_device`, which absorbs
/// the transient OOM that `cargo test`'s many concurrent device creations provoke.
#[cfg(test)]
pub(crate) fn new_test_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = pollster::block_on(request_adapter(&instance, None));
    pollster::block_on(request_device(&adapter))
}

/// Whether `LOWTEX_CPU_BAKE` forces the CPU mesh-map bake over the GPU path (read once).
/// The escape hatch for diagnosing GPU/CPU divergence and the reason the CPU `bake` /
/// `compute_light` stay reachable in release, not just from the parity tests.
fn force_cpu_bake() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("LOWTEX_CPU_BAKE").is_some_and(|v| !v.is_empty()))
}

/// Whether `LOWTEX_GPU_PAINT` opts the core surface brush into GPU dab stamping (read
/// once, the renderer's `gpu_paint` default). Off by default — the CPU `splat`+`deposit`
/// path stays the shipping default until the GPU path is proven on more hardware.
fn force_gpu_paint() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("LOWTEX_GPU_PAINT").is_some_and(|v| !v.is_empty()))
}

/// Whether `LOWTEX_GPU_DISPLAY` routes the model's texture through the GPU composite
/// (composite + palette quantize + gutter bleed on the GPU; see `gpu_layers`). Off by
/// default — the CPU `composite_display` stays the shipping default + parity oracle.
fn force_gpu_display() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("LOWTEX_GPU_DISPLAY").is_some_and(|v| !v.is_empty()))
}

async fn request_device(adapter: &wgpu::Adapter) -> (wgpu::Device, wgpu::Queue) {
    let desc = wgpu::DeviceDescriptor {
        label: Some("lowtex device"),
        required_features: wgpu::Features::empty(),
        // Use the adapter's real limits (downlevel_defaults caps textures
        // at 2048, which a HiDPI window surface can exceed).
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
    };
    // Retry on transient OutOfMemory. `cargo test` fans the suite across threads, each
    // creating its own device; past a dozen-or-so coexisting the driver briefly refuses
    // one. Siblings finish and free theirs, so a short backoff lets the queued device
    // through. In production exactly one device is created and the first attempt succeeds.
    let mut last_err = None;
    for attempt in 0u32..40 {
        match adapter.request_device(&desc, None).await {
            Ok(dq) => return dq,
            Err(e) => {
                last_err = Some(e);
                let ms = 25 * (attempt + 1).min(8) as u64;
                std::thread::sleep(std::time::Duration::from_millis(ms));
            }
        }
    }
    panic!("failed to request device: {:?}", last_err.unwrap());
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
    make_paint_bind_group_view(device, layout, uniform_buffer, &view, sampler)
}

/// Same as [`make_paint_bind_group`] but from an already-built texture view — used to
/// bind the GPU composite atlas's sRGB view in place of the single paint texture.
fn make_paint_bind_group_view(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
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
                resource: wgpu::BindingResource::TextureView(view),
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

/// Apply a GPU dab `cov` map (row-major within `rect`, one f32 per texel) onto a full
/// `size`-wide RGBA8 buffer, against the immutable `base`: erase lowers alpha, otherwise
/// blend the source colour — the per-texel `material` sample (UV-anchored, tiled) when
/// given, else the solid `color`. The shared inner loop of `resolve_gpu_stroke`'s
/// colour/mask arms, taking plain slices so the colour/mask/base borrows stay disjoint at
/// the call site. This is the only CPU per-texel paint work left on the GPU path, and it
/// runs once per frame over the dirty region — not per dab.
fn apply_coverage(
    pixels: &mut [u8],
    base: &[u8],
    cov: &[f32],
    rect: TexRect,
    size: u32,
    color: [u8; 3],
    erase: bool,
    material: Option<(&crate::material::Material, f32)>,
) {
    use rayon::prelude::*;
    let (rw, rh) = (rect.width() as usize, rect.height() as usize);
    let size_us = size as usize;
    let size_f = size as f32;
    let (x0, y0) = (rect.x0 as usize, rect.y0 as usize);
    // Each row writes disjoint texels, so fan the region across cores — the big-brush
    // resolve was tens of ms single-threaded. Slice `pixels` to the region's row band; each
    // worker gets one full texture row and writes only columns x0..x1, reading the matching
    // row of the (separate) `base`. blend4/erase4 are the exact per-dab `blend_texel` math.
    let band = &mut pixels[y0 * size_us * 4..(y0 + rh) * size_us * 4];
    let apply_row = |ry: usize, row: &mut [u8]| {
        let ty = y0 + ry;
        for rx in 0..rw {
            let a = cov[ry * rw + rx];
            if a <= 0.0 {
                continue;
            }
            let tx = x0 + rx;
            let di = tx * 4;
            let bi = (ty * size_us + tx) * 4;
            if erase {
                crate::paint::erase4(&mut row[di..di + 4], &base[bi..bi + 4], a);
            } else {
                let src = match material {
                    Some((m, tile)) => {
                        let c = m.sample(tx as f32 / size_f, ty as f32 / size_f, tile);
                        [c[0], c[1], c[2]]
                    }
                    None => color,
                };
                crate::paint::blend4(&mut row[di..di + 4], &base[bi..bi + 4], src, a);
            }
        }
    };
    // Small regions stay serial — rayon's fork/join would dominate a sub-128² footprint.
    const PAR_MIN_TEXELS: usize = 1 << 14;
    if rw * rh >= PAR_MIN_TEXELS {
        band.par_chunks_mut(size_us * 4)
            .enumerate()
            .for_each(|(ry, row)| apply_row(ry, row));
    } else {
        band.chunks_mut(size_us * 4)
            .enumerate()
            .for_each(|(ry, row)| apply_row(ry, row));
    }
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
        headless_sized(64, 64)
    }

    /// A headless cube renderer at the given surface size. Its device comes through the
    /// retrying `request_device`, so the suite's concurrent device creations don't OOM.
    fn headless_sized(width: u32, height: u32) -> Renderer {
        pollster::block_on(Renderer::new_headless(width, height, Mesh::cube()))
    }

    /// Wall-clock per stroke (begin → 10 segments, flushing each frame → end) at a paint
    /// resolution, for each path: pure CPU, GPU-paint (readback), GPU-paint+display
    /// (no-readback resolve). Ignored by default — run for numbers:
    ///   LOWTEX_BENCH_SIZE=2048 cargo test --release bench_stroke_paths -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_stroke_paths() {
        use std::time::Instant;
        let size = std::env::var("LOWTEX_BENCH_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024u32);
        let (sw, sh) = (640u32, 640u32);
        let brush = Brush {
            color: [0.85, 0.2, 0.3],
            radius: (size as f32 / 32.0).max(8.0), // ~constant visual size across resolutions
            opacity: 1.0,
            hardness: 0.6,
            ..Brush::default()
        };

        let run = |label: &str, gpu_display: bool, gpu_paint: bool| {
            let mut r = headless_sized(sw, sh);
            r.set_texture_resolution(size);
            r.gpu_display = gpu_display;
            r.gpu_paint = gpu_paint;
            let (cx, cy) = (sw as f32 / 2.0, sh as f32 / 2.0);
            let stroke = |r: &mut Renderer| {
                r.begin_stroke();
                let mut p = (cx - 80.0, cy - 60.0);
                for k in 0..10 {
                    let np = (p.0 + 16.0, p.1 + 12.0 * ((k % 3) as f32 - 1.0));
                    r.paint_segment(p, np, &brush);
                    r.flush_paint();
                    p = np;
                }
                r.end_stroke();
                r.flush_paint();
            };
            stroke(&mut r); // warm
            const ITERS: u32 = 10;
            let t = Instant::now();
            for _ in 0..ITERS {
                stroke(&mut r);
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / ITERS as f64;
            println!("  {label:28} {ms:8.2} ms/stroke");
        };

        println!("\nStroke cost @ {size}² (10 segments, per-frame flush):");
        run("CPU (splat+deposit)", false, false);
        run("GPU paint (readback)", false, true);
        run("GPU paint + display (resolve)", true, true);
        println!();
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

    /// End-to-end GPU display parity: with `gpu_display` on, the composite atlas the
    /// model samples (composite + palette quantize + Bayer dither + gutter bleed, all on
    /// the GPU) must match the CPU `composite_display` on all but a tiny fraction of
    /// texels (palette-on is byte-exact bar rare argmin flips; bleed is exact).
    #[test]
    fn gpu_display_matches_cpu_composite_display() {
        let mut r = headless();
        r.gpu_display = true;
        let size = r.tex_size;

        // A two-layer stack (Multiply on top, partial alpha) over the base.
        r.layers.add_layer();
        r.layers.layers[1].blend = crate::layers::BlendMode::Multiply;
        for (t, px) in r.layers.active_tex_mut().pixels.chunks_exact_mut(4).enumerate() {
            px.copy_from_slice(&[(t % 256) as u8, 100, 255u8.wrapping_sub(t as u8), 200]);
        }
        // Palette + dither on, so the whole quantize+bleed path is exercised.
        r.palette = crate::palette::Palette::builtins().remove(0); // PICO-8
        r.palette_settings = PaletteSettings {
            enabled: true,
            dither: true,
            dither_strength: 0.06,
        };

        r.refresh_display_texture(); // composites on CPU *and* GPU (update_gpu_display)
        let cpu = r.composite_display();
        let gpu = r.gpu_layers.read_atlas(&r.device, &r.queue);
        assert_eq!(gpu.len(), cpu.len());

        let n = (size * size) as usize;
        let mut diff = 0u32;
        for t in 0..n {
            let i = t * 4;
            let d = (0..3)
                .map(|c| (gpu[i + c] as i32 - cpu[i + c] as i32).abs())
                .max()
                .unwrap();
            if d > 1 {
                diff += 1;
            }
        }
        assert!(
            (diff as f32) < 0.02 * n as f32,
            "GPU display diverges from CPU at {diff} of {n} texels"
        );
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
        let mut r = headless_sized(w, h);
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
            let mut r = headless_sized(w, h);
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

    /// A loaded alpha tip shapes the dab from the tip image instead of the circular
    /// falloff: a fully-white tip paints its footprint, and inverting that same tip
    /// (white → no coverage) paints nothing — proving the tip drives coverage and the
    /// `alpha_invert` flag is honoured end to end.
    #[test]
    fn alpha_tip_shapes_and_inverts_the_dab() {
        let (w, h) = (256u32, 256u32);
        let brush = Brush {
            color: [0.1, 0.7, 0.3],
            radius: 8.0,
            ..Brush::default()
        };
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        // A small fully-white, fully-opaque tip: every texel in the footprint gets full
        // coverage (luma 1 × alpha 1).
        let white_tip = || crate::material::Material {
            width: 2,
            height: 2,
            pixels: vec![255u8; 16],
        };

        let painted = |invert: bool| -> usize {
            let mut r = headless_sized(w, h);
            r.brush_alpha = Some(white_tip());
            r.brush_alpha_invert = invert;
            let before = r.layers.active_tex().pixels.clone();
            r.begin_stroke();
            r.paint_at((cx, cy), &brush);
            r.end_stroke();
            before
                .chunks_exact(4)
                .zip(r.layers.active_tex().pixels.chunks_exact(4))
                .filter(|(a, b)| a != b)
                .count()
        };

        assert!(
            painted(false) > 0,
            "a fully-white alpha tip must paint its footprint"
        );
        assert_eq!(
            painted(true),
            0,
            "inverting a fully-white tip drops coverage to zero — nothing should paint"
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
            let mut r = headless_sized(w, h);
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
        let mut r = headless_sized(w, h);

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
        let mut r = headless_sized(w, h);

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
        let mut r = headless_sized(w, h);
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

    /// Under GPU display the model samples the GPU atlas, so the hover ghost must be written
    /// into the atlas (not just the unused CPU `paint_texture_gpu`) — otherwise the preview is
    /// invisible on the model. Hovering must change the atlas in the ghost footprint; pointing
    /// off the mesh must restore it to the committed composite.
    #[test]
    fn brush_preview_ghost_visible_in_atlas_under_gpu_display() {
        let (w, h) = (256u32, 256u32);
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        let brush = Brush {
            color: [0.9, 0.1, 0.1],
            radius: 8.0,
            ..Brush::default()
        };
        let mut r = headless_sized(w, h);
        r.gpu_display = true;
        r.gpu_paint = true;
        r.update_gpu_display(); // build the atlas the model samples
        let committed = r.gpu_layers.read_atlas(&r.device, &r.queue);

        r.set_brush_preview(Some((cx, cy)), &brush);
        assert!(r.preview_rect.is_some(), "hovering the mesh should arm a preview");
        let hovered = r.gpu_layers.read_atlas(&r.device, &r.queue);
        assert_ne!(
            hovered, committed,
            "ghost not written into the atlas — invisible on the model under GPU display"
        );

        // Pointing off the mesh reverts the ghost; the atlas returns to the committed image.
        r.set_brush_preview(Some((1.0, 1.0)), &brush);
        assert!(r.preview_rect.is_none(), "pointing off the mesh clears the preview");
        let reverted = r.gpu_layers.read_atlas(&r.device, &r.queue);
        assert_eq!(
            reverted, committed,
            "reverting the ghost must restore the committed atlas"
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
        let mut r = headless_sized(w, h);
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

    #[test]
    fn gpu_baked_maps_drive_layers_and_interactive_sun() {
        // End-to-end through the renderer's GPU mesh-map path (Phase 2.5): applying an
        // AO layer bakes the geometry maps + GPU AO; a Light layer bakes the GPU sun; and
        // moving the sun re-bakes the light channel live. Exercises the wiring/borrows in
        // `ensure_mesh_maps`/`ensure_light` and the baker's residency reuse — the per-texel
        // value parity itself is pinned by the gpu_bake tests.
        let mut r = headless();

        // AO layer → ensure_mesh_maps → GPU AO bake.
        r.apply_ao_layer(Levels::amount(1.0), None);
        assert_eq!(r.layers().layers.len(), 2, "AO added a layer");
        let maps = r.mesh_maps.as_ref().expect("AO bake populated mesh maps");
        assert_eq!(maps.ao.len(), (r.tex_size * r.tex_size) as usize);

        // Light layer with a near-overhead sun → ensure_light → GPU sun bake. A white
        // Normal layer whose alpha is N·L visibly lights the up-facing faces, so the
        // composite must change.
        let before = r.display_pixels();
        r.set_sun([0.3, 0.9, 0.2], true);
        r.add_map_layer(
            "Sun",
            MapSource::Light,
            Levels::amount(1.0),
            [255, 255, 255],
            crate::layers::BlendMode::Normal,
            None,
        );
        assert_eq!(r.layers().layers.len(), 3, "Light added a layer");
        assert_ne!(before, r.display_pixels(), "the sun light changed the composite");
        let light_a = r.mesh_maps.as_ref().unwrap().light.clone();
        assert!(light_a.iter().any(|&v| v > 0.1), "some faces are lit");
        assert_eq!(
            r.mesh_maps.as_ref().unwrap().light_params,
            Some((glam::Vec3::new(0.3, 0.9, 0.2).normalize(), true)),
        );

        // Drag the sun to the opposite side and re-apply: the GPU sun must re-bake, so
        // the light channel differs (the interactive-slider win).
        r.set_sun([-0.3, 0.9, -0.2], true);
        r.add_map_layer(
            "Sun2",
            MapSource::Light,
            Levels::amount(1.0),
            [255, 255, 255],
            crate::layers::BlendMode::Normal,
            None,
        );
        let light_b = &r.mesh_maps.as_ref().unwrap().light;
        assert_ne!(&light_a, light_b, "moving the sun re-baked the light channel");
    }

    #[test]
    fn gpu_paint_stroke_matches_cpu_stroke() {
        // A solid-colour surface stroke via the GPU dab path (Phase 1) must match the CPU
        // splat+deposit path within a small tolerance: same face flood, same falloff, same
        // max-coverage discipline — only f16 coverage precision and the GPU edge fill rule
        // differ, on a thin rim. Paint the identical stroke both ways; diff the layer.
        let (w, h) = (256u32, 256u32);
        let brush = Brush {
            color: [0.85, 0.2, 0.3],
            radius: 14.0,
            opacity: 1.0,
            hardness: 0.6,
            ..Brush::default()
        };
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);

        let paint = |gpu: bool| -> Vec<u8> {
            let mut r = headless_sized(w, h);
            r.gpu_paint = gpu;
            r.begin_stroke();
            r.paint_at((cx, cy), &brush);
            r.flush_paint(); // mid-stroke per-frame resolve (the realistic app cadence)
            r.paint_segment((cx, cy), (cx + 28.0, cy + 16.0), &brush);
            r.flush_paint();
            r.end_stroke();
            r.flush_paint();
            r.layers.active_tex().pixels.clone()
        };
        let cpu = paint(false);
        let gpu = paint(true);
        let base = headless_sized(w, h).layers.active_tex().pixels.clone();

        // Both paths painted a substantial, comparable region.
        let painted = |buf: &[u8]| {
            buf.chunks_exact(4)
                .zip(base.chunks_exact(4))
                .filter(|(a, b)| a != b)
                .count()
        };
        let (cpu_n, gpu_n) = (painted(&cpu), painted(&gpu));
        assert!(cpu_n > 200, "CPU stroke painted too little: {cpu_n}");
        assert!(
            gpu_n as f32 > 0.7 * cpu_n as f32 && (gpu_n as f32) < 1.4 * cpu_n as f32,
            "GPU painted region differs too much from CPU ({gpu_n} vs {cpu_n})"
        );

        // Per-channel agreement over the union of painted texels.
        let mut sum = 0i64;
        let mut n = 0i64;
        let mut big = 0i64;
        for i in 0..cpu.len() {
            if cpu[i] != base[i] || gpu[i] != base[i] {
                let d = (cpu[i] as i32 - gpu[i] as i32).abs();
                sum += d as i64;
                n += 1;
                if d > 64 {
                    big += 1;
                }
            }
        }
        let mean = sum as f64 / n.max(1) as f64;
        assert!(mean < 6.0, "GPU vs CPU stroke mean channel diff too high: {mean:.2}");
        assert!(
            (big as f64) < 0.04 * n as f64,
            "too many large GPU/CPU channel diffs: {big} of {n}"
        );
    }

    /// The *displayed atlas* after a resolve stroke (what the user sees on the model) must
    /// match the CPU `composite_display`, and a second overlapping stroke must build over the
    /// first (overpainting). Guards the resolve path's live display, which the layer-pixel
    /// parity tests don't cover.
    #[test]
    fn gpu_resolve_display_overpaints() {
        let (w, h) = (256u32, 256u32);
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        let mut r = headless_sized(w, h);
        r.gpu_display = true;
        r.gpu_paint = true;

        let stroke = |r: &mut Renderer, color: [f32; 3], from: (f32, f32), to: (f32, f32)| {
            let brush = Brush {
                color,
                radius: 16.0,
                opacity: 1.0,
                hardness: 0.7,
                ..Brush::default()
            };
            r.begin_stroke();
            r.paint_at(from, &brush);
            r.flush_paint();
            r.paint_segment(from, to, &brush);
            r.flush_paint();
            r.end_stroke();
            r.flush_paint();
        };

        // Stroke 1 (red), then stroke 2 (green) overlapping it.
        stroke(&mut r, [0.9, 0.1, 0.1], (cx - 30.0, cy), (cx + 10.0, cy));
        stroke(&mut r, [0.1, 0.9, 0.1], (cx - 10.0, cy), (cx + 30.0, cy));

        // The active layer must show *both* colours (overpainting kept stroke 1 where stroke
        // 2 didn't cover, and laid stroke 2 on top in the overlap).
        let px = &r.layers.active_tex().pixels;
        let reds = px.chunks_exact(4).filter(|p| p[0] > 150 && p[1] < 100).count();
        let greens = px.chunks_exact(4).filter(|p| p[1] > 150 && p[0] < 100).count();
        assert!(reds > 50, "stroke 1 (red) vanished — no overpainting base: {reds}");
        assert!(greens > 50, "stroke 2 (green) didn't lay down: {greens}");

        // The displayed atlas must equal the CPU composite_display (which includes the gutter
        // bleed) within ±1 / palette flips — i.e. no missing edges at UV seams.
        let size = r.tex_size as usize;
        let atlas = r.gpu_layers.read_atlas(&r.device, &r.queue);
        let cpu = r.composite_display();
        assert_eq!(atlas.len(), cpu.len());
        let mut diff = 0u32;
        for t in 0..size * size {
            let i = t * 4;
            let d = (0..3)
                .map(|c| (atlas[i + c] as i32 - cpu[i + c] as i32).abs())
                .max()
                .unwrap();
            if d > 1 {
                diff += 1;
            }
        }
        assert!(
            (diff as f32) < 0.02 * (size * size) as f32,
            "displayed atlas diverges from CPU composite at {diff} of {} texels",
            size * size
        );
    }

    /// The tiled material brush under GPU display must show a correct atlas — composite +
    /// gutter bleed — matching the CPU `composite_display`. The material brush takes the
    /// no-readback resolve path (resolve.wgsl mode 2 samples the material per texel); this
    /// guards against missing seam edges on the path the material brush actually uses.
    #[test]
    fn gpu_material_display_matches_cpu() {
        let (w, h) = (256u32, 256u32);
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        let material = crate::material::Material {
            width: 4,
            height: 4,
            pixels: (0..16u32)
                .flat_map(|i| [(i * 16) as u8, (255 - i * 15) as u8, ((i * 9) % 256) as u8, 255])
                .collect(),
        };
        let mut r = headless_sized(w, h);
        r.gpu_display = true;
        r.gpu_paint = true;
        r.brush_material = Some(material);
        r.brush_tile = 4.0;
        let brush = Brush {
            radius: 16.0,
            opacity: 1.0,
            hardness: 0.7,
            ..Brush::default()
        };
        r.begin_stroke();
        r.paint_at((cx, cy), &brush);
        r.flush_paint();
        r.paint_segment((cx, cy), (cx + 30.0, cy + 10.0), &brush);
        r.flush_paint();
        r.end_stroke();
        r.flush_paint();

        let size = r.tex_size as usize;
        let atlas = r.gpu_layers.read_atlas(&r.device, &r.queue);
        let cpu = r.composite_display();
        let mut diff = 0u32;
        for t in 0..size * size {
            let i = t * 4;
            let d = (0..3)
                .map(|c| (atlas[i + c] as i32 - cpu[i + c] as i32).abs())
                .max()
                .unwrap();
            if d > 1 {
                diff += 1;
            }
        }
        assert!(
            (diff as f32) < 0.02 * (size * size) as f32,
            "material readback display diverges from CPU composite at {diff} of {} texels",
            size * size
        );
    }

    /// The 2D UV editor, brush preview and export all read the CPU display mirror
    /// (`display_buf` via `atlas_view`), not the GPU atlas. Under GPU display the in-stroke
    /// paint paths deliberately skip the per-frame CPU composite, so without an explicit
    /// re-sync the mirror goes stale — the symptom reported as "UVs / painting on them not
    /// working" in the 2D panel while the 3D model updates correctly. `end_stroke` must leave
    /// `display_buf` byte-identical to a fresh `composite_display`, and bump `paint_version`
    /// so the panel re-uploads. Covers both in-stroke paths: the no-readback resolve path
    /// (solid colour) and the readback path (mask paint).
    #[test]
    fn gpu_display_buf_synced_after_stroke() {
        let (w, h) = (256u32, 256u32);
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        // "color" → resolve path; "mask" → readback path (mask reveal/hide isn't resolve-eligible).
        for target in ["color", "mask"] {
            let mut r = headless_sized(w, h);
            r.gpu_display = true;
            r.gpu_paint = true;
            // Mask paint needs a layer with some colour to reveal/hide, and a mask target.
            if target == "mask" {
                let base = Brush {
                    color: [0.2, 0.7, 0.9],
                    radius: 40.0,
                    opacity: 1.0,
                    ..Brush::default()
                };
                r.begin_stroke();
                r.paint_segment((cx - 50.0, cy), (cx + 50.0, cy), &base);
                r.end_stroke();
                r.flush_paint();
                r.set_paint_target(PaintTarget::Mask);
                r.set_mask_reveal(false); // hide
            }
            let brush = Brush {
                color: [0.85, 0.2, 0.3],
                radius: 16.0,
                opacity: 1.0,
                hardness: 0.7,
                ..Brush::default()
            };
            let ver_before = r.paint_version();
            r.begin_stroke();
            r.paint_at((cx, cy), &brush);
            r.flush_paint();
            r.paint_segment((cx, cy), (cx + 30.0, cy + 12.0), &brush);
            r.flush_paint();
            r.end_stroke();
            r.flush_paint();

            assert!(
                r.paint_version() != ver_before,
                "[{target}] paint_version never bumped — UV editor won't refresh",
            );
            let (_, mirror) = r.atlas_view();
            let cpu = r.composite_display();
            assert_eq!(mirror.len(), cpu.len());
            assert_eq!(
                mirror, &cpu[..],
                "[{target}] display_buf (UV-editor mirror) is stale after a GPU-display \
                 stroke — not byte-identical to composite_display",
            );
        }
    }

    /// Painting in the 2D UV editor draws straight into the panel the user is watching (the CPU
    /// `display_buf` mirror). Under GPU display the 3D-stroke optimization skips the per-frame CPU
    /// composite — but a UV stroke MUST keep it live, or the panel won't update until `end_stroke`
    /// (the stroke looks like it never lands / "only one texel"). Mid-stroke, after a segment +
    /// `flush_paint`, `paint_version` must bump and `display_buf` must show the paint — and still
    /// equal a fresh `composite_display`.
    /// The GPU gutter bleed must use the *current* mesh's UV coverage. The coverage push was
    /// keyed only on resolution, so a same-resolution mesh swap (load / re-unwrap) left the GPU
    /// bleeding the new atlas with the PREVIOUS mesh's gutter mask — jagged seams at every real
    /// island edge on the model, while the CPU path (which rebuilds coverage) stayed clean. After
    /// a swap the GPU atlas must match `composite_display` (which uses the new coverage).
    #[test]
    fn gpu_bleed_uses_current_mesh_coverage_after_swap() {
        let (w, h) = (256u32, 256u32);
        let mut r = headless_sized(w, h);
        r.gpu_display = true;
        r.gpu_paint = true;
        // Paint a stroke so islands carry a distinct colour for the bleed to spread.
        let brush = Brush {
            color: [0.9, 0.2, 0.2],
            radius: 20.0,
            opacity: 1.0,
            ..Brush::default()
        };
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        r.begin_stroke();
        r.paint_segment((cx - 40.0, cy), (cx + 40.0, cy), &brush);
        r.end_stroke();
        r.flush_paint();

        // Swap to a different UV layout at the SAME resolution (what load_model / re-unwrap do):
        // shrink every UV toward the atlas centre so the coverage/gutter mask changes a lot.
        for v in &mut r.mesh.vertices {
            v.uv = [0.25 + v.uv[0] * 0.5, 0.25 + v.uv[1] * 0.5];
        }
        r.topo_version = r.topo_version.wrapping_add(1);
        *r.coverage.get_mut() = None;
        r.refresh_display_texture(); // recompose CPU mirror + GPU atlas with the new layout

        let n = (r.tex_size * r.tex_size) as usize;
        let atlas = r.gpu_layers.read_atlas(&r.device, &r.queue);
        let cpu = r.composite_display();
        let mut diff = 0u32;
        for t in 0..n {
            let i = t * 4;
            let d = (0..3)
                .map(|c| (atlas[i + c] as i32 - cpu[i + c] as i32).abs())
                .max()
                .unwrap();
            if d > 1 {
                diff += 1;
            }
        }
        assert!(
            (diff as f32) < 0.02 * n as f32,
            "GPU atlas bleed used stale coverage after a same-resolution mesh swap: \
             {diff} of {n} texels differ from CPU composite",
        );
    }

    #[test]
    fn gpu_uv_stroke_updates_panel_live_midstroke() {
        let (w, h) = (256u32, 256u32);
        let mut r = headless_sized(w, h);
        r.gpu_display = true;
        r.gpu_paint = true;
        let base = r.display_buf.clone();
        let ver0 = r.paint_version();
        let brush = Brush {
            color: [0.9, 0.15, 0.2],
            radius: 12.0,
            opacity: 1.0,
            ..Brush::default()
        };
        r.begin_stroke();
        // A diagonal UV stroke straight in texel space — no end_stroke yet (still dragging).
        r.paint_uv_segment(glam::Vec2::new(0.3, 0.3), glam::Vec2::new(0.7, 0.7), &brush);
        r.flush_paint();

        assert!(
            r.paint_version() != ver0,
            "UV stroke didn't bump paint_version mid-stroke — the 2D panel won't refresh until release"
        );
        let (_, mirror) = r.atlas_view();
        assert_ne!(
            mirror, &base[..],
            "UV stroke not visible in the panel mirror mid-stroke (stale until end_stroke)"
        );
        let cpu = r.composite_display();
        assert_eq!(
            mirror, &cpu[..],
            "mid-stroke UV panel mirror diverges from composite_display"
        );
        // The model atlas must show it live too (the split view shows both).
        let atlas = r.gpu_layers.read_atlas(&r.device, &r.queue);
        let painted = atlas
            .chunks_exact(4)
            .zip(base.chunks_exact(4))
            .filter(|(a, b)| a != b)
            .count();
        assert!(painted > 50, "UV stroke not visible on the model atlas mid-stroke: {painted}");
    }

    #[test]
    fn gpu_resolve_stroke_matches_cpu_stroke() {
        // The no-readback resolve path (GPU display + GPU paint): the dab coverage resolves
        // straight into the active GPU layer slice, and `end_stroke` reconciles it back to
        // the CPU `Layers`. The reconciled pixels must match the pure-CPU stroke within the
        // same tolerance as the readback GPU path — same coverage, same blend math (±1).
        let (w, h) = (256u32, 256u32);
        let brush = Brush {
            color: [0.85, 0.2, 0.3],
            radius: 14.0,
            opacity: 1.0,
            hardness: 0.6,
            ..Brush::default()
        };
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);

        let paint = |resolve: bool| -> Vec<u8> {
            let mut r = headless_sized(w, h);
            if resolve {
                r.gpu_display = true;
                r.gpu_paint = true;
            }
            r.begin_stroke();
            r.paint_at((cx, cy), &brush);
            r.flush_paint();
            r.paint_segment((cx, cy), (cx + 28.0, cy + 16.0), &brush);
            r.flush_paint();
            r.end_stroke();
            r.flush_paint();
            r.layers.active_tex().pixels.clone()
        };
        let cpu = paint(false);
        let gpu = paint(true);
        let base = headless_sized(w, h).layers.active_tex().pixels.clone();

        let painted = |buf: &[u8]| {
            buf.chunks_exact(4)
                .zip(base.chunks_exact(4))
                .filter(|(a, b)| a != b)
                .count()
        };
        let (cpu_n, gpu_n) = (painted(&cpu), painted(&gpu));
        assert!(cpu_n > 200, "CPU stroke painted too little: {cpu_n}");
        assert!(
            gpu_n as f32 > 0.7 * cpu_n as f32 && (gpu_n as f32) < 1.4 * cpu_n as f32,
            "resolve painted region differs too much from CPU ({gpu_n} vs {cpu_n})"
        );

        let mut sum = 0i64;
        let mut n = 0i64;
        let mut big = 0i64;
        for i in 0..cpu.len() {
            if cpu[i] != base[i] || gpu[i] != base[i] {
                let d = (cpu[i] as i32 - gpu[i] as i32).abs();
                sum += d as i64;
                n += 1;
                if d > 64 {
                    big += 1;
                }
            }
        }
        let mean = sum as f64 / n.max(1) as f64;
        assert!(mean < 6.0, "resolve vs CPU stroke mean channel diff too high: {mean:.2}");
        assert!(
            (big as f64) < 0.04 * n as f64,
            "too many large resolve/CPU channel diffs: {big} of {n}"
        );
    }

    #[test]
    fn gpu_erase_stroke_matches_cpu_stroke() {
        // The GPU eraser path (coverage → `erase_texel` at resolve) must match the CPU
        // eraser within tolerance: erasing the opaque base lowers alpha by coverage, RGB
        // unchanged. Same dab coverage, so only f16 precision + the edge rim differ.
        let (w, h) = (256u32, 256u32);
        let brush = Brush {
            radius: 14.0,
            opacity: 1.0,
            hardness: 0.6,
            erase: true,
            ..Brush::default()
        };
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        let erase = |gpu: bool| -> Vec<u8> {
            let mut r = headless_sized(w, h);
            r.gpu_paint = gpu;
            r.begin_stroke();
            r.paint_at((cx, cy), &brush);
            r.paint_segment((cx, cy), (cx + 28.0, cy + 16.0), &brush);
            r.end_stroke();
            r.flush_paint();
            r.layers.active_tex().pixels.clone()
        };
        let cpu = erase(false);
        let gpu = erase(true);
        let base = headless_sized(w, h).layers.active_tex().pixels.clone();

        // Erasing the opaque base shows up as alpha changes; both paths must lower alpha
        // over a comparable region, and agree closely.
        let erased = |buf: &[u8]| {
            (0..buf.len() / 4)
                .filter(|&t| buf[t * 4 + 3] != base[t * 4 + 3])
                .count()
        };
        let (cpu_n, gpu_n) = (erased(&cpu), erased(&gpu));
        assert!(cpu_n > 200, "CPU eraser changed too little: {cpu_n}");
        // The GPU dab raster is conservative (covers a thin edge rim the CPU scanline omits),
        // so the GPU region is a slight superset — allow it to run a little larger, but it must
        // not be smaller (that would be the unpainted-edge bug).
        assert!(
            gpu_n as f32 > 0.85 * cpu_n as f32 && (gpu_n as f32) < 1.5 * cpu_n as f32,
            "GPU erased region differs too much from CPU ({gpu_n} vs {cpu_n})"
        );
        // GPU must not MISS texels the CPU erased (the "teeth"): count CPU-only changes.
        let cpu_only = (0..cpu.len() / 4)
            .filter(|&t| cpu[t * 4 + 3] != base[t * 4 + 3] && gpu[t * 4 + 3] == base[t * 4 + 3])
            .count();
        assert!(
            (cpu_only as f32) < 0.05 * cpu_n as f32,
            "GPU eraser misses texels the CPU erased (teeth): {cpu_only} of {cpu_n}"
        );
        // Interior agreement: where both paths erased, the alpha must match closely (the thin
        // conservative rim, where only one side changed, is excluded).
        let mut sum = 0i64;
        let mut n = 0i64;
        for t in 0..cpu.len() / 4 {
            let (ca, ga, ba) = (cpu[t * 4 + 3], gpu[t * 4 + 3], base[t * 4 + 3]);
            if ca != ba && ga != ba {
                sum += (ca as i32 - ga as i32).abs() as i64;
                n += 1;
            }
        }
        let mean = sum as f64 / n.max(1) as f64;
        assert!(mean < 6.0, "GPU vs CPU eraser interior mean alpha diff too high: {mean:.2}");
    }

    #[test]
    fn gpu_mask_stroke_matches_cpu_stroke() {
        // Mask painting on the GPU path resolves the coverage into the layer's reveal mask
        // (here hide = black into the white mask) instead of colour. Must match the CPU mask
        // deposit within tolerance; the compositor reads the mask's red channel, so compare
        // that.
        let (w, h) = (256u32, 256u32);
        let brush = Brush {
            radius: 14.0,
            opacity: 1.0,
            hardness: 0.6,
            ..Brush::default()
        };
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        let mask = |gpu: bool| -> Vec<u8> {
            let mut r = headless_sized(w, h);
            r.gpu_paint = gpu;
            r.set_paint_target(PaintTarget::Mask);
            r.set_mask_reveal(false); // hide: blend black into the white mask
            r.begin_stroke();
            r.paint_at((cx, cy), &brush);
            r.paint_segment((cx, cy), (cx + 28.0, cy + 16.0), &brush);
            r.end_stroke();
            r.flush_paint();
            r.layers.active_mask().pixels.clone()
        };
        let cpu = mask(false);
        let gpu = mask(true);

        // The mask starts white (red 255); hiding lowers red. Both paths must carve a
        // comparable region and agree on the red channel.
        let carved = |buf: &[u8]| (0..buf.len() / 4).filter(|&t| buf[t * 4] != 255).count();
        let (cpu_n, gpu_n) = (carved(&cpu), carved(&gpu));
        assert!(cpu_n > 200, "CPU mask carved too little: {cpu_n}");
        // Conservative GPU dab raster → a slight superset (thin edge rim); GPU may carve a bit
        // more, but must not carve less (the unpainted-edge bug).
        assert!(
            gpu_n as f32 > 0.85 * cpu_n as f32 && (gpu_n as f32) < 1.5 * cpu_n as f32,
            "GPU carved region differs too much from CPU ({gpu_n} vs {cpu_n})"
        );
        // GPU must not miss texels the CPU carved (the "teeth").
        let cpu_only = (0..cpu.len() / 4)
            .filter(|&t| cpu[t * 4] != 255 && gpu[t * 4] == 255)
            .count();
        assert!(
            (cpu_only as f32) < 0.05 * cpu_n as f32,
            "GPU mask misses texels the CPU carved (teeth): {cpu_only} of {cpu_n}"
        );
        // Interior agreement: where both paths carved, the red channel must match closely.
        let mut sum = 0i64;
        let mut n = 0i64;
        for t in 0..cpu.len() / 4 {
            if cpu[t * 4] != 255 && gpu[t * 4] != 255 {
                sum += (cpu[t * 4] as i32 - gpu[t * 4] as i32).abs() as i64;
                n += 1;
            }
        }
        let mean = sum as f64 / n.max(1) as f64;
        assert!(mean < 6.0, "GPU vs CPU mask interior mean red diff too high: {mean:.2}");
    }

    #[test]
    fn gpu_tiled_image_stroke_matches_cpu_stroke() {
        // The tiled image brush on the GPU path: the GPU computes coverage, the resolve
        // samples the material per texel (UV-anchored, tiled) exactly as the CPU `deposit`.
        // Must match the CPU path within tolerance — only coverage (f16 + edge rule) differs;
        // the sampled colours are identical. (This is the brush the lag report was using.)
        let (w, h) = (256u32, 256u32);
        // A 4×4 material with distinct opaque colours so tiling is visible.
        let material = {
            let mut pixels = Vec::with_capacity(16 * 4);
            for i in 0..16u32 {
                pixels.extend_from_slice(&[
                    (i * 16) as u8,
                    (255 - i * 15) as u8,
                    ((i * 9) % 256) as u8,
                    255,
                ]);
            }
            crate::material::Material {
                width: 4,
                height: 4,
                pixels,
            }
        };
        let brush = Brush {
            radius: 14.0,
            opacity: 1.0,
            hardness: 0.6,
            ..Brush::default()
        };
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        let paint = |gpu: bool| -> Vec<u8> {
            let mut r = headless_sized(w, h);
            r.gpu_paint = gpu;
            r.brush_material = Some(material.clone()); // Tiled mode by default
            r.brush_tile = 4.0;
            r.begin_stroke();
            r.paint_at((cx, cy), &brush);
            r.paint_segment((cx, cy), (cx + 28.0, cy + 16.0), &brush);
            r.end_stroke();
            r.flush_paint();
            r.layers.active_tex().pixels.clone()
        };
        let cpu = paint(false);
        let gpu = paint(true);
        let base = headless_sized(w, h).layers.active_tex().pixels.clone();

        let painted = |buf: &[u8]| {
            buf.chunks_exact(4)
                .zip(base.chunks_exact(4))
                .filter(|(a, b)| a != b)
                .count()
        };
        let (cpu_n, gpu_n) = (painted(&cpu), painted(&gpu));
        assert!(cpu_n > 200, "CPU tiled stroke painted too little: {cpu_n}");
        assert!(
            gpu_n as f32 > 0.7 * cpu_n as f32 && (gpu_n as f32) < 1.4 * cpu_n as f32,
            "GPU tiled region differs too much from CPU ({gpu_n} vs {cpu_n})"
        );
        let mut sum = 0i64;
        let mut n = 0i64;
        for i in 0..cpu.len() {
            if cpu[i] != base[i] || gpu[i] != base[i] {
                sum += (cpu[i] as i32 - gpu[i] as i32).abs() as i64;
                n += 1;
            }
        }
        let mean = sum as f64 / n.max(1) as f64;
        assert!(mean < 8.0, "GPU vs CPU tiled mean channel diff too high: {mean:.2}");
    }

    #[test]
    fn gpu_resolve_material_stroke_matches_cpu_stroke() {
        // The tiled material brush on the no-readback *resolve* path (GPU display + paint):
        // the resolve shader samples the material per texel and reconciles to the CPU layer.
        // Must match the pure-CPU stroke within tolerance (coverage f16 + nearest material
        // wrap are the only sources of difference). This is the brush from the lag report.
        let (w, h) = (256u32, 256u32);
        let material = {
            let mut pixels = Vec::with_capacity(16 * 4);
            for i in 0..16u32 {
                pixels.extend_from_slice(&[
                    (i * 16) as u8,
                    (255 - i * 15) as u8,
                    ((i * 9) % 256) as u8,
                    255,
                ]);
            }
            crate::material::Material {
                width: 4,
                height: 4,
                pixels,
            }
        };
        let brush = Brush {
            radius: 14.0,
            opacity: 1.0,
            hardness: 0.6,
            ..Brush::default()
        };
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        let paint = |resolve: bool| -> Vec<u8> {
            let mut r = headless_sized(w, h);
            if resolve {
                r.gpu_display = true;
                r.gpu_paint = true;
            }
            r.brush_material = Some(material.clone());
            r.brush_tile = 4.0;
            r.begin_stroke();
            r.paint_at((cx, cy), &brush);
            r.flush_paint();
            r.paint_segment((cx, cy), (cx + 28.0, cy + 16.0), &brush);
            r.flush_paint();
            r.end_stroke();
            r.flush_paint();
            r.layers.active_tex().pixels.clone()
        };
        let cpu = paint(false);
        let gpu = paint(true);
        let base = headless_sized(w, h).layers.active_tex().pixels.clone();

        let painted = |buf: &[u8]| {
            buf.chunks_exact(4)
                .zip(base.chunks_exact(4))
                .filter(|(a, b)| a != b)
                .count()
        };
        let (cpu_n, gpu_n) = (painted(&cpu), painted(&gpu));
        assert!(cpu_n > 200, "CPU material stroke painted too little: {cpu_n}");
        assert!(
            gpu_n as f32 > 0.7 * cpu_n as f32 && (gpu_n as f32) < 1.4 * cpu_n as f32,
            "resolve material region differs too much from CPU ({gpu_n} vs {cpu_n})"
        );
        let mut sum = 0i64;
        let mut n = 0i64;
        for i in 0..cpu.len() {
            if cpu[i] != base[i] || gpu[i] != base[i] {
                sum += (cpu[i] as i32 - gpu[i] as i32).abs() as i64;
                n += 1;
            }
        }
        let mean = sum as f64 / n.max(1) as f64;
        assert!(mean < 8.0, "resolve vs CPU material mean channel diff too high: {mean:.2}");
    }
}

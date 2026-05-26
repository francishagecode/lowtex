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
//   mouse pixel → Ray::from_screen → BVH pick_uv → Texture::stamp_stroke →
//   (palette quantize if enabled) → Queue::write_texture (CPU → GPU)
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

use crate::bake::{Levels, MapSource};
use crate::bvh::Bvh;
use crate::camera::Camera;
use crate::history::History;
use crate::layers::Layers;
use crate::mesh::{Mesh, Vertex};
use crate::paint::{Brush, Ray, Texture as PaintTexture};
use crate::palette::Palette;

const TEX_SIZE: u32 = 128; // PSX-scale. Bump to 256 if you want.

/// How many undo steps to keep. Each step is a full layer-stack snapshot (a few
/// hundred KB at PSX sizes), so this bounds history memory to a handful of MB.
const HISTORY_CAP: usize = 64;

/// The color format used for offscreen capture. sRGB to match the surface.
const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

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

/// What the brush paints into (G11).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PaintTarget {
    /// The active layer's color.
    Color,
    /// The active layer's reveal mask.
    Mask,
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

    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,

    paint_texture_gpu: wgpu::Texture,
    // Layer stack (G10); painting targets the active layer, composited bottom-up
    // into the GPU texture (then palette-quantized).
    layers: Layers,
    tex_size: u32,

    // Stroke accumulation (G6): the active-layer snapshot at stroke start +
    // per-texel coverage, so overlapping stamps within one stroke don't double-darken.
    stroke_base: Vec<u8>,
    stroke_coverage: Vec<f32>,

    // Whether the brush paints the active layer's color or its reveal mask (G11),
    // and (in mask mode) whether it reveals (white) or hides (black).
    paint_target: PaintTarget,
    mask_reveal: bool,

    // Undo/redo over the layer stack. A stroke records one entry on release (via
    // `pending` + `stroke_dirty`); discrete layer ops record one each.
    history: History,
    // The layer stack as it was when the in-progress stroke began; committed to
    // history on `end_stroke` only if the stroke actually painted anything.
    pending: Option<Layers>,
    stroke_dirty: bool,

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
    mesh: Mesh,
    bvh: Bvh,
    // Cached baked mesh maps (AO/edge) at the current resolution; invalidated on
    // model load or resolution change.
    mesh_maps: Option<crate::bake::MeshMaps>,
    // Cached UV-island map at the current resolution, driving the fill tools.
    // Invalidated alongside mesh_maps (geometry or resolution change).
    fill_map: Option<crate::fill::FillMap>,
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
        upload_texture(&queue, &paint_texture_gpu, &base);
        let layers = Layers::new(base);

        // Nearest-neighbor sampler — pure PSX, no filtering.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("nearest sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

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

        // Palette quantize is applied to the texture on the CPU (G8).
        let palette = Palette::builtins().remove(0); // PICO-8 by default
        let palette_settings = PaletteSettings::default();

        // BVH over the mesh triangles for fast ray picking (G5).
        let bvh = Bvh::build(&mesh);

        let stroke_base = layers.active_tex().pixels.clone();
        let stroke_coverage = vec![0.0; (tex_size * tex_size) as usize];

        Self {
            window: None,
            surface: None,
            device,
            queue,
            config,
            pipeline,
            vertex_buffer,
            index_buffer,
            index_count,
            uniform_buffer,
            bind_group,
            bind_group_layout,
            sampler,
            paint_texture_gpu,
            layers,
            tex_size,
            stroke_base,
            stroke_coverage,
            paint_target: PaintTarget::Color,
            mask_reveal: false,
            history: History::new(HISTORY_CAP),
            pending: None,
            stroke_dirty: false,
            depth_view,
            palette,
            palette_settings,
            egui_renderer: None,
            max_texture_dim,
            camera,
            mesh,
            bvh,
            mesh_maps: None,
            fill_map: None,
        }
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

        self.camera.aspect = width as f32 / height as f32;
        self.update_uniforms();
    }

    /// Write the view-projection matrix to the GPU.
    fn update_uniforms(&self) {
        let view_proj = self.camera.view_proj();
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[view_proj.to_cols_array_2d()]),
        );
    }

    /// Composite the layers, palette-quantize the result (if enabled), and upload
    /// it. Non-destructive — the layers keep full color.
    fn refresh_display_texture(&self) {
        let mut display = self.layers.composite();
        if self.palette_settings.enabled && !self.palette.colors.is_empty() {
            self.palette.quantize_rgba(
                &mut display,
                self.tex_size,
                self.palette_settings.dither,
                self.palette_settings.dither_strength,
            );
        }
        upload_pixels(
            &self.queue,
            &self.paint_texture_gpu,
            &display,
            self.tex_size,
            self.tex_size,
        );
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

    /// The composited (and quantized, if enabled) texture as currently shown —
    /// used for export so the PNG matches the model.
    pub fn display_pixels(&self) -> Vec<u8> {
        let mut px = self.layers.composite();
        if self.palette_settings.enabled && !self.palette.colors.is_empty() {
            self.palette.quantize_rgba(
                &mut px,
                self.tex_size,
                self.palette_settings.dither,
                self.palette_settings.dither_strength,
            );
        }
        px
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
    }

    /// Step back to the previous layer state, if any.
    pub fn undo(&mut self) {
        let current = self.layers.clone();
        if let Some(prev) = self.history.undo(current) {
            self.layers = prev;
            self.restore_after_history();
        }
    }

    /// Step forward to the next layer state, if any.
    pub fn redo(&mut self) {
        let current = self.layers.clone();
        if let Some(next) = self.history.redo(current) {
            self.layers = next;
            self.restore_after_history();
        }
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

    /// Add a generated tint layer: a flat `color` whose alpha is a baked map
    /// (`src`) read through `levels`. This is the one path behind every AO/curvature
    /// effect — the presets below are just fixed (source, color, blend) choices.
    pub fn add_map_layer(
        &mut self,
        name: &str,
        src: MapSource,
        levels: Levels,
        color: [u8; 3],
        blend: crate::layers::BlendMode,
    ) {
        self.checkpoint();
        self.ensure_mesh_maps();
        let weights = self.mesh_maps.as_ref().unwrap().sample(src, &levels);
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
        self.resync_stroke_base();
        self.refresh_display_texture();
    }

    /// Fill the *active layer's* reveal mask from a baked map — the Substance-style
    /// move: route AO/curvature into a mask so whatever that layer paints only shows
    /// where the map is high (e.g. paint a flat color, then confine it to cavities).
    pub fn fill_active_mask_from_map(&mut self, src: MapSource, levels: Levels) {
        self.checkpoint();
        self.ensure_mesh_maps();
        let weights = self.mesh_maps.as_ref().unwrap().sample(src, &levels);
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
    pub fn apply_ao_layer(&mut self, levels: Levels) {
        self.add_map_layer(
            "AO",
            MapSource::Cavities,
            levels,
            [0, 0, 0],
            crate::layers::BlendMode::Multiply,
        );
    }

    /// Preset: a white layer on convex edges/corners — brightens exposed edges and,
    /// being curvature-driven, never lands on flat lit faces or in concave creases.
    pub fn apply_highlight_layer(&mut self, levels: Levels) {
        self.add_map_layer(
            "Highlights",
            MapSource::Edges,
            levels,
            [255, 255, 255],
            crate::layers::BlendMode::Normal,
        );
    }

    /// Preset: a dark grime tint settling into the cavities (Substance "Dirt").
    pub fn apply_dirt_layer(&mut self, levels: Levels) {
        self.add_map_layer(
            "Dirt",
            MapSource::Cavities,
            levels,
            [54, 42, 30],
            crate::layers::BlendMode::Normal,
        );
    }

    /// Preset: a worn, lightened tint on convex edges (Substance "Edge wear").
    pub fn apply_edge_wear_layer(&mut self, levels: Levels) {
        self.add_map_layer(
            "Edge wear",
            MapSource::Edges,
            levels,
            [205, 200, 185],
            crate::layers::BlendMode::Normal,
        );
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
        self.mesh_maps = None; // geometry changed; baked maps are stale
        self.fill_map = None; // geometry changed; island map is stale
        Ok(())
    }

    /// Re-unwrap the current mesh's UVs (G14–G17). Geometry is unchanged but
    /// vertices are split and re-UV'd, so the GPU buffers, BVH, and cached maps are
    /// rebuilt; the painted texture stays and re-maps onto the new UVs.
    pub fn apply_unwrap(&mut self, mode: crate::unwrap::UnwrapMode) {
        self.checkpoint();
        let mesh = crate::unwrap::unwrap(&self.mesh, mode);
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
        self.mesh_maps = None;
        self.fill_map = None;
    }

    /// Save the entire editing state to a `.lowtex` file (G24).
    pub fn save_project(&self, path: &str) -> Result<(), String> {
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
        };
        doc.save(path)
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
        self.mesh_maps = None;
        self.fill_map = None;

        // Rebuild the layer stack.
        let n = (doc.tex_size * doc.tex_size * 4) as usize;
        let mut layers = Vec::with_capacity(doc.layers.len());
        for d in &doc.layers {
            let mut color = crate::project::decode_pixels(&d.color)?;
            let mut mask = crate::project::decode_pixels(&d.mask)?;
            color.resize(n, 0);
            mask.resize(n, 255);
            layers.push(crate::layers::Layer {
                name: d.name.clone(),
                tex: PaintTexture {
                    width: doc.tex_size,
                    height: doc.tex_size,
                    pixels: color,
                },
                mask: PaintTexture {
                    width: doc.tex_size,
                    height: doc.tex_size,
                    pixels: mask,
                },
                visible: d.visible,
                opacity: d.opacity,
                blend: crate::layers::BlendMode::ALL
                    .get(d.blend as usize)
                    .copied()
                    .unwrap_or(crate::layers::BlendMode::Normal),
            });
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

        // Recreate the GPU paint texture for the (possibly new) resolution and
        // re-snapshot stroke buffers + refresh the display.
        self.rebuild_paint_gpu();
        Ok(())
    }

    /// Fill the active layer's color with a loaded material image (brick, moss, …),
    /// UV-tiled `tile` times. Combine with a reveal mask (e.g. mask-from-Cavities)
    /// for "moss in the crevices". Undoable.
    pub fn fill_active_with_material(&mut self, path: &str, tile: f32) -> Result<(), String> {
        let material = crate::material::Material::load(path)?;
        self.checkpoint();
        material.fill(self.layers.active_tex_mut(), tile);
        self.resync_stroke_base();
        self.refresh_display_texture();
        Ok(())
    }

    /// Begin a new stroke: snapshot the texture and clear stroke coverage, so
    /// overlap within the stroke accumulates by max-coverage (no double-darken).
    pub fn begin_stroke(&mut self) {
        self.stroke_base = self.target_pixels().to_vec();
        self.stroke_coverage.fill(0.0);
        // Snapshot the pre-stroke stack so the whole stroke is one undo step. Held
        // until release, then committed only if the stroke actually painted.
        self.pending = Some(self.layers.clone());
        self.stroke_dirty = false;
    }

    /// End the current stroke, committing it to history as a single undo step.
    /// A stroke that never hit the mesh (`stroke_dirty` false) records nothing,
    /// so an errant click off the model doesn't leave an empty undo entry.
    pub fn end_stroke(&mut self) {
        if let Some(before) = self.pending.take() {
            if self.stroke_dirty {
                self.history.record(before);
            }
        }
        self.stroke_dirty = false;
    }

    /// The current screen size (window physical size, or the headless config).
    fn screen_size(&self) -> Vec2 {
        match &self.window {
            Some(window) => {
                let s = window.inner_size();
                Vec2::new(s.width as f32, s.height as f32)
            }
            None => Vec2::new(self.config.width as f32, self.config.height as f32),
        }
    }

    /// Build a stable world-space pick ray for a screen pixel.
    fn pick_ray(&self, mouse_px: Vec2) -> Ray {
        let inv_view_proj = self.camera.view_proj().inverse();
        let ray_origin = self.camera.eye();
        let ray = Ray::from_screen(mouse_px, self.screen_size(), inv_view_proj);
        // Pin the origin to the camera eye for stability — unproject can wobble
        // for near==0 depending on driver.
        Ray {
            origin: ray_origin,
            direction: (ray.origin + ray.direction - ray_origin).normalize(),
        }
    }

    /// Pick + stamp a single dab at a screen pixel (no GPU upload). Returns
    /// whether it hit the mesh.
    fn stamp_screen(&mut self, mouse_px: Vec2, brush: &Brush) -> bool {
        let ray = self.pick_ray(mouse_px);
        if let Some(uv) = self.bvh.pick_uv(&ray) {
            let base = &self.stroke_base;
            let coverage = &mut self.stroke_coverage;
            match self.paint_target {
                PaintTarget::Color => {
                    self.layers
                        .active_tex_mut()
                        .stamp_stroke(uv, brush, base, coverage);
                }
                PaintTarget::Mask => {
                    // Mask painting ignores the brush color: reveal=white, hide=black.
                    let v = if self.mask_reveal { 1.0 } else { 0.0 };
                    let mask_brush = Brush {
                        color: [v, v, v],
                        ..*brush
                    };
                    self.layers
                        .active_mask_mut()
                        .stamp_stroke(uv, &mask_brush, base, coverage);
                }
            }
            self.stroke_dirty = true;
            true
        } else {
            false
        }
    }

    /// Stamp a single dab at a screen pixel and upload. Used for the initial
    /// click of a stroke and for headless verification.
    pub fn paint_at(&mut self, mouse_px: (f32, f32), brush: &Brush) {
        if self.stamp_screen(Vec2::new(mouse_px.0, mouse_px.1), brush) {
            self.refresh_display_texture();
        }
    }

    /// Paint a continuous stroke segment from `from` to `to` (screen pixels),
    /// interpolating stamps so a fast drag leaves a solid line, not gappy dots.
    /// Re-picks at each sub-sample so the stroke follows the surface across faces.
    pub fn paint_segment(&mut self, from: (f32, f32), to: (f32, f32), brush: &Brush) {
        let from = Vec2::new(from.0, from.1);
        let to = Vec2::new(to.0, to.1);
        // ~2px steps guarantee overlap regardless of brush size. Cap the count so
        // a giant drag can't stall; coverage accumulation prevents double-darken.
        const STEP_PX: f32 = 2.0;
        let dist = (to - from).length();
        let steps = ((dist / STEP_PX).ceil() as u32).clamp(1, 1024);
        let mut hit_any = false;
        for k in 1..=steps {
            let t = k as f32 / steps as f32;
            hit_any |= self.stamp_screen(from.lerp(to, t), brush);
        }
        if hit_any {
            self.refresh_display_texture();
        }
    }

    /// Record the scene into `target_view`. Shared by window + offscreen paths.
    fn draw_into(&self, encoder: &mut wgpu::CommandEncoder, target_view: &wgpu::TextureView) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    // PSX-ish dark teal background.
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.04,
                        g: 0.06,
                        b: 0.08,
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

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..self.index_count, 0, 0..1);
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

fn upload_texture(queue: &wgpu::Queue, gpu: &wgpu::Texture, cpu: &PaintTexture) {
    upload_pixels(queue, gpu, &cpu.pixels, cpu.width, cpu.height);
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

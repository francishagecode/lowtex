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
//   mouse pixel → Ray::from_screen → pick_uv → Texture::paint_brush
//                                              ↓
//                                       Queue::write_texture (CPU → GPU)
//
// Flow on each frame:
//   acquire target view → encode draw → submit → present

use std::sync::Arc;

use glam::Vec2;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::bvh::Bvh;
use crate::camera::Camera;
use crate::mesh::{Mesh, Vertex};
use crate::paint::{Brush, Ray, Texture as PaintTexture};

const TEX_SIZE: u32 = 128; // PSX-scale. Bump to 256 if you want.

/// The color format used for offscreen capture. sRGB to match the surface.
const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// egui paint data for one frame, produced by the App and consumed by `render`.
pub struct UiPaint<'a> {
    pub jobs: &'a [egui::ClippedPrimitive],
    pub textures_delta: &'a egui::TexturesDelta,
    pub pixels_per_point: f32,
}

/// PSX-look render settings (G7). `enabled` is the master toggle; the individual
/// flags dial each effect. `effective_*` resolves `enabled && flag`.
#[derive(Clone, Copy)]
pub struct PsxSettings {
    pub enabled: bool,
    pub affine: bool,
    pub snap: bool,
    pub grid: f32,
    pub fog: bool,
    pub fog_color: [f32; 3],
    pub fog_start: f32,
    pub fog_end: f32,
    pub flat: bool,
}

impl Default for PsxSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            affine: true,
            snap: true,
            grid: 64.0,
            fog: false,
            fog_color: [0.04, 0.06, 0.08], // matches the viewport clear
            fog_start: 3.0,
            fog_end: 7.0,
            flat: false,
        }
    }
}

/// GPU uniform block. Layout matches `Uniforms` in main.wgsl (std140-friendly:
/// mat4 then 16-byte-aligned vec4s).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    fog_color: [f32; 4],
    params: [f32; 4],  // affine, snap, grid, fog
    params2: [f32; 4], // flat, fog_start, fog_end, unused
}

impl Uniforms {
    fn new(view_proj: glam::Mat4, psx: &PsxSettings) -> Self {
        let on = |b: bool| if psx.enabled && b { 1.0 } else { 0.0 };
        Self {
            view_proj: view_proj.to_cols_array_2d(),
            fog_color: [psx.fog_color[0], psx.fog_color[1], psx.fog_color[2], 1.0],
            params: [on(psx.affine), on(psx.snap), psx.grid.max(1.0), on(psx.fog)],
            params2: [on(psx.flat), psx.fog_start, psx.fog_end, 0.0],
        }
    }
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
    paint_texture_cpu: PaintTexture,
    tex_size: u32,

    // Stroke accumulation (G6): the texture snapshot at stroke start + per-texel
    // coverage, so overlapping stamps within one stroke don't double-darken.
    stroke_base: Vec<u8>,
    stroke_coverage: Vec<f32>,

    depth_view: wgpu::TextureView,

    // egui overlay renderer; None in headless mode.
    egui_renderer: Option<egui_wgpu::Renderer>,

    // Surface/texture dimensions are clamped to this (GPU limit).
    max_texture_dim: u32,

    camera: Camera,
    psx: PsxSettings,
    // Kept for goals that re-read geometry (unwrap G14, bake G19); the BVH owns
    // its own triangle copy for picking.
    #[allow(dead_code)]
    mesh: Mesh,
    bvh: Bvh,
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

        // Paint texture: faint checkerboard so the UV layout is visible pre-paint.
        let tex_size = TEX_SIZE;
        let paint_texture_cpu = make_checkerboard(tex_size, tex_size);
        let paint_texture_gpu = make_paint_texture(&device, &paint_texture_cpu);
        upload_texture(&queue, &paint_texture_gpu, &paint_texture_cpu);

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

        // Uniform buffer: view-projection + PSX params.
        let camera = Camera::new(width as f32 / height as f32);
        let psx = PsxSettings::default();
        let uniforms = Uniforms::new(camera.view_proj(), &psx);
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniform buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    // Vertex reads view_proj + snap; fragment reads affine/flat/fog flags.
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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

        // BVH over the mesh triangles for fast ray picking (G5).
        let bvh = Bvh::build(&mesh);

        let stroke_base = paint_texture_cpu.pixels.clone();
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
            paint_texture_cpu,
            tex_size,
            stroke_base,
            stroke_coverage,
            depth_view,
            egui_renderer: None,
            max_texture_dim,
            camera,
            psx,
            mesh,
            bvh,
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

    /// Write the view-proj + PSX uniform block to the GPU.
    fn update_uniforms(&self) {
        let uniforms = Uniforms::new(self.camera.view_proj(), &self.psx);
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Replace the PSX render settings and refresh the uniform.
    pub fn set_psx_settings(&mut self, psx: PsxSettings) {
        self.psx = psx;
        self.update_uniforms();
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

    /// Recreate the GPU paint texture + bind group from the current CPU texture.
    /// Call after the CPU texture is replaced or resized.
    fn rebuild_paint_gpu(&mut self) {
        self.tex_size = self.paint_texture_cpu.width;
        self.paint_texture_gpu = make_paint_texture(&self.device, &self.paint_texture_cpu);
        upload_texture(
            &self.queue,
            &self.paint_texture_gpu,
            &self.paint_texture_cpu,
        );
        self.bind_group = make_paint_bind_group(
            &self.device,
            &self.bind_group_layout,
            &self.uniform_buffer,
            &self.paint_texture_gpu,
            &self.sampler,
        );
        // The stroke buffers must track the (possibly resized) texture.
        self.stroke_base = self.paint_texture_cpu.pixels.clone();
        self.stroke_coverage = vec![0.0; (self.tex_size * self.tex_size) as usize];
    }

    /// Save the painted texture to a PNG. The CPU texture holds sRGB-encoded
    /// bytes (the GPU texture is `Rgba8UnormSrgb`), and PNG is sRGB, so the bytes
    /// are written through unchanged.
    pub fn save_texture_png(&self, path: &str) -> Result<(), String> {
        let cpu = &self.paint_texture_cpu;
        image::save_buffer(
            path,
            &cpu.pixels,
            cpu.width,
            cpu.height,
            image::ColorType::Rgba8,
        )
        .map_err(|e| format!("failed to save PNG: {e}"))
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
        self.paint_texture_cpu = loaded.resampled(self.tex_size, self.tex_size);
        self.rebuild_paint_gpu();
        Ok(())
    }

    /// Change the paint texture resolution, resampling existing paint into it.
    pub fn set_texture_resolution(&mut self, size: u32) {
        let size = size.clamp(8, self.max_texture_dim);
        if size == self.tex_size {
            return;
        }
        self.paint_texture_cpu = self.paint_texture_cpu.resampled(size, size);
        self.rebuild_paint_gpu();
    }

    /// Begin a new stroke: snapshot the texture and clear stroke coverage, so
    /// overlap within the stroke accumulates by max-coverage (no double-darken).
    pub fn begin_stroke(&mut self) {
        self.stroke_base.clear();
        self.stroke_base
            .extend_from_slice(&self.paint_texture_cpu.pixels);
        self.stroke_coverage.fill(0.0);
    }

    /// End the current stroke. (The painted pixels are already committed; the
    /// next `begin_stroke` re-snapshots from them.)
    pub fn end_stroke(&mut self) {}

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
            self.paint_texture_cpu.stamp_stroke(
                uv,
                brush,
                &self.stroke_base,
                &mut self.stroke_coverage,
            );
            true
        } else {
            false
        }
    }

    /// Stamp a single dab at a screen pixel and upload. Used for the initial
    /// click of a stroke and for headless verification.
    pub fn paint_at(&mut self, mouse_px: (f32, f32), brush: &Brush) {
        if self.stamp_screen(Vec2::new(mouse_px.0, mouse_px.1), brush) {
            upload_texture(
                &self.queue,
                &self.paint_texture_gpu,
                &self.paint_texture_cpu,
            );
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
            upload_texture(
                &self.queue,
                &self.paint_texture_gpu,
                &self.paint_texture_cpu,
            );
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

        // Scene first (clears + depth), then the egui overlay on top.
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
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: gpu,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &cpu.pixels,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(cpu.width * 4),
            rows_per_image: Some(cpu.height),
        },
        wgpu::Extent3d {
            width: cpu.width,
            height: cpu.height,
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

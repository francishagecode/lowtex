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

use crate::camera::Camera;
use crate::mesh::{Mesh, Vertex};
use crate::paint::{pick_uv, Brush, Ray, Texture as PaintTexture};

const TEX_SIZE: u32 = 128; // PSX-scale. Bump to 256 if you want.

/// The color format used for offscreen capture. sRGB to match the surface.
const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// egui paint data for one frame, produced by the App and consumed by `render`.
pub struct UiPaint<'a> {
    pub jobs: &'a [egui::ClippedPrimitive],
    pub textures_delta: &'a egui::TexturesDelta,
    pub pixels_per_point: f32,
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

    paint_texture_gpu: wgpu::Texture,
    paint_texture_cpu: PaintTexture,

    depth_view: wgpu::TextureView,

    // egui overlay renderer; None in headless mode.
    egui_renderer: Option<egui_wgpu::Renderer>,

    // Surface/texture dimensions are clamped to this (GPU limit).
    max_texture_dim: u32,

    camera: Camera,
    mesh: Mesh,
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
        let paint_texture_cpu = make_checkerboard(TEX_SIZE, TEX_SIZE);
        let paint_texture_gpu = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("paint texture"),
            size: wgpu::Extent3d {
                width: TEX_SIZE,
                height: TEX_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        upload_texture(&queue, &paint_texture_gpu, &paint_texture_cpu);
        let paint_texture_view =
            paint_texture_gpu.create_view(&wgpu::TextureViewDescriptor::default());

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

        // Uniform buffer for the view-projection matrix.
        let camera = Camera::new(width as f32 / height as f32);
        let view_proj = camera.view_proj();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniform buffer"),
            contents: bytemuck::cast_slice(&[view_proj.to_cols_array_2d()]),
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&paint_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

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
            paint_texture_gpu,
            paint_texture_cpu,
            depth_view,
            egui_renderer: None,
            max_texture_dim,
            camera,
            mesh,
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
        self.update_camera_uniform();
    }

    fn update_camera_uniform(&self) {
        let view_proj = self.camera.view_proj();
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[view_proj.to_cols_array_2d()]),
        );
    }

    /// Orbit the camera by drag deltas and refresh the view-proj uniform.
    pub fn orbit_camera(&mut self, dx: f32, dy: f32) {
        self.camera.orbit(dx, dy);
        self.update_camera_uniform();
    }

    /// Pan the camera target by drag deltas and refresh the view-proj uniform.
    pub fn pan_camera(&mut self, dx: f32, dy: f32) {
        self.camera.pan(dx, dy);
        self.update_camera_uniform();
    }

    /// Zoom (dolly) the camera and refresh the view-proj uniform.
    pub fn zoom_camera(&mut self, delta: f32) {
        self.camera.zoom(delta);
        self.update_camera_uniform();
    }

    /// Orbit by explicit angles (radians); for programmatic/headless control.
    pub fn orbit_view_radians(&mut self, d_azimuth: f32, d_elevation: f32) {
        self.camera.orbit_radians(d_azimuth, d_elevation);
        self.update_camera_uniform();
    }

    /// Called when the mouse moves while held, or on click.
    /// Casts a ray, finds the UV, stamps the brush, re-uploads the texture.
    pub fn paint_at(&mut self, mouse_px: (f32, f32), brush: &Brush) {
        let (w, h) = match &self.window {
            Some(window) => {
                let s = window.inner_size();
                (s.width as f32, s.height as f32)
            }
            None => (self.config.width as f32, self.config.height as f32),
        };
        let screen_size = Vec2::new(w, h);
        let mouse = Vec2::new(mouse_px.0, mouse_px.1);

        let inv_view_proj = self.camera.view_proj().inverse();
        let ray_origin = self.camera.eye();
        let ray = Ray::from_screen(mouse, screen_size, inv_view_proj);

        // Pin the origin to the camera eye for stability — unproject can wobble
        // for near==0 depending on driver.
        let ray = Ray {
            origin: ray_origin,
            direction: (ray.origin + ray.direction - ray_origin).normalize(),
        };

        if let Some(uv) = pick_uv(&ray, &self.mesh) {
            self.paint_texture_cpu.stamp(uv, brush);
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

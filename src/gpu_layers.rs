// src/gpu_layers.rs
//
// GPU-resident layer stack + the display composite pass (Phase 1 of moving the per-frame
// pixel pipeline off the CPU). The authoritative pixel store stays the CPU `Layers`
// (src/layers.rs) — save/load, undo, merge, resize all keep reading it — but the
// *runtime* composite that feeds the model runs here on the GPU.
//
// Layers are mirrored into 2D-array textures (one slice per layer: `Rgba8Unorm` colour
// + `R8Unorm` mask). `shaders/composite.wgsl` blends them, bottom-up, into a composite
// **atlas**. The atlas is one `Rgba8Unorm` texture created with an `Rgba8UnormSrgb`
// view format: the composite writes raw bytes through the Unorm view (no sRGB encode,
// matching the CPU byte math), and main.wgsl samples the sRGB view (decode at sample
// time only) — so the look is byte-for-byte the same store the CPU produced, ±1 from u8
// rounding. Later phases fold palette quantize + Bayer (P2) and gutter bleed (P3) into
// this same atlas, then paint resolves directly into the layer slices (P5).

use wgpu::util::DeviceExt;

use crate::layers::{BlendMode, Layers};

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
/// The atlas is stored as linear-`Unorm` (the composite writes raw bytes) but *sampled*
/// as `Srgb` by main.wgsl, via a second view format on the same texture.
const ATLAS_UNORM: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const ATLAS_SRGB: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Fixed palette capacity (a `vec4` per colour). 256 is the most median-cut yields, so the
/// palette buffer + bind group never need rebuilding when the palette changes.
const PALETTE_CAP: usize = 256;

/// Per-layer composite parameters, mirroring `composite.wgsl`'s `LayerParam`
/// (std430: 16 bytes). `visible` folds in the CPU's `visible && opacity > 0` filter.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerParam {
    opacity: f32,
    blend: u32,
    visible: u32,
    _pad: u32,
}

/// Quantize params, mirroring `composite.wgsl`'s `Quant` (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QuantU {
    enabled: u32,
    dither: u32,
    strength: f32,
    palette_len: u32,
}

/// Paint-resolve params, mirroring `resolve.wgsl`'s `ResolveU` (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ResolveU {
    color: [f32; 4],
    mode: u32, // 0 = blend solid colour, 1 = erase, 2 = blend material
    tile: f32,
    _pad: [u32; 2],
}

/// How a resolve paints the coverage onto the base.
pub enum ResolveKind {
    Color([f32; 3]),
    Erase,
    Material(f32), // tile factor; material is set via `set_material`
}

fn blend_code(b: BlendMode) -> u32 {
    match b {
        BlendMode::Normal => 0,
        BlendMode::Multiply => 1,
        BlendMode::Add => 2,
        BlendMode::Screen => 3,
    }
}

/// The array textures + atlas + bind group for a given (size, layer-count). Rebuilt by
/// `ensure` when either changes; the params buffer is rewritten every `upload`.
struct Resident {
    size: u32,
    count: u32,
    color: wgpu::Texture,
    mask: wgpu::Texture,
    atlas: wgpu::Texture,
    atlas_srgb_view: wgpu::TextureView,
    atlas_unorm_view: wgpu::TextureView,
    params: wgpu::Buffer,
    palette: wgpu::Buffer,
    quant: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    // Gutter-bleed ping-pong: two {colour, validity} buffers + the static coverage mask.
    ping_color: [wgpu::Texture; 2],
    ping_valid: [wgpu::Texture; 2],
    ping_color_view: [wgpu::TextureView; 2],
    ping_valid_view: [wgpu::TextureView; 2],
    coverage_tex: wgpu::Texture,
    /// `bleed_src_bg[k]` samples ping buffer `k` (colour + validity) as a pass's source.
    bleed_src_bg: [wgpu::BindGroup; 2],
    /// Immutable pre-stroke copy of the active layer's colour slice — the resolve base, so
    /// re-resolving the dirty region each frame stays idempotent.
    stroke_base: wgpu::Texture,
    stroke_base_view: wgpu::TextureView,
}

/// Owns the composite + bleed pipelines (built once) and the current residency.
pub struct GpuLayers {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    bleed_pipeline: wgpu::RenderPipeline,
    bleed_layout: wgpu::BindGroupLayout,
    resolve_pipeline: wgpu::RenderPipeline,
    resolve_layout: wgpu::BindGroupLayout,
    /// The current brush material as a GPU texture (uploaded by `set_material`), and a 1×1
    /// placeholder bound when the stroke isn't a material brush.
    material: Option<wgpu::TextureView>,
    dummy_material_view: wgpu::TextureView,
    resident: Option<Resident>,
}

impl GpuLayers {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("composite shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/composite.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite bind layout"),
            entries: &[
                // color array
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // mask array
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // per-layer params
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // palette colours
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // quantize params
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite pipeline layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("composite pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: ATLAS_UNORM,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Bleed (gutter dilation): samples a colour + validity texture, writes the
        // dilated colour + validity as MRT.
        let bleed_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bleed shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bleed.wgsl").into()),
        });
        let tex_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let bleed_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bleed bind layout"),
            entries: &[tex_entry(0), tex_entry(1)],
        });
        let bleed_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bleed pipeline layout"),
                bind_group_layouts: &[&bleed_layout],
                push_constant_ranges: &[],
            });
        let bleed_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bleed pipeline"),
            layout: Some(&bleed_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &bleed_shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &bleed_shader,
                entry_point: "fs_main",
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: COLOR_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: MASK_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Paint resolve: base colour + stroke coverage → resolved layer colour.
        let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resolve shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/resolve.wgsl").into()),
        });
        let resolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resolve bind layout"),
            entries: &[
                tex_entry(0), // base
                tex_entry(1), // coverage
                tex_entry(2), // material
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // A 1×1 placeholder bound when a stroke isn't a material brush (never sampled — the
        // shader only reads the material in mode 2, where the real material is bound instead).
        let dummy_material = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dummy material"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_material_view = dummy_material.create_view(&Default::default());
        let resolve_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("resolve pipeline layout"),
                bind_group_layouts: &[&resolve_layout],
                push_constant_ranges: &[],
            });
        let resolve_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("resolve pipeline"),
            layout: Some(&resolve_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &resolve_shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &resolve_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: COLOR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            layout,
            bleed_pipeline,
            bleed_layout,
            resolve_pipeline,
            resolve_layout,
            material: None,
            dummy_material_view,
            resident: None,
        }
    }

    /// Upload the brush material as a GPU texture for the material resolve path. Call when
    /// the material changes (keyed on the brush's material generation), not per stroke.
    pub fn set_material(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mat: &crate::material::Material,
    ) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("brush material"),
            size: wgpu::Extent3d {
                width: mat.width,
                height: mat.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &mat.pixels,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(mat.width * 4),
                rows_per_image: Some(mat.height),
            },
            wgpu::Extent3d {
                width: mat.width,
                height: mat.height,
                depth_or_array_layers: 1,
            },
        );
        self.material = Some(tex.create_view(&Default::default()));
    }

    /// (Re)build the array textures, atlas and bind group when the atlas size or layer
    /// count changes. Allocates exactly `count` array slices (layer-count changes are
    /// rare), so VRAM scales with the real stack, not a fixed cap.
    fn ensure(&mut self, device: &wgpu::Device, size: u32, count: u32) {
        let count = count.max(1);
        let fresh = self
            .resident
            .as_ref()
            .is_none_or(|r| r.size != size || r.count != count);
        if !fresh {
            return;
        }
        let array_tex = |label, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: count,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let color = array_tex(
            "layer color",
            COLOR_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC // snapshot the active slice to stroke_base
                | wgpu::TextureUsages::RENDER_ATTACHMENT, // paint resolves into a slice
        );
        let mask = array_tex(
            "layer mask",
            MASK_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("composite atlas"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: ATLAS_UNORM,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST, // receives the bled result
            view_formats: &[ATLAS_SRGB],
        });
        let array_view = |t: &wgpu::Texture| {
            t.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
        let color_view = array_view(&color);
        let mask_view = array_view(&mask);
        let atlas_unorm_view = atlas.create_view(&wgpu::TextureViewDescriptor {
            format: Some(ATLAS_UNORM),
            ..Default::default()
        });
        let atlas_srgb_view = atlas.create_view(&wgpu::TextureViewDescriptor {
            format: Some(ATLAS_SRGB),
            ..Default::default()
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("composite params"),
            size: (count as u64) * std::mem::size_of::<LayerParam>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Palette + quant start "disabled" (composite passes through) until `set_quantize`.
        let palette = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("composite palette"),
            contents: &vec![0u8; PALETTE_CAP * 16],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let quant = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("composite quant"),
            contents: bytemuck::bytes_of(&QuantU {
                enabled: 0,
                dither: 0,
                strength: 0.0,
                palette_len: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite bind group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&mask_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: palette.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: quant.as_entire_binding(),
                },
            ],
        });
        // Bleed scratch: two colour+validity ping buffers and the static coverage mask.
        let ping_tex = |label, format, extra| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | extra,
                view_formats: &[],
            })
        };
        let ping_color = [
            ping_tex("bleed ping0 color", COLOR_FORMAT, wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST),
            ping_tex("bleed ping1 color", COLOR_FORMAT, wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST),
        ];
        let ping_valid = [
            ping_tex("bleed ping0 valid", MASK_FORMAT, wgpu::TextureUsages::COPY_DST),
            ping_tex("bleed ping1 valid", MASK_FORMAT, wgpu::TextureUsages::COPY_DST),
        ];
        let coverage_tex = ping_tex(
            "bleed coverage",
            MASK_FORMAT,
            wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
        );
        let plain_view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
        let ping_color_view = [plain_view(&ping_color[0]), plain_view(&ping_color[1])];
        let ping_valid_view = [plain_view(&ping_valid[0]), plain_view(&ping_valid[1])];
        let bleed_bg = |ci: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bleed src bind group"),
                layout: &self.bleed_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&ping_color_view[ci]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&ping_valid_view[ci]),
                    },
                ],
            })
        };
        let bleed_src_bg = [bleed_bg(0), bleed_bg(1)];

        let stroke_base = ping_tex(
            "stroke base",
            COLOR_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        let stroke_base_view = plain_view(&stroke_base);

        self.resident = Some(Resident {
            size,
            count,
            color,
            mask,
            atlas,
            atlas_srgb_view,
            atlas_unorm_view,
            params,
            palette,
            quant,
            bind_group,
            ping_color,
            ping_valid,
            ping_color_view,
            ping_valid_view,
            coverage_tex,
            bleed_src_bg,
            stroke_base,
            stroke_base_view,
        });
    }

    /// Mirror the whole CPU `Layers` into the GPU arrays + params buffer. Uploads every
    /// slice (simple + correct; per-slice incremental upload is a later optimization).
    /// `layers` must have a uniform size across the stack (it always does).
    ///
    /// The colour slice is each layer's **effected** pixels (`Layer::effected`) — the
    /// non-destructive adjustment stack applied — so the GPU composite matches the CPU
    /// `composite` for layers with effects too. Effects stay CPU-computed but memoized
    /// (off the paint hot path; the active layer usually has none), so this is cheap.
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, layers: &Layers) {
        let size = layers.size();
        let count = layers.layers.len() as u32;
        self.ensure(device, size, count);
        let r = self.resident.as_ref().unwrap();

        let mut params = Vec::with_capacity(count as usize);
        for (i, layer) in layers.layers.iter().enumerate() {
            // Colour slice: the layer's effected RGBA8 (raw pixels when no effect is active).
            let eff = layer.effected();
            write_slice(queue, &r.color, i as u32, size, &eff, 4);
            // Mask slice: the red channel only, repacked to R8.
            let red: Vec<u8> = layer.mask.pixels.iter().step_by(4).copied().collect();
            write_slice(queue, &r.mask, i as u32, size, &red, 1);
            params.push(LayerParam {
                opacity: layer.opacity,
                blend: blend_code(layer.blend),
                visible: (layer.visible && layer.opacity > 0.0) as u32,
                _pad: 0,
            });
        }
        queue.write_buffer(&r.params, 0, bytemuck::cast_slice(&params));
    }

    /// Upload only the active layer's slice (colour + mask). During a stroke only the active
    /// layer changes, so this keeps its GPU slice current for the resolve base without
    /// re-uploading the whole stack (the per-stroke cost that scales with resolution × layers).
    /// Falls back to a full `upload` when the residency is missing or stale (first use / a
    /// resolution or layer-count change), so the params + non-active slices get initialized.
    pub fn upload_active(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, layers: &Layers) {
        let size = layers.size();
        let count = layers.layers.len() as u32;
        let stale = self
            .resident
            .as_ref()
            .is_none_or(|r| r.size != size || r.count != count);
        if stale {
            self.upload(device, queue, layers);
            return;
        }
        let r = self.resident.as_ref().unwrap();
        let i = layers.active;
        let layer = &layers.layers[i];
        let eff = layer.effected();
        write_slice(queue, &r.color, i as u32, r.size, &eff, 4);
        let red: Vec<u8> = layer.mask.pixels.iter().step_by(4).copied().collect();
        write_slice(queue, &r.mask, i as u32, r.size, &red, 1);
    }

    /// Read one rect of the active layer's colour slice back to RGBA8 (row-major within the
    /// rect) — the bounded stroke-end reconcile (only the painted region, not the whole slice).
    pub fn read_layer_rect(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        active: u32,
        rect: crate::paint::TexRect,
    ) -> Option<Vec<u8>> {
        let r = self.resident.as_ref()?;
        Some(read_array_rect(device, queue, &r.color, active, rect))
    }

    /// Run the composite pass into the atlas (whole atlas). `upload` must have run.
    pub fn composite(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.composite_inner(device, queue, None);
    }

    /// Composite only `scissor` (x, y, w, h) of the atlas, preserving the rest (`LoadOp::Load`)
    /// — the per-frame dirty-rect path during a stroke, so cost scales with brush area, not
    /// atlas size. Correct because the resolve only changed the active slice inside this rect,
    /// so the composite output changes only here.
    pub fn composite_region(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scissor: (u32, u32, u32, u32),
    ) {
        self.composite_inner(device, queue, Some(scissor));
    }

    fn composite_inner(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scissor: Option<(u32, u32, u32, u32)>,
    ) {
        let r = self.resident.as_ref().expect("GpuLayers::composite before upload");
        let load = if scissor.is_some() {
            wgpu::LoadOp::Load
        } else {
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("composite encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &r.atlas_unorm_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &r.bind_group, &[]);
            if let Some((x, y, w, h)) = scissor {
                pass.set_scissor_rect(x, y, w, h);
            }
            pass.draw(0..3, 0..1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Push the active palette + dither settings the composite folds in. With no palette
    /// (or `enabled == false`) the composite passes colour through unquantized. `upload`
    /// must have run (so the residency exists). Colours beyond `PALETTE_CAP` are dropped.
    pub fn set_quantize(
        &self,
        queue: &wgpu::Queue,
        palette: &crate::palette::Palette,
        enabled: bool,
        dither: bool,
        strength: f32,
    ) {
        let Some(r) = self.resident.as_ref() else {
            return;
        };
        let len = palette.colors.len().min(PALETTE_CAP);
        let mut buf = vec![[0.0f32; 4]; len];
        for (dst, c) in buf.iter_mut().zip(&palette.colors) {
            *dst = [c[0], c[1], c[2], 0.0];
        }
        if !buf.is_empty() {
            queue.write_buffer(&r.palette, 0, bytemuck::cast_slice(&buf));
        }
        let q = QuantU {
            enabled: (enabled && len > 0) as u32,
            dither: dither as u32,
            strength,
            palette_len: len as u32,
        };
        queue.write_buffer(&r.quant, 0, bytemuck::bytes_of(&q));
    }

    /// Upload the static UV-coverage mask (`bleed::coverage`) the gutter bleed grows from.
    /// One byte per texel (255 covered / 0 gutter). `upload` must have run.
    pub fn set_coverage(&self, queue: &wgpu::Queue, covered: &[bool]) {
        let Some(r) = self.resident.as_ref() else {
            return;
        };
        let bytes: Vec<u8> = covered.iter().map(|&c| if c { 255 } else { 0 }).collect();
        write_slice(queue, &r.coverage_tex, 0, r.size, &bytes, 1);
    }

    /// Dilate the composite atlas into the UV gutter by `pad` rings (the GPU analogue of
    /// `bleed::dilate`): seed a ping buffer from the atlas + coverage, ping-pong `pad`
    /// passes, copy the result back into the atlas. No-op for `pad == 0`. `composite` and
    /// `set_coverage` must have run. One submit.
    pub fn bleed(&self, device: &wgpu::Device, queue: &wgpu::Queue, pad: u32) {
        if pad == 0 {
            return;
        }
        let r = self.resident.as_ref().expect("GpuLayers::bleed before upload");
        let size = r.size;
        let extent = wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        };
        let copy = |enc: &mut wgpu::CommandEncoder, src: &wgpu::Texture, dst: &wgpu::Texture| {
            enc.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: src,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTexture {
                    texture: dst,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                extent,
            );
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bleed encoder"),
        });
        // Seed ping[0] from the composite atlas + the coverage mask.
        copy(&mut encoder, &r.atlas, &r.ping_color[0]);
        copy(&mut encoder, &r.coverage_tex, &r.ping_valid[0]);
        for ring in 0..pad as usize {
            let src = ring % 2;
            let dst = (ring + 1) % 2;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bleed pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &r.ping_color_view[dst],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &r.ping_valid_view[dst],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.bleed_pipeline);
            pass.set_bind_group(0, &r.bleed_src_bg[src], &[]);
            pass.draw(0..3, 0..1);
            drop(pass);
        }
        copy(&mut encoder, &r.ping_color[pad as usize % 2], &r.atlas);
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Resolve a stroke's coverage into a colour `target`: `out = blend(base, color, cov)`
    /// (or `erase(base, cov)`) — the GPU port of `apply_coverage`/`blend4`/`erase4`. Reads
    /// the immutable pre-stroke `base` + the dab pass's `coverage`, so it's idempotent and
    /// re-resolving the dirty region each frame composes correctly. Generic over the views
    /// (renderer targets the active layer slice; tests use standalone textures). `scissor`
    /// (x, y, w, h) bounds the work to the dab's rect. One submit.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &wgpu::TextureView,
        base: &wgpu::TextureView,
        coverage: &wgpu::TextureView,
        kind: ResolveKind,
        scissor: Option<(u32, u32, u32, u32)>,
    ) {
        let (mode, color, tile) = match kind {
            ResolveKind::Color(c) => (0u32, [c[0], c[1], c[2], 0.0], 0.0),
            ResolveKind::Erase => (1u32, [0.0; 4], 0.0),
            ResolveKind::Material(t) => (2u32, [0.0; 4], t),
        };
        let u = ResolveU {
            color,
            mode,
            tile,
            _pad: [0; 2],
        };
        let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("resolve uniform"),
            contents: bytemuck::bytes_of(&u),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let material_view = self.material.as_ref().unwrap_or(&self.dummy_material_view);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resolve bind group"),
            layout: &self.resolve_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(base),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(coverage),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(material_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ubuf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("resolve encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("resolve pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Load: only the scissor region is rewritten; the rest of the slice
                        // (already equal to `base`) is preserved.
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.resolve_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            if let Some((x, y, w, h)) = scissor {
                pass.set_scissor_rect(x, y, w, h);
            }
            pass.draw(0..3, 0..1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Snapshot the active layer's colour slice into `stroke_base` (the immutable resolve
    /// base). Call at stroke start, after `upload` has mirrored the current layers.
    pub fn begin_stroke_resolve(&self, device: &wgpu::Device, queue: &wgpu::Queue, active: u32) {
        let Some(r) = self.resident.as_ref() else {
            return;
        };
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stroke base copy"),
        });
        enc.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &r.color,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: active },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &r.stroke_base,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: r.size,
                height: r.size,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(enc.finish()));
    }

    /// Resolve the stroke `coverage` into the active layer's colour slice (no readback),
    /// blending the solid `color` (or erasing) over `stroke_base`. `scissor` bounds the
    /// work to the dab's rect. `begin_stroke_resolve` must have run.
    pub fn resolve_active(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        coverage: &wgpu::TextureView,
        active: u32,
        kind: ResolveKind,
        scissor: Option<(u32, u32, u32, u32)>,
    ) {
        let Some(r) = self.resident.as_ref() else {
            return;
        };
        let slice = r.color.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: active,
            array_layer_count: Some(1),
            ..Default::default()
        });
        self.resolve(
            device,
            queue,
            &slice,
            &r.stroke_base_view,
            coverage,
            kind,
            scissor,
        );
    }

    /// Read the active layer's colour slice back as RGBA8 (row-major) — used at stroke end
    /// to reconcile the GPU-resolved pixels into the authoritative CPU `Layers`.
    pub fn read_layer(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        active: u32,
    ) -> Option<Vec<u8>> {
        let r = self.resident.as_ref()?;
        Some(read_array_slice(device, queue, &r.color, active, r.size))
    }

    /// The sRGB view of the composite atlas, for binding into main.wgsl in place of the
    /// old single paint texture. `None` until the first `upload`.
    pub fn atlas_srgb_view(&self) -> Option<&wgpu::TextureView> {
        self.resident.as_ref().map(|r| &r.atlas_srgb_view)
    }

    /// The composite atlas texture itself (Rgba8Unorm), so the renderer can overlay a region
    /// directly — the brush-preview ghost, which under GPU display must paint into the atlas
    /// the model samples (not the unused CPU `paint_texture_gpu`). Raw RGBA8 bytes written via
    /// the Unorm format are reinterpreted by the sRGB view exactly as `bleed`'s output is, so
    /// an overlaid region matches the composite's byte math. `None` until the first `upload`.
    pub fn atlas_texture(&self) -> Option<&wgpu::Texture> {
        self.resident.as_ref().map(|r| &r.atlas)
    }

    /// Read the composite atlas back as RGBA8 (row-major) — the export path, and the
    /// parity-test oracle. Blocking.
    #[cfg(test)]
    pub fn read_atlas(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u8> {
        let r = self.resident.as_ref().expect("read_atlas before upload");
        let size = r.size;
        let unpadded = size * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atlas readback"),
            size: (padded * size) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("atlas readback encoder"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &r.atlas,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(size),
                },
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("map atlas readback");
        let data = slice.get_mapped_range();
        let mut out = vec![0u8; (unpadded * size) as usize];
        for y in 0..size as usize {
            let s = y * padded as usize;
            let d = y * unpadded as usize;
            out[d..d + unpadded as usize].copy_from_slice(&data[s..s + unpadded as usize]);
        }
        drop(data);
        buf.unmap();
        out
    }
}

/// Read one RGBA8 array slice (`layer`) of `tex` back into tightly-packed bytes. Blocking;
/// used once per stroke at `end_stroke` to reconcile the GPU-resolved layer into the CPU
/// store (off the paint hot path).
fn read_array_slice(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex: &wgpu::Texture,
    layer: u32,
    size: u32,
) -> Vec<u8> {
    let unpadded = size * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layer readback"),
        size: (padded * size) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("layer readback encoder"),
    });
    enc.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &buf,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(size),
            },
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(enc.finish()));
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().expect("map layer readback");
    let data = slice.get_mapped_range();
    let mut out = vec![0u8; (unpadded * size) as usize];
    for y in 0..size as usize {
        let s = y * padded as usize;
        let d = y * unpadded as usize;
        out[d..d + unpadded as usize].copy_from_slice(&data[s..s + unpadded as usize]);
    }
    drop(data);
    buf.unmap();
    out
}

/// Read a rect of one RGBA8 array slice (`layer`) back into tightly-packed bytes
/// (`rect.width()*rect.height()*4`, row-major). Bounds the stroke-end reconcile readback to
/// the painted region. Blocking.
fn read_array_rect(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex: &wgpu::Texture,
    layer: u32,
    rect: crate::paint::TexRect,
) -> Vec<u8> {
    let (rw, rh) = (rect.width(), rect.height());
    let unpadded = rw * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layer rect readback"),
        size: (padded * rh) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("layer rect readback encoder"),
    });
    enc.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: rect.x0,
                y: rect.y0,
                z: layer,
            },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &buf,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(rh),
            },
        },
        wgpu::Extent3d {
            width: rw,
            height: rh,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(enc.finish()));
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().expect("map layer rect readback");
    let data = slice.get_mapped_range();
    let mut out = vec![0u8; (unpadded * rh) as usize];
    for y in 0..rh as usize {
        let s = y * padded as usize;
        let d = y * unpadded as usize;
        out[d..d + unpadded as usize].copy_from_slice(&data[s..s + unpadded as usize]);
    }
    drop(data);
    buf.unmap();
    out
}

/// Upload one array slice (`layer`) of `tex` from a tightly-packed `pixels` buffer of
/// `bpp` bytes/texel. `queue.write_texture` has no row-alignment requirement.
fn write_slice(
    queue: &wgpu::Queue,
    tex: &wgpu::Texture,
    layer: u32,
    size: u32,
    pixels: &[u8],
    bpp: u32,
) {
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
            aspect: wgpu::TextureAspect::All,
        },
        pixels,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(size * bpp),
            rows_per_image: Some(size),
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paint::Texture;

    /// The GPU composite must reproduce the CPU `Layers::composite` within ±1 (u8 encode
    /// rounding). Build a multi-layer stack exercising every blend mode, a partial mask,
    /// partial opacity and a hidden layer, then diff the readback against the CPU oracle.
    #[test]
    fn gpu_composite_matches_cpu_layers() {
        let (device, queue) = crate::renderer::new_test_device();
        let size = 16u32;

        let mut layers = Layers::new(Texture::new(size, size, [200, 50, 25, 255]));
        // A Multiply layer with a varying mask + partial alpha.
        layers.add_layer();
        for (t, px) in layers.active_tex_mut().pixels.chunks_exact_mut(4).enumerate() {
            px.copy_from_slice(&[(t * 3) as u8, 100, 255u8.wrapping_sub(t as u8), 180]);
        }
        layers.layers[1].blend = BlendMode::Multiply;
        for (t, px) in layers.layers[1].mask.pixels.chunks_exact_mut(4).enumerate() {
            let v = ((t * 7) % 256) as u8;
            px.copy_from_slice(&[v, v, v, 255]);
        }
        // An Add layer at 0.6 opacity.
        layers.add_layer();
        for px in layers.active_tex_mut().pixels.chunks_exact_mut(4) {
            px.copy_from_slice(&[40, 60, 90, 200]);
        }
        layers.layers[2].blend = BlendMode::Add;
        layers.layers[2].opacity = 0.6;
        // A Screen layer that is hidden (must contribute nothing).
        layers.add_layer();
        for px in layers.active_tex_mut().pixels.chunks_exact_mut(4) {
            px.copy_from_slice(&[180, 180, 60, 255]);
        }
        layers.layers[3].blend = BlendMode::Screen;
        layers.layers[3].visible = false;

        let cpu = layers.composite();

        let mut gpu = GpuLayers::new(&device);
        gpu.upload(&device, &queue, &layers);
        gpu.composite(&device, &queue);
        let got = gpu.read_atlas(&device, &queue);

        assert_eq!(got.len(), cpu.len());
        let mut max_d = 0i32;
        for (a, b) in got.iter().zip(&cpu) {
            max_d = max_d.max((*a as i32 - *b as i32).abs());
        }
        assert!(max_d <= 1, "GPU composite diverges from CPU by {max_d} (> 1)");
    }

    /// With a palette + Bayer dither on, the GPU composite must match the CPU
    /// `composite` + `quantize_rgba` *byte-exactly* on almost every texel: the composite
    /// output snaps to an exact palette byte, so the only divergence is the rare argmin
    /// flip where a ULP of drift crosses an equidistant pair. Assert a tiny flip budget,
    /// and that non-flipped texels are identical.
    #[test]
    fn gpu_composite_quantize_matches_cpu() {
        use crate::palette::Palette;
        let (device, queue) = crate::renderer::new_test_device();
        let size = 32u32;

        let mut layers = Layers::new(Texture::new(size, size, [200, 50, 25, 255]));
        layers.add_layer();
        for (t, px) in layers.active_tex_mut().pixels.chunks_exact_mut(4).enumerate() {
            px.copy_from_slice(&[(t * 3) as u8, 100, 255u8.wrapping_sub(t as u8), 200]);
        }
        layers.layers[1].blend = BlendMode::Add;

        let palette = Palette::builtins().remove(0); // PICO-8 (16)
        let (dither, strength) = (true, 0.06f32);

        // CPU oracle: composite then quantize.
        let mut cpu = layers.composite();
        palette.quantize_rgba(&mut cpu, size, dither, strength);

        let mut gpu = GpuLayers::new(&device);
        gpu.upload(&device, &queue, &layers);
        gpu.set_quantize(&queue, &palette, true, dither, strength);
        gpu.composite(&device, &queue);
        let got = gpu.read_atlas(&device, &queue);

        let mut flips = 0u32;
        let n = (size * size) as usize;
        for t in 0..n {
            let i = t * 4;
            if got[i..i + 3] != cpu[i..i + 3] {
                flips += 1;
            }
            // Alpha is never quantized — must match within u8 rounding.
            assert!((got[i + 3] as i32 - cpu[i + 3] as i32).abs() <= 1);
        }
        assert!(
            (flips as f32) < 0.01 * n as f32,
            "too many palette argmin flips: {flips} of {n}"
        );
    }

    /// Read a standalone Rgba8 texture back to RGBA8 bytes (test helper).
    fn read_rgba8(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tex: &wgpu::Texture,
        size: u32,
    ) -> Vec<u8> {
        let unpadded = size * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (padded * size) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(size),
                },
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(enc.finish()));
        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let mut out = vec![0u8; (unpadded * size) as usize];
        for y in 0..size as usize {
            let s = y * padded as usize;
            let d = y * unpadded as usize;
            out[d..d + unpadded as usize].copy_from_slice(&data[s..s + unpadded as usize]);
        }
        drop(data);
        buf.unmap();
        out
    }

    /// The GPU resolve must reproduce `paint::blend4` (and `erase4`): given a pre-stroke
    /// base + a coverage field, `out = blend(base, color, cov)` / `erase(base, cov)` within
    /// ±1 (u8 rounding). This is the no-readback paint path's core.
    #[test]
    fn gpu_resolve_matches_cpu_blend() {
        let (device, queue) = crate::renderer::new_test_device();
        let size = 16u32;
        let n = (size * size) as usize;

        // Base: a varying opaque-ish field.
        let mut base_px = vec![0u8; n * 4];
        for (t, px) in base_px.chunks_exact_mut(4).enumerate() {
            px.copy_from_slice(&[(t * 7) as u8, 80, 255u8.wrapping_sub(t as u8), 200]);
        }
        // Coverage: a 0..1 ramp.
        let cov: Vec<f32> = (0..n).map(|t| (t as f32 / n as f32)).collect();

        let mk_tex = |fmt, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: fmt,
                usage,
                view_formats: &[],
            })
        };
        let base = mk_tex(
            COLOR_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        write_slice(&queue, &base, 0, size, &base_px, 4);
        let cov_tex = mk_tex(
            wgpu::TextureFormat::R32Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        write_slice(&queue, &cov_tex, 0, size, bytemuck::cast_slice(&cov), 4);
        let base_view = base.create_view(&Default::default());
        let cov_view = cov_tex.create_view(&Default::default());

        let gpu = GpuLayers::new(&device);
        for (erase, color) in [(false, [255u8, 0, 0]), (true, [0, 0, 0])] {
            let target = mk_tex(
                COLOR_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            );
            let target_view = target.create_view(&Default::default());
            let col = [
                color[0] as f32 / 255.0,
                color[1] as f32 / 255.0,
                color[2] as f32 / 255.0,
            ];
            let kind = if erase {
                ResolveKind::Erase
            } else {
                ResolveKind::Color(col)
            };
            gpu.resolve(&device, &queue, &target_view, &base_view, &cov_view, kind, None);
            let got = read_rgba8(&device, &queue, &target, size);

            // CPU oracle: blend4 / erase4 per texel.
            let mut want = base_px.clone();
            for t in 0..n {
                let i = t * 4;
                let a = cov[t];
                if a <= 0.0 {
                    continue;
                }
                let (mut dst, b) = ([0u8; 4], &base_px[i..i + 4]);
                dst.copy_from_slice(b);
                if erase {
                    crate::paint::erase4(&mut dst, b, a);
                } else {
                    crate::paint::blend4(&mut dst, b, color, a);
                }
                want[i..i + 4].copy_from_slice(&dst);
            }
            let mut max_d = 0i32;
            for (a, b) in got.iter().zip(&want) {
                max_d = max_d.max((*a as i32 - *b as i32).abs());
            }
            assert!(max_d <= 1, "GPU resolve (erase={erase}) diverges by {max_d}");
        }
    }

    /// A layer carrying a (non-destructive) effect must composite the same on the GPU as
    /// on the CPU: `upload` feeds the GPU each layer's *effected* pixels, so the GPU
    /// composite of an effected stack matches `Layers::composite` (which runs the same
    /// effect) within ±1.
    #[test]
    fn gpu_composite_with_effect_matches_cpu() {
        let (device, queue) = crate::renderer::new_test_device();
        let size = 16u32;

        let mut layers = Layers::new(Texture::new(size, size, [200, 40, 40, 255]));
        // A full desaturate on the base — the composite should be grey, not red.
        layers.layers[0].effects.push(crate::effects::Effect::HueSatLight {
            hue: 0.0,
            sat: -1.0,
            light: 0.0,
        });
        layers.layers[0].invalidate();

        let cpu = layers.composite();

        let mut gpu = GpuLayers::new(&device);
        gpu.upload(&device, &queue, &layers);
        gpu.composite(&device, &queue);
        let got = gpu.read_atlas(&device, &queue);

        // The effect must actually have applied (grey), and GPU == CPU within ±1.
        assert!(cpu[0] == cpu[1] && cpu[1] == cpu[2], "CPU base not desaturated");
        let mut max_d = 0i32;
        for (a, b) in got.iter().zip(&cpu) {
            max_d = max_d.max((*a as i32 - *b as i32).abs());
        }
        assert!(max_d <= 1, "GPU effected composite diverges by {max_d}");
    }

    /// The GPU gutter bleed must be byte-identical to `bleed::dilate`. Isolate the bleed
    /// stage: composite on the GPU, read the atlas back as the *source*, run the CPU
    /// dilate on that exact buffer, and assert the GPU bleed of the same source +
    /// coverage matches it exactly (same algorithm, same inputs — no float drift).
    #[test]
    fn gpu_bleed_matches_cpu_dilate() {
        let (device, queue) = crate::renderer::new_test_device();
        let size = 32u32;
        let pad = 4u32;

        // A vivid base so dilated colours are obvious.
        let mut layers = Layers::new(Texture::new(size, size, [10, 200, 60, 255]));
        for (t, px) in layers.active_tex_mut().pixels.chunks_exact_mut(4).enumerate() {
            px.copy_from_slice(&[(t * 5) as u8, 200, 60, 255]);
        }

        // Coverage: a centred block is covered, the surrounding gutter must fill from it.
        let mut covered = vec![false; (size * size) as usize];
        for y in 8..24u32 {
            for x in 8..24u32 {
                covered[(y * size + x) as usize] = true;
            }
        }

        let mut gpu = GpuLayers::new(&device);
        gpu.upload(&device, &queue, &layers);
        gpu.set_coverage(&queue, &covered);
        gpu.composite(&device, &queue);
        let pre = gpu.read_atlas(&device, &queue); // the bleed source

        gpu.bleed(&device, &queue, pad);
        let got = gpu.read_atlas(&device, &queue);

        let mut cpu = pre.clone();
        crate::bleed::dilate(&mut cpu, &covered, size, pad);

        assert_eq!(got, cpu, "GPU bleed diverges from CPU dilate");
    }
}

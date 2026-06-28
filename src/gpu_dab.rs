// src/gpu_dab.rs
//
// GPU dab stamping (Phase 1, option B). `surface::splat_faces` floods mesh adjacency to
// the face set within a dab's surface radius (cheap, face-granular); this module
// rasterizes exactly those faces into a coverage texture with the per-fragment surface
// falloff (`shaders/dab.wgsl`), reproducing `surface::splat`'s cross-seam coverage with
// the former ~95 %-of-paint per-texel cost moved to the GPU.
//
// `coverage` returns the per-texel coverage map a single dab produces, which the parity
// test diffs against the CPU `surface::splat` within a small tolerance — proving the
// "surface brush in texture space, on the GPU" hard problem. The renderer can feed this
// map straight into the existing `deposit` stroke discipline (gated by LOWTEX_GPU_PAINT),
// reusing all the tested max-coverage / dirty-rect / undo machinery unchanged.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use glam::{Vec2, Vec3};
use wgpu::util::DeviceExt;

use crate::mesh::Mesh;
use crate::paint::TexRect;
use crate::surface;

/// One dab vertex: the atlas UV (mapped to clip space), the face's world position at
/// that vertex (interpolated per fragment for the distance falloff), and the dab's
/// own parameters carried per-vertex. The params are constant across a face's three
/// vertices, so carrying them on the vertex (instead of a per-dab uniform) lets a
/// whole frame's dabs — each with its own centre/radius/opacity/hardness — rasterize
/// in a *single* draw call, collapsing the former one-submit-per-dab overhead.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DabVertex {
    uv: [f32; 2],
    world: [f32; 3],
    center: [f32; 3],
    radius: f32,
    opacity: f32,
    hardness: f32,
}

/// The coverage-texture format. R16Float is blendable (R32Float is not, in wgpu 22), and
/// half precision (~3-4 digits) is plenty for 0..1 coverage. `Max` blend accumulates the
/// per-stroke max coverage across dabs (mirroring the CPU `stroke_coverage` discipline),
/// so overlapping dabs don't double-count; readback decodes f16 → f32.
const COVERAGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

/// Decode an IEEE-754 binary16 (as a `u16`) to `f32` with pure integer bit-ops — no
/// `powi`, which matters because the readback decodes a whole dirty region per frame.
/// The two common cases (zero / normal) are branch-light and loop-free; only a tiny
/// nonzero subnormal (negligible coverage) takes the rare normalize loop.
fn f16_to_f32(h: u16) -> f32 {
    let h = h as u32;
    let sign = (h & 0x8000) << 16; // f16 bit15 → f32 bit31
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let bits = if exp == 0 {
        if mant == 0 {
            sign // ±0
        } else {
            // Subnormal f16 → normalized f32 (rare; only sub-6e-5 coverage).
            let mut e = 0i32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e += 1;
            }
            sign | (((113 - e) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13) // inf/NaN (unused for coverage)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13) // normal: bias 15 → 127
    };
    f32::from_bits(bits)
}

/// How far (in texels) to grow each dab triangle for conservative coverage. `LOWTEX_DAB_EXPAND`
/// overrides the default for tuning; half a texel covers the hardware's dropped boundary texels
/// with minimal over-paint.
fn dab_expand() -> f32 {
    std::env::var("LOWTEX_DAB_EXPAND")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.5)
}

/// Offset each edge of a UV triangle outward by `e` texels along its outward normal and
/// return the enlarged triangle (back in [0,1] UV space). Used to make the GPU dab raster
/// *conservative*: the hardware rasterizer only shades fragments whose centre is strictly
/// inside (subpixel fill rule), so it drops the very boundary texels — and sub-texel slivers
/// — that the CPU scanline (`surface::rasterize`, inclusive `w>=0`) keeps. Growing every
/// triangle by half a texel makes the hardware cover every texel the CPU would, killing the
/// unpainted "teeth" at island edges. Degenerate (near-zero-area) triangles are returned
/// unchanged; near-parallel sliver corners fall back to a centroid push so the offset can't
/// shoot off to infinity.
fn expand_tri(uv: [Vec2; 3], size: f32, e: f32) -> [Vec2; 3] {
    let p = [uv[0] * size, uv[1] * size, uv[2] * size];
    let area2 = (p[1] - p[0]).perp_dot(p[2] - p[0]); // twice the signed area
    if area2.abs() < 1e-4 {
        return uv; // degenerate in UV space — nothing meaningful to offset
    }
    let cen = (p[0] + p[1] + p[2]) / 3.0;
    // Each edge k (v[k] → v[k+1]) offset outward by `e`, stored as (point on line, direction).
    let mut lines = [(Vec2::ZERO, Vec2::ZERO); 3];
    for k in 0..3 {
        let a = p[k];
        let b = p[(k + 1) % 3];
        let dir = (b - a).normalize_or_zero();
        let mut n = Vec2::new(dir.y, -dir.x);
        if n.dot((a + b) * 0.5 - cen) < 0.0 {
            n = -n; // make the normal point away from the centroid (outward)
        }
        lines[k] = (a + n * e, dir);
    }
    // Acute corners need the vertex pushed out along the edge bisector by e/sin(half-angle),
    // which blows up for slivers — so clamp the *magnitude* but keep the (correct) bisector
    // direction, rather than collapsing to a weak centroid push that leaves the corner texel
    // uncovered (the last stubborn "teeth" at sharp island corners).
    let cap = 8.0 * e + 2.0;
    let mut out = uv;
    for k in 0..3 {
        // Vertex k is where the offset of edge (k-1) and edge k meet.
        let (p0, d0) = lines[(k + 2) % 3];
        let (p1, d1) = lines[k];
        let nv = match line_intersect(p0, d0, p1, d1) {
            Some(x) => {
                let off = x - p[k];
                if off.length() > cap {
                    p[k] + off.normalize_or_zero() * cap
                } else {
                    x
                }
            }
            // Edges (near-)parallel: fall back to a straight outward push from the centroid.
            None => p[k] + (p[k] - cen).normalize_or_zero() * cap,
        };
        out[k] = nv / size;
    }
    out
}

/// Barycentric coordinates of point `e` with respect to triangle `t` (scale-invariant, so it
/// works directly in UV space). Used to extrapolate the original triangle's affine UV→world map
/// to the conservatively-expanded vertices.
fn bary(e: Vec2, t: &[Vec2; 3]) -> [f32; 3] {
    let area2 = (t[1] - t[0]).perp_dot(t[2] - t[0]);
    if area2.abs() < 1e-12 {
        return [1.0, 0.0, 0.0];
    }
    let b0 = (t[1] - e).perp_dot(t[2] - e) / area2;
    let b1 = (t[2] - e).perp_dot(t[0] - e) / area2;
    [b0, b1, 1.0 - b0 - b1]
}

/// Intersection of two lines given as (point, direction). `None` when (near-)parallel.
fn line_intersect(p0: Vec2, d0: Vec2, p1: Vec2, d1: Vec2) -> Option<Vec2> {
    let denom = d0.perp_dot(d1);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (p1 - p0).perp_dot(d1) / denom;
    Some(p0 + d0 * t)
}

/// A persistent per-stroke coverage texture the dabs of a stroke `Max`-accumulate into,
/// plus the size it was built for so a resolution change rebuilds it.
struct StrokeTarget {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    size: u32,
}

/// An in-flight coverage readback: the GPU copy has been submitted and the buffer is being
/// mapped asynchronously. `done` flips true in the map callback once the GPU finishes, so
/// the renderer can pick it up a frame later *without blocking* — the fix for the per-frame
/// `poll(Wait)` stall that dominated paint cost. `rect` is the region it covers.
struct Pending {
    buffer: wgpu::Buffer,
    rect: TexRect,
    padded: u32, // 256-aligned bytes per row
    rw: u32,
    rh: u32,
    done: Arc<AtomicBool>,
}

/// Owns the dab render pipeline (built once), a per-stroke coverage target, and an in-flight
/// readback. Driven two ways:
///  - `coverage` (one-shot): rasterize a face set into a fresh texture and read it back —
///    the parity-test path that proves the GPU dab matches `surface::splat`.
///  - `begin_stroke` / `stamp` / `issue_readback` / `try_take_readback` / `drain_readback`
///    (accumulating, pipelined): the paint hot path — stamp many dabs into one persistent
///    target, then read each frame's dirty region back *asynchronously* (one frame of
///    latency) so the CPU never stalls on the GPU mid-stroke.
pub struct GpuDab {
    pipeline: wgpu::RenderPipeline,
    /// This frame's accumulated dab geometry (3 verts/face, per-vertex params). Stamps
    /// append here; `flush_dabs` rasterizes the whole batch in one pass + one submit.
    batch: Vec<DabVertex>,
    stroke: Option<StrokeTarget>,
    pending: Option<Pending>,
}

impl GpuDab {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dab shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/dab.wgsl").into()),
        });
        // No bind groups: every per-dab parameter rides on the vertex now.
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dab pipeline layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<DabVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // uv
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // world
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // center
                wgpu::VertexAttribute {
                    offset: 20,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // radius
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
                },
                // opacity
                wgpu::VertexAttribute {
                    offset: 36,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32,
                },
                // hardness
                wgpu::VertexAttribute {
                    offset: 40,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dab pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: COVERAGE_FORMAT,
                    // Max blend: a texel's coverage is the max over all dabs that touch it
                    // (the per-stroke max-coverage discipline), so overlap never darkens.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Max,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Max,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // UV winding is arbitrary across faces, so don't cull — every face must
                // rasterize regardless of orientation in UV space.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        Self {
            pipeline,
            batch: Vec::new(),
            stroke: None,
            pending: None,
        }
    }

    /// Build the dab geometry — 3 (uv, world, params) vertices per face — for `faces`,
    /// tagging every vertex with this dab's centre/radius/opacity/hardness.
    fn dab_verts<'a>(
        mesh: &'a Mesh,
        faces: &'a [u32],
        center: Vec3,
        radius: f32,
        opacity: f32,
        hardness: f32,
        size: u32,
    ) -> impl Iterator<Item = DabVertex> + 'a {
        let center = [center.x, center.y, center.z];
        let sz = size as f32;
        faces.iter().flat_map(move |&f| {
            let (p, uv) = surface::tri_data(mesh, f);
            // Expand the UV triangle outward by half a texel so the hardware rasterizer covers
            // every texel whose centre lies in the original triangle — its subpixel fill rule
            // otherwise drops boundary texels (and sub-texel slivers) that the CPU scanline
            // (`surface::rasterize`, `w>=0`) keeps, leaving unpainted dark teeth at island edges.
            // World stays the *original* vertex position: the per-vertex `world` attribute is
            // linear in barycentrics, so interpolating it across the enlarged triangle simply
            // extrapolates the surface point correctly, and the falloff still discards anything
            // past the radius (no over-paint).
            let euv = expand_tri(uv, sz, dab_expand());
            (0..3).map(move |k| {
                // World for each (expanded) vertex via the *original* triangle's affine UV→world
                // map: this keeps the map identical for interior texels (so their world — and the
                // distance falloff — are unchanged), and gives the newly-covered edge texels the
                // correct linear extrapolation. Reusing the original vertex world instead would
                // stretch the map and shift the falloff disc, perturbing interior coverage.
                let b = bary(euv[k], &uv);
                let world = p[0] * b[0] + p[1] * b[1] + p[2] * b[2];
                DabVertex {
                    uv: [euv[k].x, euv[k].y],
                    world: [world.x, world.y, world.z],
                    center,
                    radius,
                    opacity,
                    hardness,
                }
            })
        })
    }

    /// Render `verts` into `view` with the given `load` op (Clear to start a fresh target,
    /// Load to `Max`-accumulate onto it). One submit; no readback. With no verts it's a
    /// bare clear/no-op pass (used to clear the stroke target at `begin_stroke`).
    fn submit_pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        load: wgpu::LoadOp<wgpu::Color>,
        verts: &[DabVertex],
    ) {
        let vbuf = (!verts.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dab verts"),
                contents: bytemuck::cast_slice(verts),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dab encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dab pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
            if let Some(vbuf) = &vbuf {
                pass.set_pipeline(&self.pipeline);
                pass.set_vertex_buffer(0, vbuf.slice(..));
                pass.draw(0..verts.len() as u32, 0..1);
            }
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Start a stroke's GPU coverage accumulation: (re)create the persistent coverage
    /// target at `size` and clear it to 0. Subsequent `stamp`s `Max`-accumulate into it.
    pub fn begin_stroke(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, size: u32) {
        // Drop any readback still in flight from a previous stroke — its coverage is stale.
        self.pending = None;
        self.batch.clear();
        let need = self.stroke.as_ref().is_none_or(|s| s.size != size);
        if need {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("stroke coverage"),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: COVERAGE_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING, // sampled by the GPU paint resolve
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.stroke = Some(StrokeTarget { tex, view, size });
        }
        // Clear to 0 (a render pass with no draws).
        let view = &self.stroke.as_ref().unwrap().view;
        self.submit_pass(
            device,
            queue,
            view,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            &[], // no draws — just the clear
        );
    }

    /// Stamp one dab's `faces` into the persistent stroke target, `Max`-accumulating the
    /// per-fragment coverage. No readback — the paint hot path's per-dab cost is a draw.
    /// `begin_stroke` must have run.
    pub fn stamp(
        &mut self,
        mesh: &Mesh,
        faces: &[u32],
        center: Vec3,
        radius: f32,
        opacity: f32,
        hardness: f32,
        size: u32,
    ) {
        if faces.is_empty() || radius <= 0.0 {
            return;
        }
        self.batch
            .extend(Self::dab_verts(mesh, faces, center, radius, opacity, hardness, size));
    }

    /// Rasterize this frame's accumulated dab batch into the stroke target in one pass +
    /// one submit (`Max`-accumulating onto whatever is there), then clear the batch. The
    /// fix for the former one-submit-per-dab overhead: a fast drag's many dabs now cost a
    /// single draw. No-op when nothing was stamped. Must run before a coverage readback so
    /// the copy sees the frame's dabs.
    pub fn flush_dabs(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.batch.is_empty() {
            return;
        }
        let verts = std::mem::take(&mut self.batch);
        let view = match self.stroke.as_ref() {
            Some(s) => &s.view,
            None => return, // no target (stroke not begun) — drop the batch
        };
        self.submit_pass(device, queue, view, wgpu::LoadOp::Load, &verts);
    }

    /// Synchronous read of the accumulated coverage for `rect` (row-major, length
    /// `rect.width()*rect.height()`). Superseded on the hot path by the pipelined
    /// `issue_readback`/`try_take_readback`; kept as the parity-test path.
    #[cfg(test)]
    pub fn read_region(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rect: crate::paint::TexRect,
    ) -> Vec<f32> {
        self.flush_dabs(device, queue);
        let stroke = self
            .stroke
            .as_ref()
            .expect("GpuDab::read_region before begin_stroke");
        let (rw, rh) = (rect.width(), rect.height());
        if rw == 0 || rh == 0 {
            return Vec::new();
        }
        let unpadded = rw * 2; // 2 bytes/texel, R16Float
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stroke readback"),
            size: (padded * rh) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stroke readback encoder"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &stroke.tex,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: rect.x0,
                    y: rect.y0,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
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
        queue.submit(std::iter::once(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("failed to map stroke readback");
        let data = slice.get_mapped_range();
        let mut out = vec![0.0f32; (rw * rh) as usize];
        for y in 0..rh as usize {
            let row = &data[y * padded as usize..y * padded as usize + unpadded as usize];
            let row_h: &[u16] = bytemuck::cast_slice(row);
            for (x, &h) in row_h.iter().enumerate() {
                out[y * rw as usize + x] = f16_to_f32(h);
            }
        }
        drop(data);
        readback.unmap();
        out
    }

    /// Whether a coverage readback is in flight (its GPU copy submitted but not yet picked
    /// up). The renderer holds off issuing the next one until this clears.
    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// The per-stroke coverage texture's view, for the GPU paint resolve to sample (the
    /// no-readback path). `None` before `begin_stroke`.
    pub fn coverage_view(&self) -> Option<&wgpu::TextureView> {
        self.stroke.as_ref().map(|s| &s.view)
    }

    /// Submit a copy of `rect` of the stroke coverage to a fresh readback buffer and begin
    /// mapping it asynchronously — *no wait*. The result is collected a frame later by
    /// `try_take_readback`. This is what removes the per-frame `poll(Wait)` stall.
    pub fn issue_readback(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, rect: TexRect) {
        let stroke = match self.stroke.as_ref() {
            Some(s) => s,
            None => return,
        };
        let (rw, rh) = (rect.width(), rect.height());
        if rw == 0 || rh == 0 {
            return;
        }
        let unpadded = rw * 2; // 2 bytes/texel, R16Float
        let padded = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stroke readback (async)"),
            size: (padded * rh) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stroke readback encoder"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &stroke.tex,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: rect.x0,
                    y: rect.y0,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
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
        queue.submit(std::iter::once(encoder.finish()));

        let done = Arc::new(AtomicBool::new(false));
        let flag = done.clone();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |res| {
            if res.is_ok() {
                flag.store(true, Ordering::Release);
            }
        });
        self.pending = Some(Pending {
            buffer,
            rect,
            padded,
            rw,
            rh,
            done,
        });
    }

    /// Non-blocking: if the in-flight readback has finished on the GPU, decode and return
    /// it as `(coverage, rect)` (coverage row-major within `rect`); otherwise `None`. Pump
    /// the device's callbacks first (`Maintain::Poll` never blocks).
    pub fn try_take_readback(&mut self, device: &wgpu::Device) -> Option<(Vec<f32>, TexRect)> {
        device.poll(wgpu::Maintain::Poll);
        if !self.pending.as_ref()?.done.load(Ordering::Acquire) {
            return None;
        }
        let p = self.pending.take().unwrap();
        Some((Self::decode_pending(&p), p.rect))
    }

    /// Blocking drain (stroke end / final reconcile): wait for the in-flight readback, then
    /// decode and return it. `None` if nothing is in flight.
    pub fn drain_readback(&mut self, device: &wgpu::Device) -> Option<(Vec<f32>, TexRect)> {
        self.pending.as_ref()?;
        device.poll(wgpu::Maintain::Wait);
        let p = self.pending.take().unwrap();
        Some((Self::decode_pending(&p), p.rect))
    }

    /// Strip row padding + decode f16 → f32 from a finished readback buffer.
    fn decode_pending(p: &Pending) -> Vec<f32> {
        let data = p.buffer.slice(..).get_mapped_range();
        let (rw, rh, padded) = (p.rw as usize, p.rh as usize, p.padded as usize);
        let mut out = vec![0.0f32; rw * rh];
        for y in 0..rh {
            let row = &data[y * padded..y * padded + rw * 2];
            let row_h: &[u16] = bytemuck::cast_slice(row);
            for (x, &h) in row_h.iter().enumerate() {
                out[y * rw + x] = f16_to_f32(h);
            }
        }
        drop(data);
        p.buffer.unmap();
        out
    }

    /// Rasterize `faces` (from `surface::splat_faces`) into a `size`×`size` coverage map
    /// for a dab centred at `center` with world `radius`, `opacity`, `hardness`. Returns
    /// per-texel coverage, row-major (V down) — matching `surface::splat`'s texel order.
    /// Self-contained one-shot (creates + tears down its own target); the paint hot path
    /// uses `begin_stroke`/`stamp`/`read_region` instead, so this is the parity-test helper.
    #[cfg(test)]
    pub fn coverage(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &Mesh,
        faces: &[u32],
        center: Vec3,
        radius: f32,
        opacity: f32,
        hardness: f32,
        size: u32,
    ) -> Vec<f32> {
        let n = (size * size) as usize;
        if faces.is_empty() || radius <= 0.0 {
            return vec![0.0; n];
        }
        // Build the dab geometry: 3 (uv, world, params) vertices per face.
        let verts: Vec<DabVertex> =
            Self::dab_verts(mesh, faces, center, radius, opacity, hardness, size).collect();
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dab verts"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dab coverage"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COVERAGE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        // Readback row pitch must be 256-aligned (2 bytes/texel, R16Float).
        let unpadded = size * 2;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dab readback"),
            size: (padded * size) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dab encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dab pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), // r=0
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_vertex_buffer(0, vbuf.slice(..));
            pass.draw(0..verts.len() as u32, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
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

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("failed to map dab readback");
        let data = slice.get_mapped_range();

        // Strip the row padding and decode f16 → f32 into a tight size² map.
        let mut out = vec![0.0f32; n];
        for y in 0..size as usize {
            let row = &data[y * padded as usize..y * padded as usize + unpadded as usize];
            let row_h: &[u16] = bytemuck::cast_slice(row);
            for (x, &h) in row_h.iter().enumerate() {
                out[y * size as usize + x] = f16_to_f32(h);
            }
        }
        drop(data);
        readback.unmap();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bvh::Hit;
    use crate::paint::TexRect;
    use crate::surface::{Adjacency, SplatScratch};
    use glam::Vec2;

    /// The GPU dab coverage must reproduce the CPU `surface::splat` within a small
    /// tolerance: same falloff, same world-distance, same faces — only the GPU edge fill
    /// rule (vs the CPU `w >= 0` rasterizer) and float rounding can differ, and only on a
    /// thin rim of UV-boundary texels. Diff several dabs (soft/hard, partial opacity,
    /// cross-face) on the cube; assert tight mean error and few large outliers.
    #[test]
    fn gpu_dab_coverage_matches_cpu_splat() {
        let (device, queue) = crate::renderer::new_test_device();
        let dab = GpuDab::new(&device);
        let size = 128u32;
        let mesh = Mesh::cube();
        let adj = Adjacency::build(&mesh);
        let mut scratch = SplatScratch::new();

        for (tri, r, opacity, hardness) in [
            (0u32, 0.3f32, 1.0f32, 0.8f32),
            (0, 0.6, 0.7, 0.5),
            (3, 0.4, 1.0, 1.0),
            (7, 0.9, 1.0, 0.6), // wraps across faces
        ] {
            let (p, _) = surface::tri_data(&mesh, tri);
            let centroid = (p[0] + p[1] + p[2]) / 3.0;
            let normal = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
            let hit = Hit {
                uv: Vec2::ZERO,
                tri,
                pos: centroid,
                normal,
            };

            // CPU reference: max coverage per texel.
            let mut cpu = vec![0.0f32; (size * size) as usize];
            for (texel, a) in
                surface::splat(&mesh, &adj, &hit, r, opacity, hardness, size, &mut scratch)
            {
                cpu[texel] = cpu[texel].max(a);
            }

            // GPU coverage over the same dab's face set.
            let faces = surface::splat_faces(&mesh, &adj, &hit, r, &mut scratch);
            let gpu = dab.coverage(
                &device, &queue, &mesh, &faces, centroid, r, opacity, hardness, size,
            );

            let mut sum = 0.0f32;
            // The dab raster is deliberately *conservative* (`expand_tri`): GPU coverage is a
            // superset of the CPU splat. `under` = CPU painted but GPU missed — the unpainted-
            // edge "teeth" bug; must be ~0. `over` = the thin conservative edge rim (the fix
            // doing its job) — allowed, but bounded to stay an edge.
            let mut under = 0u32;
            let mut over = 0u32;
            let mut cpu_nz = 0u32;
            for i in 0..cpu.len() {
                let d = cpu[i] - gpu[i];
                sum += d.abs();
                if d > 0.1 {
                    under += 1;
                } else if d < -0.1 {
                    over += 1;
                }
                if cpu[i] > 0.0 {
                    cpu_nz += 1;
                }
            }
            let mean = sum / cpu.len() as f32;
            assert!(cpu_nz > 0, "the CPU dab must cover texels (tri {tri}, r {r})");
            assert!(
                mean < 0.006,
                "GPU dab mean error too high: {mean:.5} (tri {tri}, r {r})"
            );
            // The whole point of the fix: the GPU dab must not leave texels the CPU paints
            // (those become dark "teeth" on the model where an island edge crosses them).
            assert!(
                (under as f32) < 0.01 * cpu_nz as f32,
                "GPU under-covers the CPU dab (teeth): {under} of {cpu_nz} (tri {tri}, r {r})"
            );
            // The conservative rim is expected, but must stay a thin edge band.
            assert!(
                (over as f32) < 0.05 * cpu_nz as f32,
                "GPU over-covers far beyond a thin rim: {over} of {cpu_nz} (tri {tri}, r {r})"
            );
        }
    }

    #[test]
    fn gpu_stroke_accumulation_matches_cpu_max() {
        // Several overlapping dabs (a stroke) `Max`-accumulated into one persistent target
        // must match the CPU per-stroke max coverage — the discipline the renderer relies
        // on so overlapping dabs don't double-darken. Stamp the dabs, read the whole atlas
        // back once, and diff against the CPU max.
        let (device, queue) = crate::renderer::new_test_device();
        let mut dab = GpuDab::new(&device);
        let size = 128u32;
        let mesh = Mesh::cube();
        let adj = Adjacency::build(&mesh);
        let mut scratch = SplatScratch::new();

        let (p, _) = surface::tri_data(&mesh, 0);
        let normal = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
        // Overlapping points on face 0 (centroid + two interior offsets).
        let centers = [
            (p[0] + p[1] + p[2]) / 3.0,
            p[0] * 0.5 + p[1] * 0.5,
            p[1] * 0.4 + p[2] * 0.6,
        ];
        let (r, opacity, hardness) = (0.4f32, 0.8f32, 0.7f32);

        let mut cpu = vec![0.0f32; (size * size) as usize];
        dab.begin_stroke(&device, &queue, size);
        for &c in &centers {
            let hit = Hit {
                uv: Vec2::ZERO,
                tri: 0,
                pos: c,
                normal,
            };
            for (texel, a) in
                surface::splat(&mesh, &adj, &hit, r, opacity, hardness, size, &mut scratch)
            {
                cpu[texel] = cpu[texel].max(a);
            }
            let faces = surface::splat_faces(&mesh, &adj, &hit, r, &mut scratch);
            dab.stamp(&mesh, &faces, c, r, opacity, hardness, size);
        }
        let gpu = dab.read_region(
            &device,
            &queue,
            TexRect {
                x0: 0,
                y0: 0,
                x1: size,
                y1: size,
            },
        );

        let mut sum = 0.0f32;
        // Conservative dab raster (`expand_tri`): GPU coverage is a superset of CPU, so split
        // the disagreement. `under` (CPU>GPU) is the teeth bug and must be ~0; `over` is the
        // intended thin conservative rim.
        let mut under = 0u32;
        let mut over = 0u32;
        let mut cpu_nz = 0u32;
        for i in 0..cpu.len() {
            let d = cpu[i] - gpu[i];
            sum += d.abs();
            if d > 0.1 {
                under += 1;
            } else if d < -0.1 {
                over += 1;
            }
            if cpu[i] > 0.0 {
                cpu_nz += 1;
            }
        }
        let mean = sum / cpu.len() as f32;
        assert!(cpu_nz > 0);
        assert!(mean < 0.006, "accumulated stroke mean error too high: {mean:.5}");
        assert!(
            (under as f32) < 0.01 * cpu_nz as f32,
            "GPU under-covers the accumulated stroke (teeth): {under} of {cpu_nz}"
        );
        assert!(
            (over as f32) < 0.05 * cpu_nz as f32,
            "GPU over-covers far beyond a thin rim: {over} of {cpu_nz}"
        );
    }
}

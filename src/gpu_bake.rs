// src/gpu_bake.rs
//
// GPU mesh-map bake (Phase 2.5). The BVH lives in GPU buffers and `shaders/bake.wgsl`
// ray-traces ambient occlusion + the sun's cast shadow per atlas texel. This moves the
// expensive part of `bake.rs` off the CPU so AO scales to 2K/4K and dragging the sun
// re-bakes in real time (the current pain point: a 2K sun drag was a 16M-shadow-ray
// CPU recompute per slider tick).
//
// The CPU still rasterizes the geometry maps (`bake::bake_geometry` → pos/nrm/mask);
// only the ray-trace runs here. That keeps the result diffable against the clean CPU
// reference (`bake::bake_ao_into` / `MeshMaps::compute_light`) — see the parity tests
// below, which assert GPU ≈ CPU within a small tolerance on a known mesh.

use glam::Vec3;
use wgpu::util::DeviceExt;

use crate::bvh::Bvh;

/// Uniform params for one bake dispatch. Layout mirrors `Params` in `bake.wgsl`
/// (std140: `sun_dir` is a vec3, so `ao_dist` packs into its 16-byte row; total 48).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    sun_dir: [f32; 3],
    ao_dist: f32,
    bias: f32,
    reach: f32,
    n: u32,
    samples: u32,
    shadow: u32,
    n_tris: u32,
    _pad: [u32; 2],
}

/// Per-mesh GPU residency: the uploaded BVH + per-texel inputs + output buffers and the
/// bind group over them. Rebuilt by `upload` when the mesh or atlas size changes; reused
/// across sun re-bakes, where only the params and the dispatch differ.
struct Resident {
    n: u32,
    n_tris: u32,
    diag: f32,
    params: wgpu::Buffer,
    ao_out: wgpu::Buffer,
    light_out: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

/// Owns the two compute pipelines (built once) and the current mesh residency. The
/// renderer holds one of these; `ensure_mesh_maps` uploads + bakes AO, `ensure_light`
/// re-bakes the sun against the same residency.
pub struct GpuBaker {
    layout: wgpu::BindGroupLayout,
    ao_pipeline: wgpu::ComputePipeline,
    sun_pipeline: wgpu::ComputePipeline,
    resident: Option<Resident>,
}

impl GpuBaker {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bake compute shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bake.wgsl").into()),
        });

        // params(uniform) + nodes/tri_idx/tris/pos/nrm(read storage) + ao/light(rw storage).
        let buf = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let read = wgpu::BufferBindingType::Storage { read_only: true };
        let write = wgpu::BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bake bind layout"),
            entries: &[
                buf(0, wgpu::BufferBindingType::Uniform),
                buf(1, read),
                buf(2, read),
                buf(3, read),
                buf(4, read),
                buf(5, read),
                buf(6, write),
                buf(7, write),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bake pipeline layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = |entry_point: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point,
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            ao_pipeline: pipeline("ao_main"),
            sun_pipeline: pipeline("sun_main"),
            layout,
            resident: None,
        }
    }

    /// (Re)upload the BVH + per-texel `pos`/`nrm`/`mask` for a freshly rasterized map
    /// set (call after `bake::bake_geometry`). `pos`/`nrm` are the bake's world position
    /// + smooth normal per texel; `mask` marks covered texels (folded into `pos.w` on the
    /// GPU). A later `ao`/`sun` reuses this residency without re-uploading.
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        bvh: &Bvh,
        pos: &[Vec3],
        nrm: &[Vec3],
        mask: &[bool],
        diag: f32,
    ) {
        let n = pos.len() as u32;
        // pos.xyz + mask → vec4; nrm.xyz → vec4 (std430 array<vec4> stride is 16).
        let pos4: Vec<[f32; 4]> = pos
            .iter()
            .zip(mask)
            .map(|(p, &m)| [p.x, p.y, p.z, if m { 1.0 } else { 0.0 }])
            .collect();
        let nrm4: Vec<[f32; 4]> = nrm.iter().map(|v| [v.x, v.y, v.z, 0.0]).collect();

        let nodes = bvh.gpu_nodes();
        let tri_idx = bvh.gpu_tri_indices();
        let tris = bvh.gpu_tri_positions();
        // wgpu rejects zero-sized buffers; a degenerate 0-triangle mesh gets a dummy
        // entry the shader ignores via the `n_tris == 0` guard.
        let tri_idx_data: Vec<u32> = if tri_idx.is_empty() {
            vec![0]
        } else {
            tri_idx.to_vec()
        };
        let tris_data: Vec<f32> = if tris.is_empty() { vec![0.0; 9] } else { tris };

        let init = |label, contents: &[u8], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let nodes_buf = init("bvh nodes", bytemuck::cast_slice(&nodes), storage);
        let tri_idx_buf = init("bvh tri_idx", bytemuck::cast_slice(&tri_idx_data), storage);
        let tris_buf = init("bvh tris", bytemuck::cast_slice(&tris_data), storage);
        let pos_buf = init("bake pos", bytemuck::cast_slice(&pos4), storage);
        let nrm_buf = init("bake nrm", bytemuck::cast_slice(&nrm4), storage);

        let out_bytes = (n.max(1) as u64) * 4;
        let out_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let ao_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bake ao out"),
            size: out_bytes,
            usage: out_usage,
            mapped_at_creation: false,
        });
        let light_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bake light out"),
            size: out_bytes,
            usage: out_usage,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bake params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bake bind group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: nodes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: tri_idx_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: tris_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: nrm_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: ao_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: light_out.as_entire_binding() },
            ],
        });

        self.resident = Some(Resident {
            n,
            n_tris: bvh.tri_count() as u32,
            diag,
            params,
            ao_out,
            light_out,
            bind_group,
        });
    }

    /// Ray-trace ambient occlusion for the uploaded mesh and read it back (one f32 per
    /// texel, row-major like the CPU `MeshMaps::ao`). Panics if `upload` hasn't run.
    pub fn ao(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
        let r = self.resident.as_ref().expect("GpuBaker::ao before upload");
        let params = Params {
            sun_dir: [0.0, 0.0, 0.0],
            ao_dist: crate::bake::ao_reach(r.diag),
            bias: crate::bake::ray_bias(r.diag),
            reach: r.diag,
            n: r.n,
            samples: crate::bake::AO_SAMPLES,
            shadow: 0,
            n_tris: r.n_tris,
            _pad: [0, 0],
        };
        queue.write_buffer(&r.params, 0, bytemuck::bytes_of(&params));
        self.run(device, queue, &self.ao_pipeline, r, &r.ao_out)
    }

    /// Bake the directional ("sun") light for `dir` (toward the light), with an optional
    /// BVH shadow ray, and read it back. Reuses the residency, so an interactive sun
    /// drag is just a params write + dispatch + readback. Panics if `upload` hasn't run.
    pub fn sun(&self, device: &wgpu::Device, queue: &wgpu::Queue, dir: Vec3, shadow: bool) -> Vec<f32> {
        let r = self.resident.as_ref().expect("GpuBaker::sun before upload");
        let d = dir.normalize_or_zero();
        let params = Params {
            sun_dir: [d.x, d.y, d.z],
            ao_dist: crate::bake::ao_reach(r.diag),
            bias: crate::bake::ray_bias(r.diag),
            reach: r.diag,
            n: r.n,
            samples: crate::bake::AO_SAMPLES,
            shadow: shadow as u32,
            n_tris: r.n_tris,
            _pad: [0, 0],
        };
        queue.write_buffer(&r.params, 0, bytemuck::bytes_of(&params));
        self.run(device, queue, &self.sun_pipeline, r, &r.light_out)
    }

    /// Dispatch one compute pass over all texels and read `out_buf` back as `Vec<f32>`.
    /// The grid is 2D — X extent `num_workgroups.x * 64` — so the per-dimension 65535
    /// workgroup cap isn't hit at 2K/4K (the shader maps (x,y) back to a linear index).
    fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &wgpu::ComputePipeline,
        r: &Resident,
        out_buf: &wgpu::Buffer,
    ) -> Vec<f32> {
        const WG: u32 = 64;
        let groups = r.n.div_ceil(WG).max(1);
        let gx = groups.min(65535);
        let gy = groups.div_ceil(gx);
        let bytes = (r.n.max(1) as u64) * 4;

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bake readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bake encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bake pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &r.bind_group, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        encoder.copy_buffer_to_buffer(out_buf, 0, &readback, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("failed to map bake readback");
        let data = slice.get_mapped_range();
        let mut out: Vec<f32> = bytemuck::cast_slice(&data[..]).to_vec();
        drop(data);
        readback.unmap();
        out.truncate(r.n as usize);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bake;
    use crate::mesh::{Mesh, Vertex};

    // A device via the renderer's retrying creator (see `renderer::request_device`), so
    // the parity/bench tests survive the suite's concurrent device creations.
    fn device_queue() -> (wgpu::Device, wgpu::Queue) {
        crate::renderer::new_test_device()
    }

    /// Mean + max absolute error between two equal-length channels over covered texels.
    fn error(a: &[f32], b: &[f32], mask: &[bool]) -> (f32, f32) {
        let mut sum = 0.0f32;
        let mut max = 0.0f32;
        let mut count = 0u32;
        for i in 0..a.len() {
            if !mask[i] {
                continue;
            }
            let d = (a[i] - b[i]).abs();
            sum += d;
            max = max.max(d);
            count += 1;
        }
        (sum / count.max(1) as f32, max)
    }

    /// A concave "L" cross-section: a floor (+Y) and a wall (+X) sharing an edge, with
    /// disjoint UV halves. Concave, so it self-shadows — the wall casts onto the floor
    /// for a low side-lit sun, which exercises the GPU shadow-ray branch against the CPU.
    fn l_shape() -> Mesh {
        let v = |p: [f32; 3], n: [f32; 3], uv: [f32; 2]| Vertex {
            position: p,
            normal: n,
            uv,
        };
        let vertices = vec![
            // floor: x in [0,2], z in [0,1], y=0, normal +Y, u in [0,0.5]
            v([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0]),
            v([2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0]),
            v([2.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.5, 1.0]),
            v([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0]),
            // wall: x=0, y in [0,1], z in [0,1], normal +X, u in [0.5,1]
            v([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0]),
            v([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0]),
            v([0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0]),
            v([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.5, 1.0]),
        ];
        let indices = vec![0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];
        Mesh {
            vertices,
            indices,
            needs_normals: false,
            needs_uvs: false,
            source_transform: crate::mesh::SourceTransform::IDENTITY,
        }
    }

    #[test]
    fn gpu_ao_matches_cpu_reference() {
        // The GPU AO bake must reproduce the CPU `bake_ao_into` within a small tolerance
        // (the BVH-on-GPU traversal mirrors `occludes`; only transcendental rounding can
        // flip a grazing ray, and only by ~1/AO_SAMPLES on a handful of silhouette
        // texels). Feed the GPU the *same* rasterized pos/nrm/mask so the diff isolates
        // the ray-trace.
        let (device, queue) = device_queue();
        let mesh = Mesh::cube();
        let bvh = Bvh::build(&mesh);
        let maps = bake::bake(&mesh, &bvh, 64); // CPU reference (geometry + CPU AO)

        let mut baker = GpuBaker::new(&device);
        baker.upload(&device, &bvh, &maps.pos, &maps.nrm, &maps.mask, maps.diag);
        let gpu_ao = baker.ao(&device, &queue);

        assert_eq!(gpu_ao.len(), maps.ao.len());
        let (mean, max) = error(&gpu_ao, &maps.ao, &maps.mask);
        assert!(
            mean < 0.01 && max < 0.2,
            "GPU AO diverges from CPU: mean={mean:.4} max={max:.4}"
        );
        // And it must actually vary (cube edges occlude), not be a flat zero.
        let spread = gpu_ao.iter().cloned().fold(0.0f32, f32::max);
        assert!(spread > 0.1, "GPU AO is flat — the bake did nothing");
    }

    /// CPU vs GPU AO bake wall-clock at a paint-relevant atlas size. The CPU path is
    /// the (already rayon-parallel) `bake_ao_into`; the GPU path is upload + dispatch +
    /// readback. Ignored by default — run for numbers:
    ///   cargo test --release gpu_bake::tests::bench_ao -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_ao() {
        use std::time::Instant;
        let size = std::env::var("LOWTEX_BENCH_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024u32);
        let mesh = l_shape();
        let bvh = Bvh::build(&mesh);

        // `bake_ao_into` only writes `ao`, so the same maps feed the GPU's pos/nrm/mask.
        let mut maps = bake::bake_geometry(&mesh, size);
        let t = Instant::now();
        bake::bake_ao_into(&mut maps, &bvh);
        let cpu_ms = t.elapsed().as_secs_f64() * 1e3;

        let (device, queue) = device_queue();
        let mut baker = GpuBaker::new(&device);
        baker.upload(&device, &bvh, &maps.pos, &maps.nrm, &maps.mask, maps.diag);
        let _ = baker.ao(&device, &queue); // warm
        let t = Instant::now();
        let _ = baker.ao(&device, &queue);
        let gpu_ms = t.elapsed().as_secs_f64() * 1e3;

        println!(
            "\nAO bake @ {size}² ({} tris):\n  CPU (rayon): {cpu_ms:8.2} ms\n  GPU:         {gpu_ms:8.2} ms\n  speedup:     {:.1}×\n",
            bvh.tri_count(),
            cpu_ms / gpu_ms.max(1e-6),
        );
    }

    #[test]
    fn gpu_sun_matches_cpu_reference() {
        // Sun parity on the self-shadowing L-shape with a low side-lit sun: `max(N·L,0)`
        // plus the BVH shadow ray must match `compute_light`. Asserts the CPU actually
        // shadows some lit texels (so the GPU shadow branch is genuinely exercised), then
        // that GPU ≈ CPU within tolerance.
        let (device, queue) = device_queue();
        let mesh = l_shape();
        let bvh = Bvh::build(&mesh);
        let mut maps = bake::bake_geometry(&mesh, 96);
        let dir = Vec3::new(-1.0, 0.5, 0.0).normalize();
        maps.compute_light(&bvh, dir, true); // CPU reference into maps.light

        // Confirm the geometry self-shadows on the CPU: some lit-facing floor texels are
        // zeroed by the shadow ray (light 0 while N·L > 0).
        let dnorm = dir.normalize_or_zero();
        let shadowed = (0..maps.light.len())
            .filter(|&i| maps.mask[i] && maps.nrm[i].dot(dnorm) > 0.05 && maps.light[i] == 0.0)
            .count();
        assert!(shadowed > 0, "test mesh/sun cast no shadow — branch not exercised");

        let mut baker = GpuBaker::new(&device);
        baker.upload(&device, &bvh, &maps.pos, &maps.nrm, &maps.mask, maps.diag);
        let gpu_light = baker.sun(&device, &queue, dir, true);

        let (mean, max) = error(&gpu_light, &maps.light, &maps.mask);
        assert!(
            mean < 0.02 && max < 0.35,
            "GPU sun diverges from CPU: mean={mean:.4} max={max:.4}, shadowed={shadowed}"
        );
    }
}

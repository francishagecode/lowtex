// src/shaders/bake.wgsl
//
// GPU mesh-map bake (Phase 2.5). Two compute entry points ray-trace the BVH per
// atlas texel — the expensive part of `src/bake.rs` — so AO scales to 2K/4K and the
// sun becomes an interactive slider:
//
//   - ao_main  : cosine-weighted hemisphere occlusion (mirrors `bake`'s AO loop +
//                `ao_for_texel`).
//   - sun_main : max(N·L, 0) with one optional BVH shadow ray (mirrors `compute_light`).
//
// Every primitive below is a line-for-line port of the CPU reference so a GPU bake
// diffs against `src/bake.rs` within a small tolerance (see the parity tests in
// gpu_bake.rs): the same `hash2`, the same cosine sampling, the same Möller–Trumbore
// and slab test, the same `safe()` inv-dir clamp and the same 64-deep traversal stack.
// The CPU rasterization still feeds `pos`/`nrm`/`mask`, so only the ray-trace moves.

struct Params {
    sun_dir: vec3<f32>,
    ao_dist: f32,
    bias: f32,
    reach: f32,
    n: u32,        // texel count
    samples: u32,  // AO_SAMPLES
    shadow: u32,   // sun: cast shadows when != 0
    n_tris: u32,   // empty-mesh guard
    pad0: u32,
    pad1: u32,
};

// A BVH node, matching `bvh::GpuBvhNode` (std430: 8 words / 32 bytes).
struct Node {
    bmin: vec3<f32>,
    left_or_first: u32,
    bmax: vec3<f32>,
    count: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> tri_idx: array<u32>;
@group(0) @binding(3) var<storage, read> tris: array<f32>; // 9 floats per triangle
@group(0) @binding(4) var<storage, read> pos: array<vec4<f32>>; // xyz world pos, w = mask
@group(0) @binding(5) var<storage, read> nrm: array<vec4<f32>>; // xyz normal
@group(0) @binding(6) var<storage, read_write> ao_out: array<f32>;
@group(0) @binding(7) var<storage, read_write> light_out: array<f32>;

// Two deterministic pseudo-random values in [0,1) from a texel + sample index.
// Identical integer hash to `bake::hash2` — u32 ops wrap mod 2^32 in WGSL, matching
// Rust's `wrapping_mul`, so r1/r2 are bit-exact across CPU and GPU.
fn hash2(a: u32, b: u32) -> vec2<f32> {
    var h: u32 = (a * 0x9E3779B1u) ^ (b * 0x85EBCA77u);
    h = h ^ (h >> 15u);
    h = h * 0x2C1B3C6Du;
    h = h ^ (h >> 12u);
    let r1 = f32(h & 0xFFFFu) / 65536.0;
    let r2 = f32((h >> 16u) & 0xFFFFu) / 65536.0;
    return vec2<f32>(r1, r2);
}

// `glam::normalize_or_zero`: unit vector, or zero when degenerate.
fn norm0(v: vec3<f32>) -> vec3<f32> {
    let l = length(v);
    if (l > 0.0) {
        return v / l;
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// `bvh::ray_hits_aabb`: padded slab test, does the ray reach the box within max_t?
fn hits_aabb(ro: vec3<f32>, inv_dir: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>, max_t: f32) -> bool {
    let pad = vec3<f32>(1e-4, 1e-4, 1e-4);
    let t1 = (bmin - pad - ro) * inv_dir;
    let t2 = (bmax + pad - ro) * inv_dir;
    let tmin = min(t1, t2);
    let tmax = max(t1, t2);
    let enter = max(tmin.x, max(tmin.y, tmin.z));
    let exit = min(tmax.x, min(tmax.y, tmax.z));
    return exit >= max(enter, 0.0) && enter <= max_t;
}

// `paint::intersect_triangle` (Möller–Trumbore): distance along the ray to the hit,
// or -1 on miss. Same EPS = 1e-7 and the same u/v acceptance bounds as the CPU.
fn intersect_tri(ro: vec3<f32>, rd: vec3<f32>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> f32 {
    let eps = 1e-7;
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = cross(rd, e2);
    let a = dot(e1, h);
    if (abs(a) < eps) {
        return -1.0;
    }
    let f = 1.0 / a;
    let s = ro - v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) {
        return -1.0;
    }
    let q = cross(s, e1);
    let v = f * dot(rd, q);
    if (v < 0.0 || u + v > 1.0) {
        return -1.0;
    }
    let t = f * dot(e2, q);
    if (t > eps) {
        return t;
    }
    return -1.0;
}

// `|x| < 1e-8 -> 1e-8` (note: clamps to +1e-8 regardless of sign, matching the CPU).
fn safe(x: f32) -> f32 {
    if (abs(x) < 1e-8) {
        return 1e-8;
    }
    return x;
}

// `bvh::occludes`: does anything block the ray within max_dist? Explicit 64-deep stack
// traversal, front-to-back-ish, early-out on the first real triangle hit.
fn occludes(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32) -> bool {
    if (params.n_tris == 0u) {
        return false;
    }
    let inv_dir = vec3<f32>(1.0 / safe(dir.x), 1.0 / safe(dir.y), 1.0 / safe(dir.z));
    var stack: array<u32, 64>;
    var sp: i32 = 0;
    stack[0] = 0u;
    sp = 1;
    loop {
        if (sp <= 0) {
            break;
        }
        sp = sp - 1;
        let node = nodes[stack[sp]];
        if (!hits_aabb(origin, inv_dir, node.bmin, node.bmax, max_dist)) {
            continue;
        }
        if (node.count > 0u) {
            let first = node.left_or_first;
            for (var k: u32 = 0u; k < node.count; k = k + 1u) {
                let ti = tri_idx[first + k];
                let base = ti * 9u;
                let v0 = vec3<f32>(tris[base], tris[base + 1u], tris[base + 2u]);
                let v1 = vec3<f32>(tris[base + 3u], tris[base + 4u], tris[base + 5u]);
                let v2 = vec3<f32>(tris[base + 6u], tris[base + 7u], tris[base + 8u]);
                let t = intersect_tri(origin, dir, v0, v1, v2);
                if (t > 0.0 && t < max_dist) {
                    return true;
                }
            }
        } else {
            // Internal: push both children (LIFO order is irrelevant for occlusion).
            stack[sp] = node.left_or_first;
            sp = sp + 1;
            stack[sp] = node.left_or_first + 1u;
            sp = sp + 1;
        }
    }
    return false;
}

// Map the 2D dispatch back to a linear texel index. The X extent is
// num_workgroups.x * 64 (the workgroup size below); a 2D grid keeps us under the
// 65535-groups-per-dimension cap at 2K/4K. Threads past `n` early-out.
fn texel_index(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.y * (nwg.x * 64u) + gid.x;
}

@compute @workgroup_size(64, 1, 1)
fn ao_main(@builtin(global_invocation_id) gid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = texel_index(gid, nwg);
    if (idx >= params.n) {
        return;
    }
    if (pos[idx].w < 0.5) {
        ao_out[idx] = 0.0; // uncovered texel
        return;
    }
    let p = pos[idx].xyz;
    let n = nrm[idx].xyz;
    // Orthonormal tangent frame — `bake::basis`.
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(n.y) >= 0.99) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent = norm0(cross(up, n));
    let bitangent = cross(n, tangent);

    var occ: u32 = 0u;
    let origin = p + n * params.bias;
    for (var s: u32 = 0u; s < params.samples; s = s + 1u) {
        let r = hash2(idx, s);
        let phi = 6.2831853071795864769 * r.x; // TAU
        let cos_t = sqrt(1.0 - r.y);
        let sin_t = sqrt(r.y);
        let dir = tangent * (cos(phi) * sin_t) + bitangent * (sin(phi) * sin_t) + n * cos_t;
        if (occludes(origin, dir, params.ao_dist)) {
            occ = occ + 1u;
        }
    }
    ao_out[idx] = f32(occ) / f32(params.samples);
}

@compute @workgroup_size(64, 1, 1)
fn sun_main(@builtin(global_invocation_id) gid: vec3<u32>,
            @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = texel_index(gid, nwg);
    if (idx >= params.n) {
        return;
    }
    if (pos[idx].w < 0.5) {
        light_out[idx] = 0.0;
        return;
    }
    let n = nrm[idx].xyz;
    let ndotl = max(dot(n, params.sun_dir), 0.0);
    var v = ndotl;
    if (params.shadow != 0u && ndotl > 0.0) {
        if (occludes(pos[idx].xyz + n * params.bias, params.sun_dir, params.reach)) {
            v = 0.0;
        }
    }
    light_out[idx] = v;
}

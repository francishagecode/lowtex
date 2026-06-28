// src/shaders/dab.wgsl
//
// GPU dab stamping (Phase 1, option B). The CPU floods mesh adjacency to a *face set*
// (`surface::splat_faces`); this pass rasterizes exactly those faces into the atlas with
// a per-fragment surface falloff, reproducing `surface::splat`'s cross-seam coverage with
// the per-texel cost on the GPU instead of the CPU.
//
// Vertex stage: map a face's UVs into atlas clip space (UV-as-position) and carry the
// face's interpolated *world* position. Fragment stage: coverage = opacity ·
// falloff(distance(worldPos, hit_center) / radius, hardness) — a line-for-line port of
// `paint::falloff` and `surface::splat`'s weight, so the GPU coverage diffs against the
// CPU splat within a small tolerance (see the parity test in gpu_dab.rs). Zero-coverage
// fragments are discarded so a face past the radius can't overwrite a covered texel.

// Per-dab parameters now ride on the vertex (constant across a face's 3 verts), so a
// whole frame's dabs rasterize in one draw — see DabVertex in gpu_dab.rs.
struct VsIn {
    @location(0) uv: vec2<f32>,
    @location(1) world: vec3<f32>,
    @location(2) center: vec3<f32>, // hit point in world space
    @location(3) radius: f32,       // world-space brush radius
    @location(4) opacity: f32,
    @location(5) hardness: f32,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) center: vec3<f32>,
    @location(2) radius: f32,
    @location(3) opacity: f32,
    @location(4) hardness: f32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    // UV (v down, 0 at top) → clip space. x: [0,1]→[-1,1]; y flips so v=0 lands at the
    // top framebuffer row, matching the atlas's row-major (V-down) texel layout.
    out.clip = vec4<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0, 0.0, 1.0);
    out.world = in.world;
    out.center = in.center;
    out.radius = in.radius;
    out.opacity = in.opacity;
    out.hardness = in.hardness;
    return out;
}

// `paint::falloff`: 1 inside the hard core, linear ramp to 0 at the rim, 0 past it.
fn falloff(d: f32, hardness: f32) -> f32 {
    if (d >= 1.0) {
        return 0.0;
    }
    let h = clamp(hardness, 0.0, 1.0);
    if (d <= h) {
        return 1.0;
    }
    return 1.0 - (d - h) / max(1.0 - h, 1e-4);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) f32 {
    let dist = distance(in.world, in.center);
    let cov = in.opacity * falloff(dist / in.radius, in.hardness);
    if (cov <= 0.0) {
        discard; // past the radius — don't overwrite a texel a nearer face covered
    }
    return cov;
}

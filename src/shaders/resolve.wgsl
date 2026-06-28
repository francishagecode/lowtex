// src/shaders/resolve.wgsl
//
// GPU paint resolve (Phase 5): blend a stroke's accumulated coverage into the active
// layer with no GPU→CPU readback. The dab pass (`gpu_dab`) accumulates per-stroke max
// coverage; this pass reads the immutable pre-stroke base + that coverage and writes the
// resolved layer colour, a line-for-line port of `paint::blend4`/`erase4` (the CPU
// `apply_coverage`). Idempotent from the base, so re-resolving the dirty region each
// frame composes correctly. Rendered into the active layer's array slice; the display
// then composites from the layer arrays — the whole stroke stays on the GPU.

struct ResolveU {
    color: vec4<f32>, // rgb = solid brush colour (0..1); a unused
    mode: u32,        // 0 = blend solid colour, 1 = erase (lower alpha), 2 = blend material
    tile: f32,        // material UV tiling (mode 2)
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var base_tex: texture_2d<f32>;
@group(0) @binding(1) var cov_tex: texture_2d<f32>;
@group(0) @binding(2) var material_tex: texture_2d<f32>; // tiled brush material (mode 2)
@group(0) @binding(3) var<uniform> u: ResolveU;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var xy = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(xy[vi], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(i32(floor(pos.x)), i32(floor(pos.y)));
    let base = textureLoad(base_tex, p, 0);
    let a = textureLoad(cov_tex, p, 0).r;
    if (a <= 0.0) {
        return base; // untouched by the stroke
    }
    if (u.mode == 1u) {
        // erase: leave rgb at the base, lower alpha toward transparent.
        return vec4<f32>(base.rgb, base.a * (1.0 - a));
    }
    var src = u.color.rgb;
    if (u.mode == 2u) {
        // Tiled material: sample at the texel's own atlas UV × tile, wrapped, nearest —
        // a port of `material::Material::sample` (rem_euclid → fract, truncate → floor).
        let atlas = vec2<f32>(textureDimensions(base_tex));
        let mdim = vec2<f32>(textureDimensions(material_tex));
        let uv = vec2<f32>(f32(p.x), f32(p.y)) / atlas;
        let w = fract(uv * u.tile);
        let mc = vec2<i32>(i32(floor(w.x * mdim.x)), i32(floor(w.y * mdim.y)));
        src = textureLoad(material_tex, mc, 0).rgb;
    }
    // blend: composite the solid colour / material sample over the base at coverage `a`.
    let rgb = base.rgb * (1.0 - a) + src * a;
    let al = base.a * (1.0 - a) + a;
    return vec4<f32>(rgb, al);
}

// src/shaders/post.wgsl
//
// Full-screen post-process (G8): palette quantization + ordered (Bayer) dither.
// Samples the rendered scene texture and snaps each pixel to the nearest palette
// color. Optional 4×4 Bayer dithering biases the color before snapping, breaking
// up banding on gradients at low color counts.
//
// Sampling an sRGB scene texture yields LINEAR values, so the palette is uploaded
// in linear space too (renderer converts). Output goes to an sRGB target, which
// re-encodes — so comparisons and the result are consistent.

struct Palette {
    // x = enabled, y = color count, z = dither (0/1), w = dither strength
    params: vec4<f32>,
    colors: array<vec4<f32>, 256>,
};

@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> pal: Palette;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VsOut {
    // Single oversized triangle covering the screen.
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let xy = corners[vid];
    var out: VsOut;
    out.pos = vec4<f32>(xy, 0.0, 1.0);
    // NDC → uv (0..1); flip Y since texture origin is top-left.
    out.uv = vec2<f32>(xy.x * 0.5 + 0.5, 1.0 - (xy.y * 0.5 + 0.5));
    return out;
}

// 4×4 Bayer threshold matrix, normalized to (0,1).
fn bayer(px: vec2<u32>) -> f32 {
    // `var` (not `let`): WGSL only allows dynamic indexing of a memory location.
    var m = array<f32, 16>(
        0.0, 8.0, 2.0, 10.0,
        12.0, 4.0, 14.0, 6.0,
        3.0, 11.0, 1.0, 9.0,
        15.0, 7.0, 13.0, 5.0,
    );
    let i = (px.y % 4u) * 4u + (px.x % 4u);
    return (m[i] + 0.5) / 16.0;
}

fn nearest_palette(c: vec3<f32>) -> vec3<f32> {
    let count = i32(pal.params.y);
    var best = pal.colors[0].rgb;
    var best_d = 1e9;
    for (var i = 0; i < count; i = i + 1) {
        let p = pal.colors[i].rgb;
        let d = dot(c - p, c - p);
        if (d < best_d) {
            best_d = d;
            best = p;
        }
    }
    return best;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let src = textureSample(scene, samp, in.uv);
    if (pal.params.x < 0.5 || pal.params.y < 1.0) {
        return src; // quantize disabled (or empty palette) → passthrough
    }
    var c = src.rgb;
    if (pal.params.z > 0.5) {
        // Ordered dither: bias by (bayer - 0.5) * strength before snapping.
        let px = vec2<u32>(in.pos.xy);
        let bias = (bayer(px) - 0.5) * pal.params.w;
        c = clamp(c + vec3<f32>(bias), vec3<f32>(0.0), vec3<f32>(1.0));
    }
    return vec4<f32>(nearest_palette(c), src.a);
}

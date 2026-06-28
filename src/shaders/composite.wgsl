// src/shaders/composite.wgsl
//
// GPU layer compositing (Phase 1 of the GPU display port). A single fullscreen pass
// reproduces `layers::composite_texel` (src/layers.rs) bit-for-bit (modulo ±1 u8
// rounding): bottom-up `over` of every visible layer, each gated by its red-channel
// mask, scaled by opacity, combined through one of four blend modes against the running
// premultiplied-ish accumulator. Layers live as 2D-array textures (one slice per layer),
// read with `textureLoad` (integer, nearest — the atlas is rendered 1:1). The output is
// written to an `Rgba8Unorm` view so the stored bytes match the CPU's, then sampled
// through an `Rgba8UnormSrgb` view by main.wgsl (decode at sample time only).

struct LayerParam {
    opacity: f32,
    blend: u32,   // 0 Normal, 1 Multiply, 2 Add, 3 Screen
    visible: u32, // 0 = skip (hidden / zero-opacity)
    pad: u32,
};

// Palette quantize + dither params (mirrors PaletteSettings + the renderer's palette).
struct Quant {
    enabled: u32,     // 0 = no palette constraint (pass composite through)
    dither: u32,      // 1 = 4x4 Bayer ordered dither
    strength: f32,    // dither bias strength
    palette_len: u32, // number of valid entries in `palette`
};

@group(0) @binding(0) var color_tex: texture_2d_array<f32>;
@group(0) @binding(1) var mask_tex: texture_2d_array<f32>;
@group(0) @binding(2) var<storage, read> params: array<LayerParam>;
@group(0) @binding(3) var<storage, read> palette: array<vec4<f32>>; // xyz = sRGB 0..1
@group(0) @binding(4) var<uniform> quant: Quant;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // A single oversized triangle covering the whole atlas.
    var xy = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(xy[vi], 0.0, 1.0);
}

// `layers::BlendMode::apply` — combine a source channel with the backdrop (both 0..1).
fn blend_apply(mode: u32, dst: f32, src: f32) -> f32 {
    if (mode == 1u) { return dst * src; }                       // Multiply
    if (mode == 2u) { return min(dst + src, 1.0); }             // Add
    if (mode == 3u) { return 1.0 - (1.0 - dst) * (1.0 - src); } // Screen
    return src;                                                 // Normal
}

// `palette::bayer4` — 4x4 Bayer threshold (normalized 0..1), indexed [y%4][x%4].
fn bayer4(x: i32, y: i32) -> f32 {
    var m = array<f32, 16>(
        0.0, 8.0, 2.0, 10.0,
        12.0, 4.0, 14.0, 6.0,
        3.0, 11.0, 1.0, 9.0,
        15.0, 7.0, 13.0, 5.0,
    );
    let yy = u32(y) % 4u;
    let xx = u32(x) % 4u;
    return (m[yy * 4u + xx] + 0.5) / 16.0;
}

// Round-half-up to a u8 level then back to 0..1 — matches Rust `(x*255).round()` for the
// non-negative values here (WGSL `round` is banker's rounding, which would diverge).
fn to_u8_unit(x: f32) -> f32 {
    return floor(clamp(x, 0.0, 1.0) * 255.0 + 0.5) / 255.0;
}

// `palette::nearest` — closest palette colour by squared sRGB distance, *first index wins*
// on ties (strict `<`, matching Rust `min_by`). Returns the palette colour (0..1).
fn nearest_palette(c: vec3<f32>) -> vec3<f32> {
    let n = i32(quant.palette_len);
    var best = palette[0].xyz;
    var best_d = dot(c - best, c - best);
    for (var k = 1; k < n; k = k + 1) {
        let pc = palette[k].xyz;
        let d = dot(c - pc, c - pc);
        if (d < best_d) {
            best_d = d;
            best = pc;
        }
    }
    return best;
}

// `palette::quantize_rgba` over one already-composited texel: emulate the CPU's u8 store,
// add the ordered-dither bias, snap to the nearest palette colour. Returns 0..1 (the
// palette colour, so writing it to the Unorm atlas yields the exact palette byte).
fn quantize(rgb: vec3<f32>, x: i32, y: i32) -> vec3<f32> {
    if (quant.enabled == 0u || quant.palette_len == 0u) {
        return rgb;
    }
    let cu = vec3<f32>(to_u8_unit(rgb.r), to_u8_unit(rgb.g), to_u8_unit(rgb.b));
    var bias = 0.0;
    if (quant.dither != 0u) {
        bias = (bayer4(x, y) - 0.5) * quant.strength;
    }
    let cand = clamp(cu + vec3<f32>(bias), vec3<f32>(0.0), vec3<f32>(1.0));
    return nearest_palette(cand);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(i32(floor(pos.x)), i32(floor(pos.y)));
    let n = i32(arrayLength(&params));
    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var i = 0; i < n; i = i + 1) {
        let pr = params[i];
        if (pr.visible == 0u || pr.opacity <= 0.0) {
            continue;
        }
        let col = textureLoad(color_tex, p, i, 0);
        let m = textureLoad(mask_tex, p, i, 0).r;
        let sa = col.a * pr.opacity * m;
        if (sa <= 0.0) {
            continue;
        }
        // Per-channel blend over the running accumulator (channels independent).
        acc.r = acc.r * (1.0 - sa) + blend_apply(pr.blend, acc.r, col.r) * sa;
        acc.g = acc.g * (1.0 - sa) + blend_apply(pr.blend, acc.g, col.g) * sa;
        acc.b = acc.b * (1.0 - sa) + blend_apply(pr.blend, acc.b, col.b) * sa;
        acc.a = sa + acc.a * (1.0 - sa);
    }
    let rgb = quantize(clamp(acc.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), p.x, p.y);
    return vec4<f32>(rgb, clamp(acc.a, 0.0, 1.0));
}

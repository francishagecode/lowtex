// src/shaders/bleed.wgsl
//
// GPU UV-gutter dilation (Phase 3 of the GPU display port) — the exact analogue of
// `bleed::dilate` (src/bleed.rs), run as a ping-pong. The CPU grows island colours
// outward one ring at a time: each ring freezes a snapshot of validity+colour, then
// every still-invalid texel takes the colour of the *first valid 4-neighbour* in the
// fixed order [(-1,0),(1,0),(0,-1),(0,1)]. A ping-pong reproduces this precisely — each
// pass reads the *previous* texture (= the frozen snapshot) and writes the next — so the
// result is bit-identical to the CPU dilate given the same coverage and source colours.
//
// MRT: location 0 = colour (Rgba8Unorm), location 1 = validity (R8Unorm, 0/1). The
// renderer runs `pad` of these, seeded from the composite atlas + the static coverage
// mask, then copies the final colour back into the atlas main.wgsl samples.

@group(0) @binding(0) var src_color: texture_2d<f32>;
@group(0) @binding(1) var src_valid: texture_2d<f32>;

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) valid: f32,
};

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
fn fs_main(@builtin(position) pos: vec4<f32>) -> FsOut {
    let p = vec2<i32>(i32(floor(pos.x)), i32(floor(pos.y)));
    let dims = vec2<i32>(textureDimensions(src_color));
    var out: FsOut;

    // Already valid (covered, or filled by an earlier ring): pass straight through.
    if (textureLoad(src_valid, p, 0).r >= 0.5) {
        out.color = textureLoad(src_color, p, 0);
        out.valid = 1.0;
        return out;
    }

    // Otherwise take the first valid 4-neighbour in the CPU's fixed scan order.
    var offs = array<vec2<i32>, 4>(
        vec2<i32>(-1, 0),
        vec2<i32>(1, 0),
        vec2<i32>(0, -1),
        vec2<i32>(0, 1),
    );
    for (var k = 0; k < 4; k = k + 1) {
        let n = p + offs[k];
        if (n.x < 0 || n.y < 0 || n.x >= dims.x || n.y >= dims.y) {
            continue;
        }
        if (textureLoad(src_valid, n, 0).r >= 0.5) {
            out.color = textureLoad(src_color, n, 0);
            out.valid = 1.0;
            return out;
        }
    }

    // No valid neighbour this ring — unchanged, still invalid.
    out.color = textureLoad(src_color, p, 0);
    out.valid = 0.0;
    return out;
}

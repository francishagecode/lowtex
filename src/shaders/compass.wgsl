// src/shaders/compass.wgsl
//
// Thick screen-space lines for the orientation compass. wgpu can't draw lines
// wider than 1px portably, so each axis segment arrives as a quad (6 vertices)
// carrying both endpoints; the vertex shader projects them, then offsets the
// corners perpendicular to the segment's screen direction to give it width.
// The compass draws into a *square* corner viewport, so a single NDC half-width
// reads as a uniform thickness without an aspect correction.
//
// Per-vertex `param` packs the corner: x = t (0 at start, 1 at end), y = side
// (signed half-width multiplier — magnitude < 1 thins a segment, e.g. the dim
// negative stubs). Colors are linear (the sRGB surface encodes on write).

struct VertexInput {
    @location(0) start: vec3<f32>,
    @location(1) end: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) param: vec2<f32>, // x = t (0|1), y = side (signed width)
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@group(0) @binding(0) var<uniform> view_proj: mat4x4<f32>;

// Half-width of a full-weight axis, in NDC (the viewport is square, so this is
// the same fraction of the gizmo box in both directions).
const THICKNESS: f32 = 0.04;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let cs = view_proj * vec4<f32>(in.start, 1.0);
    let ce = view_proj * vec4<f32>(in.end, 1.0);
    let ndc_s = cs.xy / cs.w;
    let ndc_e = ce.xy / ce.w;

    var dir = ndc_e - ndc_s;
    let len = length(dir);
    if (len > 1e-6) {
        dir = dir / len;
    } else {
        dir = vec2<f32>(1.0, 0.0);
    }
    let normal = vec2<f32>(-dir.y, dir.x);

    let base = mix(ndc_s, ndc_e, in.param.x);
    let pos = base + normal * in.param.y * THICKNESS;
    // Depth is ignored (the pipeline compares Always, never writes), so any z in
    // range is fine; w = 1 since the gizmo projection is orthographic.
    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}

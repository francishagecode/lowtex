// src/shaders/lines.wgsl
//
// Unlit colored lines for viewport furniture: the ground grid and the
// orientation compass. Each vertex carries its own color; the only uniform is a
// view-projection matrix (the scene's for the grid, a rotation-only one for the
// compass). Colors are linear — the sRGB surface encodes them on write, same as
// the main shader — so pass linear values, not gamma-space ones.

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@group(0) @binding(0) var<uniform> view_proj: mat4x4<f32>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}

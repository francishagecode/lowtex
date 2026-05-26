// src/shaders/main.wgsl
//
// Minimal vertex + fragment shader for v0.1.
//
// Vertex shader: transforms vertex position by view-projection matrix,
//                passes UV and normal through to the fragment.
// Fragment shader: samples the paint texture and applies a tiny bit of
//                  flat shading so faces are visually distinguishable.
//
// PSX effects (vertex snap, affine UVs, palette quantize) come in v0.2.
// Keeping this clean for the proof of paint.

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) uv:       vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
};

@group(0) @binding(0) var<uniform> view_proj: mat4x4<f32>;
@group(0) @binding(1) var paint_tex: texture_2d<f32>;
@group(0) @binding(2) var paint_smp: sampler;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = view_proj * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    out.normal = in.normal;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(paint_tex, paint_smp, in.uv);

    // Cheap flat shading: dot the normal with a fixed light dir.
    let light_dir = normalize(vec3<f32>(0.4, 0.8, 0.5));
    let shade = clamp(dot(normalize(in.normal), light_dir), 0.0, 1.0);
    let lit = mix(0.55, 1.0, shade); // ambient floor so dark sides aren't black

    return vec4<f32>(tex_color.rgb * lit, tex_color.a);
}

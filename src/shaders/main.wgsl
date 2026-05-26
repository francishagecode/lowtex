// src/shaders/main.wgsl
//
// Scene shader with the PSX look (G7). All effects are runtime-toggleable via
// uniform flags — no pipeline switching — by carrying both perspective-correct
// and affine/flat varyings and selecting in the fragment stage.
//
//   - affine UV warp : a @interpolate(linear) UV copy is interpolated WITHOUT
//                      perspective correction, reproducing PSX's warped textures.
//   - vertex snap    : projected NDC.xy is snapped to a low-res grid (the wobble).
//   - flat / Gouraud : a @interpolate(flat) normal copy gives flat shading.
//   - depth fog      : mix toward a fog color by view-space distance.
//   - nearest sample : set on the sampler (renderer.rs).

struct Uniforms {
    view_proj: mat4x4<f32>,
    fog_color: vec4<f32>,
    // x = affine UV (0/1), y = vertex snap (0/1), z = snap grid, w = fog (0/1)
    params: vec4<f32>,
    // x = flat shading (0/1), y = fog start dist, z = fog end dist, w = unused
    params2: vec4<f32>,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(linear) uv_affine: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) world_pos: vec3<f32>,
    @location(4) view_dist: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var paint_tex: texture_2d<f32>;
@group(0) @binding(2) var paint_smp: sampler;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    var clip = u.view_proj * vec4<f32>(in.position, 1.0);

    // Vertex snap: quantize projected NDC.xy to a coarse grid, then scale back by
    // w so the hardware perspective divide reproduces the snapped position.
    if (u.params.y > 0.5 && clip.w > 0.0) {
        let grid = max(u.params.z, 1.0);
        let ndc = clip.xy / clip.w;
        let snapped = round(ndc * grid) / grid;
        clip = vec4<f32>(snapped * clip.w, clip.z, clip.w);
    }

    out.clip_position = clip;
    out.uv = in.uv;
    out.uv_affine = in.uv;
    out.normal = in.normal;
    out.world_pos = in.position; // model == world (no model matrix)
    out.view_dist = clip.w; // perspective w ≈ view-space distance
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // select(false_val, true_val, cond)
    let uv = select(in.uv, in.uv_affine, u.params.x > 0.5);
    let tex_color = textureSample(paint_tex, paint_smp, uv);

    // Flat shading: derive the true geometric face normal from screen-space
    // derivatives of world position, oriented to match the smooth normal.
    let smooth_n = normalize(in.normal);
    var n = smooth_n;
    if (u.params2.x > 0.5) {
        let face_n = normalize(cross(dpdx(in.world_pos), dpdy(in.world_pos)));
        n = select(-face_n, face_n, dot(face_n, smooth_n) >= 0.0);
    }
    let light_dir = normalize(vec3<f32>(0.4, 0.8, 0.5));
    let shade = clamp(dot(n, light_dir), 0.0, 1.0);
    let lit = mix(0.55, 1.0, shade); // ambient floor so dark sides aren't black
    var color = tex_color.rgb * lit;

    // Depth fog: linear blend toward fog color across [start, end].
    if (u.params.w > 0.5) {
        let start = u.params2.y;
        let end = max(u.params2.z, start + 1e-3);
        let f = clamp((in.view_dist - start) / (end - start), 0.0, 1.0);
        color = mix(color, u.fog_color.rgb, f);
    }

    return vec4<f32>(color, tex_color.a);
}

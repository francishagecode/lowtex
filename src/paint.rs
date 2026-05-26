// src/paint.rs
//
// The painting core. Two responsibilities:
//
//   1. Given a screen-space mouse position and a camera, find which point on
//      the mesh the cursor is over (ray/triangle intersection in world space,
//      barycentric interpolation to get the UV).
//
//   2. Write a brush stamp into a CPU-side texture at that UV.
//
// Why CPU-side first? It's the simplest thing that works and lets us validate
// the whole pipeline. v0.2 moves brush stamps to a GPU compute shader for
// performance (essential once textures are 1024² and brushes are large).
//
// The triangle intersection is the standard Möller–Trumbore algorithm.

use glam::{Mat4, Vec2, Vec3};

use crate::mesh::Mesh;

/// Live brush settings, driven by the UI panel (G3). Color is sRGB in 0..1 to
/// match egui's color picker; radius is in texels.
#[derive(Clone, Copy)]
pub struct Brush {
    pub color: [f32; 3],
    pub radius: f32,
    pub opacity: f32,  // 0..1, per-stamp coverage
    pub hardness: f32, // 0..1, fraction of the radius that is fully opaque
}

impl Default for Brush {
    fn default() -> Self {
        Self {
            color: [0.86, 0.24, 0.31], // the retro red from v0.1
            radius: 4.0,
            opacity: 1.0,
            hardness: 0.8,
        }
    }
}

impl Brush {
    /// Brush color as sRGB u8, for writing into the RGBA8 texture.
    pub fn color_u8(&self) -> [u8; 3] {
        [
            (self.color[0].clamp(0.0, 1.0) * 255.0).round() as u8,
            (self.color[1].clamp(0.0, 1.0) * 255.0).round() as u8,
            (self.color[2].clamp(0.0, 1.0) * 255.0).round() as u8,
        ]
    }
}

pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>, // RGBA8, length = width * height * 4
}

impl Texture {
    pub fn new(width: u32, height: u32, fill: [u8; 4]) -> Self {
        let pixels = fill
            .iter()
            .copied()
            .cycle()
            .take((width * height * 4) as usize)
            .collect();
        Self {
            width,
            height,
            pixels,
        }
    }

    /// Return a copy resampled to `new_w` × `new_h` with nearest-neighbor
    /// sampling (no blur — PSX-correct, and the right choice for chunky textures).
    pub fn resampled(&self, new_w: u32, new_h: u32) -> Texture {
        let mut out = Texture::new(new_w, new_h, [0, 0, 0, 255]);
        for y in 0..new_h {
            let sy = (y * self.height / new_h).min(self.height - 1);
            for x in 0..new_w {
                let sx = (x * self.width / new_w).min(self.width - 1);
                let si = ((sy * self.width + sx) * 4) as usize;
                let di = ((y * new_w + x) * 4) as usize;
                out.pixels[di..di + 4].copy_from_slice(&self.pixels[si..si + 4]);
            }
        }
        out
    }

    /// Stamp a brush at a UV coordinate with per-stroke coverage accumulation.
    ///
    /// `base` is the texture as it was when the stroke began; `coverage` is the
    /// stroke's accumulated per-texel coverage (0..1). Overlapping stamps within
    /// one stroke take the *max* coverage rather than adding, so a slow drag or a
    /// dense interpolation doesn't double-darken — the stroke tops out at the
    /// brush's opacity. UV is in [0,1] with V=0 at the top.
    pub fn stamp_stroke(&mut self, uv: Vec2, brush: &Brush, base: &[u8], coverage: &mut [f32]) {
        let cx = uv.x * self.width as f32;
        let cy = uv.y * self.height as f32;
        let radius = brush.radius.max(0.5);
        let r = radius.ceil() as i32;
        let color = brush.color_u8();

        for dy in -r..=r {
            for dx in -r..=r {
                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                let a = brush.opacity * falloff(dist / radius, brush.hardness);
                if a <= 0.0 {
                    continue;
                }
                let x = cx as i32 + dx;
                let y = cy as i32 + dy;
                if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
                    continue;
                }
                let texel = (y as u32 * self.width + x as u32) as usize;
                if a <= coverage[texel] {
                    continue; // already covered at least this much this stroke
                }
                coverage[texel] = a;
                let cov = a;
                let i = texel * 4;
                for (c, &src) in color.iter().enumerate() {
                    let dst = base[i + c] as f32;
                    self.pixels[i + c] = (dst * (1.0 - cov) + src as f32 * cov)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                }
                self.pixels[i + 3] = 255;
            }
        }
    }
}

/// Radial brush coverage at normalized distance `d` (0 at center, 1 at the rim)
/// for a given `hardness` (1 = hard edge, 0 = fully soft). Returns 0..1.
fn falloff(d: f32, hardness: f32) -> f32 {
    if d >= 1.0 {
        return 0.0;
    }
    let h = hardness.clamp(0.0, 1.0);
    if d <= h {
        1.0
    } else {
        // Linear ramp from the hard core out to the rim.
        1.0 - (d - h) / (1.0 - h).max(1e-4)
    }
}

/// A ray in world space.
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    /// Build a world-space ray from a pixel on the screen, "unprojecting" through
    /// the inverse view-projection.
    pub fn from_screen(mouse_px: Vec2, screen_size: Vec2, inv_view_proj: Mat4) -> Self {
        // Normalize to clip space: x ∈ [-1, 1], y ∈ [-1, 1] with y flipped
        // because screen Y goes down but clip-space Y goes up.
        let ndc_x = (mouse_px.x / screen_size.x) * 2.0 - 1.0;
        let ndc_y = 1.0 - (mouse_px.y / screen_size.y) * 2.0;

        // Unproject a near point and a far point, ray = far - near.
        let near = inv_view_proj.project_point3(Vec3::new(ndc_x, ndc_y, 0.0));
        let far = inv_view_proj.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));

        Self {
            origin: near,
            direction: (far - near).normalize(),
        }
    }
}

/// Möller–Trumbore ray/triangle intersection.
/// Returns (t, u, v) where t is distance along ray, (u, v) are barycentric
/// coordinates suitable for interpolating vertex attributes:
///   point = (1 - u - v) * v0 + u * v1 + v * v2
pub(crate) fn intersect_triangle(
    ray: &Ray,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
) -> Option<(f32, f32, f32)> {
    const EPS: f32 = 1e-7;
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = ray.direction.cross(edge2);
    let a = edge1.dot(h);
    if a.abs() < EPS {
        return None; // ray parallel to triangle
    }
    let f = 1.0 / a;
    let s = ray.origin - v0;
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(edge1);
    let v = f * ray.direction.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * edge2.dot(q);
    if t > EPS {
        Some((t, u, v))
    } else {
        None
    }
}

/// Cast a ray against every triangle in the mesh. Returns the UV at the closest
/// hit, or None if the ray misses everything.
///
/// O(n) per click. Superseded for real picking by the BVH (G5); kept as the
/// brute-force oracle the BVH is tested against.
#[allow(dead_code)]
pub fn pick_uv(ray: &Ray, mesh: &Mesh) -> Option<Vec2> {
    let mut best: Option<(f32, Vec2)> = None;

    for tri in mesh.indices.chunks_exact(3) {
        let v0 = mesh.vertices[tri[0] as usize];
        let v1 = mesh.vertices[tri[1] as usize];
        let v2 = mesh.vertices[tri[2] as usize];

        let p0 = Vec3::from(v0.position);
        let p1 = Vec3::from(v1.position);
        let p2 = Vec3::from(v2.position);

        if let Some((t, u, v)) = intersect_triangle(ray, p0, p1, p2) {
            if best.is_none_or(|(bt, _)| t < bt) {
                let w = 1.0 - u - v;
                let uv = Vec2::from(v0.uv) * w + Vec2::from(v1.uv) * u + Vec2::from(v2.uv) * v;
                best = Some((t, uv));
            }
        }
    }

    best.map(|(_, uv)| uv)
}

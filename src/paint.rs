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
#[derive(Clone, Copy, PartialEq)]
pub struct Brush {
    pub color: [f32; 3],
    pub radius: f32,
    pub opacity: f32,        // 0..1, per-stamp coverage
    pub hardness: f32,       // 0..1, fraction of the radius that is fully opaque
    pub snap_to_texel: bool, // round each dab's center to the grid for crisp, grid-aligned PSX strokes
    pub snap_grid: f32,      // grid cell size in texels (only meaningful when snap_to_texel is on)
    pub erase: bool, // remove paint (lower alpha toward transparent) instead of laying color
}

impl Default for Brush {
    fn default() -> Self {
        Self {
            color: [0.86, 0.24, 0.31], // the retro red from v0.1
            radius: 4.0,
            opacity: 1.0,
            hardness: 0.8,
            snap_to_texel: false,
            snap_grid: 4.0,
            erase: false,
        }
    }
}

impl Brush {
    /// A copy of this brush with pen-tablet `pressure` (0..1) applied to the
    /// enabled axes. `size`/`opacity` toggle which axes respond; `min_size` is the
    /// radius fraction at zero pressure, so a light touch still leaves a visible
    /// mark instead of collapsing to nothing. A plain mouse reports full pressure
    /// (1.0), which leaves the brush unchanged — so this is a no-op without a pen.
    pub fn with_pressure(self, pressure: f32, size: bool, opacity: bool, min_size: f32) -> Brush {
        let p = pressure.clamp(0.0, 1.0);
        let mut b = self;
        if size {
            let min = min_size.clamp(0.0, 1.0);
            b.radius = (b.radius * (min + (1.0 - min) * p)).max(0.5);
        }
        if opacity {
            b.opacity = (b.opacity * p).clamp(0.0, 1.0);
        }
        b
    }

    /// Brush color as sRGB u8, for writing into the RGBA8 texture.
    pub fn color_u8(&self) -> [u8; 3] {
        [
            (self.color[0].clamp(0.0, 1.0) * 255.0).round() as u8,
            (self.color[1].clamp(0.0, 1.0) * 255.0).round() as u8,
            (self.color[2].clamp(0.0, 1.0) * 255.0).round() as u8,
        ]
    }
}

#[derive(Clone)]
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
}

/// Composite `src` (sRGB) into texel `texel` over the stroke's `base` snapshot at
/// coverage `a` (0..1), raising alpha toward opaque. An opaque base (255) stays
/// opaque; a transparent layer (0) gains paint where stamped. Shared by every
/// brush path — solid color, material image, mask, and the cross-face surface
/// splat — so they all composite identically. `texel` must be in range.
pub(crate) fn blend_texel(pixels: &mut [u8], base: &[u8], texel: usize, src: [u8; 3], a: f32) {
    let i = texel * 4;
    blend4(&mut pixels[i..i + 4], &base[i..i + 4], src, a);
}

/// The per-texel blend of [`blend_texel`] on standalone 4-byte `dst`/`base` slices, so a
/// caller that has already split a buffer into rows (the rayon GPU-coverage resolve) shares
/// the exact same math — no parity drift between the per-dab and per-row paths.
pub(crate) fn blend4(dst: &mut [u8], base: &[u8], src: [u8; 3], a: f32) {
    for c in 0..3 {
        let d = base[c] as f32;
        dst[c] = (d * (1.0 - a) + src[c] as f32 * a).round().clamp(0.0, 255.0) as u8;
    }
    let base_a = base[3] as f32;
    dst[3] = (base_a * (1.0 - a) + 255.0 * a).round() as u8;
}

/// Erase texel `texel` over the stroke's `base` snapshot at coverage `a` (0..1),
/// lowering alpha toward transparent — the inverse of [`blend_texel`]. RGB is left at
/// the base color (it's invisible where alpha is 0, and keeping it avoids a black
/// fringe if the layer is later flattened). `texel` must be in range.
pub(crate) fn erase_texel(pixels: &mut [u8], base: &[u8], texel: usize, a: f32) {
    let i = texel * 4;
    erase4(&mut pixels[i..i + 4], &base[i..i + 4], a);
}

/// The per-texel erase of [`erase_texel`] on standalone 4-byte slices (see [`blend4`]).
pub(crate) fn erase4(dst: &mut [u8], base: &[u8], a: f32) {
    dst[0..3].copy_from_slice(&base[0..3]);
    let base_a = base[3] as f32;
    dst[3] = (base_a * (1.0 - a)).round().clamp(0.0, 255.0) as u8;
}

/// A rectangular region of texels, `[x0, x1) × [y0, y1)` (exclusive upper bound),
/// in texture space. Used to bound a stroke's display refresh to the texels a brush
/// actually touched, so paint cost is proportional to brush area, not the whole
/// texture (the dirty-rectangle optimization).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TexRect {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

impl TexRect {
    /// Smallest rect covering both.
    pub fn union(self, other: TexRect) -> TexRect {
        TexRect {
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
        }
    }

    /// Grow by `pad` texels on every side, clamped to `[0, size)`.
    pub fn expanded(self, pad: u32, size: u32) -> TexRect {
        TexRect {
            x0: self.x0.saturating_sub(pad),
            y0: self.y0.saturating_sub(pad),
            x1: (self.x1 + pad).min(size),
            y1: (self.y1 + pad).min(size),
        }
    }

    pub fn width(self) -> u32 {
        self.x1 - self.x0
    }

    pub fn height(self) -> u32 {
        self.y1 - self.y0
    }

    /// True if `(x, y)` lies inside the rect.
    pub fn contains(self, x: u32, y: u32) -> bool {
        x >= self.x0 && x < self.x1 && y >= self.y0 && y < self.y1
    }
}

/// Radial brush coverage at normalized distance `d` (0 at center, 1 at the rim)
/// for a given `hardness` (1 = hard edge, 0 = fully soft). Returns 0..1.
/// Shared by the solid-color stamp here and the texture-brush stamp in `material`.
pub(crate) fn falloff(d: f32, hardness: f32) -> f32 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erase_lowers_alpha_by_coverage() {
        // An opaque painted texel...
        let base = [200u8, 100, 50, 255];
        let mut px = base;

        // ...fully erased becomes transparent.
        erase_texel(&mut px, &base, 0, 1.0);
        assert_eq!(px[3], 0, "full coverage clears alpha");

        // ...half erased keeps half its alpha; RGB stays at the base color.
        let mut px = base;
        erase_texel(&mut px, &base, 0, 0.5);
        assert_eq!(px[3], 128, "half coverage halves alpha");
        assert_eq!(&px[0..3], &base[0..3], "erase leaves RGB at the base color");
    }

    #[test]
    fn full_pressure_leaves_the_brush_unchanged() {
        // A mouse reports pressure 1.0, so painting must be identical to the raw
        // brush regardless of which axes are enabled.
        let b = Brush {
            radius: 10.0,
            opacity: 0.5,
            ..Brush::default()
        };
        let scaled = b.with_pressure(1.0, true, true, 0.1);
        assert_eq!(scaled.radius, b.radius);
        assert_eq!(scaled.opacity, b.opacity);
    }

    #[test]
    fn pressure_scales_only_enabled_axes() {
        let b = Brush {
            radius: 20.0,
            opacity: 0.8,
            ..Brush::default()
        };

        // Size only: radius interpolates from min_size*radius (here 0) up to radius;
        // opacity is untouched.
        let size = b.with_pressure(0.5, true, false, 0.0);
        assert!((size.radius - 10.0).abs() < 1e-4, "half pressure → half radius");
        assert_eq!(size.opacity, b.opacity, "opacity left alone when size-only");

        // Opacity only: opacity scales linearly; radius is untouched.
        let opa = b.with_pressure(0.25, false, true, 0.0);
        assert_eq!(opa.radius, b.radius, "radius left alone when opacity-only");
        assert!((opa.opacity - 0.2).abs() < 1e-4, "quarter pressure → quarter opacity");
    }

    #[test]
    fn min_size_floors_the_radius_at_low_pressure() {
        let b = Brush {
            radius: 30.0,
            ..Brush::default()
        };
        // At zero pressure with a 0.1 floor, radius is 10% of nominal — visible, not zero.
        let near_zero = b.with_pressure(0.0, true, false, 0.1);
        assert!((near_zero.radius - 3.0).abs() < 1e-4);
        // And never collapses below the 0.5-texel hard floor even with min_size 0.
        let tiny = b.with_pressure(0.0, true, false, 0.0);
        assert!(tiny.radius >= 0.5);
    }

    #[test]
    fn erase_then_blend_are_inverse_at_full_coverage() {
        // Blend to opaque, then erase fully → back to transparent.
        let transparent = [0u8, 0, 0, 0];
        let mut px = transparent;
        blend_texel(&mut px, &transparent, 0, [255, 0, 0], 1.0);
        assert_eq!(px[3], 255);
        let painted = px;
        erase_texel(&mut px, &painted, 0, 1.0);
        assert_eq!(px[3], 0);
    }
}

// src/material.rs
//
// Material textures (Substance-style): load an image — a brick, moss, metal, etc.
// — and paint or fill it onto a layer instead of a flat brush color. Combined with
// the layer reveal mask (G11) and mask-from-map (AO/curvature, G20) this is how you
// get "moss in the crevices": fill a layer with the moss material, then set its
// mask from Cavities.
//
// Sampling is UV-tiled: a texel's own UV (its position in the atlas) times `tile`,
// wrapped, indexes the material. So the material repeats `tile` times across the
// 0–1 UV space and rides along with whatever unwrap the mesh has.

use glam::Vec2;

use crate::paint::{falloff, Brush, Texture};

/// A loaded source texture used as paint.
#[derive(Clone)]
pub struct Material {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>, // RGBA8
}

impl Material {
    /// Load an image file as a material (decoded to RGBA8).
    pub fn load(path: &str) -> Result<Material, String> {
        let img = image::open(path).map_err(|e| format!("failed to open material: {e}"))?;
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();
        Ok(Material {
            width,
            height,
            pixels: rgba.into_raw(),
        })
    }

    /// Sample the material at UV `(u, v)` repeated `tile` times, nearest-neighbour
    /// (PSX-correct, no blur). Inputs outside 0..1 wrap.
    pub fn sample(&self, u: f32, v: f32, tile: f32) -> [u8; 4] {
        let wrap = |c: f32| {
            let t = (c * tile).rem_euclid(1.0);
            t.clamp(0.0, 0.999_999)
        };
        let mx = (wrap(u) * self.width as f32) as u32;
        let my = (wrap(v) * self.height as f32) as u32;
        let i = ((my * self.width + mx) * 4) as usize;
        [
            self.pixels[i],
            self.pixels[i + 1],
            self.pixels[i + 2],
            self.pixels[i + 3],
        ]
    }

    /// Stamp this material into `dst` at UV `uv`, gated by the brush's radius and
    /// falloff with per-stroke coverage accumulation — the texture-brush counterpart
    /// to `Texture::stamp_stroke`. The source color for each painted texel is the
    /// material sampled at *that texel's own* UV (tiled), so a stroke reveals exactly
    /// what a full `fill` would put there: brushing is "fill, but only where you drag".
    ///
    /// `base` is the layer as it was when the stroke began and `coverage` the stroke's
    /// per-texel coverage, both shared with the solid-color path so overlap within one
    /// stroke takes the max coverage (no double-application). UV has V=0 at the top.
    pub fn stamp(
        &self,
        dst: &mut Texture,
        uv: Vec2,
        brush: &Brush,
        tile: f32,
        base: &[u8],
        coverage: &mut [f32],
    ) {
        let cx = uv.x * dst.width as f32;
        let cy = uv.y * dst.height as f32;
        let radius = brush.radius.max(0.5);
        let r = radius.ceil() as i32;

        for dy in -r..=r {
            for dx in -r..=r {
                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                let a = brush.opacity * falloff(dist / radius, brush.hardness);
                if a <= 0.0 {
                    continue;
                }
                let x = cx as i32 + dx;
                let y = cy as i32 + dy;
                if x < 0 || y < 0 || x >= dst.width as i32 || y >= dst.height as i32 {
                    continue;
                }
                let texel = (y as u32 * dst.width + x as u32) as usize;
                if a <= coverage[texel] {
                    continue; // already covered at least this much this stroke
                }
                coverage[texel] = a;
                let cov = a;
                let src = self.sample(
                    x as f32 / dst.width as f32,
                    y as f32 / dst.height as f32,
                    tile,
                );
                let i = texel * 4;
                for c in 0..3 {
                    let under = base[i + c] as f32;
                    dst.pixels[i + c] = (under * (1.0 - cov) + src[c] as f32 * cov)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                }
                // Raise alpha toward opaque by coverage, like the color stamp.
                let base_a = base[i + 3] as f32;
                dst.pixels[i + 3] = (base_a * (1.0 - cov) + 255.0 * cov).round() as u8;
            }
        }
    }

    /// Fill `dst` (a `size`×`size` layer texture) with this material, UV-tiled. Each
    /// destination texel maps to UV `(x/size, y/size)`. Fully opaque output.
    pub fn fill(&self, dst: &mut Texture, tile: f32) {
        let size = dst.width;
        for y in 0..size {
            for x in 0..size {
                let c = self.sample(x as f32 / size as f32, y as f32 / size as f32, tile);
                let i = ((y * size + x) * 4) as usize;
                dst.pixels[i] = c[0];
                dst.pixels[i + 1] = c[1];
                dst.pixels[i + 2] = c[2];
                dst.pixels[i + 3] = 255;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checker_material() -> Material {
        // 2×2 RG/BW so sampling is easy to reason about.
        Material {
            width: 2,
            height: 2,
            pixels: vec![
                255, 0, 0, 255, 0, 255, 0, 255, // red, green
                0, 0, 255, 255, 255, 255, 255, 255, // blue, white
            ],
        }
    }

    #[test]
    fn sample_wraps_and_picks_cells() {
        let m = checker_material();
        // tile=1: u,v in [0,0.5) → cell (0,0)=red; [0.5,1) → far cells.
        assert_eq!(m.sample(0.1, 0.1, 1.0), [255, 0, 0, 255]);
        assert_eq!(m.sample(0.9, 0.1, 1.0), [0, 255, 0, 255]); // x→col1
                                                               // Wrapping: u=1.1 ≡ 0.1.
        assert_eq!(m.sample(1.1, 0.1, 1.0), [255, 0, 0, 255]);
    }

    #[test]
    fn stamp_reveals_what_fill_would_put_there() {
        // A full-coverage stamp over a texel must equal the fill result there: the
        // texture brush is "fill, but only where you drag".
        let m = checker_material();
        let size = 8u32;
        let mut filled = Texture::new(size, size, [0, 0, 0, 0]);
        m.fill(&mut filled, 2.0);

        let mut painted = Texture::new(size, size, [0, 0, 0, 255]);
        let base = painted.pixels.clone();
        let mut coverage = vec![0.0; (size * size) as usize];
        // Hard, opaque brush centered so its full-coverage core covers texel (4,4).
        let brush = Brush {
            color: [0.0, 0.0, 0.0],
            radius: 3.0,
            opacity: 1.0,
            hardness: 1.0,
        };
        let center = Vec2::new(4.5 / size as f32, 4.5 / size as f32);
        m.stamp(&mut painted, center, &brush, 2.0, &base, &mut coverage);

        let i = ((4 * size + 4) * 4) as usize;
        assert_eq!(painted.pixels[i..i + 4], filled.pixels[i..i + 4]);
    }

    #[test]
    fn stamp_leaves_outside_the_footprint_untouched() {
        let m = checker_material();
        let size = 16u32;
        let mut painted = Texture::new(size, size, [9, 9, 9, 255]);
        let base = painted.pixels.clone();
        let mut coverage = vec![0.0; (size * size) as usize];
        let brush = Brush {
            color: [0.0, 0.0, 0.0],
            radius: 2.0,
            opacity: 1.0,
            hardness: 1.0,
        };
        // Stamp near one corner; the opposite corner is well outside the radius.
        m.stamp(
            &mut painted,
            Vec2::new(0.1, 0.1),
            &brush,
            2.0,
            &base,
            &mut coverage,
        );
        let far = (((size - 1) * size + (size - 1)) * 4) as usize;
        assert_eq!(painted.pixels[far..far + 4], [9, 9, 9, 255]);
    }

    #[test]
    fn fill_is_opaque_and_tiles() {
        let m = checker_material();
        let mut tex = Texture::new(8, 8, [0, 0, 0, 0]);
        m.fill(&mut tex, 2.0);
        // Every texel opaque and one of the four material colors.
        let allowed = [[255u8, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]];
        for px in tex.pixels.chunks_exact(4) {
            assert_eq!(px[3], 255);
            assert!(
                allowed.contains(&[px[0], px[1], px[2]]),
                "unexpected {px:?}"
            );
        }
    }
}

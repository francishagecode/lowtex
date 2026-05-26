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

use crate::paint::Texture;

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

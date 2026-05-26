// src/palette.rs
//
// Constrained color palettes (G8): an ordered list of sRGB colors. The renderer
// applies these to the *paint texture* on the CPU (`quantize_rgba`) — nearest
// palette color + optional Bayer dither — so the model and the exported PNG show
// the quantized result (WYSIWYG). Includes a few built-ins and median-cut
// generation from an arbitrary image, so an artist can lift a palette off
// reference art.
//
// Colors are sRGB in 0..1 (matching the UI color pickers); quantization compares
// in sRGB byte space, which is fine for chunky low-res textures.

/// An ordered palette. Empty is treated as "no constraint" by the renderer.
#[derive(Clone)]
pub struct Palette {
    pub name: String,
    pub colors: Vec<[f32; 3]>,
}

impl Palette {
    fn from_hex(name: &str, hexes: &[u32]) -> Self {
        let colors = hexes
            .iter()
            .map(|h| {
                [
                    ((h >> 16) & 0xff) as f32 / 255.0,
                    ((h >> 8) & 0xff) as f32 / 255.0,
                    (h & 0xff) as f32 / 255.0,
                ]
            })
            .collect();
        Self {
            name: name.to_string(),
            colors,
        }
    }

    /// The built-in palettes offered in the UI.
    pub fn builtins() -> Vec<Palette> {
        vec![
            // PICO-8 — the canonical chunky 16, great default for the vibe.
            Palette::from_hex(
                "PICO-8 (16)",
                &[
                    0x000000, 0x1D2B53, 0x7E2553, 0x008751, 0xAB5236, 0x5F574F, 0xC2C3C7, 0xFFF1E8,
                    0xFF004D, 0xFFA300, 0xFFEC27, 0x00E436, 0x29ADFF, 0x83769C, 0xFF77A8, 0xFFCCAA,
                ],
            ),
            // Game Boy DMG — 4 greens.
            Palette::from_hex("Game Boy (4)", &[0x0F380F, 0x306230, 0x8BAC0F, 0x9BBC0F]),
            // 4-step grayscale — extreme PSX constraint.
            Palette::from_hex("Grayscale (4)", &[0x000000, 0x555555, 0xAAAAAA, 0xFFFFFF]),
        ]
    }

    /// Generate a palette of up to `n` colors from RGBA8 image pixels using
    /// median-cut quantization. Large images are subsampled for speed.
    pub fn from_image_median_cut(rgba: &[u8], n: usize) -> Palette {
        let n = n.clamp(2, 256);
        // Collect (subsampled) opaque-ish pixels as [u8;3].
        let px_count = rgba.len() / 4;
        let step = (px_count / 20_000).max(1); // cap at ~20k samples
        let mut samples: Vec<[u8; 3]> = Vec::new();
        for i in (0..px_count).step_by(step) {
            let b = i * 4;
            samples.push([rgba[b], rgba[b + 1], rgba[b + 2]]);
        }
        if samples.is_empty() {
            return Palette {
                name: "Generated".into(),
                colors: vec![[0.0, 0.0, 0.0]],
            };
        }

        // Median-cut: repeatedly split the box with the widest channel at its median.
        let mut boxes: Vec<Vec<[u8; 3]>> = vec![samples];
        while boxes.len() < n {
            // Find the splittable box with the largest channel range.
            let mut best: Option<(usize, usize, u8)> = None; // (box, axis, range)
            for (bi, b) in boxes.iter().enumerate() {
                if b.len() < 2 {
                    continue;
                }
                for axis in 0..3 {
                    let (mut lo, mut hi) = (255u8, 0u8);
                    for p in b {
                        lo = lo.min(p[axis]);
                        hi = hi.max(p[axis]);
                    }
                    let range = hi - lo;
                    if best.is_none_or(|(_, _, r)| range > r) {
                        best = Some((bi, axis, range));
                    }
                }
            }
            let Some((bi, axis, range)) = best else { break };
            if range == 0 {
                break; // nothing left to split
            }
            let mut b = boxes.swap_remove(bi);
            b.sort_unstable_by_key(|p| p[axis]);
            let mid = b.len() / 2;
            let right = b.split_off(mid);
            boxes.push(b);
            boxes.push(right);
        }

        // Average each box to its representative color.
        let colors = boxes
            .iter()
            .filter(|b| !b.is_empty())
            .map(|b| {
                let mut sum = [0u64; 3];
                for p in b.iter() {
                    for c in 0..3 {
                        sum[c] += p[c] as u64;
                    }
                }
                let len = b.len() as u64;
                [
                    (sum[0] / len) as f32 / 255.0,
                    (sum[1] / len) as f32 / 255.0,
                    (sum[2] / len) as f32 / 255.0,
                ]
            })
            .collect();
        Palette {
            name: format!("Generated ({n})"),
            colors,
        }
    }

    /// Nearest palette color to an sRGB color (squared distance).
    pub fn nearest(&self, c: [f32; 3]) -> [f32; 3] {
        self.colors
            .iter()
            .copied()
            .min_by(|a, b| dist2(*a, c).total_cmp(&dist2(*b, c)))
            .unwrap_or(c)
    }

    /// Nearest palette color as u8, with an optional pre-quantize bias (for
    /// ordered dithering).
    fn nearest_u8(&self, rgb: [u8; 3], bias: f32) -> [u8; 3] {
        let c = [
            (rgb[0] as f32 / 255.0 + bias).clamp(0.0, 1.0),
            (rgb[1] as f32 / 255.0 + bias).clamp(0.0, 1.0),
            (rgb[2] as f32 / 255.0 + bias).clamp(0.0, 1.0),
        ];
        let p = self.nearest(c);
        [
            (p[0] * 255.0).round() as u8,
            (p[1] * 255.0).round() as u8,
            (p[2] * 255.0).round() as u8,
        ]
    }

    /// Quantize an RGBA8 image in place to this palette, optionally adding 4×4
    /// ordered (Bayer) dithering to break up banding. Works directly in sRGB
    /// byte space — fine for chunky textures and keeps export WYSIWYG. Alpha is
    /// left untouched. `width` is needed to index the dither matrix per texel.
    pub fn quantize_rgba(&self, pixels: &mut [u8], width: u32, dither: bool, strength: f32) {
        if self.colors.is_empty() {
            return;
        }
        for (i, px) in pixels.chunks_exact_mut(4).enumerate() {
            let bias = if dither {
                let x = (i as u32 % width) as usize;
                let y = (i as u32 / width) as usize;
                (bayer4(x, y) - 0.5) * strength
            } else {
                0.0
            };
            let q = self.nearest_u8([px[0], px[1], px[2]], bias);
            px[0] = q[0];
            px[1] = q[1];
            px[2] = q[2];
        }
    }
}

/// 4×4 Bayer threshold (normalized 0..1), used for ordered dithering.
fn bayer4(x: usize, y: usize) -> f32 {
    const M: [[u8; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
    (M[y % 4][x % 4] as f32 + 0.5) / 16.0
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_cut_separates_distinct_colors() {
        // An image of pure red / green / blue / white in quarters.
        let blocks = [[255u8, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]];
        let mut rgba = Vec::new();
        for c in blocks {
            for _ in 0..256 {
                rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        let pal = Palette::from_image_median_cut(&rgba, 4);
        assert_eq!(pal.colors.len(), 4);
        // Each source block should have a near-exact representative in the palette.
        for c in blocks {
            let target = [
                c[0] as f32 / 255.0,
                c[1] as f32 / 255.0,
                c[2] as f32 / 255.0,
            ];
            let near = pal.nearest(target);
            let d = dist2(near, target);
            assert!(
                d < 0.01,
                "no palette color near {target:?} (closest {near:?})"
            );
        }
    }

    #[test]
    fn builtins_are_nonempty() {
        for p in Palette::builtins() {
            assert!(!p.colors.is_empty(), "{} has no colors", p.name);
        }
    }
}

// src/palette.rs
//
// Constrained color palettes (G8): an ordered list of sRGB colors used by the
// quantize post-process (renderer + post.wgsl). Includes a few built-ins and
// median-cut generation from an arbitrary image, so an artist can lift a palette
// straight off reference art.
//
// Colors are stored as sRGB in 0..1 (matching the UI color pickers). The renderer
// converts to linear before upload, since the post pass samples the (sRGB) scene
// texture as linear.

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

    /// Nearest palette color to an sRGB color (squared distance). Used by export
    /// (G23) and as a CPU oracle; the live view quantizes on the GPU.
    #[allow(dead_code)]
    pub fn nearest(&self, c: [f32; 3]) -> [f32; 3] {
        self.colors
            .iter()
            .copied()
            .min_by(|a, b| dist2(*a, c).total_cmp(&dist2(*b, c)))
            .unwrap_or(c)
    }
}

#[allow(dead_code)]
fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

/// sRGB component (0..1) → linear.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
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

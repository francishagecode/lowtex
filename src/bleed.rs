// src/bleed.rs
//
// Island bleed / UV gutter dilation (G18). Nearest-neighbour sampling at a UV
// island edge can land on a texel just outside the island — the unpainted gutter
// — and reveal background, leaving a visible seam on the model. Especially at
// 64×64, where one texel is a big chunk of surface.
//
// Fix: after compositing, grow the painted texels outward into the gutter by a few
// pixels, so whatever the sampler grabs near an edge is still the island's color.
// `coverage` marks which texels a UV triangle actually covers; `dilate` floods the
// rest outward from there.

use glam::Vec2;

use crate::mesh::Mesh;

/// Per-texel boolean: does any UV triangle cover this texel's center? Row-major,
/// `size`×`size`, V down (matching the paint texture).
pub fn coverage(mesh: &Mesh, size: u32) -> Vec<bool> {
    let mut covered = vec![false; (size * size) as usize];
    for tri in mesh.indices.chunks_exact(3) {
        let t: [Vec2; 3] = [
            Vec2::from(mesh.vertices[tri[0] as usize].uv) * size as f32,
            Vec2::from(mesh.vertices[tri[1] as usize].uv) * size as f32,
            Vec2::from(mesh.vertices[tri[2] as usize].uv) * size as f32,
        ];
        let area = edge(t[0], t[1], t[2]);
        if area.abs() < 1e-6 {
            continue;
        }
        let min_x = t[0].x.min(t[1].x).min(t[2].x).floor().max(0.0) as i32;
        let max_x = t[0].x.max(t[1].x).max(t[2].x).ceil().min(size as f32) as i32;
        let min_y = t[0].y.min(t[1].y).min(t[2].y).floor().max(0.0) as i32;
        let max_y = t[0].y.max(t[1].y).max(t[2].y).ceil().min(size as f32) as i32;
        for y in min_y..max_y {
            for x in min_x..max_x {
                let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let w0 = edge(t[1], t[2], p) / area;
                let w1 = edge(t[2], t[0], p) / area;
                let w2 = edge(t[0], t[1], p) / area;
                if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                    covered[(y as u32 * size + x as u32) as usize] = true;
                }
            }
        }
    }
    covered
}

fn edge(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Grow the colors of `covered` texels outward into uncovered ones by `pad`
/// rings (4-neighbour flood). `pixels` is RGBA8 `size`×`size`; `covered` is the
/// island-coverage mask (not mutated). Cheap and in-place.
pub fn dilate(pixels: &mut [u8], covered: &[bool], size: u32, pad: u32) {
    if pad == 0 {
        return;
    }
    // `valid[i]` = texel i already has a color to spread (covered, or filled by a
    // previous ring).
    let mut valid = covered.to_vec();
    let w = size as i32;
    for _ in 0..pad {
        let frozen = valid.clone();
        for y in 0..size as i32 {
            for x in 0..size as i32 {
                let i = (y * w + x) as usize;
                if frozen[i] {
                    continue;
                }
                // Take the first valid 4-neighbour's color.
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w || ny >= w {
                        continue;
                    }
                    let ni = (ny * w + nx) as usize;
                    if frozen[ni] {
                        let (s, d) = (ni * 4, i * 4);
                        pixels.copy_within(s..s + 4, d);
                        valid[i] = true;
                        break;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dilate_grows_into_the_gutter() {
        // 4×4: only texel (1,1) is covered (red); one ring of dilation should fill
        // its 4-neighbours with red.
        let size = 4;
        let mut px = vec![0u8; (size * size * 4) as usize];
        let mut covered = vec![false; (size * size) as usize];
        let c = (size + 1) as usize;
        covered[c] = true;
        px[c * 4..c * 4 + 4].copy_from_slice(&[255, 0, 0, 255]);

        dilate(&mut px, &covered, size, 1);

        for (nx, ny) in [(1, 0), (1, 2), (0, 1), (2, 1)] {
            let i = (ny * size + nx) as usize;
            assert_eq!(
                &px[i * 4..i * 4 + 4],
                &[255, 0, 0, 255],
                "neighbour ({nx},{ny}) not bled"
            );
        }
    }

    #[test]
    fn cube_coverage_is_mostly_filled() {
        // The cube's box UVs tile the whole 0–1 space, so coverage is near-total.
        let cov = coverage(&Mesh::cube(), 64);
        let filled = cov.iter().filter(|&&c| c).count();
        assert!(
            filled > cov.len() * 3 / 4,
            "cube UV coverage too low: {filled}"
        );
    }
}

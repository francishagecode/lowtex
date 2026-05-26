// src/bake.rs
//
// Mesh-aware maps baked into UV/texture space (the moat, principle #4). For each
// texel that a triangle covers we know the surface point's world position and
// normal, from which we bake:
//
//   - ao        : ambient occlusion — fraction of a cosine-weighted hemisphere
//                 blocked by other geometry within a local radius (dark crevices).
//   - curvature : signed surface convexity in roughly [-1, 1] — the smooth-vs-face
//                 normal divergence (peaks on edges/corners) signed by whether the
//                 surface bulges out (convex, > 0, an edge) or pinches in (concave,
//                 < 0, a crease).
//
// These two channels are *inputs*, in the Substance-Painter sense: rather than
// each producing one fixed layer, `MeshMaps::sample` reads a `MapSource` through a
// `Levels` remap (invert / contrast / strength) into a 0..1 weight that drives
// either a generated tint layer or any layer's reveal mask. The AO suite ("Darken
// (AO)", "Highlights"), the Dirt and Edge-wear presets, and "mask from map" are all
// the same path with different source + color + blend.

use std::collections::{HashMap, HashSet};

use glam::{Vec2, Vec3};

use crate::bvh::Bvh;
use crate::mesh::Mesh;

/// Per-texel baked maps, all `size`×`size`, row-major (V down, matching paint).
pub struct MeshMaps {
    pub size: u32,
    pub ao: Vec<f32>,
    /// Signed convexity, ≈[-1, 1]: > 0 on convex edges, < 0 in concave creases.
    pub curvature: Vec<f32>,
    pub mask: Vec<bool>, // true where a triangle covered the texel
}

/// Hemisphere ray count per texel for AO. Modest — bakes stay sub-second.
const AO_SAMPLES: u32 = 24;

/// Scales raw convexity (≈0.4 at a cube corner) up toward 1 for edge effects.
const EDGE_SCALE: f32 = 1.8;

/// Which baked channel an effect reads from. Phrased as what it *selects* on the
/// surface, not the raw map name (principle #1: painter words).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MapSource {
    /// Occlusion — the crevices/cavities (high `ao`).
    Cavities,
    /// Exposure — the lit, unoccluded surface (`1 − ao`).
    Exposed,
    /// Convex edges and corners (positive `curvature`) — where wear collects.
    Edges,
    /// Concave creases (negative `curvature`) — deep seams dirt sinks into.
    Creases,
}

impl MapSource {
    pub const ALL: [MapSource; 4] = [
        MapSource::Cavities,
        MapSource::Exposed,
        MapSource::Edges,
        MapSource::Creases,
    ];

    pub fn name(self) -> &'static str {
        match self {
            MapSource::Cavities => "Cavities",
            MapSource::Exposed => "Exposed",
            MapSource::Edges => "Edges",
            MapSource::Creases => "Creases",
        }
    }
}

/// A Substance-style "Levels" remap of a 0..1 map value: optionally invert, push
/// contrast around the midpoint, then scale by an overall amount.
#[derive(Clone, Copy)]
pub struct Levels {
    pub invert: bool,
    /// Contrast 0..1 — 0 leaves the map linear; higher pinches mid-tones toward
    /// black/white (sharper, more selective masks).
    pub contrast: f32,
    /// Output amount 0..1 — the effect's overall strength.
    pub strength: f32,
}

impl Levels {
    /// A plain linear remap at the given strength (no invert, no contrast).
    pub fn amount(strength: f32) -> Self {
        Self {
            invert: false,
            contrast: 0.0,
            strength,
        }
    }

    /// Remap a raw 0..1 map value through invert → contrast → strength.
    pub fn apply(&self, v: f32) -> f32 {
        let v = if self.invert { 1.0 - v } else { v };
        // Linear contrast about 0.5: factor 1 at contrast 0, up to 4 at contrast 1.
        let v = if self.contrast > 0.0 {
            (0.5 + (v - 0.5) * (1.0 + self.contrast * 3.0)).clamp(0.0, 1.0)
        } else {
            v
        };
        (v * self.strength).clamp(0.0, 1.0)
    }
}

impl MeshMaps {
    /// The per-texel weight (0..1) an effect should use, reading `src` through the
    /// `levels` remap. Texels no triangle covers stay 0.
    pub fn sample(&self, src: MapSource, levels: &Levels) -> Vec<f32> {
        let mut out = vec![0.0f32; self.ao.len()];
        for (i, o) in out.iter_mut().enumerate() {
            if !self.mask[i] {
                continue;
            }
            let raw = match src {
                MapSource::Cavities => self.ao[i],
                MapSource::Exposed => 1.0 - self.ao[i],
                MapSource::Edges => (self.curvature[i].max(0.0) * EDGE_SCALE).min(1.0),
                MapSource::Creases => ((-self.curvature[i]).max(0.0) * EDGE_SCALE).min(1.0),
            };
            *o = levels.apply(raw);
        }
        out
    }
}

pub fn bake(mesh: &Mesh, bvh: &Bvh, size: u32) -> MeshMaps {
    let n = (size * size) as usize;
    let mut ao = vec![0.0f32; n];
    let mut curvature = vec![0.0f32; n];
    let mut mask = vec![false; n];
    let mut pos_map = vec![Vec3::ZERO; n];
    let mut nrm_map = vec![Vec3::Y; n];

    let (smooth, convexity) = welded_attributes(mesh);

    // Scale-dependent AO reach + ray bias from the model's bounding box.
    let (mn, mx) = mesh.bounds();
    let diag = (mx - mn).length().max(1e-3);
    let ao_dist = diag * 0.25;
    let bias = diag * 1e-3;

    // --- Rasterize triangles into UV space ---
    for tri in mesh.indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let p = [
            Vec3::from(mesh.vertices[i0].position),
            Vec3::from(mesh.vertices[i1].position),
            Vec3::from(mesh.vertices[i2].position),
        ];
        let sn = [smooth[i0], smooth[i1], smooth[i2]];
        let cv = [convexity[i0], convexity[i1], convexity[i2]];
        let face_n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();

        // UVs → texel space (V down, like the paint texture).
        let t = [
            Vec2::from(mesh.vertices[i0].uv) * size as f32,
            Vec2::from(mesh.vertices[i1].uv) * size as f32,
            Vec2::from(mesh.vertices[i2].uv) * size as f32,
        ];
        let area = edge_fn(t[0], t[1], t[2]);
        if area.abs() < 1e-6 {
            continue; // degenerate in UV space
        }

        let min_x = t[0].x.min(t[1].x).min(t[2].x).floor().max(0.0) as i32;
        let max_x = t[0].x.max(t[1].x).max(t[2].x).ceil().min(size as f32) as i32;
        let min_y = t[0].y.min(t[1].y).min(t[2].y).floor().max(0.0) as i32;
        let max_y = t[0].y.max(t[1].y).max(t[2].y).ceil().min(size as f32) as i32;

        for y in min_y..max_y {
            for x in min_x..max_x {
                let pt = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                // Barycentric via edge functions.
                let w0 = edge_fn(t[1], t[2], pt) / area;
                let w1 = edge_fn(t[2], t[0], pt) / area;
                let w2 = edge_fn(t[0], t[1], pt) / area;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                let idx = (y as u32 * size + x as u32) as usize;
                let pos = p[0] * w0 + p[1] * w1 + p[2] * w2;
                let smooth_n = (sn[0] * w0 + sn[1] * w1 + sn[2] * w2).normalize_or_zero();
                mask[idx] = true;
                pos_map[idx] = pos;
                let nrm = if smooth_n == Vec3::ZERO {
                    face_n
                } else {
                    smooth_n
                };
                nrm_map[idx] = nrm;
                // Curvature = smooth-vs-face divergence (peaks on edges/corners),
                // signed by interpolated convexity so we can tell a convex edge
                // (where wear collects) from a concave crease (where dirt sinks).
                let magnitude = (1.0 - nrm.dot(face_n)).clamp(0.0, 1.0);
                let convex = cv[0] * w0 + cv[1] * w1 + cv[2] * w2;
                curvature[idx] = if convex > 0.0 {
                    magnitude
                } else if convex < 0.0 {
                    -magnitude
                } else {
                    0.0
                };
            }
        }
    }

    // --- Ambient occlusion: hemisphere ray casts against the BVH ---
    for idx in 0..n {
        if !mask[idx] {
            continue;
        }
        let p = pos_map[idx];
        let nrm = nrm_map[idx];
        let (tangent, bitangent) = basis(nrm);
        let mut occluded = 0u32;
        for s in 0..AO_SAMPLES {
            let (r1, r2) = hash2(idx as u32, s);
            // Cosine-weighted hemisphere sample in the tangent frame.
            let phi = std::f32::consts::TAU * r1;
            let cos_t = (1.0 - r2).sqrt();
            let sin_t = r2.sqrt();
            let dir = tangent * (phi.cos() * sin_t) + bitangent * (phi.sin() * sin_t) + nrm * cos_t;
            if bvh.occludes(p + nrm * bias, dir, ao_dist) {
                occluded += 1;
            }
        }
        ao[idx] = occluded as f32 / AO_SAMPLES as f32;
    }

    MeshMaps {
        size,
        ao,
        curvature,
        mask,
    }
}

/// 2D edge function (twice the signed area of triangle a,b,c).
fn edge_fn(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// An orthonormal tangent/bitangent for a unit normal.
fn basis(n: Vec3) -> (Vec3, Vec3) {
    let up = if n.y.abs() < 0.99 { Vec3::Y } else { Vec3::X };
    let t = up.cross(n).normalize_or_zero();
    let b = n.cross(t);
    (t, b)
}

/// Two deterministic pseudo-random values in [0,1) from a texel + sample index.
fn hash2(a: u32, b: u32) -> (f32, f32) {
    let mut h = a.wrapping_mul(0x9E3779B1) ^ b.wrapping_mul(0x85EBCA77);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B3C6D);
    h ^= h >> 12;
    let r1 = (h & 0xFFFF) as f32 / 65536.0;
    let r2 = ((h >> 16) & 0xFFFF) as f32 / 65536.0;
    (r1, r2)
}

/// Per-mesh-vertex smooth normal and signed convexity, computed on a *welded*
/// copy of the mesh (vertices sharing a position are merged). The mesh may store
/// split per-face normals; welding recovers the cross-face adjacency that hides,
/// which is what both smooths the AO hemisphere at hard edges and lets us
/// distinguish a convex edge from a concave crease.
///
/// Convexity = −mean over welded neighbors of `normalize(q − p) · n`: neighbors
/// behind the tangent plane (along −n) mean the surface bulges out (convex, > 0);
/// neighbors in front mean a crease (concave, < 0).
fn welded_attributes(mesh: &Mesh) -> (Vec<Vec3>, Vec<f32>) {
    let quant = |p: [f32; 3]| {
        (
            (p[0] * 1e4).round() as i64,
            (p[1] * 1e4).round() as i64,
            (p[2] * 1e4).round() as i64,
        )
    };

    // Weld: assign each mesh vertex a welded index; accumulate averaged normals.
    let mut index_of: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let mut wpos: Vec<Vec3> = Vec::new();
    let mut wnrm: Vec<Vec3> = Vec::new();
    let mut wid: Vec<usize> = Vec::with_capacity(mesh.vertices.len());
    for v in &mesh.vertices {
        let key = quant(v.position);
        let id = *index_of.entry(key).or_insert_with(|| {
            wpos.push(Vec3::from(v.position));
            wnrm.push(Vec3::ZERO);
            wpos.len() - 1
        });
        wnrm[id] += Vec3::from(v.normal);
        wid.push(id);
    }
    for nrm in &mut wnrm {
        *nrm = nrm.normalize_or_zero();
    }

    // Welded adjacency from triangle edges.
    let mut nbrs: Vec<HashSet<usize>> = vec![HashSet::new(); wpos.len()];
    for tri in mesh.indices.chunks_exact(3) {
        let (a, b, c) = (
            wid[tri[0] as usize],
            wid[tri[1] as usize],
            wid[tri[2] as usize],
        );
        for (i, j) in [(a, b), (b, c), (c, a)] {
            if i != j {
                nbrs[i].insert(j);
                nbrs[j].insert(i);
            }
        }
    }

    // Signed convexity per welded vertex.
    let mut wcurv = vec![0.0f32; wpos.len()];
    for i in 0..wpos.len() {
        if nbrs[i].is_empty() {
            continue;
        }
        let mut sum = 0.0;
        for &j in &nbrs[i] {
            let d = (wpos[j] - wpos[i]).normalize_or_zero();
            sum += d.dot(wnrm[i]);
        }
        wcurv[i] = -sum / nbrs[i].len() as f32;
    }

    let smooth = wid.iter().map(|&id| wnrm[id]).collect();
    let convexity = wid.iter().map(|&id| wcurv[id]).collect();
    (smooth, convexity)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn levels_amount_is_linear_scale() {
        // No invert, no contrast: just multiply by strength.
        let lv = Levels::amount(0.5);
        assert!(close(lv.apply(1.0), 0.5));
        assert!(close(lv.apply(0.4), 0.2));
        assert!(close(lv.apply(0.0), 0.0));
    }

    #[test]
    fn levels_invert_flips_before_scaling() {
        let lv = Levels {
            invert: true,
            contrast: 0.0,
            strength: 1.0,
        };
        assert!(close(lv.apply(0.0), 1.0));
        assert!(close(lv.apply(1.0), 0.0));
    }

    #[test]
    fn levels_contrast_pushes_away_from_mid() {
        // Above 0.5 rises, below 0.5 falls; the midpoint is fixed.
        let lv = Levels {
            invert: false,
            contrast: 1.0,
            strength: 1.0,
        };
        assert!(close(lv.apply(0.5), 0.5));
        assert!(lv.apply(0.6) > 0.6);
        assert!(lv.apply(0.4) < 0.4);
        // Stays clamped to [0, 1].
        assert!(close(lv.apply(1.0), 1.0));
        assert!(close(lv.apply(0.0), 0.0));
    }

    #[test]
    fn sample_selects_complementary_sources() {
        // A 2-texel map: one deep cavity (convex creases), one exposed edge.
        let maps = MeshMaps {
            size: 1,
            ao: vec![0.9, 0.1],
            curvature: vec![-0.5, 0.5], // crease, then convex edge
            mask: vec![true, true],
        };
        let full = Levels::amount(1.0);

        let cav = maps.sample(MapSource::Cavities, &full);
        assert!(cav[0] > cav[1], "cavities favor the occluded texel");

        let exp = maps.sample(MapSource::Exposed, &full);
        assert!(exp[1] > exp[0], "exposed favors the open texel");

        let edges = maps.sample(MapSource::Edges, &full);
        assert!(
            edges[1] > 0.0 && close(edges[0], 0.0),
            "edges only on convex"
        );

        let creases = maps.sample(MapSource::Creases, &full);
        assert!(
            creases[0] > 0.0 && close(creases[1], 0.0),
            "creases only on concave"
        );
    }

    #[test]
    fn sample_skips_uncovered_texels() {
        let maps = MeshMaps {
            size: 1,
            ao: vec![1.0],
            curvature: vec![0.0],
            mask: vec![false], // no triangle covered it
        };
        assert!(close(
            maps.sample(MapSource::Cavities, &Levels::amount(1.0))[0],
            0.0
        ));
    }
}

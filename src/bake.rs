// src/bake.rs
//
// Mesh-aware maps baked into UV/texture space (the moat, principle #4). For each
// texel that a triangle covers we know the surface point's world position and
// normal, from which we bake:
//
//   - ao   : ambient occlusion — fraction of a cosine-weighted hemisphere that is
//            blocked by other geometry within a local radius (dark crevices).
//   - edge : *convex*-edge weight — signed surface convexity (how neighboring
//            vertices sit relative to the tangent plane), keeping only the convex
//            part. Concave creases (where AO lives) read negative and are
//            excluded, so highlights don't land on top of AO.
//
// These drive the AO suite: "Darken (AO)" adds a black Multiply layer weighted by
// `ao`; "Highlights" adds a white layer weighted by `edge`.

use std::collections::{HashMap, HashSet};

use glam::{Vec2, Vec3};

use crate::bvh::Bvh;
use crate::mesh::Mesh;

/// Scales edge magnitude (≈0.4 at a cube corner) up toward 1 for highlights.
const EDGE_SCALE: f32 = 1.8;

/// Per-texel baked maps, all `size`×`size`, row-major (V down, matching paint).
pub struct MeshMaps {
    pub size: u32,
    pub ao: Vec<f32>,
    pub edge: Vec<f32>,
    pub mask: Vec<bool>, // true where a triangle covered the texel
}

/// Hemisphere ray count per texel for AO. Modest — bakes stay sub-second.
const AO_SAMPLES: u32 = 24;

pub fn bake(mesh: &Mesh, bvh: &Bvh, size: u32) -> MeshMaps {
    let n = (size * size) as usize;
    let mut ao = vec![0.0f32; n];
    let mut edge = vec![0.0f32; n];
    let mut mask = vec![false; n];
    let mut pos_map = vec![Vec3::ZERO; n];
    let mut nrm_map = vec![Vec3::Y; n];

    let (smooth, curvature) = welded_attributes(mesh);

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
        let cv = [curvature[i0], curvature[i1], curvature[i2]];
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
                nrm_map[idx] = if smooth_n == Vec3::ZERO {
                    face_n
                } else {
                    smooth_n
                };
                // Highlight = edge magnitude (smooth-vs-face divergence, which
                // peaks on edges/corners) GATED by convex sign, so concave creases
                // (where AO lives) are excluded.
                let magnitude = (1.0 - smooth_n.dot(face_n)).clamp(0.0, 1.0);
                let convex = cv[0] * w0 + cv[1] * w1 + cv[2] * w2;
                edge[idx] = if convex > 0.0 {
                    (magnitude * EDGE_SCALE).clamp(0.0, 1.0)
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
        edge,
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
/// copy of the mesh (vertices sharing a position are merged). Welding recovers
/// cross-face adjacency that split (per-face) normals hide, which is what lets us
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
    for n in &mut wnrm {
        *n = n.normalize_or_zero();
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
    let curvature = wid.iter().map(|&id| wcurv[id]).collect();
    (smooth, curvature)
}

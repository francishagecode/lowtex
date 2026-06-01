// src/bake.rs
//
// Mesh-aware maps baked into UV/texture space (the moat, principle #4). For each
// texel that a triangle covers we know the surface point's world position and
// normal, from which we bake:
//
//   - ao        : ambient occlusion — fraction of a cosine-weighted hemisphere
//                 blocked by other geometry within a local radius (dark crevices).
//   - curvature : signed hard-edge proximity in [-1, 1] — a falloff band hugging the
//                 mesh's hard edges, +ve on convex edges (where wear collects), −ve in
//                 concave creases (where dirt sinks), 0 on smooth/flat surface. Keyed
//                 off the *dihedral angle* between adjacent faces and the texel's
//                 distance to such an edge, so the band stays thin and local instead of
//                 spreading across a big low-poly face the way a per-texel normal
//                 divergence does.
//
// These two channels are *inputs*, in the Substance-Painter sense: rather than
// each producing one fixed layer, `MeshMaps::sample` reads a `MapSource` through a
// `Levels` remap (invert / contrast / strength) into a 0..1 weight that drives
// either a generated tint layer or any layer's reveal mask. The AO suite ("Darken
// (AO)", "Highlights"), the Dirt and Edge-wear presets, and "mask from map" are all
// the same path with different source + color + blend.

use std::collections::HashMap;

use glam::{Vec2, Vec3};

use crate::bvh::Bvh;
use crate::mesh::Mesh;
use crate::noise::{self, NoiseKind, NoiseParams};

/// Per-texel baked maps, all `size`×`size`, row-major (V down, matching paint).
pub struct MeshMaps {
    pub size: u32,
    pub ao: Vec<f32>,
    /// Signed hard-edge proximity in [-1, 1]: a band that is +1 on a convex hard edge
    /// and −1 in a concave crease, falling to 0 within `EDGE_WIDTH_FRAC` of the edge
    /// (and 0 everywhere on smooth/flat surface). Drives `Edges`/`Creases`.
    pub curvature: Vec<f32>,
    pub mask: Vec<bool>, // true where a triangle covered the texel
    /// World position per texel — the sampling point for procedural noise (so it
    /// reads the same across UV seams). `ZERO` where no triangle covered the texel.
    pub pos: Vec<Vec3>,
    /// Smooth surface normal per texel (face normal where smoothing collapses).
    /// Kept from the bake so a directional light can be (re)computed cheaply without
    /// re-rasterizing. `Y` where no triangle covered the texel.
    pub nrm: Vec<Vec3>,
    /// Directional ("sun") light per texel: N·L clamped, optionally shadowed. Filled
    /// lazily by `compute_light` for the chosen direction — zero until then.
    pub light: Vec<f32>,
    /// The (direction, shadow) the `light` channel was last baked for, so callers can
    /// skip recomputing when the sun hasn't moved. `None` = never computed.
    pub light_params: Option<(Vec3, bool)>,
    /// The mesh's bounding-box diagonal. Noise positions are divided by this so a
    /// given `NoiseMod::scale` means the same feature size regardless of model size.
    pub diag: f32,
}

/// Hemisphere ray count per texel for AO. Modest — bakes stay sub-second.
const AO_SAMPLES: u32 = 24;

/// A mesh edge counts as "hard" (and so seeds an edge/crease band) when its two
/// adjacent faces diverge by more than this in `1 − n₁·n₂`: 0 is coplanar, ~0.13 is
/// a 30° fold, 1.0 a right-angle cube edge. Below it the edge is treated as a smooth
/// interior edge and contributes nothing.
const SHARP_COS: f32 = 0.1;

/// Half-width of the edge/crease band as a fraction of the model's bounding-box
/// diagonal. The `Edges`/`Creases` weight is 1 *on* a hard edge and falls to 0 this
/// far away (in world space, so the band is a consistent physical width regardless of
/// texture resolution or how big the adjoining faces are — the fix for the old
/// curvature "dome" that washed whole low-poly faces).
const EDGE_WIDTH_FRAC: f32 = 0.04;

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
    /// The whole painted surface (weight 1 everywhere covered). On its own a flat
    /// object tint; paired with a noise breakup it becomes procedural grunge.
    Surface,
    /// Directional ("sun") light: high on faces turned toward the light, dark in its
    /// shadow. Unlike the others this depends on a chosen sun direction (and optional
    /// cast shadows), baked into `MeshMaps::light` by `compute_light` before sampling.
    Light,
}

impl MapSource {
    pub const ALL: [MapSource; 6] = [
        MapSource::Cavities,
        MapSource::Exposed,
        MapSource::Edges,
        MapSource::Creases,
        MapSource::Surface,
        MapSource::Light,
    ];

    pub fn name(self) -> &'static str {
        match self {
            MapSource::Cavities => "Cavities",
            MapSource::Exposed => "Exposed",
            MapSource::Edges => "Edges",
            MapSource::Creases => "Creases",
            MapSource::Surface => "Surface",
            MapSource::Light => "Light",
        }
    }
}

/// Procedural-noise breakup layered on top of a map source — the documented
/// "edge wear = curvature × noise". The source weight (after its `Levels` remap)
/// is multiplied by a noise factor, so the effect lands in patches instead of a
/// perfect curvature ring. With `MapSource::Surface` the source is uniform, so
/// the noise itself becomes the pattern (grunge, splotches, weathering).
#[derive(Clone, Copy)]
pub struct NoiseMod {
    pub kind: NoiseKind,
    /// Feature frequency across the model's span (higher = finer detail).
    pub scale: f32,
    /// Sharpens the noise toward hard patches (0 = soft/cloudy, 1 = blotchy).
    pub contrast: f32,
    /// 0 = noise off (source unchanged); 1 = source fully multiplied by noise.
    pub amount: f32,
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
    /// `levels` remap and, when `noise` is given, multiplied by a procedural-noise
    /// breakup. Texels no triangle covers stay 0.
    pub fn sample(&self, src: MapSource, levels: &Levels, noise: Option<&NoiseMod>) -> Vec<f32> {
        let mut out = vec![0.0f32; self.ao.len()];
        // Only set up noise when it would actually change anything.
        let noise = noise.filter(|n| n.amount > 0.0);
        let nparams = noise.map(|n| NoiseParams {
            scale: n.scale,
            ..Default::default()
        });
        // The noise field gets its own contrast remap to turn mushy fBm into patches.
        let ncontrast = noise.map(|n| Levels {
            invert: false,
            contrast: n.contrast,
            strength: 1.0,
        });
        let inv_diag = 1.0 / self.diag.max(1e-6);

        for (i, o) in out.iter_mut().enumerate() {
            if !self.mask[i] {
                continue;
            }
            let raw = match src {
                MapSource::Cavities => self.ao[i],
                MapSource::Exposed => 1.0 - self.ao[i],
                MapSource::Edges => self.curvature[i].max(0.0),
                MapSource::Creases => (-self.curvature[i]).max(0.0),
                MapSource::Surface => 1.0,
                MapSource::Light => self.light[i],
            };
            let mut w = levels.apply(raw);
            if let (Some(n), Some(params), Some(c)) = (noise, &nparams, &ncontrast) {
                let nv = c.apply(noise::sample(n.kind, self.pos[i] * inv_diag, params));
                // Lerp the factor 1→noise by `amount`: amount 0 leaves `w` untouched,
                // amount 1 multiplies fully. Staying ≤ 1 keeps the result in range.
                w *= 1.0 - n.amount + n.amount * nv;
            }
            *o = w;
        }
        out
    }

    /// Bake a directional ("sun") light into the `light` channel: per covered texel,
    /// `max(N·L, 0)` (Lambert), and — when `shadow` — zeroed if a ray cast toward the
    /// light hits other geometry first (cast shadows). `dir` points *toward* the
    /// light. Records `light_params` so the renderer can skip recomputing when the
    /// sun hasn't moved. Like AO, this raycasts against the BVH, so it's the one map
    /// that needs the mesh's acceleration structure rather than just baked channels.
    pub fn compute_light(&mut self, bvh: &Bvh, dir: Vec3, shadow: bool) {
        let dir = dir.normalize_or_zero();
        let bias = self.diag * 1e-3;
        // A directional light is infinitely far; cast the shadow ray across the
        // whole model so anything between the texel and the sun counts as occluder.
        let reach = self.diag;
        for i in 0..self.light.len() {
            if !self.mask[i] {
                self.light[i] = 0.0;
                continue;
            }
            let ndotl = self.nrm[i].dot(dir).max(0.0);
            self.light[i] = if shadow
                && ndotl > 0.0
                && bvh.occludes(self.pos[i] + self.nrm[i] * bias, dir, reach)
            {
                0.0
            } else {
                ndotl
            };
        }
        self.light_params = Some((dir, shadow));
    }
}

/// A low→high color ramp applied to a 0..1 weight — the "gradient map". Where the
/// per-texel tint layers paint one flat color masked by a map, a gradient map paints
/// the surface *by value*, so a single channel (height, AO, light…) reads as a full
/// material: deep crevices through to bright tops in one pass. sRGB lerp — fine at
/// PSX color depth and consistent with how paint colors are stored.
#[derive(Clone, Copy)]
pub struct Gradient {
    pub low: [u8; 3],
    pub high: [u8; 3],
}

impl Gradient {
    /// The color at weight `t` (clamped to 0..1), linearly between `low` and `high`.
    pub fn sample(&self, t: f32) -> [u8; 3] {
        let t = t.clamp(0.0, 1.0);
        let lerp = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t).round() as u8;
        [
            lerp(self.low[0], self.high[0]),
            lerp(self.low[1], self.high[1]),
            lerp(self.low[2], self.high[2]),
        ]
    }
}

pub fn bake(mesh: &Mesh, bvh: &Bvh, size: u32) -> MeshMaps {
    let n = (size * size) as usize;
    let mut ao = vec![0.0f32; n];
    let mut curvature = vec![0.0f32; n];
    let mut mask = vec![false; n];
    let mut pos_map = vec![Vec3::ZERO; n];
    let mut nrm_map = vec![Vec3::Y; n];

    let smooth = welded_smooth_normals(mesh);
    // Hard edges and their convex/concave sign, keyed by welded vertex-id pair, plus
    // the welded id of each mesh vertex so the rasterizer can look its edges up.
    let (sharp, wid) = sharp_edges(mesh);

    // Scale-dependent AO reach + ray bias from the model's bounding box.
    let (mn, mx) = mesh.bounds();
    let diag = (mx - mn).length().max(1e-3);
    let ao_dist = diag * 0.25;
    let bias = diag * 1e-3;
    let edge_width = diag * EDGE_WIDTH_FRAC;

    // --- Rasterize triangles into UV space ---
    for tri in mesh.indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let p = [
            Vec3::from(mesh.vertices[i0].position),
            Vec3::from(mesh.vertices[i1].position),
            Vec3::from(mesh.vertices[i2].position),
        ];
        let sn = [smooth[i0], smooth[i1], smooth[i2]];
        let face_n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();

        // This triangle's hard edges as world segments + sign (+convex / −concave).
        // Only edges that fold sharply enough seed a band; at most 3 per triangle.
        let mut tri_sharp: [(Vec3, Vec3, f32); 3] = [(Vec3::ZERO, Vec3::ZERO, 0.0); 3];
        let mut n_sharp = 0usize;
        for &(va, vb, pa, pb) in &[
            (wid[i0], wid[i1], p[0], p[1]),
            (wid[i1], wid[i2], p[1], p[2]),
            (wid[i2], wid[i0], p[2], p[0]),
        ] {
            let key = if va < vb { (va, vb) } else { (vb, va) };
            if let Some(&sign) = sharp.get(&key) {
                tri_sharp[n_sharp] = (pa, pb, sign);
                n_sharp += 1;
            }
        }

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
                // Hard-edge proximity: distance from this surface point to the
                // triangle's sharpest nearby hard edge, as a falloff band. Keep the
                // strongest (closest) edge and carry its convex/concave sign.
                let mut best = 0.0f32;
                for &(ea, eb, sign) in &tri_sharp[..n_sharp] {
                    let dist = point_segment_dist(pos, ea, eb);
                    let w = 1.0 - smoothstep(0.0, edge_width, dist);
                    if w > best.abs() {
                        best = sign * w;
                    }
                }
                curvature[idx] = best;
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
        pos: pos_map,
        nrm: nrm_map,
        // The directional light is baked lazily once a sun direction is chosen.
        light: vec![0.0f32; n],
        light_params: None,
        diag,
    }
}

/// Hermite smoothstep: 0 below `lo`, 1 above `hi`, smooth ramp between.
fn smoothstep(lo: f32, hi: f32, x: f32) -> f32 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Distance from point `p` to the segment `a`–`b`.
fn point_segment_dist(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let len2 = ab.length_squared();
    let t = if len2 > 1e-12 {
        ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p - (a + ab * t)).length()
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

/// Quantize a position to a welding grid key (~1e-4 units), so vertices the importer
/// split at a UV seam or hard edge but that share a position are recognised as one.
fn weld_key(p: [f32; 3]) -> (i64, i64, i64) {
    (
        (p[0] * 1e4).round() as i64,
        (p[1] * 1e4).round() as i64,
        (p[2] * 1e4).round() as i64,
    )
}

/// Per-mesh-vertex smooth normal, computed on a *welded* copy of the mesh (vertices
/// sharing a position are merged). The mesh may store split per-face normals; welding
/// recovers the cross-face adjacency that hides, which is what smooths the AO
/// hemisphere across hard edges.
fn welded_smooth_normals(mesh: &Mesh) -> Vec<Vec3> {
    let mut index_of: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let mut wnrm: Vec<Vec3> = Vec::new();
    let mut wid: Vec<usize> = Vec::with_capacity(mesh.vertices.len());
    for v in &mesh.vertices {
        let id = *index_of.entry(weld_key(v.position)).or_insert_with(|| {
            wnrm.push(Vec3::ZERO);
            wnrm.len() - 1
        });
        wnrm[id] += Vec3::from(v.normal);
        wid.push(id);
    }
    for nrm in &mut wnrm {
        *nrm = nrm.normalize_or_zero();
    }
    wid.iter().map(|&id| wnrm[id]).collect()
}

/// Classify the mesh's hard edges for the edge/crease band. Returns, per welded
/// vertex-id pair that forms a hard edge, its sign (`+1` convex / `−1` concave), and
/// — for the rasterizer — the welded id of every mesh vertex.
///
/// An edge is "hard" when its two adjacent faces fold by more than `SHARP_COS`. The
/// sign comes from where the *opposite* vertex of the second face sits relative to the
/// first face's plane: behind it (along −n₁) means the surface bulges out (convex);
/// in front means a crease (concave). Boundary edges (one face) and non-manifold edges
/// (>2 faces) are skipped.
fn sharp_edges(mesh: &Mesh) -> (HashMap<(u32, u32), f32>, Vec<u32>) {
    // Weld vertices by position so split copies share an id.
    let mut index_of: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut wpos: Vec<Vec3> = Vec::new();
    let mut wid: Vec<u32> = Vec::with_capacity(mesh.vertices.len());
    for v in &mesh.vertices {
        let id = *index_of.entry(weld_key(v.position)).or_insert_with(|| {
            wpos.push(Vec3::from(v.position));
            (wpos.len() - 1) as u32
        });
        wid.push(id);
    }

    // Per welded edge, the adjacent faces as (normal, opposite-vertex position).
    let mut faces: HashMap<(u32, u32), Vec<(Vec3, Vec3)>> = HashMap::new();
    for tri in mesh.indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (a, b, c) = (wid[i0], wid[i1], wid[i2]);
        let p = [
            Vec3::from(mesh.vertices[i0].position),
            Vec3::from(mesh.vertices[i1].position),
            Vec3::from(mesh.vertices[i2].position),
        ];
        let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
        // Each edge paired with the triangle's opposite vertex.
        for &(u, v, opp) in &[(a, b, p[2]), (b, c, p[0]), (c, a, p[1])] {
            if u == v {
                continue;
            }
            let key = if u < v { (u, v) } else { (v, u) };
            faces.entry(key).or_default().push((n, opp));
        }
    }

    let mut sharp = HashMap::new();
    for (&key, fs) in &faces {
        if fs.len() != 2 {
            continue; // boundary or non-manifold edge
        }
        let (n1, _c1) = fs[0];
        let (n2, c2) = fs[1];
        if 1.0 - n1.dot(n2) <= SHARP_COS {
            continue; // too flat to be a hard edge
        }
        let edge_a = wpos[key.0 as usize];
        let convex = n1.dot(c2 - edge_a) < 0.0;
        sharp.insert(key, if convex { 1.0 } else { -1.0 });
    }
    (sharp, wid)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn edges_form_a_thin_band_on_the_hard_edges() {
        // A bare cube is the worst case for the old smooth-vs-face metric: every
        // face is one quad, so the divergence "dome" washed the whole surface. The
        // distance-to-hard-edge metric must instead keep the Edges weight to a thin
        // band hugging the cube's edges, and leave the flat face centres at 0.
        let mesh = crate::mesh::Mesh::cube();
        let bvh = Bvh::build(&mesh);
        let maps = bake(&mesh, &bvh, 64);
        let edges = maps.sample(MapSource::Edges, &Levels::amount(1.0), None);

        let covered = maps.mask.iter().filter(|&&m| m).count();
        let banded = edges.iter().filter(|&&w| w > 0.5).count();
        // The band is a small fraction of the surface — not the whole-face wash the
        // old dome produced (which lit ~all covered texels).
        assert!(covered > 0);
        let frac = banded as f32 / covered as f32;
        assert!(frac > 0.0, "the cube's hard edges must register at all");
        assert!(
            frac < 0.35,
            "edge band should be local, got {frac:.2} of the surface"
        );

        // A texel right at the centre of a +Z face sits far from any edge → ~0; the
        // cube's convex edges read positive (Edges), never negative.
        assert!(edges.iter().all(|&w| w >= 0.0));
        assert!(
            edges.iter().cloned().fold(0.0, f32::max) > 0.9,
            "on-edge texels reach full weight"
        );
        // A bare convex cube has no concave creases.
        let creases = maps.sample(MapSource::Creases, &Levels::amount(1.0), None);
        assert!(
            creases.iter().all(|&w| close(w, 0.0)),
            "a convex cube has no creases"
        );
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
            pos: vec![Vec3::ZERO, Vec3::ZERO],
            nrm: vec![Vec3::Y, Vec3::Y],
            light: vec![0.0, 0.0],
            light_params: None,
            diag: 1.0,
        };
        let full = Levels::amount(1.0);

        let cav = maps.sample(MapSource::Cavities, &full, None);
        assert!(cav[0] > cav[1], "cavities favor the occluded texel");

        let exp = maps.sample(MapSource::Exposed, &full, None);
        assert!(exp[1] > exp[0], "exposed favors the open texel");

        let edges = maps.sample(MapSource::Edges, &full, None);
        assert!(
            edges[1] > 0.0 && close(edges[0], 0.0),
            "edges only on convex"
        );

        let creases = maps.sample(MapSource::Creases, &full, None);
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
            pos: vec![Vec3::ZERO],
            nrm: vec![Vec3::Y],
            light: vec![0.0],
            light_params: None,
            diag: 1.0,
        };
        assert!(close(
            maps.sample(MapSource::Cavities, &Levels::amount(1.0), None)[0],
            0.0
        ));
    }

    fn one_texel(ao: f32, pos: Vec3) -> MeshMaps {
        MeshMaps {
            size: 1,
            ao: vec![ao],
            curvature: vec![0.0],
            mask: vec![true],
            pos: vec![pos],
            nrm: vec![Vec3::Y],
            light: vec![0.0],
            light_params: None,
            diag: 1.0,
        }
    }

    #[test]
    fn surface_source_is_full_before_noise() {
        let maps = one_texel(0.5, Vec3::ZERO);
        let w = maps.sample(MapSource::Surface, &Levels::amount(1.0), None);
        assert!(close(w[0], 1.0), "Surface ignores AO/curvature — flat 1.0");
    }

    #[test]
    fn compute_light_is_lambert_against_the_sun() {
        // Bake a cube and shine the sun down (+Y). With shadows off the contract is
        // exactly Lambert per texel — `max(N·L, 0)` — which we can check directly
        // (robust to the cube's corner-bent welded normals).
        let mesh = crate::mesh::Mesh::cube();
        let bvh = Bvh::build(&mesh);
        let mut maps = bake(&mesh, &bvh, 32);

        maps.compute_light(&bvh, Vec3::Y, false);
        assert_eq!(maps.light_params, Some((Vec3::Y, false)));

        let (mut lit, mut dark) = (0u32, 0u32);
        for i in 0..maps.light.len() {
            if !maps.mask[i] {
                assert!(close(maps.light[i], 0.0), "uncovered texels stay dark");
                continue;
            }
            let expect = maps.nrm[i].dot(Vec3::Y).max(0.0);
            assert!(close(maps.light[i], expect), "light is clamped N·L");
            if maps.light[i] > 0.5 {
                lit += 1;
            } else if maps.light[i] == 0.0 {
                dark += 1;
            }
        }
        // The cube has faces toward the sun (lit) and away from it (the down-clamped
        // dark side), so neither bucket is empty — the light actually varies.
        assert!(lit > 0 && dark > 0, "sun lights some faces and not others");

        // A convex cube can't shadow itself, so casting shadows changes nothing.
        let no_shadow = maps.light.clone();
        maps.compute_light(&bvh, Vec3::Y, true);
        assert!(
            maps.light
                .iter()
                .zip(&no_shadow)
                .all(|(a, b)| close(*a, *b)),
            "a convex mesh casts no self-shadow"
        );
    }

    #[test]
    fn light_source_reads_the_light_channel() {
        // Two covered texels: one fully lit, one in shadow. The Light source should
        // pass the channel straight through the (full) Levels remap.
        let maps = MeshMaps {
            size: 1,
            ao: vec![0.0, 0.0],
            curvature: vec![0.0, 0.0],
            mask: vec![true, true],
            pos: vec![Vec3::ZERO, Vec3::ZERO],
            nrm: vec![Vec3::Y, Vec3::Y],
            light: vec![0.9, 0.1],
            light_params: Some((Vec3::Y, false)),
            diag: 1.0,
        };
        let w = maps.sample(MapSource::Light, &Levels::amount(1.0), None);
        assert!(
            close(w[0], 0.9) && close(w[1], 0.1),
            "Light reads its channel"
        );
        // Inverting turns the lit map into a shadow mask (dark side high).
        let inv = maps.sample(
            MapSource::Light,
            &Levels {
                invert: true,
                contrast: 0.0,
                strength: 1.0,
            },
            None,
        );
        assert!(inv[1] > inv[0], "inverted Light favors the shadowed texel");
    }

    #[test]
    fn gradient_interpolates_low_to_high() {
        let g = Gradient {
            low: [0, 0, 0],
            high: [255, 100, 50],
        };
        assert_eq!(g.sample(0.0), [0, 0, 0]);
        assert_eq!(g.sample(1.0), [255, 100, 50]);
        // Midpoint is the channel-wise average (clamped input).
        assert_eq!(g.sample(0.5), [128, 50, 25]);
        assert_eq!(g.sample(2.0), [255, 100, 50], "weights clamp to 1");
    }

    #[test]
    fn noise_amount_zero_is_identity() {
        let maps = one_texel(0.2, Vec3::new(1.0, 2.0, 3.0));
        let lv = Levels::amount(1.0);
        let nm = NoiseMod {
            kind: NoiseKind::Perlin,
            scale: 4.0,
            contrast: 0.5,
            amount: 0.0,
        };
        let with = maps.sample(MapSource::Cavities, &lv, Some(&nm));
        let without = maps.sample(MapSource::Cavities, &lv, None);
        assert!(
            close(with[0], without[0]),
            "amount 0 must not change the weight"
        );
    }

    #[test]
    fn noise_full_amount_multiplies_and_stays_in_range() {
        // Surface (raw 1.0) fully multiplied by noise → the weight *is* the noise.
        let maps = one_texel(1.0, Vec3::new(0.3, 0.7, 0.1));
        let nm = NoiseMod {
            kind: NoiseKind::Value,
            scale: 5.0,
            contrast: 0.0,
            amount: 1.0,
        };
        let w = maps.sample(MapSource::Surface, &Levels::amount(1.0), Some(&nm))[0];
        assert!(
            (0.0..=1.0).contains(&w),
            "noise-multiplied weight out of range: {w}"
        );
        // A second call is deterministic (noise is hash-based).
        let w2 = maps.sample(MapSource::Surface, &Levels::amount(1.0), Some(&nm))[0];
        assert!(close(w, w2));
    }
}

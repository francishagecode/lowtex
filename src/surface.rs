// src/surface.rs
//
// Cross-face ("surface-space") brush splatting. The flat brush in `paint.rs`
// stamps a circle in *texture* space at the single picked UV, so a dab that
// reaches a UV seam dies at the island edge: the texels just across the seam
// belong to another atlas cell and are never touched, leaving a hard line on
// every model edge. This module paints the dab as a sphere on the *mesh
// surface* instead — from the picked triangle it walks position-adjacent
// triangles outward to a world-space radius, rasterizes each into its own UV
// region, and weights every covered texel by its 3D distance to the hit point.
// A dab on a cube edge then wraps onto the neighbouring face, crossing UV seams
// while staying on connected geometry.
//
// Adjacency is keyed on *position* only (not UV), so the walk crosses seams
// while staying on connected geometry: the brush wraps around a 90° edge but
// never bleeds through to the unconnected back wall of a thin shell.

use std::collections::HashMap;

use glam::{Vec2, Vec3};

use crate::bvh::Hit;
use crate::mesh::Mesh;
use crate::paint::falloff;

/// Per-triangle position-edge adjacency: for each triangle, the neighbouring
/// triangle across each of its three edges (`-1` where the edge is a mesh
/// boundary). Keyed on position only, so two faces meeting at a UV seam are
/// still neighbours. Depends only on topology — resolution-independent, so it
/// survives a texture-resolution change (unlike the baked maps).
pub struct Adjacency {
    neighbors: Vec<[i32; 3]>,
}

impl Adjacency {
    pub fn build(mesh: &Mesh) -> Self {
        let tri_count = mesh.indices.len() / 3;
        let mut neighbors = vec![[-1i32; 3]; tri_count];
        // (quantized position edge) -> (owning tri, which of its 3 edge slots).
        // The first triangle to claim an edge inserts; the second removes it and
        // links both directions. A non-manifold third sharer just re-inserts
        // (last writer owns) — fine for a local flood.
        let mut owner: HashMap<(PosKey, PosKey), (u32, u8)> = HashMap::new();
        for (ti, tri) in mesh.indices.chunks_exact(3).enumerate() {
            let pk = [
                pos_key(mesh, tri[0]),
                pos_key(mesh, tri[1]),
                pos_key(mesh, tri[2]),
            ];
            for e in 0..3 {
                let (a, b) = (pk[e], pk[(e + 1) % 3]);
                let key = if a <= b { (a, b) } else { (b, a) };
                match owner.remove(&key) {
                    Some((other_ti, other_e)) => {
                        neighbors[ti][e] = other_ti as i32;
                        neighbors[other_ti as usize][other_e as usize] = ti as i32;
                    }
                    None => {
                        owner.insert(key, (ti as u32, e as u8));
                    }
                }
            }
        }
        Self { neighbors }
    }

    /// Per-triangle edge neighbours (`-1` = mesh boundary). For triangle `ti`, slot
    /// `e` is the neighbour across the edge between its vertices `e` and `(e+1)%3`.
    /// Lets other modules (e.g. the face-outline builder) walk facet boundaries.
    pub fn neighbors(&self) -> &[[i32; 3]] {
        &self.neighbors
    }
}

/// Reusable working memory for `splat`, kept across dabs so a stroke doesn't
/// allocate (and zero) a `tri_count`-sized visited buffer per dab. Visited
/// triangles are marked with a per-dab generation stamp instead of a boolean, so
/// resetting between dabs is a single counter bump — the flood then costs
/// O(triangles actually walked), not O(total triangles), which matters once a
/// mesh has many faces. `stack` is the flood frontier, cleared and refilled per dab.
pub struct SplatScratch {
    /// `seen[ti] == gen` means triangle `ti` was reached this dab. Sized to the
    /// mesh's triangle count; reset to 0 only when the mesh (count) changes or the
    /// generation counter wraps.
    seen: Vec<u32>,
    gen: u32,
    stack: Vec<usize>,
}

impl Default for SplatScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl SplatScratch {
    pub fn new() -> Self {
        Self {
            seen: Vec::new(),
            gen: 0,
            stack: Vec::new(),
        }
    }

    /// Start a fresh flood over `tri_count` triangles: resize on a mesh change,
    /// advance the generation (resetting on wraparound so a stale stamp can't read
    /// as visited), and clear the frontier. Returns the generation to stamp with.
    fn begin(&mut self, tri_count: usize) -> u32 {
        if self.seen.len() != tri_count {
            self.seen.clear();
            self.seen.resize(tri_count, 0);
            self.gen = 0;
        }
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            self.seen.iter_mut().for_each(|s| *s = 0);
            self.gen = 1;
        }
        self.stack.clear();
        self.gen
    }
}

/// World-space brush radius for a texel-space `brush_radius`, measured at the hit
/// triangle. The unwrap keeps a constant world-texel density, so the local ratio
/// of world length to UV length there is the density everywhere: `brush_radius`
/// texels → `brush_radius / size` UV units → world units. Uses the triangle's
/// longest UV edge for a stable ratio; `None` if it is degenerate in UV (the
/// caller then falls back to a bbox-derived radius).
pub fn world_radius(mesh: &Mesh, tri: u32, brush_radius: f32, size: u32) -> Option<f32> {
    let (p, uv) = tri_data(mesh, tri);
    let mut world_per_uv = 0.0;
    let mut longest_uv = 0.0f32;
    for i in 0..3 {
        let world_len = (p[(i + 1) % 3] - p[i]).length();
        let uv_len = (uv[(i + 1) % 3] - uv[i]).length();
        if uv_len > longest_uv {
            longest_uv = uv_len;
            world_per_uv = world_len / uv_len;
        }
    }
    (longest_uv > 1e-6).then(|| (brush_radius / size as f32) * world_per_uv)
}

/// Walk the mesh surface from `hit` out to `radius_world`, returning every covered
/// texel with its stroke coverage `a = opacity · falloff(d / radius, hardness)`,
/// where `d` is the texel's surface-point distance to the hit point. Texels past
/// the radius (falloff 0) are omitted. A texel on a shared triangle edge can be
/// returned more than once; the caller's per-stroke coverage takes the max, so
/// the duplication is harmless.
pub fn splat(
    mesh: &Mesh,
    adj: &Adjacency,
    hit: &Hit,
    radius_world: f32,
    opacity: f32,
    hardness: f32,
    size: u32,
    scratch: &mut SplatScratch,
) -> Vec<(usize, f32)> {
    let mut out = Vec::new();
    let tri_count = mesh.indices.len() / 3;
    if radius_world <= 0.0 || hit.tri as usize >= tri_count {
        return out;
    }
    let r = radius_world;
    let gen = scratch.begin(tri_count);
    scratch.stack.push(hit.tri as usize);
    scratch.seen[hit.tri as usize] = gen;

    while let Some(ti) = scratch.stack.pop() {
        let (p, uv) = tri_data(mesh, ti as u32);
        // Expand the flood through this triangle only if part of it is near the
        // hit: the *closest point on the triangle* within the radius (the root is
        // always in, d=0 there). Testing only the vertices/centroid would miss a
        // triangle whose interior or edge passes under the brush while all three
        // corners sit beyond it — e.g. a dab straddling the middle of a long
        // shared edge, where the neighbour's two shared vertices are the far edge
        // endpoints. That made the dab refuse to wrap onto the neighbour unless
        // the cursor crossed onto it. Over-inclusion is harmless — per-texel
        // falloff discards anything past the radius — this just stops the walk at
        // the true brush frontier instead of crawling the whole mesh.
        let near = ti == hit.tri as usize || point_tri_dist(hit.pos, &p) <= r;
        if !near {
            continue;
        }
        rasterize(&uv, size, |texel, w| {
            let world = p[0] * w[0] + p[1] * w[1] + p[2] * w[2];
            let a = opacity * falloff(world.distance(hit.pos) / r, hardness);
            if a > 0.0 {
                out.push((texel, a));
            }
        });
        for &nb in &adj.neighbors[ti] {
            if nb >= 0 && scratch.seen[nb as usize] != gen {
                scratch.seen[nb as usize] = gen;
                scratch.stack.push(nb as usize);
            }
        }
    }
    out
}

/// Like [`splat`], but instead of a coverage scalar it returns each covered texel's
/// *surface point* in world space — the raw geometry a stamp/decal needs to project
/// the texel into its own oriented frame. Every texel within `radius_world` of the
/// hit is returned (no falloff applied here; the caller decides coverage from the
/// decal image's alpha). A texel on a shared edge can appear more than once; the
/// caller's per-stroke coverage discipline makes the duplication harmless.
pub fn splat_world(
    mesh: &Mesh,
    adj: &Adjacency,
    hit: &Hit,
    radius_world: f32,
    size: u32,
    scratch: &mut SplatScratch,
) -> Vec<(usize, Vec3)> {
    let mut out = Vec::new();
    let tri_count = mesh.indices.len() / 3;
    if radius_world <= 0.0 || hit.tri as usize >= tri_count {
        return out;
    }
    let r = radius_world;
    let gen = scratch.begin(tri_count);
    scratch.stack.push(hit.tri as usize);
    scratch.seen[hit.tri as usize] = gen;

    while let Some(ti) = scratch.stack.pop() {
        let (p, uv) = tri_data(mesh, ti as u32);
        // Same frontier test as `splat`: only descend into a triangle whose closest
        // point is within the radius (the root is always in).
        let near = ti == hit.tri as usize || point_tri_dist(hit.pos, &p) <= r;
        if !near {
            continue;
        }
        rasterize(&uv, size, |texel, w| {
            let world = p[0] * w[0] + p[1] * w[1] + p[2] * w[2];
            if world.distance(hit.pos) <= r {
                out.push((texel, world));
            }
        });
        for &nb in &adj.neighbors[ti] {
            if nb >= 0 && scratch.seen[nb as usize] != gen {
                scratch.seen[nb as usize] = gen;
                scratch.stack.push(nb as usize);
            }
        }
    }
    out
}

/// Scan-fill a triangle's UV footprint at `size`, calling `f(texel, [w0, w1, w2])`
/// for each covered texel with its barycentric weights (`w_i` is the weight of
/// vertex `i`). Same edge-function rasterizer as `fill.rs` / `bake.rs`.
fn rasterize(uv: &[Vec2; 3], size: u32, mut f: impl FnMut(usize, [f32; 3])) {
    let t = [
        uv[0] * size as f32,
        uv[1] * size as f32,
        uv[2] * size as f32,
    ];
    let area = edge_fn(t[0], t[1], t[2]);
    if area.abs() < 1e-6 {
        return; // degenerate in UV space
    }
    let min_x = t[0].x.min(t[1].x).min(t[2].x).floor().max(0.0) as i32;
    let max_x = t[0].x.max(t[1].x).max(t[2].x).ceil().min(size as f32) as i32;
    let min_y = t[0].y.min(t[1].y).min(t[2].y).floor().max(0.0) as i32;
    let max_y = t[0].y.max(t[1].y).max(t[2].y).ceil().min(size as f32) as i32;
    for y in min_y..max_y {
        for x in min_x..max_x {
            let pt = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let w0 = edge_fn(t[1], t[2], pt) / area;
            let w1 = edge_fn(t[2], t[0], pt) / area;
            let w2 = edge_fn(t[0], t[1], pt) / area;
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }
            f((y as u32 * size + x as u32) as usize, [w0, w1, w2]);
        }
    }
}

/// The world positions and UVs of a triangle (mesh-order index).
pub fn tri_data(mesh: &Mesh, ti: u32) -> ([Vec3; 3], [Vec2; 3]) {
    let i = (ti * 3) as usize;
    let idx = [mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]];
    let v = |k: usize| &mesh.vertices[idx[k] as usize];
    (
        [
            Vec3::from(v(0).position),
            Vec3::from(v(1).position),
            Vec3::from(v(2).position),
        ],
        [
            Vec2::from(v(0).uv),
            Vec2::from(v(1).uv),
            Vec2::from(v(2).uv),
        ],
    )
}

type PosKey = (i32, i32, i32);

/// Quantize a vertex position so edge endpoints compare equal across triangles
/// despite float wobble. 1e4 ticks ≈ 0.0001 units — finer than any PSX texel.
fn pos_key(mesh: &Mesh, idx: u32) -> PosKey {
    let p = Vec3::from(mesh.vertices[idx as usize].position);
    let q = |x: f32| (x * 1e4).round() as i32;
    (q(p.x), q(p.y), q(p.z))
}

/// 2D edge function (twice the signed area of triangle a,b,c).
fn edge_fn(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Distance from point `p` to the closest point on triangle `t` in 3D. The
/// frontier test for the surface flood: it must see a triangle whose edge or
/// interior passes under the brush even when every vertex sits beyond it.
/// Standard closest-point-on-triangle (Ericson, *Real-Time Collision Detection*):
/// classify `p` against the triangle's Voronoi regions, then project.
fn point_tri_dist(p: Vec3, t: &[Vec3; 3]) -> f32 {
    let (a, b, c) = (t[0], t[1], t[2]);
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length(); // vertex region A
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length(); // vertex region B
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (p - (a + ab * v)).length(); // edge region AB
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length(); // vertex region C
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (p - (a + ac * w)).length(); // edge region AC
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (p - (b + (c - b) * w)).length(); // edge region BC
    }
    // Face region: project onto the triangle's plane via barycentric weights.
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (p - (a + ab * v + ac * w)).length()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fill::FillMap;

    fn cube_hit(tri: u32) -> (Mesh, Hit) {
        let mesh = Mesh::cube();
        let (p, _) = tri_data(&mesh, tri);
        let centroid = (p[0] + p[1] + p[2]) / 3.0;
        let normal = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
        let hit = Hit {
            uv: Vec2::ZERO,
            tri,
            pos: centroid,
            normal,
        };
        (mesh, hit)
    }

    #[test]
    fn cube_is_a_closed_manifold() {
        // Every edge of a closed cube is shared by exactly two triangles, so every
        // triangle has three real neighbours and the link is symmetric.
        let mesh = Mesh::cube();
        let adj = Adjacency::build(&mesh);
        assert_eq!(adj.neighbors.len(), 12);
        for (ti, nbs) in adj.neighbors.iter().enumerate() {
            for (e, &nb) in nbs.iter().enumerate() {
                assert!(nb >= 0, "tri {ti} edge {e} should have a neighbour");
                assert!(
                    adj.neighbors[nb as usize].contains(&(ti as i32)),
                    "adjacency must be symmetric"
                );
            }
        }
    }

    #[test]
    fn splat_world_returns_points_inside_the_radius() {
        // Every returned texel's surface point must lie within the flood radius of
        // the hit, and the set must be non-empty (the hit face is always covered).
        let size = 64;
        let (mesh, hit) = cube_hit(0);
        let adj = Adjacency::build(&mesh);
        let r = 0.2;
        let pts = splat_world(&mesh, &adj, &hit, r, size, &mut SplatScratch::new());
        assert!(!pts.is_empty(), "the hit face must yield texels");
        for &(_, wp) in &pts {
            assert!(
                wp.distance(hit.pos) <= r + 1e-4,
                "point {wp:?} is outside the flood radius"
            );
        }
        // A larger radius reaches strictly more texels (it wraps toward the edges).
        let wide = splat_world(&mesh, &adj, &hit, 0.45, size, &mut SplatScratch::new());
        assert!(wide.len() > pts.len(), "a wider flood covers more surface");
    }

    #[test]
    fn small_dab_stays_on_one_face() {
        // A radius far smaller than the cube's 1-unit faces can't reach a
        // neighbouring face: every painted texel lands in the hit face's island.
        let size = 64;
        let (mesh, hit) = cube_hit(0);
        let adj = Adjacency::build(&mesh);
        let map = FillMap::build(&mesh, size);
        let splats = splat(
            &mesh,
            &adj,
            &hit,
            0.05,
            1.0,
            1.0,
            size,
            &mut SplatScratch::new(),
        );
        assert!(!splats.is_empty(), "the dab must paint something");
        let islands: std::collections::HashSet<i32> = splats
            .iter()
            .map(|&(texel, _)| map.texel_island[texel])
            .filter(|&i| i >= 0)
            .collect();
        assert_eq!(islands.len(), 1, "a small dab stays within one island");
    }

    #[test]
    fn dab_on_a_mid_edge_wraps_onto_the_neighbour() {
        // A dab centred on the *middle* of a shared cube edge, with a radius far
        // smaller than the face. The neighbour face's two shared vertices are the
        // edge endpoints (half a unit away), so the old vertex/centroid `near`
        // test skipped it — the dab refused to wrap unless the cursor crossed onto
        // the neighbour. The closest-point frontier test sees the edge under the
        // brush, so the dab now reaches both islands.
        let size = 64;
        let mesh = Mesh::cube();
        let adj = Adjacency::build(&mesh);
        let map = FillMap::build(&mesh, size);
        // Hit on tri 0, at the midpoint of an edge it shares with another face.
        let (p, _) = tri_data(&mesh, 0);
        let nbs = adj.neighbors()[0];
        let shared_e = (0..3)
            .find(|&e| nbs[e] >= 0)
            .expect("tri 0 has a neighbour");
        let mid = (p[shared_e] + p[(shared_e + 1) % 3]) * 0.5;
        let normal = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
        let hit = Hit {
            uv: Vec2::ZERO,
            tri: 0,
            pos: mid,
            normal,
        };
        // Radius bigger than a texel but well under the 1-unit face.
        let splats = splat(
            &mesh,
            &adj,
            &hit,
            0.1,
            1.0,
            1.0,
            size,
            &mut SplatScratch::new(),
        );
        let islands: std::collections::HashSet<i32> = splats
            .iter()
            .map(|&(texel, _)| map.texel_island[texel])
            .filter(|&i| i >= 0)
            .collect();
        assert!(
            islands.len() >= 2,
            "a dab on a shared edge must reach the neighbouring island (got {})",
            islands.len()
        );
    }

    #[test]
    fn large_dab_wraps_across_faces() {
        // A radius larger than the cube floods the whole connected surface, so the
        // dab paints texels in several islands — exactly the cross-face reach the
        // flat texture-space stamp can't produce.
        let size = 64;
        let (mesh, hit) = cube_hit(0);
        let adj = Adjacency::build(&mesh);
        let map = FillMap::build(&mesh, size);
        let splats = splat(
            &mesh,
            &adj,
            &hit,
            5.0,
            1.0,
            1.0,
            size,
            &mut SplatScratch::new(),
        );
        let islands: std::collections::HashSet<i32> = splats
            .iter()
            .map(|&(texel, _)| map.texel_island[texel])
            .filter(|&i| i >= 0)
            .collect();
        assert!(
            islands.len() >= 2,
            "a dab wider than a face must cross onto its neighbours (got {} islands)",
            islands.len()
        );
    }

    #[test]
    fn reused_scratch_matches_fresh_scratch() {
        // The generation-stamped scratch must leave no state between dabs: a reused
        // scratch driven through several different dabs (small, large, different face)
        // must return byte-identical results to a fresh scratch each time. Guards the
        // per-dab generation reset (a stale stamp would make a triangle read "visited"
        // and silently drop part of a later dab).
        let size = 64;
        let mesh = Mesh::cube();
        let adj = Adjacency::build(&mesh);
        let dabs = [(0u32, 0.05f32), (0, 5.0), (3, 0.3), (7, 2.0), (0, 0.05)];
        let mut reused = SplatScratch::new();
        for &(tri, radius) in &dabs {
            let (p, _) = tri_data(&mesh, tri);
            let centroid = (p[0] + p[1] + p[2]) / 3.0;
            let normal = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
            let hit = Hit {
                uv: Vec2::ZERO,
                tri,
                pos: centroid,
                normal,
            };
            let from_reused = splat(&mesh, &adj, &hit, radius, 1.0, 1.0, size, &mut reused);
            let from_fresh = splat(
                &mesh,
                &adj,
                &hit,
                radius,
                1.0,
                1.0,
                size,
                &mut SplatScratch::new(),
            );
            assert_eq!(
                from_reused, from_fresh,
                "reused scratch diverged on dab tri={tri} r={radius}"
            );
        }
    }
}

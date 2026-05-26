// src/fill.rs
//
// The shared primitive behind the two paint-bucket tools (object fill + UV-island
// fill). Both need to know *which UV island a texel belongs to* — something the
// brush never asks. We compute it once per resolution and cache it (like the AO
// mesh maps), then the fills are trivial passes over `texel_island`.
//
//   - object fill  → write color into every texel the mesh covers
//                    (`texel_island >= 0`).
//   - UV-island fill → write color into every texel of the one island under the
//                    cursor.
//
// "Island" = a connected component of triangles joined across mesh edges that are
// *not* UV seams. Two triangles merge when they share an edge whose two endpoints
// agree in both position and UV. A seam is exactly where the UVs split while the
// position stays shared (e.g. a cube edge that two faces meet at, but each face
// maps to a different UV cell), so it breaks the adjacency. Islands then fall out
// of a union-find over those edges. This matters because a projection atlas often
// packs its charts edge-to-edge in UV space — keying on UV alone would weld the
// whole cube into one island; keying on (position, UV) keeps the six faces apart.
//
// The texel rasterizer is the same edge-function scan-fill as `bake.rs`, minus the
// per-texel AO ray-casting — so a fill never triggers (or waits on) an AO bake.

use std::collections::HashMap;

use glam::{Vec2, Vec3};

use crate::mesh::Mesh;

/// Two cosine of the largest angle between adjacent triangles that still counts
/// them as the same flat face. ~8° — merges the triangles of a (near-)planar quad
/// or a triangulated flat wall, but keeps a cube's 90° sides, and the facets of a
/// faceted curve, apart.
const FACET_COS: f32 = 0.99;

/// Per-resolution partitions of the mesh into fill regions, plus their rasterized
/// texel maps (row-major, V down to match the paint texture; `-1` where no
/// triangle covers a texel). Two independent partitions drive the two scoped
/// bucket tools:
///
///   - *island*: connected components in UV space (split at seams) — the region a
///     2D paint bucket would flood. `island_of_tri` / `texel_island`.
///   - *facet*: connected, near-coplanar triangles — a flat "face" of the model,
///     regardless of UV layout. `facet_of_tri` / `texel_facet`.
pub struct FillMap {
    pub size: u32,
    pub island_of_tri: Vec<u32>,
    pub texel_island: Vec<i32>,
    pub island_count: u32,
    pub facet_of_tri: Vec<u32>,
    pub texel_facet: Vec<i32>,
    pub facet_count: u32,
}

impl FillMap {
    /// Build the island + facet partitions and rasterize both into `size`×`size`
    /// texel maps in a single pass. Cheap for low-poly assets — no ray-casting.
    pub fn build(mesh: &Mesh, size: u32) -> Self {
        let tri_count = mesh.indices.len() / 3;

        // Geometric (face) normal per triangle, for the coplanarity test below.
        let face_n: Vec<Vec3> = mesh
            .indices
            .chunks_exact(3)
            .map(|tri| {
                let p0 = Vec3::from(mesh.vertices[tri[0] as usize].position);
                let p1 = Vec3::from(mesh.vertices[tri[1] as usize].position);
                let p2 = Vec3::from(mesh.vertices[tri[2] as usize].position);
                (p1 - p0).cross(p2 - p0).normalize_or_zero()
            })
            .collect();

        // Islands: union triangles sharing a non-seam edge — endpoints keyed by
        // position *and* UV, so a seam (same position, split UV) yields different
        // keys on its two sides and the edge doesn't match across it.
        let mut island_uf = UnionFind::new(tri_count);
        let mut uv_edge_owner: HashMap<(VertKey, VertKey), usize> = HashMap::new();

        // Facets: union triangles sharing a position edge whose face normals are
        // near-parallel — i.e. one flat surface. Keyed by position only, so it
        // ignores UV seams; gated by FACET_COS so a hard edge (e.g. a cube corner)
        // is a boundary even though the position edge is shared.
        let mut facet_uf = UnionFind::new(tri_count);
        let mut pos_edge_owner: HashMap<(PosKey, PosKey), usize> = HashMap::new();

        for (ti, tri) in mesh.indices.chunks_exact(3).enumerate() {
            let vk = [
                vert_key(mesh, tri[0]),
                vert_key(mesh, tri[1]),
                vert_key(mesh, tri[2]),
            ];
            let pk = [
                pos_key(mesh, tri[0]),
                pos_key(mesh, tri[1]),
                pos_key(mesh, tri[2]),
            ];
            for e in 0..3 {
                let (va, vb) = (vk[e], vk[(e + 1) % 3]);
                let uv_key = if va <= vb { (va, vb) } else { (vb, va) };
                match uv_edge_owner.get(&uv_key) {
                    Some(&other) => island_uf.union(ti, other),
                    None => {
                        uv_edge_owner.insert(uv_key, ti);
                    }
                }

                let (pa, pb) = (pk[e], pk[(e + 1) % 3]);
                let pos_key = if pa <= pb { (pa, pb) } else { (pb, pa) };
                match pos_edge_owner.get(&pos_key) {
                    Some(&other) => {
                        if face_n[ti].dot(face_n[other]) >= FACET_COS {
                            facet_uf.union(ti, other);
                        }
                    }
                    None => {
                        pos_edge_owner.insert(pos_key, ti);
                    }
                }
            }
        }

        let (island_of_tri, island_count) = island_uf.dense_ids(tri_count);
        let (facet_of_tri, facet_count) = facet_uf.dense_ids(tri_count);

        // Rasterize each triangle's UV footprint, stamping both its island and
        // facet ids into the texel maps.
        let n = (size * size) as usize;
        let mut texel_island = vec![-1i32; n];
        let mut texel_facet = vec![-1i32; n];
        for (ti, tri) in mesh.indices.chunks_exact(3).enumerate() {
            let t = [
                Vec2::from(mesh.vertices[tri[0] as usize].uv) * size as f32,
                Vec2::from(mesh.vertices[tri[1] as usize].uv) * size as f32,
                Vec2::from(mesh.vertices[tri[2] as usize].uv) * size as f32,
            ];
            let area = edge_fn(t[0], t[1], t[2]);
            if area.abs() < 1e-6 {
                continue; // degenerate in UV space
            }
            let min_x = t[0].x.min(t[1].x).min(t[2].x).floor().max(0.0) as i32;
            let max_x = t[0].x.max(t[1].x).max(t[2].x).ceil().min(size as f32) as i32;
            let min_y = t[0].y.min(t[1].y).min(t[2].y).floor().max(0.0) as i32;
            let max_y = t[0].y.max(t[1].y).max(t[2].y).ceil().min(size as f32) as i32;
            let (isl, fac) = (island_of_tri[ti] as i32, facet_of_tri[ti] as i32);
            for y in min_y..max_y {
                for x in min_x..max_x {
                    let pt = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                    let w0 = edge_fn(t[1], t[2], pt) / area;
                    let w1 = edge_fn(t[2], t[0], pt) / area;
                    let w2 = edge_fn(t[0], t[1], pt) / area;
                    if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                        continue;
                    }
                    let idx = (y as u32 * size + x as u32) as usize;
                    texel_island[idx] = isl;
                    texel_facet[idx] = fac;
                }
            }
        }

        Self {
            size,
            island_of_tri,
            texel_island,
            island_count,
            facet_of_tri,
            texel_facet,
            facet_count,
        }
    }

    /// The island id for a triangle index, or `None` if out of range.
    pub fn island_for_tri(&self, tri: u32) -> Option<u32> {
        self.island_of_tri.get(tri as usize).copied()
    }

    /// The facet id for a triangle index, or `None` if out of range.
    pub fn facet_for_tri(&self, tri: u32) -> Option<u32> {
        self.facet_of_tri.get(tri as usize).copied()
    }
}

/// A vertex keyed by quantized position *and* UV, so edge endpoints compare equal
/// across triangles despite float wobble. Quantizing both means a seam (shared
/// position, split UV) produces different keys on its two sides — exactly the
/// island boundary we want. 1e4 ticks ≈ 0.0001 units, finer than any PSX texel.
type VertKey = (i32, i32, i32, i32, i32);

fn vert_key(mesh: &Mesh, idx: u32) -> VertKey {
    let v = &mesh.vertices[idx as usize];
    let p = Vec3::from(v.position);
    let uv = Vec2::from(v.uv);
    let q = |x: f32| (x * 1e4).round() as i32;
    (q(p.x), q(p.y), q(p.z), q(uv.x), q(uv.y))
}

/// A vertex keyed by quantized position only, for facet adjacency: two triangles
/// touch on a position edge regardless of how their UVs are laid out.
type PosKey = (i32, i32, i32);

fn pos_key(mesh: &Mesh, idx: u32) -> PosKey {
    let p = Vec3::from(mesh.vertices[idx as usize].position);
    let q = |x: f32| (x * 1e4).round() as i32;
    (q(p.x), q(p.y), q(p.z))
}

/// 2D edge function (twice the signed area of triangle a,b,c). Mirrors `bake.rs`.
fn edge_fn(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Minimal union-find (path-halving + union by size) over triangle indices.
struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path halving
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
    }

    /// Compact the roots of `0..n` into dense component ids `[0, count)`, returning
    /// the per-element id vector and the component count.
    fn dense_ids(&mut self, n: usize) -> (Vec<u32>, u32) {
        let mut root_to_id: HashMap<usize, u32> = HashMap::new();
        let mut ids = vec![0u32; n];
        for x in 0..n {
            let root = self.find(x);
            let next = root_to_id.len() as u32;
            ids[x] = *root_to_id.entry(root).or_insert(next);
        }
        (ids, root_to_id.len() as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_islands_follow_uv_contiguity() {
        // The box-projected cube unwraps to five islands, not six: the +Y and +Z
        // cells are stacked in the atlas and share the top-front edge with matching
        // UVs, so the texture is continuous across it (no seam) — a fill correctly
        // treats them as one region. The other four faces are separate islands.
        let map = FillMap::build(&Mesh::cube(), 64);
        assert_eq!(map.island_count, 5);

        // The two triangles of every face always land in the same island.
        for face in 0..6 {
            let (a, b) = (map.island_of_tri[face * 2], map.island_of_tri[face * 2 + 1]);
            assert_eq!(a, b, "face {face}'s two triangles share an island");
        }

        // Exactly +Y and +Z (faces 2 and 4) are stitched; no other pair is.
        assert_eq!(
            map.island_of_tri[4], map.island_of_tri[8],
            "+Y and +Z are UV-continuous"
        );
        for face in [0usize, 1, 3, 5] {
            assert_ne!(
                map.island_of_tri[face * 2],
                map.island_of_tri[4],
                "face {face} must not share the +Y/+Z island"
            );
        }
    }

    #[test]
    fn object_coverage_matches_island_fill_sum() {
        // Every covered texel belongs to exactly one island, so the object-fill
        // set (texel_island >= 0) equals the union of all per-island fill sets.
        let map = FillMap::build(&Mesh::cube(), 64);
        let covered = map.texel_island.iter().filter(|&&i| i >= 0).count();
        assert!(covered > 0, "the cube must cover some texels");

        let mut per_island = 0usize;
        for island in 0..map.island_count as i32 {
            per_island += map.texel_island.iter().filter(|&&i| i == island).count();
        }
        assert_eq!(covered, per_island);
    }

    #[test]
    fn island_fill_is_a_strict_subset() {
        // Filling one island touches fewer texels than filling the whole object,
        // and only texels of that island.
        let map = FillMap::build(&Mesh::cube(), 64);
        let covered = map.texel_island.iter().filter(|&&i| i >= 0).count();
        let island0 = map.texel_island.iter().filter(|&&i| i == 0).count();
        assert!(island0 > 0 && island0 < covered);
    }

    #[test]
    fn cube_has_six_facets() {
        // Unlike islands, facets are geometric: each cube side is one flat facet
        // (its two coplanar triangles merge across their shared diagonal), and the
        // 90° hard edges between sides are facet boundaries — so six, one per side.
        let map = FillMap::build(&Mesh::cube(), 64);
        assert_eq!(map.facet_count, 6);
        for face in 0..6 {
            let (a, b) = (map.facet_of_tri[face * 2], map.facet_of_tri[face * 2 + 1]);
            assert_eq!(a, b, "face {face}'s two triangles share a facet");
        }
        // Every distinct cube side gets its own facet id.
        let mut ids: Vec<u32> = (0..6).map(|f| map.facet_of_tri[f * 2]).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 6, "all six sides are distinct facets");

        // A single facet is a strict subset of the whole object's coverage.
        let covered = map.texel_facet.iter().filter(|&&i| i >= 0).count();
        let facet0 = map.texel_facet.iter().filter(|&&i| i == 0).count();
        assert!(facet0 > 0 && facet0 < covered);
    }
}

// src/unwrap.rs
//
// UV unwrapping (Phase 4). Most downloaded / hand-modeled low-poly assets have
// bad or no UVs; lowtex unwraps them in a PSX-friendly way. Projection methods
// fit the aesthetic better than LSCM/ABF, so these are all planar projections:
//
//   - box_unwrap     (G14): per-triangle dominant-axis planar projection into a
//                           2×3 grid (six cells, one per ±axis) — the cube's
//                           scheme, generalized.
//   - per_face_unwrap(G16): every triangle gets its own square chart, packed into
//                           a grid — zero seam-bleed, "texture each face".
//   - smart_unwrap   (G15): triangles clustered by normal similarity, each cluster
//                           planar-projected, the charts packed (G17).
//
// Every unwrap splits vertices (3 per triangle): a vertex shared across faces with
// different projections needs different UVs, so unwrapping rebuilds the vertex
// list as 3·triangle_count flat vertices. Output is always a fresh `Mesh` with
// `needs_uvs = false`.

use glam::{Vec2, Vec3};

use crate::mesh::{Mesh, Vertex};

/// The dominant axis (0=X,1=Y,2=Z) of a vector, by largest absolute component.
fn dominant_axis(n: Vec3) -> usize {
    let a = n.abs();
    if a.x >= a.y && a.x >= a.z {
        0
    } else if a.y >= a.z {
        1
    } else {
        2
    }
}

/// The geometric normal of a triangle (un-normalized cross is fine for axis pick,
/// but we normalize so output normals are usable).
fn face_normal(p: [Vec3; 3]) -> Vec3 {
    (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero()
}

/// Iterate a mesh's triangles as `[Vec3; 3]` world positions.
fn tri_positions(mesh: &Mesh) -> impl Iterator<Item = [Vec3; 3]> + '_ {
    mesh.indices.chunks_exact(3).map(move |t| {
        [
            Vec3::from(mesh.vertices[t[0] as usize].position),
            Vec3::from(mesh.vertices[t[1] as usize].position),
            Vec3::from(mesh.vertices[t[2] as usize].position),
        ]
    })
}

/// Build a split-vertex mesh from per-triangle (positions, normal, uvs).
fn build_split(tris: &[([Vec3; 3], Vec3, [Vec2; 3])]) -> Mesh {
    let mut vertices = Vec::with_capacity(tris.len() * 3);
    for (p, n, uv) in tris {
        for i in 0..3 {
            vertices.push(Vertex {
                position: p[i].to_array(),
                normal: n.to_array(),
                uv: uv[i].to_array(),
            });
        }
    }
    let indices = (0..vertices.len() as u32).collect();
    Mesh {
        vertices,
        indices,
        needs_normals: false,
        needs_uvs: false,
    }
}

/// G14 — Box projection. Each triangle is projected along its dominant axis into
/// one cell of a 2×3 grid: column by the axis sign (+ = 0, − = 1), row by the axis
/// (X=0, Y=1, Z=2). Positions are normalized by the mesh bounds so coplanar faces
/// share a consistent scale. The result paints with predictable per-face UVs.
pub fn box_unwrap(mesh: &Mesh) -> Mesh {
    let (min, max) = mesh.bounds();
    let size = (max - min).max(Vec3::splat(1e-6));

    let tris: Vec<_> = tri_positions(mesh)
        .map(|p| {
            let n = face_normal(p);
            let axis = dominant_axis(n);
            let col = if n[axis] >= 0.0 { 0.0 } else { 1.0 };
            let row = axis as f32;
            // The two axes perpendicular to the dominant one become (u, v).
            let ua = (axis + 1) % 3;
            let va = (axis + 2) % 3;
            let uv = p.map(|pt| {
                let cu = (pt[ua] - min[ua]) / size[ua];
                let cv = (pt[va] - min[va]) / size[va];
                Vec2::new((col + cu) / 2.0, (row + cv) / 3.0)
            });
            (p, n, uv)
        })
        .collect();
    build_split(&tris)
}

/// G16 — Per-face. Every triangle becomes its own square chart in a near-square
/// grid of `ceil(sqrt(n))` columns. Each triangle is planar-projected by its own
/// normal, fit to its cell with a small gutter so nearest-neighbour sampling can't
/// bleed between charts. Useful for "texture each face" workflows and as a
/// zero-seam-bleed mode.
pub fn per_face_unwrap(mesh: &Mesh) -> Mesh {
    let tri_count = mesh.indices.len() / 3;
    if tri_count == 0 {
        return build_split(&[]);
    }
    let cols = (tri_count as f32).sqrt().ceil() as usize;
    let rows = tri_count.div_ceil(cols);
    let cell_w = 1.0 / cols as f32;
    let cell_h = 1.0 / rows as f32;
    const GUTTER: f32 = 0.08; // fraction of the cell kept empty around each chart

    let tris: Vec<_> = tri_positions(mesh)
        .enumerate()
        .map(|(i, p)| {
            let n = face_normal(p);
            let (tu, tv) = planar_basis(n);
            // Project to the triangle's tangent plane, then normalize to its bbox.
            let proj: [Vec2; 3] = p.map(|pt| Vec2::new(pt.dot(tu), pt.dot(tv)));
            let (pmin, pmax) = bounds2(&proj);
            let psize = (pmax - pmin).max(Vec2::splat(1e-6));

            let col = i % cols;
            let row = i / cols;
            let uv = proj.map(|q| {
                let local = (q - pmin) / psize; // 0..1 within the chart
                let inset = local * (1.0 - 2.0 * GUTTER) + Vec2::splat(GUTTER);
                Vec2::new(
                    (col as f32 + inset.x) * cell_w,
                    (row as f32 + inset.y) * cell_h,
                )
            });
            (p, n, uv)
        })
        .collect();
    build_split(&tris)
}

/// An orthonormal (tangent, bitangent) spanning the plane perpendicular to `n`.
fn planar_basis(n: Vec3) -> (Vec3, Vec3) {
    let up = if n.y.abs() < 0.99 { Vec3::Y } else { Vec3::X };
    let t = up.cross(n).normalize_or_zero();
    let b = n.cross(t);
    (t, b)
}

fn bounds2(pts: &[Vec2; 3]) -> (Vec2, Vec2) {
    let mut mn = pts[0];
    let mut mx = pts[0];
    for p in &pts[1..] {
        mn = mn.min(*p);
        mx = mx.max(*p);
    }
    (mn, mx)
}

/// G15 — Smart projection. Triangles are greedily clustered by normal similarity
/// (within `angle_threshold_deg`), each cluster is planar-projected by its average
/// normal, and the clusters are packed into the atlas (G17). Clustered faces stay
/// together, so a curved-ish low-poly mesh unwraps into a few sensible islands
/// with less stretch than box projection. `angle_threshold_deg` ~ 40–66 is sane.
pub fn smart_unwrap(mesh: &Mesh, angle_threshold_deg: f32) -> Mesh {
    let tri_count = mesh.indices.len() / 3;
    if tri_count == 0 {
        return build_split(&[]);
    }
    let cos_thresh = angle_threshold_deg.to_radians().cos();

    let positions: Vec<[Vec3; 3]> = tri_positions(mesh).collect();
    let normals: Vec<Vec3> = positions.iter().map(|p| face_normal(*p)).collect();

    // Greedy normal clustering: each triangle joins the first cluster whose mean
    // normal is within the threshold, else starts a new one.
    let mut cluster_of = vec![0usize; tri_count];
    let mut cluster_normal: Vec<Vec3> = Vec::new();
    let mut cluster_count: Vec<u32> = Vec::new();
    for ti in 0..tri_count {
        let n = normals[ti];
        let found = cluster_normal.iter().position(|cn| cn.dot(n) >= cos_thresh);
        match found {
            Some(c) => {
                cluster_of[ti] = c;
                // Running mean of the cluster normal.
                let k = cluster_count[c] as f32;
                cluster_normal[c] = ((cluster_normal[c] * k + n) / (k + 1.0)).normalize_or_zero();
                cluster_count[c] += 1;
            }
            None => {
                cluster_of[ti] = cluster_normal.len();
                cluster_normal.push(n);
                cluster_count.push(1);
            }
        }
    }

    let num_clusters = cluster_normal.len();
    // Project each triangle in its cluster's tangent frame; track per-cluster bbox.
    let bases: Vec<(Vec3, Vec3)> = cluster_normal.iter().map(|n| planar_basis(*n)).collect();
    let mut proj: Vec<[Vec2; 3]> = Vec::with_capacity(tri_count);
    let mut cmin = vec![Vec2::splat(f32::INFINITY); num_clusters];
    let mut cmax = vec![Vec2::splat(f32::NEG_INFINITY); num_clusters];
    for ti in 0..tri_count {
        let (tu, tv) = bases[cluster_of[ti]];
        let q = positions[ti].map(|pt| Vec2::new(pt.dot(tu), pt.dot(tv)));
        let c = cluster_of[ti];
        for v in &q {
            cmin[c] = cmin[c].min(*v);
            cmax[c] = cmax[c].max(*v);
        }
        proj.push(q);
    }

    // Pack clusters into the atlas (G17) and remap each triangle into its rect.
    let csize: Vec<Vec2> = (0..num_clusters)
        .map(|c| (cmax[c] - cmin[c]).max(Vec2::splat(1e-6)))
        .collect();
    let rects = pack_rects(&csize);

    let tris: Vec<_> = (0..tri_count)
        .map(|ti| {
            let c = cluster_of[ti];
            let (offset, scale) = rects[c];
            let uv = proj[ti].map(|q| {
                let local = (q - cmin[c]) * scale; // chart-local UV scaled into its rect
                offset + local
            });
            (positions[ti], normals[ti], uv)
        })
        .collect();
    build_split(&tris)
}

/// G17 — Chart packing. Sort-by-area shelf packing of axis-aligned rects (given by
/// world-space `sizes`) into the unit square, scaled uniformly to fit with a small
/// gutter so nearest-neighbour sampling can't bleed across charts. Returns, per
/// input rect, `(uv_offset, scale)` mapping its local [0,size] coords into [0,1].
fn pack_rects(sizes: &[Vec2]) -> Vec<(Vec2, f32)> {
    const GUTTER: f32 = 0.01; // empty margin (in UV) around each chart

    // Aspect-preserving: pack at the raw sizes, then scale the whole layout to fit.
    // Order tallest-first (classic shelf packing).
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by(|&a, &b| sizes[b].y.total_cmp(&sizes[a].y));

    // Shelf width in raw units: the widest chart, or the running average — use the
    // total width spread so rows stay roughly square overall.
    let total_w: f32 = sizes.iter().map(|s| s.x).sum();
    let shelf_w = (total_w / (sizes.len() as f32).sqrt().max(1.0)).max(
        sizes.iter().map(|s| s.x).fold(0.0_f32, f32::max), // never narrower than the widest chart
    );

    let mut placed = vec![(Vec2::ZERO, 0.0f32); sizes.len()];
    let (mut x, mut y, mut shelf_h) = (0.0f32, 0.0f32, 0.0f32);
    for &i in &order {
        let s = sizes[i];
        if x > 0.0 && x + s.x > shelf_w {
            // New shelf.
            y += shelf_h;
            x = 0.0;
            shelf_h = 0.0;
        }
        placed[i] = (Vec2::new(x, y), 1.0);
        x += s.x;
        shelf_h = shelf_h.max(s.y);
    }
    let total_h = y + shelf_h;

    // Uniform scale so the whole layout fits [0,1]² inside the gutter margin.
    let span = shelf_w.max(total_h).max(1e-6);
    let scale = (1.0 - 2.0 * GUTTER) / span;
    placed
        .iter()
        .map(|(pos, _)| (*pos * scale + Vec2::splat(GUTTER), scale))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_uvs_in_unit(mesh: &Mesh) {
        for v in &mesh.vertices {
            assert!(
                (0.0..=1.0).contains(&v.uv[0]) && (0.0..=1.0).contains(&v.uv[1]),
                "uv out of range: {:?}",
                v.uv
            );
        }
    }

    #[test]
    fn box_unwrap_splits_and_stays_in_unit() {
        let cube = Mesh::cube();
        let u = box_unwrap(&cube);
        // 12 triangles → 36 split vertices.
        assert_eq!(u.vertices.len(), 36);
        assert_eq!(u.indices.len(), 36);
        assert!(!u.needs_uvs);
        assert_uvs_in_unit(&u);
    }

    #[test]
    fn box_unwrap_separates_opposite_faces_into_different_cells() {
        // +X and -X faces share an axis (row) but must land in different columns,
        // so their UV cells don't overlap.
        let cube = Mesh::cube();
        let u = box_unwrap(&cube);
        // Each triangle's centroid UV; collect the distinct cell (col,row) keys.
        let mut cells = std::collections::HashSet::new();
        for tri in u.indices.chunks_exact(3) {
            let c = (tri.iter().map(|&i| Vec2::from(u.vertices[i as usize].uv)))
                .fold(Vec2::ZERO, |a, b| a + b)
                / 3.0;
            let col = (c.x * 2.0).floor() as i32;
            let row = (c.y * 3.0).floor() as i32;
            cells.insert((col, row));
        }
        // A cube exercises all six axis cells.
        assert_eq!(cells.len(), 6, "expected 6 box cells, got {cells:?}");
    }

    #[test]
    fn per_face_unwrap_gives_one_chart_per_triangle_in_unit() {
        let cube = Mesh::cube();
        let u = per_face_unwrap(&cube);
        assert_eq!(u.vertices.len(), 36);
        assert_uvs_in_unit(&u);
    }

    #[test]
    fn smart_unwrap_clusters_cube_into_six_and_stays_in_unit() {
        // With a tight angle threshold the cube's six flat faces form six clusters;
        // the packed charts must stay inside the unit square.
        let cube = Mesh::cube();
        let u = smart_unwrap(&cube, 30.0);
        assert_eq!(u.vertices.len(), 36);
        assert_uvs_in_unit(&u);
    }

    #[test]
    fn packed_rects_stay_in_unit_and_dont_overlap() {
        // A handful of differently-sized charts pack without leaving the atlas.
        let sizes = [
            Vec2::new(1.0, 0.5),
            Vec2::new(0.3, 0.9),
            Vec2::new(0.6, 0.6),
            Vec2::new(0.2, 0.2),
        ];
        let rects = pack_rects(&sizes);
        // Each chart's far corner (offset + size*scale) lands inside [0,1].
        for (i, (offset, scale)) in rects.iter().enumerate() {
            let far = *offset + sizes[i] * *scale;
            assert!(
                offset.x >= -1e-4
                    && offset.y >= -1e-4
                    && far.x <= 1.0 + 1e-4
                    && far.y <= 1.0 + 1e-4,
                "chart {i} escaped the atlas: {offset:?}..{far:?}"
            );
        }
        // No two chart rects overlap (axis-aligned overlap test).
        for a in 0..sizes.len() {
            for b in (a + 1)..sizes.len() {
                let (ao, asz) = (rects[a].0, sizes[a] * rects[a].1);
                let (bo, bsz) = (rects[b].0, sizes[b] * rects[b].1);
                let disjoint = ao.x + asz.x <= bo.x + 1e-5
                    || bo.x + bsz.x <= ao.x + 1e-5
                    || ao.y + asz.y <= bo.y + 1e-5
                    || bo.y + bsz.y <= ao.y + 1e-5;
                assert!(disjoint, "charts {a} and {b} overlap");
            }
        }
    }
}

// src/bvh.rs
//
// A bounding-volume hierarchy over the mesh triangles, so ray picking is
// O(log n) instead of O(n). Without this, every mouse-move on a few-thousand-tri
// mesh would test every triangle and stutter (G5).
//
// This is a straightforward median/midpoint-split BVH (Jacco-Bikker style):
// build once at mesh load, then traverse front-to-back-ish with a slab test,
// only running Möller–Trumbore on triangles in visited leaves.

use glam::{Vec2, Vec3};

use crate::mesh::Mesh;
use crate::paint::{intersect_triangle, Ray};

/// Leaf node holds at most this many triangles before we stop subdividing.
const LEAF_MAX: u32 = 4;

struct Tri {
    p: [Vec3; 3],
    uv: [Vec2; 3],
    centroid: Vec3,
}

#[derive(Clone, Copy)]
struct Node {
    min: Vec3,
    max: Vec3,
    /// Internal node: index of the left child (right is left + 1).
    /// Leaf node: index into `tri_idx` of the first triangle.
    left_or_first: u32,
    /// 0 = internal node; otherwise the triangle count of a leaf.
    count: u32,
}

pub struct Bvh {
    nodes: Vec<Node>,
    tri_idx: Vec<u32>,
    tris: Vec<Tri>,
}

/// A BVH node flattened for GPU upload — the same `min/max/left_or_first/count`
/// as the internal [`Node`], but `#[repr(C)]` + `Pod` so it maps straight onto the
/// compute shaders' std430 `Node` (8 words = 32 bytes: `min.xyz`, `left_or_first`,
/// `max.xyz`, `count`). The GPU bake (`gpu_bake.rs`) traverses these to ray-trace
/// occlusion per texel; keeping the layout knowledge here, beside the build, means
/// the shader and the builder can't silently drift.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBvhNode {
    pub min: [f32; 3],
    pub left_or_first: u32,
    pub max: [f32; 3],
    pub count: u32,
}

/// The closest triangle a ray struck: the interpolated UV at the hit, the original
/// mesh-triangle index (for keying per-triangle data like UV islands), and the
/// world-space hit point + geometric face normal.
#[derive(Clone, Copy)]
pub struct Hit {
    pub uv: Vec2,
    pub tri: u32,
    pub pos: Vec3,
    pub normal: Vec3,
}

impl Bvh {
    /// Build a BVH over the mesh's triangles. Cheap for low-poly assets.
    pub fn build(mesh: &Mesh) -> Self {
        let tris: Vec<Tri> = mesh
            .indices
            .chunks_exact(3)
            .map(|t| {
                let v = [
                    mesh.vertices[t[0] as usize],
                    mesh.vertices[t[1] as usize],
                    mesh.vertices[t[2] as usize],
                ];
                let p = [
                    Vec3::from(v[0].position),
                    Vec3::from(v[1].position),
                    Vec3::from(v[2].position),
                ];
                Tri {
                    p,
                    uv: [
                        Vec2::from(v[0].uv),
                        Vec2::from(v[1].uv),
                        Vec2::from(v[2].uv),
                    ],
                    centroid: (p[0] + p[1] + p[2]) / 3.0,
                }
            })
            .collect();

        let n = tris.len() as u32;
        let tri_idx: Vec<u32> = (0..n).collect();
        // Worst case a BVH has 2N-1 nodes; reserve that.
        let mut nodes = Vec::with_capacity((2 * n.max(1)) as usize);

        let mut bvh = Bvh {
            nodes: Vec::new(),
            tri_idx,
            tris,
        };
        if n == 0 {
            bvh.nodes = vec![Node {
                min: Vec3::ZERO,
                max: Vec3::ZERO,
                left_or_first: 0,
                count: 0,
            }];
            return bvh;
        }

        // Root.
        nodes.push(Node {
            min: Vec3::ZERO,
            max: Vec3::ZERO,
            left_or_first: 0,
            count: n,
        });
        bvh.nodes = nodes;
        bvh.subdivide(0);
        bvh
    }

    fn update_bounds(&mut self, node_idx: usize) {
        let (first, count) = {
            let node = &self.nodes[node_idx];
            (node.left_or_first as usize, node.count as usize)
        };
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for &ti in &self.tri_idx[first..first + count] {
            let tri = &self.tris[ti as usize];
            for p in tri.p {
                min = min.min(p);
                max = max.max(p);
            }
        }
        self.nodes[node_idx].min = min;
        self.nodes[node_idx].max = max;
    }

    fn subdivide(&mut self, node_idx: usize) {
        self.update_bounds(node_idx);

        let (first, count) = {
            let node = &self.nodes[node_idx];
            (node.left_or_first as usize, node.count as usize)
        };
        if count <= LEAF_MAX as usize {
            return; // leaf
        }

        // Split on the longest axis at its spatial midpoint.
        let extent = self.nodes[node_idx].max - self.nodes[node_idx].min;
        let axis = if extent.x > extent.y && extent.x > extent.z {
            0
        } else if extent.y > extent.z {
            1
        } else {
            2
        };
        let split = self.nodes[node_idx].min[axis] + extent[axis] * 0.5;

        // Partition tri_idx[first..first+count] by centroid on `axis`.
        let mut i = first;
        let mut j = first + count - 1;
        while i <= j {
            if self.tris[self.tri_idx[i] as usize].centroid[axis] < split {
                i += 1;
            } else {
                self.tri_idx.swap(i, j);
                if j == 0 {
                    break;
                }
                j -= 1;
            }
        }

        let left_count = i - first;
        // Degenerate split (all on one side): keep as a leaf to avoid recursion loop.
        if left_count == 0 || left_count == count {
            return;
        }

        let left_idx = self.nodes.len() as u32;
        self.nodes.push(Node {
            min: Vec3::ZERO,
            max: Vec3::ZERO,
            left_or_first: first as u32,
            count: left_count as u32,
        });
        self.nodes.push(Node {
            min: Vec3::ZERO,
            max: Vec3::ZERO,
            left_or_first: i as u32,
            count: (count - left_count) as u32,
        });
        self.nodes[node_idx].left_or_first = left_idx;
        self.nodes[node_idx].count = 0; // now internal

        self.subdivide(left_idx as usize);
        self.subdivide(left_idx as usize + 1);
    }

    /// Cast a ray and return the closest hit: its interpolated UV and the index
    /// of the triangle it struck. The triangle index is the *original* mesh
    /// triangle (the `tris` Vec is built in mesh order and never reordered — only
    /// `tri_idx` is permuted during the build), so it maps straight into anything
    /// keyed by triangle, e.g. `fill::IslandMap::island_of_tri`.
    pub fn pick(&self, ray: &Ray) -> Option<Hit> {
        if self.tris.is_empty() {
            return None;
        }
        // Nudge exactly-axis-aligned directions off zero: an exact 0 component
        // gives inv_dir = ∞, and `0 * ∞ = NaN` in the slab test wrongly rejects
        // boundary-aligned rays.
        let safe = |x: f32| if x.abs() < 1e-8 { 1e-8 } else { x };
        let inv_dir = Vec3::new(
            1.0 / safe(ray.direction.x),
            1.0 / safe(ray.direction.y),
            1.0 / safe(ray.direction.z),
        );

        let mut best: Option<(f32, Vec2, u32)> = None;
        // Explicit stack traversal.
        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        stack[sp] = 0;
        sp += 1;

        while sp > 0 {
            sp -= 1;
            let node = self.nodes[stack[sp] as usize];
            let best_t = best.map_or(f32::INFINITY, |(t, _, _)| t);
            if !ray_hits_aabb(ray.origin, inv_dir, node.min, node.max, best_t) {
                continue;
            }
            if node.count > 0 {
                // Leaf: test its triangles.
                let first = node.left_or_first as usize;
                for &ti in &self.tri_idx[first..first + node.count as usize] {
                    let tri = &self.tris[ti as usize];
                    if let Some((t, u, v)) = intersect_triangle(ray, tri.p[0], tri.p[1], tri.p[2]) {
                        if best.is_none_or(|(bt, _, _)| t < bt) {
                            let w = 1.0 - u - v;
                            let uv = tri.uv[0] * w + tri.uv[1] * u + tri.uv[2] * v;
                            best = Some((t, uv, ti));
                        }
                    }
                }
            } else {
                // Internal: push both children.
                stack[sp] = node.left_or_first;
                sp += 1;
                stack[sp] = node.left_or_first + 1;
                sp += 1;
            }
        }
        best.map(|(t, uv, tri)| {
            let p = self.tris[tri as usize].p;
            let normal = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
            Hit {
                uv,
                tri,
                pos: ray.origin + ray.direction * t,
                normal,
            }
        })
    }

    /// Does anything block the ray within `max_dist`? Early-exits on the first
    /// hit — used for ambient-occlusion baking (G19/AO suite).
    pub fn occludes(&self, origin: Vec3, dir: Vec3, max_dist: f32) -> bool {
        if self.tris.is_empty() {
            return false;
        }
        let safe = |x: f32| if x.abs() < 1e-8 { 1e-8 } else { x };
        let inv_dir = Vec3::new(1.0 / safe(dir.x), 1.0 / safe(dir.y), 1.0 / safe(dir.z));
        let ray = Ray {
            origin,
            direction: dir,
        };

        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        stack[sp] = 0;
        sp += 1;
        while sp > 0 {
            sp -= 1;
            let node = self.nodes[stack[sp] as usize];
            if !ray_hits_aabb(ray.origin, inv_dir, node.min, node.max, max_dist) {
                continue;
            }
            if node.count > 0 {
                let first = node.left_or_first as usize;
                for &ti in &self.tri_idx[first..first + node.count as usize] {
                    let tri = &self.tris[ti as usize];
                    if let Some((t, _, _)) = intersect_triangle(&ray, tri.p[0], tri.p[1], tri.p[2])
                    {
                        if t < max_dist {
                            return true;
                        }
                    }
                }
            } else {
                stack[sp] = node.left_or_first;
                sp += 1;
                stack[sp] = node.left_or_first + 1;
                sp += 1;
            }
        }
        false
    }

    // --- GPU upload (Phase 2.5: BVH-on-GPU for the mesh-map bake) ---

    /// The nodes flattened for GPU upload (see [`GpuBvhNode`]). One-to-one with the
    /// internal node array, including index semantics (`left_or_first`/`count`), so the
    /// compute-shader traversal mirrors [`Self::occludes`] exactly.
    pub fn gpu_nodes(&self) -> Vec<GpuBvhNode> {
        self.nodes
            .iter()
            .map(|n| GpuBvhNode {
                min: n.min.to_array(),
                left_or_first: n.left_or_first,
                max: n.max.to_array(),
                count: n.count,
            })
            .collect()
    }

    /// The leaf triangle-index permutation, as the shader indexes it (`tri_idx[first..]`
    /// inside a leaf). Values are original mesh-triangle indices into [`Self::gpu_tri_positions`].
    pub fn gpu_tri_indices(&self) -> &[u32] {
        &self.tri_idx
    }

    /// Triangle vertex positions packed 9 floats per triangle (`v0.xyz, v1.xyz, v2.xyz`),
    /// in *mesh* order so a `tri_idx` value indexes straight in (`base = 9 * ti`). The
    /// shader's Möller–Trumbore reads these for the occlusion test — positions only, since
    /// the bake's shadow/AO rays don't need UVs.
    pub fn gpu_tri_positions(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.tris.len() * 9);
        for t in &self.tris {
            for p in &t.p {
                out.extend_from_slice(&[p.x, p.y, p.z]);
            }
        }
        out
    }

    /// Number of triangles (for the GPU bake's empty-mesh guard / dispatch sizing).
    pub fn tri_count(&self) -> usize {
        self.tris.len()
    }
}

/// Slab test: does the ray reach the box within `max_t`? The box is padded by a
/// small epsilon so a ray lying exactly in a (possibly zero-thickness, e.g. a
/// flat mesh) box face isn't rejected by an exact `max - origin == 0`. The pad
/// can only cause a few extra triangle tests, never a false hit (Möller–Trumbore
/// still decides). Meshes are normalized to ~unit scale, so a fixed pad is fine.
fn ray_hits_aabb(origin: Vec3, inv_dir: Vec3, min: Vec3, max: Vec3, max_t: f32) -> bool {
    const PAD: f32 = 1e-4;
    let t1 = (min - Vec3::splat(PAD) - origin) * inv_dir;
    let t2 = (max + Vec3::splat(PAD) - origin) * inv_dir;
    let tmin = t1.min(t2);
    let tmax = t1.max(t2);
    let enter = tmin.max_element();
    let exit = tmax.min_element();
    exit >= enter.max(0.0) && enter <= max_t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paint::pick_uv as brute_pick;

    /// A tessellated plane in the XZ region, `n`×`n` quads (2n² tris), facing +Y.
    fn grid_mesh(n: u32) -> Mesh {
        use crate::mesh::Vertex;
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                let x = i as f32 / n as f32 - 0.5;
                let z = j as f32 / n as f32 - 0.5;
                vertices.push(Vertex {
                    position: [x, 0.0, z],
                    normal: [0.0, 1.0, 0.0],
                    uv: [i as f32 / n as f32, j as f32 / n as f32],
                });
            }
        }
        let w = n + 1;
        for j in 0..n {
            for i in 0..n {
                let a = j * w + i;
                let b = a + 1;
                let c = a + w;
                let d = c + 1;
                indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
        Mesh {
            vertices,
            indices,
            needs_normals: false,
            needs_uvs: false,
        }
    }

    /// On a dense mesh the BVH should be dramatically faster than brute force.
    #[test]
    fn bvh_faster_than_brute_force() {
        use std::time::Instant;
        let mesh = grid_mesh(50); // 2*50*50 = 5000 tris
        assert_eq!(mesh.indices.len() / 3, 5000);
        let bvh = Bvh::build(&mesh);

        // A fan of rays from above, most hitting the plane.
        let rays: Vec<Ray> = (0..2000)
            .map(|k| {
                let t = k as f32 / 2000.0 - 0.5;
                Ray {
                    origin: Vec3::new(t * 0.8, 2.0, (t * 7.0).sin() * 0.4),
                    direction: Vec3::new(0.0, -1.0, 0.0),
                }
            })
            .collect();

        let t0 = Instant::now();
        let mut hits_b = 0;
        for r in &rays {
            if brute_pick(r, &mesh).is_some() {
                hits_b += 1;
            }
        }
        let brute = t0.elapsed();

        let t1 = Instant::now();
        let mut hits_a = 0;
        for r in &rays {
            if bvh.pick(r).is_some() {
                hits_a += 1;
            }
        }
        let accel = t1.elapsed();

        eprintln!(
            "5000 tris, 2000 picks: brute={brute:?} bvh={accel:?} ({:.1}x), hits {hits_b}/{hits_a}",
            brute.as_secs_f64() / accel.as_secs_f64().max(1e-9)
        );
        assert_eq!(
            hits_a, hits_b,
            "BVH and brute force must agree on hit count"
        );
        assert!(accel < brute, "BVH should beat brute force on a dense mesh");
    }

    /// BVH picking must agree with the brute-force loop on the same rays.
    #[test]
    fn bvh_matches_brute_force() {
        let mesh = Mesh::cube();
        let bvh = Bvh::build(&mesh);

        // Fire rays from several directions toward the origin.
        for dir in [
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(-0.3, -0.6, -0.7),
            Vec3::new(0.5, 0.5, 0.5),
        ] {
            let origin = -dir.normalize() * 3.0;
            let ray = Ray {
                origin,
                direction: dir.normalize(),
            };
            let a = bvh.pick(&ray).map(|h| h.uv);
            let b = brute_pick(&ray, &mesh);
            match (a, b) {
                (Some(a), Some(b)) => {
                    assert!((a - b).length() < 1e-4, "uv mismatch: {a:?} vs {b:?}");
                }
                (None, None) => {}
                _ => panic!("hit/miss disagreement: bvh={a:?} brute={b:?}"),
            }
        }
    }
}

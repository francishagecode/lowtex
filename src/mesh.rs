// src/mesh.rs
//
// Vertex layout and a hardcoded cube with box-projected UVs.
//
// Why a cube instead of loading a glTF? v0.1 is about proving the paint loop.
// Mesh loading is its own can of worms (importers, normals, tangents, errors)
// and we don't need it to validate "click → texel updates → see it on screen."
// v0.2 adds glTF import.
//
// The UVs here use straightforward box projection: each face gets a corner of
// a 2x3 grid in the texture. This is exactly the simplest PSX-style unwrap.

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl Vertex {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // normal
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // uv
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

/// The recenter+rescale lowtex applies on import so any model is framed by the
/// camera (`recenter_and_normalize`). Stored so export can undo it and write the
/// geometry back in the source's original coordinates: `original = p / scale + center`.
/// The identity (`center = 0`, `scale = 1`) means "no transform applied" — used by
/// the built-in cube and any procedurally-built mesh.
#[derive(Copy, Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SourceTransform {
    pub center: [f32; 3],
    pub scale: f32,
}

impl SourceTransform {
    pub const IDENTITY: SourceTransform = SourceTransform {
        center: [0.0, 0.0, 0.0],
        scale: 1.0,
    };

    /// Map a normalized (in-lowtex) position back to the source's coordinates.
    pub fn to_source(&self, p: [f32; 3]) -> [f32; 3] {
        let inv = 1.0 / self.scale;
        [
            p[0] * inv + self.center[0],
            p[1] * inv + self.center[1],
            p[2] * inv + self.center[2],
        ]
    }
}

impl Default for SourceTransform {
    fn default() -> Self {
        SourceTransform::IDENTITY
    }
}

/// One named object/group from the source file (OBJ `o`/`g`, glTF mesh/primitive),
/// as a contiguous run of triangles. lowtex merges everything into one paintable mesh,
/// but keeps these so export can re-emit the original `o <name>` split. Triangle indices
/// are stable across unwrap (it preserves triangle order), so the ranges stay valid.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MeshGroup {
    pub name: String,
    /// First triangle (0-based) and triangle count in the mesh's triangle list.
    pub start_tri: usize,
    pub tri_count: usize,
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// True if the source had no normals and they were (or must be) synthesized.
    pub needs_normals: bool,
    /// True if the source had no UVs and a projection fallback is (or must be) used.
    pub needs_uvs: bool,
    /// The import recenter/rescale, kept so OBJ export can restore original coordinates.
    pub source_transform: SourceTransform,
    /// The source file's object/group split, kept so export can re-emit it. Empty when
    /// the source had no named groups (or for procedural/built-in meshes).
    pub groups: Vec<MeshGroup>,
}

impl Mesh {
    /// Build a unit cube centered at the origin.
    ///
    /// UV layout: 2 columns × 3 rows, so each face occupies a third of the
    /// vertical space and half the horizontal. Faces are ordered:
    ///   +X (right) | -X (left)
    ///   +Y (top)   | -Y (bottom)
    ///   +Z (front) | -Z (back)
    pub fn cube() -> Self {
        let h = 0.5;
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Each face: (normal, four corner positions CCW from bottom-left, uv cell)
        let faces: [(Vec3, [Vec3; 4], (u32, u32)); 6] = [
            // +X right
            (
                Vec3::X,
                [
                    Vec3::new(h, -h, h),
                    Vec3::new(h, -h, -h),
                    Vec3::new(h, h, -h),
                    Vec3::new(h, h, h),
                ],
                (0, 0),
            ),
            // -X left
            (
                -Vec3::X,
                [
                    Vec3::new(-h, -h, -h),
                    Vec3::new(-h, -h, h),
                    Vec3::new(-h, h, h),
                    Vec3::new(-h, h, -h),
                ],
                (1, 0),
            ),
            // +Y top
            (
                Vec3::Y,
                [
                    Vec3::new(-h, h, h),
                    Vec3::new(h, h, h),
                    Vec3::new(h, h, -h),
                    Vec3::new(-h, h, -h),
                ],
                (0, 1),
            ),
            // -Y bottom
            (
                -Vec3::Y,
                [
                    Vec3::new(-h, -h, -h),
                    Vec3::new(h, -h, -h),
                    Vec3::new(h, -h, h),
                    Vec3::new(-h, -h, h),
                ],
                (1, 1),
            ),
            // +Z front
            (
                Vec3::Z,
                [
                    Vec3::new(-h, -h, h),
                    Vec3::new(h, -h, h),
                    Vec3::new(h, h, h),
                    Vec3::new(-h, h, h),
                ],
                (0, 2),
            ),
            // -Z back
            (
                -Vec3::Z,
                [
                    Vec3::new(h, -h, -h),
                    Vec3::new(-h, -h, -h),
                    Vec3::new(-h, h, -h),
                    Vec3::new(h, h, -h),
                ],
                (1, 2),
            ),
        ];

        for (normal, corners, (col, row)) in faces {
            let base = vertices.len() as u32;

            // UV cell occupies [col/2 .. (col+1)/2] x [row/3 .. (row+1)/3]
            let u0 = col as f32 / 2.0;
            let u1 = (col + 1) as f32 / 2.0;
            let v0 = row as f32 / 3.0;
            let v1 = (row + 1) as f32 / 3.0;

            let uvs = [
                Vec2::new(u0, v1),
                Vec2::new(u1, v1),
                Vec2::new(u1, v0),
                Vec2::new(u0, v0),
            ];

            for i in 0..4 {
                vertices.push(Vertex {
                    position: corners[i].to_array(),
                    normal: normal.to_array(),
                    uv: uvs[i].to_array(),
                });
            }

            // Two triangles per quad (CCW).
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        Self {
            vertices,
            indices,
            needs_normals: false,
            needs_uvs: false,
            source_transform: SourceTransform::IDENTITY,
            groups: Vec::new(),
        }
    }

    /// Axis-aligned bounding box (min, max) over all vertex positions.
    /// Returns a unit box centered at the origin for an empty mesh.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        if self.vertices.is_empty() {
            return (Vec3::splat(-0.5), Vec3::splat(0.5));
        }
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for v in &self.vertices {
            let p = Vec3::from(v.position);
            min = min.min(p);
            max = max.max(p);
        }
        (min, max)
    }

    /// Recenter the mesh on the origin and scale it so its largest dimension is
    /// `target` units. This keeps any imported model framed by the static camera
    /// (which sits ~3.5 units out and expects a roughly unit-sized subject).
    pub fn recenter_and_normalize(&mut self, target: f32) {
        let (min, max) = self.bounds();
        let center = (min + max) * 0.5;
        let extent = (max - min).max_element().max(1e-6);
        let scale = target / extent;
        for v in &mut self.vertices {
            let p = (Vec3::from(v.position) - center) * scale;
            v.position = p.to_array();
        }
        // Remember the transform so export can map positions back to the source's
        // original placement and scale (the painter brings the texture *and* geometry
        // into an engine, and the geometry must land where the original did).
        self.source_transform = SourceTransform {
            center: center.to_array(),
            scale,
        };
    }

    /// Recompute smooth (area-weighted) vertex normals from the triangle faces.
    /// Used when an imported mesh ships without normals.
    pub fn compute_smooth_normals(&mut self) {
        let mut accum = vec![Vec3::ZERO; self.vertices.len()];
        for tri in self.indices.chunks_exact(3) {
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            let p0 = Vec3::from(self.vertices[i0].position);
            let p1 = Vec3::from(self.vertices[i1].position);
            let p2 = Vec3::from(self.vertices[i2].position);
            // Cross product magnitude is proportional to triangle area, so summing
            // un-normalized face normals area-weights the result naturally.
            let face = (p1 - p0).cross(p2 - p0);
            accum[i0] += face;
            accum[i1] += face;
            accum[i2] += face;
        }
        for (v, n) in self.vertices.iter_mut().zip(accum) {
            let n = n.normalize_or_zero();
            let n = if n == Vec3::ZERO { Vec3::Y } else { n };
            v.normal = n.to_array();
        }
    }

    /// Assign box-projected UVs to every vertex, based on the dominant axis of
    /// its normal. This is the fallback that makes a UV-less mesh paintable; it
    /// matches the cube's scheme and is generalized properly in G14.
    pub fn box_project_uvs(&mut self) {
        let (min, max) = self.bounds();
        let size = (max - min).max(Vec3::splat(1e-6));
        for v in &mut self.vertices {
            let p = Vec3::from(v.position);
            let n = Vec3::from(v.normal).abs();
            // Normalize position into [0,1] per axis, then pick the two axes
            // perpendicular to the dominant normal axis as (u, v).
            let t = (p - min) / size;
            let uv = if n.x >= n.y && n.x >= n.z {
                Vec2::new(t.z, t.y)
            } else if n.y >= n.x && n.y >= n.z {
                Vec2::new(t.x, t.z)
            } else {
                Vec2::new(t.x, t.y)
            };
            v.uv = uv.to_array();
        }
    }
}

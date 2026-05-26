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

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
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

        Self { vertices, indices }
    }
}

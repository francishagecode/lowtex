// src/model.rs
//
// Mesh import. Loads glTF (.gltf/.glb) and OBJ (.obj) into our Vertex/index
// representation, then normalizes the result so *any* model is immediately
// paintable and framed by the camera:
//
//   - missing normals    → area-weighted smooth normals
//   - missing UVs         → box-projection fallback (generalized at G14)
//   - arbitrary scale/pos → recentered on origin, scaled to ~unit size
//
// v1 collapses all primitives/shapes into a single mesh (one texture set).
// Multi-material support is a later question (see DESIGN.md).

use std::path::Path;

use crate::mesh::{Mesh, Vertex};

/// Target largest-dimension size after normalization. The static camera sits
/// ~3.5 units out, so ~1.6 units frames the model without clipping.
const FRAME_SIZE: f32 = 1.6;

/// Load a mesh from `path`, dispatching on file extension. On failure returns a
/// human-readable error; the caller can fall back to the sample cube.
pub fn load(path: &str) -> Result<Mesh, String> {
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();

    let mut mesh = match ext.as_str() {
        "gltf" | "glb" => load_gltf(path)?,
        "obj" => load_obj(path)?,
        other => return Err(format!("unsupported mesh format: .{other}")),
    };

    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
        return Err(format!("'{path}' contained no triangles"));
    }

    finalize(&mut mesh);
    log::info!(
        "loaded '{}' — {} verts, {} tris",
        path,
        mesh.vertices.len(),
        mesh.indices.len() / 3
    );
    Ok(mesh)
}

/// Shared post-processing: fill in missing attributes and frame the model.
fn finalize(mesh: &mut Mesh) {
    if mesh.needs_normals {
        mesh.compute_smooth_normals();
    }
    if mesh.needs_uvs {
        mesh.box_project_uvs();
    }
    mesh.recenter_and_normalize(FRAME_SIZE);
}

fn load_gltf(path: &str) -> Result<Mesh, String> {
    let (doc, buffers, _images) =
        gltf::import(path).map_err(|e| format!("glTF import failed: {e}"))?;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut had_normals = true;
    let mut had_uvs = true;

    for mesh in doc.meshes() {
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                continue; // we only paint triangles
            }
            let reader = prim.reader(|b| Some(&buffers[b.index()]));

            let positions: Vec<[f32; 3]> = match reader.read_positions() {
                Some(p) => p.collect(),
                None => continue,
            };
            let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|n| n.collect());
            let uvs: Option<Vec<[f32; 2]>> =
                reader.read_tex_coords(0).map(|t| t.into_f32().collect());

            had_normals &= normals.is_some();
            had_uvs &= uvs.is_some();

            let base = vertices.len() as u32;
            for i in 0..positions.len() {
                vertices.push(Vertex {
                    position: positions[i],
                    normal: normals.as_ref().map_or([0.0, 1.0, 0.0], |n| n[i]),
                    uv: uvs.as_ref().map_or([0.0, 0.0], |u| u[i]),
                });
            }

            match reader.read_indices() {
                Some(read) => indices.extend(read.into_u32().map(|i| base + i)),
                // No index buffer: vertices are in sequential triangle order.
                None => indices.extend(base..base + positions.len() as u32),
            }
        }
    }

    Ok(Mesh {
        vertices,
        indices,
        needs_normals: !had_normals,
        needs_uvs: !had_uvs,
    })
}

fn load_obj(path: &str) -> Result<Mesh, String> {
    let load_opts = tobj::LoadOptions {
        triangulate: true,
        single_index: true,
        ..Default::default()
    };
    let (models, _materials) =
        tobj::load_obj(path, &load_opts).map_err(|e| format!("OBJ load failed: {e}"))?;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut had_normals = true;
    let mut had_uvs = true;

    for model in &models {
        let m = &model.mesh;
        let vert_count = m.positions.len() / 3;
        had_normals &= !m.normals.is_empty();
        had_uvs &= !m.texcoords.is_empty();

        let base = vertices.len() as u32;
        for i in 0..vert_count {
            let position = [
                m.positions[i * 3],
                m.positions[i * 3 + 1],
                m.positions[i * 3 + 2],
            ];
            let normal = if m.normals.is_empty() {
                [0.0, 1.0, 0.0]
            } else {
                [m.normals[i * 3], m.normals[i * 3 + 1], m.normals[i * 3 + 2]]
            };
            // OBJ V runs bottom-up; flip to match wgpu's top-down texture space.
            let uv = if m.texcoords.is_empty() {
                [0.0, 0.0]
            } else {
                [m.texcoords[i * 2], 1.0 - m.texcoords[i * 2 + 1]]
            };
            vertices.push(Vertex {
                position,
                normal,
                uv,
            });
        }
        indices.extend(m.indices.iter().map(|i| base + i));
    }

    Ok(Mesh {
        vertices,
        indices,
        needs_normals: !had_normals,
        needs_uvs: !had_uvs,
    })
}

// src/export.rs
//
// Opinionated export (principle #6). The headline feature is a *true indexed PNG*
// — a real paletted image with a PLTE chunk — which is what retro / PS1 pipelines
// and many pixel-art tools expect, versus a 32-bit RGBA PNG that merely happens to
// use few colors. When a palette is active the exported file is genuinely indexed
// (≤256 colors, 8-bit indices); otherwise it falls back to RGBA8.
//
// Engine "presets" mainly fix the suggested filename; the import flags that matter
// (point/nearest filtering, no mipmaps, correct color space) are settings on the
// engine side, not encodable in the PNG — see the per-preset note.

use std::fs::File;
use std::io::{BufWriter, Write};

use crate::mesh::Mesh;

/// Target pipeline for an export. Affects the suggested filename + guidance.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ExportPreset {
    Plain,
    Unity,
    Unreal,
    Godot,
    Gltf,
}

impl ExportPreset {
    pub const ALL: [ExportPreset; 5] = [
        ExportPreset::Plain,
        ExportPreset::Unity,
        ExportPreset::Unreal,
        ExportPreset::Godot,
        ExportPreset::Gltf,
    ];

    pub fn name(self) -> &'static str {
        match self {
            ExportPreset::Plain => "Plain",
            ExportPreset::Unity => "Unity",
            ExportPreset::Unreal => "Unreal",
            ExportPreset::Godot => "Godot",
            ExportPreset::Gltf => "glTF",
        }
    }

    /// Suggested file name for this preset.
    pub fn suggested_filename(self) -> &'static str {
        match self {
            // A `_albedo`/`_BaseColor` suffix is the convention each engine reads.
            ExportPreset::Plain => "texture.png",
            ExportPreset::Unity => "texture_albedo.png",
            ExportPreset::Unreal => "T_texture_BaseColor.png",
            ExportPreset::Godot => "texture_albedo.png",
            ExportPreset::Gltf => "texture_baseColor.png",
        }
    }

    /// One-line reminder of the engine-side import setting that can't live in a PNG.
    pub fn import_hint(self) -> &'static str {
        match self {
            ExportPreset::Plain => "Set the sampler to Point/Nearest, no mipmaps.",
            ExportPreset::Unity => "Texture: Filter Mode = Point (no filter), Mip Maps off.",
            ExportPreset::Unreal => "Texture: Filter = Nearest, Mip Gen = NoMipmaps, sRGB on.",
            ExportPreset::Godot => "Import: Filter = Nearest, Mipmaps off.",
            ExportPreset::Gltf => "Sampler magFilter/minFilter = NEAREST (9728).",
        }
    }
}

/// Write `rgba` (`width`×`height`, 8-bit) to a PNG. If `palette` (sRGB u8 triples,
/// ≤256 entries) is given and every pixel matches a palette color, the output is a
/// genuine *indexed* PNG; otherwise it's RGBA8.
pub fn export_png(
    path: &str,
    rgba: &[u8],
    width: u32,
    height: u32,
    palette: Option<&[[u8; 3]]>,
) -> Result<(), String> {
    match palette {
        Some(pal) if !pal.is_empty() && pal.len() <= 256 => {
            save_indexed_png(path, rgba, width, height, pal)
        }
        _ => image::save_buffer(path, rgba, width, height, image::ColorType::Rgba8)
            .map_err(|e| format!("failed to write PNG: {e}")),
    }
}

/// True paletted PNG: a PLTE chunk + one 8-bit index per pixel. Each pixel is
/// mapped to its nearest palette entry (exact when the texture is already quantized
/// to this palette).
fn save_indexed_png(
    path: &str,
    rgba: &[u8],
    width: u32,
    height: u32,
    palette: &[[u8; 3]],
) -> Result<(), String> {
    let mut indices = Vec::with_capacity((width * height) as usize);
    for px in rgba.chunks_exact(4) {
        indices.push(nearest_index(palette, [px[0], px[1], px[2]]) as u8);
    }

    let mut plte = Vec::with_capacity(palette.len() * 3);
    for c in palette {
        plte.extend_from_slice(c);
    }

    let file = File::create(path).map_err(|e| format!("failed to create {path}: {e}"))?;
    let mut enc = png::Encoder::new(BufWriter::new(file), width, height);
    enc.set_color(png::ColorType::Indexed);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_palette(plte);
    let mut writer = enc.write_header().map_err(|e| format!("png header: {e}"))?;
    writer
        .write_image_data(&indices)
        .map_err(|e| format!("png data: {e}"))?;
    Ok(())
}

/// Write `mesh` to a Wavefront OBJ at `path`: positions, texcoords, normals, and
/// triangle faces (`v/vt/vn`, 1-based).
///
/// This is how a painter gets the *unwrapped* geometry back out. The unwrap rebuilds
/// the UVs (and splits vertices to do it), and those UVs exist only inside lowtex —
/// the exported texture maps onto them and nothing else. So the mesh has to travel
/// with the texture, or the PNG is unusable in an engine.
///
/// Texcoord V is flipped back to OBJ's bottom-up convention (the loader flips it the
/// other way on import, `1.0 - v`), so a round trip is stable and the exported PNG
/// lines up the same way it does in the lowtex viewport.
pub fn export_obj(path: &str, mesh: &Mesh) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("failed to create {path}: {e}"))?;
    let mut w = BufWriter::new(file);
    let mut write = |line: std::fmt::Arguments| -> Result<(), String> {
        writeln!(w, "{line}").map_err(|e| format!("failed to write {path}: {e}"))
    };

    write(format_args!("# exported by lowtex"))?;
    write(format_args!(
        "# {} vertices, {} triangles",
        mesh.vertices.len(),
        mesh.indices.len() / 3
    ))?;
    // Undo lowtex's import recenter/rescale so the geometry leaves in the source's
    // original coordinates (an engine expects the model where it originally sat, not at
    // lowtex's camera-framing scale). Identity for the cube / procedural meshes.
    for v in &mesh.vertices {
        let p = mesh.source_transform.to_source(v.position);
        write(format_args!("v {} {} {}", p[0], p[1], p[2]))?;
    }
    for v in &mesh.vertices {
        write(format_args!("vt {} {}", v.uv[0], 1.0 - v.uv[1]))?;
    }
    for v in &mesh.vertices {
        write(format_args!(
            "vn {} {} {}",
            v.normal[0], v.normal[1], v.normal[2]
        ))?;
    }
    // One face per triangle. OBJ indices are 1-based; position/texcoord/normal share one
    // index because the (unwrapped) mesh is a flat, split-vertex mesh.
    let face = |w: &mut dyn std::io::Write, ti: usize| -> Result<(), String> {
        let t = &mesh.indices[ti * 3..ti * 3 + 3];
        let (a, b, c) = (t[0] + 1, t[1] + 1, t[2] + 1);
        writeln!(w, "f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}")
            .map_err(|e| format!("failed to write {path}: {e}"))
    };
    let tri_count = mesh.indices.len() / 3;
    if mesh.groups.is_empty() {
        for ti in 0..tri_count {
            face(&mut w, ti)?;
        }
    } else {
        // Re-emit the source's object split: `o <name>` then that group's faces. Clamp the
        // ranges defensively so a stale group can't index past the triangle list.
        for grp in &mesh.groups {
            writeln!(w, "o {}", grp.name).map_err(|e| format!("failed to write {path}: {e}"))?;
            let end = grp.start_tri.saturating_add(grp.tri_count).min(tri_count);
            for ti in grp.start_tri.min(tri_count)..end {
                face(&mut w, ti)?;
            }
        }
    }
    w.flush().map_err(|e| format!("failed to write {path}: {e}"))
}

fn nearest_index(palette: &[[u8; 3]], c: [u8; 3]) -> usize {
    let mut best = 0;
    let mut best_d = i32::MAX;
    for (i, p) in palette.iter().enumerate() {
        let d = (p[0] as i32 - c[0] as i32).pow(2)
            + (p[1] as i32 - c[1] as i32).pow(2)
            + (p[2] as i32 - c[2] as i32).pow(2);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_png_round_trips_to_a_paletted_image() {
        // A 2×2 of two palette colors → an indexed PNG that decodes back as Indexed
        // with the right dimensions.
        let palette = [[255u8, 0, 0], [0, 0, 255]];
        let rgba = [
            255, 0, 0, 255, 0, 0, 255, 255, // row 0: red, blue
            0, 0, 255, 255, 255, 0, 0, 255, // row 1: blue, red
        ];
        let path = std::env::temp_dir().join("lowtex_idx_test.png");
        let p = path.to_string_lossy();
        export_png(&p, &rgba, 2, 2, Some(&palette)).unwrap();

        let decoder = png::Decoder::new(File::open(&path).unwrap());
        let reader = decoder.read_info().unwrap();
        let info = reader.info();
        assert_eq!(info.color_type, png::ColorType::Indexed);
        assert_eq!((info.width, info.height), (2, 2));
        assert!(info.palette.is_some());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn exported_obj_reloads_with_matching_uvs() {
        // Unwrap a cube, export it, and re-import: vertex count and UVs must survive
        // the round trip (the V flip on the way out cancels the loader's flip back).
        use crate::mesh::Mesh;
        let unwrapped = crate::unwrap::auto_unwrap(&Mesh::cube(), &Default::default()).mesh;
        let path = std::env::temp_dir().join("lowtex_obj_test.obj");
        let p = path.to_string_lossy();
        export_obj(&p, &unwrapped).unwrap();

        let reloaded = crate::model::load(&p).unwrap();
        assert_eq!(reloaded.vertices.len(), unwrapped.vertices.len());
        assert_eq!(reloaded.indices.len(), unwrapped.indices.len());
        // model::load recenters/normalizes positions, so compare UVs (untouched).
        for (a, b) in reloaded.vertices.iter().zip(&unwrapped.vertices) {
            assert!((a.uv[0] - b.uv[0]).abs() < 1e-4 && (a.uv[1] - b.uv[1]).abs() < 1e-4);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn exported_obj_restores_source_coordinates() {
        // A model that sat off-origin at a large scale: lowtex recenters+rescales it on
        // import (recenter_and_normalize), but export must write the ORIGINAL coordinates
        // back — otherwise the geometry lands at the wrong place/size in an engine.
        use crate::mesh::{Mesh, Vertex};
        let source_positions = [
            [10.0f32, 20.0, 30.0],
            [14.0, 20.0, 30.0],
            [10.0, 28.0, 30.0],
        ];
        let mut mesh = Mesh {
            vertices: source_positions
                .iter()
                .map(|&p| Vertex {
                    position: p,
                    normal: [0.0, 0.0, 1.0],
                    uv: [0.0, 0.0],
                })
                .collect(),
            indices: vec![0, 1, 2],
            needs_normals: false,
            needs_uvs: false,
            source_transform: Default::default(),
            groups: Vec::new(),
        };
        mesh.recenter_and_normalize(1.6); // mutates positions, records the transform

        let path = std::env::temp_dir().join("lowtex_obj_src_test.obj");
        let p = path.to_string_lossy();
        export_obj(&p, &mesh).unwrap();

        // Parse the `v` lines straight out of the file and compare to the source.
        let text = std::fs::read_to_string(&path).unwrap();
        let verts: Vec<[f32; 3]> = text
            .lines()
            .filter_map(|l| l.strip_prefix("v "))
            .map(|rest| {
                let c: Vec<f32> = rest.split_whitespace().map(|s| s.parse().unwrap()).collect();
                [c[0], c[1], c[2]]
            })
            .collect();
        assert_eq!(verts.len(), source_positions.len());
        for (got, want) in verts.iter().zip(&source_positions) {
            for k in 0..3 {
                assert!(
                    (got[k] - want[k]).abs() < 1e-3,
                    "exported {got:?} should match source {want:?}"
                );
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn exported_obj_re_emits_object_groups() {
        // Two triangles in two named groups: export must write an `o <name>` header for
        // each, with that group's faces under it.
        use crate::mesh::{Mesh, MeshGroup, Vertex};
        let vert = |x: f32| Vertex {
            position: [x, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
        };
        let mesh = Mesh {
            vertices: (0..6).map(|i| vert(i as f32)).collect(),
            indices: vec![0, 1, 2, 3, 4, 5],
            needs_normals: false,
            needs_uvs: false,
            source_transform: Default::default(),
            groups: vec![
                MeshGroup { name: "alpha".into(), start_tri: 0, tri_count: 1 },
                MeshGroup { name: "beta".into(), start_tri: 1, tri_count: 1 },
            ],
        };
        let path = std::env::temp_dir().join("lowtex_obj_groups_test.obj");
        let p = path.to_string_lossy();
        export_obj(&p, &mesh).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();

        let objs: Vec<&str> = text.lines().filter(|l| l.starts_with("o ")).collect();
        assert_eq!(objs, ["o alpha", "o beta"]);
        // Each group's single face sits under its own `o` header, in order.
        let order: Vec<&str> = text
            .lines()
            .filter(|l| l.starts_with("o ") || l.starts_with("f "))
            .collect();
        assert_eq!(order.len(), 4); // 2 headers + 2 faces
        assert!(order[0] == "o alpha" && order[1].starts_with("f "));
        assert!(order[2] == "o beta" && order[3].starts_with("f "));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn nearest_index_picks_closest() {
        let pal = [[0u8, 0, 0], [255, 255, 255], [255, 0, 0]];
        assert_eq!(nearest_index(&pal, [250, 10, 10]), 2); // ~red
        assert_eq!(nearest_index(&pal, [10, 10, 10]), 0); // ~black
    }
}

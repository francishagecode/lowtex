// src/project.rs
//
// Project save/load (G24): the whole editing state as a `.lowtex` file so you can
// reopen exactly where you left off. The format is RON (human-readable structure)
// with each layer's pixel buffers base64'd so the file stays compact.
//
// This is a self-contained DTO that the renderer maps to/from its internal state.
// Keeping serde off the core types (Mesh/Layers/Texture) avoids deriving on the
// fast-moving render structs and keeps the on-disk format decoupled from them.
//
// We store the mesh geometry itself (not just a path): an unwrap or a procedurally
// modified mesh must round-trip, and a sample cube has no path.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde::{Deserialize, Serialize};

/// Bump when the on-disk layout changes incompatibly.
///
/// v2 (G28) added per-layer `effects`; v1 files load unchanged because the field
/// is `#[serde(default)]` (an empty stack) and the version is only rejected if it
/// is *newer* than this build understands.
///
/// v3 added the optional `texture_folder` the project was painted against; older
/// files load unchanged (the field is `#[serde(default)]` → `None`).
pub const FORMAT_VERSION: u32 = 3;

/// One per-layer effect, mirroring `effects::Effect` so serde stays off the core
/// type (same split as `BlendMode` ↔ the `blend` index). The renderer converts.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub enum EffectDoc {
    HueSatLight { hue: f32, sat: f32, light: f32 },
    BrightnessContrast { brightness: f32, contrast: f32 },
    Blur { radius: f32 },
    Warp { amount: f32, scale: f32 },
}

#[derive(Serialize, Deserialize)]
pub struct LayerDoc {
    pub name: String,
    /// Index into `layers::BlendMode::ALL`.
    pub blend: u8,
    pub visible: bool,
    pub opacity: f32,
    /// Base64 RGBA8, `tex_size`².
    pub color: String,
    pub mask: String,
    /// Non-destructive effect stack (G28). Absent in v1 files → empty.
    #[serde(default)]
    pub effects: Vec<EffectDoc>,
}

#[derive(Serialize, Deserialize)]
pub struct ProjectDoc {
    pub version: u32,
    pub tex_size: u32,
    pub active_layer: usize,
    pub palette: Vec<[f32; 3]>,
    pub quantize: bool,
    pub dither: bool,
    pub dither_strength: f32,
    // Mesh geometry (split or not — whatever's live).
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    pub layers: Vec<LayerDoc>,
    /// The texture folder the project was painted against, so reopening the file
    /// reopens the same brush browser. Absent in v1/v2 files → `None`.
    #[serde(default)]
    pub texture_folder: Option<String>,
}

impl ProjectDoc {
    pub fn save(&self, path: &str) -> Result<(), String> {
        let s = ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
            .map_err(|e| format!("serialize failed: {e}"))?;
        std::fs::write(path, s).map_err(|e| format!("write failed: {e}"))
    }

    pub fn load(path: &str) -> Result<ProjectDoc, String> {
        let s = std::fs::read_to_string(path).map_err(|e| format!("read failed: {e}"))?;
        let doc: ProjectDoc = ron::from_str(&s).map_err(|e| format!("parse failed: {e}"))?;
        if doc.version > FORMAT_VERSION {
            return Err(format!(
                "project is version {} but this build only understands {FORMAT_VERSION}",
                doc.version
            ));
        }
        Ok(doc)
    }
}

/// Base64-encode a pixel buffer for the document.
pub fn encode_pixels(pixels: &[u8]) -> String {
    STANDARD.encode(pixels)
}

/// Decode a base64 pixel buffer; errors if it isn't valid base64.
pub fn decode_pixels(s: &str) -> Result<Vec<u8>, String> {
    STANDARD
        .decode(s)
        .map_err(|e| format!("bad pixel data: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixels_round_trip_through_base64() {
        let px: Vec<u8> = (0..=255).collect();
        assert_eq!(decode_pixels(&encode_pixels(&px)).unwrap(), px);
    }

    #[test]
    fn doc_round_trips_through_ron() {
        let doc = ProjectDoc {
            version: FORMAT_VERSION,
            tex_size: 64,
            active_layer: 1,
            palette: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            quantize: true,
            dither: false,
            dither_strength: 0.1,
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            normals: vec![[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            uvs: vec![[0.0, 0.0], [1.0, 1.0]],
            indices: vec![0, 1, 0],
            texture_folder: Some("/tmp/textures".into()),
            layers: vec![LayerDoc {
                name: "Base".into(),
                blend: 0,
                visible: true,
                opacity: 0.8,
                color: encode_pixels(&[1, 2, 3, 4]),
                mask: encode_pixels(&[255, 255, 255, 255]),
                effects: vec![EffectDoc::Blur { radius: 2.5 }],
            }],
        };
        let path = std::env::temp_dir().join("lowtex_proj_test.lowtex");
        let p = path.to_string_lossy().to_string();
        doc.save(&p).unwrap();
        let back = ProjectDoc::load(&p).unwrap();
        assert_eq!(back.tex_size, 64);
        assert_eq!(back.active_layer, 1);
        assert_eq!(back.palette.len(), 2);
        assert_eq!(back.indices, vec![0, 1, 0]);
        assert_eq!(back.layers.len(), 1);
        assert_eq!(
            decode_pixels(&back.layers[0].color).unwrap(),
            vec![1, 2, 3, 4]
        );
        assert_eq!(
            back.layers[0].effects,
            vec![EffectDoc::Blur { radius: 2.5 }]
        );
        let _ = std::fs::remove_file(&path);
    }
}

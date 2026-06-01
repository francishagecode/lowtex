// src/preset.rs
//
// Smart palettes / preset looks (G21) — the shareable-ecosystem hook.
//
// A `Preset` is a *recipe*, not pixels: an ordered list of mesh-aware generator
// layers (each = a map source + Levels remap + color + blend) plus an optional
// palette. Because every layer is evaluated against the *currently loaded* mesh's
// baked maps when applied, the same preset produces a coherent result on any
// model — "Mossy Stone" sinks moss into whatever crevices the new geometry has.
//
// Stored as human-readable, hand-editable, diffable RON (versioned). Map sources
// and blend modes are written as names rather than indices so a shared `.lowpreset`
// survives reordering of the underlying enums.

use serde::{Deserialize, Serialize};

use crate::bake::{Levels, MapSource, NoiseMod};
use crate::layers::BlendMode;
use crate::noise::NoiseKind;

/// Bumped on any breaking change to the on-disk schema. v2 adds the optional
/// per-layer noise breakup; v1 files still load (the noise fields default off).
pub const PRESET_VERSION: u32 = 2;

/// One mesh-aware generator layer in a preset. Mirrors the arguments of
/// `Renderer::add_map_layer`, flattened for a clean RON representation.
#[derive(Clone, Serialize, Deserialize)]
pub struct PresetLayer {
    pub name: String,
    /// Map source name: "Cavities" | "Exposed" | "Edges" | "Creases" | "Surface".
    pub source: String,
    pub invert: bool,
    pub contrast: f32,
    pub strength: f32,
    pub color: [u8; 3],
    /// Blend mode name: "Normal" | "Multiply" | "Add" | "Screen".
    pub blend: String,
    /// Optional procedural-noise breakup (v2). `noise_kind` None = no noise; the
    /// other fields are ignored then. `#[serde(default)]` lets v1 files load.
    #[serde(default)]
    pub noise_kind: Option<String>,
    #[serde(default)]
    pub noise_scale: f32,
    #[serde(default)]
    pub noise_contrast: f32,
    #[serde(default)]
    pub noise_amount: f32,
}

impl PresetLayer {
    /// Capture a generator invocation as a recipe entry (used while recording).
    pub fn from_op(
        name: &str,
        src: MapSource,
        levels: Levels,
        color: [u8; 3],
        blend: BlendMode,
        noise: Option<NoiseMod>,
    ) -> Self {
        let (noise_kind, noise_scale, noise_contrast, noise_amount) = match noise {
            Some(n) => (
                Some(n.kind.name().to_string()),
                n.scale,
                n.contrast,
                n.amount,
            ),
            None => (None, 0.0, 0.0, 0.0),
        };
        Self {
            name: name.to_string(),
            source: src.name().to_string(),
            invert: levels.invert,
            contrast: levels.contrast,
            strength: levels.strength,
            color,
            blend: blend.name().to_string(),
            noise_kind,
            noise_scale,
            noise_contrast,
            noise_amount,
        }
    }

    /// Resolve back into the concrete generator arguments. Unknown names fall back
    /// to sensible defaults so a partially-understood file still does something
    /// reasonable rather than failing.
    pub fn to_op(
        &self,
    ) -> (
        String,
        MapSource,
        Levels,
        [u8; 3],
        BlendMode,
        Option<NoiseMod>,
    ) {
        let src = map_source_from_name(&self.source).unwrap_or(MapSource::Cavities);
        let blend = blend_from_name(&self.blend).unwrap_or(BlendMode::Normal);
        let levels = Levels {
            invert: self.invert,
            contrast: self.contrast,
            strength: self.strength,
        };
        let noise = self
            .noise_kind
            .as_deref()
            .and_then(noise_kind_from_name)
            .map(|kind| NoiseMod {
                kind,
                scale: self.noise_scale,
                contrast: self.noise_contrast,
                amount: self.noise_amount,
            });
        (self.name.clone(), src, levels, self.color, blend, noise)
    }
}

/// A complete, shareable look.
#[derive(Clone, Serialize, Deserialize)]
pub struct Preset {
    pub version: u32,
    pub name: String,
    /// Optional built-in palette to apply with the look (by name).
    pub palette: Option<String>,
    pub layers: Vec<PresetLayer>,
}

impl Preset {
    pub fn new(name: impl Into<String>, layers: Vec<PresetLayer>) -> Self {
        Self {
            version: PRESET_VERSION,
            name: name.into(),
            palette: None,
            layers,
        }
    }

    /// Serialize to pretty RON.
    pub fn to_ron(&self) -> Result<String, String> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
            .map_err(|e| format!("serialize preset: {e}"))
    }

    /// Parse from RON, rejecting a newer format version we can't understand.
    pub fn from_ron(s: &str) -> Result<Self, String> {
        let preset: Preset = ron::from_str(s).map_err(|e| format!("parse preset: {e}"))?;
        if preset.version > PRESET_VERSION {
            return Err(format!(
                "preset format v{} is newer than supported v{PRESET_VERSION} — update lowtex",
                preset.version
            ));
        }
        Ok(preset)
    }

    /// Write the preset to `path`.
    pub fn save(&self, path: &str) -> Result<(), String> {
        std::fs::write(path, self.to_ron()?).map_err(|e| format!("write '{path}': {e}"))
    }

    /// Read a preset from `path`.
    pub fn load(path: &str) -> Result<Self, String> {
        let s = std::fs::read_to_string(path).map_err(|e| format!("read '{path}': {e}"))?;
        Self::from_ron(&s)
    }
}

fn map_source_from_name(s: &str) -> Option<MapSource> {
    MapSource::ALL.iter().copied().find(|m| m.name() == s)
}

fn blend_from_name(s: &str) -> Option<BlendMode> {
    BlendMode::ALL.iter().copied().find(|b| b.name() == s)
}

fn noise_kind_from_name(s: &str) -> Option<NoiseKind> {
    NoiseKind::ALL.iter().copied().find(|k| k.name() == s)
}

/// The built-in shareable looks shipped with lowtex. Each is a pure recipe, so it
/// re-evaluates against whatever mesh is loaded.
pub fn builtins() -> Vec<Preset> {
    // Helper to keep the recipes terse — a plain generator layer, no noise.
    let layer =
        |name: &str, source: &str, color: [u8; 3], blend: &str, strength, contrast| PresetLayer {
            name: name.to_string(),
            source: source.to_string(),
            invert: false,
            contrast,
            strength,
            color,
            blend: blend.to_string(),
            noise_kind: None,
            noise_scale: 0.0,
            noise_contrast: 0.0,
            noise_amount: 0.0,
        };

    // Same, but broken up by procedural noise (Surface × noise = grunge).
    #[allow(clippy::too_many_arguments)]
    let noisy = |name: &str,
                 source: &str,
                 color: [u8; 3],
                 blend: &str,
                 strength,
                 contrast,
                 kind: &str,
                 scale,
                 noise_contrast,
                 amount| PresetLayer {
        name: name.to_string(),
        source: source.to_string(),
        invert: false,
        contrast,
        strength,
        color,
        blend: blend.to_string(),
        noise_kind: Some(kind.to_string()),
        noise_scale: scale,
        noise_contrast,
        noise_amount: amount,
    };

    vec![
        // Stone with moss settling into the crevices and lightly worn edges.
        Preset::new(
            "Mossy Stone",
            vec![
                layer("AO", "Cavities", [0, 0, 0], "Multiply", 0.75, 0.3),
                layer("Moss", "Cavities", [54, 84, 38], "Normal", 0.85, 0.45),
                layer("Edge light", "Edges", [188, 190, 170], "Normal", 0.5, 0.2),
            ],
        ),
        // Grimy metal: dark in the recesses, bright bare metal on the worn edges.
        Preset::new(
            "Worn Metal",
            vec![
                layer("AO", "Cavities", [0, 0, 0], "Multiply", 0.65, 0.35),
                layer("Grime", "Cavities", [38, 33, 28], "Normal", 0.6, 0.4),
                layer("Edge wear", "Edges", [205, 205, 215], "Normal", 0.75, 0.25),
            ],
        ),
        // Pale dust catching on the upward-facing surfaces of an old relic.
        Preset::new(
            "Dusty Relic",
            vec![
                layer("AO", "Cavities", [0, 0, 0], "Multiply", 0.6, 0.3),
                layer("Dust", "Exposed", [178, 168, 146], "Normal", 0.45, 0.15),
            ],
        ),
        // Iron eaten by rust: cellular Worley blotches of oxide across the whole
        // surface, darkened cavities, and bare metal still catching the edges.
        Preset::new(
            "Rusty Iron",
            vec![
                layer("AO", "Cavities", [0, 0, 0], "Multiply", 0.65, 0.35),
                noisy(
                    "Rust",
                    "Surface",
                    [120, 64, 38],
                    "Normal",
                    0.8,
                    0.0,
                    "Worley",
                    9.0,
                    0.45,
                    0.9,
                ),
                layer("Bare metal", "Edges", [188, 188, 198], "Normal", 0.55, 0.25),
            ],
        ),
    ]
}

/// Look up a built-in preset by (case-insensitive) name.
pub fn builtin(name: &str) -> Option<Preset> {
    builtins()
        .into_iter()
        .find(|p| p.name.eq_ignore_ascii_case(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_ron() {
        let p = builtin("Mossy Stone").unwrap();
        let s = p.to_ron().unwrap();
        let back = Preset::from_ron(&s).unwrap();
        assert_eq!(back.name, "Mossy Stone");
        assert_eq!(back.layers.len(), p.layers.len());
        assert_eq!(back.layers[1].source, "Cavities");
    }

    #[test]
    fn op_conversion_preserves_known_values() {
        let pl = PresetLayer::from_op(
            "Edge",
            MapSource::Edges,
            Levels {
                invert: true,
                contrast: 0.4,
                strength: 0.6,
            },
            [10, 20, 30],
            BlendMode::Multiply,
            None,
        );
        let (name, src, lv, col, bl, noise) = pl.to_op();
        assert_eq!(name, "Edge");
        assert!(matches!(src, MapSource::Edges));
        assert!(matches!(bl, BlendMode::Multiply));
        assert!(lv.invert && (lv.strength - 0.6).abs() < 1e-6);
        assert_eq!(col, [10, 20, 30]);
        assert!(noise.is_none());
    }

    #[test]
    fn op_conversion_round_trips_noise() {
        let pl = PresetLayer::from_op(
            "Rust",
            MapSource::Surface,
            Levels::amount(0.8),
            [120, 64, 38],
            BlendMode::Normal,
            Some(NoiseMod {
                kind: NoiseKind::Worley,
                scale: 9.0,
                contrast: 0.45,
                amount: 0.9,
            }),
        );
        let (_, src, _, _, _, noise) = pl.to_op();
        assert!(matches!(src, MapSource::Surface));
        let n = noise.expect("noise should round-trip");
        assert!(matches!(n.kind, NoiseKind::Worley));
        assert!((n.scale - 9.0).abs() < 1e-6 && (n.amount - 0.9).abs() < 1e-6);
    }

    #[test]
    fn v1_preset_without_noise_fields_still_loads() {
        // A v1 file predates the noise fields; serde(default) must fill them.
        let v1 = r#"(version:1,name:"Old",palette:None,layers:[(name:"AO",source:"Cavities",invert:false,contrast:0.3,strength:0.7,color:(0,0,0),blend:"Multiply")])"#;
        let p = Preset::from_ron(v1).expect("v1 preset should load under v2 reader");
        let (_, _, _, _, _, noise) = p.layers[0].to_op();
        assert!(noise.is_none());
    }

    #[test]
    fn rejects_newer_version() {
        let mut p = builtin("Worn Metal").unwrap();
        p.version = PRESET_VERSION + 1;
        let s = p.to_ron().unwrap();
        assert!(Preset::from_ron(&s).is_err());
    }

    #[test]
    fn unknown_names_fall_back() {
        let pl = PresetLayer {
            name: "x".into(),
            source: "Bogus".into(),
            invert: false,
            contrast: 0.0,
            strength: 1.0,
            color: [1, 2, 3],
            blend: "Nope".into(),
            noise_kind: Some("AlsoBogus".into()),
            noise_scale: 4.0,
            noise_contrast: 0.0,
            noise_amount: 1.0,
        };
        let (_, src, _, _, bl, noise) = pl.to_op();
        assert!(matches!(src, MapSource::Cavities));
        assert!(matches!(bl, BlendMode::Normal));
        assert!(noise.is_none(), "an unknown noise kind drops to no noise");
    }
}

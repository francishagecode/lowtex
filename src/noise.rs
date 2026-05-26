// src/noise.rs
//
// Procedural noise (G22) for the generator/mask system and as texture brushes.
// Sampled in *world space* (Vec3) so it doesn't break across UV seams, with the
// usual fBm controls (scale, octaves, persistence). All hash-based and
// deterministic — same input + seed → same value — so bakes are reproducible.
//
// Three kinds, each returning 0..1:
//   - Value   : smooth interpolated lattice hash — soft, cloudy.
//   - Perlin  : gradient noise — the classic organic look.
//   - Worley  : cellular (F1 distance to feature points) — cracks, scales, stone.
//
// This is a self-contained primitive library (G22). It's consumed by the
// generator/mask system (G20) — "edge wear = curvature × noise" — which is still
// being wired up; until then the public API is unused, hence the module allow.
#![allow(dead_code)]

use glam::Vec3;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NoiseKind {
    Value,
    Perlin,
    Worley,
}

impl NoiseKind {
    pub const ALL: [NoiseKind; 3] = [NoiseKind::Value, NoiseKind::Perlin, NoiseKind::Worley];

    pub fn name(self) -> &'static str {
        match self {
            NoiseKind::Value => "Value",
            NoiseKind::Perlin => "Perlin",
            NoiseKind::Worley => "Worley",
        }
    }
}

#[derive(Clone, Copy)]
pub struct NoiseParams {
    /// Spatial frequency: features are ~`1/scale` apart. Higher = finer.
    pub scale: f32,
    /// fBm octaves (1 = single frequency).
    pub octaves: u32,
    /// Amplitude falloff per octave (0..1).
    pub persistence: f32,
    pub seed: u32,
}

impl Default for NoiseParams {
    fn default() -> Self {
        Self {
            scale: 4.0,
            octaves: 4,
            persistence: 0.5,
            seed: 0,
        }
    }
}

/// fBm sum of `kind` at world position `p`, returned in 0..1. Octaves double in
/// frequency and shrink by `persistence`; the sum is normalized by its maximum
/// possible amplitude so the result stays in range.
pub fn sample(kind: NoiseKind, p: Vec3, params: &NoiseParams) -> f32 {
    let mut freq = params.scale;
    let mut amp = 1.0;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for o in 0..params.octaves.max(1) {
        let seed = params.seed.wrapping_add(o.wrapping_mul(0x9E3779B1));
        let v = match kind {
            NoiseKind::Value => value3(p * freq, seed),
            NoiseKind::Perlin => perlin3(p * freq, seed) * 0.5 + 0.5,
            NoiseKind::Worley => worley3(p * freq, seed),
        };
        sum += v * amp;
        norm += amp;
        freq *= 2.0;
        amp *= params.persistence.clamp(0.0, 1.0);
    }
    (sum / norm.max(1e-6)).clamp(0.0, 1.0)
}

// --- Hashing ---

/// Integer hash → u32. Three coords + seed mixed into a well-distributed value.
fn hash3(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x8DA6B343)
        ^ (y as u32).wrapping_mul(0xD8163841)
        ^ (z as u32).wrapping_mul(0xCB1AB31F)
        ^ seed.wrapping_mul(0x165667B1);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B3C6D);
    h ^= h >> 13;
    h = h.wrapping_mul(0x297A2D39);
    h ^= h >> 16;
    h
}

/// Hash → f32 in [0,1).
fn hash_unit(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    (hash3(x, y, z, seed) & 0xFFFFFF) as f32 / 0x1000000 as f32
}

/// Quintic smootherstep fade.
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// --- Value noise (0..1) ---

fn value3(p: Vec3, seed: u32) -> f32 {
    let xi = p.x.floor() as i32;
    let yi = p.y.floor() as i32;
    let zi = p.z.floor() as i32;
    let (fx, fy, fz) = (
        fade(p.x.fract_gl()),
        fade(p.y.fract_gl()),
        fade(p.z.fract_gl()),
    );

    let c = |dx, dy, dz| hash_unit(xi + dx, yi + dy, zi + dz, seed);
    let x00 = lerp(c(0, 0, 0), c(1, 0, 0), fx);
    let x10 = lerp(c(0, 1, 0), c(1, 1, 0), fx);
    let x01 = lerp(c(0, 0, 1), c(1, 0, 1), fx);
    let x11 = lerp(c(0, 1, 1), c(1, 1, 1), fx);
    let y0 = lerp(x00, x10, fy);
    let y1 = lerp(x01, x11, fy);
    lerp(y0, y1, fz)
}

// --- Perlin gradient noise (-1..1) ---

/// The 12 edge-midpoint gradient directions of a cube (Perlin's improved set).
const GRAD: [[f32; 3]; 12] = [
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0],
    [0.0, -1.0, 1.0],
    [0.0, 1.0, -1.0],
    [0.0, -1.0, -1.0],
];

fn grad_dot(x: i32, y: i32, z: i32, dx: f32, dy: f32, dz: f32, seed: u32) -> f32 {
    let g = GRAD[(hash3(x, y, z, seed) % 12) as usize];
    g[0] * dx + g[1] * dy + g[2] * dz
}

fn perlin3(p: Vec3, seed: u32) -> f32 {
    let xi = p.x.floor() as i32;
    let yi = p.y.floor() as i32;
    let zi = p.z.floor() as i32;
    let (rx, ry, rz) = (p.x.fract_gl(), p.y.fract_gl(), p.z.fract_gl());
    let (u, v, w) = (fade(rx), fade(ry), fade(rz));

    let g = |dx: i32, dy: i32, dz: i32| {
        grad_dot(
            xi + dx,
            yi + dy,
            zi + dz,
            rx - dx as f32,
            ry - dy as f32,
            rz - dz as f32,
            seed,
        )
    };
    let x00 = lerp(g(0, 0, 0), g(1, 0, 0), u);
    let x10 = lerp(g(0, 1, 0), g(1, 1, 0), u);
    let x01 = lerp(g(0, 0, 1), g(1, 0, 1), u);
    let x11 = lerp(g(0, 1, 1), g(1, 1, 1), u);
    let y0 = lerp(x00, x10, v);
    let y1 = lerp(x01, x11, v);
    // Perlin 3D output is within ~±1; clamp guards the corners.
    lerp(y0, y1, w).clamp(-1.0, 1.0)
}

// --- Worley / cellular noise (0..1, F1 distance) ---

fn worley3(p: Vec3, seed: u32) -> f32 {
    let xi = p.x.floor() as i32;
    let yi = p.y.floor() as i32;
    let zi = p.z.floor() as i32;
    let mut best = f32::INFINITY;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (cx, cy, cz) = (xi + dx, yi + dy, zi + dz);
                // One feature point per cell, jittered by a per-cell hash.
                let h = hash3(cx, cy, cz, seed);
                let fx = (h & 0xFF) as f32 / 255.0;
                let fy = ((h >> 8) & 0xFF) as f32 / 255.0;
                let fz = ((h >> 16) & 0xFF) as f32 / 255.0;
                let feat = Vec3::new(cx as f32 + fx, cy as f32 + fy, cz as f32 + fz);
                best = best.min((feat - p).length_squared());
            }
        }
    }
    // Distance to nearest feature ∈ ~[0, ~1.7]; normalize and clamp to 0..1.
    (best.sqrt()).clamp(0.0, 1.0)
}

/// `f32::fract` returns the signed fractional part; for noise we want the
/// always-positive fract matching `floor` (GLSL `fract`).
trait FractGl {
    fn fract_gl(self) -> f32;
}
impl FractGl for f32 {
    fn fract_gl(self) -> f32 {
        self - self.floor()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_samples(kind: NoiseKind) -> Vec<f32> {
        let params = NoiseParams::default();
        let mut out = Vec::new();
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    let p = Vec3::new(x as f32, y as f32, z as f32) * 0.13;
                    out.push(sample(kind, p, &params));
                }
            }
        }
        out
    }

    #[test]
    fn all_kinds_stay_in_unit_range() {
        for kind in NoiseKind::ALL {
            for v in grid_samples(kind) {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "{} out of range: {v}",
                    kind.name()
                );
            }
        }
    }

    #[test]
    fn noise_is_deterministic() {
        let p = Vec3::new(1.7, -2.3, 0.9);
        let params = NoiseParams::default();
        for kind in NoiseKind::ALL {
            assert_eq!(sample(kind, p, &params), sample(kind, p, &params));
        }
    }

    #[test]
    fn noise_is_non_uniform() {
        // A flat field would defeat the purpose; require real variation.
        for kind in NoiseKind::ALL {
            let s = grid_samples(kind);
            let mean = s.iter().sum::<f32>() / s.len() as f32;
            let var = s.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / s.len() as f32;
            assert!(var > 1e-4, "{} is nearly uniform (var={var})", kind.name());
        }
    }

    #[test]
    fn seed_changes_the_field() {
        let p = Vec3::new(0.5, 0.5, 0.5);
        let a = NoiseParams {
            seed: 1,
            ..NoiseParams::default()
        };
        let b = NoiseParams {
            seed: 2,
            ..NoiseParams::default()
        };
        // At least one kind must differ between seeds (Value/Perlin certainly do).
        let differs = NoiseKind::ALL
            .iter()
            .any(|&k| (sample(k, p, &a) - sample(k, p, &b)).abs() > 1e-6);
        assert!(differs, "seed had no effect");
    }
}

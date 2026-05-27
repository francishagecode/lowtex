// src/particle.rs
//
// The fluid particle brush (oil / water / blood): a click-hold-drag tool that, at
// each dab, emits a burst of particles at the cursor's surface point and lets them
// flow *downhill across the real geometry* under gravity, depositing a streak of
// "wetness" as they go. Unlike the AO/curvature generators this isn't a baked map —
// it rides the normal stroke lifecycle (begin/dab/segment/end) and paints color into
// the active layer, so it's non-destructive via the layer stack like any brush.
//
// The simulation is a surface walk. A particle has a world position on the mesh and
// a tangent velocity; each step it slides along gravity *projected onto the tangent
// plane*, then re-projects back onto the surface via a short BVH cast. Because all
// motion is world-space and UV is only read at deposit time, streaks cross UV seams
// correctly — a drip started on one face flows onto the next without the seam
// artifacts the island-bleed pass (G18) exists to hide.
//
// One "viscosity" knob derives the feel: thin fluids (water) run far in straight
// rivulets; thick ones (oil, blood) move slowly, wander less, and pool where the
// surface flattens or pinches into a crease.

use std::collections::HashMap;

use glam::{Vec2, Vec3};

use crate::bvh::{Bvh, Hit};
use crate::paint::Ray;

/// Which fluid the brush lays down. Each is a starting point for `FluidSpec`
/// (color + viscosity); the user can tweak from there.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FluidKind {
    Water,
    Oil,
    Blood,
}

impl FluidKind {
    pub const ALL: [FluidKind; 3] = [FluidKind::Water, FluidKind::Oil, FluidKind::Blood];

    pub fn name(self) -> &'static str {
        match self {
            FluidKind::Water => "Water",
            FluidKind::Oil => "Oil",
            FluidKind::Blood => "Blood",
        }
    }

    /// Starting (sRGB color, viscosity) for this fluid. Water is thin and runs far
    /// (low viscosity) and reads as a subtle darkening of the albedo; oil and blood
    /// are dark, thick, slow, and pool.
    pub fn defaults(self) -> ([f32; 3], f32) {
        match self {
            FluidKind::Water => ([0.06, 0.07, 0.09], 0.15),
            FluidKind::Oil => ([0.03, 0.03, 0.04], 0.80),
            FluidKind::Blood => ([0.34, 0.02, 0.03], 0.70),
        }
    }
}

/// Live fluid-brush settings, synced from the UI.
#[derive(Clone, Copy)]
pub struct FluidSpec {
    /// Deposited color, sRGB 0..1.
    pub color: [f32; 3],
    /// 0..1. Low = thin, fast, long runs (water); high = thick, slow, pools (oil/blood).
    pub viscosity: f32,
    /// Overall deposit strength 0..1 — the per-stroke coverage a streak tops out at.
    pub amount: f32,
    /// World-space gravity direction (need not be normalized). Default −Y.
    pub gravity: Vec3,
}

impl FluidSpec {
    /// A spec with the standard world-down gravity.
    pub fn with(color: [f32; 3], viscosity: f32, amount: f32) -> Self {
        Self {
            color,
            viscosity,
            amount,
            gravity: Vec3::NEG_Y,
        }
    }

    pub fn from_kind(kind: FluidKind) -> Self {
        let (color, viscosity) = kind.defaults();
        Self::with(color, viscosity, 0.85)
    }
}

impl Default for FluidSpec {
    fn default() -> Self {
        Self::from_kind(FluidKind::Water)
    }
}

/// Particles per dab. Modest so a burst is a sub-millisecond handful of BVH casts
/// and a drag stays interactive; density comes from overlap, not raw count.
const BURST: u32 = 48;

/// Where a burst starts: a surface point, its face normal, and the radius (in world
/// units, from the brush Size) of the disk particles spawn within.
#[derive(Clone, Copy)]
pub struct Emitter {
    pub origin: Vec3,
    pub normal: Vec3,
    pub spawn: f32,
}

/// Simulate one burst from `emit`, returning the texels it wet as `(texel_index,
/// density)` with density in 0..1. `size` is the square texture size; `scale` the
/// model's bounding-box diagonal (sets step/reach); `seed` makes the burst
/// deterministic (same inputs → same streak).
pub fn simulate_burst(
    bvh: &Bvh,
    emit: Emitter,
    spec: &FluidSpec,
    size: u32,
    scale: f32,
    seed: u32,
) -> Vec<(usize, f32)> {
    let gravity = spec.gravity.normalize_or_zero();
    if gravity == Vec3::ZERO || scale <= 0.0 || size == 0 {
        return Vec::new();
    }

    let visc = spec.viscosity.clamp(0.0, 1.0);
    // Viscosity → motion. Thin fluid: little drag, long life, runs far in straight
    // rivulets. Thick fluid: heavy drag, short life, wanders little, pools sooner.
    let damping = lerp(0.04, 0.45, visc);
    let max_steps = lerp(180.0, 50.0, visc) as u32; // streak length in steps
    let spread = lerp(0.5, 0.12, visc); // lateral wander
    let retention = lerp(0.992, 0.96, visc); // fluid-budget taper per step
    // Below this tangent-gravity magnitude the surface is ~flat/upward → pool & stop.
    let pool_threshold = lerp(0.04, 0.18, visc);

    let step = scale * 0.01; // world distance moved per integration step
    let bias = scale * 1e-3; // lift off the surface before re-casting
    let max_drop = scale * 0.08; // farthest a reproject/fall cast may land

    let amount = spec.amount.clamp(0.0, 1.0);
    let mut deposit: HashMap<usize, f32> = HashMap::new();
    let (tan, bitan) = basis(emit.normal);

    // A drip starts roughly brush-sized but not identically so — `head` is the spawn
    // radius scaled down a random amount per burst, so successive dabs along a drag
    // don't stamp the same width. `body` is the rivulet thickness in texels, capped so
    // a fat brush stays interactive: the head reads wide because particles spread across
    // `head`, and the tail tapers to a point because rim particles run short (see `len`)
    // and each rivulet thins along its run.
    let mut brng = Rng::new(seed.wrapping_add(0xB5297A4D));
    let head = emit.spawn * lerp(0.6, 1.0, brng.next());
    let body = (head / scale * size as f32).min(2.0);

    for pi in 0..BURST {
        let mut rng = Rng::new(seed ^ pi.wrapping_mul(0x9E3779B1).wrapping_add(1));

        // Spawn within the head disk in the tangent plane, then snap onto the surface.
        // `edge` (0 at the centre, 1 at the rim) shapes the streak: rim particles run
        // short, so the wet band narrows from a brush-wide head to a pointed tail.
        let edge = rng.next().sqrt();
        let ja = rng.next() * std::f32::consts::TAU;
        let jr = edge * head;
        let mut p = emit.origin + tan * (jr * ja.cos()) + bitan * (jr * ja.sin());
        let mut n = emit.normal;
        // Random length with the brush size as the start: centre runs the full streak,
        // the rim ~40%, each scaled by a further 0.55–1.0 so no two drips match.
        let len = (max_steps as f32 * lerp(1.0, 0.4, edge) * lerp(0.55, 1.0, rng.next()))
            .max(1.0) as u32;
        if let Some(h) = reproject(bvh, p, n, gravity, bias, max_drop) {
            p = h.pos;
            n = h.normal;
            splat(&mut deposit, h.uv, size, amount, body);
        }

        let mut vel = Vec3::ZERO;
        let mut fluid = 1.0f32;
        for i in 0..len {
            // Gravity projected onto the tangent plane is the downhill direction.
            let gt = gravity - n * gravity.dot(n);
            let gmag = gt.length();
            if gmag < pool_threshold {
                // Flat or upward surface: the fluid pools here and dumps its budget.
                let uv = uv_of(p, n, bvh, gravity, bias, max_drop);
                splat(&mut deposit, uv, size, amount * fluid, body);
                break;
            }
            let down = gt / gmag;
            // Wander perpendicular to flow, in the tangent plane — natural rivulets.
            let side = n.cross(down).normalize_or_zero();
            vel = vel * (1.0 - damping) + (down + side * ((rng.next() - 0.5) * spread)) * step;
            p += vel;

            let Some(h) = reproject(bvh, p, n, gravity, bias, max_drop) else {
                break; // dripped off into empty space
            };
            p = h.pos;
            n = h.normal;
            // The rivulet starts fat (`body`) and tapers to a thin tip along its run.
            let r = lerp(body, 0.5, i as f32 / len as f32);
            splat(&mut deposit, h.uv, size, amount * fluid, r);

            fluid *= retention;
            if fluid < 0.02 {
                break;
            }
        }
    }

    deposit.into_iter().collect()
}

/// Re-attach a particle to the surface after a tangential step. First cast straight
/// down the normal (stay glued to a curving face); if that misses or lands too far,
/// let the particle fall along gravity onto whatever surface is below (so a drip
/// wraps over an edge onto the face beneath). `None` if it left the mesh entirely.
fn reproject(bvh: &Bvh, p: Vec3, n: Vec3, gravity: Vec3, bias: f32, max_drop: f32) -> Option<Hit> {
    let origin = p + n * bias;
    for dir in [-n, gravity] {
        if let Some(h) = bvh.pick(&Ray {
            origin,
            direction: dir,
        }) {
            if (h.pos - origin).length() <= max_drop {
                return Some(h);
            }
        }
    }
    None
}

/// The UV of the surface point under `p` (for the pooling deposit); falls back to a
/// projection of `p` if the cast somehow misses.
fn uv_of(p: Vec3, n: Vec3, bvh: &Bvh, gravity: Vec3, bias: f32, max_drop: f32) -> Vec2 {
    reproject(bvh, p, n, gravity, bias, max_drop).map_or(Vec2::ZERO, |h| h.uv)
}

/// Accumulate `density` into the texels within `radius` (in texels) of where `uv`
/// lands, keeping the max (so re-tracing a path within one burst doesn't over-darken —
/// the same discipline the brush uses). A radius under ~¾ of a texel takes the fast
/// path and deposits the single texel the UV falls in.
fn splat(map: &mut HashMap<usize, f32>, uv: Vec2, size: u32, density: f32, radius: f32) {
    let last = (size - 1) as f32;
    let cx = (uv.x * size as f32).clamp(0.0, last);
    let cy = (uv.y * size as f32).clamp(0.0, last);
    let d = density.clamp(0.0, 1.0);
    if radius < 0.75 {
        let i = (cy as u32 * size + cx as u32) as usize;
        let e = map.entry(i).or_insert(0.0);
        *e = e.max(d);
        return;
    }
    let ri = radius.ceil() as i32;
    let r2 = radius * radius;
    let (icx, icy) = (cx as i32, cy as i32);
    for dy in -ri..=ri {
        let y = icy + dy;
        if y < 0 || y >= size as i32 {
            continue;
        }
        for dx in -ri..=ri {
            let x = icx + dx;
            if x < 0 || x >= size as i32 || (dx * dx + dy * dy) as f32 > r2 {
                continue;
            }
            let i = (y as u32 * size + x as u32) as usize;
            let e = map.entry(i).or_insert(0.0);
            *e = e.max(d);
        }
    }
}

/// An orthonormal tangent/bitangent for a unit normal (same construction as the bake).
fn basis(n: Vec3) -> (Vec3, Vec3) {
    let up = if n.y.abs() < 0.99 { Vec3::Y } else { Vec3::X };
    let t = up.cross(n).normalize_or_zero();
    (t, n.cross(t))
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// A tiny deterministic xorshift32, so a burst (and therefore a saved/replayed
/// stroke) is reproducible.
struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self {
        Rng(seed | 1) // a zero state would stay zero
    }

    /// Next value in [0, 1).
    fn next(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x & 0x00FF_FFFF) as f32 / 0x0100_0000 as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{Mesh, Vertex};

    /// A unit quad in a plane, two triangles, with UVs that put `+along_v` at the top
    /// of the texture (V down). `face` is the outward normal axis.
    fn quad(normal: Vec3) -> Mesh {
        // Build a quad spanning [-0.5, 0.5]² in the two axes perpendicular to `normal`.
        let (a, b) = super::basis(normal);
        let corner = |su: f32, sv: f32| (a * su + b * sv) * 0.5;
        let pos = [
            corner(-1.0, -1.0),
            corner(1.0, -1.0),
            corner(1.0, 1.0),
            corner(-1.0, 1.0),
        ];
        // UV: map the `a` axis to U and the `b` axis to V, V flipped (top = high b).
        let uv = [
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 0.0),
        ];
        let vertices = (0..4)
            .map(|i| Vertex {
                position: pos[i].to_array(),
                normal: normal.to_array(),
                uv: uv[i].to_array(),
            })
            .collect();
        Mesh {
            vertices,
            indices: vec![0, 1, 2, 0, 2, 3],
            needs_normals: false,
            needs_uvs: false,
        }
    }

    fn rows(deposits: &[(usize, f32)], size: u32) -> (u32, u32) {
        let ys: Vec<u32> = deposits.iter().map(|(i, _)| *i as u32 / size).collect();
        (*ys.iter().min().unwrap(), *ys.iter().max().unwrap())
    }

    #[test]
    fn flows_downhill_on_a_vertical_wall() {
        // A wall facing +Z; gravity −Y. A burst near the top must stream downward
        // (V increases) and never climb above where it started.
        let mesh = quad(Vec3::Z);
        let bvh = Bvh::build(&mesh);
        let size = 64u32;
        // Emit near the top of the wall (world +Y → texture V≈0).
        let emit = Emitter {
            origin: Vec3::new(0.0, 0.4, 0.0),
            normal: Vec3::Z,
            spawn: 0.01,
        };
        let spec = FluidSpec::with([0.0, 0.0, 0.0], 0.15, 1.0);
        let d = simulate_burst(&bvh, emit, &spec, size, 1.414, 7);
        assert!(!d.is_empty(), "burst deposited nothing");
        let (min_y, max_y) = rows(&d, size);
        // Origin V ≈ (0.5 − 0.4) = 0.1 → row ~6.
        assert!(min_y <= 12, "streak started too low: {min_y}");
        assert!(max_y > 30, "streak didn't run down the wall: {max_y}");
    }

    #[test]
    fn pools_on_a_flat_top() {
        // A horizontal face (normal +Y) under −Y gravity: tangent gravity ≈ 0, so the
        // fluid pools at the cursor instead of streaking across the surface.
        let mesh = quad(Vec3::Y);
        let bvh = Bvh::build(&mesh);
        let size = 64u32;
        let spec = FluidSpec::with([0.0, 0.0, 0.0], 0.7, 1.0);
        let emit = Emitter {
            origin: Vec3::ZERO,
            normal: Vec3::Y,
            spawn: 0.01,
        };
        let d = simulate_burst(&bvh, emit, &spec, size, 1.414, 3);
        assert!(!d.is_empty());
        // Every wet texel sits within a few of the centre (≈ row/col 32).
        for (i, _) in &d {
            let (x, y) = (*i as u32 % size, *i as u32 / size);
            assert!(x.abs_diff(32) <= 4 && y.abs_diff(32) <= 4, "spread off the puddle: ({x},{y})");
        }
    }

    #[test]
    fn is_deterministic_and_bounded() {
        let mesh = quad(Vec3::Z);
        let bvh = Bvh::build(&mesh);
        let size = 64u32;
        let spec = FluidSpec::with([0.0, 0.0, 0.0], 0.2, 0.9);
        let emit = Emitter {
            origin: Vec3::new(0.0, 0.4, 0.0),
            normal: Vec3::Z,
            spawn: 0.01,
        };
        let run = || {
            let mut v = simulate_burst(&bvh, emit, &spec, size, 1.414, 42);
            v.sort_by_key(|(i, _)| *i);
            v
        };
        let a = run();
        let b = run();
        assert_eq!(a.len(), b.len(), "same seed gave a different texel count");
        for ((ia, da), (ib, db)) in a.iter().zip(b.iter()) {
            assert_eq!(ia, ib);
            assert!((da - db).abs() < 1e-6);
        }
        // Densities stay in (0, amount]; indices stay in range.
        for (i, d) in &a {
            assert!(*i < (size * size) as usize);
            assert!(*d > 0.0 && *d <= 0.9 + 1e-6, "density out of range: {d}");
        }
    }
}

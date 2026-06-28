// src/unwrap.rs
//
// UV unwrapping (Phase 4). Most downloaded / hand-modeled low-poly assets have
// bad or no UVs; lowtex unwraps them in a PSX-friendly way.
//
// One auto-unwrap, built for "it just works" over tight packing — the user paints
// at very low resolution, so there's headroom to trade atlas space for correctness:
//
//   1. Connectivity charts (no overlaps). Triangles are welded by position and
//      region-grown into charts along shared edges, staying within an angle cone of
//      the seed normal. Because charts are *connected* and span < 2·cone < 180°, two
//      separate parts of the mesh can never share UV space and a chart's planar
//      projection can't fold onto itself. This kills the old normal-only clustering
//      bug where parallel-but-separate faces stacked in the atlas.
//
//   2. Constant world-space texel density. Each chart is projected onto its
//      area-weighted average normal (an orthonormal basis, so the 2D coords are in
//      world units), then *every* chart is scaled by one global "texels per world
//      unit" D. One world unit is D texels everywhere → the same physical pixel size
//      across the whole surface, regardless of which chart a face landed in.
//
//   3. Derived atlas size. Charts are packed in pixels at scale D and the atlas
//      resolution is rounded up to the next power of two to fit them. Denser meshes
//      (or a higher `Density`) just produce a bigger texture — the texel size stays
//      put. The caller resizes its paint layers to `UnwrapResult::atlas_size`.
//
//   4. Texel snapping (`snap_texels`, on by default). A free-form unwrap puts edges
//      at fractional, off-axis UVs, so at PSX resolutions a face edge cuts diagonally
//      across texels and reads as jaggy/blurry — and tighter packing can't fix it.
//      So each chart is first *rectified* (rotated so its longest edge is axis-
//      aligned), then every vertex is rounded to a whole texel. Charts already pack at
//      integer pixel offsets, so a snapped vertex lands exactly on a texel corner and
//      face edges coincide with the grid. The trade is blockier non-rectangular faces
//      and a little wasted atlas — the density-over-packing call this unwrap makes.
//
// Every unwrap splits vertices (3 per triangle): a vertex shared across faces with
// different projections needs different UVs, so unwrapping rebuilds the vertex list
// as 3·triangle_count flat vertices. Output is always a fresh `Mesh` with
// `needs_uvs = false`.

use std::collections::HashMap;

use glam::{Mat2, Vec2, Vec3};

use crate::mesh::{Mesh, Vertex};

/// Coarse texel-density knob for the UI. The constant-density invariant holds at
/// every setting — the multiplier only scales the absolute texels-per-world-unit
/// (and therefore the derived atlas size).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Density {
    Low,
    #[default]
    Medium,
    High,
}

impl Density {
    pub const ALL: [Density; 3] = [Density::Low, Density::Medium, Density::High];

    pub fn name(self) -> &'static str {
        match self {
            Density::Low => "Low",
            Density::Medium => "Medium",
            Density::High => "High",
        }
    }

    fn multiplier(self) -> f32 {
        match self {
            Density::Low => 0.5,
            Density::Medium => 1.0,
            Density::High => 2.0,
        }
    }
}

/// Tunables for `auto_unwrap`. `Default` is the sane GUI configuration.
#[derive(Clone, Copy)]
pub struct UnwrapOptions {
    pub density: Density,
    /// Chart growth cone, in degrees, measured from the seed normal. Kept < 90° so a
    /// chart can't fold over its own projection. Tighter → flatter charts (more
    /// uniform density) at the cost of more charts / a bigger atlas.
    pub angle_cone_deg: f32,
    /// Empty pixels reserved around each chart so nearest-neighbour sampling and the
    /// island-bleed dilate can't cross a seam.
    pub gutter_px: u32,
    /// Hard upper bound on the atlas (the renderer passes its GPU max texture dim).
    pub max_atlas: u32,
    /// The `Medium` density aims the atlas near this size for the current mesh.
    pub target_atlas_px: u32,
    /// Snap charts to the texel grid: rectify each chart so its longest edge is
    /// axis-aligned, then round every vertex to a whole texel. Face edges then land
    /// on texel boundaries instead of cutting diagonally across texels — the crisp,
    /// jaggy-free look you want at PSX resolutions. Costs some atlas (charts round up
    /// to whole texels) and makes non-rectangular faces blocky. See `snap_charts`.
    pub snap_texels: bool,
    /// Stack congruent charts on top of each other in the atlas instead of packing each
    /// into its own slot. Any two charts that are the same shape — at any position or
    /// orientation, including mirror images — collapse onto one shared region so
    /// corresponding surface points land on the same texels (paint one, paint all).
    /// Identical parts then cost the atlas nothing, which also lets the density fill the
    /// texture with fewer slots (slightly sharper). Off by default; matching relies on
    /// the rectify normalization, which is forced on whenever this is set. See
    /// `group_congruent_charts`.
    pub overlap_identical: bool,
    /// Exact texels-per-world-unit to unwrap at, if the caller wants to pin the
    /// density numerically (e.g. "128 texels per meter") rather than let `density`
    /// pick it. When `Some`, the atlas is sized to *hold* the charts at this density
    /// (rounded up to a power of two) instead of being filled — so the achieved
    /// `density_d` equals this value, except when it would overflow `max_atlas`, in
    /// which case it's trimmed and `clamped` is set. `None` uses the preset path.
    pub target_density: Option<f32>,
}

impl Default for UnwrapOptions {
    fn default() -> Self {
        Self {
            density: Density::Medium,
            angle_cone_deg: 40.0,
            // Breathing room between charts. Snapped charts pack to tight integer
            // rectangles, and display-time edge bleed dilates several px into this
            // margin, so keep enough gutter that one island's bleed can't reach its
            // neighbour at the seam.
            gutter_px: 4,
            max_atlas: 8192,
            target_atlas_px: 128,
            snap_texels: true,
            overlap_identical: false,
            target_density: None,
        }
    }
}

/// The product of an unwrap: the re-UV'd mesh plus the atlas size the caller should
/// resize its paint layers to.
pub struct UnwrapResult {
    /// Split-vertex mesh, `needs_uvs = false`, all UVs in `[0,1]`.
    pub mesh: Mesh,
    /// Square, power-of-two atlas resolution that holds the packed charts.
    pub atlas_size: u32,
    /// `true` if density was reduced to keep the atlas within `max_atlas`.
    pub clamped: bool,
    /// The final texels-per-world-unit actually used (after any clamp).
    pub density_d: f32,
}

/// Unwrap `mesh` into connectivity-based charts at a constant world-space texel
/// density, deriving the atlas size from `opts.density`.
pub fn auto_unwrap(mesh: &Mesh, opts: &UnwrapOptions) -> UnwrapResult {
    let mut result = unwrap_impl(mesh, opts).0;
    // Unwrap rebuilds UVs and splits vertices but leaves positions in place, so the
    // source import transform carries over unchanged — keep it for export.
    result.mesh.source_transform = mesh.source_transform;
    result
}

/// The geometric normal of a triangle (zero for a degenerate triangle).
fn face_normal(p: [Vec3; 3]) -> Vec3 {
    (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero()
}

/// Twice the area vector's length → triangle area (0 for degenerate).
fn tri_area(p: [Vec3; 3]) -> f32 {
    0.5 * (p[1] - p[0]).cross(p[2] - p[0]).length()
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

/// An orthonormal (tangent, bitangent) spanning the plane perpendicular to `n`.
fn planar_basis(n: Vec3) -> (Vec3, Vec3) {
    let up = if n.y.abs() < 0.99 { Vec3::Y } else { Vec3::X };
    let t = up.cross(n).normalize_or_zero();
    let b = n.cross(t);
    (t, b)
}

/// For each chart, the rotation — returned as `(sinθ, cosθ)` — that, applied as a
/// `-θ` rotation, gives the chart its *minimum-area* axis-aligned bounding box. By a
/// standard result the min-area box of a polygon has a side collinear with one of its
/// edges, so we test every edge direction and keep the one whose box is smallest.
/// This squares a face to its own sides rather than its diagonal: a quad split into
/// two triangles has the diagonal as its longest edge, so an "align the longest edge"
/// rule would rotate the whole face 45° into a diamond — twice the box area, and it
/// no longer tiles or snaps cleanly. A chart with no measurable edge keeps the
/// identity. `proj` holds each triangle's three projected UVs.
fn chart_rectify(proj: &[[Vec2; 3]], chart_of: &[usize], num_charts: usize) -> Vec<(f32, f32)> {
    // Group triangles by chart so each chart's box is measured over all its points.
    let mut tris_by_chart: Vec<Vec<usize>> = vec![Vec::new(); num_charts];
    for (ti, _) in proj.iter().enumerate() {
        tris_by_chart[chart_of[ti]].push(ti);
    }

    let mut rot = vec![(0.0f32, 1.0f32); num_charts]; // identity: θ = 0
    for (c, tris) in tris_by_chart.iter().enumerate() {
        let mut best_area = f32::INFINITY;
        for &ti in tris {
            for e in 0..3 {
                let d = proj[ti][(e + 1) % 3] - proj[ti][e];
                let len = d.length();
                if len < 1e-9 {
                    continue;
                }
                let (sin, cos) = (d.y / len, d.x / len);
                // Area of the chart's box when this edge is made axis-aligned.
                let (mut mn, mut mx) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
                for &tj in tris {
                    for p in &proj[tj] {
                        let r = Vec2::new(p.x * cos + p.y * sin, -p.x * sin + p.y * cos);
                        mn = mn.min(r);
                        mx = mx.max(r);
                    }
                }
                let area = (mx.x - mn.x) * (mx.y - mn.y);
                if area < best_area {
                    best_area = area;
                    rot[c] = (sin, cos);
                }
            }
        }
    }
    rot
}

/// Build a split-vertex mesh from per-triangle (positions, per-vertex normals, uvs).
/// Normals are carried per vertex (not a single face normal) so the unwrap preserves
/// the source's smooth shading — a flat face normal would export a faceted model.
fn build_split(tris: &[([Vec3; 3], [Vec3; 3], [Vec2; 3])]) -> Mesh {
    let mut vertices = Vec::with_capacity(tris.len() * 3);
    for (p, n, uv) in tris {
        for i in 0..3 {
            vertices.push(Vertex {
                position: p[i].to_array(),
                normal: n[i].to_array(),
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
        // auto_unwrap copies the source mesh's transform onto the result; unwrap leaves
        // positions untouched, so the import transform still maps them back to source coords.
        source_transform: crate::mesh::SourceTransform::IDENTITY,
    }
}

/// Weld vertices by quantized position → a representative id per input vertex. This
/// recovers topology on meshes that are already split (e.g. a prior unwrap), which
/// is what lets connectivity (not normals) drive chart membership.
fn weld_positions(mesh: &Mesh, eps: f32) -> Vec<u32> {
    let inv = 1.0 / eps as f64;
    let mut map: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut weld = Vec::with_capacity(mesh.vertices.len());
    let mut next = 0u32;
    for v in &mesh.vertices {
        let key = (
            (v.position[0] as f64 * inv).round() as i64,
            (v.position[1] as f64 * inv).round() as i64,
            (v.position[2] as f64 * inv).round() as i64,
        );
        let id = *map.entry(key).or_insert_with(|| {
            let id = next;
            next += 1;
            id
        });
        weld.push(id);
    }
    weld
}

/// Per-triangle adjacency: two triangles are neighbours if they share a welded edge.
fn build_adjacency(mesh: &Mesh, weld: &[u32]) -> Vec<Vec<usize>> {
    let tri_count = mesh.indices.len() / 3;
    let mut edges: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (ti, t) in mesh.indices.chunks_exact(3).enumerate() {
        let w = [
            weld[t[0] as usize],
            weld[t[1] as usize],
            weld[t[2] as usize],
        ];
        for k in 0..3 {
            let (a, b) = (w[k], w[(k + 1) % 3]);
            let key = if a <= b { (a, b) } else { (b, a) };
            edges.entry(key).or_default().push(ti);
        }
    }
    let mut adj = vec![Vec::new(); tri_count];
    for tris in edges.values() {
        for i in 0..tris.len() {
            for j in (i + 1)..tris.len() {
                adj[tris[i]].push(tris[j]);
                adj[tris[j]].push(tris[i]);
            }
        }
    }
    adj
}

/// Region-grow charts by flood fill. A triangle joins a chart only if its normal is
/// within `cos_cone` of the chart's *seed* normal — frozen, so the chart spans
/// < 2·cone and the projection onto its average normal stays injective. Returns the
/// chart id per triangle and each chart's area-weighted average normal.
fn grow_charts(
    normals: &[Vec3],
    areas: &[f32],
    adj: &[Vec<usize>],
    cos_cone: f32,
) -> (Vec<usize>, Vec<Vec3>) {
    let tri_count = normals.len();
    let mut chart_of = vec![usize::MAX; tri_count];
    let mut chart_normal: Vec<Vec3> = Vec::new();
    for seed in 0..tri_count {
        if chart_of[seed] != usize::MAX {
            continue;
        }
        let c = chart_normal.len();
        let seed_n = normals[seed];
        let valid_seed = seed_n.length_squared() > 1e-12;
        chart_of[seed] = c;
        let mut acc = seed_n * areas[seed];
        let mut stack = vec![seed];
        while let Some(t) = stack.pop() {
            for &nb in &adj[t] {
                if chart_of[nb] != usize::MAX {
                    continue;
                }
                let n = normals[nb];
                // Zero-area neighbours carry no orientation — let them ride along.
                let degenerate = n.length_squared() < 1e-12;
                let joins = degenerate || (valid_seed && seed_n.dot(n) >= cos_cone);
                if joins {
                    chart_of[nb] = c;
                    acc += n * areas[nb];
                    stack.push(nb);
                }
            }
        }
        let avg = if acc.length_squared() > 1e-20 {
            acc.normalize()
        } else {
            seed_n
        };
        chart_normal.push(avg);
    }
    (chart_of, chart_normal)
}

/// Bottom-left **skyline** packing of chart *pixel* footprints (chart size · `d`, plus
/// a gutter on every side). Returns each chart's pixel offset and the used extent.
///
/// This replaces a tallest-first shelf packer. A shelf is as tall as its tallest box,
/// so every shorter box on the shelf wastes the gap above it; the skyline instead lets
/// the next box drop into that gap, packing heterogeneous charts noticeably tighter.
/// Because `pack_pixels` rescales density to *fill* the derived atlas, that tighter
/// layout doesn't shrink the texture — it buys a higher texels-per-world-unit `d` at
/// the same atlas size (sharper paint at PSX resolutions).
///
/// Footprints are whole pixels and offsets land on integer skyline edges, so texel
/// snapping still puts every vertex on a texel corner. Charts never overlap: each box
/// is placed strictly above the current skyline over its span.
fn skyline_pack(csize: &[Vec2], d: f32, gutter_px: u32) -> (Vec<Vec2>, f32, f32) {
    let n = csize.len();
    let g = gutter_px as f32;
    let psize: Vec<Vec2> = csize
        .iter()
        .map(|s| Vec2::new((s.x * d).ceil() + 2.0 * g, (s.y * d).ceil() + 2.0 * g))
        .collect();

    // Tallest-first placement (the classic skyline heuristic: tall boxes first leave a
    // flatter skyline for the short ones to nestle into).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| psize[b].y.total_cmp(&psize[a].y));

    let total_area: f32 = psize.iter().map(|s| s.x * s.y).sum();
    let widest = psize.iter().map(|s| s.x).fold(0.0_f32, f32::max);
    // Aim for a *square* layout (width ≈ √area) so the square power-of-two atlas
    // wastes little — never narrower than the widest chart.
    let bound_w = total_area.sqrt().max(widest);

    // Skyline: a contiguous, left-to-right list of segments `(x, width, top_y)`,
    // starting flat at y = 0 across the whole bound.
    let mut sky: Vec<(f32, f32, f32)> = vec![(0.0, bound_w.max(1.0), 0.0)];

    let mut offsets = vec![Vec2::ZERO; n];
    let (mut max_x, mut max_y) = (0.0f32, 0.0f32);
    for &i in &order {
        let s = psize[i];
        let (px, py) = skyline_fit(&sky, s.x, bound_w);
        offsets[i] = Vec2::new(px, py);
        skyline_add(&mut sky, px, s.x, py + s.y);
        max_x = max_x.max(px + s.x);
        max_y = max_y.max(py + s.y);
    }
    (offsets, max_x, max_y)
}

/// The highest skyline top over the span `[x, x+w]` — the y a box of width `w` placed
/// at `x` would rest on.
fn skyline_top(sky: &[(f32, f32, f32)], x: f32, w: f32) -> f32 {
    let (l, r) = (x, x + w);
    let mut top = 0.0f32;
    for &(sx, sw, sy) in sky {
        if sx < r - 1e-3 && sx + sw > l + 1e-3 {
            top = top.max(sy);
        }
    }
    top
}

/// Lowest resting position for a `w`-wide box, trying each skyline segment's left edge
/// as a candidate x and preferring smaller y, then smaller x. Candidates that fit
/// within `bound_w` win over any that would overflow it.
fn skyline_fit(sky: &[(f32, f32, f32)], w: f32, bound_w: f32) -> (f32, f32) {
    // Lexicographic (y, then x) compare for f32 tuples.
    let better = |a: (f32, f32), b: (f32, f32)| match a.0.total_cmp(&b.0) {
        std::cmp::Ordering::Equal => a.1 < b.1,
        o => o.is_lt(),
    };
    let mut best: Option<(f32, f32)> = None; // (y, x) within bound
    let mut overflow: Option<(f32, f32)> = None; // (y, x) needing overflow
    for seg in sky {
        let x = seg.0;
        let y = skyline_top(sky, x, w);
        if x + w <= bound_w + 1e-3 {
            if best.is_none_or(|b| better((y, x), b)) {
                best = Some((y, x));
            }
        } else if overflow.is_none_or(|b| better((y, x), b)) {
            overflow = Some((y, x));
        }
    }
    let (y, x) = best.or(overflow).unwrap_or((0.0, 0.0));
    (x, y)
}

/// Raise the skyline over `[x, x+w]` to `ny`, splitting boundary segments at the box
/// edges and merging neighbours that end up level. Keeps `sky` a contiguous list.
fn skyline_add(sky: &mut Vec<(f32, f32, f32)>, x: f32, w: f32, ny: f32) {
    let (l, r) = (x, x + w);
    // Extend coverage rightward if the box overflowed the current skyline.
    if let Some(&(lx, lw, _)) = sky.last() {
        let end = lx + lw;
        if r > end + 1e-3 {
            sky.push((end, r - end, 0.0));
        }
    }
    let mut out: Vec<(f32, f32, f32)> = Vec::with_capacity(sky.len() + 2);
    for &(sx, sw, sy) in sky.iter() {
        let send = sx + sw;
        // Keep the portions of each segment outside [l, r] at their old height; the
        // covered middle is dropped and replaced by one flat segment below.
        if sx < l - 1e-3 {
            out.push((sx, send.min(l) - sx, sy));
        }
        if send > r + 1e-3 {
            let cut = sx.max(r);
            out.push((cut, send - cut, sy));
        }
    }
    out.push((l, w, ny));
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Merge adjacent same-height segments so the skyline doesn't fragment unboundedly.
    let mut merged: Vec<(f32, f32, f32)> = Vec::with_capacity(out.len());
    for seg in out {
        if let Some(last) = merged.last_mut() {
            if (last.2 - seg.2).abs() < 1e-3 && (last.0 + last.1 - seg.0).abs() < 1e-3 {
                last.1 += seg.1;
                continue;
            }
        }
        merged.push(seg);
    }
    *sky = merged;
}

/// Smallest power of two ≥ `x`, floored at 8.
fn next_pow2(x: u32) -> u32 {
    let mut p = 8u32;
    while p < x {
        p <<= 1;
    }
    p
}

/// Pack charts at density `d`, deriving a square power-of-two atlas and then scaling
/// density so the layout *fills* it (the rounded-up slack becomes extra texels, not
/// blank space). If the natural atlas would exceed `max_atlas`, density is reduced to
/// fit instead (and `clamped` is set). Returns chart pixel offsets, the atlas size,
/// the final (filled) density, and the flag.
fn pack_pixels(
    csize: &[Vec2],
    d: f32,
    gutter_px: u32,
    max_atlas: u32,
    fill: bool,
) -> (Vec<Vec2>, u32, f32, bool) {
    if csize.is_empty() {
        return (Vec::new(), 8, d, false);
    }
    // Largest power of two that still fits the GPU limit (≥ 8).
    let mut max_pow2 = 8u32;
    while (max_pow2 << 1) <= max_atlas {
        max_pow2 <<= 1;
    }

    // Pack once at the requested density to discover the natural footprint, then pick
    // the power-of-two atlas that holds it (clamped to the GPU limit).
    let (mut offsets, mut w, mut h) = skyline_pack(csize, d, gutter_px);
    let natural = next_pow2(w.max(h).ceil() as u32);
    let clamped = natural > max_pow2;
    let atlas = natural.min(max_pow2);

    // Scale density so the layout *fills* the atlas rather than leaving the rounded-up
    // power-of-two slack blank: since we're paying for the texture either way, spend
    // the leftover room on more (still uniform) texels. The same loop shrinks density
    // when the content is too big for the GPU max. Fixed-pixel gutters make this
    // non-linear, so we re-pack and converge (toward the atlas from below). Skipped
    // when `fill` is off (an exact target density): the slack stays blank so the
    // density isn't stretched away from what the caller asked for.
    let mut d = d;
    if fill {
        for _ in 0..8 {
            let cur = w.max(h).max(1.0);
            if (atlas as f32 / cur - 1.0).abs() < 0.01 {
                break;
            }
            d *= atlas as f32 / cur;
            let r = skyline_pack(csize, d, gutter_px);
            offsets = r.0;
            w = r.1;
            h = r.2;
        }
    }
    // Never exceed the atlas (keep UVs ≤ 1) — trim density if a re-pack overshot.
    let mut guard = 0;
    while w.max(h) > atlas as f32 && guard < 6 {
        d *= atlas as f32 / w.max(h);
        let r = skyline_pack(csize, d, gutter_px);
        offsets = r.0;
        w = r.1;
        h = r.2;
        guard += 1;
    }
    (offsets, atlas, d, clamped)
}

/// How a chart is placed into its shared slot: the affine map `lin · q + trans` (in the
/// chart's projected coords) that lands it on the slot representative's layout, so
/// corresponding vertices coincide texel-for-texel. `lin` is a rotation, or a
/// rotation+reflection for a mirror image; `trans` folds in the translation. The
/// representative — and *every* chart when overlap is off — gets `lin = I`,
/// `trans = -cmin`, i.e. the plain `q - cmin` layout the packer already expects.
#[derive(Clone, Copy)]
struct SlotRemap {
    lin: Mat2,
    trans: Vec2,
}

/// The identity placement for a chart drawn in its own slot: `q - cmin`.
fn id_remap(cmin: Vec2) -> SlotRemap {
    SlotRemap {
        lin: Mat2::IDENTITY,
        trans: -cmin,
    }
}

/// Deduplicate a chart's projected vertices by quantized position — charts arrive split, so
/// shared edges repeat a vertex. First-seen order; the isometry matching below is
/// order-independent.
fn dedup_verts(pts: &[Vec2], eps: f32) -> Vec<Vec2> {
    let inv = 1.0 / eps;
    let mut seen: HashMap<(i64, i64), ()> = HashMap::new();
    let mut out = Vec::new();
    for &p in pts {
        let key = ((p.x * inv).round() as i64, (p.y * inv).round() as i64);
        if seen.insert(key, ()).is_none() {
            out.push(p);
        }
    }
    out
}

/// The farthest-apart pair of points (the set's diameter), or `None` if all coincide. A
/// well-defined, isometry-stable anchor for registration.
fn diameter(v: &[Vec2]) -> Option<(usize, usize)> {
    let mut best = (0usize, 0usize);
    let mut best_d2 = 0.0f32;
    for i in 0..v.len() {
        for j in (i + 1)..v.len() {
            let d2 = v[i].distance_squared(v[j]);
            if d2 > best_d2 {
                best_d2 = d2;
                best = (i, j);
            }
        }
    }
    (best_d2 > 0.0).then_some(best)
}

/// The rotation taking unit vector `from` onto unit vector `to`.
fn rot_between(from: Vec2, to: Vec2) -> Mat2 {
    let c = from.dot(to); // cos θ
    let s = from.perp_dot(to); // sin θ
    Mat2::from_cols(Vec2::new(c, s), Vec2::new(-s, c))
}

/// Find the rigid (optionally mirrored) transform mapping every point of `m` onto a point
/// of `r` within `tol`, anchored on each set's diameter: returns `(lin, a_m, a_r)` for the
/// map `lin · (p - a_m) + a_r`. `None` if no orientation/reflection lines the sets up —
/// i.e. a distance-key collision between genuinely different shapes, which then stays
/// unmerged. `r` and `m` share a pairwise-distance key, so they're congruence candidates.
fn register(r: &[Vec2], m: &[Vec2], tol: f32) -> Option<(Mat2, Vec2, Vec2)> {
    let (ri, rj) = diameter(r)?;
    let (mi, mj) = diameter(m)?;
    let dir_r = (r[rj] - r[ri]).normalize();
    let tol2 = tol * tol;
    let reflect = Mat2::from_cols(Vec2::new(1.0, 0.0), Vec2::new(0.0, -1.0));
    // Map either end of the member's diameter onto r[ri], with and without a reflection —
    // four hypotheses, the first that lines every vertex up wins.
    for &(ma, mb) in &[(mi, mj), (mj, mi)] {
        let to_x = rot_between((m[mb] - m[ma]).normalize(), Vec2::X);
        for &mirror in &[false, true] {
            let refl = if mirror { reflect } else { Mat2::IDENTITY };
            // dir_m → +x, optionally reflect across it, then +x → dir_r.
            let lin = rot_between(Vec2::X, dir_r) * refl * to_x;
            let (a_m, a_r) = (m[ma], r[ri]);
            let ok = m.iter().all(|&p| {
                let q = lin * (p - a_m) + a_r;
                r.iter().any(|&rp| rp.distance_squared(q) <= tol2)
            });
            if ok {
                return Some((lin, a_m, a_r));
            }
        }
    }
    None
}

/// Collapse congruent charts onto shared atlas slots — the "overlap identical UVs"
/// feature. Two charts are congruent when one maps onto the other by a rigid motion or a
/// mirror; congruent charts share a `slot_of`, and each carries a `SlotRemap` placing it on
/// the slot's representative so corresponding vertices coincide texel-for-texel.
/// `slot_sizes` holds one box per slot (the representative's). With `overlap = false` every
/// chart is its own slot with an identity remap, so the caller's packing and UV math are
/// unchanged.
///
/// Charts are first bucketed by an isometry-invariant key (vertex count + sorted pairwise
/// distances), then each candidate is *registered* against its bucket's representatives and
/// the alignment verified vertex-by-vertex — so a key collision between genuinely different
/// shapes can't produce a bad overlap (it just falls back to a private slot). This is
/// independent of `chart_rectify`, whose minimum-area-box ties can otherwise orient
/// congruent charts inconsistently.
fn group_congruent_charts(
    proj: &[[Vec2; 3]],
    chart_of: &[usize],
    cmin: &[Vec2],
    csize: &[Vec2],
    num_charts: usize,
    eps: f32,
    overlap: bool,
) -> (Vec<usize>, Vec<Vec2>, Vec<SlotRemap>) {
    if !overlap {
        return (
            (0..num_charts).collect(),
            csize.to_vec(),
            (0..num_charts).map(|c| id_remap(cmin[c])).collect(),
        );
    }

    // Deduplicated projected vertices per chart.
    let mut chart_pts: Vec<Vec<Vec2>> = vec![Vec::new(); num_charts];
    for (ti, tri) in proj.iter().enumerate() {
        chart_pts[chart_of[ti]].extend_from_slice(tri);
    }
    let verts: Vec<Vec<Vec2>> = chart_pts.iter().map(|p| dedup_verts(p, eps)).collect();

    // The distance-key cell is coarse enough to absorb the ~1e-6 float drift between
    // congruent charts yet far finer than the gap between genuinely different shapes; it
    // also serves as the registration tolerance.
    let cell = eps * 10.0;
    let inv_cell = 1.0 / cell;
    // Skip matching for very large charts: the O(v²) distance key would blow up, and a huge
    // chart almost never has an exact duplicate anyway.
    const MAX_MATCH_VERTS: usize = 256;
    let dist_key = |v: &[Vec2]| -> Vec<i64> {
        let mut ds = vec![v.len() as i64];
        for i in 0..v.len() {
            for j in (i + 1)..v.len() {
                ds.push((v[i].distance(v[j]) * inv_cell).round() as i64);
            }
        }
        ds[1..].sort_unstable();
        ds
    };

    let mut slot_of = vec![0usize; num_charts];
    let mut slot_sizes: Vec<Vec2> = Vec::new();
    let mut remap: Vec<SlotRemap> = (0..num_charts).map(|c| id_remap(cmin[c])).collect();
    // distance key → representative chart indices sharing that key (usually exactly one).
    let mut buckets: HashMap<Vec<i64>, Vec<usize>> = HashMap::new();

    for c in 0..num_charts {
        let mut placed = false;
        if verts[c].len() <= MAX_MATCH_VERTS {
            let reps = buckets.entry(dist_key(&verts[c])).or_default();
            for &rep in reps.iter() {
                if let Some((lin, a_m, a_r)) = register(&verts[rep], &verts[c], cell) {
                    // slot-local: lin·(p−a_m)+a_r − cmin[rep]
                    //           = lin·p + (a_r − cmin[rep] − lin·a_m).
                    slot_of[c] = slot_of[rep];
                    remap[c] = SlotRemap {
                        lin,
                        trans: a_r - cmin[rep] - lin * a_m,
                    };
                    placed = true;
                    break;
                }
            }
            if !placed {
                reps.push(c); // new representative for this distance key
            }
        }
        if !placed {
            // Private slot, identity layout (already set in `remap`).
            slot_of[c] = slot_sizes.len();
            slot_sizes.push(csize[c]);
        }
    }
    (slot_of, slot_sizes, remap)
}

/// The full pipeline; returns the chart id per output triangle alongside the result
/// (used by tests). Output triangle order matches `chart_of` order.
fn unwrap_impl(mesh: &Mesh, opts: &UnwrapOptions) -> (UnwrapResult, Vec<usize>) {
    let tri_count = mesh.indices.len() / 3;
    if tri_count == 0 {
        return (
            UnwrapResult {
                mesh: build_split(&[]),
                atlas_size: opts.target_atlas_px.max(8),
                clamped: false,
                density_d: 1.0,
            },
            Vec::new(),
        );
    }

    let positions: Vec<[Vec3; 3]> = tri_positions(mesh).collect();
    let normals: Vec<Vec3> = positions.iter().map(|p| face_normal(*p)).collect();
    let areas: Vec<f32> = positions.iter().map(|p| tri_area(*p)).collect();
    // The source's per-vertex normals, kept for the split output so smooth shading
    // survives the unwrap. The face `normals` above drive chart projection only.
    let vert_normals: Vec<[Vec3; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|t| {
            [
                Vec3::from(mesh.vertices[t[0] as usize].normal),
                Vec3::from(mesh.vertices[t[1] as usize].normal),
                Vec3::from(mesh.vertices[t[2] as usize].normal),
            ]
        })
        .collect();

    let (min, max) = mesh.bounds();
    let eps = ((max - min).length() * 1e-5).max(1e-6);
    let weld = weld_positions(mesh, eps);
    let adj = build_adjacency(mesh, &weld);

    let cos_cone = opts.angle_cone_deg.to_radians().cos();
    let (chart_of, chart_normal) = grow_charts(&normals, &areas, &adj, cos_cone);
    let num_charts = chart_normal.len();

    // Project each triangle into its chart's tangent frame (coords in world units).
    let bases: Vec<(Vec3, Vec3)> = chart_normal.iter().map(|n| planar_basis(*n)).collect();
    let mut proj: Vec<[Vec2; 3]> = Vec::with_capacity(tri_count);
    for ti in 0..tri_count {
        let (tu, tv) = bases[chart_of[ti]];
        proj.push(positions[ti].map(|pt| Vec2::new(pt.dot(tu), pt.dot(tv))));
    }

    // Rectify (snap mode only): rotate each chart so its longest edge is axis-aligned.
    // A boxy face then becomes an axis-aligned rectangle whose edges can fall on texel
    // boundaries once snapped — instead of a diagonal that cuts across texels at any
    // resolution. Rigid rotation, so the no-fold projection guarantee is preserved.
    if opts.snap_texels {
        let rot = chart_rectify(&proj, &chart_of, num_charts);
        for ti in 0..tri_count {
            let (sin, cos) = rot[chart_of[ti]];
            // Rotate by -θ so the chart's longest edge lands along +U.
            proj[ti] = proj[ti].map(|p| Vec2::new(p.x * cos + p.y * sin, -p.x * sin + p.y * cos));
        }
    }

    // Per-chart world-space bounding box, measured after any rectify rotation.
    let mut cmin = vec![Vec2::splat(f32::INFINITY); num_charts];
    let mut cmax = vec![Vec2::splat(f32::NEG_INFINITY); num_charts];
    for ti in 0..tri_count {
        let c = chart_of[ti];
        for v in &proj[ti] {
            cmin[c] = cmin[c].min(*v);
            cmax[c] = cmax[c].max(*v);
        }
    }
    let csize: Vec<Vec2> = (0..num_charts)
        .map(|c| (cmax[c] - cmin[c]).max(Vec2::splat(1e-6)))
        .collect();

    // Collapse congruent charts onto shared slots when overlap is requested. `slot_of`
    // maps each chart to a packing slot, `slot_sizes` holds one box per slot, and `remap`
    // re-frames a chart's local coords onto its slot representative. With overlap off this
    // is a no-op: one slot per chart, identity remaps — the math below is unchanged.
    let match_eps = ((max - min).length() * 1e-4).max(1e-6);
    let (slot_of, slot_sizes, remap) = group_congruent_charts(
        &proj,
        &chart_of,
        &cmin,
        &csize,
        num_charts,
        match_eps,
        opts.overlap_identical,
    );

    // Choose density D so a `Medium` mesh lands near `target_atlas_px`. Total *slot* bbox
    // area is the natural scale (stacked charts share a slot, so they don't inflate it);
    // √η accounts for packing slack (gutters + the gaps the skyline can't fill).
    // `pack_pixels` then rescales D to fill the real atlas, so this only needs to be the
    // right ballpark.
    const ETA: f32 = 0.65;
    let a_world: f32 = slot_sizes.iter().map(|s| s.x * s.y).sum::<f32>().max(1e-6);
    let d_base = ETA.sqrt() * opts.target_atlas_px as f32 / a_world.sqrt();
    // An explicit `target_density` pins texels-per-world-unit exactly, so the atlas is
    // sized to hold the charts at it rather than stretched to fill (which would change
    // the density). The preset path keeps the fill-to-atlas behaviour for sharpness.
    let (d, fill) = match opts.target_density {
        Some(td) if td > 0.0 => (td, false),
        _ => (d_base * opts.density.multiplier(), true),
    };

    let (offsets, atlas, d, clamped) =
        pack_pixels(&slot_sizes, d, opts.gutter_px, opts.max_atlas, fill);
    let g = Vec2::splat(opts.gutter_px as f32);
    let inv_atlas = 1.0 / atlas as f32;

    let tris: Vec<_> = (0..tri_count)
        .map(|ti| {
            let c = chart_of[ti];
            let rm = remap[c];
            let uv = proj[ti].map(|q| {
                // Re-frame the chart's projected point into its slot (identity `q - cmin[c]`
                // unless this chart is overlapped onto a representative), then scale to px.
                let r = rm.lin * q + rm.trans;
                let mut local = r * d;
                // Snap each vertex to a whole texel. The slot's min corner sits at the
                // origin and `offsets`/`g` are integer pixels, so a rounded local pixel
                // puts every vertex exactly on a texel corner and face edges land on the
                // grid. Shared world positions project identically, so a seam edge snaps
                // the same on both charts and stays watertight; congruent charts re-framed
                // onto one slot snap onto the same texels, so they overlap exactly.
                if opts.snap_texels {
                    local = local.round();
                }
                let px = local + offsets[slot_of[c]] + g;
                px * inv_atlas
            });
            (positions[ti], vert_normals[ti], uv)
        })
        .collect();

    (
        UnwrapResult {
            mesh: build_split(&tris),
            atlas_size: atlas,
            clamped,
            density_d: d,
        },
        chart_of,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_uvs_in_unit(mesh: &Mesh) {
        for v in &mesh.vertices {
            assert!(
                (-1e-4..=1.0 + 1e-4).contains(&v.uv[0]) && (-1e-4..=1.0 + 1e-4).contains(&v.uv[1]),
                "uv out of range: {:?}",
                v.uv
            );
        }
    }

    #[test]
    fn unwrap_preserves_smooth_vertex_normals() {
        // A triangle whose vertex normals are NOT its face normal (+Z): the unwrap must
        // carry the smooth per-vertex normals through to the split output, not flatten
        // them to the face normal (which would export a faceted model).
        let n = [
            Vec3::new(0.0, 0.6, 0.8).normalize(),
            Vec3::new(0.6, 0.0, 0.8).normalize(),
            Vec3::new(-0.4, -0.4, 0.82).normalize(),
        ];
        let pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mesh = Mesh {
            vertices: (0..3)
                .map(|i| Vertex {
                    position: pos[i],
                    normal: n[i].to_array(),
                    uv: [0.0, 0.0],
                })
                .collect(),
            indices: vec![0, 1, 2],
            needs_normals: false,
            needs_uvs: false,
            source_transform: Default::default(),
        };
        let r = auto_unwrap(&mesh, &UnwrapOptions::default());
        assert_eq!(r.mesh.vertices.len(), 3);
        for (out, want) in r.mesh.vertices.iter().zip(n) {
            let got = Vec3::from(out.normal);
            assert!(
                got.distance(want) < 1e-5,
                "normal {got} should match the source smooth normal {want}, not the face normal"
            );
        }
    }

    /// A flat quad (two triangles, +Z normal) at x ∈ [x0, x0+1], y ∈ [0,1].
    fn quad(x0: f32) -> [Vertex; 6] {
        let v = |x: f32, y: f32| Vertex {
            position: [x, y, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
        };
        [
            v(x0, 0.0),
            v(x0 + 1.0, 0.0),
            v(x0 + 1.0, 1.0),
            v(x0, 0.0),
            v(x0 + 1.0, 1.0),
            v(x0, 1.0),
        ]
    }

    /// Two coplanar, same-normal quads that share no edge — the exact case the old
    /// normal-only clustering stacked on top of each other.
    fn two_separate_quads() -> Mesh {
        let mut vertices = quad(0.0).to_vec();
        vertices.extend_from_slice(&quad(5.0)); // gap so no shared vertices/edges
        mesh_of(vertices)
    }

    /// A split-vertex mesh (one index per vertex) from a flat vertex list.
    fn mesh_of(vertices: Vec<Vertex>) -> Mesh {
        Mesh {
            indices: (0..vertices.len() as u32).collect(),
            vertices,
            needs_normals: false,
            needs_uvs: false,
            source_transform: crate::mesh::SourceTransform::IDENTITY,
        }
    }

    /// Default options with the "overlap identical UVs" toggle on.
    fn overlap_opts() -> UnwrapOptions {
        UnwrapOptions {
            overlap_identical: true,
            ..Default::default()
        }
    }

    /// Per-triangle UV bounding boxes in pixel space, in output triangle order.
    fn pixel_bboxes(mesh: &Mesh, atlas: u32) -> Vec<(Vec2, Vec2)> {
        mesh.indices
            .chunks_exact(3)
            .map(|t| {
                let uvs =
                    [0, 1, 2].map(|k| Vec2::from(mesh.vertices[t[k] as usize].uv) * atlas as f32);
                let mn = uvs
                    .iter()
                    .copied()
                    .fold(Vec2::splat(f32::INFINITY), Vec2::min);
                let mx = uvs
                    .iter()
                    .copied()
                    .fold(Vec2::splat(f32::NEG_INFINITY), Vec2::max);
                (mn, mx)
            })
            .collect()
    }

    fn disjoint(a: (Vec2, Vec2), b: (Vec2, Vec2)) -> bool {
        a.1.x <= b.0.x + 1e-3
            || b.1.x <= a.0.x + 1e-3
            || a.1.y <= b.0.y + 1e-3
            || b.1.y <= a.0.y + 1e-3
    }

    #[test]
    fn auto_unwrap_uvs_in_unit() {
        let r = auto_unwrap(&Mesh::cube(), &UnwrapOptions::default());
        assert_eq!(r.mesh.vertices.len(), 36); // 12 tris → 36 split verts
        assert!(!r.mesh.needs_uvs);
        assert!(r.atlas_size >= 8 && r.atlas_size.is_power_of_two());
        assert_uvs_in_unit(&r.mesh);
    }

    #[test]
    fn packing_fills_the_atlas() {
        // Density is scaled so the layout fills the power-of-two atlas rather than
        // leaving the rounded-up slack blank. The global UV bbox must therefore reach
        // the atlas edge in its limiting dimension, and cover a healthy area overall.
        for mesh in [Mesh::cube(), two_separate_quads()] {
            let r = auto_unwrap(&mesh, &UnwrapOptions::default());
            let (mut mn, mut mx) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
            for v in &r.mesh.vertices {
                let uv = Vec2::from(v.uv);
                mn = mn.min(uv);
                mx = mx.max(uv);
            }
            let span = mx - mn;
            assert!(
                span.x.max(span.y) >= 0.9,
                "layout leaves the atlas mostly blank: span {span:?}"
            );
            // Sum of chart bbox areas vs atlas — a lower bound on real coverage.
            let bb = pixel_bboxes(&r.mesh, r.atlas_size);
            let covered: f32 = bb.iter().map(|(a, b)| (b.x - a.x) * (b.y - a.y)).sum();
            let frac = covered / (r.atlas_size * r.atlas_size) as f32;
            assert!(frac >= 0.4, "atlas only {:.0}% covered", frac * 100.0);
        }
    }

    #[test]
    fn disconnected_same_normal_parts_get_separate_charts() {
        let (r, chart_of) = unwrap_impl(&two_separate_quads(), &UnwrapOptions::default());
        // The two quads (tris 0-1 and 2-3) must land in different charts...
        let charts: std::collections::HashSet<_> = chart_of.iter().copied().collect();
        assert!(charts.len() >= 2, "separate quads collapsed into one chart");
        assert_ne!(chart_of[0], chart_of[2]);
        // ...and occupy non-overlapping pixel regions (the old overlap bug).
        let bb = pixel_bboxes(&r.mesh, r.atlas_size);
        for a in [0usize, 1] {
            for b in [2usize, 3] {
                assert!(
                    disjoint(bb[a], bb[b]),
                    "charts overlap: {:?} {:?}",
                    bb[a],
                    bb[b]
                );
            }
        }
        assert_uvs_in_unit(&r.mesh);
    }

    #[test]
    fn overlap_stacks_identical_charts() {
        // The exact opposite of the test above: with overlap on, the two congruent quads
        // collapse onto the *same* slot instead of getting separate regions.
        let opts = overlap_opts();
        let r = auto_unwrap(&two_separate_quads(), &opts);
        assert_uvs_in_unit(&r.mesh);
        let bb = pixel_bboxes(&r.mesh, r.atlas_size);
        for a in [0usize, 1] {
            for b in [2usize, 3] {
                assert!(
                    !disjoint(bb[a], bb[b]),
                    "stacked charts should overlap: {:?} {:?}",
                    bb[a],
                    bb[b]
                );
            }
        }
        // Corresponding triangles land on the same texels (exact overlap, not just same box).
        assert!(
            (bb[0].0 - bb[2].0).length() < 1.0 && (bb[0].1 - bb[2].1).length() < 1.0,
            "quad A tri0 {:?} and quad B tri0 {:?} should coincide",
            bb[0],
            bb[2]
        );
        // Stacking two identical quads costs no more atlas than a single quad.
        let single = auto_unwrap(&mesh_of(quad(0.0).to_vec()), &opts);
        assert!(
            r.atlas_size <= single.atlas_size,
            "overlap atlas {} exceeds single-quad atlas {}",
            r.atlas_size,
            single.atlas_size
        );
    }

    #[test]
    fn overlap_collapses_cube_faces() {
        // Every cube face is the same square, so with overlap on all six faces collapse
        // onto one slot — every triangle's UV bbox is the same box. (Triangulation runs
        // both diagonal directions across the faces, so this also exercises the D4
        // rotation/reflection canonicalization.)
        let r = auto_unwrap(&Mesh::cube(), &overlap_opts());
        assert_uvs_in_unit(&r.mesh);
        let bb = pixel_bboxes(&r.mesh, r.atlas_size);
        let first = bb[0];
        for (i, b) in bb.iter().enumerate() {
            assert!(
                (b.0 - first.0).length() < 1.0 && (b.1 - first.1).length() < 1.0,
                "cube tri {i} bbox {b:?} differs from {first:?}; faces did not collapse"
            );
        }
    }

    #[test]
    fn overlap_stacks_mirror_images() {
        // Two chiral (right) triangles in the same plane, one the mirror of the other.
        // Only a reflection maps one onto the other, so they stack *only* because D4
        // includes reflections — without them they'd stay in separate slots.
        let pv = |x: f32, y: f32| Vertex {
            position: [x, y, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
        };
        let mesh = mesh_of(vec![
            pv(0.0, 0.0),
            pv(2.0, 0.0),
            pv(0.0, 1.0), // tri A (CCW, +Z)
            pv(5.0, 0.0),
            pv(5.0, -1.0),
            pv(7.0, 0.0), // tri B: A mirrored across the X axis (still CCW, +Z)
        ]);

        // Off: the two mirror charts get separate, non-overlapping slots.
        let off = auto_unwrap(&mesh, &UnwrapOptions::default());
        let obb = pixel_bboxes(&off.mesh, off.atlas_size);
        assert!(
            disjoint(obb[0], obb[1]),
            "without overlap, mirror charts must not stack: {:?} {:?}",
            obb[0],
            obb[1]
        );

        // On: they collapse onto the same slot.
        let r = auto_unwrap(&mesh, &overlap_opts());
        assert_uvs_in_unit(&r.mesh);
        let bb = pixel_bboxes(&r.mesh, r.atlas_size);
        assert!(
            !disjoint(bb[0], bb[1]),
            "mirror charts should stack: {:?} {:?}",
            bb[0],
            bb[1]
        );
        assert!(
            (bb[0].0 - bb[1].0).length() < 1.5 && (bb[0].1 - bb[1].1).length() < 1.5,
            "mirror charts didn't coincide: {:?} vs {:?}",
            bb[0],
            bb[1]
        );
    }

    #[test]
    fn welding_recovers_adjacency() {
        // The cube ships as split vertices; welding must reconnect each face's two
        // triangles into one chart (6 charts, 2 tris each) — without welding it would
        // fragment into 12.
        let (_, chart_of) = unwrap_impl(&Mesh::cube(), &UnwrapOptions::default());
        let charts: std::collections::HashSet<_> = chart_of.iter().copied().collect();
        assert_eq!(charts.len(), 6, "expected 6 charts, got {}", charts.len());
        for c in charts {
            let n = chart_of.iter().filter(|&&x| x == c).count();
            assert_eq!(n, 2, "chart {c} has {n} tris, expected 2");
        }
    }

    #[test]
    fn constant_density_within_cone_bound() {
        // The constant-density guarantee is a property of the projection. Texel
        // snapping deliberately quantizes vertices on top of it, perturbing each
        // triangle's exact area, so this invariant is checked with snapping off.
        let opts = UnwrapOptions {
            snap_texels: false,
            ..UnwrapOptions::default()
        };
        let r = auto_unwrap(&Mesh::cube(), &opts);
        let d2 = r.density_d * r.density_d;
        let cos2 = opts.angle_cone_deg.to_radians().cos().powi(2);
        for t in r.mesh.indices.chunks_exact(3) {
            let p = [0, 1, 2].map(|k| Vec3::from(r.mesh.vertices[t[k] as usize].position));
            let uv = [0, 1, 2]
                .map(|k| Vec2::from(r.mesh.vertices[t[k] as usize].uv) * r.atlas_size as f32);
            let world_area = tri_area(p);
            let uv_area = 0.5 * (uv[1] - uv[0]).perp_dot(uv[2] - uv[0]).abs();
            let ratio = uv_area / world_area; // texels² per world unit²
                                              // Each cube face is flat (θ=0), so the ratio should equal D² exactly;
                                              // the cone bound is the general guarantee.
            assert!(
                ratio >= d2 * cos2 - 1.0 && ratio <= d2 + 1.0,
                "density ratio {ratio} outside [{}, {}]",
                d2 * cos2,
                d2
            );
        }
    }

    #[test]
    fn atlas_within_max() {
        let opts = UnwrapOptions {
            max_atlas: 64,
            target_atlas_px: 128, // natural atlas would exceed 64 → must clamp
            ..Default::default()
        };
        let r = auto_unwrap(&Mesh::cube(), &opts);
        assert!(r.atlas_size <= 64, "atlas {} exceeded max", r.atlas_size);
        assert!(r.clamped, "expected density to be clamped");
        assert_uvs_in_unit(&r.mesh);
    }

    #[test]
    fn explicit_density_is_honored_exactly() {
        // Pinning texels-per-world-unit must yield exactly that density: the atlas is
        // sized to hold the charts rather than stretched to fill, so unlike the preset
        // path the request isn't rescaled away.
        let opts = UnwrapOptions {
            target_density: Some(50.0),
            ..Default::default()
        };
        let r = auto_unwrap(&Mesh::cube(), &opts);
        assert!(!r.clamped, "50 texels/unit should fit the default max atlas");
        assert!(
            (r.density_d - 50.0).abs() < 1e-3,
            "explicit density {} != requested 50",
            r.density_d
        );
        assert!(r.atlas_size.is_power_of_two(), "atlas not a power of two");
        assert_uvs_in_unit(&r.mesh);
    }

    #[test]
    fn explicit_density_clamps_when_too_dense() {
        // A density that would overflow the GPU limit is trimmed (and flagged), never
        // exceeding the atlas cap.
        let opts = UnwrapOptions {
            target_density: Some(100_000.0),
            max_atlas: 64,
            ..Default::default()
        };
        let r = auto_unwrap(&Mesh::cube(), &opts);
        assert!(r.clamped, "an impossible density should clamp");
        assert!(r.atlas_size <= 64, "atlas {} exceeded max", r.atlas_size);
        assert!(r.density_d < 100_000.0, "clamped density not reduced");
        assert_uvs_in_unit(&r.mesh);
    }

    #[test]
    fn single_triangle_and_empty() {
        let tri = Mesh {
            vertices: vec![
                Vertex {
                    position: [0.0, 0.0, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    uv: [0.0, 0.0],
                },
                Vertex {
                    position: [1.0, 0.0, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    uv: [0.0, 0.0],
                },
                Vertex {
                    position: [0.0, 1.0, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    uv: [0.0, 0.0],
                },
            ],
            indices: vec![0, 1, 2],
            needs_normals: false,
            needs_uvs: false,
            source_transform: crate::mesh::SourceTransform::IDENTITY,
        };
        let r = auto_unwrap(&tri, &UnwrapOptions::default());
        assert_eq!(r.mesh.vertices.len(), 3);
        assert_uvs_in_unit(&r.mesh);

        let empty = Mesh {
            vertices: vec![],
            indices: vec![],
            needs_normals: false,
            needs_uvs: false,
            source_transform: crate::mesh::SourceTransform::IDENTITY,
        };
        let r = auto_unwrap(&empty, &UnwrapOptions::default());
        assert_eq!(r.mesh.vertices.len(), 0);
        assert!(r.atlas_size >= 8);
    }

    #[test]
    fn snapped_uvs_land_on_texel_boundaries() {
        // With snapping on (the default), every vertex sits on a whole texel, so face
        // edges coincide with the texel grid instead of cutting across it.
        let r = auto_unwrap(&Mesh::cube(), &UnwrapOptions::default());
        for v in &r.mesh.vertices {
            for px in [v.uv[0], v.uv[1]].map(|c| c * r.atlas_size as f32) {
                assert!(
                    (px - px.round()).abs() < 1e-3,
                    "vertex UV at {px} px is not on a texel boundary"
                );
            }
        }
    }

    #[test]
    fn rectify_squares_faces_to_their_sides_not_diagonals() {
        // Each cube face is an axis-aligned square split into two triangles, so a
        // correctly rectified triangle keeps two of its three edges axis-aligned (a
        // horizontal leg and a vertical leg). The old "align the longest edge" rule
        // squared faces to the diagonal instead, rotating them 45° — every edge would
        // then be off-axis and this would fail.
        let r = auto_unwrap(&Mesh::cube(), &UnwrapOptions::default());
        let atlas = r.atlas_size as f32;
        for t in r.mesh.indices.chunks_exact(3) {
            let uv = [0, 1, 2].map(|k| Vec2::from(r.mesh.vertices[t[k] as usize].uv) * atlas);
            let axis_aligned = (0..3)
                .filter(|&e| {
                    let d = uv[(e + 1) % 3] - uv[e];
                    d.x.abs() < 1e-3 || d.y.abs() < 1e-3
                })
                .count();
            assert!(
                axis_aligned >= 2,
                "face triangle is not axis-aligned (diamond regression): {uv:?}"
            );
        }
    }

    #[test]
    fn skyline_packs_tightly_without_overlap() {
        // A heterogeneous box set — exactly where shelf packing wasted the gaps above
        // shorter boxes. The skyline must (a) never overlap two boxes and (b) fill a
        // high fraction of its bounding extent.
        let csize = [
            Vec2::new(10.0, 30.0),
            Vec2::new(20.0, 5.0),
            Vec2::new(8.0, 8.0),
            Vec2::new(15.0, 22.0),
            Vec2::new(5.0, 12.0),
            Vec2::new(25.0, 6.0),
            Vec2::new(12.0, 18.0),
            Vec2::new(7.0, 7.0),
        ];
        let (offsets, w, h) = skyline_pack(&csize, 1.0, 0);

        for i in 0..csize.len() {
            for j in (i + 1)..csize.len() {
                let (amn, amx) = (offsets[i], offsets[i] + csize[i]);
                let (bmn, bmx) = (offsets[j], offsets[j] + csize[j]);
                let overlap = amn.x < bmx.x - 1e-3
                    && bmn.x < amx.x - 1e-3
                    && amn.y < bmx.y - 1e-3
                    && bmn.y < amx.y - 1e-3;
                assert!(!overlap, "boxes {i} and {j} overlap");
            }
        }

        let used: f32 = csize.iter().map(|s| s.x * s.y).sum();
        let frac = used / (w * h);
        assert!(frac >= 0.75, "skyline only {:.0}% efficient", frac * 100.0);
    }

    #[test]
    fn snapping_keeps_faces_non_degenerate() {
        // Rounding to whole texels must not collapse a face: every triangle still spans
        // at least one texel in each axis after the snap.
        let r = auto_unwrap(&Mesh::cube(), &UnwrapOptions::default());
        for (lo, hi) in pixel_bboxes(&r.mesh, r.atlas_size) {
            assert!(
                hi.x - lo.x >= 1.0 - 1e-3 && hi.y - lo.y >= 1.0 - 1e-3,
                "a face snapped down to a sub-texel sliver: {lo:?}..{hi:?}"
            );
        }
    }
}

# GPU Painting Plan (Tier 2 → "crazy performance")

Status: **proposed** · supersedes/expands roadmap goal **G12 (GPU compositing)**
Scope: move the paint hot path off the CPU so painting stays interactive at
2K–4K atlases. Companion to the Tier 0/1 CPU work already landed (rayon-parallel
compositing + version-stamped stroke buffers).

---

## TL;DR

Today every painted texel is touched on the **CPU**; the GPU only samples one
pre-composited texture. After the Tier 1 CPU wins, profiling shows the remaining
cost is almost entirely the **per-dab surface brush** (`splat` + `deposit`), not
the display pipeline. The fix is to **stamp dabs and composite layers on the
GPU**, leaving the CPU to compute only a tiny per-dab *face set* and to feed
stroke positions. End state: paint cost scales with screen/atlas *throughput*,
not with CPU single-thread speed, and 4K becomes viable.

---

## Where the time goes today (measured)

Tier 0 profiler (`LOWTEX_PROFILE=1`, `src/perf.rs`), steady-state painting on a
**2048²** atlas with a large brush, per 1-second window:

| Phase              | ms/sec   | calls/sec | per-call |
|--------------------|----------|-----------|----------|
| `deposit`          | ~500 ms  | ~140      | ~3.5 ms  |
| `splat`            | ~465 ms  | ~140      | ~3.1 ms  |
| `composite_region` | ~2–12 ms | 1–4       | ~2.2 ms  |
| `bleed_region`     | ~0.5–28 ms | 1–4     | ~6 ms    |
| `upload_region`    | ~0.5–5 ms | 1–4      | ~0.6 ms  |

**`splat` + `deposit` ≈ 96–99 % of paint CPU.** The display pipeline
(`composite` + `quantize` + `bleed` + `upload`) is now ~2–4 % — the dirty-rect +
rayon + per-stroke O(n²) removal already did their job there. So Tier 2 must
target the per-dab surface brush first; GPU compositing is a secondary win.

### Current data flow (per dab)

```
hit (ray pick, BVH)
  └─ surface::splat        flood-fill across mesh adjacency within a WORLD radius,
                           emitting every atlas texel the dab covers   [CPU, ~3ms]
  └─ Renderer::deposit     blend each (texel, coverage) into the active layer's
                           CPU pixels, max-coverage per stroke         [CPU, ~3.5ms]
        ↓ (coalesced once per frame, dirty rect only)
  layers::composite_into_region   blend visible layers → display_buf   [CPU, rayon]
  Palette::quantize_region        palette + Bayer dither               [CPU, rayon]
  bleed::dilate_region            gutter dilation across seams         [CPU]
  upload_region (queue.write_texture)  → the single GPU paint texture  [CPU→GPU]
        ↓
  main.wgsl samples the paint texture on the model
```

Key code: `src/surface.rs` (`splat`), `src/renderer.rs` (`deposit`,
`deposit_rgb`, `refresh_display_region`, `flush_paint`), `src/layers.rs`
(`composite`, `composite_into_region`), `src/palette.rs`, `src/bleed.rs`,
`src/shaders/main.wgsl`.

### The other CPU cost: the mesh-map bake (AO / curvature / sun)

Not in the per-dab loop above, but the second-largest CPU pixel job and the one
behind the product's moat ("the mesh tells you where to paint"). `src/bake.rs`
computes per-texel, **at atlas resolution**, by ray-casting the BVH:

- **AO** — `AO_SAMPLES` (24) cosine-weighted hemisphere rays per texel. At 2K
  that's ~16M texels × 24 ≈ **400M ray casts**; the "sub-second bake" assumption
  holds at PSX sizes, not at 2K. Rebaked on unwrap / resolution change
  (`ensure_mesh_maps`).
- **Sun / shadows** — `compute_light`: `max(N·L,0)` plus one shadow ray per texel
  against the BVH, **re-baked every time the sun direction or shadow toggle
  changes** — i.e. on an *interactive* slider. At 2K dragging the sun is a
  16M-shadow-ray recompute per change → unusable.
- **Curvature/edge + world position** — rasterized per texel (cheap).

Key code: `src/bake.rs` (`bake`, `compute_light`, `MeshMaps`), `src/renderer.rs`
(`ensure_mesh_maps`, `set_sun`), `src/effects.rs` (consumers via `MeshMaps::sample`).

**Immediate CPU win (Tier 1, available now):** the AO loop (`for idx in 0..n`)
and the `compute_light` loop are per-texel independent — the same rayon
row/index parallelization used for compositing applies directly, for a near-linear
speedup with no behaviour change. Worth landing before the GPU port.

---

## Target architecture

1. **Layers live as GPU textures** (one RGBA8 color + one R8 mask per layer; an
   array texture or a small pool). The CPU keeps shadow copies only where needed
   (undo, export) — ideally lazily, via readback.
2. **Dabs are stamped on the GPU** directly into the active layer, by rendering
   the affected mesh triangles into the atlas (UV-as-position) with a per-fragment
   brush falloff. No CPU per-texel loop.
3. **Compositing is a GPU pass** (or folded into the model fragment shader), with
   palette quantize + Bayer dither + gutter bleed as shader stages. No CPU
   composite, no `write_texture` upload during strokes.
4. **Mesh maps (AO / curvature / sun) bake on the GPU** — the BVH lives in a GPU
   buffer and a compute pass ray-traces occlusion/shadows per texel, so AO scales
   to 2K/4K and the sun is an interactive slider. The maps stay GPU-resident and
   feed the effect/compositing passes.
5. CPU's painting job shrinks to: ray-pick the hit, compute the **face set**
   within brush range (cheap adjacency walk — bounded by faces, not texels), and
   drive the GPU passes.

---

## The hard problem: a *surface* brush in texture space

`surface::splat` is a **geodesic-ish flood** across mesh adjacency: it paints
across UV seams and onto neighbouring faces, but it will *not* bleed through to
the back of a thin part or across a fold that is near in 3D yet far on the
surface. Preserving that on the GPU is the crux. Three options:

- **(A) Euclidean projection paint.** Render every triangle near the brush into
  the atlas (vertex shader emits UV → clip space; fragment gets world position);
  the fragment shader writes coverage where `distance(worldPos, center) < radius`,
  gated by a normal-facing test. Simplest and fastest, but it's a *behaviour
  change* — it can bleed through thin geometry the flood currently blocks.

- **(B) Hybrid: CPU face-flood + GPU per-texel.** Keep the adjacency flood, but
  stop it at **face granularity** — emit the *set of faces* within surface range
  (cheap: we already walk this graph; we just skip the expensive per-texel
  enumeration). Upload that face set and rasterize *only those faces* into the
  atlas with the per-fragment falloff. Preserves cross-seam + geodesic
  correctness at face resolution while moving the per-texel cost (the ~7 ms) to
  the GPU. **Recommended.**

- **(C) True geodesic on GPU.** Distance-in-texture-space precompute per dab.
  Overkill; rejected.

Plan adopts **(B)**, with **(A)** available as a "fast/loose" toggle. Note (B)
turns today's `splat` from *texel-emitting* into *face-emitting*, which is a small
change to `src/surface.rs` and removes most of `splat`'s cost too — not just
`deposit`'s.

---

## Phased plan

Each phase is independently shippable and guarded by parity tests (see below).

### Phase 1 — GPU dab stamping (the ~95 % win)

Move `splat` + `deposit` to the GPU.

- New `src/shaders/dab.wgsl`: vertex stage maps a triangle's UVs into atlas clip
  space; fragment stage computes brush falloff from interpolated world position
  (radius + hardness, matching `surface::splat`'s falloff curve) and outputs
  coverage·color.
- New render target: the **active layer** texture (render-to-texture). Add the
  `RENDER_ATTACHMENT` usage to layer textures.
- **Stroke coverage texture** (R8/R16F) for the max-coverage discipline: each dab
  `max`-blends into it; the stroke is composited onto the layer from the stroke
  buffer (per frame or on stroke end), mirroring today's `stroke_coverage` +
  `stroke_base` logic but on the GPU. `begin_stroke` clears the stroke texture
  (a cheap clear, or the same version-stamp idea via a "stroke id" uniform).
- CPU changes: `surface::splat` → emit the face set (option B); `deposit` is
  deleted from the hot path. A small vertex/index scratch buffer holds the dab's
  faces.
- Brush-image / stamp / eraser / mask-paint variants become fragment-shader
  branches (sample the brush material texture; write to mask target instead of
  color).

Touches: `surface.rs`, `renderer.rs` (paint path, new pipeline), new `dab.wgsl`.
Expected: removes ~95 % of current paint CPU; per-dab cost becomes a small draw.

### Phase 2 — GPU compositing (folds in G12)

- Layer stack → GPU textures; a compositing pass blends them (fragment over a
  fullscreen quad into a composite texture, **or** composite on-the-fly in
  `main.wgsl` — preferred, makes cost independent of layer count).
- Port `Palette::quantize_rgba` (nearest + 4×4 Bayer, `bayer4`) and
  `bleed::dilate` into shader stages. Quantize is trivially per-fragment; gutter
  bleed becomes a small ping-pong dilation pass over the composite (or a
  precomputed "nearest valid texel" map, since UV coverage is static between
  unwraps).
- Deletes `composite_into_region`, `quantize_region`, `dilate_region`,
  `upload_region` from the runtime path (kept for headless/export parity).

Touches: `layers.rs`, `palette.rs`, `bleed.rs`, `main.wgsl`, new composite shader.

### Phase 2.5 — Mesh maps (AO / curvature / sun) on the GPU

The bake suite is the moat *and* a heavy per-texel ray-tracing job — a textbook
GPU compute workload. Move `bake.rs` onto the GPU so AO scales to 2K/4K and the
sun becomes truly interactive.

- **Upload the BVH + triangles to GPU buffers** (we already build a `Bvh` in
  `src/bvh.rs` for picking). A compute shader traverses it per texel — the same
  algorithm as the CPU bake, just parallel over millions of texels.
- **`bake_ao.wgsl`** (compute): for each atlas texel, read its world position +
  normal (from the rasterized `pos`/normal maps), cast the 24 hemisphere rays
  against the BVH, write occlusion → an `ao` GPU texture. Matches `AO_SAMPLES`
  and the cosine-weighted sampling for parity.
- **`bake_sun.wgsl`** (compute): `max(N·L,0)` + one BVH shadow ray per texel →
  `light` texture. Because it's now a GPU pass, **dragging the sun re-bakes in
  real time** even at 2K (the current pain point). Alternative for the shadow
  term: a light-space depth/shadow map (cheaper, slightly less exact) — but BVH
  rays match the CPU result exactly, so prefer them for parity.
- **Curvature/edge + world position**: rasterization passes into GPU textures
  (the `pos`/curvature raster is already simple), feeding both the AO/sun compute
  passes and the effects.
- The mesh maps become **GPU textures** sampled directly by the effect /
  compositing passes (ties into Phase 2): `MeshMaps::sample` consumers
  (Darken-AO, Highlights, Dirt/Edge-wear presets, "mask from map", gradient map)
  read them on the GPU instead of the CPU.

Touches: `bake.rs`, `bvh.rs` (GPU-uploadable layout), `effects.rs`,
`renderer.rs` (`ensure_mesh_maps`, `set_sun`), new `bake_ao.wgsl` / `bake_sun.wgsl`.
Payoff: AO bake drops from seconds to milliseconds at 2K; sun/shadow becomes a
live slider; unblocks high-res use of the mesh-aware generators (the moat).

Dependency note: this shares the **BVH-on-GPU** infrastructure with option (B) of
the dab pass and could be the first place that lands, since it's a self-contained
compute job with a clean CPU reference to diff against.

### Phase 3 — History, save & export against GPU state

- **Undo/redo**: today `history.rs` snapshots CPU layer pixels. Options: (a) keep
  a CPU shadow updated by reading back the dirty rect on `end_stroke` (simple,
  one readback per stroke); (b) a GPU snapshot ring. Start with (a).
- **Export / Save PNG / `.lowtex`**: read back the composited texture on demand
  (`display_pixels`, `save_texture_png`, `export_png`, `project.rs`). Already
  off-hot-path.

Touches: `history.rs`, `renderer.rs`, `project.rs`, `export.rs`.

### Phase 4 — Tier 3: scale past 4K (virtual / sparse textures)

- Tiled, sparsely-resident atlas so memory and cost ∝ *painted* area, not atlas
  size. Unlocks 4K/8K. Large, optional, only if users push past 2K.

---

## Cross-cutting correctness

- **Parity tests are the safety net.** The repo already has byte-identical guards
  (`composite_into_region_matches_full`, `quantize_region_matches_full`,
  `real_stroke_flush_matches_full`, `region_matches_full_*`,
  `dilate_region_matches_full_*`). Strategy: keep the CPU reference paths, add
  GPU-vs-CPU comparison tests (render a dab/composite on GPU, read back, assert
  within a small tolerance of the CPU result). The headless `--screenshot` path
  and lavapipe in CI make GPU tests runnable without a real GPU.
- **PSX look must be preserved exactly**: nearest sampling, palette + Bayer
  dither, gutter bleed across seams (DESIGN principle #5). Quantize and bleed
  ports must match `bayer4` and `dilate` pixel-for-pixel (tolerance 0).
- **Stroke discipline**: max-coverage-per-stroke (no double-darken) must survive
  the move — the stroke coverage texture replicates `stroke_coverage`.
- **Determinism**: GPU rasterization of overlapping triangles into the same atlas
  texel must be order-independent (it is, with max-coverage blending) — important
  for the overlap-identical-UVs feature where charts share texels.
- **Bake parity**: GPU AO/sun must match the CPU `bake`/`compute_light` within a
  small tolerance (ray-tracing the same BVH with the same sample pattern). Diff a
  GPU bake against the CPU reference on a known mesh; the cosine-weighted
  hemisphere sampling and `AO_SAMPLES`/`ao_dist` must line up.

---

## Open questions

1. Geodesic fidelity: is option (A)'s thin-wall bleed-through acceptable as a
   default, or is (B) required? (Leaning (B).)
2. Layer storage: texture array vs. pool vs. atlas-of-layers — pick by layer
   count limits and bind-group churn.
3. Composite-in-`main.wgsl` (no intermediate texture) vs. cached composite
   texture updated per dirty tile — the former is simpler and removes upload
   entirely; the latter helps the 2D UV editor which wants the composited atlas.
4. Undo readback cost at 2K (one dirty-rect readback per stroke) — measure before
   choosing a GPU snapshot ring.
5. wgpu 22 specifics: render-to-texture with the existing surface format, storage
   textures vs. render attachments for the dab pass.

## Non-goals

- PBR / Substance surface area (per DESIGN.md).
- Changing the painting *model* (cross-seam surface brush, layers, palette,
  bleed) — this is a performance port, behaviour stays the same (modulo the
  geodesic option above, which would be opt-in).

---

## Sequencing & expected payoff

| Phase | Effort | Payoff |
|-------|--------|--------|
| (Tier 1 now) rayon AO/sun bake | hours | near-linear bake speedup, no GPU needed |
| 1 — GPU dab stamping | days | removes ~95 % of paint CPU; 2K interactive |
| 2 — GPU compositing  | days | removes the rest of CPU pixel work; cost ∝ cores→GPU |
| 2.5 — GPU mesh maps (AO/sun) | days | AO at 2K in ms; live sun/shadow slider; unblocks the moat |
| 3 — history/export   | 1–2 days | correctness parity with GPU state |
| 4 — virtual textures | weeks | 4K/8K painting |

Do **Phase 1 first** — it targets the measured paint bottleneck. **Phase 2.5**
(GPU mesh maps) is the most self-contained and can land early, sharing the
BVH-on-GPU work and diffed against a clean CPU reference; it's also the only path
to interactive AO/sun at 2K. Phases 2–3 complete the win and keep undo/export
honest. Phase 4 only if users go past 2K. The rayon bake parallelization is a
no-GPU stopgap worth doing immediately.

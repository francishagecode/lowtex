# GPU paint/display port — handoff (continue in fresh context)

Status as of 2026-06-28. Branch: `main` (uncommitted working tree). All **139 `cargo test` pass**.

## UPDATE 2026-06-28 (session 2 cont'd): the dark "TEETH" at island edges — FIXED

This was the bug the user actually meant by "GPU painting not working": painting in GPU mode left a
**dark sawtooth of unpainted texels along every UV-island edge** on the 3D model; CPU painting was
clean. Verified the user's report by loading their model (`alien-crusade/.../cell holder/cell.lowtex`).

**Root cause — GPU paint UNDER-COVERS vs CPU (not a display bug).** The CPU brush
(`surface::splat` → `rasterize`, inclusive `w>=0` "centre-in-triangle") paints the island-edge
texels; the GPU dab rasterizes the face set via `dab.wgsl` and the **hardware fill rule drops those
boundary texels** (plus sub-texel slivers). Those texels keep the dark base colour, and the gutter
bleed can't fix them (they're *inside* the coverage mask, treated as valid). Measured: 282 texels
CPU-painted-but-GPU-missed (251 full-strength), 0 the other way.

**Why earlier display comparisons missed it:** headless CPU-vs-GPU *renders* of the same stored
texture were pixel-identical (max diff 2). The bug is in *paint coverage*, not display — you only see
it by **diffing the painted textures**, not the renders. Repro that finally worked:
`--open-project P --stroke --brush-color 0,1,0 --save-texture out.png` for CPU vs GPU, then diff the
two PNGs' green texels.

**Fix (`src/gpu_dab.rs`): conservative dab raster.**
- `expand_tri()` grows each face's UV triangle outward by `LOWTEX_DAB_EXPAND` texels (default 0.5)
  so the hardware covers every texel whose centre is in the original triangle.
- **Critical detail:** each expanded vertex's *world* position is recomputed via the **original**
  triangle's affine UV→world map (`bary()` extrapolation), NOT the original vertex world. Reusing the
  vertex world stretches the map and shifts the falloff disc, perturbing interior coverage (caused a
  ~194-texel ring). With the affine-correct world, interior coverage is byte-unchanged and only the
  edge gains.
- Acute corners/slivers: the edge-offset intersection blows up, so its **magnitude** is clamped
  (cap `8e+2`) while keeping the **bisector direction** — collapsing to a centroid push left the last
  corner teeth.

**Verified on the user's mesh:** small (size 8), large (400), and material brushes all 0 missed vs
CPU; a freshly fully-painted region renders 0 dark-teeth-inside-paint, pixel-identical to CPU
(max diff 2). Parity tests updated to assert GPU is a conservative **superset** (under-coverage ≈ 0 =
no teeth; thin over-rim allowed): `gpu_dab_coverage_matches_cpu_splat`,
`gpu_stroke_accumulation_matches_cpu_max`, `gpu_erase_stroke_matches_cpu_stroke`,
`gpu_mask_stroke_matches_cpu_stroke`.

**Pre-fix art cleanup — SHIPPED (region-aware).** Teeth baked into a file *before* this fix are
unpainted island-edge texels in the stored layers. `Renderer::clean_seams` (Edit ▸ "Clean Seams"
menu button, or headless `--clean-seams`) repairs them: `bleed::fill_island_rim_teeth` fills each
unpainted **rim** texel from a **same-facet** painted neighbour (`fill_map.texel_facet`). Two
safeguards make it safe where the first naive attempt was not: (1) same-facet only, so it can't smear
an adjacent island's colour across a seam — the naive first-valid-neighbour version added +137 *dark*
texels at red/dark island borders; the region-aware version adds **0** red→dark and fills 181 red
teeth from red on cell.lowtex; (2) rim-bounded (within `pad` of the gutter) so intentional interior
transparency is untouched. Undoable; no-op on fully-opaque layers. Test
`fill_island_rim_teeth_uses_own_facet_color`. New painting never needs this (the dab is conservative);
it just repairs old files without a full repaint. To clean an existing project headlessly:
`./target/release/lowtex --open-project P.lowtex --clean-seams --save-project P.lowtex`.

Genuine multi-colour island *boundaries* (one island's colour meeting another's) are real edges, not
teeth — present in CPU too — and are left as-is.

**Also this session (2026-06-27):**
- **GPU gutter mask was stale after a same-resolution mesh swap** — `gpu_compose` keyed the coverage
  push on `tex_size` only; re-keyed on `(tex_size, topo_version)` (field `gpu_coverage_gen`). Test
  `gpu_bleed_uses_current_mesh_coverage_after_swap`. This was a *real* GPU-worse-than-CPU bug.
- **2D UV-editor live painting** — a UV-editor stroke now keeps the cheap CPU display path live each
  frame (`uv_stroke` flag in `flush_paint`) so the panel updates as you drag (was only refreshing at
  `end_stroke`). Test `gpu_uv_stroke_updates_panel_live_midstroke`.

**KEY LESSON for next time:** when the user says "GPU *painting* is broken," diff the painted
**textures** (paint coverage), not just the display renders.

---

## UPDATE 2026-06-27 (session 2): the "UVs / painting on them" bug is FIXED

Root cause was exactly the leading hypothesis below: under `LOWTEX_GPU_DISPLAY` the CPU display
mirror (`display_buf` / `paint_texture_gpu`) that feeds the **2D UV editor, brush preview and
export** went **stale** during/after a stroke (the in-stroke paths recomposite the GPU atlas the
3D model samples, but skip the CPU composite as wasted work). Fixes landed:

1. **UV-editor / mirror stale → fixed.** New field `stroke_paint_rect` accumulates the whole
   stroke's painted texels (not consumed per-frame like `dirty_rect`). At `end_stroke`, under GPU
   display, `refresh_display_region(stroke_paint_rect)` re-syncs just that region from the
   now-reconciled CPU layers + bumps `paint_version`. Brush-area bounded, once per stroke, off the
   hot path. Test: `gpu_display_buf_synced_after_stroke` (covers resolve = solid + readback = mask).
2. **Brush-preview ghost invisible on the model → fixed.** `set_brush_preview` now mirrors the
   ghost into the GPU atlas too (`GpuLayers::atlas_texture()`), and reverts it from `display_buf`.
   Test: `brush_preview_ghost_visible_in_atlas_under_gpu_display`.
3. **Material GPU resolve re-enabled.** The deferred-bleed regression that forced material onto the
   readback path is gone (resolve path does full compose+bleed every frame), so the tiled material
   brush — the user's brush from the lag report — now takes the no-readback resolve path. Tests:
   `gpu_resolve_material_stroke_matches_cpu_stroke`, `gpu_material_display_matches_cpu`.

Verified end-to-end: headless `--screenshot` CPU vs GPU diff (plain stroke, material brush, `--ui`
panel) all agree within **max abs channel diff = 2** (sRGB rounding), 0 pixels off by >8.

**Still recommended:** the user should confirm live (load their actual mesh, flags on, paint, watch
the 2D panel + hover ghost). Remaining work is perf (dirty-rect bleed) + P6/P7 (flip GPU default,
byte-budget history, remove CPU runtime path) — see "Suggested next steps". The original
description of the bug is kept below for context.

---

## TL;DR for the next session

- The whole GPU pipeline is **opt-in behind two env flags**; with **no flags the app is the
  original, fully-correct CPU path** (the shipping default). To get correct behavior right
  now: run `cargo run --release` with **no** `LOWTEX_*` flags.
- GPU path: `LOWTEX_GPU_DISPLAY=1 LOWTEX_GPU_PAINT=1 cargo run --release`. This is what's broken
  in the user's eyes.
- **Leading hypothesis for the current bug: the 2D UV-editor panel (and anything reading
  `display_buf`) is STALE under `LOWTEX_GPU_DISPLAY`.** See "Leading bug" below. Tests pass
  because they check the *3D model atlas* and the *CPU layer pixels*, not the 2D UV-editor
  mirror.
- Profiler: `LOWTEX_PROFILE=1 RUST_LOG=lowtex=info ./target/release/lowtex` prints `[perf/1s …]`
  lines. (Don't use `RUST_LOG=info` — wgpu floods it.)

## What works vs what the user sees

- **Tests (all green):** GPU composite/quantize/bleed match CPU `composite_display` (±1 / palette
  flips); GPU resolve matches `blend4`/`erase4`; per-layer undo; the *displayed model atlas* after
  a stroke matches CPU including bleed (`gpu_resolve_display_overpaints`,
  `gpu_readback_material_display_matches_cpu`).
- **User report (live app):** painting/UVs "not working." Earlier they confirmed the readback path
  was "nice and smooth" and correct; then the material-resolve change broke the look ("ugly, no
  overpainting"); then "edges missing, need to overpaint slightly" (the gutter bleed — now fixed);
  now "issues with the UVs or painting on them."

## Leading bug: `display_buf` / UV editor stale under GPU display

`atlas_view()` (renderer.rs:3885) hands the **2D UV-editor panel** `self.display_buf` (the CPU
mirror), and `paint_version()` (3890) tells it when to re-upload. Under `LOWTEX_GPU_DISPLAY`:

- The model (3D) samples the **GPU atlas** (`gpu_bind_group` → `gpu_layers` atlas sRGB view),
  which IS updated each stroke (`gpu_compose`).
- But `display_buf` / `paint_texture_gpu` / `paint_version` are **only** updated by
  `refresh_display_texture` / `refresh_display_region` (CPU path). Under GPU display we
  deliberately **skip** those (see `flush_paint`, renderer.rs ~1413, and
  `finish_gpu_resolve_stroke`). So:
  - the **2D UV editor shows a stale image** and doesn't even know to refresh (paint_version not
    bumped), and
  - the hover **brush-preview ghost** writes `paint_texture_gpu` (not the atlas), so it doesn't
    appear on the model.

This is the most likely "UVs / painting on them not working." **Verify first**: with the flags on,
paint and watch the 2D UV-editor panel — is it stale/blank/wrong while the 3D model updates?

### Fix options (pick one, P6 work)
1. Re-route the UV editor to read the **GPU atlas** (read it back lazily into a small CPU buffer
   on `paint_version` change, or have the panel sample the atlas texture directly). Cleanest.
2. Keep `display_buf` updated under GPU display: at `end_stroke` (and after each readback-path
   frame) do a **CPU `composite_display` into `display_buf`** for the painted rect + bump
   `paint_version`. Simpler but re-introduces some CPU cost (off the hot path if only at
   `end_stroke`).
3. Also fix the **brush preview** under GPU display (compose the ghost into the atlas, or read the
   atlas for the ghost). Currently the ghost is invisible on the model under GPU display.

If the UV editor turns out NOT to be the user's complaint, other suspects: picking/UV mapping on
the user's specific mesh (tests use `Mesh::cube()` — try the user's actual model), or the
deferred… (already fixed). Reproduce with the user's mesh + watch both the model and the panel.

## Architecture (GPU path)

Two GPU subsystems, both opt-in:

- **`LOWTEX_GPU_PAINT`** (older, `gpu_dab.rs`): the surface brush stamps dab coverage on the GPU
  (`splat_faces` → `gpu_dab`), into a per-stroke R16F coverage texture (`Max`-blended).
- **`LOWTEX_GPU_DISPLAY`** (new, `gpu_layers.rs`): the layer stack composites + palette-quantizes +
  Bayer-dithers + gutter-bleeds **on the GPU** into a single atlas texture (`Rgba8Unorm` with an
  `Rgba8UnormSrgb` view; the model samples the sRGB view — this preserves the exact sRGB-byte
  math). The model's bind group points at this atlas (`gpu_bind_group`) instead of
  `paint_texture_gpu`.

**Source of truth:** the CPU `Layers` (`layers.rs`) stays authoritative (save/load/undo/merge/
resize/export). The GPU runs the per-frame display + in-stroke paint; the active layer is
reconciled back to CPU at `end_stroke`.

### Two in-stroke paint paths (both under GPU display)
- **Resolve path (no readback)** — solid colour / eraser, active layer effect-free:
  `gpu_stroke_resolve=true` (set in `gpu_surface_dab`). Per frame (`resolve_gpu_stroke_frame`):
  coverage resolves straight into the active GPU layer slice (`resolve.wgsl` over a pre-stroke
  `stroke_base` copy), then **full `gpu_compose`** (composite + bleed) every frame. At
  `end_stroke` (`finish_gpu_resolve_stroke`): final resolve + compose + **read back only the
  painted rect** into the CPU layer.
- **Readback path** — everything else (mask, **tiled material brush (current)**, effected active
  layer, decal stamp): `pump_gpu_stroke` reads the coverage back (pipelined, 1-frame latency) and
  CPU-blends it into the layer (`apply_gpu_coverage`/`apply_coverage`), then `flush_paint` does
  `upload_active` + `gpu_compose` (GPU display; the CPU `refresh_display_region` is skipped under
  GPU display).

The **material GPU resolve** (`resolve.wgsl` mode 2, `ResolveKind::Material`, `set_material`) exists
but is **currently DISABLED** (material stays on the readback path) — it caused the "edges missing"
regression via a since-removed deferred-bleed optimization. Re-enable by dropping
`brush_material.is_none()` from the resolve eligibility in `gpu_surface_dab` and setting
`gpu_stroke_material` (see git history of that function) — but verify the display first.

## File map

New:
- `src/gpu_layers.rs` — GPU layer arrays + composite/bleed/resolve pipelines, atlas, material,
  `upload`/`upload_active`/`composite`/`composite_region`/`bleed`/`resolve`/`resolve_active`/
  `begin_stroke_resolve`/`read_layer_rect`/`read_atlas`(test)/`set_quantize`/`set_coverage`/
  `set_material`. Parity tests at the bottom.
- `src/shaders/composite.wgsl` (composite + palette quantize + Bayer), `bleed.wgsl` (ping-pong
  gutter dilation), `resolve.wgsl` (paint resolve: solid/erase/material), `dab.wgsl` (per-vertex
  dab params — batched), `bake.wgsl` (GPU AO/sun, separate older work).
- `src/gpu_dab.rs`, `src/gpu_bake.rs` — GPU dab stamping + GPU mesh-map bake (older Tier-2 work).

Modified (key ones):
- `src/renderer.rs` (+~1470 lines) — all GPU integration. Key fns:
  `begin_stroke` (~3097, per-layer snapshot + gpu prep), `end_stroke` (~3170),
  `flush_paint` (~1413, GPU-display branch), `pump_gpu_stroke`/`resolve_gpu_stroke_frame`/
  `finish_gpu_resolve_stroke`/`finish_gpu_stroke` (~3524+), `apply_gpu_coverage` (~3660),
  `gpu_surface_dab`/`gpu_paint_eligible`/`gpu_resolve_kind`, `update_gpu_display`/
  `gpu_sync_layers`/`gpu_compose` (~1355), `refresh_display_texture`/`refresh_display_region`
  (~1243/1265, CPU path), `atlas_view`/`paint_version` (3885/3890, UV editor),
  `draw_into` (model bind-group selection ~3770), `make_paint_bind_group_view`,
  `force_gpu_display`/`force_gpu_paint`. New struct `StrokeBackup`; fields `gpu_layers`,
  `gpu_display`, `gpu_bind_group`, `gpu_stroke_resolve`, `gpu_stroke_material`, `gpu_material_gen`,
  `gpu_coverage_size`. Bench `bench_stroke_paths` (`#[ignore]`).
- `src/history.rs` — `enum Snapshot { Stack(Layers), Layer{index,tex,mask} }` + swap-based
  `apply_into`; `record(Snapshot)`, `undo/redo(&mut Layers)->bool` (no clones).
- `src/layers.rs` — `Layer::effected` + `enum Effected` made `pub` (GPU upload reads effected px).
- `src/gpu_dab.rs` — per-vertex dab params + per-frame batch (`flush_dabs`), `coverage_view`.
- `src/paint.rs`/`bvh.rs`/`bake.rs`/`surface.rs` — older Tier-1/2 support (splat_faces, GPU BVH).

Plan: `docs/gpu-painting-plan.md` (Tier-2 plan). Approved plan that started this:
`~/.claude/plans/goofy-inventing-cocke.md`. Project memory:
`~/.claude/projects/-home-francis-lowtex/memory/gpu-pixel-pipeline-port.md` (detailed running log).

## Perf findings (validated on the user's hardware)

Profiling a real session (5 layers, material brush) found and fixed:
- `stroke_snapshot` = **280ms** (begin_stroke cloned the whole stack) → **~60ms** via per-layer
  snapshot (`history.rs` `Snapshot::Layer`, `StrokeBackup`).
- CPU `composite_region`/`bleed_region`/`upload_region` (~74ms/frame) ran as **wasted work** under
  GPU display → routed to GPU (`flush_paint` GPU branch). Gone from the profile.
- Bench `LOWTEX_BENCH_SIZE=4096 cargo test --release bench_stroke_paths -- --ignored --nocapture`:
  resolve ~150ms/stroke (10 segs) vs CPU ~1310ms. (Cube is a worst case for scissor; the bench is a
  rough guide.)

Known remaining cost: full bleed every frame is ~15ms/frame at 4K. A **correct** dirty-rect bleed
(bound the seed copies + ping-pong to dab-rect+pad, keeping gutters filled live) is the future
optimization — the earlier attempt broke gutters and was reverted.

## How to run / test / debug

```
# Correct (CPU) baseline:
cargo run --release
# GPU path (the one under test):
LOWTEX_GPU_DISPLAY=1 LOWTEX_GPU_PAINT=1 cargo run --release
# Profiler (app perf only, not wgpu spam):
LOWTEX_GPU_DISPLAY=1 LOWTEX_GPU_PAINT=1 LOWTEX_PROFILE=1 RUST_LOG=lowtex=info ./target/release/lowtex
# Force CPU bake / disable a piece if needed: LOWTEX_CPU_BAKE=1
# Tests + parity:
cargo test --release            # 135 pass; GPU tests use lavapipe, occasionally flake under
                                #   concurrency — re-run (consider a shared OnceLock test device).
cargo test --release gpu_       # all GPU parity tests
```

Useful existing parity tests (in `gpu_layers.rs` / `renderer.rs` test mods):
`gpu_composite_matches_cpu_layers`, `gpu_composite_quantize_matches_cpu`,
`gpu_bleed_matches_cpu_dilate`, `gpu_resolve_matches_cpu_blend`,
`gpu_resolve_display_overpaints`, `gpu_readback_material_display_matches_cpu`,
`gpu_resolve_stroke_matches_cpu_stroke`, `gpu_resolve_material_stroke_matches_cpu_stroke`,
`gpu_tiled_image_stroke_matches_cpu_stroke`, `history::tests::*`.

## Suggested next steps (in order)

1. **Reproduce with the user's actual mesh** (tests use a cube). Load their model, turn flags on,
   paint, and watch **both** the 3D model **and** the 2D UV-editor panel. Note exactly what's wrong
   (panel stale? paint in wrong UV spot? nothing shows? seams?).
2. If it's the **UV-editor panel** (most likely): implement fix option 1 or 2 above (re-route the
   panel to the GPU atlas, or refresh `display_buf` + bump `paint_version` at `end_stroke`).
3. Fix the **brush-preview ghost** under GPU display (invisible today).
4. Only then revisit the material-resolve re-enable and the dirty-rect bleed optimization.
5. Longer term (plan P6/P7): make GPU default, byte-budget history further, remove the CPU runtime
   path (keep CPU as export + parity oracle).

## Quick orientation commands
```
grep -n "fn flush_paint\|fn pump_gpu_stroke\|fn resolve_gpu_stroke_frame\|fn finish_gpu_resolve_stroke\|fn gpu_compose\|fn update_gpu_display\|fn atlas_view\|fn gpu_surface_dab\|fn apply_gpu_coverage\|gpu_stroke_resolve =" src/renderer.rs
```

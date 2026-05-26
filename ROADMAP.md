# lowtex — Roadmap

The full arc from where the code is today (`v0.1`, proof of paint) to the
final goal: **the PSX/low-poly texture painter indie devs reach for first.**

This file is the source of truth for *what to build next*. It is meant to be
worked through one goal at a time (see [How to use this roadmap](#how-to-use-this-roadmap)).

---

## Vision

A focused 3D texture painter for the PS1/low-poly aesthetic. It deliberately
*does not* chase Substance Painter's PBR surface area. Instead it leans into the
constraints of the style (low-res textures, limited palettes, no PBR, affine
warp, vertex wobble) and turns them into features — then steals the few ideas
that make Substance feel magical and applies them at 64×64 with 16 colors.

The differentiator nobody has built: **the mesh tells you where to paint.** Bake
AO/curvature once, then "rust on the edges" or "grime in the crevices" happens
automatically — at PSX resolution, on a limited palette. PS1-era artists did all
of that by hand.

## Design principles

These are the rules every goal is measured against.

1. **Speak plain language, not PBR.** "Shiny," not "roughness." The audience
   comes from 2D pixel art; Substance's vocabulary is an activation barrier.
2. **The feedback loop is sacred.** The PSX look is *live* in the viewport while
   painting — never an export-then-look step. (Open question G9: clean paint view
   vs. wobble-while-painting.)
3. **Non-destructive by default.** Strokes, fills, generators, palettes are
   editable after the fact. Nothing is flattened until export.
4. **The mesh is an input, not just a canvas.** AO/curvature/world-position drive
   masks. This is the moat.
5. **Seamless across UV islands.** Even at 64×64 — chunky seams are *more*
   visible, so island bleed/dilation matters more, not less.
6. **Opinionated export.** One click to a correctly-named, correctly-packed,
   engine-ready result (incl. true indexed PNG for retro pipelines).
7. **Cut hard.** Ship the smallest thing that proves each idea; get it in front
   of the Haunted PS1 / low-poly community early.

## What we are explicitly NOT building (for v1)

PBR (metallic/roughness/normal maps, IBL, tonemapping), HDR, shadow maps,
real-time raytracing, a node-graph material editor (Designer-style), animation.
LSCM/ABF unwrapping and xatlas integration are *deferred*, not core — projection
unwraps match the aesthetic better (see G14–G16).

---

## How to use this roadmap

- Goals have stable IDs (`G1`, `G2`, …). **Never renumber them** — commands and
  notes reference them. Insert new work as `G7a`, `G7b`, etc.
- Work the **lowest-numbered unchecked goal whose dependencies are all done**,
  unless told otherwise.
- Each goal carries everything needed to execute it: **Outcome** (the one-line
  "done when"), **Build** (tasks), **Touches** (files/crates), **Done when**
  (a verifiable check — for this GUI app, usually "run it and observe X"), and
  **Depends on**.
- When a goal is finished: tick its checkbox, tick its sub-tasks, and add a
  one-line note under it (date + what actually shipped / what differed).
- Build/run in this environment:
  ```bash
  export PATH="$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH"
  cargo run --release --manifest-path /Users/francishage/Documents/GitHub/lowtex/Cargo.toml
  ```

---

## Phase 0 — Foundation

### [x] G0 — Proof of paint (v0.1)
**Outcome:** A window renders a textured cube; left-click-drag paints on it.
- Hardcoded box-projected-UV cube, static camera, CPU-side 128² texture,
  Möller–Trumbore ray pick → barycentric UV → brush stamp → full GPU re-upload.
- _2026-05-26: compiles & runs on Rust 1.95 / wgpu 22.1. Fixed the two
  `entry_point: Some(..)` → `entry_point: ".."` errors (wgpu 22 takes `&str`)._

### [x] G0.1 — Project hygiene
**Outcome:** The repo is version-controlled and consistently formatted.
- **Build:** `git init` + `.gitignore` (`/target`, `*.lowtex` scratch, OS junk);
  add `rustfmt.toml` (defaults are fine); run `cargo clippy` and clear warnings;
  recreate the design doc as `DESIGN.md` so it travels with the code.
- **Touches:** new `.gitignore`, `rustfmt.toml`, `DESIGN.md`.
- **Done when:** `git status` is clean after an initial commit; `cargo clippy`
  is warning-free.
- **Depends on:** —
- _2026-05-26: `git init`, `.gitignore`, `rustfmt.toml`, `DESIGN.md` added;
  cleared 2 clippy warnings (clean now). Bonus: added a headless `--screenshot`
  offscreen render mode (`renderer.rs::capture`, `main.rs::run_screenshot`) so
  rendering goals can be verified without a window — the offscreen path shares
  `draw_into` with the window renderer._

---

## Phase 1 — From demo to usable tool (v0.2)

Target: **load your own model → orbit to every face → pick color + brush size →
paint → export a PNG.** That single loop is the line between tech demo and tool.

### [x] G1 — Load real meshes (glTF + OBJ)
**Outcome:** lowtex opens a `.gltf`/`.glb`/`.obj` instead of the hardcoded cube.
- **Build:** add a `model` loader that produces our `Vertex`/`index` buffers;
  handle missing normals (compute face/smooth normals); handle missing UVs by
  falling back to box-projection (reuse the cube's logic) so *any* mesh is
  paintable; recenter/normalize scale to fit the camera. Keep `Mesh::cube()` as a
  default/sample.
- **Touches:** `src/mesh.rs` (or new `src/model.rs`); `renderer.rs` (load path);
  **crates:** `gltf`, `tobj` (OBJ).
- **Done when:** running with a path arg loads and renders an external low-poly
  model, and clicking paints on its surface.
- **Depends on:** G0
- _2026-05-26: `src/model.rs` loads glTF (`gltf`) + OBJ (`tobj`); missing normals
  → area-weighted smooth normals, missing UVs → box-projection fallback,
  recenter/normalize to ~1.6u. `Mesh` gained `needs_normals`/`needs_uvs`. Wired
  through `main.rs` (path arg) → `App::new(mesh)` → `Renderer::new(window, mesh)`.
  Verified headless: cube/octahedron.obj/tetra.gltf all load, render, and paint
  (added `--paint` to the screenshot tool). Known limit: shared-vertex meshes get
  degenerate fallback UVs until proper per-tri unwrap (G14)._

### [x] G2 — Orbit camera
**Outcome:** Drag rotates the model, scroll zooms, optional middle-drag pans.
- **Build:** convert `Camera` to an orbit camera (azimuth/elevation/distance
  around `target`); wire RMB/MMB drag + wheel in `app.rs`; **disambiguate paint
  vs. orbit** (e.g. LMB = paint, RMB/MMB = camera, or a modifier). Update the
  view-proj uniform each frame the camera moves.
- **Touches:** `src/camera.rs`, `src/app.rs`, `renderer.rs`.
- **Done when:** you can orbit to the back of the model and paint there.
- **Depends on:** G1
- _2026-05-26: `camera.rs` is now a spherical orbit camera (azimuth/elevation/
  distance around `target`). `app.rs` routes LMB=paint, RMB=orbit, MMB=pan,
  wheel=zoom via a `Drag` state so they never fight. Renderer exposes
  orbit/pan/zoom that refresh the view-proj uniform. Verified headless with
  `--orbit 130 --paint`: view changes and pre-orbit paint persists on the surface._

### [x] G3 — egui UI shell + brush controls
**Outcome:** A side panel with a color picker, brush size, hardness/opacity.
- **Build:** integrate `egui` with the wgpu/winit setup; route events so UI
  clicks don't paint through to the mesh; replace the hardcoded `BRUSH_COLOR`/
  `BRUSH_RADIUS` constants with live state; add color picker, size slider,
  opacity, hardness. Chunky/pixel theme to match the vibe (polish, not blocking).
- **Touches:** `app.rs`, `renderer.rs`, new `src/ui.rs`; **crates:** `egui`,
  `egui-wgpu`, `egui-winit`.
- **Done when:** changing color/size in the panel changes the next stroke; UI
  hover doesn't paint.
- **Depends on:** G1
- _2026-05-26: egui 0.29 (matched to wgpu 22) integrated. `src/ui.rs` has a right
  side panel — Color picker + Size/Opacity/Hardness sliders (plain-language per
  principle #1) — driving a `paint::Brush`. `app.rs` feeds events to egui first
  and gates drag-start on `consumed` so panel clicks don't paint through.
  `renderer::render` takes `UiPaint` and draws egui in a second (load) pass via
  `forget_lifetime`. `paint::Texture::stamp` now blends by opacity + hardness
  falloff (per-stamp; per-stroke accumulation is G6). Verified headless
  (`--ui --paint --brush-color --brush-size`): panel renders, color/size change
  the stroke. Verified live in the window too._

### [x] G4 — Texture import/export (PNG)
**Outcome:** Save the painted texture to PNG; optionally load a starting texture.
- **Build:** "Save PNG" writes the CPU texture; "Open texture" loads one into the
  buffer (resampled to `TEX_SIZE` if needed). Configurable texture resolution
  (64/128/256). Watch sRGB: the GPU texture is `Rgba8UnormSrgb`, so encode PNG
  consistently.
- **Touches:** `renderer.rs`/`paint.rs`, `ui.rs`; **crates:** `image`, `rfd`
  (native file dialog).
- **Done when:** paint → Save PNG → reopen the file in an image viewer shows the
  same result; reloading it back into lowtex round-trips.
- **Depends on:** G3
- _2026-05-26: `rfd` native dialogs. UI "Texture" section: Resolution combo
  (64/128/256), Open…, Save PNG…. Renderer gained `save_texture_png`,
  `load_texture_png` (resampled to current res), `set_texture_resolution`, and
  nearest-neighbour `Texture::resampled`. CPU texture holds sRGB bytes
  (`Rgba8UnormSrgb`) written straight to PNG, so it round-trips. UI actions drain
  via `UiState::actions` handled outside the egui closure (`App::handle_ui_actions`).
  Verified headless: paint→save→load reproduces the paint on a fresh cube; 64²
  renders chunkier._

### [x] G5 — BVH-accelerated picking
**Outcome:** Picking is fast on real (thousands-of-tris) meshes.
- **Build:** build a BVH over triangles once at load; replace the O(all-tris)
  loop in `pick_uv`. Rebuild on mesh change only.
- **Touches:** `paint.rs`; **crates:** `bvh` (or hand-rolled).
- **Done when:** picking a 5k-tri mesh has no perceptible lag on click-drag.
- **Depends on:** G1 (value appears once meshes are non-trivial)
- _2026-05-26: `src/bvh.rs` — hand-rolled midpoint-split BVH (LEAF_MAX=4), built
  once in `Renderer::build`, used by `paint_at`. Padded-slab traversal (epsilon
  pad handles flat meshes / boundary-aligned rays; zero-dir components nudged).
  Tests: `bvh_matches_brute_force` (UVs match the brute-force oracle) and
  `bvh_faster_than_brute_force` (5000 tris, 2000 picks: ~190× faster, 63ms→0.3ms,
  identical hit set)._

### [x] G6 — Stroke engine
**Outcome:** Drags paint continuous strokes, not gappy dots.
- **Build:** interpolate brush stamps between consecutive mouse samples by
  spacing (% of brush size); soft falloff by hardness; opacity accumulation per
  stroke (not per stamp, so overlap within one stroke doesn't double-darken).
  Stylus pressure later (stretch).
- **Touches:** `paint.rs`, `app.rs`.
- **Done when:** a fast drag across the model leaves a solid, even line.
- **Depends on:** G3
- _2026-05-26: `Renderer::begin_stroke`/`paint_segment`/`end_stroke`.
  `paint_segment` interpolates ~2px screen steps between mouse samples (capped at
  1024), re-picking at each so strokes follow the surface across faces.
  `Texture::stamp_stroke` accumulates per-texel coverage as a *max* against a
  stroke-start snapshot, so overlap tops out at the brush opacity (no
  double-darkening). app.rs drives begin/segment/end on LMB. Verified headless
  (`--stroke`): a single big diagonal jump renders a solid even line; at
  `--brush-opacity 0.4` the heavily-overlapping band stays uniform 40%._

---

## Phase 2 — The PSX look (v0.3)

Target: the viewport *is* the vibe. This is the hook that makes someone choose
lowtex over a 2D editor.

### [~] G7 — PSX render shader *(implemented then removed — descoped)*
**Outcome:** ~~Toggleable affine-UV warp, vertex snap/jitter, optional fog.~~
Descoped 2026-05-26: the look comes from the texture, not screen-space effects.
- **Build:** affine (perspective-incorrect) UV interpolation — emit UVs without
  perspective divide to get the warp; vertex position snapping to a low-res grid
  in clip/screen space (the wobble); nearest sampling is already in place; flat /
  Gouraud vertex lighting; optional depth fog. Expose toggles + grid resolution.
- **Touches:** `src/shaders/main.wgsl`, `renderer.rs`, `ui.rs`.
- **Done when:** toggling "PSX mode" visibly warps textures and wobbles vertices
  in real time.
- **Depends on:** G2
- _2026-05-26: Implemented (affine warp, vertex wobble, flat shading, fog as
  runtime-toggleable uniform flags) **then removed the same day** per a design
  decision: the PSX/low-poly look should be driven by the **texture** (low-res +
  limited palette + dither) and nearest-neighbor sampling, not screen-space
  warp/wobble. main.wgsl reverted to a clean perspective-correct shader. See G9._

### [x] G8 — Palette system
**Outcome:** Constrained palettes with quantize + dithering, generatable from images.
- **Build:** palette as an ordered color list (16/32/256); quantize post-process
  (nearest palette color) as a final pass; ordered (Bayer) dithering brush/mode;
  generate a palette from a loaded image (median-cut or k-means); swap palettes
  live; show the active palette in the UI.
- **Touches:** new `src/palette.rs`, post-process pass in `renderer.rs`,
  `main.wgsl`, `ui.rs`; **crates:** `palette` (optional, for perceptual spaces).
- **Done when:** painting with a 16-color palette + dither looks like a PS1
  texture; swapping palettes recolors the view instantly.
- **Depends on:** G7
- _2026-05-26: `src/palette.rs` (built-ins PICO-8/Game Boy/Grayscale + median-cut
  `from_image_median_cut`). **Reworked same day to a texture-space effect** (per
  the G7/G9 pivot): `Palette::quantize_rgba` applies nearest-palette + 4×4 Bayer
  dither to the paint texture on the CPU, non-destructively (full-color
  `paint_texture_cpu` preserved; `refresh_display_texture` uploads the quantized
  result; `display_pixels` used for export). Originally a fullscreen post pass —
  removed. `PaletteSettings`/active `Palette` in renderer; UI Quantize/Dither,
  swatch row, built-ins, "From image…". Tests: median-cut separates R/G/B/W.
  Verified headless: the texture posterizes to PICO-8 with dither; background
  stays un-quantized (proving it's texture-space); export PNG matches the model._

### [x] G9 — Paint-view vs PSX-preview UX *(design decision)*
**Outcome:** A resolved, deliberate answer to "do you paint in the wobble?"
- **Build:** implement the chosen model — most likely a clean editing view + a
  live PSX preview (toggle or split/secondary view). Decide and document in
  `DESIGN.md`. This is a UX call that shapes the whole tool; settle it before
  layers add complexity.
- **Touches:** `ui.rs`, `renderer.rs`, `DESIGN.md`.
- **Done when:** the decision is implemented and written down with rationale.
- **Depends on:** G7, G8
- _2026-05-26: **Resolved — there is no wobble to paint in.** PSX screen-space
  rendering (G7) was removed; the look is driven entirely by the texture (low
  resolution + limited palette + dither) and nearest-neighbor sampling. The
  viewport always shows the final, clean, perspective-correct result, so painting
  is precise and WYSIWYG. Palette quantize is applied to the texture
  non-destructively (toggle on/off without losing the full-color paint), and the
  exported PNG is exactly what's on the model. Rationale documented in DESIGN.md._

---

## Phase 3 — Layers & non-destructive editing (v0.4)

Target: a Photoshop-style stack — the principle that makes iteration fearless.

### [ ] G10 — Layer stack + blend modes
**Outcome:** Multiple layers (albedo + emissive channels), opacity, blend modes,
visibility, reorder; composited on the GPU.
- **Build:** layer = set of GPU textures; compositor pass blends bottom-up
  (normal/multiply/add/screen at least); palette-quantize sits at the very bottom
  of the stack (per principle). Painting targets the active layer.
- **Touches:** new `src/layers.rs`, big changes to `renderer.rs`, `main.wgsl`,
  `ui.rs`.
- **Done when:** painting on layer 2 above a layer-1 fill composites correctly;
  hiding a layer removes it from the view.
- **Depends on:** G6, G8

### [ ] G11 — Layer masks
**Outcome:** Each layer has a paintable mask that reveals/hides it.
- **Build:** mask = single-channel texture per layer; brush can target mask or
  color; masks feed the compositor.
- **Touches:** `layers.rs`, `paint.rs`, `main.wgsl`, `ui.rs`.
- **Done when:** painting black into a layer's mask hides those texels; white
  reveals them.
- **Depends on:** G10

### [ ] G12 — Move painting & compositing to the GPU
**Outcome:** Brush stamps and compositing run on the GPU; CPU is no longer the
source of truth for paint.
- **Build:** brush stamp as a compute (or render-to-texture) pass writing
  directly into the layer texture; keep a CPU mirror only for export. Removes the
  full-texture re-upload per stamp (the current `upload_texture` path).
- **Touches:** `paint.rs`, `renderer.rs`, new compute shader.
- **Done when:** painting on a 1024² texture with a large brush stays smooth.
- **Depends on:** G10

### [ ] G13 — Undo/redo
**Outcome:** Ctrl-Z/Ctrl-Y across strokes and layer ops.
- **Build:** command/history stack; for strokes, store affected-region deltas
  (tiles), not whole-texture snapshots, to bound memory.
- **Touches:** new `src/history.rs`, `app.rs`, `layers.rs`, `paint.rs`.
- **Done when:** a stroke, a layer add, and a fill can each be undone and redone.
- **Depends on:** G10

---

## Phase 4 — Unwrapping (v0.5)

Target: most downloaded/hand-modeled low-poly assets have bad or no UVs.
lowtex should unwrap them in-style. Projection methods fit PSX better than LSCM.

### [ ] G14 — Box-projection unwrap
**Outcome:** "Box Unwrap" assigns UVs by each triangle's dominant axis (≤6 charts).
- **Build:** per-triangle dominant normal axis → planar project → 2×3 grid (this
  is exactly the cube's existing scheme, generalized).
- **Touches:** new `src/unwrap.rs`, `ui.rs`.
- **Done when:** an unwrapped imported mesh paints with predictable per-face UVs.
- **Depends on:** G1

### [ ] G15 — Smart-projection unwrap *(default)*
**Outcome:** Faces clustered by normal similarity, each cluster planar-projected.
- **Build:** greedy normal clustering (angle threshold, à la Blender Smart UV
  Project) → planar project each cluster → hand charts to the packer (G17).
- **Touches:** `unwrap.rs`, `ui.rs`.
- **Done when:** a curved-ish low-poly mesh unwraps into sensible islands with
  less stretch than box projection.
- **Depends on:** G14, G17

### [ ] G16 — Per-face unwrap
**Outcome:** "Per-Face Unwrap" gives every triangle its own island.
- **Build:** trivial chart-per-triangle; useful for "texture each face" workflows
  and as a zero-seam-bleed mode.
- **Touches:** `unwrap.rs`, `ui.rs`.
- **Done when:** selecting it produces one island per triangle in the packed atlas.
- **Depends on:** G14, G17

### [ ] G17 — Chart packing
**Outcome:** 2D charts arranged in the 0–1 atlas without overlap.
- **Build:** sort-by-area first-fit with 90° rotation; **snap chart positions/
  sizes to 8px / power-of-two** boundaries for PSX-correct UVs; optional final
  snap of all UVs to a 256× integer grid.
- **Touches:** `unwrap.rs`; **crates:** `rectangle-pack` or `crunch`.
- **Done when:** packed charts don't overlap and fill the atlas reasonably.
- **Depends on:** G14

### [ ] G18 — Seam-aware painting + island bleed
**Outcome:** Strokes across UV-island boundaries don't leave visible seams.
- **Build:** dilate/bleed painted texels outward past island edges into the gutter
  so nearest-neighbor sampling can't reveal background; ideally paint in 3D space
  so a stroke crossing a seam writes both sides.
- **Touches:** `paint.rs`/compute, `unwrap.rs`.
- **Done when:** a stroke crossing a seam shows no gap on the rendered mesh.
- **Depends on:** G12, G17

---

## Phase 5 — The moat: mesh-aware generators (v0.6)

Target: the unique value. "Rust on the edges," automatically, at 64×64.

### [ ] G19 — Bake mesh maps
**Outcome:** Per-mesh AO, curvature, world-space normal, position, thickness maps.
- **Build:** rasterize each map into UV space; AO via hemisphere sampling against
  the BVH (G5); curvature from dihedral angles / normal divergence. Bake once,
  cache. Add a cheap vertex-AO fallback for speed.
- **Touches:** new `src/bake.rs`, reuse BVH from G5.
- **Done when:** baked AO/curvature maps visibly match the geometry (crevices
  dark, edges high-curvature).
- **Depends on:** G5, G17 (needs UVs to bake into)

### [ ] G20 — Generator / mask system
**Outcome:** Procedural masks driven by mesh maps (curvature, AO, world-Y) ×
noise, used as layer masks.
- **Build:** generators output a mask from mesh-map inputs with thresholds/curves;
  e.g. "edge wear" = curvature mask × noise; "dirt" = AO mask; "snow on top" =
  world-Y mask. Plug into the layer-mask slot (G11).
- **Touches:** new `src/generators.rs`, `layers.rs`, `ui.rs`.
- **Done when:** adding an "edge wear" generator to a layer puts paint on the
  mesh's actual edges with zero hand-painting.
- **Depends on:** G11, G19, G22

### [ ] G21 — Smart palettes / preset looks
**Outcome:** A draggable preset (layer stack + generators + palette) that adapts
to any mesh via its baked maps.
- **Build:** serialize a layer stack + generators + palette as a reusable asset;
  applying it re-evaluates against the current mesh's maps. ("Rusty Metal PSX",
  "Mossy Stone PSX".) This is the shareable-ecosystem hook.
- **Touches:** `generators.rs`, `layers.rs`, serialization (shared with G24).
- **Done when:** applying a saved preset to a *different* mesh produces a coherent
  result that follows the new geometry.
- **Depends on:** G20

### [ ] G22 — Procedural noise library
**Outcome:** Value/Perlin/Worley noise available to generators and as brushes.
- **Build:** noise sampled in UV or world space; parameters (scale, octaves);
  used to break up masks and as dither/texture brushes.
- **Touches:** new `src/noise.rs`; **crates:** `noise` (or hand-rolled).
- **Done when:** a generator mask modulated by noise looks non-uniform/organic.
- **Depends on:** —

---

## Phase 6 — Ship it (v1.0)

Target: an indie dev can adopt lowtex, finish a model, and get it into their engine.

### [ ] G23 — Export presets
**Outcome:** One click to engine-ready output, including true indexed PNG.
- **Build:** presets for Unity/Unreal/Godot/glTF (correct names, packed channels
  e.g. albedo + emissive, power-of-two sizing, nearest/no-mip flags); **indexed
  PNG** with a real palette for retro pipelines; integer-UV snap option on export.
- **Touches:** new `src/export.rs`, `ui.rs`; **crates:** `image`.
- **Done when:** exporting drops correctly-named files that import into a target
  engine with the intended look (no filtering, right channels).
- **Depends on:** G10, G8

### [ ] G24 — Project save/load (`.lowtex`)
**Outcome:** Reopen a project with mesh reference, layers, palette, generators intact.
- **Build:** serialize project state (mesh path + transform, layer stack, masks,
  palette, generators, baked-map cache refs). Versioned format.
- **Touches:** new `src/project.rs`; **crates:** `serde`, `ron`/`serde_json`.
- **Done when:** save → quit → reopen restores the exact editing state.
- **Depends on:** G10, G20

### [ ] G25 — Robustness & performance pass
**Outcome:** lowtex doesn't fall over on real-world input.
- **Build:** graceful errors for malformed/huge meshes and missing UVs; bounds on
  texture/mesh size; profile paint/composite/bake hot paths; sanity-cap memory.
- **Touches:** across the codebase.
- **Done when:** a stress set of community low-poly assets all load, paint, and
  export without crashes or stalls.
- **Depends on:** G18, G20

### [ ] G26 — Onboarding & docs
**Outcome:** A new user reaches load→paint→export in under a minute.
- **Build:** sample meshes + starter palettes bundled; README/quickstart; a short
  "60-second first texture" walkthrough; tooltips in plain language (principle 1).
- **Touches:** `README.md`, `assets/`, `docs/`.
- **Done when:** someone unfamiliar follows the quickstart and exports a texture
  without help.
- **Depends on:** G23

### [ ] G27 — Distribution
**Outcome:** Downloadable builds for macOS/Windows/Linux.
- **Build:** release builds + bundling per platform; tagged releases; consider an
  itch.io page aimed at the Haunted PS1 / low-poly communities.
- **Touches:** CI config, packaging.
- **Done when:** a non-developer can download and run lowtex on their OS.
- **Depends on:** G25, G26

---

## Phase 7 — Beyond v1 (backlog, unordered)

Stylus pressure & tilt · symmetry/mirror painting · stamp & decal brushes ·
dynamic strokes (warp brush along curvature) · node-graph generator editor ·
LSCM/ABF + optional `xatlas-rs` for organic meshes · vertex-color painting ·
animated-texture / UV-scroll support · community palette & smart-material sharing ·
multi-material / texture-set support.

---

## Open design questions (resolve as we hit them)

- **G9:** Paint in the PSX wobble, or clean view + live preview? (Leaning clean +
  preview.) Blocks how the viewport is structured.
- **Texture resolution model:** fixed per project, or resolution-independent
  re-evaluation like Substance? (Affects layers/generators architecture — decide
  before G10.)
- **3D vs UV-space painting:** screen-space projection (simpler, Substance-like)
  vs. UV-space rasterization (better at seams). Likely projection first, UV-space
  for fixups. (Affects G6/G18.)
- **Multi-material meshes:** one texture set in v1, or many? (Affects G1/G10/G23.)

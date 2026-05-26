# lowtex — Design

This is the design doc that travels with the code. The *what to build next* lives
in [ROADMAP.md](ROADMAP.md); this file is the *why* and the *how it fits together*.

## Vision

A focused 3D texture painter for the PS1 / low-poly aesthetic. It deliberately
does **not** chase Substance Painter's PBR surface area. Instead it leans into the
constraints of the style — low-res textures, limited palettes, no PBR, affine
texture warp, vertex wobble — and turns them into features. Then it steals the few
ideas that make Substance feel magical and applies them at 64×64 with 16 colors.

The differentiator nobody has built: **the mesh tells you where to paint.** Bake
AO / curvature once, then "rust on the edges" or "grime in the crevices" happens
automatically — at PSX resolution, on a limited palette. PS1-era artists did all of
that by hand.

## Design principles

1. **Speak plain language, not PBR.** "Shiny," not "roughness." The audience comes
   from 2D pixel art; Substance's vocabulary is an activation barrier.
2. **The feedback loop is sacred.** The PSX look is *live* in the viewport while
   painting — never an export-then-look step.
3. **Non-destructive by default.** Strokes, fills, generators, palettes are editable
   after the fact. Nothing is flattened until export.
4. **The mesh is an input, not just a canvas.** AO / curvature / world-position drive
   masks. This is the moat.
5. **Seamless across UV islands.** Even at 64×64 — chunky seams are *more* visible,
   so island bleed / dilation matters more, not less.
6. **Opinionated export.** One click to a correctly-named, correctly-packed,
   engine-ready result (incl. true indexed PNG for retro pipelines).
7. **Cut hard.** Ship the smallest thing that proves each idea.

## What we are NOT building (for v1)

PBR (metallic/roughness/normal maps, IBL, tonemapping), HDR, shadow maps, real-time
raytracing, a node-graph material editor, animation. LSCM/ABF unwrapping and xatlas
integration are *deferred* — projection unwraps match the aesthetic better.

## Architecture

lowtex is a single native binary: `winit` window → `wgpu` renderer → `egui` overlay.

```
src/
  main.rs       entry point, event loop kickoff, CLI arg parsing
  app.rs        winit ApplicationHandler; owns window + input state; routes events
  camera.rs     orbit camera (azimuth / elevation / distance around a target)
  mesh.rs       Vertex layout + procedural cube + bounds/normal helpers
  model.rs      glTF / OBJ loaders → our Vertex/index buffers
  paint.rs      ray/triangle intersection, UV picking, CPU brush stamping
  renderer.rs   wgpu setup, GPU resources, draw + paint plumbing, offscreen capture
  ui.rs         egui panels (brush, palette, layers, generators, export)
  shaders/
    main.wgsl   scene vertex + fragment shaders (incl. PSX effects)
```

Read in that order. `renderer.rs` is the biggest piece because wgpu has a lot of
boilerplate for resource setup.

### The paint loop (v0.1 baseline)

```
mouse pixel
  → Ray::from_screen (unproject through inverse view-projection)
  → pick_uv (Möller–Trumbore vs. triangles → barycentric UV)
  → Texture::paint_brush (CPU-side RGBA stamp)
  → upload to GPU + redraw (nearest-neighbor sampling)
```

CPU is the source of truth for paint in v0.1. This moves to the GPU at G12; the CPU
side then survives only as an export mirror.

### Headless verification (`--screenshot`)

The renderer can target an **offscreen texture** instead of the window surface, copy
it back, and write a PNG. Run:

```bash
cargo run --release -- --screenshot out.png [--mesh path] [--width N --height N]
```

This exists so the rendering pipeline can be verified without a visible window
(CI, automated checks, headless dev). It is not a user-facing feature.

## Resolved design questions

(Open questions live in ROADMAP.md and migrate here once settled.)

- **Headless rendering:** offscreen render-to-texture + readback, gated behind a CLI
  flag. The window path and the offscreen path share the same scene-draw code so a
  screenshot is faithful to what the window shows. *(decided G0.1, 2026-05-26)*

## Open design questions

These shape multiple goals; resolved answers move up to the section above.

- **G9 — Paint in the PSX wobble, or clean view + live preview?** Leaning clean +
  preview. Blocks how the viewport is structured.
- **Texture resolution model:** fixed per project, or resolution-independent
  re-evaluation like Substance? Affects layers/generators architecture — decide
  before G10.
- **3D vs UV-space painting:** screen-space projection (simpler, Substance-like) vs.
  UV-space rasterization (better at seams). Likely projection first, UV-space for
  fixups. Affects G6 / G18.
- **Multi-material meshes:** one texture set in v1, or many? Affects G1 / G10 / G23.
</content>

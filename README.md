# lowtex

A focused **3D texture painter for the PS1 / low-poly aesthetic** — written in
Rust (winit + wgpu + egui). Load a low-poly model, paint straight onto it, and
constrain the result to a limited palette with dithering, the way PS1-era artists
worked. The differentiator: **the mesh tells you where to paint** — bake ambient
occlusion and curvature once, and "grime in the crevices" or "wear on the edges"
happens automatically, at 64×64 with 16 colors.

It deliberately does *not* chase Substance Painter's PBR surface area. It leans
into the constraints of the style and turns them into features. See
[DESIGN.md](DESIGN.md) for the rationale and [ROADMAP.md](ROADMAP.md) for status.

## Run it

You need [Rust](https://rustup.rs/) (recent stable).

```bash
cargo run --release                       # opens the sample cube
cargo run --release -- assets/samples/octahedron.obj   # or your own .gltf/.glb/.obj
```

First build pulls a lot of wgpu deps (a couple of minutes); after that it's
seconds.

## 60-second first texture

1. **Open** a model (top of the panel) or use the cube that's already there.
2. **Unwrap → Box** if the model's UVs are bad or missing (most downloaded
   low-poly assets).
3. Pick a **Color** and a **Size**, then **left-drag** on the model to paint.
4. **Ambient occlusion → Darken (AO)** drops shadow into the crevices;
   **Highlights** brightens the edges — both as their own layers you can dial back.
5. Turn on **Palette → Quantize** and pick **PICO-8** to snap everything to 16
   colors with dithering — the PS1 look.
6. **Export indexed…** writes a true paletted PNG ready for your engine.

## Controls

| Input | Action |
|-------|--------|
| **Left-drag** | Paint (brush) / bucket-fill (with a fill tool selected) |
| **Right-drag** | Orbit the camera |
| **Middle-drag** | Pan |
| **Scroll** | Zoom |
| **Ctrl/⌘ + Z** | Undo |
| **Ctrl/⌘ + Shift + Z** | Redo |

## What's in it

- **Load** glTF / glb / OBJ; missing normals/UVs are synthesized so any mesh is
  paintable. **Unwrap** Box / Smart / Per-face when UVs are bad.
- **Brush** with size / opacity / hardness; gap-free interpolated strokes; bucket
  **fill** (island / object / face).
- **Layers** with blend modes (Normal/Multiply/Add/Screen), opacity, visibility,
  reorder, and a paintable **reveal mask** per layer.
- **Palette** quantize + ordered dithering (PICO-8 / Game Boy / Grayscale, or
  generate one from any image). Applied to the texture, so it's WYSIWYG.
- **Mesh-aware effects**: baked AO + curvature drive **Darken (AO)**,
  **Highlights**, **Dirt**, **Edge-wear**, and "mask from map" — paint that
  follows the geometry with zero hand-work.
- **Materials**: fill a layer with an image (brick, moss, …); mask it by AO for
  material-in-the-crevices.
- **Undo/redo**, **save/load** projects (`.lowtex`), and **export** (true indexed
  PNG, engine-named files).

## Project files & export

- **`.lowtex`** — your whole project (mesh, layers, masks, palette, settings),
  a versioned RON file. Save/Open from the top of the panel.
- **Export** — choose your engine (Unity/Unreal/Godot/glTF) for the right
  filename, then *Export indexed…* for a true paletted PNG (retro pipelines) or
  *Export RGBA…*. Set the texture to **Point/Nearest filtering, no mipmaps** on
  import — the panel reminds you of the exact flag per engine.

## Source layout

```
src/
  main.rs     entry point, CLI, headless --screenshot
  app.rs      winit handler: window, input routing, egui
  camera.rs   orbit camera
  mesh.rs / model.rs   Vertex + cube; glTF/OBJ loaders
  bvh.rs      BVH for fast ray picking
  paint.rs    ray/UV picking + CPU brush stamping
  layers.rs   layer stack + blend + masks + compositor
  palette.rs  palettes + median-cut + CPU quantize/dither
  bake.rs     AO + curvature mesh maps (the "moat")
  fill.rs     UV-island flood fill
  noise.rs    value/Perlin/Worley noise
  unwrap.rs   box / smart / per-face unwrap + chart packing
  material.rs material-texture fill
  export.rs   indexed-PNG + engine export presets
  project.rs  .lowtex save/load
  renderer.rs wgpu setup + GPU resources + plumbing
  ui.rs       egui side panel
  shaders/main.wgsl   scene shader (nearest-sampled)
```

Build/test: `cargo build --release`, `cargo test`. Headless render for checks:
`cargo run --release -- --screenshot out.png [MESH]`.

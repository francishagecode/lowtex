# Lowtex — v0.1 (proof of paint)

The smallest possible version of the PSX-style 3D texture painter: opens a window, renders a cube, lets you click and drag to paint directly on it. That's it.

## What this proves

- Mouse pixel → world-space ray (unproject through inverse view-projection)
- Ray → mesh intersection (Möller–Trumbore against every triangle)
- Hit → UV (barycentric interpolation of vertex UVs)
- UV → texel write (CPU-side brush stamp into an RGBA buffer)
- Texel → screen (re-upload texture to GPU, re-render with nearest-neighbor sampling)

If this loop feels good, the rest of the tool is built on this foundation. If it doesn't, better to find out now.

## Run it

You need [Rust](https://rustup.rs/) installed (any recent stable, 1.75+).

```bash
cargo run --release
```

First build will take a couple of minutes (wgpu has a lot of transitive deps). After that it's seconds.

## Controls

- **Left-click + drag** — paint
- **Close window** — exit

That's all there is in v0.1.

## What you should see

A dark teal background with a white-and-grey checkerboard cube. The checkerboard reveals the box-projection UV layout (each face is one cell of a 2×3 grid). Click on the cube and a red dot appears. Drag and you get a stroke. Each face paints independently — the UV unwrap is per-face, no seam-blending yet.

## File layout

```
src/
  main.rs       — entry point, event loop kickoff
  app.rs        — winit ApplicationHandler, owns window + input state
  camera.rs     — static orbiting camera
  mesh.rs       — Vertex layout + hand-built cube
  paint.rs      — ray/triangle intersection + brush stamping
  renderer.rs   — wgpu setup, GPU resources, draw + paint plumbing
  shaders/
    main.wgsl   — vertex + fragment shaders
```

Read in that order. `main.rs` is 10 lines; `renderer.rs` is the biggest piece because wgpu has a lot of boilerplate for resource setup.

## What's next (v0.2 — minimum viable painter)

In rough priority order:

1. **glTF mesh loading** — replace the hardcoded cube with the `gltf` crate
2. **Orbit camera** — middle-mouse to orbit, scroll to zoom
3. **BVH for ray picking** — O(n) per-triangle test won't scale past a few thousand tris; build a BVH with the `bvh` crate
4. **Smart projection unwrap** — angle-clustered planar projection (the PSX-friendly default from the design doc)
5. **PSX shader effects** — affine UVs, vertex snap, palette quantize as a post-process
6. **GPU brush compute shader** — move the brush stamp to a compute pass so it scales to 1024² textures
7. **egui UI** — brush size slider, color picker, palette picker
8. **PNG export** — save the painted texture to disk

Each of those is a sensible 1–3 day chunk for the next iterations.

## Known limitations in v0.1

- Cube only (no mesh import yet)
- One color, one brush size (constants in `renderer.rs`)
- No undo
- No save/load
- Painting near UV seams writes to one side only (seam dilation is a v0.3 feature)
- Camera is static (no orbit)
- CPU-side texture is re-uploaded in full on every paint stroke — fine at 128² but would thrash at higher resolutions. The compute shader rewrite in v0.2 fixes this.

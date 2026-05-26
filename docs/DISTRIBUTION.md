# Distribution

How lowtex ships to non-developers (G27).

## Builds

`Cargo.toml` sets a lean release profile (LTO, one codegen unit, stripped). A
local release binary:

```bash
cargo build --release        # -> target/release/lowtex (or lowtex.exe)
```

## Continuous integration

- **`.github/workflows/ci.yml`** runs `fmt --check`, `clippy`, `test`, and
  `build` on Linux / macOS / Windows for every push and PR. (Linux installs the
  X11/Wayland/xkbcommon dev packages winit/wgpu link against.)

## Cutting a release

Tag a commit and push it:

```bash
git tag v0.2.0
git push origin v0.2.0
```

**`.github/workflows/release.yml`** then builds for:

- `aarch64-apple-darwin` and `x86_64-apple-darwin` (macOS)
- `x86_64-pc-windows-msvc` (Windows)
- `x86_64-unknown-linux-gnu` (Linux)

packages each as a `.tar.gz` / `.zip` (binary + README + `assets/`), and attaches
them to the GitHub Release for the tag.

## itch.io

The Haunted PS1 / low-poly communities live on itch.io — that's the target
storefront. After a release, upload the per-platform archives there (butler:
`butler push lowtex-<tag>-<target>.zip <user>/lowtex:<channel>`). Mark the channel
per OS (`windows`, `osx`, `linux`) so the itch app installs the right one.

## Not yet automated

- Code signing / notarization (macOS Gatekeeper, Windows SmartScreen). Unsigned
  builds will warn on first launch until this is set up.
- A bundled `.app` / installer — currently a raw binary in an archive.

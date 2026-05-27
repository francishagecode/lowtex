// src/layers.rs
//
// A Photoshop-style layer stack (G10). Each layer is an RGBA8 texture; painting
// targets the active layer, and the stack is composited bottom-up into a single
// image that the renderer quantizes (palette, at the very bottom of the pipeline)
// and uploads. Compositing is on the CPU for now — moved to the GPU at G12.
//
// Alpha encodes paint presence: the bottom layer starts opaque (the base albedo);
// layers added on top start fully transparent and gain alpha where painted.

use std::cell::RefCell;
use std::sync::Arc;

use crate::effects::Effect;
use crate::paint::{TexRect, Texture};

/// How a layer combines with everything beneath it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Normal,
    Multiply,
    Add,
    Screen,
}

impl BlendMode {
    pub const ALL: [BlendMode; 4] = [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Add,
        BlendMode::Screen,
    ];

    pub fn name(self) -> &'static str {
        match self {
            BlendMode::Normal => "Normal",
            BlendMode::Multiply => "Multiply",
            BlendMode::Add => "Add",
            BlendMode::Screen => "Screen",
        }
    }

    /// Combine a source channel with the backdrop channel (both 0..1).
    fn apply(self, dst: f32, src: f32) -> f32 {
        match self {
            BlendMode::Normal => src,
            BlendMode::Multiply => dst * src,
            BlendMode::Add => (dst + src).min(1.0),
            BlendMode::Screen => 1.0 - (1.0 - dst) * (1.0 - src),
        }
    }
}

/// Memoized output of a layer's effect stack, shared by reference-counting so it
/// survives cloning cheaply. Re-running every layer's effects over the whole texture
/// on every composite is what made both painting and undo/redo scale badly with layer
/// count: a stroke (or a restored history snapshot) changes only one layer, yet all
/// layers re-ran their effects. We memoize the effected pixels in an `Arc` and drop it
/// (`invalidate`) only when the layer's `tex` or `effects` change.
///
/// The `Arc` is what makes undo/redo cheap: cloning a layer (a history snapshot, a
/// redo bank) shares the *same* buffer as the live layer until one is edited, so a
/// snapshot costs a refcount bump rather than a copy, and restoring one reuses its
/// cached effects instead of recomputing them. The buffer is never mutated in place —
/// an edit replaces the whole `Arc` — so sharing it across snapshots stays sound.
///
/// `None` means "not computed / stale". The source of truth stays `tex` + `effects`;
/// this is pure derived state, never persisted.
type EffectCache = RefCell<Option<Arc<Vec<u8>>>>;

/// A layer's pixels after its effect stack: borrowed straight from `tex` when no
/// effect is active (the common case, zero-copy) or a shared handle to the memoized
/// effect output.
enum Effected<'a> {
    Raw(&'a [u8]),
    Cached(Arc<Vec<u8>>),
}

impl std::ops::Deref for Effected<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Effected::Raw(p) => p,
            Effected::Cached(a) => a.as_slice(),
        }
    }
}

#[derive(Clone)]
pub struct Layer {
    pub name: String,
    pub tex: Texture,
    /// Single-channel reveal mask (stored as a `Texture`; the compositor reads the
    /// red channel). White (255) fully reveals, black (0) hides — paint into it to
    /// carve a layer away non-destructively (G11).
    pub mask: Texture,
    pub visible: bool,
    pub opacity: f32,
    pub blend: BlendMode,
    /// Non-destructive adjustment stack (G28), applied in order over a scratch copy
    /// of `tex` at composite time. Empty by default; the painted pixels are never
    /// modified, so effects stay live and removable.
    pub effects: Vec<Effect>,
    /// Ordered, de-duplicated tokens for the content-changing ops applied to this
    /// layer (`["AO", "Blur", "Stroke"]`). Drives the auto-name — see `record_op` /
    /// `derived_name`. The *source* leads; modifiers follow in apply order.
    ops: Vec<String>,
    /// True once the user has typed a name: auto-naming backs off and `name` is left
    /// alone. The base layer starts locked ("Base"), as do layers restored from disk.
    locked: bool,
    /// Memoized effect output (see `EffectCache`). Derived from `tex` + `effects`;
    /// `invalidate` is called whenever either changes.
    cache: EffectCache,
}

impl Layer {
    /// A layer wrapping `tex`, with a fully-revealing (white) mask of matching size.
    pub fn new(name: String, tex: Texture, blend: BlendMode, opacity: f32) -> Self {
        let mask = Texture::new(tex.width, tex.height, [255, 255, 255, 255]);
        Self {
            name,
            tex,
            mask,
            visible: true,
            opacity,
            blend,
            effects: Vec::new(),
            ops: Vec::new(),
            locked: false,
            cache: EffectCache::default(),
        }
    }

    /// Build a layer from fully-specified parts (used when loading a project, which
    /// restores mask and effects). The effect cache starts empty.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        name: String,
        tex: Texture,
        mask: Texture,
        visible: bool,
        opacity: f32,
        blend: BlendMode,
        effects: Vec<Effect>,
    ) -> Self {
        Self {
            name,
            tex,
            mask,
            visible,
            opacity,
            blend,
            effects,
            // A restored layer keeps its saved name verbatim: lock it so a later op
            // doesn't rewrite a name the user (or a past session) deliberately set.
            ops: Vec::new(),
            locked: true,
            cache: EffectCache::default(),
        }
    }

    /// Record a content-changing op against this layer for auto-naming, then refresh
    /// the derived name. De-duplicated, so fifty brush dabs add one `Stroke`, and a
    /// no-op on a locked (manually named) layer leaves its name untouched.
    pub fn record_op(&mut self, token: &str) {
        if self.locked || self.ops.iter().any(|t| t == token) {
            return;
        }
        self.ops.push(token.to_string());
        self.name = derived_name(&self.ops);
    }

    /// Set a name by hand and lock auto-naming for this layer from here on.
    pub fn rename(&mut self, name: String) {
        self.name = name;
        self.locked = true;
    }

    /// Drop the memoized effect output. Must be called after any change to `tex` or
    /// `effects` so the next composite recomputes it (and existing history snapshots,
    /// holding their own `Arc`, are unaffected).
    pub fn invalidate(&self) {
        *self.cache.borrow_mut() = None;
    }

    /// The layer's color after running its effect stack. With no (non-identity)
    /// effects this borrows `tex.pixels` directly — no allocation, the common case.
    /// Otherwise it runs the stack once over a copy and memoizes the result in an
    /// `Arc`, handing back a cheap shared handle on later composites until
    /// `invalidate` is called — so a paint drag (or an undo/redo restore) re-runs
    /// effects for the changed layer only, not every layer. The stored `tex.pixels`
    /// are left untouched — effects stay non-destructive.
    fn effected(&self) -> Effected<'_> {
        if self.effects.iter().all(Effect::is_identity) {
            return Effected::Raw(&self.tex.pixels);
        }
        // `let` (not `if let`/condition) so the immutable borrow is released before the
        // mutable borrow below — RefCell would otherwise panic on the overlap.
        let miss = self.cache.borrow().is_none();
        if miss {
            let mut buf = self.tex.pixels.clone();
            for fx in &self.effects {
                fx.apply(&mut buf, self.tex.width, self.tex.height);
            }
            *self.cache.borrow_mut() = Some(Arc::new(buf));
        }
        Effected::Cached(Arc::clone(self.cache.borrow().as_ref().unwrap()))
    }
}

/// The display name for a layer from its op tokens: the source leads, modifiers
/// follow in apply order. Up to two modifiers show in full; beyond that the tail
/// collapses to `+N` so the *source* always stays visible — `AO`, `AO + Blur`,
/// `AO + Blur + Stroke`, then `AO + Blur +3`.
fn derived_name(ops: &[String]) -> String {
    match ops.split_first() {
        None => String::new(),
        Some((lead, [])) => lead.clone(),
        Some((lead, mods)) if mods.len() <= 2 => format!("{lead} + {}", mods.join(" + ")),
        Some((lead, mods)) => format!("{lead} + {} +{}", mods[0], mods.len() - 1),
    }
}

#[derive(Clone)]
pub struct Layers {
    pub layers: Vec<Layer>, // index 0 = bottom
    pub active: usize,
}

impl Layers {
    /// Start with a single opaque base layer holding `base`.
    pub fn new(base: Texture) -> Self {
        let mut base = Layer::new("Base".to_string(), base, BlendMode::Normal, 1.0);
        base.locked = true; // the canvas keeps the name "Base", even once painted on
        Self {
            layers: vec![base],
            active: 0,
        }
    }

    /// Record a content-op token against the active layer (auto-naming).
    pub fn record_active_op(&mut self, token: &str) {
        let a = self.active;
        self.layers[a].record_op(token);
    }

    /// Rename a layer by hand, locking its auto-name.
    pub fn rename(&mut self, index: usize, name: String) {
        if let Some(l) = self.layers.get_mut(index) {
            l.rename(name);
        }
    }

    pub fn size(&self) -> u32 {
        self.layers[0].tex.width
    }

    pub fn active_tex(&self) -> &Texture {
        &self.layers[self.active].tex
    }

    pub fn active_tex_mut(&mut self) -> &mut Texture {
        let a = self.active;
        // The caller is about to mutate the layer's pixels (paint, fill, load), so the
        // memoized effect output is now stale.
        self.layers[a].invalidate();
        &mut self.layers[a].tex
    }

    pub fn active_mask(&self) -> &Texture {
        &self.layers[self.active].mask
    }

    pub fn active_mask_mut(&mut self) -> &mut Texture {
        &mut self.layers[self.active].mask
    }

    /// Add a transparent layer on top and make it active.
    pub fn add_layer(&mut self) {
        let size = self.size();
        let n = self.layers.len() + 1;
        self.layers.push(Layer::new(
            format!("Layer {n}"),
            Texture::new(size, size, [0, 0, 0, 0]),
            BlendMode::Normal,
            1.0,
        ));
        self.active = self.layers.len() - 1;
    }

    /// Push a pre-built layer (e.g. a baked AO/highlight layer) on top and make
    /// it active. `name` seeds the auto-name as the layer's lead (source) token, so
    /// later edits extend it — "AO" then a blur becomes "AO + Blur".
    pub fn push_generated(&mut self, name: String, tex: Texture, blend: BlendMode, opacity: f32) {
        let mut layer = Layer::new(name.clone(), tex, blend, opacity);
        layer.record_op(&name);
        self.layers.push(layer);
        self.active = self.layers.len() - 1;
    }

    /// Remove the active layer (never the last remaining one).
    pub fn remove_active(&mut self) {
        if self.layers.len() <= 1 {
            return;
        }
        self.layers.remove(self.active);
        self.active = self.active.min(self.layers.len() - 1);
    }

    /// Move the active layer up (toward the top) or down in the stack.
    pub fn move_active(&mut self, up: bool) {
        let i = self.active;
        if up && i + 1 < self.layers.len() {
            self.layers.swap(i, i + 1);
            self.active = i + 1;
        } else if !up && i > 0 {
            self.layers.swap(i, i - 1);
            self.active = i - 1;
        }
    }

    /// Resample every layer (and its mask) to a new square resolution.
    pub fn resize(&mut self, n: u32) {
        for l in &mut self.layers {
            l.tex = l.tex.resampled(n, n);
            l.mask = l.mask.resampled(n, n);
            l.invalidate(); // pixels changed
        }
    }

    /// Composite the stack bottom-up into a single RGBA8 image.
    pub fn composite(&self) -> Vec<u8> {
        let size = self.size();
        let count = (size * size) as usize;
        let mut acc = vec![0.0f32; count * 4]; // premultiplied-ish accumulator (0..1)

        for layer in &self.layers {
            if !layer.visible || layer.opacity <= 0.0 {
                continue;
            }
            // The layer's color after its non-destructive effect stack (G28).
            let effected = layer.effected();
            let px: &[u8] = &effected;
            let mask = &layer.mask.pixels;
            for t in 0..count {
                let i = t * 4;
                // Mask (red channel) gates the layer's contribution: 0 hides, 255
                // reveals (G11).
                let m = mask[i] as f32 / 255.0;
                let sa = (px[i + 3] as f32 / 255.0) * layer.opacity * m;
                if sa <= 0.0 {
                    continue;
                }
                for c in 0..3 {
                    let dst = acc[i + c];
                    let src = layer.blend.apply(dst, px[i + c] as f32 / 255.0);
                    acc[i + c] = dst * (1.0 - sa) + src * sa;
                }
                acc[i + 3] = sa + acc[i + 3] * (1.0 - sa);
            }
        }

        acc.iter()
            .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect()
    }

    /// Composite only the texels inside `rect` into `out` (a full-size RGBA8 buffer),
    /// leaving the rest of `out` untouched. Byte-identical to `composite()` on those
    /// texels — the dirty-rectangle counterpart used by the per-stroke refresh so cost
    /// scales with brush area, not texture size. Each layer's effect stack is run once
    /// (over the whole layer, via `effected`), so this is cheap precisely when no
    /// effect is active — the common case while painting.
    pub fn composite_into_region(&self, out: &mut [u8], rect: TexRect) {
        let size = self.size() as usize;
        let (rw, rh) = (rect.width() as usize, rect.height() as usize);
        let mut acc = vec![0.0f32; rw * rh * 4]; // region-sized accumulator

        for layer in &self.layers {
            if !layer.visible || layer.opacity <= 0.0 {
                continue;
            }
            let effected = layer.effected();
            let px: &[u8] = &effected;
            let mask = &layer.mask.pixels;
            for ry in 0..rh {
                let y = rect.y0 as usize + ry;
                for rx in 0..rw {
                    let x = rect.x0 as usize + rx;
                    let i = (y * size + x) * 4;
                    let ai = (ry * rw + rx) * 4;
                    let m = mask[i] as f32 / 255.0;
                    let sa = (px[i + 3] as f32 / 255.0) * layer.opacity * m;
                    if sa <= 0.0 {
                        continue;
                    }
                    for c in 0..3 {
                        let dst = acc[ai + c];
                        let src = layer.blend.apply(dst, px[i + c] as f32 / 255.0);
                        acc[ai + c] = dst * (1.0 - sa) + src * sa;
                    }
                    acc[ai + 3] = sa + acc[ai + 3] * (1.0 - sa);
                }
            }
        }

        for ry in 0..rh {
            let y = rect.y0 as usize + ry;
            for rx in 0..rw {
                let x = rect.x0 as usize + rx;
                let oi = (y * size + x) * 4;
                let ai = (ry * rw + rx) * 4;
                for c in 0..4 {
                    out[oi + c] = (acc[ai + c].clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill_active(layers: &mut Layers, rgba: [u8; 4]) {
        layers.active_tex_mut().pixels.copy_from_slice(&rgba);
    }

    fn one_px_base(rgba: [u8; 4]) -> Layers {
        Layers::new(Texture::new(1, 1, rgba))
    }

    #[test]
    fn derived_name_leads_with_source_then_truncates() {
        let s = |xs: &[&str]| derived_name(&xs.iter().map(|x| x.to_string()).collect::<Vec<_>>());
        assert_eq!(s(&["AO"]), "AO");
        assert_eq!(s(&["AO", "Blur"]), "AO + Blur");
        assert_eq!(s(&["AO", "Blur", "Stroke"]), "AO + Blur + Stroke");
        // Beyond two modifiers the tail collapses, keeping the source visible.
        assert_eq!(s(&["AO", "Blur", "Stroke", "Hue", "Warp"]), "AO + Blur +3");
    }

    #[test]
    fn record_op_builds_and_dedups_the_name() {
        let mut l = Layer::new("Layer 2".into(), Texture::new(1, 1, [0, 0, 0, 0]), BlendMode::Normal, 1.0);
        l.record_op("AO");
        assert_eq!(l.name, "AO");
        l.record_op("Blur");
        l.record_op("Stroke");
        assert_eq!(l.name, "AO + Blur + Stroke");
        // A repeated op (many brush dabs) doesn't grow the name.
        l.record_op("Stroke");
        assert_eq!(l.name, "AO + Blur + Stroke");
    }

    #[test]
    fn rename_locks_out_auto_naming() {
        let mut l = Layer::new("AO".into(), Texture::new(1, 1, [0, 0, 0, 0]), BlendMode::Normal, 1.0);
        l.record_op("AO");
        l.rename("My shadows".into());
        l.record_op("Blur"); // ignored — the layer is hand-named now
        assert_eq!(l.name, "My shadows");
    }

    #[test]
    fn base_layer_keeps_its_name_when_painted() {
        let mut l = one_px_base([10, 20, 30, 255]);
        l.record_active_op("Stroke"); // painting the base
        assert_eq!(l.layers[0].name, "Base");
    }

    #[test]
    fn generated_layer_seeds_lead_then_extends() {
        let mut l = one_px_base([0, 0, 0, 255]);
        l.push_generated("AO".into(), Texture::new(1, 1, [0, 0, 0, 0]), BlendMode::Multiply, 1.0);
        assert_eq!(l.layers[1].name, "AO");
        l.record_active_op("Blur");
        l.record_active_op("Stroke");
        assert_eq!(l.layers[1].name, "AO + Blur + Stroke");
    }

    #[test]
    fn normal_blend_over() {
        // Opaque red base, half-alpha blue on top → 50/50.
        let mut l = one_px_base([255, 0, 0, 255]);
        l.add_layer();
        fill_active(&mut l, [0, 0, 255, 128]);
        let out = l.composite();
        assert!((out[0] as i32 - 127).abs() <= 2, "r={}", out[0]);
        assert_eq!(out[1], 0);
        assert!((out[2] as i32 - 128).abs() <= 2, "b={}", out[2]);
        assert_eq!(out[3], 255); // base opaque
    }

    #[test]
    fn multiply_darkens() {
        // Red base × full blue multiply → black (no shared channel).
        let mut l = one_px_base([255, 0, 0, 255]);
        l.add_layer();
        fill_active(&mut l, [0, 0, 255, 255]);
        l.layers[1].blend = BlendMode::Multiply;
        let out = l.composite();
        assert_eq!([out[0], out[1], out[2]], [0, 0, 0]);
    }

    #[test]
    fn black_mask_hides_a_layer() {
        // Opaque green layer over red base, but its mask is black → base shows.
        let mut l = one_px_base([255, 0, 0, 255]);
        l.add_layer();
        fill_active(&mut l, [0, 255, 0, 255]);
        l.active_mask_mut().pixels.copy_from_slice(&[0, 0, 0, 255]); // hide
        assert_eq!(
            [l.composite()[0], l.composite()[1], l.composite()[2]],
            [255, 0, 0]
        );
        // White mask reveals the green again.
        l.active_mask_mut()
            .pixels
            .copy_from_slice(&[255, 255, 255, 255]);
        assert_eq!(
            [l.composite()[0], l.composite()[1], l.composite()[2]],
            [0, 255, 0]
        );
    }

    #[test]
    fn layer_effect_changes_composite_but_not_stored_pixels() {
        // A full-desaturate effect on the base greys the composite, yet the layer's
        // own pixels stay the original color (non-destructive, G28).
        let mut l = one_px_base([200, 40, 40, 255]);
        l.layers[0]
            .effects
            .push(crate::effects::Effect::HueSatLight {
                hue: 0.0,
                sat: -1.0,
                light: 0.0,
            });
        let out = l.composite();
        assert_eq!(out[0], out[1], "composite not greyed");
        assert_eq!(out[1], out[2]);
        // Stored pixels untouched.
        assert_eq!(&l.layers[0].tex.pixels[0..3], &[200, 40, 40]);
    }

    #[test]
    fn composite_into_region_matches_full() {
        // Two layers (Multiply on top) with a varying partial mask + alpha, so blend,
        // mask and over-compositing are all exercised. The region result must be
        // byte-identical to the full composite — recomputing the rect, touching nothing
        // outside it.
        let mut l = Layers::new(Texture::new(8, 8, [200, 50, 25, 255]));
        l.add_layer();
        for (t, px) in l.active_tex_mut().pixels.chunks_exact_mut(4).enumerate() {
            px.copy_from_slice(&[(t * 3) as u8, 100, 255u8.wrapping_sub(t as u8), 180]);
        }
        l.layers[1].blend = BlendMode::Multiply;
        for (t, px) in l.layers[1].mask.pixels.chunks_exact_mut(4).enumerate() {
            let v = ((t * 7) % 256) as u8;
            px.copy_from_slice(&[v, v, v, 255]);
        }

        let full = l.composite();
        let rect = TexRect {
            x0: 2,
            y0: 1,
            x1: 6,
            y1: 7,
        };
        // Start from the full result but corrupt the rect, so the assert proves the
        // region writes it back exactly (and leaves the rest alone).
        let mut out = full.clone();
        for y in rect.y0..rect.y1 {
            for x in rect.x0..rect.x1 {
                let i = ((y * 8 + x) * 4) as usize;
                out[i..i + 4].copy_from_slice(&[1, 2, 3, 4]);
            }
        }
        l.composite_into_region(&mut out, rect);
        assert_eq!(out, full);
    }

    #[test]
    fn hidden_layer_is_skipped() {
        let mut l = one_px_base([200, 100, 50, 255]);
        l.add_layer();
        fill_active(&mut l, [0, 0, 0, 255]);
        l.layers[1].visible = false;
        let out = l.composite();
        assert_eq!([out[0], out[1], out[2]], [200, 100, 50]);
    }

    /// Per-stamp composite cost on a 10-layer stack with blur effects, modelling a
    /// paint drag. "old" = every layer re-runs its effects each frame (the behaviour
    /// before effect memoization); "new" = only the active layer does. Ignored by
    /// default — run for numbers (release matters):
    ///   cargo test --release bench_effect_memoization -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_effect_memoization() {
        use std::time::Instant;

        const SIZE: u32 = 128;
        const LAYERS: usize = 10;
        const ITERS: u32 = 300;

        // Base + 9 layers, each upper layer carrying a (real, non-identity) blur over
        // some painted content so `effected` actually does work.
        let mut l = Layers::new(Texture::new(SIZE, SIZE, [120, 60, 30, 255]));
        for k in 1..LAYERS {
            l.add_layer();
            for (t, px) in l.active_tex_mut().pixels.chunks_exact_mut(4).enumerate() {
                let v = ((t + k * 17) % 256) as u8;
                px.copy_from_slice(&[v, 128, 255 - v, 200]);
            }
            l.layers[k]
                .effects
                .push(crate::effects::Effect::Blur { radius: 3.0 });
            l.layers[k].invalidate();
        }

        let rect = TexRect::from_stamp(64.0, 64.0, 8.0, SIZE).unwrap();
        let mut out = vec![0u8; (SIZE * SIZE * 4) as usize];
        l.composite_into_region(&mut out, rect); // warm caches

        let active = l.layers.len() - 1;
        let t = Instant::now();
        for _ in 0..ITERS {
            l.layers[active].invalidate(); // a stroke dirties only the active layer
            l.composite_into_region(&mut out, rect);
        }
        let new_us = t.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

        let t = Instant::now();
        for _ in 0..ITERS {
            for ly in &l.layers {
                ly.invalidate(); // pre-memoization: every layer recomputes
            }
            l.composite_into_region(&mut out, rect);
        }
        let old_us = t.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

        println!(
            "\n{LAYERS} layers @ {SIZE}², blur on {} of them, per stamp:\n  \
             before (all layers recompute): {old_us:8.1} µs\n  \
             after  (active only, memoized): {new_us:8.1} µs\n  \
             speedup: {:.1}×\n",
            LAYERS - 1,
            old_us / new_us,
        );
    }

    /// Cost of an undo/redo restore (clone a snapshot, recomposite it) on a 10-layer
    /// stack with effects. "before" = the cache was reset on clone, so every restore
    /// recomputed all effects; "after" = the `Arc` cache survives the clone, so the
    /// restored snapshot reuses it. Ignored by default:
    ///   cargo test --release bench_history_restore -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_history_restore() {
        use std::time::Instant;

        const SIZE: u32 = 128;
        const LAYERS: usize = 10;
        const ITERS: u32 = 200;

        let mut l = Layers::new(Texture::new(SIZE, SIZE, [120, 60, 30, 255]));
        for k in 1..LAYERS {
            l.add_layer();
            for (t, px) in l.active_tex_mut().pixels.chunks_exact_mut(4).enumerate() {
                let v = ((t + k * 17) % 256) as u8;
                px.copy_from_slice(&[v, 128, 255 - v, 200]);
            }
            l.layers[k]
                .effects
                .push(crate::effects::Effect::Blur { radius: 3.0 });
            l.layers[k].invalidate();
        }
        let _ = l.composite(); // warm the live stack, as it is before any history op

        let time = |iters: u32, mut f: Box<dyn FnMut()>| {
            let t = Instant::now();
            for _ in 0..iters {
                f();
            }
            t.elapsed().as_secs_f64() * 1e6 / iters as f64
        };

        // Before: cloning reset the cache, so the restored snapshot recomputed every
        // effect (modelled by invalidating the clone before compositing).
        let old_us = time(
            ITERS,
            Box::new(|| {
                let c = l.clone();
                for ly in &c.layers {
                    ly.invalidate();
                }
                std::hint::black_box(c.composite());
            }),
        );
        // After: the Arc cache survives the clone, so the restore reuses it.
        let new_us = time(
            ITERS,
            Box::new(|| {
                let c = l.clone();
                std::hint::black_box(c.composite());
            }),
        );

        println!(
            "\n{LAYERS} layers @ {SIZE}², blur on {} of them, per undo/redo restore:\n  \
             before (cache reset on clone): {old_us:8.1} µs\n  \
             after  (Arc cache shared):     {new_us:8.1} µs\n  \
             speedup: {:.1}×\n",
            LAYERS - 1,
            old_us / new_us,
        );
    }
}

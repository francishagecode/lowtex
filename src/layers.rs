// src/layers.rs
//
// A Photoshop-style layer stack (G10). Each layer is an RGBA8 texture; painting
// targets the active layer, and the stack is composited bottom-up into a single
// image that the renderer quantizes (palette, at the very bottom of the pipeline)
// and uploads. Compositing is on the CPU for now — moved to the GPU at G12.
//
// Alpha encodes paint presence: the bottom layer starts opaque (the base albedo);
// layers added on top start fully transparent and gain alpha where painted.

use crate::paint::Texture;

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

#[derive(Clone)]
pub struct Layer {
    pub name: String,
    pub tex: Texture,
    pub visible: bool,
    pub opacity: f32,
    pub blend: BlendMode,
}

#[derive(Clone)]
pub struct Layers {
    pub layers: Vec<Layer>, // index 0 = bottom
    pub active: usize,
}

impl Layers {
    /// Start with a single opaque base layer holding `base`.
    pub fn new(base: Texture) -> Self {
        Self {
            layers: vec![Layer {
                name: "Base".to_string(),
                tex: base,
                visible: true,
                opacity: 1.0,
                blend: BlendMode::Normal,
            }],
            active: 0,
        }
    }

    pub fn size(&self) -> u32 {
        self.layers[0].tex.width
    }

    pub fn active_tex(&self) -> &Texture {
        &self.layers[self.active].tex
    }

    pub fn active_tex_mut(&mut self) -> &mut Texture {
        &mut self.layers[self.active].tex
    }

    /// Add a transparent layer on top and make it active.
    pub fn add_layer(&mut self) {
        let size = self.size();
        let n = self.layers.len() + 1;
        self.layers.push(Layer {
            name: format!("Layer {n}"),
            tex: Texture::new(size, size, [0, 0, 0, 0]),
            visible: true,
            opacity: 1.0,
            blend: BlendMode::Normal,
        });
        self.active = self.layers.len() - 1;
    }

    /// Push a pre-built layer (e.g. a baked AO/highlight layer) on top and make
    /// it active.
    pub fn push_generated(&mut self, name: String, tex: Texture, blend: BlendMode, opacity: f32) {
        self.layers.push(Layer {
            name,
            tex,
            visible: true,
            opacity,
            blend,
        });
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

    /// Resample every layer to a new square resolution.
    pub fn resize(&mut self, n: u32) {
        for l in &mut self.layers {
            l.tex = l.tex.resampled(n, n);
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
            let px = &layer.tex.pixels;
            for t in 0..count {
                let i = t * 4;
                let sa = (px[i + 3] as f32 / 255.0) * layer.opacity;
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
    fn hidden_layer_is_skipped() {
        let mut l = one_px_base([200, 100, 50, 255]);
        l.add_layer();
        fill_active(&mut l, [0, 0, 0, 255]);
        l.layers[1].visible = false;
        let out = l.composite();
        assert_eq!([out[0], out[1], out[2]], [200, 100, 50]);
    }
}

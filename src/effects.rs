// src/effects.rs
//
// Per-layer, non-destructive image effects (classic paint.net-style adjustments).
// A layer carries an ordered stack of these; the compositor runs them over a
// scratch copy of the layer's pixels just before blending (see `Layer::effected`),
// so the painted pixels themselves are never altered — tweak a slider or remove an
// effect and the original paint is exactly as it was. This matches how masks and
// the palette already work: live, re-evaluated each composite, never baked in.
//
// Everything here is plain CPU image math on the RGBA8 buffer. At PSX texture sizes
// (64²–256²) re-running the stack each composite is cheap; the neighbourhood ops
// (blur, warp) are premultiplied so they stay halo-free. Warp reuses the project's
// `noise` primitives to drive its displacement field.

use crate::noise::{self, NoiseKind, NoiseParams};
use glam::Vec3;

/// One non-destructive adjustment. Parameters are chosen so that the zero value is
/// the identity (no change) — `is_identity` skips those so an effect freshly added
/// (or zeroed out) costs nothing until you move a slider.
#[derive(Clone, Copy, PartialEq)]
pub enum Effect {
    /// HSL shift. `hue` in degrees (−180..180), `sat`/`light` in −1..1 (0 = no
    /// change). sat is a multiplier (−1 → grayscale, +1 → double); light is an
    /// additive offset (−1 → black, +1 → white).
    HueSatLight { hue: f32, sat: f32, light: f32 },
    /// `brightness` −1..1 additive; `contrast` −1..1 as a gain around mid-grey
    /// (−1 → flat grey, +1 → doubled contrast).
    BrightnessContrast { brightness: f32, contrast: f32 },
    /// Gaussian blur, `radius` in texels. Premultiplied by alpha so transparent
    /// regions don't bleed dark halos into the painted edge.
    Blur { radius: f32 },
    /// Domain warp (distortion). Each texel is resampled from a position pushed by a
    /// smooth procedural vector field: `amount` is the peak displacement in texels
    /// (0 = identity), `scale` the field's frequency across the texture (higher =
    /// finer, busier wobble). Premultiplied like blur to stay halo-free at edges.
    Warp { amount: f32, scale: f32 },
}

/// The kinds offered in the "Add effect" menu — `Effect` without its parameters.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum EffectKind {
    HueSatLight,
    BrightnessContrast,
    Blur,
    Warp,
}

impl EffectKind {
    pub const ALL: [EffectKind; 4] = [
        EffectKind::HueSatLight,
        EffectKind::BrightnessContrast,
        EffectKind::Blur,
        EffectKind::Warp,
    ];

    pub fn name(self) -> &'static str {
        match self {
            EffectKind::HueSatLight => "Hue / Saturation",
            EffectKind::BrightnessContrast => "Brightness / Contrast",
            EffectKind::Blur => "Blur",
            EffectKind::Warp => "Warp",
        }
    }

    /// A terse token for auto-naming a layer from the ops applied to it — short
    /// enough to read in a stack row (`AO + Blur + Hue`), unlike the full `name`.
    pub fn token(self) -> &'static str {
        match self {
            EffectKind::HueSatLight => "Hue",
            EffectKind::BrightnessContrast => "Levels",
            EffectKind::Blur => "Blur",
            EffectKind::Warp => "Warp",
        }
    }

    /// A freshly-added effect of this kind, parameters at their identity values.
    pub fn default_effect(self) -> Effect {
        match self {
            EffectKind::HueSatLight => Effect::HueSatLight {
                hue: 0.0,
                sat: 0.0,
                light: 0.0,
            },
            EffectKind::BrightnessContrast => Effect::BrightnessContrast {
                brightness: 0.0,
                contrast: 0.0,
            },
            EffectKind::Blur => Effect::Blur { radius: 2.0 },
            EffectKind::Warp => Effect::Warp {
                amount: 4.0,
                scale: 6.0,
            },
        }
    }
}

impl Effect {
    pub fn name(&self) -> &'static str {
        match self {
            Effect::HueSatLight { .. } => "Hue / Saturation",
            Effect::BrightnessContrast { .. } => "Brightness / Contrast",
            Effect::Blur { .. } => "Blur",
            Effect::Warp { .. } => "Warp",
        }
    }

    /// True when the effect would leave the image untouched, so the compositor can
    /// skip it entirely.
    pub fn is_identity(&self) -> bool {
        match *self {
            Effect::HueSatLight { hue, sat, light } => hue == 0.0 && sat == 0.0 && light == 0.0,
            Effect::BrightnessContrast {
                brightness,
                contrast,
            } => brightness == 0.0 && contrast == 0.0,
            Effect::Blur { radius } => radius < 0.5,
            // Sub-half-texel peak displacement is invisible after resampling.
            Effect::Warp { amount, .. } => amount < 0.5,
        }
    }

    /// Apply this effect in place to an RGBA8 `width`×`height` buffer.
    pub fn apply(&self, pixels: &mut [u8], width: u32, height: u32) {
        if self.is_identity() {
            return;
        }
        match *self {
            Effect::HueSatLight { hue, sat, light } => apply_hsl(pixels, hue, sat, light),
            Effect::BrightnessContrast {
                brightness,
                contrast,
            } => apply_brightness_contrast(pixels, brightness, contrast),
            Effect::Blur { radius } => apply_blur(pixels, width, height, radius),
            Effect::Warp { amount, scale } => apply_warp(pixels, width, height, amount, scale),
        }
    }

    /// How far (in texels) this effect can move a painted change across the layer.
    /// A neighbourhood op spreads a brush stamp outward by this much, so the display
    /// refresh must widen the dirtied region by it; point ops return 0.
    pub fn display_spread(&self) -> u32 {
        match *self {
            Effect::Blur { radius } if radius >= 0.5 => radius.ceil() as u32,
            Effect::Warp { amount, .. } if amount >= 0.5 => amount.ceil() as u32,
            _ => 0,
        }
    }
}

/// HSL shift over every texel. Alpha is left untouched.
fn apply_hsl(pixels: &mut [u8], hue_deg: f32, sat: f32, light: f32) {
    let hue_shift = hue_deg / 360.0;
    for px in pixels.chunks_exact_mut(4) {
        let (mut h, mut s, mut l) = rgb_to_hsl(
            px[0] as f32 / 255.0,
            px[1] as f32 / 255.0,
            px[2] as f32 / 255.0,
        );
        h = (h + hue_shift).rem_euclid(1.0);
        s = (s * (1.0 + sat)).clamp(0.0, 1.0);
        l = (l + light).clamp(0.0, 1.0);
        let (r, g, b) = hsl_to_rgb(h, s, l);
        px[0] = (r * 255.0).round() as u8;
        px[1] = (g * 255.0).round() as u8;
        px[2] = (b * 255.0).round() as u8;
    }
}

/// Brightness (additive) then contrast (gain around 0.5), per channel. Alpha is
/// left untouched.
fn apply_brightness_contrast(pixels: &mut [u8], brightness: f32, contrast: f32) {
    let gain = 1.0 + contrast; // −1 → 0 (flat grey), +1 → 2 (doubled)
    for px in pixels.chunks_exact_mut(4) {
        for c in px.iter_mut().take(3) {
            let mut v = *c as f32 / 255.0;
            v += brightness;
            v = (v - 0.5) * gain + 0.5;
            *c = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
}

/// Separable Gaussian blur, premultiplied by alpha (so transparent texels don't
/// drag the painted edge toward black) with clamp-to-edge sampling.
fn apply_blur(pixels: &mut [u8], width: u32, height: u32, radius: f32) {
    let (w, h) = (width as i32, height as i32);
    if w == 0 || h == 0 {
        return;
    }

    // 1-D Gaussian kernel.
    let sigma = (radius / 2.0).max(0.5);
    let kr = radius.ceil() as i32;
    let mut kernel = Vec::with_capacity((2 * kr + 1) as usize);
    let mut ksum = 0.0;
    for d in -kr..=kr {
        let weight = (-(d * d) as f32 / (2.0 * sigma * sigma)).exp();
        kernel.push(weight);
        ksum += weight;
    }
    for k in &mut kernel {
        *k /= ksum;
    }

    let n = (width * height) as usize;
    // Premultiplied float buffer: rgb scaled by alpha, plus alpha itself (0..1).
    let mut prem = vec![0.0f32; n * 4];
    for (i, px) in pixels.chunks_exact(4).enumerate() {
        let a = px[3] as f32 / 255.0;
        prem[i * 4] = (px[0] as f32 / 255.0) * a;
        prem[i * 4 + 1] = (px[1] as f32 / 255.0) * a;
        prem[i * 4 + 2] = (px[2] as f32 / 255.0) * a;
        prem[i * 4 + 3] = a;
    }

    let sample = |buf: &[f32], x: i32, y: i32| -> [f32; 4] {
        let xi = x.clamp(0, w - 1);
        let yi = y.clamp(0, h - 1);
        let i = ((yi * w + xi) * 4) as usize;
        [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]
    };

    // Horizontal pass → tmp.
    let mut tmp = vec![0.0f32; n * 4];
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 4];
            for (ki, &k) in kernel.iter().enumerate() {
                let s = sample(&prem, x + ki as i32 - kr, y);
                for c in 0..4 {
                    acc[c] += s[c] * k;
                }
            }
            let i = ((y * w + x) * 4) as usize;
            tmp[i..i + 4].copy_from_slice(&acc);
        }
    }

    // Vertical pass → back into prem, then unpremultiply into the output.
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 4];
            for (ki, &k) in kernel.iter().enumerate() {
                let s = sample(&tmp, x, y + ki as i32 - kr);
                for c in 0..4 {
                    acc[c] += s[c] * k;
                }
            }
            let i = ((y * w + x) * 4) as usize;
            let a = acc[3];
            let out = if a > 1e-4 {
                [acc[0] / a, acc[1] / a, acc[2] / a]
            } else {
                [0.0, 0.0, 0.0]
            };
            pixels[i] = (out[0].clamp(0.0, 1.0) * 255.0).round() as u8;
            pixels[i + 1] = (out[1].clamp(0.0, 1.0) * 255.0).round() as u8;
            pixels[i + 2] = (out[2].clamp(0.0, 1.0) * 255.0).round() as u8;
            pixels[i + 3] = (a.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
}

/// Domain warp: resample each texel from a position pushed by a smooth procedural
/// vector field. Two decorrelated Perlin fields (one per axis) supply the x/y
/// displacement, scaled by `amount` (peak texels). Sampling the field in normalized
/// texture space makes `scale` read as "features across the texture" regardless of
/// resolution. The resample is premultiplied bilinear with clamp-to-edge, so warping
/// across an alpha edge doesn't drag dark fringes into the painted region (like blur).
fn apply_warp(pixels: &mut [u8], width: u32, height: u32, amount: f32, scale: f32) {
    let (w, h) = (width as i32, height as i32);
    if w == 0 || h == 0 {
        return;
    }
    let n = (width * height) as usize;

    // Premultiplied float source: rgb scaled by alpha, plus alpha (0..1).
    let mut prem = vec![0.0f32; n * 4];
    for (i, px) in pixels.chunks_exact(4).enumerate() {
        let a = px[3] as f32 / 255.0;
        prem[i * 4] = (px[0] as f32 / 255.0) * a;
        prem[i * 4 + 1] = (px[1] as f32 / 255.0) * a;
        prem[i * 4 + 2] = (px[2] as f32 / 255.0) * a;
        prem[i * 4 + 3] = a;
    }

    // Two seeds → two independent fields, so dx and dy don't move in lockstep. A
    // couple of octaves keeps the flow smooth with a touch of finer detail.
    let dx_params = NoiseParams {
        scale,
        octaves: 2,
        persistence: 0.5,
        seed: 0x5717_3D21,
    };
    let dy_params = NoiseParams {
        seed: 0xA34F_91C7,
        ..dx_params
    };

    // Bilinear fetch from the premultiplied source, clamped to the texture.
    let sample_prem = |x: f32, y: f32| -> [f32; 4] {
        let x = x.clamp(0.0, (w - 1) as f32);
        let y = y.clamp(0.0, (h - 1) as f32);
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);
        let (tx, ty) = (x - x0 as f32, y - y0 as f32);
        let at = |xi: i32, yi: i32, c: usize| prem[((yi * w + xi) * 4) as usize + c];
        let mut out = [0.0f32; 4];
        for (c, o) in out.iter_mut().enumerate() {
            let top = at(x0, y0, c) * (1.0 - tx) + at(x1, y0, c) * tx;
            let bot = at(x0, y1, c) * (1.0 - tx) + at(x1, y1, c) * tx;
            *o = top * (1.0 - ty) + bot * ty;
        }
        out
    };

    for y in 0..h {
        for x in 0..w {
            // Field sampled in 0..1 texture space; z held at 0 (a 2D slice).
            let p = Vec3::new(x as f32 / w as f32, y as f32 / h as f32, 0.0);
            let dx = (noise::sample(NoiseKind::Perlin, p, &dx_params) * 2.0 - 1.0) * amount;
            let dy = (noise::sample(NoiseKind::Perlin, p, &dy_params) * 2.0 - 1.0) * amount;
            let s = sample_prem(x as f32 + dx, y as f32 + dy);
            let a = s[3];
            let out = if a > 1e-4 {
                [s[0] / a, s[1] / a, s[2] / a]
            } else {
                [0.0, 0.0, 0.0]
            };
            let i = ((y * w + x) * 4) as usize;
            pixels[i] = (out[0].clamp(0.0, 1.0) * 255.0).round() as u8;
            pixels[i + 1] = (out[1].clamp(0.0, 1.0) * 255.0).round() as u8;
            pixels[i + 2] = (out[2].clamp(0.0, 1.0) * 255.0).round() as u8;
            pixels[i + 3] = (a.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
}

/// RGB (0..1) → HSL (each 0..1, hue normalized).
fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) / 2.0;
    if (max - min).abs() < 1e-6 {
        return (0.0, 0.0, l); // grey: hue/sat undefined
    }
    let d = max - min;
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };
    let h = if max == r {
        (g - b) / d + if g < b { 6.0 } else { 0.0 }
    } else if max == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    (h / 6.0, s, l)
}

/// HSL (each 0..1, hue normalized) → RGB (0..1).
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s < 1e-6 {
        return (l, l, l);
    }
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    (
        hue_to_rgb(p, q, h + 1.0 / 3.0),
        hue_to_rgb(p, q, h),
        hue_to_rgb(p, q, h - 1.0 / 3.0),
    )
}

fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        p + (q - p) * 6.0 * t
    } else if t < 1.0 / 2.0 {
        q
    } else if t < 2.0 / 3.0 {
        p + (q - p) * (2.0 / 3.0 - t) * 6.0
    } else {
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_effects_are_skipped() {
        assert!(Effect::HueSatLight {
            hue: 0.0,
            sat: 0.0,
            light: 0.0
        }
        .is_identity());
        assert!(Effect::BrightnessContrast {
            brightness: 0.0,
            contrast: 0.0
        }
        .is_identity());
        assert!(Effect::Blur { radius: 0.0 }.is_identity());
        assert!(!Effect::Blur { radius: 2.0 }.is_identity());

        // An identity apply leaves pixels byte-identical.
        let mut px = vec![10, 20, 30, 255, 200, 100, 50, 128];
        let before = px.clone();
        Effect::HueSatLight {
            hue: 0.0,
            sat: 0.0,
            light: 0.0,
        }
        .apply(&mut px, 2, 1);
        assert_eq!(px, before);
    }

    #[test]
    fn full_desaturation_makes_grey() {
        // sat = −1 collapses chroma: R == G == B per texel.
        let mut px = vec![200, 40, 40, 255];
        Effect::HueSatLight {
            hue: 0.0,
            sat: -1.0,
            light: 0.0,
        }
        .apply(&mut px, 1, 1);
        assert_eq!(px[0], px[1]);
        assert_eq!(px[1], px[2]);
        assert_eq!(px[3], 255); // alpha untouched
    }

    #[test]
    fn hue_180_rotates_red_to_cyan() {
        // Pure red rotated 180° lands on cyan (low R, high G/B).
        let mut px = vec![255, 0, 0, 255];
        Effect::HueSatLight {
            hue: 180.0,
            sat: 0.0,
            light: 0.0,
        }
        .apply(&mut px, 1, 1);
        assert!(px[0] < 16, "r={}", px[0]);
        assert!(px[1] > 240 && px[2] > 240, "g={} b={}", px[1], px[2]);
    }

    #[test]
    fn brightness_pushes_toward_white_and_black() {
        let mut up = vec![128, 128, 128, 255];
        Effect::BrightnessContrast {
            brightness: 1.0,
            contrast: 0.0,
        }
        .apply(&mut up, 1, 1);
        assert_eq!(&up[0..3], &[255, 255, 255]);

        let mut down = vec![128, 128, 128, 255];
        Effect::BrightnessContrast {
            brightness: -1.0,
            contrast: 0.0,
        }
        .apply(&mut down, 1, 1);
        assert_eq!(&down[0..3], &[0, 0, 0]);
    }

    #[test]
    fn full_negative_contrast_flattens_to_mid_grey() {
        // gain = 0 maps every channel to 0.5 → 128.
        let mut px = vec![10, 200, 90, 255];
        Effect::BrightnessContrast {
            brightness: 0.0,
            contrast: -1.0,
        }
        .apply(&mut px, 1, 1);
        assert_eq!(&px[0..3], &[128, 128, 128]);
    }

    #[test]
    fn blur_spreads_a_bright_texel_into_neighbours() {
        // 3×3 opaque field, only the centre is white; a blur lifts its neighbours
        // above black and keeps the field fully opaque.
        let size = 3u32;
        let mut px = vec![0u8; (size * size * 4) as usize];
        for i in 0..(size * size) as usize {
            px[i * 4 + 3] = 255; // opaque everywhere
        }
        let c = (size * size / 2) as usize; // centre texel (1,1)
        px[c * 4] = 255;
        px[c * 4 + 1] = 255;
        px[c * 4 + 2] = 255;

        Effect::Blur { radius: 2.0 }.apply(&mut px, size, size);

        let center_brightness = px[c * 4];
        let left = c - 1;
        assert!(px[left * 4] > 0, "neighbour stayed black");
        assert!(
            px[left * 4] < center_brightness,
            "neighbour brighter than centre"
        );
        // Opacity preserved.
        for t in 0..(size * size) as usize {
            assert_eq!(px[t * 4 + 3], 255, "texel {t} lost opacity");
        }
    }

    #[test]
    fn blur_leaves_a_uniform_field_unchanged() {
        let size = 4u32;
        let mut px = vec![0u8; (size * size * 4) as usize];
        for t in 0..(size * size) as usize {
            px[t * 4..t * 4 + 4].copy_from_slice(&[120, 60, 200, 255]);
        }
        let before = px.clone();
        Effect::Blur { radius: 3.0 }.apply(&mut px, size, size);
        // Edge clamping means a flat field reconstructs itself (±1 rounding).
        for (a, b) in px.iter().zip(before.iter()) {
            assert!((*a as i32 - *b as i32).abs() <= 1, "{a} vs {b}");
        }
    }

    #[test]
    fn warp_identity_below_half_texel() {
        assert!(Effect::Warp {
            amount: 0.0,
            scale: 6.0
        }
        .is_identity());
        assert!(!Effect::Warp {
            amount: 4.0,
            scale: 6.0
        }
        .is_identity());

        // A zero-amount warp is skipped and leaves pixels byte-identical.
        let mut px = vec![10, 20, 30, 255, 200, 100, 50, 128];
        let before = px.clone();
        Effect::Warp {
            amount: 0.0,
            scale: 6.0,
        }
        .apply(&mut px, 2, 1);
        assert_eq!(px, before);
    }

    #[test]
    fn warp_displaces_a_patterned_field() {
        // A high-frequency checker is moved by the warp, so at least some texels
        // change; opacity (fully opaque here) is preserved everywhere.
        let size = 16u32;
        let mut px = vec![0u8; (size * size * 4) as usize];
        for y in 0..size {
            for x in 0..size {
                let i = ((y * size + x) * 4) as usize;
                let v = if (x + y) % 2 == 0 { 240 } else { 16 };
                px[i..i + 4].copy_from_slice(&[v, v, v, 255]);
            }
        }
        let before = px.clone();
        Effect::Warp {
            amount: 3.0,
            scale: 6.0,
        }
        .apply(&mut px, size, size);

        assert_ne!(px, before, "warp left the field untouched");
        for t in 0..(size * size) as usize {
            assert_eq!(px[t * 4 + 3], 255, "texel {t} lost opacity");
        }
    }

    #[test]
    fn warp_leaves_a_uniform_field_unchanged() {
        // Displacing a constant field samples the same colour everywhere.
        let size = 8u32;
        let mut px = vec![0u8; (size * size * 4) as usize];
        for t in 0..(size * size) as usize {
            px[t * 4..t * 4 + 4].copy_from_slice(&[120, 60, 200, 255]);
        }
        let before = px.clone();
        Effect::Warp {
            amount: 5.0,
            scale: 8.0,
        }
        .apply(&mut px, size, size);
        for (a, b) in px.iter().zip(before.iter()) {
            assert!((*a as i32 - *b as i32).abs() <= 1, "{a} vs {b}");
        }
    }
}

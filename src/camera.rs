// src/camera.rs
//
// Static orbiting camera looking at the origin. v0.1 doesn't need mouse-orbit yet;
// we just want a clean angle on the cube. Will become an OrbitCamera in v0.2.

use glam::{Mat4, Vec3};

pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_radians: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn new(aspect: f32) -> Self {
        Self {
            eye: Vec3::new(2.5, 2.0, 3.5),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y_radians: 60.0_f32.to_radians(),
            aspect,
            near: 0.1,
            far: 100.0,
        }
    }

    pub fn view_proj(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye, self.target, self.up);
        let proj = Mat4::perspective_rh(self.fov_y_radians, self.aspect, self.near, self.far);
        proj * view
    }
}

// src/camera.rs
//
// Orbit camera: spherical coordinates (azimuth, elevation, distance) around a
// `target` point. RMB/MMB drag orbits/pans, the wheel zooms (wired in app.rs).
// Paint stays on LMB so camera control and painting never fight (see G2).

use glam::{Mat4, Vec3};

/// Clamp elevation just shy of the poles so the view-up never degenerates.
const ELEV_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

pub struct Camera {
    pub target: Vec3,
    pub azimuth: f32,   // radians, around the world Y axis
    pub elevation: f32, // radians, from the XZ plane
    pub distance: f32,
    pub up: Vec3,
    pub fov_y_radians: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn new(aspect: f32) -> Self {
        Self {
            target: Vec3::ZERO,
            // A three-quarter view, matching v0.1's static angle.
            azimuth: 0.7,
            elevation: 0.5,
            distance: 4.2,
            up: Vec3::Y,
            fov_y_radians: 60.0_f32.to_radians(),
            aspect,
            near: 0.1,
            far: 100.0,
        }
    }

    /// World-space eye position derived from the spherical orbit parameters.
    pub fn eye(&self) -> Vec3 {
        let (sa, ca) = self.azimuth.sin_cos();
        let (se, ce) = self.elevation.sin_cos();
        let dir = Vec3::new(ce * sa, se, ce * ca);
        self.target + dir * self.distance
    }

    pub fn view_proj(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye(), self.target, self.up);
        let proj = Mat4::perspective_rh(self.fov_y_radians, self.aspect, self.near, self.far);
        proj * view
    }

    /// View-projection for the orientation compass: the scene camera's rotation
    /// only — looking at the origin from the same direction, but at a fixed
    /// distance through an orthographic projection. So the little XYZ gizmo shows
    /// which way is up/forward without inheriting the scene's zoom, pan, or
    /// perspective. Drawn into its own square corner viewport, so the box is
    /// symmetric (aspect 1).
    pub fn gizmo_view_proj(&self) -> Mat4 {
        let (sa, ca) = self.azimuth.sin_cos();
        let (se, ce) = self.elevation.sin_cos();
        let dir = Vec3::new(ce * sa, se, ce * ca);
        let view = Mat4::look_at_rh(dir * 3.0, Vec3::ZERO, self.up);
        // Half-extent 1.4 fits the unit-length axes with a margin.
        let proj = Mat4::orthographic_rh(-1.4, 1.4, -1.4, 1.4, 0.1, 10.0);
        proj * view
    }

    /// Snap the view so the eye sits along `dir` from the target — i.e. look down
    /// the given world axis at the origin. Used by the clickable compass: clicking
    /// the +X arm calls `look_from(Vec3::X)`, etc. Distance and target are kept;
    /// only the orbit angles change. Inverts `eye()`'s spherical mapping
    /// (`dir = (ce·sa, se, ce·ca)`), so elevation = asin(y), azimuth = atan2(x, z).
    pub fn look_from(&mut self, dir: Vec3) {
        let d = dir.normalize_or_zero();
        if d == Vec3::ZERO {
            return;
        }
        self.elevation = d.y.clamp(-1.0, 1.0).asin().clamp(-ELEV_LIMIT, ELEV_LIMIT);
        self.azimuth = d.x.atan2(d.z);
    }

    /// Orbit by mouse-drag deltas (pixels). Positive dx swings right, dy looks down.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        const SPEED: f32 = 0.006;
        self.orbit_radians(-dx * SPEED, dy * SPEED);
    }

    /// Orbit by explicit angles (radians). Used for programmatic/headless control.
    pub fn orbit_radians(&mut self, d_azimuth: f32, d_elevation: f32) {
        self.azimuth += d_azimuth;
        self.elevation = (self.elevation + d_elevation).clamp(-ELEV_LIMIT, ELEV_LIMIT);
    }

    /// Dolly in/out. `delta` is a signed wheel amount; positive zooms in.
    pub fn zoom(&mut self, delta: f32) {
        const SPEED: f32 = 0.1;
        let factor = (1.0 - delta * SPEED).clamp(0.2, 5.0);
        self.distance = (self.distance * factor).clamp(0.3, 50.0);
    }

    /// Pan the target in the camera's screen plane (pixels).
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let speed = self.distance * 0.0015;
        let forward = (self.target - self.eye()).normalize();
        let right = forward.cross(self.up).normalize();
        let up = right.cross(forward).normalize();
        self.target += (-right * dx + up * dy) * speed;
    }
}

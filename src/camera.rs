// src/camera.rs
//
// Orbit camera: spherical coordinates (azimuth, elevation, distance) around a
// `target` point. The same parameters drive two navigation styles (wired in
// app.rs):
//   - Orbit (Alt+LMB drag / MMB pan / wheel zoom): swing the eye around the
//     target — the model-inspection workflow.
//   - Fly (RMB drag + WASD): a game-engine free camera. `look` turns the view in
//     place (pivots at the eye); `fly` translates eye and target together.
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

    /// Unit vector from the target toward the eye (the spherical orbit direction).
    /// `eye = target + dir·distance`, so its negation is the camera's forward.
    fn dir(&self) -> Vec3 {
        let (sa, ca) = self.azimuth.sin_cos();
        let (se, ce) = self.elevation.sin_cos();
        Vec3::new(ce * sa, se, ce * ca)
    }

    /// World-space eye position derived from the spherical orbit parameters.
    pub fn eye(&self) -> Vec3 {
        self.target + self.dir() * self.distance
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
        let view = Mat4::look_at_rh(self.dir() * 3.0, Vec3::ZERO, self.up);
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

    /// First-person mouse-look (RMB fly): yaw/pitch the view while keeping the eye
    /// fixed, so the camera turns in place rather than swinging around the target
    /// like `orbit`. Same angular deltas + pole clamp as `orbit`; the target is
    /// re-derived to hold the eye still (`target = eye − dir·distance`).
    pub fn look(&mut self, dx: f32, dy: f32) {
        let eye = self.eye();
        self.orbit(dx, dy);
        self.target = eye - self.dir() * self.distance;
    }

    /// Fly the camera through the scene by translating eye and target together.
    /// `f`/`r`/`u` are signed world-space distances along the view forward (with
    /// pitch), the horizontal strafe right, and world-up. Drives WASD/QE while RMB
    /// is held. Moving `target` carries the eye with it, since `dir`/`distance`
    /// are unchanged.
    pub fn fly(&mut self, f: f32, r: f32, u: f32) {
        let forward = -self.dir(); // eye → scene
        let right = forward.cross(self.up).normalize_or_zero();
        self.target += forward * f + right * r + self.up * u;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Mouse-look must turn the view without moving the eye (FPS pivot), unlike
    /// `orbit`, which swings the eye around the target.
    #[test]
    fn look_keeps_eye_fixed_and_turns_view() {
        let mut cam = Camera::new(1.0);
        let eye_before = cam.eye();
        let dir_before = (cam.target - eye_before).normalize();

        cam.look(40.0, 25.0);

        assert!(
            (cam.eye() - eye_before).length() < 1e-4,
            "look moved the eye by {}",
            (cam.eye() - eye_before).length()
        );
        let dir_after = (cam.target - cam.eye()).normalize();
        assert!(
            dir_before.dot(dir_after) < 0.9999,
            "look did not change the view direction"
        );
    }

    /// Flying forward must carry the eye toward where it is looking, keeping the
    /// view direction and distance unchanged (it's a translation, not a rotation).
    #[test]
    fn fly_forward_moves_along_view() {
        let mut cam = Camera::new(1.0);
        let eye_before = cam.eye();
        let forward = (cam.target - eye_before).normalize();
        let dist_before = cam.distance;

        cam.fly(1.0, 0.0, 0.0);

        let moved = cam.eye() - eye_before;
        assert!(
            (moved.normalize() - forward).length() < 1e-4,
            "fly-forward moved off the view axis"
        );
        assert!(
            (moved.length() - 1.0).abs() < 1e-4,
            "unexpected step length"
        );
        assert!(
            (cam.distance - dist_before).abs() < 1e-4,
            "fly changed distance"
        );
    }

    /// Strafing right moves horizontally (no vertical drift) and perpendicular to
    /// the forward axis.
    #[test]
    fn fly_strafe_is_horizontal_and_sideways() {
        let mut cam = Camera::new(1.0);
        let eye_before = cam.eye();
        let forward = (cam.target - eye_before).normalize();

        cam.fly(0.0, 1.0, 0.0);

        let moved = cam.eye() - eye_before;
        assert!(moved.y.abs() < 1e-4, "strafe drifted vertically");
        assert!(
            forward.dot(moved.normalize()).abs() < 1e-4,
            "strafe not perpendicular"
        );
    }
}

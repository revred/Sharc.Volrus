//! Perspective camera for raycasting and rendering.

/// Perspective camera for raycasting and rendering.
pub struct Camera {
    pub eye: [f64; 3],
    pub target: [f64; 3],
    pub up: [f64; 3],
    pub fov_y: f64,
    pub aspect: f64,
    pub near: f64,
    pub far: f64,
}

impl Camera {
    pub fn new(eye: [f64; 3], target: [f64; 3], up: [f64; 3], fov_y: f64, aspect: f64) -> Self {
        Self {
            eye,
            target,
            up,
            fov_y,
            aspect,
            near: 0.1,
            far: 1000.0,
        }
    }

    /// Generate a ray for pixel (u, v) where u,v in [0,1].
    /// Returns (origin, direction) in world space. Direction is NOT normalized.
    pub fn ray_at(&self, u: f64, v: f64) -> ([f64; 3], [f64; 3]) {
        let (forward, right, up) = self.basis();

        let half_h = (self.fov_y * 0.5).tan();
        let half_w = half_h * self.aspect;

        // Map [0,1] to [-1,1]
        let sx = (2.0 * u - 1.0) * half_w;
        let sy = (2.0 * v - 1.0) * half_h;

        let dir = [
            forward[0] + sx * right[0] + sy * up[0],
            forward[1] + sx * right[1] + sy * up[1],
            forward[2] + sx * right[2] + sy * up[2],
        ];

        // Normalize direction
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        let dir_n = [dir[0] / len, dir[1] / len, dir[2] / len];

        (self.eye, dir_n)
    }

    /// View matrix (4x4, world -> camera), row-major.
    pub fn view_matrix(&self) -> [[f64; 4]; 4] {
        let (forward, right, up) = self.basis();

        // View matrix: rotation part transposes the basis, then translates by -eye
        let tx = -(right[0] * self.eye[0] + right[1] * self.eye[1] + right[2] * self.eye[2]);
        let ty = -(up[0] * self.eye[0] + up[1] * self.eye[1] + up[2] * self.eye[2]);
        let tz = -(-forward[0] * self.eye[0] - forward[1] * self.eye[1] - forward[2] * self.eye[2]);

        [
            [right[0], right[1], right[2], tx],
            [up[0], up[1], up[2], ty],
            [-forward[0], -forward[1], -forward[2], tz],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// Projection matrix (4x4, camera -> clip), row-major.
    /// Standard perspective projection (OpenGL-style, depth [-1,1]).
    pub fn projection_matrix(&self) -> [[f64; 4]; 4] {
        let f = 1.0 / (self.fov_y * 0.5).tan();
        let nf = 1.0 / (self.near - self.far);

        [
            [f / self.aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (self.far + self.near) * nf, 2.0 * self.far * self.near * nf],
            [0.0, 0.0, -1.0, 0.0],
        ]
    }

    /// Combined view-projection matrix.
    pub fn view_projection(&self) -> [[f64; 4]; 4] {
        mat4_mul(&self.projection_matrix(), &self.view_matrix())
    }

    /// Compute the camera basis vectors: (forward, right, up) all unit length.
    fn basis(&self) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let forward = normalize3([
            self.target[0] - self.eye[0],
            self.target[1] - self.eye[1],
            self.target[2] - self.eye[2],
        ]);

        let right = normalize3(cross3(&forward, &self.up));
        let true_up = cross3(&right, &forward);

        (forward, right, true_up)
    }
}

fn normalize3(v: [f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-15 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// 4x4 matrix multiply (row-major): C = A * B.
pub(crate) fn mat4_mul(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[i][k] * b[k][j];
            }
            out[i][j] = sum;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-8;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn make_camera() -> Camera {
        Camera::new(
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            std::f64::consts::FRAC_PI_4, // 45 degrees
            1.0,
        )
    }

    #[test]
    fn center_ray_points_along_view_direction() {
        let cam = make_camera();
        let (origin, dir) = cam.ray_at(0.5, 0.5);

        // Origin should be the eye
        assert!(approx(origin[0], 0.0));
        assert!(approx(origin[1], 0.0));
        assert!(approx(origin[2], 10.0));

        // Center ray should point along -Z (toward target)
        assert!(dir[2] < -0.9, "Center ray z should be negative, got {}", dir[2]);
        assert!(dir[0].abs() < 0.01, "Center ray x should be ~0, got {}", dir[0]);
        assert!(dir[1].abs() < 0.01, "Center ray y should be ~0, got {}", dir[1]);
    }

    #[test]
    fn corner_rays_span_fov() {
        let cam = make_camera();

        let (_, dir_tl) = cam.ray_at(0.0, 1.0); // top-left
        let (_, dir_br) = cam.ray_at(1.0, 0.0); // bottom-right

        // Top-left should have negative x, positive y
        assert!(dir_tl[0] < 0.0, "Top-left x should be negative");
        assert!(dir_tl[1] > 0.0, "Top-left y should be positive");

        // Bottom-right should have positive x, negative y
        assert!(dir_br[0] > 0.0, "Bottom-right x should be positive");
        assert!(dir_br[1] < 0.0, "Bottom-right y should be negative");
    }

    #[test]
    fn view_matrix_transforms_eye_to_origin() {
        let cam = make_camera();
        let view = cam.view_matrix();

        // Transform eye point through view matrix
        let e = cam.eye;
        let tx = view[0][0] * e[0] + view[0][1] * e[1] + view[0][2] * e[2] + view[0][3];
        let ty = view[1][0] * e[0] + view[1][1] * e[1] + view[1][2] * e[2] + view[1][3];
        let tz = view[2][0] * e[0] + view[2][1] * e[1] + view[2][2] * e[2] + view[2][3];

        assert!(
            approx(tx, 0.0),
            "Eye in camera space should have x=0, got {}",
            tx
        );
        assert!(
            approx(ty, 0.0),
            "Eye in camera space should have y=0, got {}",
            ty
        );
        assert!(
            approx(tz, 0.0),
            "Eye in camera space should have z=0, got {}",
            tz
        );
    }

    #[test]
    fn projection_matrix_is_valid() {
        let cam = make_camera();
        let proj = cam.projection_matrix();

        // The projection matrix should have -1 in [3][2] for perspective divide
        assert!(approx(proj[3][2], -1.0));
        // proj[3][3] should be 0
        assert!(approx(proj[3][3], 0.0));
    }

    #[test]
    fn view_projection_is_product() {
        let cam = make_camera();
        let vp = cam.view_projection();
        let expected = mat4_mul(&cam.projection_matrix(), &cam.view_matrix());

        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    approx(vp[i][j], expected[i][j]),
                    "vp[{}][{}] = {} expected {}",
                    i,
                    j,
                    vp[i][j],
                    expected[i][j]
                );
            }
        }
    }

    #[test]
    fn ray_at_is_unit_direction() {
        let cam = make_camera();
        for &(u, v) in &[(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.3, 0.7)] {
            let (_, dir) = cam.ray_at(u, v);
            let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-10,
                "Ray direction should be unit length, got {} at ({},{})",
                len,
                u,
                v
            );
        }
    }

    #[test]
    fn mat4_mul_identity() {
        let id = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let m = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ];
        let result = mat4_mul(&id, &m);
        for i in 0..4 {
            for j in 0..4 {
                assert!(approx(result[i][j], m[i][j]));
            }
        }
    }
}

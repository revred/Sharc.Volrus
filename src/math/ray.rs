//! Ray for DDA traversal and raycasting — mirrors OpenVDB `openvdb::math::Ray`.

use super::Vec3d;

/// A ray defined by origin, direction, and parametric interval `[t_min, t_max]`.
///
/// `inv_dir` is precomputed for efficient slab-based AABB intersection tests.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3d,
    pub dir: Vec3d,
    pub inv_dir: Vec3d,
    pub t_min: f64,
    pub t_max: f64,
}

impl Ray {
    /// Create a new ray. `inv_dir` is computed automatically from `dir`.
    #[inline]
    pub fn new(origin: Vec3d, dir: Vec3d, t_min: f64, t_max: f64) -> Self {
        Self {
            origin,
            dir,
            inv_dir: Vec3d::new(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z),
            t_min,
            t_max,
        }
    }

    /// Evaluate the point along the ray at parameter `t`.
    #[inline]
    pub fn at(&self, t: f64) -> Vec3d {
        self.origin + self.dir * t
    }

    /// Slab-based ray-AABB intersection test.
    ///
    /// `bbox_min` and `bbox_max` define the axis-aligned box corners.
    /// Returns `Some((t_enter, t_exit))` if the ray hits the box within
    /// `[self.t_min, self.t_max]`, otherwise `None`.
    pub fn intersects_bbox(&self, bbox_min: Vec3d, bbox_max: Vec3d) -> Option<(f64, f64)> {
        let mut t_near = self.t_min;
        let mut t_far = self.t_max;

        // X slab
        let tx1 = (bbox_min.x - self.origin.x) * self.inv_dir.x;
        let tx2 = (bbox_max.x - self.origin.x) * self.inv_dir.x;
        let (tx_min, tx_max) = if tx1 <= tx2 { (tx1, tx2) } else { (tx2, tx1) };
        t_near = t_near.max(tx_min);
        t_far = t_far.min(tx_max);
        if t_near > t_far {
            return None;
        }

        // Y slab
        let ty1 = (bbox_min.y - self.origin.y) * self.inv_dir.y;
        let ty2 = (bbox_max.y - self.origin.y) * self.inv_dir.y;
        let (ty_min, ty_max) = if ty1 <= ty2 { (ty1, ty2) } else { (ty2, ty1) };
        t_near = t_near.max(ty_min);
        t_far = t_far.min(ty_max);
        if t_near > t_far {
            return None;
        }

        // Z slab
        let tz1 = (bbox_min.z - self.origin.z) * self.inv_dir.z;
        let tz2 = (bbox_max.z - self.origin.z) * self.inv_dir.z;
        let (tz_min, tz_max) = if tz1 <= tz2 { (tz1, tz2) } else { (tz2, tz1) };
        t_near = t_near.max(tz_min);
        t_far = t_far.min(tz_max);
        if t_near > t_far {
            return None;
        }

        Some((t_near, t_far))
    }
}

impl std::fmt::Display for Ray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Ray({} + t*{}, t=[{}, {}])",
            self.origin, self.dir, self.t_min, self.t_max
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    #[test]
    fn ray_at() {
        let r = Ray::new(
            Vec3d::new(0.0, 0.0, 0.0),
            Vec3d::new(1.0, 0.0, 0.0),
            0.0,
            100.0,
        );
        let p = r.at(5.0);
        assert!((p.x - 5.0).abs() < EPS);
        assert!((p.y).abs() < EPS);
        assert!((p.z).abs() < EPS);
    }

    #[test]
    fn ray_at_origin() {
        let r = Ray::new(
            Vec3d::new(1.0, 2.0, 3.0),
            Vec3d::new(0.0, 1.0, 0.0),
            0.0,
            10.0,
        );
        let p = r.at(0.0);
        assert_eq!(p, Vec3d::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn inv_dir_precomputed() {
        let r = Ray::new(
            Vec3d::zero(),
            Vec3d::new(2.0, 4.0, 8.0),
            0.0,
            1.0,
        );
        assert!((r.inv_dir.x - 0.5).abs() < EPS);
        assert!((r.inv_dir.y - 0.25).abs() < EPS);
        assert!((r.inv_dir.z - 0.125).abs() < EPS);
    }

    #[test]
    fn intersects_bbox_hit() {
        // Ray along +X hitting a unit cube at origin
        let r = Ray::new(
            Vec3d::new(-5.0, 0.5, 0.5),
            Vec3d::new(1.0, 0.0, 0.0),
            0.0,
            100.0,
        );
        let hit = r.intersects_bbox(Vec3d::zero(), Vec3d::splat(1.0));
        assert!(hit.is_some());
        let (t_enter, t_exit) = hit.unwrap();
        assert!((t_enter - 5.0).abs() < EPS);
        assert!((t_exit - 6.0).abs() < EPS);
    }

    #[test]
    fn intersects_bbox_miss() {
        // Ray along +X, but offset in Y so it misses the cube
        let r = Ray::new(
            Vec3d::new(-5.0, 5.0, 0.5),
            Vec3d::new(1.0, 0.0, 0.0),
            0.0,
            100.0,
        );
        assert!(r
            .intersects_bbox(Vec3d::zero(), Vec3d::splat(1.0))
            .is_none());
    }

    #[test]
    fn intersects_bbox_behind_ray() {
        // Box is behind the ray origin
        let r = Ray::new(
            Vec3d::new(5.0, 0.5, 0.5),
            Vec3d::new(1.0, 0.0, 0.0),
            0.0,
            100.0,
        );
        assert!(r
            .intersects_bbox(Vec3d::zero(), Vec3d::splat(1.0))
            .is_none());
    }

    #[test]
    fn intersects_bbox_ray_inside() {
        // Ray origin inside the box
        let r = Ray::new(
            Vec3d::new(0.5, 0.5, 0.5),
            Vec3d::new(1.0, 0.0, 0.0),
            0.0,
            100.0,
        );
        let hit = r.intersects_bbox(Vec3d::zero(), Vec3d::splat(1.0));
        assert!(hit.is_some());
        let (t_enter, t_exit) = hit.unwrap();
        assert!(t_enter <= 0.0 + EPS); // clipped to t_min
        assert!((t_exit - 0.5).abs() < EPS);
    }

    #[test]
    fn intersects_bbox_diagonal() {
        // Ray along the diagonal
        let r = Ray::new(
            Vec3d::new(-1.0, -1.0, -1.0),
            Vec3d::new(1.0, 1.0, 1.0).normalize(),
            0.0,
            100.0,
        );
        let hit = r.intersects_bbox(Vec3d::zero(), Vec3d::splat(1.0));
        assert!(hit.is_some());
    }

    #[test]
    fn intersects_bbox_negative_dir() {
        // Ray going in -X direction
        let r = Ray::new(
            Vec3d::new(5.0, 0.5, 0.5),
            Vec3d::new(-1.0, 0.0, 0.0),
            0.0,
            100.0,
        );
        let hit = r.intersects_bbox(Vec3d::zero(), Vec3d::splat(1.0));
        assert!(hit.is_some());
        let (t_enter, t_exit) = hit.unwrap();
        assert!((t_enter - 4.0).abs() < EPS);
        assert!((t_exit - 5.0).abs() < EPS);
    }

    #[test]
    fn intersects_bbox_t_range_clamp() {
        // Ray hits box but intersection is outside t_min..t_max
        let r = Ray::new(
            Vec3d::new(-5.0, 0.5, 0.5),
            Vec3d::new(1.0, 0.0, 0.0),
            0.0,
            3.0, // t_max too small to reach the box at t=5
        );
        assert!(r
            .intersects_bbox(Vec3d::zero(), Vec3d::splat(1.0))
            .is_none());
    }

    #[test]
    fn display() {
        let r = Ray::new(
            Vec3d::new(0.0, 0.0, 0.0),
            Vec3d::new(1.0, 0.0, 0.0),
            0.0,
            10.0,
        );
        let s = format!("{}", r);
        assert!(s.contains("Ray"));
    }
}

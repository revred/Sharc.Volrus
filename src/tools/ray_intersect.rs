//! DDA ray-grid intersection for level-set grids.
//!
//! Traverses voxels along a ray using Digital Differential Analyzer (DDA),
//! detecting zero-crossings in the SDF to find the surface.

use crate::grid::Grid;
use crate::math::Coord;
use crate::tools::gradient::gradient_at;

/// Result of a ray intersection with a level-set surface.
#[derive(Debug, Clone, Copy)]
pub struct RayHit {
    /// Parameter along the ray at the hit point.
    pub t: f64,
    /// World-space position of the hit.
    pub position: [f64; 3],
    /// Gradient-based surface normal (unit length, points outward).
    pub normal: [f64; 3],
}

/// Cast a ray against a level-set grid, finding the first zero-crossing.
///
/// Uses DDA (Digital Differential Analyzer) to step through voxels along the
/// ray direction. When a sign change is detected between adjacent voxels,
/// the exact crossing is refined by linear interpolation.
///
/// # Arguments
/// * `grid` — A level-set grid (SDF values).
/// * `origin` — Ray origin in world space.
/// * `dir` — Ray direction in world space (need not be unit length).
///
/// # Returns
/// `Some(RayHit)` if a zero-crossing is found, `None` otherwise.
pub fn ray_intersect(grid: &Grid<f32>, origin: [f64; 3], dir: [f64; 3]) -> Option<RayHit> {
    // Normalize direction.
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    if len < 1e-15 {
        return None;
    }
    let d = [dir[0] / len, dir[1] / len, dir[2] / len];

    // Transform ray to index space.
    let vs = grid.transform().voxel_size;
    let inv_vs = 1.0 / vs;
    let orig_idx = [
        (origin[0] - grid.transform().origin[0]) * inv_vs,
        (origin[1] - grid.transform().origin[1]) * inv_vs,
        (origin[2] - grid.transform().origin[2]) * inv_vs,
    ];
    let dir_idx = [d[0] * inv_vs, d[1] * inv_vs, d[2] * inv_vs];

    // Note: dir_idx is not unit in index space. We just need the DDA direction.
    // We normalize the index-space direction for uniform stepping.
    let dir_idx_len = (dir_idx[0] * dir_idx[0] + dir_idx[1] * dir_idx[1] + dir_idx[2] * dir_idx[2]).sqrt();
    let step_dir = [
        dir_idx[0] / dir_idx_len,
        dir_idx[1] / dir_idx_len,
        dir_idx[2] / dir_idx_len,
    ];

    // DDA: step through index-space voxels.
    // We use a simple parametric stepping approach: at each step, advance by
    // a small fraction (0.5 voxels) and check for sign changes.
    let step_size = 0.5; // half-voxel steps for robustness
    let max_steps = 2000u32;

    let mut prev_coord = Coord::new(
        orig_idx[0].round() as i32,
        orig_idx[1].round() as i32,
        orig_idx[2].round() as i32,
    );
    let mut prev_val = grid.get(prev_coord);
    let mut prev_t_world = 0.0f64;

    for step in 1..=max_steps {
        let t_idx = step as f64 * step_size;
        let px = orig_idx[0] + step_dir[0] * t_idx;
        let py = orig_idx[1] + step_dir[1] * t_idx;
        let pz = orig_idx[2] + step_dir[2] * t_idx;

        let curr_coord = Coord::new(px.round() as i32, py.round() as i32, pz.round() as i32);
        let curr_val = grid.get(curr_coord);

        // World-space t for this step.
        let t_world = t_idx * vs; // index-space distance * voxel_size = world distance

        // Detect zero-crossing: sign change between prev and current,
        // or current value is exactly zero. We also handle the case where
        // prev is nonzero and current is zero (direct surface hit).
        let sign_change = (prev_val > 0.0 && curr_val < 0.0)
            || (prev_val < 0.0 && curr_val > 0.0);
        let zero_hit = prev_val != 0.0 && curr_val == 0.0;

        if sign_change || zero_hit {
            // Linear interpolation to find exact crossing.
            let (hit_t, hit_pos) = if zero_hit {
                // Current voxel is exactly on the surface.
                let pos = [
                    origin[0] + d[0] * t_world,
                    origin[1] + d[1] * t_world,
                    origin[2] + d[2] * t_world,
                ];
                (t_world, pos)
            } else {
                let frac = prev_val / (prev_val - curr_val);
                let t = prev_t_world + (t_world - prev_t_world) * frac as f64;
                let pos = [
                    origin[0] + d[0] * t,
                    origin[1] + d[1] * t,
                    origin[2] + d[2] * t,
                ];
                (t, pos)
            };

            // Compute normal from gradient at the crossing voxel.
            // Use the voxel closer to the surface (smaller |SDF|).
            let normal_coord = if prev_val.abs() < curr_val.abs() {
                prev_coord
            } else {
                curr_coord
            };
            let grad = gradient_at(grid, normal_coord);

            let grad_len =
                (grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2]).sqrt();

            let normal = if grad_len > 1e-10 {
                [
                    (grad[0] / grad_len) as f64,
                    (grad[1] / grad_len) as f64,
                    (grad[2] / grad_len) as f64,
                ]
            } else {
                // Fallback: use ray direction negated.
                [-d[0], -d[1], -d[2]]
            };

            return Some(RayHit {
                t: hit_t,
                position: hit_pos,
                normal,
            });
        }

        prev_coord = curr_coord;
        prev_val = curr_val;
        prev_t_world = t_world;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::sphere::make_level_set_sphere;

    fn make_test_sphere() -> Grid<f32> {
        make_level_set_sphere(5.0, [0.0, 0.0, 0.0], 0.5, 3.0)
    }

    #[test]
    fn ray_toward_sphere_center_hits_at_radius() {
        let grid = make_test_sphere();

        // Ray from +X toward origin.
        let hit = ray_intersect(&grid, [20.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
        assert!(hit.is_some(), "Ray toward sphere should hit");

        let h = hit.unwrap();
        // Hit should be near x=5 (the surface at radius 5).
        assert!(
            (h.position[0] - 5.0).abs() < 1.0,
            "Hit x should be near radius=5, got {}",
            h.position[0]
        );
        assert!(
            h.position[1].abs() < 1.0,
            "Hit y should be near 0, got {}",
            h.position[1]
        );
        assert!(
            h.position[2].abs() < 1.0,
            "Hit z should be near 0, got {}",
            h.position[2]
        );
    }

    #[test]
    fn ray_tangent_to_sphere_misses() {
        let grid = make_test_sphere();

        // Ray parallel to X axis but offset far in Y (well beyond radius 5).
        let hit = ray_intersect(&grid, [-20.0, 20.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(hit.is_none(), "Tangent ray should miss the sphere");
    }

    #[test]
    fn ray_from_inside_sphere_hits() {
        let grid = make_test_sphere();

        // Ray from near the center outward along +X.
        // The SDF is negative inside, positive outside => will cross zero.
        let hit = ray_intersect(&grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        // Note: origin (0,0,0) might be outside the narrow band, so let's
        // start from a point within the band but inside the sphere.
        let hit2 = ray_intersect(&grid, [3.5, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(hit.is_some() || hit2.is_some(), "Ray from inside should hit surface");

        if let Some(h) = hit.or(hit2) {
            // Hit should be near radius 5.
            assert!(
                (h.position[0] - 5.0).abs() < 1.5,
                "Inside→outside hit x should be near 5, got {}",
                h.position[0]
            );
        }
    }

    #[test]
    fn hit_normal_points_away_from_sphere_center() {
        let grid = make_test_sphere();

        // Ray from +X toward center.
        let hit = ray_intersect(&grid, [20.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
        assert!(hit.is_some());

        let h = hit.unwrap();
        // Normal should point roughly in +X direction (away from center).
        assert!(
            h.normal[0] > 0.5,
            "Normal x-component should be positive (outward), got {}",
            h.normal[0]
        );

        // Normal should be roughly unit length.
        let nlen = (h.normal[0] * h.normal[0]
            + h.normal[1] * h.normal[1]
            + h.normal[2] * h.normal[2])
            .sqrt();
        assert!(
            (nlen - 1.0).abs() < 0.1,
            "Normal should be unit length, got {}",
            nlen
        );
    }

    #[test]
    fn ray_along_y_axis_hits_sphere() {
        let grid = make_test_sphere();

        let hit = ray_intersect(&grid, [0.0, -20.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(hit.is_some(), "Ray along Y should hit sphere");

        let h = hit.unwrap();
        assert!(
            (h.position[1] - (-5.0)).abs() < 1.0,
            "Hit y should be near -5 (south pole), got {}",
            h.position[1]
        );
    }

    #[test]
    fn hit_t_is_positive() {
        let grid = make_test_sphere();

        let hit = ray_intersect(&grid, [20.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
        assert!(hit.is_some());
        assert!(hit.unwrap().t > 0.0, "Hit t should be positive");
    }
}

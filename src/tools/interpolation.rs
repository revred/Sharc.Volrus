//! World-space sampling with trilinear interpolation.

use crate::grid::Grid;
use crate::math::Coord;

/// Trilinear interpolation at a world-space point.
///
/// Converts the world position to continuous index space, identifies
/// the 8 surrounding voxel corners, and blends their values using
/// trilinear weights.
pub fn sample_trilinear(grid: &Grid<f32>, world_pos: [f64; 3]) -> f32 {
    let idx = grid.transform().world_to_index_f64(world_pos);

    // Floor to get the base corner.
    let ix = idx[0].floor();
    let iy = idx[1].floor();
    let iz = idx[2].floor();

    // Fractional parts (interpolation weights).
    let fx = (idx[0] - ix) as f32;
    let fy = (idx[1] - iy) as f32;
    let fz = (idx[2] - iz) as f32;

    let i = ix as i32;
    let j = iy as i32;
    let k = iz as i32;

    // Fetch 8 corner values.
    let c000 = grid.get(Coord::new(i, j, k));
    let c100 = grid.get(Coord::new(i + 1, j, k));
    let c010 = grid.get(Coord::new(i, j + 1, k));
    let c110 = grid.get(Coord::new(i + 1, j + 1, k));
    let c001 = grid.get(Coord::new(i, j, k + 1));
    let c101 = grid.get(Coord::new(i + 1, j, k + 1));
    let c011 = grid.get(Coord::new(i, j + 1, k + 1));
    let c111 = grid.get(Coord::new(i + 1, j + 1, k + 1));

    // Trilinear blend.
    let c00 = c000 * (1.0 - fx) + c100 * fx;
    let c01 = c001 * (1.0 - fx) + c101 * fx;
    let c10 = c010 * (1.0 - fx) + c110 * fx;
    let c11 = c011 * (1.0 - fx) + c111 * fx;

    let c0 = c00 * (1.0 - fy) + c10 * fy;
    let c1 = c01 * (1.0 - fy) + c11 * fy;

    c0 * (1.0 - fz) + c1 * fz
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Coord;

    #[test]
    fn trilinear_at_voxel_center_returns_exact() {
        let mut grid = Grid::<f32>::new(0.0, 1.0);
        grid.set(Coord::new(5, 5, 5), 42.0);

        // At integer index, trilinear should return the exact value
        // (assuming all neighbours are background = 0).
        // Actually at (5.0,5.0,5.0) the floor is (5,5,5) with frac=0,
        // so the result is just c000 = 42.0.
        let val = sample_trilinear(&grid, [5.0, 5.0, 5.0]);
        assert!((val - 42.0).abs() < 1e-6, "Got {}", val);
    }

    #[test]
    fn trilinear_midpoint_interpolates() {
        let mut grid = Grid::<f32>::new(0.0, 1.0);
        grid.set(Coord::new(0, 0, 0), 0.0);
        grid.set(Coord::new(1, 0, 0), 10.0);

        // At (0.5, 0, 0), trilinear should give 5.0.
        let val = sample_trilinear(&grid, [0.5, 0.0, 0.0]);
        assert!((val - 5.0).abs() < 1e-5, "Got {}", val);
    }

    #[test]
    fn trilinear_respects_voxel_size() {
        let mut grid = Grid::<f32>::new(0.0, 0.5);
        // Voxel at index (0,0,0) → world (0,0,0)
        // Voxel at index (1,0,0) → world (0.5, 0, 0)
        grid.set(Coord::new(0, 0, 0), 0.0);
        grid.set(Coord::new(1, 0, 0), 10.0);

        // World 0.25 → index 0.5 → midpoint → value 5.0
        let val = sample_trilinear(&grid, [0.25, 0.0, 0.0]);
        assert!((val - 5.0).abs() < 1e-5, "Got {}", val);
    }

    #[test]
    fn trilinear_on_sphere_sdf_smooth() {
        use crate::tools::sphere::make_level_set_sphere;

        let grid = make_level_set_sphere(5.0, [0.0, 0.0, 0.0], 0.5, 3.0);

        // On the surface (distance = radius), SDF should be ~0.
        let val = sample_trilinear(&grid, [5.0, 0.0, 0.0]);
        assert!(
            val.abs() < 1.0,
            "SDF at surface should be near zero, got {}",
            val
        );

        // Just inside the narrow band: at distance 4 from center, SDF = 4-5 = -1.
        // |SDF| = 1 < band = 1.5, so this is stored.
        let val_inside = sample_trilinear(&grid, [4.0, 0.0, 0.0]);
        assert!(
            val_inside < 0.0,
            "SDF inside narrow band should be negative, got {}",
            val_inside
        );
    }
}

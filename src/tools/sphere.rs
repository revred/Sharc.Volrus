//! Level-set sphere generator — the classic OpenVDB "hello world".

use crate::grid::{Grid, GridClass, VoxelTransform};
use crate::math::Coord;

/// Create a narrow-band level-set sphere.
///
/// Iterates over the bounding box of the sphere in index space, computes
/// `SDF = distance_to_center - radius`, and stores only voxels within
/// the narrow band `|sdf| < half_width * voxel_size`.
///
/// # Arguments
/// * `radius` — sphere radius in world units.
/// * `center` — sphere center in world units.
/// * `voxel_size` — uniform voxel dimension.
/// * `half_width` — narrow-band half-width in voxels.
pub fn make_level_set_sphere(
    radius: f64,
    center: [f64; 3],
    voxel_size: f64,
    half_width: f64,
) -> Grid<f32> {
    let band = half_width * voxel_size;
    let bg = band as f32;

    let mut grid = Grid::<f32>::new(bg, voxel_size);
    grid.set_grid_class(GridClass::LevelSet);
    grid.set_name("sdf");
    grid.set_transform(VoxelTransform::uniform(voxel_size));

    // Compute bounding box in index space.
    let extent = radius + band;
    let inv = 1.0 / voxel_size;

    let imin = ((center[0] - extent) * inv).floor() as i32;
    let jmin = ((center[1] - extent) * inv).floor() as i32;
    let kmin = ((center[2] - extent) * inv).floor() as i32;
    let imax = ((center[0] + extent) * inv).ceil() as i32;
    let jmax = ((center[1] + extent) * inv).ceil() as i32;
    let kmax = ((center[2] + extent) * inv).ceil() as i32;

    for i in imin..=imax {
        let wx = i as f64 * voxel_size - center[0];
        for j in jmin..=jmax {
            let wy = j as f64 * voxel_size - center[1];
            for k in kmin..=kmax {
                let wz = k as f64 * voxel_size - center[2];
                let dist = (wx * wx + wy * wy + wz * wz).sqrt();
                let sdf = (dist - radius) as f32;

                if sdf.abs() < bg {
                    grid.set(Coord::new(i, j, k), sdf);
                }
            }
        }
    }

    grid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_has_active_voxels() {
        let grid = make_level_set_sphere(5.0, [0.0, 0.0, 0.0], 1.0, 3.0);
        assert!(grid.active_voxel_count() > 0);
        assert_eq!(grid.grid_class(), GridClass::LevelSet);
    }

    #[test]
    fn sphere_sdf_value_at_center_is_negative() {
        let grid = make_level_set_sphere(5.0, [0.0, 0.0, 0.0], 1.0, 3.0);
        // At center (0,0,0), distance = 0, sdf = 0 - 5 = -5.
        // But |sdf| = 5 > band = 3, so it may not be stored.
        // At index (3,0,0), distance = 3, sdf = 3-5 = -2. Within band.
        let val = grid.get(Coord::new(3, 0, 0));
        assert!(val < 0.0, "SDF inside sphere should be negative: {}", val);
    }

    #[test]
    fn sphere_sdf_value_outside_is_positive() {
        let grid = make_level_set_sphere(5.0, [0.0, 0.0, 0.0], 1.0, 3.0);
        // At index (7,0,0), distance = 7, sdf = 7-5 = 2. Within band.
        let val = grid.get(Coord::new(7, 0, 0));
        assert!(val > 0.0, "SDF outside sphere should be positive: {}", val);
    }

    #[test]
    fn sphere_sdf_value_on_surface_is_near_zero() {
        let grid = make_level_set_sphere(5.0, [0.0, 0.0, 0.0], 1.0, 3.0);
        let val = grid.get(Coord::new(5, 0, 0));
        assert!(
            val.abs() < 0.5,
            "SDF on surface should be near zero: {}",
            val
        );
    }

    #[test]
    fn sphere_narrow_band_width() {
        let grid = make_level_set_sphere(10.0, [0.0, 0.0, 0.0], 1.0, 3.0);
        // Voxels far outside the narrow band should return background.
        let far_val = grid.get(Coord::new(20, 0, 0));
        let bg = grid.tree().background();
        assert_eq!(far_val, bg, "Far voxel should be background");
    }

    #[test]
    fn sphere_with_offset_center() {
        let grid = make_level_set_sphere(3.0, [10.0, 20.0, 30.0], 0.5, 3.0);
        assert!(grid.active_voxel_count() > 0);

        // Index of center: (10/0.5, 20/0.5, 30/0.5) = (20, 40, 60)
        // At surface: index (26, 40, 60) → world (13, 20, 30) → dist = 3 → sdf ~ 0
        let val = grid.get(Coord::new(26, 40, 60));
        assert!(
            val.abs() < 1.0,
            "SDF at surface of offset sphere should be near zero: {}",
            val
        );
    }
}

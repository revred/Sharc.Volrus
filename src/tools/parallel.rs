//! Parallel tools — feature-gated behind `parallel` (rayon).
//!
//! These provide data-parallel versions of key operations on sparse grids.
//! The pattern is: collect active voxels, process in parallel, insert
//! results into a new grid sequentially.

#[cfg(feature = "parallel")]
use crate::grid::Grid;
#[cfg(feature = "parallel")]
use crate::math::Coord;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Apply a function to every active voxel in parallel, returning a new grid.
///
/// The output grid inherits the same background, voxel size, and grid class.
/// Each active voxel's value is replaced by `f(coord, old_value)`.
///
/// # Example
/// ```ignore
/// let negated = par_apply(&sdf_grid, |_coord, val| -val);
/// ```
#[cfg(feature = "parallel")]
pub fn par_apply<F>(grid: &Grid<f32>, f: F) -> Grid<f32>
where
    F: Fn(Coord, f32) -> f32 + Sync + Send,
{
    // Step 1: collect all active voxels
    let active: Vec<(Coord, f32)> = grid.tree().iter_active().collect();

    // Step 2: transform values in parallel
    let transformed: Vec<(Coord, f32)> = active
        .into_par_iter()
        .map(|(coord, val)| (coord, f(coord, val)))
        .collect();

    // Step 3: build output grid
    let mut out = Grid::<f32>::new(grid.tree().background(), grid.transform().voxel_size);
    out.set_grid_class(grid.grid_class());

    for (coord, val) in transformed {
        out.set(coord, val);
    }

    out
}

#[cfg(test)]
#[cfg(feature = "parallel")]
mod tests {
    use super::*;
    use crate::tools::make_level_set_sphere;

    #[test]
    fn par_apply_identity_preserves_values() {
        let grid = make_level_set_sphere(3.0, [0.0; 3], 1.0, 3.0);
        let result = par_apply(&grid, |_coord, val| val);

        assert_eq!(result.active_voxel_count(), grid.active_voxel_count());
        assert_eq!(result.leaf_count(), grid.leaf_count());

        for (coord, val) in grid.tree().iter_active() {
            let rval = result.get(coord);
            assert!(
                (rval - val).abs() < 1e-9,
                "mismatch at {coord}: expected {val}, got {rval}"
            );
        }
    }

    #[test]
    fn par_apply_negate_flips_signs() {
        let grid = make_level_set_sphere(3.0, [0.0; 3], 1.0, 3.0);
        let negated = par_apply(&grid, |_coord, val| -val);

        assert_eq!(negated.active_voxel_count(), grid.active_voxel_count());

        for (coord, val) in grid.tree().iter_active() {
            let nval = negated.get(coord);
            assert!(
                (nval + val).abs() < 1e-9,
                "negate mismatch at {coord}: expected {}, got {nval}",
                -val
            );
        }
    }

    #[test]
    fn par_apply_scale() {
        let mut grid = Grid::<f32>::new(0.0, 1.0);
        grid.set(Coord::new(0, 0, 0), 2.0);
        grid.set(Coord::new(10, 10, 10), 5.0);

        let scaled = par_apply(&grid, |_coord, val| val * 3.0);
        assert_eq!(scaled.get(Coord::new(0, 0, 0)), 6.0);
        assert_eq!(scaled.get(Coord::new(10, 10, 10)), 15.0);
    }
}

//! Spatial clip and crop operations on sparse VDB grids.
//!
//! - `clip_to_bbox` — copy only the active voxels inside an AABB into a new grid.
//! - `crop_in_place` — deactivate active voxels outside an AABB in-place.

use crate::grid::Grid;
use crate::math::{Coord, CoordBBox};

/// Return a new grid containing only the active voxels of `grid` that lie
/// inside `bbox` (inclusive).
///
/// The output shares `grid`'s transform, name, and class.
pub fn clip_to_bbox<T: Copy + Default>(grid: &Grid<T>, bbox: &CoordBBox) -> Grid<T> {
    let affine = grid.affine_map();
    let bg = grid.tree().background();
    let mut out = Grid::<T>::with_affine(bg, *affine);
    out.set_name(grid.name());
    out.set_grid_class(grid.grid_class());

    for (coord, value) in grid.tree().iter_active() {
        if bbox.contains(coord) {
            out.set(coord, value);
        }
    }
    out
}

/// Deactivate all active voxels in `grid` that lie outside `bbox`, in-place.
///
/// Voxels inside `bbox` are untouched.  Empty leaf nodes are pruned after
/// deactivation.
pub fn crop_in_place<T: Copy + Default>(grid: &mut Grid<T>, bbox: &CoordBBox) {
    let to_deactivate: Vec<Coord> = grid
        .tree()
        .iter_active()
        .filter(|(coord, _)| !bbox.contains(*coord))
        .map(|(coord, _)| coord)
        .collect();

    for coord in to_deactivate {
        grid.tree_mut().deactivate(coord);
    }
    grid.tree_mut().remove_empty_leaves();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Coord;

    /// Helper: 5×5×5 grid with value = x+y+z at each voxel.
    fn make_5x5x5() -> Grid<f32> {
        let mut g = Grid::<f32>::new(0.0, 1.0);
        for x in 0..5i32 {
            for y in 0..5i32 {
                for z in 0..5i32 {
                    g.set(Coord::new(x, y, z), (x + y + z) as f32);
                }
            }
        }
        g
    }

    #[test]
    fn clip_keeps_only_inside_voxels() {
        let g = make_5x5x5();
        let bbox = CoordBBox::new(Coord::new(1, 1, 1), Coord::new(3, 3, 3));
        let clipped = clip_to_bbox(&g, &bbox);

        for (coord, _) in clipped.tree().iter_active() {
            assert!(bbox.contains(coord), "coord ({},{},{}) outside bbox", coord.x, coord.y, coord.z);
        }
        // 3×3×3 = 27 voxels remain
        assert_eq!(clipped.tree().active_voxel_count(), 27);
    }

    #[test]
    fn clip_outside_bbox_returns_empty() {
        let g = make_5x5x5();
        let bbox = CoordBBox::new(Coord::new(20, 20, 20), Coord::new(30, 30, 30));
        let clipped = clip_to_bbox(&g, &bbox);
        assert_eq!(clipped.tree().active_voxel_count(), 0);
    }

    #[test]
    fn clip_preserves_values() {
        let g = make_5x5x5();
        let bbox = CoordBBox::new(Coord::new(0, 0, 0), Coord::new(2, 2, 2));
        let clipped = clip_to_bbox(&g, &bbox);

        for (coord, val) in clipped.tree().iter_active() {
            let expected = (coord.x + coord.y + coord.z) as f32;
            assert_eq!(val, expected, "value mismatch at ({},{},{})", coord.x, coord.y, coord.z);
        }
    }

    #[test]
    fn clip_empty_grid_returns_empty() {
        let g = Grid::<f32>::new(0.0, 1.0);
        let bbox = CoordBBox::new(Coord::new(0, 0, 0), Coord::new(10, 10, 10));
        let clipped = clip_to_bbox(&g, &bbox);
        assert_eq!(clipped.tree().active_voxel_count(), 0);
    }

    #[test]
    fn crop_in_place_removes_outside() {
        let mut g = make_5x5x5();
        let before = g.tree().active_voxel_count(); // 125
        let bbox = CoordBBox::new(Coord::new(0, 0, 0), Coord::new(2, 2, 2));
        crop_in_place(&mut g, &bbox);

        let after = g.tree().active_voxel_count();
        assert!(after < before, "crop should remove voxels: before={} after={}", before, after);
        assert_eq!(after, 27, "3×3×3 = 27 should remain");

        for (coord, _) in g.tree().iter_active() {
            assert!(bbox.contains(coord), "({},{},{}) should have been cropped", coord.x, coord.y, coord.z);
        }
    }

    #[test]
    fn crop_in_place_full_bbox_noop() {
        let mut g = make_5x5x5();
        let before = g.tree().active_voxel_count();
        // bbox covers the whole grid
        let bbox = CoordBBox::new(Coord::new(0, 0, 0), Coord::new(4, 4, 4));
        crop_in_place(&mut g, &bbox);
        assert_eq!(g.tree().active_voxel_count(), before);
    }

    #[test]
    fn crop_in_place_empty_grid_noop() {
        let mut g = Grid::<f32>::new(0.0, 1.0);
        let bbox = CoordBBox::new(Coord::new(0, 0, 0), Coord::new(10, 10, 10));
        crop_in_place(&mut g, &bbox);
        assert_eq!(g.tree().active_voxel_count(), 0);
    }
}

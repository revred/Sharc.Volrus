//! Morphological operations on sparse grids: dilate, erode, opening, closing.

use crate::grid::Grid;
use crate::math::Coord;

/// The 6-connected neighbor offsets (face-adjacent voxels).
const NEIGHBORS_6: [Coord; 6] = [
    Coord::new(1, 0, 0),
    Coord::new(-1, 0, 0),
    Coord::new(0, 1, 0),
    Coord::new(0, -1, 0),
    Coord::new(0, 0, 1),
    Coord::new(0, 0, -1),
];

/// Dilate active voxels: activate all voxels within `iterations` steps
/// of any currently active voxel (6-connected neighborhood).
///
/// Newly activated voxels receive the value of their nearest active neighbor.
pub fn dilate(grid: &mut Grid<f32>, iterations: u32) {
    for _ in 0..iterations {
        // Collect current active voxels.
        let active: Vec<(Coord, f32)> = grid.tree().iter_active().collect();

        // For each active voxel, activate its 6 neighbors if not already active.
        for (coord, value) in &active {
            for offset in &NEIGHBORS_6 {
                let neighbor = *coord + *offset;
                if !grid.tree().is_active(neighbor) {
                    grid.set(neighbor, *value);
                }
            }
        }
    }
}

/// Erode active voxels: deactivate boundary voxels `iterations` times.
///
/// A boundary voxel is one where at least one of its 6 neighbors is inactive.
pub fn erode(grid: &mut Grid<f32>, iterations: u32) {
    let bg = grid.tree().background();

    for _ in 0..iterations {
        // Find voxels to deactivate: those with at least one inactive neighbor.
        let to_deactivate: Vec<Coord> = grid
            .tree()
            .iter_active()
            .filter_map(|(coord, _)| {
                let is_boundary = NEIGHBORS_6
                    .iter()
                    .any(|offset| !grid.tree().is_active(coord + *offset));
                if is_boundary {
                    Some(coord)
                } else {
                    None
                }
            })
            .collect();

        // Deactivate boundary voxels by setting them to background.
        // We need leaf-level deactivation. Since Grid doesn't expose set_inactive
        // directly, we use the tree's leaf_mut path.
        for coord in to_deactivate {
            if let Some(leaf) = grid.tree_mut().leaf_mut(coord) {
                leaf.set_inactive(coord, bg);
            }
        }
    }
}

/// Opening: erode then dilate (removes small protrusions).
pub fn opening(grid: &mut Grid<f32>, iterations: u32) {
    erode(grid, iterations);
    dilate(grid, iterations);
}

/// Closing: dilate then erode (fills small holes).
pub fn closing(grid: &mut Grid<f32>, iterations: u32) {
    dilate(grid, iterations);
    erode(grid, iterations);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dilate_single_voxel_produces_7_active() {
        let mut grid = Grid::<f32>::new(0.0, 1.0);
        grid.set(Coord::new(0, 0, 0), 1.0);
        assert_eq!(grid.active_voxel_count(), 1);

        dilate(&mut grid, 1);

        // Center + 6 face neighbors = 7.
        assert_eq!(
            grid.active_voxel_count(),
            7,
            "Expected 7 active voxels after dilating a single voxel"
        );

        // All neighbors should have value 1.0 (copied from center).
        for offset in &NEIGHBORS_6 {
            let c = Coord::new(0, 0, 0) + *offset;
            assert!(
                grid.tree().is_active(c),
                "Neighbor {:?} should be active",
                c
            );
        }
    }

    #[test]
    fn erode_3x3x3_solid_leaves_center() {
        let mut grid = Grid::<f32>::new(0.0, 1.0);
        // Fill a 3x3x3 cube: coords -1..=1 in each axis.
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    grid.set(Coord::new(x, y, z), 1.0);
                }
            }
        }
        assert_eq!(grid.active_voxel_count(), 27);

        erode(&mut grid, 1);

        // Only the center voxel (0,0,0) has all 6 neighbors active in the
        // original 3x3x3, so only it survives.
        assert_eq!(
            grid.active_voxel_count(),
            1,
            "Only center should survive erosion of 3x3x3"
        );
        assert!(grid.tree().is_active(Coord::new(0, 0, 0)));
    }

    #[test]
    fn dilate_two_iterations() {
        let mut grid = Grid::<f32>::new(0.0, 1.0);
        grid.set(Coord::new(0, 0, 0), 1.0);

        dilate(&mut grid, 2);

        // After 2 dilations from a single point, we get a diamond shape
        // (L1 ball of radius 2). Count = 1 + 6 + 18 = 25.
        // L1 distance <= 2: sum_{d=0}^{2} count(d).
        // d=0: 1, d=1: 6, d=2: 18 => total 25.
        let count = grid.active_voxel_count();
        assert_eq!(count, 25, "L1 ball radius 2 should have 25 voxels, got {}", count);
    }

    #[test]
    fn opening_preserves_large_feature() {
        let mut grid = Grid::<f32>::new(0.0, 1.0);
        // Fill a 5x5x5 cube.
        for x in -2..=2 {
            for y in -2..=2 {
                for z in -2..=2 {
                    grid.set(Coord::new(x, y, z), 1.0);
                }
            }
        }
        let _before = grid.active_voxel_count();

        opening(&mut grid, 1);
        let after = grid.active_voxel_count();

        // The interior of the 5x5x5 cube (3x3x3 = 27 voxels) survives erosion,
        // then dilation restores the shape. The result should be close to the
        // original (corners may differ slightly).
        assert!(
            after > 0,
            "Opening should preserve large features"
        );
        // The 5x5x5 cube is large enough that opening(1) preserves most of it.
        assert!(
            after >= 27,
            "Opening should preserve at least the interior: got {}",
            after
        );
    }

    #[test]
    fn closing_fills_small_hole() {
        let mut grid = Grid::<f32>::new(0.0, 1.0);
        // Fill a 5x5x5 cube but leave center empty (a "hole").
        for x in -2..=2 {
            for y in -2..=2 {
                for z in -2..=2 {
                    if !(x == 0 && y == 0 && z == 0) {
                        grid.set(Coord::new(x, y, z), 1.0);
                    }
                }
            }
        }
        assert!(!grid.tree().is_active(Coord::new(0, 0, 0)));

        closing(&mut grid, 1);

        // After closing, the center hole should be filled.
        assert!(
            grid.tree().is_active(Coord::new(0, 0, 0)),
            "Closing should fill the center hole"
        );
    }
}

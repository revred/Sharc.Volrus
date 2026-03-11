//! Sign flood-fill for level-set grids.
//!
//! After building an SDF grid, inactive voxels should hold the correct sign:
//! positive outside the surface, negative inside.  This module performs a
//! leaf-local sign propagation: for each inactive voxel in a leaf, if the
//! majority of active neighbours within the same leaf are negative, the
//! inactive voxel is set to -background; otherwise +background.

use crate::grid::Grid;
use crate::tree::LEAF_LOG2DIM;

/// Flood-fill inactive voxels of a level-set grid so that outside voxels
/// are +background and inside voxels are -background.
///
/// This is a leaf-local approximation: each inactive voxel's sign is
/// determined by the active voxels in the same leaf.  For most narrow-band
/// level sets this produces correct results.
pub fn flood_fill_sign(grid: &mut Grid<f32>) {
    let bg = grid.tree().background();
    let dim = 1i32 << LEAF_LOG2DIM; // 8

    for leaf in grid.tree_mut().leaves_mut() {
        // First pass: determine the dominant sign from active voxels.
        let mut neg_count: i32 = 0;
        let mut pos_count: i32 = 0;
        for off in 0..512usize {
            let word = off / 64;
            let bit = off % 64;
            if (leaf.active_mask()[word] >> bit) & 1 == 1 {
                if leaf.values()[off] < 0.0 {
                    neg_count += 1;
                } else {
                    pos_count += 1;
                }
            }
        }

        // If no active voxels, leave the leaf as-is.
        if neg_count == 0 && pos_count == 0 {
            continue;
        }

        // Second pass: for each inactive voxel, check its 6-connected
        // neighbours within this leaf.  If more negative neighbours than
        // positive, assign -bg; else +bg.  Fall back to leaf-global
        // majority if no active neighbours are found.
        let global_sign: f32 = if neg_count > pos_count { -bg } else { bg };

        // We need a snapshot of the active mask to avoid read/write conflicts.
        let mask_snapshot = *leaf.active_mask();
        let values_snapshot: Vec<f32> = leaf.values().to_vec();

        for off in 0..512usize {
            let word = off / 64;
            let bit = off % 64;
            if (mask_snapshot[word] >> bit) & 1 == 1 {
                continue; // active voxel — skip
            }

            // Recover local (lx, ly, lz) from ZYX-packed offset.
            let lx = (off >> 6) as i32; // off / 64
            let ly = ((off >> 3) & 7) as i32; // (off / 8) % 8
            let lz = (off & 7) as i32; // off % 8

            // Check 6-connected neighbours within this leaf.
            let mut local_neg = 0i32;
            let mut local_pos = 0i32;
            let deltas: [(i32, i32, i32); 6] = [
                (-1, 0, 0),
                (1, 0, 0),
                (0, -1, 0),
                (0, 1, 0),
                (0, 0, -1),
                (0, 0, 1),
            ];
            for (dx, dy, dz) in &deltas {
                let nx = lx + dx;
                let ny = ly + dy;
                let nz = lz + dz;
                if nx < 0 || nx >= dim || ny < 0 || ny >= dim || nz < 0 || nz >= dim {
                    continue; // outside this leaf
                }
                let n_off = ((nx as usize) << 6) | ((ny as usize) << 3) | (nz as usize);
                let n_word = n_off / 64;
                let n_bit = n_off % 64;
                if (mask_snapshot[n_word] >> n_bit) & 1 == 1 {
                    if values_snapshot[n_off] < 0.0 {
                        local_neg += 1;
                    } else {
                        local_pos += 1;
                    }
                }
            }

            let fill_val = if local_neg + local_pos > 0 {
                if local_neg > local_pos {
                    -bg
                } else {
                    bg
                }
            } else {
                global_sign
            };

            leaf.values_mut()[off] = fill_val;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::GridClass;
    use crate::math::Coord;

    #[test]
    fn flood_fill_sets_inside_negative() {
        // Create a level-set grid with bg = 3.0.
        // Simulate a thin shell: active voxels at z=4 form a "wall",
        // values > 0 on one side, < 0 on the other.
        let voxel_size = 1.0;
        let half_width = 3.0;
        let bg = (half_width * voxel_size) as f32;
        let mut grid = Grid::<f32>::new(bg, voxel_size);
        grid.set_grid_class(GridClass::LevelSet);

        // Place active voxels forming a plane at z=4 in the first leaf.
        // Negative side: z < 4, positive side: z > 4.
        for x in 0..8 {
            for y in 0..8 {
                // Active narrow-band voxels around the plane.
                grid.set(Coord::new(x, y, 3), -1.0);
                grid.set(Coord::new(x, y, 4), 0.0);
                grid.set(Coord::new(x, y, 5), 1.0);
            }
        }

        flood_fill_sign(&mut grid);

        // Inactive voxel at (0,0,2) is adjacent to active voxel at (0,0,3)
        // which is negative, so it should be set to -bg.
        let val = grid.get(Coord::new(0, 0, 2));
        assert!(
            val < 0.0,
            "Expected negative inside, got {} at (0,0,2)",
            val
        );

        // Inactive voxel at (0,0,6) is adjacent to active voxel at (0,0,5)
        // which is positive, so it should be set to +bg.
        let val = grid.get(Coord::new(0, 0, 6));
        assert!(
            val > 0.0,
            "Expected positive outside, got {} at (0,0,6)",
            val
        );
    }

    #[test]
    fn flood_fill_all_positive_stays_positive() {
        let bg = 3.0f32;
        let mut grid = Grid::<f32>::new(bg, 1.0);
        grid.set(Coord::new(0, 0, 0), 1.0);
        grid.set(Coord::new(1, 0, 0), 2.0);

        flood_fill_sign(&mut grid);

        // Inactive neighbour should remain positive.
        assert!(grid.get(Coord::new(2, 0, 0)) > 0.0);
    }
}

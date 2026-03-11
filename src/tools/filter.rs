//! Smoothing filters for scalar grids.

use crate::grid::Grid;
use crate::math::Coord;

/// The 6-connected neighbor offsets.
const NEIGHBORS_6: [Coord; 6] = [
    Coord::new(1, 0, 0),
    Coord::new(-1, 0, 0),
    Coord::new(0, 1, 0),
    Coord::new(0, -1, 0),
    Coord::new(0, 0, 1),
    Coord::new(0, 0, -1),
];

/// Gaussian-like smoothing: replace each active voxel with the average
/// of itself and its 6 face-connected neighbors (7 values total).
///
/// Repeats for `iterations` passes.
pub fn mean_filter(grid: &mut Grid<f32>, iterations: u32) {
    for _ in 0..iterations {
        // Collect current active voxels and their smoothed values.
        let updates: Vec<(Coord, f32)> = grid
            .tree()
            .iter_active()
            .map(|(coord, val)| {
                let sum: f32 = NEIGHBORS_6
                    .iter()
                    .map(|off| grid.get(coord + *off))
                    .sum::<f32>()
                    + val;
                (coord, sum / 7.0)
            })
            .collect();

        for (coord, value) in updates {
            grid.set(coord, value);
        }
    }
}

/// Median filter: replace each active voxel with the median of its
/// 6 neighbors plus itself (7 values total).
///
/// Repeats for `iterations` passes.
pub fn median_filter(grid: &mut Grid<f32>, iterations: u32) {
    for _ in 0..iterations {
        let updates: Vec<(Coord, f32)> = grid
            .tree()
            .iter_active()
            .map(|(coord, val)| {
                let mut values = [0.0f32; 7];
                values[0] = val;
                for (i, off) in NEIGHBORS_6.iter().enumerate() {
                    values[i + 1] = grid.get(coord + *off);
                }
                // Sort and pick median (index 3 of 7).
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (coord, values[3])
            })
            .collect();

        for (coord, value) in updates {
            grid.set(coord, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::sphere::make_level_set_sphere;

    #[test]
    fn mean_filter_reduces_noise_on_sphere() {
        // Use a wide narrow band so interior voxels have all-active neighbors.
        let mut grid = make_level_set_sphere(10.0, [0.0, 0.0, 0.0], 0.5, 5.0);

        let active: Vec<(Coord, f32)> = grid.tree().iter_active().collect();

        // Identify interior voxels (all 6 neighbors are also active).
        let interior: Vec<Coord> = active
            .iter()
            .filter(|(c, _)| {
                NEIGHBORS_6.iter().all(|off| grid.tree().is_active(*c + *off))
            })
            .map(|(c, _)| *c)
            .collect();
        assert!(interior.len() > 100, "Need enough interior voxels for a meaningful test");

        // Add noise to ALL active voxels.
        let mut noise_seed = 12345u64;
        for (coord, val) in &active {
            noise_seed = noise_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((noise_seed >> 33) as f32 / (u32::MAX >> 1) as f32 - 1.0) * 0.3;
            grid.set(*coord, val + noise);
        }

        let original = make_level_set_sphere(10.0, [0.0, 0.0, 0.0], 0.5, 5.0);

        // Measure RMS deviation of interior voxels only.
        let rms_before: f64 = {
            let sum: f64 = interior
                .iter()
                .map(|c| {
                    let d = (grid.get(*c) - original.get(*c)) as f64;
                    d * d
                })
                .sum();
            (sum / interior.len() as f64).sqrt()
        };

        mean_filter(&mut grid, 2);

        let rms_after: f64 = {
            let sum: f64 = interior
                .iter()
                .map(|c| {
                    let d = (grid.get(*c) - original.get(*c)) as f64;
                    d * d
                })
                .sum();
            (sum / interior.len() as f64).sqrt()
        };

        assert!(
            rms_after < rms_before,
            "Mean filter should reduce RMS deviation on interior voxels: before={}, after={}",
            rms_before,
            rms_after
        );
    }

    #[test]
    fn median_filter_preserves_sphere_shape() {
        let mut grid = make_level_set_sphere(5.0, [0.0, 0.0, 0.0], 0.5, 3.0);
        let original = make_level_set_sphere(5.0, [0.0, 0.0, 0.0], 0.5, 3.0);

        // Add salt-and-pepper noise to a few voxels.
        let active: Vec<(Coord, f32)> = grid.tree().iter_active().collect();
        for (i, (coord, _)) in active.iter().enumerate() {
            if i % 17 == 0 {
                grid.set(*coord, 100.0); // outlier
            }
        }

        median_filter(&mut grid, 1);

        // After median filtering, the outliers should be suppressed.
        // Compute RMS error relative to the clean sphere.
        let mut sum_sq = 0.0f64;
        let mut count = 0u64;
        for (coord, _) in &active {
            let diff = (grid.get(*coord) - original.get(*coord)) as f64;
            sum_sq += diff * diff;
            count += 1;
        }
        let rms = (sum_sq / count as f64).sqrt();

        // Median filter should bring RMS well below the outlier amplitude.
        assert!(
            rms < 5.0,
            "Median filter should suppress outliers, RMS = {}",
            rms
        );
    }

    #[test]
    fn mean_filter_on_constant_field_is_noop() {
        let mut grid = Grid::<f32>::new(42.0, 1.0);
        // Fill a 7x7x7 cube with constant value 42 (same as background).
        // Since background matches the constant, all neighbors (even
        // outside the active region) return 42, making the filter a no-op.
        for x in 0..7 {
            for y in 0..7 {
                for z in 0..7 {
                    grid.set(Coord::new(x, y, z), 42.0);
                }
            }
        }

        mean_filter(&mut grid, 3);

        // All voxels should remain 42 since every neighbor is also 42.
        let val = grid.get(Coord::new(3, 3, 3));
        assert!(
            (val - 42.0).abs() < 1e-5,
            "Constant field should not change under mean filter: got {}",
            val
        );
        // Check a boundary voxel too.
        let val_edge = grid.get(Coord::new(0, 0, 0));
        assert!(
            (val_edge - 42.0).abs() < 1e-5,
            "Edge voxel should also remain 42: got {}",
            val_edge
        );
    }
}

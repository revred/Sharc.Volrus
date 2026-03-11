//! Level-set tracking — narrow-band maintenance and velocity extension.
//!
//! After advection or CSG operations, the narrow band can develop holes
//! or push voxels outside the desired half-width.  These utilities restore
//! band integrity without full Eikonal re-initialization.

use crate::grid::Grid;
use crate::math::Coord;

/// Find all active voxels that straddle the zero level set.
///
/// A voxel is a "zero crossing" if it or any 6-connected neighbor has the
/// opposite sign (product ≤ 0).  This includes voxels on the zero itself.
pub fn find_zero_crossings(grid: &Grid<f32>) -> Vec<Coord> {
    const OFFSETS: [Coord; 6] = [
        Coord { x: 1, y: 0, z: 0 },
        Coord { x: -1, y: 0, z: 0 },
        Coord { x: 0, y: 1, z: 0 },
        Coord { x: 0, y: -1, z: 0 },
        Coord { x: 0, y: 0, z: 1 },
        Coord { x: 0, y: 0, z: -1 },
    ];

    let mut crossings = Vec::new();
    for (coord, val) in grid.tree().iter_active() {
        // Exact zero is always a crossing
        if val == 0.0 {
            crossings.push(coord);
            continue;
        }
        // Only compare against active neighbors to avoid false crossings at
        // the outer narrow-band boundary where the background value would
        // generate spurious sign changes far from the real iso-surface.
        for off in &OFFSETS {
            let nb = Coord::new(coord.x + off.x, coord.y + off.y, coord.z + off.z);
            if !grid.tree().is_active(nb) {
                continue;
            }
            let nb_val = grid.tree().get(nb);
            if val * nb_val < 0.0 {
                crossings.push(coord);
                break;
            }
        }
    }
    crossings
}

/// Deactivate all voxels whose |value| ≥ `half_width` (in world units).
///
/// `half_width` is given in voxels; multiplication by `voxel_size` converts
/// to world units for comparison with SDF values.
pub fn trim_narrow_band(grid: &mut Grid<f32>, half_width: f32) {
    let voxel_size = grid.transform().voxel_size as f32;
    let threshold = half_width * voxel_size;

    let to_deactivate: Vec<Coord> = grid
        .tree()
        .iter_active()
        .filter(|(_, v)| v.abs() >= threshold)
        .map(|(c, _)| c)
        .collect();

    for coord in to_deactivate {
        grid.tree_mut().deactivate(coord);
    }
    grid.tree_mut().remove_empty_leaves();
}

/// Rebuild SDF values in the narrow band from the zero-crossing voxels.
///
/// For each active voxel, recomputes the signed distance as the Euclidean
/// distance (in index space, scaled by `voxel_size`) to the nearest
/// zero-crossing voxel, preserving the sign of the original value.
/// Voxels whose recomputed distance exceeds `half_width * voxel_size` are
/// deactivated (trimmed).
///
/// This is a simplified redistancing — not full Eikonal, but sufficient for
/// maintaining band integrity after smooth advection steps.
pub fn rebuild_narrow_band(grid: &mut Grid<f32>, half_width: f32) {
    let voxel_size = grid.transform().voxel_size as f32;
    let band = half_width * voxel_size;

    let crossings = find_zero_crossings(grid);
    if crossings.is_empty() {
        return;
    }

    let active: Vec<(Coord, f32)> = grid.tree().iter_active().collect();

    for (coord, old_val) in active {
        let sign = if old_val >= 0.0 { 1.0f32 } else { -1.0f32 };

        let min_dist = crossings
            .iter()
            .map(|c| {
                let dx = (coord.x - c.x) as f32;
                let dy = (coord.y - c.y) as f32;
                let dz = (coord.z - c.z) as f32;
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .fold(f32::MAX, f32::min);

        let new_val = sign * min_dist * voxel_size;

        if new_val.abs() < band {
            grid.tree_mut().set(coord, new_val);
        } else {
            grid.tree_mut().deactivate(coord);
        }
    }

    grid.tree_mut().remove_empty_leaves();
}

/// Extend a velocity field from the zero-crossing voxels outward into the band.
///
/// Interface voxels (|sdf| < 1 voxel length) keep their existing velocity.
/// Off-interface band voxels receive the velocity of the nearest interface
/// voxel.  This prevents velocity blow-up far from the surface during
/// repeated advection steps.
///
/// # Arguments
/// * `sdf`        — signed-distance field defining the interface
/// * `vx/vy/vz`   — velocity component grids, modified in-place
/// * `half_width` — narrow band half-width in voxels
pub fn extend_velocity(
    sdf: &Grid<f32>,
    vx: &mut Grid<f32>,
    vy: &mut Grid<f32>,
    vz: &mut Grid<f32>,
    half_width: f32,
) {
    let voxel_size = sdf.transform().voxel_size as f32;
    let band = half_width * voxel_size;

    // Interface: voxels within one voxel of the surface
    let interface: Vec<Coord> = sdf
        .tree()
        .iter_active()
        .filter(|(_, v)| v.abs() < voxel_size)
        .map(|(c, _)| c)
        .collect();

    if interface.is_empty() {
        return;
    }

    // Off-interface band voxels
    let band_voxels: Vec<Coord> = sdf
        .tree()
        .iter_active()
        .filter(|(_, v)| v.abs() >= voxel_size && v.abs() < band)
        .map(|(c, _)| c)
        .collect();

    for coord in band_voxels {
        let src = interface
            .iter()
            .copied()
            .min_by_key(|&c| dist_sq(coord, c));

        if let Some(src_coord) = src {
            vx.set(coord, vx.get(src_coord));
            vy.set(coord, vy.get(src_coord));
            vz.set(coord, vz.get(src_coord));
        }
    }
}

#[inline]
fn dist_sq(a: Coord, b: Coord) -> i64 {
    let dx = (a.x - b.x) as i64;
    let dy = (a.y - b.y) as i64;
    let dz = (a.z - b.z) as i64;
    dx * dx + dy * dy + dz * dz
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use crate::tools::sphere::make_level_set_sphere;

    #[test]
    fn find_zero_crossings_sphere_nonempty() {
        let grid = make_level_set_sphere(5.0, [0.0, 0.0, 0.0], 1.0, 3.0);
        let crossings = find_zero_crossings(&grid);
        assert!(
            !crossings.is_empty(),
            "sphere should have zero crossings"
        );
        // All crossings should be close to the sphere surface at r=5
        for c in &crossings {
            let r = ((c.x * c.x + c.y * c.y + c.z * c.z) as f64).sqrt();
            assert!(
                (r - 5.0).abs() < 2.5,
                "crossing at ({},{},{}) r={} is too far from r=5",
                c.x, c.y, c.z, r
            );
        }
    }

    #[test]
    fn find_zero_crossings_empty_grid() {
        let grid = Grid::<f32>::level_set(1.0, 3.0);
        assert!(find_zero_crossings(&grid).is_empty());
    }

    #[test]
    fn trim_narrow_band_shrinks_count() {
        // Build sphere with half_width=5 (wide band), then trim to 2
        let mut grid = make_level_set_sphere(5.0, [0.0, 0.0, 0.0], 1.0, 5.0);
        let before = grid.tree().active_voxel_count();

        trim_narrow_band(&mut grid, 2.0);
        let after = grid.tree().active_voxel_count();

        assert!(
            after < before,
            "trim should remove voxels: before={} after={}",
            before, after
        );
        // Every surviving voxel must be within 2 voxels of the surface
        for (_, val) in grid.tree().iter_active() {
            assert!(
                val.abs() < 2.0,
                "voxel val={} exceeded trim threshold 2.0",
                val
            );
        }
    }

    #[test]
    fn trim_narrow_band_empty_grid_noop() {
        let mut grid = Grid::<f32>::level_set(1.0, 3.0);
        trim_narrow_band(&mut grid, 2.0);
        assert_eq!(grid.tree().active_voxel_count(), 0);
    }

    #[test]
    fn rebuild_narrow_band_preserves_sign() {
        let mut grid = make_level_set_sphere(5.0, [0.0, 0.0, 0.0], 1.0, 3.0);
        let before: Vec<(Coord, f32)> = grid.tree().iter_active().collect();

        rebuild_narrow_band(&mut grid, 3.0);

        for (coord, old_val) in &before {
            if grid.tree().is_active(*coord) {
                let new_val = grid.tree().get(*coord);
                // Allow a tiny zero-band exception
                if old_val.abs() > 0.1 {
                    assert_eq!(
                        old_val.signum(),
                        new_val.signum(),
                        "sign changed at ({},{},{}): {} -> {}",
                        coord.x, coord.y, coord.z, old_val, new_val
                    );
                }
            }
        }
    }

    #[test]
    fn rebuild_narrow_band_empty_grid_noop() {
        let mut grid = Grid::<f32>::level_set(1.0, 3.0);
        rebuild_narrow_band(&mut grid, 3.0);
        assert_eq!(grid.tree().active_voxel_count(), 0);
    }

    #[test]
    fn extend_velocity_fills_band_voxels() {
        let sdf = make_level_set_sphere(5.0, [0.0, 0.0, 0.0], 1.0, 3.0);
        let voxel_size = 1.0_f32;

        let mut vx = Grid::<f32>::new(0.0, voxel_size as f64);
        let mut vy = Grid::<f32>::new(0.0, voxel_size as f64);
        let mut vz = Grid::<f32>::new(0.0, voxel_size as f64);

        // Set velocity only at the interface (|sdf| < 1 voxel)
        for (coord, val) in sdf.tree().iter_active() {
            if val.abs() < voxel_size {
                vx.set(coord, 1.0);
                vy.set(coord, 2.0);
                vz.set(coord, 3.0);
            }
        }

        let before_active = vx.tree().active_voxel_count();
        extend_velocity(&sdf, &mut vx, &mut vy, &mut vz, 3.0);
        let after_active = vx.tree().active_voxel_count();

        assert!(
            after_active >= before_active,
            "extend_velocity should add voxels: before={} after={}",
            before_active, after_active
        );
    }

    #[test]
    fn extend_velocity_empty_sdf_noop() {
        let sdf = Grid::<f32>::level_set(1.0, 3.0);
        let mut vx = Grid::<f32>::new(0.0, 1.0);
        let mut vy = Grid::<f32>::new(0.0, 1.0);
        let mut vz = Grid::<f32>::new(0.0, 1.0);
        extend_velocity(&sdf, &mut vx, &mut vy, &mut vz, 3.0);
        assert_eq!(vx.tree().active_voxel_count(), 0);
    }
}

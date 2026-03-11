//! Point rasterization — convert particles into density or SDF grids.
//!
//! This bridges PointDataGrid → Grid<f32>:
//! - `rasterize_density()` — particle count per voxel → fog volume
//! - `rasterize_splatted()` — Gaussian splat with kernel radius
//! - `rasterize_to_sdf()` — point cloud → signed distance field via closest-point

use crate::grid::{Grid, GridClass};
use crate::math::Coord;
use crate::points::PointDataGrid;

/// Rasterize particles into a density (fog) grid.
///
/// Each active voxel gets a value = number of particles that fall within it.
/// The result is a FogVolume grid.
pub fn rasterize_density(points: &PointDataGrid) -> Grid<f32> {
    let voxel_size = points.voxel_size();
    let mut grid = Grid::new(0.0, voxel_size);
    grid.set_name("density");
    grid.set_grid_class(GridClass::FogVolume);

    for particle in points.iter_particles() {
        let coord = grid.transform().world_to_index(particle.position);
        let current = grid.tree().get(coord);
        grid.tree_mut().set(coord, current + 1.0);
    }

    grid
}

/// Rasterize particles with Gaussian splatting.
///
/// Each particle deposits a Gaussian kernel (exp(-r²/2σ²)) into nearby voxels.
/// `radius` is in world units (typically 2-3x voxel size).
/// `sigma` controls the Gaussian width (typically radius/3).
pub fn rasterize_splatted(
    points: &PointDataGrid,
    radius: f64,
    sigma: f64,
) -> Grid<f32> {
    let voxel_size = points.voxel_size();
    let mut grid = Grid::new(0.0, voxel_size);
    grid.set_name("density");
    grid.set_grid_class(GridClass::FogVolume);

    let r_voxels = (radius / voxel_size).ceil() as i32;
    let inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);

    for particle in points.iter_particles() {
        let center_f = grid.transform().world_to_index_f64(particle.position);
        let cx = center_f[0].round() as i32;
        let cy = center_f[1].round() as i32;
        let cz = center_f[2].round() as i32;

        for dz in -r_voxels..=r_voxels {
            for dy in -r_voxels..=r_voxels {
                for dx in -r_voxels..=r_voxels {
                    let coord = Coord::new(cx + dx, cy + dy, cz + dz);
                    let world_pos = grid.transform().index_to_world(coord);
                    let dist2 = (world_pos[0] - particle.position[0]).powi(2)
                        + (world_pos[1] - particle.position[1]).powi(2)
                        + (world_pos[2] - particle.position[2]).powi(2);

                    if dist2 <= radius * radius {
                        let weight = (-dist2 * inv_2sigma2).exp() as f32;
                        let current = grid.tree().get(coord);
                        grid.tree_mut().set(coord, current + weight);
                    }
                }
            }
        }
    }

    grid
}

/// Rasterize a point cloud into a signed distance field (SDF).
///
/// For each voxel in a narrow band, computes the distance to the nearest particle.
/// `half_width` controls the narrow-band width (in voxels).
/// The result is a LevelSet grid.
pub fn rasterize_to_sdf(
    points: &PointDataGrid,
    half_width: f64,
) -> Grid<f32> {
    let voxel_size = points.voxel_size();
    let background = (half_width * voxel_size) as f32;
    let mut grid = Grid::new(background, voxel_size);
    grid.set_name("sdf");
    grid.set_grid_class(GridClass::LevelSet);

    let half_w_voxels = half_width.ceil() as i32;

    // For each particle, stamp distance values into nearby voxels
    for particle in points.iter_particles() {
        let center_f = grid.transform().world_to_index_f64(particle.position);
        let cx = center_f[0].round() as i32;
        let cy = center_f[1].round() as i32;
        let cz = center_f[2].round() as i32;

        for dz in -half_w_voxels..=half_w_voxels {
            for dy in -half_w_voxels..=half_w_voxels {
                for dx in -half_w_voxels..=half_w_voxels {
                    let coord = Coord::new(cx + dx, cy + dy, cz + dz);
                    let world_pos = grid.transform().index_to_world(coord);
                    let dist = ((world_pos[0] - particle.position[0]).powi(2)
                        + (world_pos[1] - particle.position[1]).powi(2)
                        + (world_pos[2] - particle.position[2]).powi(2))
                    .sqrt() as f32;

                    if dist < background {
                        let current = grid.tree().get(coord);
                        if dist < current {
                            grid.tree_mut().set(coord, dist);
                        }
                    }
                }
            }
        }
    }

    grid
}

/// Convert a fog volume density grid into a level set by thresholding.
///
/// Voxels above `iso` are inside (negative distance), below are outside (positive).
/// The resulting SDF is approximate — for exact SDF, use mesh_to_level_set.
pub fn density_to_level_set(density: &Grid<f32>, iso: f32, half_width: f32) -> Grid<f32> {
    let background = half_width * density.transform().voxel_size as f32;
    let mut sdf = Grid::new(background, density.transform().voxel_size);
    sdf.set_name("sdf");
    sdf.set_grid_class(GridClass::LevelSet);

    for (coord, value) in density.tree().iter_active() {
        // Simple threshold-based SDF approximation
        let dist = (iso - value) * density.transform().voxel_size as f32;
        let clamped = dist.clamp(-background, background);
        sdf.tree_mut().set(coord, clamped);
    }

    sdf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::points::Particle;

    fn make_test_particles(n: usize, voxel_size: f64) -> PointDataGrid {
        let mut pg = PointDataGrid::new(voxel_size);
        for i in 0..n {
            pg.insert(Particle {
                position: [i as f64 * voxel_size, 0.0, 0.0],
                velocity: [0.0; 3],
                id: i as u64,
            });
        }
        pg
    }

    #[test]
    fn rasterize_density_single_particle() {
        let mut pg = PointDataGrid::new(1.0);
        pg.insert(Particle {
            position: [0.5, 0.5, 0.5],
            velocity: [0.0; 3],
            id: 0,
        });
        let grid = rasterize_density(&pg);
        assert_eq!(grid.grid_class(), GridClass::FogVolume);
        // Single particle → single active voxel with value 1.0
        let count: usize = grid.tree().iter_active().count();
        assert_eq!(count, 1);
        let (_, val) = grid.tree().iter_active().next().unwrap();
        assert_eq!(val, 1.0);
    }

    #[test]
    fn rasterize_density_two_particles_same_voxel() {
        let mut pg = PointDataGrid::new(1.0);
        pg.insert(Particle {
            position: [0.1, 0.1, 0.1],
            velocity: [0.0; 3],
            id: 0,
        });
        pg.insert(Particle {
            position: [0.9, 0.9, 0.9],
            velocity: [0.0; 3],
            id: 1,
        });
        let grid = rasterize_density(&pg);
        // Both particles map to Coord(0,0,0) or (1,1,1) — either way, density should be 2
        let total: f32 = grid.tree().iter_active().map(|(_, v)| v).sum();
        assert_eq!(total, 2.0);
    }

    #[test]
    fn rasterize_density_multiple_voxels() {
        let pg = make_test_particles(5, 1.0);
        let grid = rasterize_density(&pg);
        let active_count = grid.tree().active_voxel_count();
        assert!(active_count >= 3); // particles at 0,1,2,3,4 — at least 3 distinct voxels
    }

    #[test]
    fn rasterize_density_empty_particles() {
        let pg = PointDataGrid::new(1.0);
        let grid = rasterize_density(&pg);
        assert_eq!(grid.tree().active_voxel_count(), 0);
    }

    #[test]
    fn rasterize_splatted_kernel_spreads() {
        let mut pg = PointDataGrid::new(1.0);
        pg.insert(Particle {
            position: [4.0, 4.0, 4.0],
            velocity: [0.0; 3],
            id: 0,
        });
        let grid = rasterize_splatted(&pg, 2.0, 0.7);
        // Single particle with radius=2 voxels should splat into multiple voxels
        let active = grid.tree().active_voxel_count();
        assert!(active > 1, "splatted should spread to neighbors, got {active}");
        // Center voxel should have highest value
        let center_val = grid.tree().get(Coord::new(4, 4, 4));
        assert!(center_val > 0.0);
    }

    #[test]
    fn rasterize_splatted_empty_particles() {
        let pg = PointDataGrid::new(1.0);
        let grid = rasterize_splatted(&pg, 2.0, 0.7);
        assert_eq!(grid.tree().active_voxel_count(), 0);
    }

    #[test]
    fn rasterize_to_sdf_single_point() {
        let mut pg = PointDataGrid::new(1.0);
        pg.insert(Particle {
            position: [4.0, 4.0, 4.0],
            velocity: [0.0; 3],
            id: 0,
        });
        let grid = rasterize_to_sdf(&pg, 3.0);
        assert_eq!(grid.grid_class(), GridClass::LevelSet);

        // Voxel at particle center should have distance ≈ 0
        let center_val = grid.tree().get(Coord::new(4, 4, 4));
        assert!(center_val < 1.0, "center dist should be small, got {center_val}");

        // Voxel far away should be background
        let far_val = grid.tree().get(Coord::new(100, 100, 100));
        assert!((far_val - 3.0).abs() < 0.01, "far voxel should be background");
    }

    #[test]
    fn rasterize_to_sdf_multiple_points_min_distance() {
        let mut pg = PointDataGrid::new(1.0);
        pg.insert(Particle {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0; 3],
            id: 0,
        });
        pg.insert(Particle {
            position: [5.0, 0.0, 0.0],
            velocity: [0.0; 3],
            id: 1,
        });
        let grid = rasterize_to_sdf(&pg, 4.0);
        // Midpoint between particles: distance should be ≈ 2.5 (closer to either)
        let mid_val = grid.tree().get(Coord::new(3, 0, 0));
        assert!(mid_val < 4.0, "midpoint should be in narrow band");
    }

    #[test]
    fn density_to_level_set_threshold() {
        let mut density = Grid::new(0.0, 1.0);
        density.set_grid_class(GridClass::FogVolume);
        // High density region
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..5 {
                    density.tree_mut().set(Coord::new(x, y, z), 10.0);
                }
            }
        }
        // Low density outside
        density.tree_mut().set(Coord::new(10, 10, 10), 0.1);

        let sdf = density_to_level_set(&density, 5.0, 3.0);
        assert_eq!(sdf.grid_class(), GridClass::LevelSet);

        // High-density voxel (10.0 > iso=5.0) → negative distance (inside)
        let inside = sdf.tree().get(Coord::new(2, 2, 2));
        assert!(inside < 0.0, "inside should be negative, got {inside}");

        // Low-density voxel (0.1 < iso=5.0) → positive distance (outside)
        let outside = sdf.tree().get(Coord::new(10, 10, 10));
        assert!(outside > 0.0, "outside should be positive, got {outside}");
    }
}

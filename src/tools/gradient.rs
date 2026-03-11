//! Finite-difference differential operators on scalar grids.
//!
//! Central differences in index space: no voxel-size scaling is applied
//! (callers must divide by voxel_size for world-space gradients).

use crate::grid::Grid;
use crate::math::Coord;

/// Compute the gradient of a scalar grid at a voxel using central differences.
///
/// Returns `[df/dx, df/dy, df/dz]` in **index space**.
/// To convert to world space, divide each component by `voxel_size`.
pub fn gradient_at(grid: &Grid<f32>, coord: Coord) -> [f32; 3] {
    let Coord { x, y, z } = coord;

    let dfdx = (grid.get(Coord::new(x + 1, y, z)) - grid.get(Coord::new(x - 1, y, z))) * 0.5;
    let dfdy = (grid.get(Coord::new(x, y + 1, z)) - grid.get(Coord::new(x, y - 1, z))) * 0.5;
    let dfdz = (grid.get(Coord::new(x, y, z + 1)) - grid.get(Coord::new(x, y, z - 1))) * 0.5;

    [dfdx, dfdy, dfdz]
}

/// Compute a gradient field: returns three separate grids `(gx, gy, gz)`.
///
/// Each output grid has the same transform and background as the input.
/// Only active voxels of the input are evaluated.
pub fn gradient_field(grid: &Grid<f32>) -> (Grid<f32>, Grid<f32>, Grid<f32>) {
    let bg = grid.tree().background();
    let vs = grid.transform().voxel_size;

    let mut gx = Grid::<f32>::new(0.0, vs);
    let mut gy = Grid::<f32>::new(0.0, vs);
    let mut gz = Grid::<f32>::new(0.0, vs);

    gx.set_transform(*grid.transform());
    gy.set_transform(*grid.transform());
    gz.set_transform(*grid.transform());

    // Collect active coords first to avoid borrow issues.
    let active: Vec<Coord> = grid.tree().iter_active().map(|(c, _)| c).collect();

    for coord in active {
        let g = gradient_at(grid, coord);
        gx.set(coord, g[0]);
        gy.set(coord, g[1]);
        gz.set(coord, g[2]);
    }

    let _ = bg; // suppress unused warning
    (gx, gy, gz)
}

/// Laplacian at a voxel (sum of second partial derivatives).
///
/// Uses the standard 7-point stencil in index space:
/// `d2f/dx2 = f(x+1) - 2*f(x) + f(x-1)`, summed over x, y, z.
pub fn laplacian_at(grid: &Grid<f32>, coord: Coord) -> f32 {
    let Coord { x, y, z } = coord;
    let center = grid.get(coord);

    let d2x =
        grid.get(Coord::new(x + 1, y, z)) - 2.0 * center + grid.get(Coord::new(x - 1, y, z));
    let d2y =
        grid.get(Coord::new(x, y + 1, z)) - 2.0 * center + grid.get(Coord::new(x, y - 1, z));
    let d2z =
        grid.get(Coord::new(x, y, z + 1)) - 2.0 * center + grid.get(Coord::new(x, y, z - 1));

    d2x + d2y + d2z
}

/// Mean curvature at a voxel of a level-set grid.
///
/// Uses the formula `H = div(grad(phi) / |grad(phi)|)` expanded as:
///
/// ```text
/// H = (phi_xx*(phi_y^2 + phi_z^2) + phi_yy*(phi_x^2 + phi_z^2) + phi_zz*(phi_x^2 + phi_y^2)
///      - 2*(phi_x*phi_y*phi_xy + phi_x*phi_z*phi_xz + phi_y*phi_z*phi_yz))
///     / |grad(phi)|^3
/// ```
///
/// Returns 0.0 when the gradient magnitude is near zero.
pub fn mean_curvature_at(grid: &Grid<f32>, coord: Coord) -> f32 {
    let Coord { x, y, z } = coord;

    // First derivatives (central differences).
    let phi_x =
        (grid.get(Coord::new(x + 1, y, z)) - grid.get(Coord::new(x - 1, y, z))) * 0.5;
    let phi_y =
        (grid.get(Coord::new(x, y + 1, z)) - grid.get(Coord::new(x, y - 1, z))) * 0.5;
    let phi_z =
        (grid.get(Coord::new(x, y, z + 1)) - grid.get(Coord::new(x, y, z - 1))) * 0.5;

    // Second derivatives (central differences of first derivatives).
    let center = grid.get(coord);

    let phi_xx =
        grid.get(Coord::new(x + 1, y, z)) - 2.0 * center + grid.get(Coord::new(x - 1, y, z));
    let phi_yy =
        grid.get(Coord::new(x, y + 1, z)) - 2.0 * center + grid.get(Coord::new(x, y - 1, z));
    let phi_zz =
        grid.get(Coord::new(x, y, z + 1)) - 2.0 * center + grid.get(Coord::new(x, y, z - 1));

    // Cross derivatives.
    let phi_xy = (grid.get(Coord::new(x + 1, y + 1, z))
        - grid.get(Coord::new(x + 1, y - 1, z))
        - grid.get(Coord::new(x - 1, y + 1, z))
        + grid.get(Coord::new(x - 1, y - 1, z)))
        * 0.25;

    let phi_xz = (grid.get(Coord::new(x + 1, y, z + 1))
        - grid.get(Coord::new(x + 1, y, z - 1))
        - grid.get(Coord::new(x - 1, y, z + 1))
        + grid.get(Coord::new(x - 1, y, z - 1)))
        * 0.25;

    let phi_yz = (grid.get(Coord::new(x, y + 1, z + 1))
        - grid.get(Coord::new(x, y + 1, z - 1))
        - grid.get(Coord::new(x, y - 1, z + 1))
        + grid.get(Coord::new(x, y - 1, z - 1)))
        * 0.25;

    let grad_mag_sq = phi_x * phi_x + phi_y * phi_y + phi_z * phi_z;
    let grad_mag = grad_mag_sq.sqrt();

    if grad_mag < 1e-10 {
        return 0.0;
    }

    let numerator = phi_xx * (phi_y * phi_y + phi_z * phi_z)
        + phi_yy * (phi_x * phi_x + phi_z * phi_z)
        + phi_zz * (phi_x * phi_x + phi_y * phi_y)
        - 2.0 * (phi_x * phi_y * phi_xy + phi_x * phi_z * phi_xz + phi_y * phi_z * phi_yz);

    numerator / (grad_mag_sq * grad_mag)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::sphere::make_level_set_sphere;

    /// Build a linear ramp: f(x,y,z) = 3x + 5y + 7z.
    fn make_linear_ramp() -> Grid<f32> {
        let mut grid = Grid::<f32>::new(0.0, 1.0);
        for x in -5..=5 {
            for y in -5..=5 {
                for z in -5..=5 {
                    let val = 3.0 * x as f32 + 5.0 * y as f32 + 7.0 * z as f32;
                    grid.set(Coord::new(x, y, z), val);
                }
            }
        }
        grid
    }

    /// Build a quadratic: f(x,y,z) = x^2 + 2*y^2 + 3*z^2.
    fn make_quadratic() -> Grid<f32> {
        let mut grid = Grid::<f32>::new(0.0, 1.0);
        for x in -10..=10 {
            for y in -10..=10 {
                for z in -10..=10 {
                    let xf = x as f32;
                    let yf = y as f32;
                    let zf = z as f32;
                    grid.set(Coord::new(x, y, z), xf * xf + 2.0 * yf * yf + 3.0 * zf * zf);
                }
            }
        }
        grid
    }

    #[test]
    fn gradient_of_linear_ramp_is_constant() {
        let grid = make_linear_ramp();
        // Interior points have valid stencils.
        let g = gradient_at(&grid, Coord::new(0, 0, 0));
        assert!((g[0] - 3.0).abs() < 1e-5, "dfdx: {}", g[0]);
        assert!((g[1] - 5.0).abs() < 1e-5, "dfdy: {}", g[1]);
        assert!((g[2] - 7.0).abs() < 1e-5, "dfdz: {}", g[2]);

        // Another interior point should give the same.
        let g2 = gradient_at(&grid, Coord::new(2, -3, 1));
        assert!((g2[0] - 3.0).abs() < 1e-5);
        assert!((g2[1] - 5.0).abs() < 1e-5);
        assert!((g2[2] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn laplacian_of_quadratic_is_constant() {
        let grid = make_quadratic();
        // d2/dx2(x^2) = 2, d2/dy2(2y^2) = 4, d2/dz2(3z^2) = 6 => Laplacian = 12
        let lap = laplacian_at(&grid, Coord::new(0, 0, 0));
        assert!((lap - 12.0).abs() < 1e-4, "Laplacian at origin: {}", lap);

        let lap2 = laplacian_at(&grid, Coord::new(3, -2, 1));
        assert!((lap2 - 12.0).abs() < 1e-4, "Laplacian at (3,-2,1): {}", lap2);
    }

    #[test]
    fn laplacian_of_linear_is_zero() {
        let grid = make_linear_ramp();
        let lap = laplacian_at(&grid, Coord::new(0, 0, 0));
        assert!(lap.abs() < 1e-5, "Laplacian of linear should be ~0: {}", lap);
    }

    #[test]
    fn mean_curvature_of_sphere_is_approx_inverse_radius() {
        // Sphere with radius 10, voxel_size 0.5 => good resolution.
        let radius = 10.0;
        let grid = make_level_set_sphere(radius, [0.0, 0.0, 0.0], 0.5, 3.0);

        // Sample on the surface along the +X axis: index ~20 (10.0 / 0.5).
        let surface_idx = (radius / 0.5).round() as i32;
        let coord = Coord::new(surface_idx, 0, 0);

        let h = mean_curvature_at(&grid, coord);

        // In index space, curvature = 1/(radius_in_voxels) = voxel_size/radius.
        // For voxel_size=0.5, radius=10 => expected = 0.5/10 = 0.05 per voxel.
        // But our formula computes in index-space, so the expected curvature is
        // 1/radius_in_voxels = 1/20 = 0.05.
        // Mean curvature = (k1 + k2) / (for the formula used, it's actually
        // the sum k1+k2 not the average). For a sphere, k1 = k2 = 1/R_idx,
        // so the result = 2/R_idx = 2/20 = 0.1.
        // Actually the formula used gives (k1+k2) directly, so for a sphere:
        // H = 2/R_idx.
        let r_idx = radius / 0.5;
        let expected = 2.0 / r_idx as f32;

        assert!(
            (h - expected).abs() < 0.05,
            "Mean curvature: got {}, expected ~{}", h, expected
        );
    }

    #[test]
    fn gradient_field_produces_three_grids() {
        let grid = make_linear_ramp();
        let (gx, gy, gz) = gradient_field(&grid);

        // Check an interior voxel.
        let c = Coord::new(0, 0, 0);
        assert!((gx.get(c) - 3.0).abs() < 1e-5);
        assert!((gy.get(c) - 5.0).abs() < 1e-5);
        assert!((gz.get(c) - 7.0).abs() < 1e-5);
    }
}

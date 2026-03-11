//! Semi-Lagrangian advection for level sets and particle advection through
//! velocity fields.

use crate::grid::Grid;
use crate::math::Coord;
use crate::points::PointDataGrid;
use crate::tools::interpolation::sample_trilinear;

/// Velocity field: three scalar grids (vx, vy, vz).
pub struct VelocityField {
    pub vx: Grid<f32>,
    pub vy: Grid<f32>,
    pub vz: Grid<f32>,
}

impl VelocityField {
    /// Sample velocity at a world-space position using trilinear interpolation.
    pub fn sample(&self, pos: [f64; 3]) -> [f64; 3] {
        [
            sample_trilinear(&self.vx, pos) as f64,
            sample_trilinear(&self.vy, pos) as f64,
            sample_trilinear(&self.vz, pos) as f64,
        ]
    }
}

/// Advect a level-set grid through a velocity field using the semi-Lagrangian method.
///
/// For each active voxel at position x, traces backward: x_prev = x - dt * v(x),
/// then samples the SDF at x_prev using trilinear interpolation and writes that
/// value to the output grid at x.
pub fn advect_level_set(grid: &Grid<f32>, velocity: &VelocityField, dt: f64) -> Grid<f32> {
    let affine = grid.affine_map();
    let bg = grid.tree().background();
    let mut output = Grid::<f32>::with_affine(bg, *affine);
    output.set_name(grid.name());
    output.set_grid_class(grid.grid_class());

    for (coord, _val) in grid.tree().iter_active() {
        let world = affine.index_to_world(coord);
        let v = velocity.sample(world);

        // Trace backward
        let x_prev = [
            world[0] - dt * v[0],
            world[1] - dt * v[1],
            world[2] - dt * v[2],
        ];

        // Sample SDF at previous position
        let new_val = sample_trilinear(grid, x_prev);
        output.set(coord, new_val);
    }

    output
}

/// Advect particles through a velocity field using RK4 integration.
///
/// After advection, particles are re-binned into leaves since their
/// positions may have moved to different leaf regions.
pub fn advect_particles(points: &mut PointDataGrid, velocity: &VelocityField, dt: f64) {
    for p in points.iter_particles_mut() {
        let pos = p.position;

        // k1 = dt * v(pos)
        let v1 = velocity.sample(pos);
        let k1 = [dt * v1[0], dt * v1[1], dt * v1[2]];

        // k2 = dt * v(pos + k1/2)
        let p2 = [pos[0] + k1[0] * 0.5, pos[1] + k1[1] * 0.5, pos[2] + k1[2] * 0.5];
        let v2 = velocity.sample(p2);
        let k2 = [dt * v2[0], dt * v2[1], dt * v2[2]];

        // k3 = dt * v(pos + k2/2)
        let p3 = [pos[0] + k2[0] * 0.5, pos[1] + k2[1] * 0.5, pos[2] + k2[2] * 0.5];
        let v3 = velocity.sample(p3);
        let k3 = [dt * v3[0], dt * v3[1], dt * v3[2]];

        // k4 = dt * v(pos + k3)
        let p4 = [pos[0] + k3[0], pos[1] + k3[1], pos[2] + k3[2]];
        let v4 = velocity.sample(p4);
        let k4 = [dt * v4[0], dt * v4[1], dt * v4[2]];

        // new_pos = pos + (k1 + 2*k2 + 2*k3 + k4) / 6
        p.position = [
            pos[0] + (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            pos[1] + (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            pos[2] + (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
        ];
    }

    // Re-bin particles into correct leaves after position update
    points.rebin();
}

/// Advect particles using simple forward Euler (faster, less accurate).
///
/// new_pos = pos + dt * v(pos)
///
/// After advection, particles are re-binned into leaves.
pub fn advect_particles_euler(points: &mut PointDataGrid, velocity: &VelocityField, dt: f64) {
    for p in points.iter_particles_mut() {
        let v = velocity.sample(p.position);
        p.position[0] += dt * v[0];
        p.position[1] += dt * v[1];
        p.position[2] += dt * v[2];
    }

    // Re-bin particles into correct leaves after position update
    points.rebin();
}

/// Helper: create a uniform velocity field where every voxel has the same velocity.
/// Useful for testing.
fn make_uniform_velocity(voxel_size: f64, vel: [f32; 3], bbox_min: Coord, bbox_max: Coord) -> VelocityField {
    let mut vx = Grid::<f32>::new(vel[0], voxel_size);
    let mut vy = Grid::<f32>::new(vel[1], voxel_size);
    let mut vz = Grid::<f32>::new(vel[2], voxel_size);

    // Set some active voxels so the field is well-defined in the region.
    // For a uniform field, the background value already gives the correct
    // interpolation result everywhere.
    let c = bbox_min;
    vx.set(c, vel[0]);
    vy.set(c, vel[1]);
    vz.set(c, vel[2]);
    let c = bbox_max;
    vx.set(c, vel[0]);
    vy.set(c, vel[1]);
    vz.set(c, vel[2]);

    VelocityField { vx, vy, vz }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Coord;
    use crate::points::Particle;
    use crate::tools::sphere::make_level_set_sphere;

    fn make_particle(id: u64, x: f64, y: f64, z: f64) -> Particle {
        Particle {
            position: [x, y, z],
            velocity: [0.0, 0.0, 0.0],
            id,
        }
    }

    #[test]
    fn uniform_velocity_moves_sphere_sdf() {
        // Create a sphere SDF centered at origin.
        // radius=5, voxel_size=0.5, half_width=3 -> background=1.5
        // Narrow band stores voxels where |SDF| < 1.5, i.e., distances 3.5..6.5 from center.
        let voxel_size = 0.5;
        let half_width = 3.0;
        let grid = make_level_set_sphere(5.0, [0.0, 0.0, 0.0], voxel_size, half_width);

        // Uniform velocity in +x direction: (1, 0, 0)
        let velocity = make_uniform_velocity(
            voxel_size,
            [1.0, 0.0, 0.0],
            Coord::new(-30, -30, -30),
            Coord::new(30, 30, 30),
        );

        let dt = 1.0;
        let advected = advect_level_set(&grid, &velocity, dt);

        // After advecting with v=(1,0,0) for dt=1, the sphere effectively
        // shifts by +1 in x. Original surface at x=5 moves to x=6.
        let val_at_new_surface = sample_trilinear(&advected, [6.0, 0.0, 0.0]);
        assert!(
            val_at_new_surface.abs() < 1.0,
            "SDF at shifted surface should be near zero, got {}",
            val_at_new_surface
        );

        // Just inside the shifted narrow band: at x=5 (was x=4 before shift),
        // original SDF(4,0,0) = 4-5 = -1 (inside, within narrow band).
        let val_inside = sample_trilinear(&advected, [5.0, 0.0, 0.0]);
        assert!(
            val_inside < 0.0,
            "SDF inside shifted narrow band should be negative, got {}",
            val_inside
        );

        // Outside the shifted sphere: at x=7 (was x=6 before shift),
        // original SDF(6,0,0) = 6-5 = 1 (outside, within narrow band).
        let val_outside = sample_trilinear(&advected, [7.0, 0.0, 0.0]);
        assert!(
            val_outside > 0.0,
            "SDF outside shifted sphere should be positive, got {}",
            val_outside
        );
    }

    #[test]
    fn rk4_advection_uniform_field() {
        let mut points = PointDataGrid::new(1.0);
        points.insert(make_particle(1, 0.0, 0.0, 0.0));

        let velocity = make_uniform_velocity(
            1.0,
            [1.0, 2.0, 3.0],
            Coord::new(-5, -5, -5),
            Coord::new(15, 15, 15),
        );

        let dt = 1.0;
        advect_particles(&mut points, &velocity, dt);

        let p = points.iter_particles().next().unwrap();
        // For uniform field, RK4 gives exact result: pos + dt * v
        assert!(
            (p.position[0] - 1.0).abs() < 1e-6,
            "x: expected 1.0, got {}",
            p.position[0]
        );
        assert!(
            (p.position[1] - 2.0).abs() < 1e-6,
            "y: expected 2.0, got {}",
            p.position[1]
        );
        assert!(
            (p.position[2] - 3.0).abs() < 1e-6,
            "z: expected 3.0, got {}",
            p.position[2]
        );
    }

    #[test]
    fn rk4_vs_euler_accuracy_rotating_field() {
        // Create a rotating velocity field: v = (-y, x, 0)
        // A particle at (1, 0, 0) should travel in a circle.
        // For a circle of radius 1, after time t the exact position is (cos(t), sin(t), 0).
        let voxel_size = 0.1;

        // Build velocity grids for a rotating field around origin
        let mut vx = Grid::<f32>::new(0.0, voxel_size);
        let mut vy = Grid::<f32>::new(0.0, voxel_size);
        let vz = Grid::<f32>::new(0.0, voxel_size);

        // Populate velocity field: v = (-y, x, 0) in a region
        for ix in -20i32..=20 {
            for iy in -20i32..=20 {
                let coord = Coord::new(ix, iy, 0);
                let world = [ix as f64 * voxel_size, iy as f64 * voxel_size, 0.0];
                vx.set(coord, -world[1] as f32);
                vy.set(coord, world[0] as f32);
            }
        }

        let velocity = VelocityField { vx, vy, vz };

        // Advect with RK4
        let mut points_rk4 = PointDataGrid::new(voxel_size);
        points_rk4.insert(make_particle(1, 1.0, 0.0, 0.0));

        // Advect with Euler
        let mut points_euler = PointDataGrid::new(voxel_size);
        points_euler.insert(make_particle(1, 1.0, 0.0, 0.0));

        let dt = 0.1;
        let steps = 10; // total time = 1.0

        for _ in 0..steps {
            advect_particles(&mut points_rk4, &velocity, dt);
            advect_particles_euler(&mut points_euler, &velocity, dt);
        }

        // Exact position at t=1: (cos(1), sin(1), 0) = (0.5403, 0.8415, 0)
        let exact = [1.0_f64.cos(), 1.0_f64.sin(), 0.0];

        let p_rk4 = points_rk4.iter_particles().next().unwrap();
        let p_euler = points_euler.iter_particles().next().unwrap();

        let err_rk4 = ((p_rk4.position[0] - exact[0]).powi(2)
            + (p_rk4.position[1] - exact[1]).powi(2))
            .sqrt();
        let err_euler = ((p_euler.position[0] - exact[0]).powi(2)
            + (p_euler.position[1] - exact[1]).powi(2))
            .sqrt();

        // RK4 should be significantly more accurate than Euler
        assert!(
            err_rk4 < err_euler,
            "RK4 error ({}) should be less than Euler error ({})",
            err_rk4,
            err_euler
        );
        // RK4 should be quite accurate for this case
        assert!(
            err_rk4 < 0.01,
            "RK4 error should be small, got {}",
            err_rk4
        );
    }

    #[test]
    fn advect_particles_then_verify_moved() {
        let mut points = PointDataGrid::new(1.0);
        points.insert(make_particle(1, 0.0, 0.0, 0.0));
        points.insert(make_particle(2, 5.0, 5.0, 5.0));

        let velocity = make_uniform_velocity(
            1.0,
            [1.0, 0.0, 0.0],
            Coord::new(-5, -5, -5),
            Coord::new(15, 15, 15),
        );

        advect_particles(&mut points, &velocity, 3.0);

        // Both particles should have moved +3 in x
        for p in points.iter_particles() {
            if p.id == 1 {
                assert!(
                    (p.position[0] - 3.0).abs() < 1e-6,
                    "Particle 1 x: expected 3.0, got {}",
                    p.position[0]
                );
            } else if p.id == 2 {
                assert!(
                    (p.position[0] - 8.0).abs() < 1e-6,
                    "Particle 2 x: expected 8.0, got {}",
                    p.position[0]
                );
            }
        }
    }

    #[test]
    fn particle_rebinning_after_advection() {
        let mut points = PointDataGrid::new(1.0);
        // Particle starts in leaf origin (0,0,0)
        points.insert(make_particle(1, 1.0, 1.0, 1.0));
        assert_eq!(points.leaf_count(), 1);

        let velocity = make_uniform_velocity(
            1.0,
            [20.0, 0.0, 0.0],
            Coord::new(-5, -5, -5),
            Coord::new(25, 5, 5),
        );

        advect_particles(&mut points, &velocity, 1.0);

        // Particle should have moved to ~(21, 1, 1), which is in leaf origin (16, 0, 0)
        assert_eq!(points.particle_count(), 1);
        let p = points.iter_particles().next().unwrap();
        assert!(
            (p.position[0] - 21.0).abs() < 1e-6,
            "Expected x=21, got {}",
            p.position[0]
        );

        // Verify it's in the correct leaf now
        let leaf_origin = Coord::new(16, 0, 0);
        let bbox = points.active_bbox().unwrap();
        assert!(bbox.contains(leaf_origin));
    }

    #[test]
    fn euler_advection_uniform_field() {
        let mut points = PointDataGrid::new(1.0);
        points.insert(make_particle(1, 0.0, 0.0, 0.0));

        let velocity = make_uniform_velocity(
            1.0,
            [2.0, -1.0, 0.5],
            Coord::new(-5, -5, -5),
            Coord::new(10, 10, 10),
        );

        advect_particles_euler(&mut points, &velocity, 1.0);

        let p = points.iter_particles().next().unwrap();
        assert!(
            (p.position[0] - 2.0).abs() < 1e-6,
            "x: expected 2.0, got {}",
            p.position[0]
        );
        assert!(
            (p.position[1] - (-1.0)).abs() < 1e-6,
            "y: expected -1.0, got {}",
            p.position[1]
        );
        assert!(
            (p.position[2] - 0.5).abs() < 1e-6,
            "z: expected 0.5, got {}",
            p.position[2]
        );
    }
}

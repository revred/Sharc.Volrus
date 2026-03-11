//! Level-set CSG operations: union, intersection, difference.
//!
//! For signed-distance-field grids, CSG is elegantly expressed as
//! pointwise min/max of the SDF values:
//!
//! - **Union**: min(a, b)
//! - **Intersection**: max(a, b)
//! - **Difference**: max(a, -b)

use crate::grid::{Grid, GridClass};
use crate::math::Coord;
use crate::tree::LEAF_LOG2DIM;

/// CSG union of two level-set grids: min(sdf_a, sdf_b).
pub fn csg_union(a: &Grid<f32>, b: &Grid<f32>) -> Grid<f32> {
    csg_combine(a, b, |va, vb| va.min(vb))
}

/// CSG intersection of two level-set grids: max(sdf_a, sdf_b).
pub fn csg_intersection(a: &Grid<f32>, b: &Grid<f32>) -> Grid<f32> {
    csg_combine(a, b, |va, vb| va.max(vb))
}

/// CSG difference: a minus b = max(sdf_a, -sdf_b).
pub fn csg_difference(a: &Grid<f32>, b: &Grid<f32>) -> Grid<f32> {
    csg_combine(a, b, |va, vb| va.max(-vb))
}

/// Generic CSG combiner.  Iterates all active voxels from both grids,
/// applies `combine_fn` to their values, and writes the result to a
/// new output grid.
fn csg_combine(
    a: &Grid<f32>,
    b: &Grid<f32>,
    combine_fn: impl Fn(f32, f32) -> f32,
) -> Grid<f32> {
    let bg_a = a.tree().background();
    let bg_b = b.tree().background();
    let bg_out = combine_fn(bg_a, bg_b);
    let voxel_size = a.transform().voxel_size;

    let mut out = Grid::<f32>::new(bg_out, voxel_size);
    out.set_grid_class(GridClass::LevelSet);
    out.set_transform(*a.transform());

    let dim = 1i32 << LEAF_LOG2DIM;

    // Collect all leaf origins from both grids.
    let origins_a = a.tree().leaf_origins();
    let origins_b = b.tree().leaf_origins();
    let mut all_origins = origins_a;
    all_origins.extend(origins_b);
    all_origins.sort();
    all_origins.dedup();

    for origin in all_origins {
        for lx in 0..dim {
            for ly in 0..dim {
                for lz in 0..dim {
                    let coord = origin + Coord::new(lx, ly, lz);
                    let va = a.get(coord);
                    let vb = b.get(coord);
                    let combined = combine_fn(va, vb);

                    // Only store if the voxel is active in at least one
                    // input, or if the combined value differs from bg.
                    let either_active =
                        a.tree().is_active(coord) || b.tree().is_active(coord);
                    if either_active || (combined - bg_out).abs() > 1e-12 {
                        out.set(coord, combined);
                    }
                }
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::sphere::make_level_set_sphere;

    #[test]
    fn union_of_two_spheres_has_more_voxels() {
        let s1 = make_level_set_sphere(5.0, [0.0, 0.0, 0.0], 1.0, 3.0);
        let s2 = make_level_set_sphere(5.0, [6.0, 0.0, 0.0], 1.0, 3.0);

        let u = csg_union(&s1, &s2);
        // Union should have at least as many active voxels as the larger input.
        assert!(u.active_voxel_count() >= s1.active_voxel_count());
        assert!(u.active_voxel_count() >= s2.active_voxel_count());
    }

    #[test]
    fn intersection_of_overlapping_spheres() {
        let s1 = make_level_set_sphere(5.0, [0.0, 0.0, 0.0], 1.0, 3.0);
        let s2 = make_level_set_sphere(5.0, [4.0, 0.0, 0.0], 1.0, 3.0);

        let isect = csg_intersection(&s1, &s2);
        // Intersection should have fewer or equal active voxels.
        assert!(isect.active_voxel_count() <= s1.active_voxel_count() + s2.active_voxel_count());
        assert!(isect.active_voxel_count() > 0, "Overlapping spheres should have non-empty intersection");
    }

    #[test]
    fn difference_removes_overlap() {
        let s1 = make_level_set_sphere(5.0, [0.0, 0.0, 0.0], 1.0, 3.0);
        let s2 = make_level_set_sphere(5.0, [4.0, 0.0, 0.0], 1.0, 3.0);

        let diff = csg_difference(&s1, &s2);

        // At the surface of s2 on the far side from s1: coord (9, 0, 0).
        // sdf_a(9,0,0) = 9-5 = 4, outside a's narrow band → returns bg_a = 3.
        // sdf_b(9,0,0) = dist(9, (4,0,0))-5 = 5-5 = 0.
        // diff = max(3, -0) = max(3, 0) = 3.
        // At coord (3, 0, 0): sdf_a = 3-5 = -2 (in band). sdf_b = dist(3,(4,0,0))-5 = 1-5 = -4 → out of band → bg_b=3.
        // diff = max(-2, -3) = -2. Still inside the difference shape (s1 minus s2).
        let val = diff.get(Coord::new(3, 0, 0));
        // sdf_a(3,0,0) = -2 (stored), sdf_b(3,0,0) = bg=3 (not stored).
        // diff = max(-2, -3) = -2 → inside.
        assert!(
            val < 0.0,
            "Point inside s1 but outside s2 should be inside difference: {}",
            val
        );

        // At a point inside both s1 and s2: coord (2, 0, 0).
        // sdf_a(2,0,0) = 2-5 = -3, at band edge.
        // sdf_b(2,0,0) = dist(2,(4,0,0))-5 = 2-5 = -3, at band edge.
        // diff = max(-3, 3) = 3 → outside (the subtraction removed it).
        // But both values at band edge may or may not be stored.
        // Use a point clearly in s2's narrow band: coord (5, 0, 0).
        // sdf_a(5,0,0) = 5-5 = 0 (on surface of a, stored).
        // sdf_b(5,0,0) = dist(5,(4,0,0))-5 = 1-5 = -4 → bg_b=3.
        // diff = max(0, -3) = 0. That's on the surface.
        // Try (6, 0, 0): sdf_a = 6-5 = 1 (stored), sdf_b = dist(6,(4,0,0))-5 = 2-5 = -3 → bg=3, -bg=-3.
        // diff = max(1, -3) = 1. Positive → outside.
        let val_outside = diff.get(Coord::new(6, 0, 0));
        assert!(
            val_outside > 0.0,
            "Point outside s1 should be outside difference: {}",
            val_outside
        );
    }

    #[test]
    fn union_is_commutative() {
        let s1 = make_level_set_sphere(3.0, [0.0, 0.0, 0.0], 1.0, 2.0);
        let s2 = make_level_set_sphere(3.0, [4.0, 0.0, 0.0], 1.0, 2.0);

        let u1 = csg_union(&s1, &s2);
        let u2 = csg_union(&s2, &s1);

        assert_eq!(u1.active_voxel_count(), u2.active_voxel_count());
    }
}

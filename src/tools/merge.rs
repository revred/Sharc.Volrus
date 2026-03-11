//! Tree merge operations — combine active regions between grids.
//!
//! - `merge_copy`         — copy all active voxels from `src` into `dst`.
//! - `merge_union`        — new grid with all active voxels from two grids (`a` wins at overlap).
//! - `merge_intersection` — new grid with only voxels active in both grids (values from `a`).

use crate::grid::Grid;

/// Copy all active voxels from `src` into `dst`.
///
/// Where `src` is active, the value overwrites `dst`.  Voxels active in `dst`
/// but not in `src` are left unchanged.
pub fn merge_copy<T: Copy + Default>(src: &Grid<T>, dst: &mut Grid<T>) {
    for (coord, value) in src.tree().iter_active() {
        dst.set(coord, value);
    }
}

/// Return a new grid that is the union of active regions from `a` and `b`.
///
/// Where both grids are active at the same coordinate, `a`'s value is kept.
/// The output shares `a`'s transform, name, and class.
pub fn merge_union<T: Copy + Default>(a: &Grid<T>, b: &Grid<T>) -> Grid<T> {
    let affine = a.affine_map();
    let bg = a.tree().background();
    let mut out = Grid::<T>::with_affine(bg, *affine);
    out.set_name(a.name());
    out.set_grid_class(a.grid_class());

    // Insert b's voxels first (lower priority)
    for (coord, value) in b.tree().iter_active() {
        out.set(coord, value);
    }
    // Insert a's voxels second (overwrites b at overlaps)
    for (coord, value) in a.tree().iter_active() {
        out.set(coord, value);
    }
    out
}

/// Return a new grid with only voxels active in BOTH `a` and `b`.
///
/// Values are taken from `a`.  The output shares `a`'s transform, name, and class.
pub fn merge_intersection<T: Copy + Default>(a: &Grid<T>, b: &Grid<T>) -> Grid<T> {
    let affine = a.affine_map();
    let bg = a.tree().background();
    let mut out = Grid::<T>::with_affine(bg, *affine);
    out.set_name(a.name());
    out.set_grid_class(a.grid_class());

    for (coord, value) in a.tree().iter_active() {
        if b.tree().is_active(coord) {
            out.set(coord, value);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Coord;

    fn grid(coords_vals: &[(Coord, f32)]) -> Grid<f32> {
        let mut g = Grid::<f32>::new(0.0, 1.0);
        for &(c, v) in coords_vals {
            g.set(c, v);
        }
        g
    }

    #[test]
    fn merge_copy_adds_src_voxels() {
        let src = grid(&[(Coord::new(0, 0, 0), 1.0), (Coord::new(1, 0, 0), 2.0)]);
        let mut dst = grid(&[(Coord::new(5, 5, 5), 9.0)]);
        merge_copy(&src, &mut dst);

        assert_eq!(dst.tree().active_voxel_count(), 3);
        assert_eq!(dst.get(Coord::new(0, 0, 0)), 1.0);
        assert_eq!(dst.get(Coord::new(1, 0, 0)), 2.0);
        assert_eq!(dst.get(Coord::new(5, 5, 5)), 9.0);
    }

    #[test]
    fn merge_copy_overwrites_dst_at_overlap() {
        let src = grid(&[(Coord::new(0, 0, 0), 99.0)]);
        let mut dst = grid(&[(Coord::new(0, 0, 0), 1.0)]);
        merge_copy(&src, &mut dst);

        assert_eq!(dst.get(Coord::new(0, 0, 0)), 99.0);
        assert_eq!(dst.tree().active_voxel_count(), 1);
    }

    #[test]
    fn merge_copy_empty_src_leaves_dst_unchanged() {
        let src = Grid::<f32>::new(0.0, 1.0);
        let mut dst = grid(&[(Coord::new(1, 2, 3), 5.0)]);
        merge_copy(&src, &mut dst);

        assert_eq!(dst.tree().active_voxel_count(), 1);
        assert_eq!(dst.get(Coord::new(1, 2, 3)), 5.0);
    }

    #[test]
    fn merge_union_contains_all_voxels() {
        let a = grid(&[(Coord::new(0, 0, 0), 1.0), (Coord::new(1, 0, 0), 2.0)]);
        let b = grid(&[(Coord::new(2, 0, 0), 3.0), (Coord::new(3, 0, 0), 4.0)]);
        let result = merge_union(&a, &b);

        assert_eq!(result.tree().active_voxel_count(), 4);
        assert_eq!(result.get(Coord::new(0, 0, 0)), 1.0);
        assert_eq!(result.get(Coord::new(2, 0, 0)), 3.0);
    }

    #[test]
    fn merge_union_a_wins_at_overlap() {
        let a = grid(&[(Coord::new(0, 0, 0), 10.0)]);
        let b = grid(&[(Coord::new(0, 0, 0), 20.0)]);
        let result = merge_union(&a, &b);

        assert_eq!(result.tree().active_voxel_count(), 1);
        assert_eq!(result.get(Coord::new(0, 0, 0)), 10.0);
    }

    #[test]
    fn merge_union_both_empty_returns_empty() {
        let a = Grid::<f32>::new(0.0, 1.0);
        let b = Grid::<f32>::new(0.0, 1.0);
        let result = merge_union(&a, &b);
        assert_eq!(result.tree().active_voxel_count(), 0);
    }

    #[test]
    fn merge_intersection_only_common_voxels() {
        let a = grid(&[(Coord::new(0, 0, 0), 1.0), (Coord::new(1, 0, 0), 2.0)]);
        let b = grid(&[(Coord::new(1, 0, 0), 99.0), (Coord::new(2, 0, 0), 3.0)]);
        let result = merge_intersection(&a, &b);

        // Only (1,0,0) is in both; value from a
        assert_eq!(result.tree().active_voxel_count(), 1);
        assert_eq!(result.get(Coord::new(1, 0, 0)), 2.0);
    }

    #[test]
    fn merge_intersection_no_overlap_empty() {
        let a = grid(&[(Coord::new(0, 0, 0), 1.0)]);
        let b = grid(&[(Coord::new(100, 100, 100), 2.0)]);
        let result = merge_intersection(&a, &b);
        assert_eq!(result.tree().active_voxel_count(), 0);
    }

    #[test]
    fn merge_intersection_full_overlap_matches_a() {
        let a = grid(&[(Coord::new(0, 0, 0), 5.0), (Coord::new(1, 1, 1), 7.0)]);
        let b = grid(&[(Coord::new(0, 0, 0), 99.0), (Coord::new(1, 1, 1), 99.0)]);
        let result = merge_intersection(&a, &b);

        assert_eq!(result.tree().active_voxel_count(), 2);
        assert_eq!(result.get(Coord::new(0, 0, 0)), 5.0);
        assert_eq!(result.get(Coord::new(1, 1, 1)), 7.0);
    }
}

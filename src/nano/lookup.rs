//! Fast voxel lookup in a linearized NanoGrid.

use crate::math::Coord;
use crate::tree::leaf::LEAF_LOG2DIM;

use super::layout::NanoGrid;

impl NanoGrid {
    /// Look up a value at the given coordinate.
    /// Returns the background value if no leaf contains this coordinate.
    pub fn get(&self, coord: Coord) -> f32 {
        let hdr = self.header();
        match self.find_leaf(coord) {
            Some(leaf) => {
                let off = coord.offset_in_tile(LEAF_LOG2DIM);
                leaf.values[off]
            }
            None => hdr.background,
        }
    }

    /// Check if a voxel is active.
    pub fn is_active(&self, coord: Coord) -> bool {
        match self.find_leaf(coord) {
            Some(leaf) => {
                let off = coord.offset_in_tile(LEAF_LOG2DIM);
                (leaf.active_mask[off / 64] >> (off % 64)) & 1 == 1
            }
            None => false,
        }
    }

    /// Iterate over all active voxels, yielding `(Coord, f32)` pairs.
    pub fn iter_active(&self) -> impl Iterator<Item = (Coord, f32)> + '_ {
        (0..self.leaf_count()).flat_map(move |i| {
            let leaf = self.leaf(i);
            let origin = Coord::new(leaf.origin[0], leaf.origin[1], leaf.origin[2]);
            let active_mask = leaf.active_mask;
            let values = leaf.values;

            (0..8usize).flat_map(move |word_idx| {
                IterBits {
                    word: active_mask[word_idx],
                    base: word_idx * 64,
                }
                .map(move |off| {
                    let x = (off >> 6) as i32;
                    let y = ((off >> 3) & 0x7) as i32;
                    let z = (off & 0x7) as i32;
                    let coord = Coord::new(origin.x + x, origin.y + y, origin.z + z);
                    (coord, values[off])
                })
            })
        })
    }

    /// Find the leaf containing `coord` via binary search on sorted origins.
    fn find_leaf(&self, coord: Coord) -> Option<&super::layout::NanoLeaf> {
        let leaf_origin = coord.aligned(LEAF_LOG2DIM);
        let target = [leaf_origin.x, leaf_origin.y, leaf_origin.z];
        let count = self.leaf_count();
        if count == 0 {
            return None;
        }

        // Binary search on sorted leaf origins.
        let mut lo = 0usize;
        let mut hi = count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let mid_origin = self.leaf(mid).origin;
            match super::layout::cmp_origin(&mid_origin, &target) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Equal => return Some(self.leaf(mid)),
                std::cmp::Ordering::Greater => hi = mid,
            }
        }
        None
    }
}

/// Bit iterator: yields set-bit indices from a u64 word.
struct IterBits {
    word: u64,
    base: usize,
}

impl Iterator for IterBits {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.word == 0 {
            return None;
        }
        let bit = self.word.trailing_zeros() as usize;
        self.word &= self.word - 1;
        Some(self.base + bit)
    }
}

#[cfg(test)]
mod tests {
    use crate::grid::{Grid, GridClass};
    use crate::math::Coord;
    use crate::nano::NanoGrid;

    #[test]
    fn round_trip_grid_to_nano_to_grid() {
        let mut grid = Grid::<f32>::new(-1.0, 0.5);
        grid.set(Coord::new(0, 0, 0), 1.0);
        grid.set(Coord::new(1, 2, 3), 2.0);
        grid.set(Coord::new(10, 20, 5), 3.0);

        let nano = NanoGrid::from_grid(&grid);
        let grid2 = nano.to_grid();

        assert_eq!(grid2.get(Coord::new(0, 0, 0)), 1.0);
        assert_eq!(grid2.get(Coord::new(1, 2, 3)), 2.0);
        assert_eq!(grid2.get(Coord::new(10, 20, 5)), 3.0);
        // Background
        assert_eq!(grid2.get(Coord::new(99, 99, 99)), -1.0);
        assert_eq!(grid2.tree().background(), -1.0);
    }

    #[test]
    fn header_fields_match_source_grid() {
        let mut grid = Grid::<f32>::new(0.5, 0.25);
        grid.set_grid_class(GridClass::LevelSet);
        grid.set(Coord::new(5, 10, 15), 42.0);
        grid.set(Coord::new(-3, -7, 2), -1.0);

        let nano = NanoGrid::from_grid(&grid);
        let hdr = nano.header();

        assert_eq!(hdr.magic, *b"NANO");
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.grid_class, 1); // LevelSet
        assert!((hdr.voxel_size - 0.25).abs() < 1e-10);
        assert_eq!(hdr.background, 0.5);
        assert_eq!(hdr.bbox_min, [-3, -7, 2]);
        assert_eq!(hdr.bbox_max, [5, 10, 15]);
    }

    #[test]
    fn get_returns_correct_values() {
        let mut grid = Grid::<f32>::new(0.0, 1.0);
        grid.set(Coord::new(1, 2, 3), 42.0);
        grid.set(Coord::new(4, 5, 6), 99.0);

        let nano = NanoGrid::from_grid(&grid);

        assert_eq!(nano.get(Coord::new(1, 2, 3)), 42.0);
        assert_eq!(nano.get(Coord::new(4, 5, 6)), 99.0);
    }

    #[test]
    fn get_returns_background_for_unset() {
        let mut grid = Grid::<f32>::new(-7.5, 1.0);
        grid.set(Coord::new(0, 0, 0), 1.0);

        let nano = NanoGrid::from_grid(&grid);

        // Coordinate in a leaf that doesn't exist
        assert_eq!(nano.get(Coord::new(100, 100, 100)), -7.5);
    }

    #[test]
    fn is_active_matches_source() {
        let mut grid = Grid::<f32>::new(0.0, 1.0);
        grid.set(Coord::new(3, 3, 3), 5.0);

        let nano = NanoGrid::from_grid(&grid);

        assert!(nano.is_active(Coord::new(3, 3, 3)));
        assert!(!nano.is_active(Coord::new(0, 0, 0))); // in same leaf but not set
        assert!(!nano.is_active(Coord::new(100, 100, 100))); // no leaf
    }

    #[test]
    fn as_bytes_length_matches_expected() {
        let mut grid = Grid::<f32>::new(0.0, 1.0);
        grid.set(Coord::new(0, 0, 0), 1.0);
        grid.set(Coord::new(8, 0, 0), 2.0); // different leaf

        let nano = NanoGrid::from_grid(&grid);
        let header_size = std::mem::size_of::<super::super::NanoHeader>();
        let leaf_size = std::mem::size_of::<super::super::NanoLeaf>();
        let expected = header_size + leaf_size * 2;

        assert_eq!(nano.as_bytes().len(), expected);
    }

    #[test]
    fn from_bytes_rejects_invalid_magic() {
        let mut data = vec![0u8; std::mem::size_of::<super::super::NanoHeader>()];
        data[0] = b'X';
        data[1] = b'Y';
        data[2] = b'Z';
        data[3] = b'W';

        let result = NanoGrid::from_bytes(data);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid magic"));
    }

    #[test]
    fn from_bytes_rejects_truncated_buffer() {
        let data = vec![0u8; 10]; // way too small
        let result = NanoGrid::from_bytes(data);
        assert!(result.is_err());
    }

    #[test]
    fn from_bytes_round_trip() {
        let mut grid = Grid::<f32>::new(0.0, 1.0);
        grid.set(Coord::new(1, 2, 3), 10.0);
        grid.set(Coord::new(9, 10, 11), 20.0);

        let nano = NanoGrid::from_grid(&grid);
        let bytes = nano.as_bytes().to_vec();
        let nano2 = NanoGrid::from_bytes(bytes).unwrap();

        assert_eq!(nano2.get(Coord::new(1, 2, 3)), 10.0);
        assert_eq!(nano2.get(Coord::new(9, 10, 11)), 20.0);
        assert_eq!(nano2.leaf_count(), nano.leaf_count());
    }

    #[test]
    fn leaf_ordering_is_sorted() {
        let mut grid = Grid::<f32>::new(0.0, 1.0);
        // Create leaves in various locations
        grid.set(Coord::new(100, 0, 0), 1.0);
        grid.set(Coord::new(0, 0, 0), 2.0);
        grid.set(Coord::new(50, 50, 50), 3.0);
        grid.set(Coord::new(-10, -10, -10), 4.0);

        let nano = NanoGrid::from_grid(&grid);

        for i in 1..nano.leaf_count() {
            let prev = nano.leaf(i - 1).origin;
            let curr = nano.leaf(i).origin;
            assert!(
                super::super::layout::cmp_origin(&prev, &curr) == std::cmp::Ordering::Less,
                "Leaf {} origin {:?} should be < leaf {} origin {:?}",
                i - 1,
                prev,
                i,
                curr
            );
        }
    }

    #[test]
    fn iter_active_yields_same_set_as_source() {
        let mut grid = Grid::<f32>::new(0.0, 1.0);
        grid.set(Coord::new(0, 0, 0), 1.0);
        grid.set(Coord::new(1, 2, 3), 2.0);
        grid.set(Coord::new(10, 20, 5), 3.0);
        grid.set(Coord::new(130, 130, 130), 4.0);

        let nano = NanoGrid::from_grid(&grid);

        let mut source_active: Vec<(Coord, f32)> = grid.tree().iter_active().collect();
        let mut nano_active: Vec<(Coord, f32)> = nano.iter_active().collect();

        source_active.sort_by(|a, b| a.0.cmp(&b.0));
        nano_active.sort_by(|a, b| a.0.cmp(&b.0));

        assert_eq!(source_active.len(), nano_active.len());
        for (s, n) in source_active.iter().zip(nano_active.iter()) {
            assert_eq!(s.0, n.0, "Coord mismatch");
            assert_eq!(s.1, n.1, "Value mismatch at {:?}", s.0);
        }
    }

    #[test]
    fn empty_grid_produces_valid_nanogrid() {
        let grid = Grid::<f32>::new(5.0, 1.0);
        let nano = NanoGrid::from_grid(&grid);

        assert_eq!(nano.leaf_count(), 0);
        assert_eq!(nano.header().background, 5.0);
        assert_eq!(nano.header().magic, *b"NANO");

        // get() returns background
        assert_eq!(nano.get(Coord::new(0, 0, 0)), 5.0);
        assert!(!nano.is_active(Coord::new(0, 0, 0)));
        assert_eq!(nano.iter_active().count(), 0);

        // Round-trip
        let grid2 = nano.to_grid();
        assert_eq!(grid2.tree().background(), 5.0);
        assert_eq!(grid2.active_voxel_count(), 0);
    }

    #[test]
    fn fog_volume_grid_class_preserved() {
        let mut grid = Grid::<f32>::new(0.0, 1.0);
        grid.set_grid_class(GridClass::FogVolume);
        grid.set(Coord::new(0, 0, 0), 1.0);

        let nano = NanoGrid::from_grid(&grid);
        assert_eq!(nano.header().grid_class, 2);

        let grid2 = nano.to_grid();
        assert_eq!(grid2.grid_class(), GridClass::FogVolume);
    }

    #[test]
    fn many_leaves_round_trip() {
        let mut grid = Grid::<f32>::new(0.0, 1.0);
        // Set voxels across many different leaves
        for i in 0..20 {
            let c = Coord::new(i * 8, i * 8, i * 8);
            grid.set(c, i as f32);
        }

        let nano = NanoGrid::from_grid(&grid);
        assert_eq!(nano.leaf_count(), 20);

        for i in 0..20 {
            let c = Coord::new(i * 8, i * 8, i * 8);
            assert_eq!(nano.get(c), i as f32, "Mismatch at leaf {}", i);
            assert!(nano.is_active(c));
        }
    }

    #[test]
    fn negative_coordinates() {
        let mut grid = Grid::<f32>::new(-1.0, 1.0);
        grid.set(Coord::new(-5, -10, -3), 77.0);
        grid.set(Coord::new(-100, -200, -300), 88.0);

        let nano = NanoGrid::from_grid(&grid);

        assert_eq!(nano.get(Coord::new(-5, -10, -3)), 77.0);
        assert_eq!(nano.get(Coord::new(-100, -200, -300)), 88.0);
        assert!(nano.is_active(Coord::new(-5, -10, -3)));
        assert!(nano.is_active(Coord::new(-100, -200, -300)));

        // Round-trip
        let grid2 = nano.to_grid();
        assert_eq!(grid2.get(Coord::new(-5, -10, -3)), 77.0);
        assert_eq!(grid2.get(Coord::new(-100, -200, -300)), 88.0);
    }

    #[test]
    fn affine_transform_preserved() {
        let mut grid = Grid::<f32>::new(0.0, 0.25);
        grid.set(Coord::new(4, 8, 12), 7.0);

        let nano = NanoGrid::from_grid(&grid);
        let hdr = nano.header();

        // Check the affine matrix encodes voxel_size=0.25
        assert!((hdr.affine_mat[0][0] - 0.25).abs() < 1e-10);
        assert!((hdr.affine_mat[1][1] - 0.25).abs() < 1e-10);
        assert!((hdr.affine_mat[2][2] - 0.25).abs() < 1e-10);

        let grid2 = nano.to_grid();
        assert!((grid2.affine_map().voxel_size() - 0.25).abs() < 1e-10);
    }
}

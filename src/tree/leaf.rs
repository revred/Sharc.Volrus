//! LeafNode — 8^3 dense voxel tile.
//!
//! Mirrors `openvdb::tree::LeafNode<T, 3>`.

use crate::math::Coord;

/// LOG2DIM for leaf nodes: 3 -> 8^3 = 512 voxels per leaf.
pub const LEAF_LOG2DIM: u32 = 3;
/// Total voxels per leaf: 8^3.
pub const LEAF_SIZE: usize = 1 << (3 * LEAF_LOG2DIM); // 512
/// Dimension of a leaf node per axis: 8.
pub const LEAF_DIM: i32 = 1 << LEAF_LOG2DIM; // 8

/// Dense 8^3 voxel tile with an activity bitmask.
///
/// Active voxels are "interesting" (surface, occupied, etc.); inactive
/// voxels hold the background value.
pub struct LeafNode<T: Copy> {
    /// Lower-left corner of this leaf in index space (tile-aligned).
    origin: Coord,
    /// Dense value storage: 512 entries in ZYX order.
    values: Box<[T; LEAF_SIZE]>,
    /// 512-bit activity mask: 8 x u64 words.
    active: [u64; 8],
}

impl<T: Copy + Default> LeafNode<T> {
    /// Create a leaf filled with `background`, all voxels inactive.
    pub fn new(origin: Coord, background: T) -> Self {
        Self {
            origin: origin.aligned(LEAF_LOG2DIM),
            values: Box::new([background; LEAF_SIZE]),
            active: [0u64; 8],
        }
    }

    /// Origin coordinate of this leaf (tile-aligned).
    #[inline]
    pub fn origin(&self) -> Coord {
        self.origin
    }

    /// Get the value at `coord`.  Panics if `coord` is outside this leaf.
    #[inline]
    pub fn get(&self, coord: Coord) -> T {
        let off = coord.offset_in_tile(LEAF_LOG2DIM);
        self.values[off]
    }

    /// Set the value at `coord` and mark it active.
    #[inline]
    pub fn set(&mut self, coord: Coord, value: T) {
        let off = coord.offset_in_tile(LEAF_LOG2DIM);
        self.values[off] = value;
        self.active[off / 64] |= 1u64 << (off % 64);
    }

    /// Clear the active flag at `coord` and reset to `background`.
    #[inline]
    pub fn set_inactive(&mut self, coord: Coord, background: T) {
        let off = coord.offset_in_tile(LEAF_LOG2DIM);
        self.values[off] = background;
        self.active[off / 64] &= !(1u64 << (off % 64));
    }

    /// Whether the voxel at `coord` is active.
    #[inline]
    pub fn is_active(&self, coord: Coord) -> bool {
        let off = coord.offset_in_tile(LEAF_LOG2DIM);
        (self.active[off / 64] >> (off % 64)) & 1 == 1
    }

    /// Count of active voxels in this leaf.
    pub fn active_count(&self) -> u32 {
        self.active.iter().map(|w| w.count_ones()).sum()
    }

    /// Alias for `set_inactive` — deactivate a voxel and reset to background.
    #[inline]
    pub fn deactivate(&mut self, coord: Coord, background: T) {
        self.set_inactive(coord, background);
    }

    /// Returns `true` if no voxels are active (active_count() == 0).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.active.iter().all(|&w| w == 0)
    }

    /// Raw bitmask access for zero-copy bridging with kit-sdf VoxelBrick.
    pub fn active_mask(&self) -> &[u64; 8] {
        &self.active
    }

    /// Mutable bitmask access for bulk import.
    pub fn active_mask_mut(&mut self) -> &mut [u64; 8] {
        &mut self.active
    }

    /// Raw value slice for bulk read.
    pub fn values(&self) -> &[T; LEAF_SIZE] {
        &self.values
    }

    /// Mutable value slice for bulk write.
    pub fn values_mut(&mut self) -> &mut [T; LEAF_SIZE] {
        &mut self.values
    }

    /// Iterate over all active voxels, yielding `(Coord, T)` pairs.
    pub fn iter_active(&self) -> impl Iterator<Item = (Coord, T)> + '_ {
        let origin = self.origin;
        self.active
            .iter()
            .enumerate()
            .flat_map(move |(word_idx, &word)| {
                let base = word_idx * 64;
                IterBits {
                    word,
                    base,
                }
            })
            .map(move |off| {
                // Reverse of ZYX: off = x * 64 + y * 8 + z
                let x = (off >> 6) as i32;
                let y = ((off >> 3) & 0x7) as i32;
                let z = (off & 0x7) as i32;
                let coord = Coord::new(origin.x + x, origin.y + y, origin.z + z);
                (coord, self.values[off])
            })
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
        self.word &= self.word - 1; // clear lowest set bit
        Some(self.base + bit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_new_all_inactive() {
        let leaf = LeafNode::<f32>::new(Coord::origin(), 0.0);
        assert_eq!(leaf.active_count(), 0);
        assert_eq!(leaf.origin(), Coord::origin());
    }

    #[test]
    fn leaf_set_get_active() {
        let mut leaf = LeafNode::<f32>::new(Coord::origin(), -1.0);
        let c = Coord::new(3, 5, 7);
        leaf.set(c, 42.0);
        assert_eq!(leaf.get(c), 42.0);
        assert!(leaf.is_active(c));
        assert_eq!(leaf.active_count(), 1);
    }

    #[test]
    fn leaf_set_inactive_resets() {
        let mut leaf = LeafNode::<f32>::new(Coord::origin(), -1.0);
        let c = Coord::new(1, 1, 1);
        leaf.set(c, 5.0);
        assert!(leaf.is_active(c));
        leaf.set_inactive(c, -1.0);
        assert!(!leaf.is_active(c));
        assert_eq!(leaf.get(c), -1.0);
    }

    #[test]
    fn leaf_origin_aligned() {
        let leaf = LeafNode::<f32>::new(Coord::new(13, 7, 20), 0.0);
        assert_eq!(leaf.origin(), Coord::new(8, 0, 16));
    }

    #[test]
    fn leaf_active_mask_512_bits() {
        let mut leaf = LeafNode::<bool>::new(Coord::origin(), false);
        for x in 0..8 {
            for y in 0..8 {
                for z in 0..8 {
                    leaf.set(Coord::new(x, y, z), true);
                }
            }
        }
        assert_eq!(leaf.active_count(), 512);
        assert_eq!(leaf.active_mask(), &[u64::MAX; 8]);
    }

    #[test]
    fn leaf_iter_active_empty() {
        let leaf = LeafNode::<f32>::new(Coord::origin(), 0.0);
        assert_eq!(leaf.iter_active().count(), 0);
    }

    #[test]
    fn leaf_iter_active_yields_set_voxels() {
        let mut leaf = LeafNode::<f32>::new(Coord::origin(), 0.0);
        leaf.set(Coord::new(1, 2, 3), 10.0);
        leaf.set(Coord::new(7, 7, 7), 20.0);
        let active: Vec<_> = leaf.iter_active().collect();
        assert_eq!(active.len(), 2);
        assert!(active.contains(&(Coord::new(1, 2, 3), 10.0)));
        assert!(active.contains(&(Coord::new(7, 7, 7), 20.0)));
    }

    #[test]
    fn leaf_deactivate_alias() {
        let mut leaf = LeafNode::<f32>::new(Coord::origin(), -1.0);
        let c = Coord::new(1, 1, 1);
        leaf.set(c, 5.0);
        assert!(leaf.is_active(c));
        leaf.deactivate(c, -1.0);
        assert!(!leaf.is_active(c));
        assert_eq!(leaf.get(c), -1.0);
    }

    #[test]
    fn leaf_is_empty_when_no_active() {
        let leaf = LeafNode::<f32>::new(Coord::origin(), 0.0);
        assert!(leaf.is_empty());
    }

    #[test]
    fn leaf_is_not_empty_with_active() {
        let mut leaf = LeafNode::<f32>::new(Coord::origin(), 0.0);
        leaf.set(Coord::new(0, 0, 0), 1.0);
        assert!(!leaf.is_empty());
    }

    #[test]
    fn leaf_becomes_empty_after_deactivation() {
        let mut leaf = LeafNode::<f32>::new(Coord::origin(), 0.0);
        leaf.set(Coord::new(0, 0, 0), 1.0);
        assert!(!leaf.is_empty());
        leaf.deactivate(Coord::new(0, 0, 0), 0.0);
        assert!(leaf.is_empty());
    }

    #[test]
    fn leaf_iter_active_with_offset_origin() {
        let mut leaf = LeafNode::<i32>::new(Coord::new(8, 16, 24), 0);
        leaf.set(Coord::new(8, 16, 24), 1);
        leaf.set(Coord::new(10, 18, 26), 2);
        let active: Vec<_> = leaf.iter_active().collect();
        assert_eq!(active.len(), 2);
        assert!(active.contains(&(Coord::new(8, 16, 24), 1)));
        assert!(active.contains(&(Coord::new(10, 18, 26), 2)));
    }
}

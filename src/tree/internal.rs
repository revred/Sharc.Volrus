//! InternalNode — single level of fan-out between Root and Leaf.
//!
//! Implements a 16^3 (LOG2DIM=4) internal node that bridges the root
//! hash-map to dense 8^3 leaf tiles.  Each slot in the internal node
//! is either a child LeafNode pointer or a tile value (uniform region).
//!
//! The combined tree is: Root (HashMap) -> InternalNode (16^3) -> LeafNode (8^3).
//! This gives a total resolution of 128^3 per internal node.

use crate::math::Coord;
use crate::tree::leaf::{LeafNode, LEAF_DIM, LEAF_LOG2DIM, LEAF_SIZE};

/// LOG2DIM for the internal node: 4 -> 16^3 = 4096 child slots.
pub const INTERNAL_LOG2DIM: u32 = 4;
/// Number of child slots per axis.
pub const INTERNAL_DIM: usize = 1 << INTERNAL_LOG2DIM; // 16
/// Total child slots.
pub const INTERNAL_SIZE: usize = 1 << (3 * INTERNAL_LOG2DIM); // 4096

/// Total LOG2DIM of the internal node including its children.
/// Each child covers 8 voxels per axis, and the internal node has 16 children
/// per axis, so it covers 128 voxels per axis = 2^7.
pub const INTERNAL_TOTAL_LOG2DIM: u32 = INTERNAL_LOG2DIM + LEAF_LOG2DIM; // 7
/// Dimension of the internal node in voxels per axis: 128.
pub const INTERNAL_TOTAL_DIM: i32 = 1 << INTERNAL_TOTAL_LOG2DIM; // 128

/// A slot in the internal node: either a child leaf or a tile value.
enum ChildOrTile<T: Copy> {
    /// A child leaf node.
    Child(Box<LeafNode<T>>),
    /// A uniform tile value.  If active, the entire region is that value.
    Tile(T),
}

/// Internal node with 16^3 branching factor, holding LeafNode children.
///
/// Each slot covers an 8^3 voxel region.  The node as a whole covers
/// 128^3 voxels in index space.
pub struct InternalNode<T: Copy> {
    /// Lower-left corner aligned to INTERNAL_TOTAL_DIM boundary.
    origin: Coord,
    /// 4096 slots: child leaf or tile value.
    children: Vec<ChildOrTile<T>>,
    /// Bitmask tracking which slots have child nodes (vs tile values).
    /// 4096 bits = 64 x u64 words.
    child_mask: [u64; 64],
    /// Bitmask tracking which slots are active tiles (only meaningful
    /// for slots that are tiles, not children).
    value_mask: [u64; 64],
    /// Background value for new leaves.
    background: T,
}

impl<T: Copy + Default> InternalNode<T> {
    /// Create an internal node with all slots set to inactive background tiles.
    pub fn new(origin: Coord, background: T) -> Self {
        let aligned = origin.aligned(INTERNAL_TOTAL_LOG2DIM);
        let mut children = Vec::with_capacity(INTERNAL_SIZE);
        for _ in 0..INTERNAL_SIZE {
            children.push(ChildOrTile::Tile(background));
        }
        Self {
            origin: aligned,
            children,
            child_mask: [0u64; 64],
            value_mask: [0u64; 64],
            background,
        }
    }

    /// Origin of this internal node (aligned to 128-boundary).
    #[inline]
    pub fn origin(&self) -> Coord {
        self.origin
    }

    /// Background value.
    #[inline]
    pub fn background(&self) -> T {
        self.background
    }

    /// Compute the child slot index for a given coordinate.
    ///
    /// The coordinate is first shifted into the internal node's local space,
    /// then the leaf-level bits are stripped to get the child index.
    #[inline]
    fn child_index(&self, coord: Coord) -> usize {
        // Local coord within this internal node's 128^3 region
        let lx = ((coord.x - self.origin.x) >> LEAF_LOG2DIM) as usize;
        let ly = ((coord.y - self.origin.y) >> LEAF_LOG2DIM) as usize;
        let lz = ((coord.z - self.origin.z) >> LEAF_LOG2DIM) as usize;
        // ZYX order: x * 256 + y * 16 + z
        (lx << (2 * INTERNAL_LOG2DIM)) | (ly << INTERNAL_LOG2DIM) | lz
    }

    /// Test if slot `idx` has a child node.
    #[inline]
    fn has_child(&self, idx: usize) -> bool {
        (self.child_mask[idx / 64] >> (idx % 64)) & 1 == 1
    }

    /// Mark slot `idx` as having a child node.
    #[inline]
    fn set_child_bit(&mut self, idx: usize) {
        self.child_mask[idx / 64] |= 1u64 << (idx % 64);
        // A child slot is never an active tile
        self.value_mask[idx / 64] &= !(1u64 << (idx % 64));
    }

    /// Test if slot `idx` is an active tile.
    #[inline]
    fn is_tile_active(&self, idx: usize) -> bool {
        !self.has_child(idx) && (self.value_mask[idx / 64] >> (idx % 64)) & 1 == 1
    }

    /// Get the value at `coord`.
    pub fn get(&self, coord: Coord) -> T {
        let idx = self.child_index(coord);
        match &self.children[idx] {
            ChildOrTile::Child(leaf) => leaf.get(coord),
            ChildOrTile::Tile(val) => *val,
        }
    }

    /// Set the value at `coord`, allocating a child leaf if needed.
    pub fn set(&mut self, coord: Coord, value: T) {
        let idx = self.child_index(coord);
        if !self.has_child(idx) {
            // Promote tile to child leaf
            let leaf_origin = coord.aligned(LEAF_LOG2DIM);
            let bg = match &self.children[idx] {
                ChildOrTile::Tile(v) => *v,
                _ => unreachable!(),
            };
            let leaf = LeafNode::new(leaf_origin, bg);
            self.children[idx] = ChildOrTile::Child(Box::new(leaf));
            self.set_child_bit(idx);
        }
        if let ChildOrTile::Child(ref mut leaf) = self.children[idx] {
            leaf.set(coord, value);
        }
    }

    /// Test if a voxel at `coord` is active.
    pub fn is_active(&self, coord: Coord) -> bool {
        let idx = self.child_index(coord);
        match &self.children[idx] {
            ChildOrTile::Child(leaf) => leaf.is_active(coord),
            ChildOrTile::Tile(_) => self.is_tile_active(idx),
        }
    }

    /// Deactivate a voxel at `coord`, setting it back to background.
    /// If the containing leaf exists, deactivates the voxel in-place.
    pub fn deactivate(&mut self, coord: Coord) {
        let idx = self.child_index(coord);
        if self.has_child(idx) {
            if let ChildOrTile::Child(ref mut leaf) = self.children[idx] {
                leaf.deactivate(coord, self.background);
            }
        }
        // If it's a tile, there's no individual voxel to deactivate unless
        // it's an active tile — in that case we'd need to expand. For now,
        // only handle leaf children (matching OpenVDB behavior for simple deactivate).
    }

    /// Count of active voxels in this internal node.
    pub fn active_voxel_count(&self) -> u64 {
        let mut count = 0u64;
        for (idx, slot) in self.children.iter().enumerate() {
            match slot {
                ChildOrTile::Child(leaf) => count += leaf.active_count() as u64,
                ChildOrTile::Tile(_) => {
                    if self.is_tile_active(idx) {
                        // Active tile covers an entire leaf's worth of voxels
                        count += crate::tree::leaf::LEAF_SIZE as u64;
                    }
                }
            }
        }
        count
    }

    /// Number of child leaf nodes allocated in this internal node.
    pub fn child_count(&self) -> usize {
        self.child_mask.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Iterate over all child leaf nodes.
    pub fn leaves(&self) -> impl Iterator<Item = &LeafNode<T>> {
        self.children.iter().filter_map(|slot| match slot {
            ChildOrTile::Child(leaf) => Some(leaf.as_ref()),
            ChildOrTile::Tile(_) => None,
        })
    }

    /// Mutable access to a child leaf at `origin` (if allocated).
    pub fn leaf_mut(&mut self, origin: Coord) -> Option<&mut LeafNode<T>> {
        let aligned = origin.aligned(LEAF_LOG2DIM);
        let idx = self.child_index(aligned);
        if self.has_child(idx) {
            if let ChildOrTile::Child(ref mut leaf) = self.children[idx] {
                if leaf.origin() == aligned {
                    return Some(leaf.as_mut());
                }
            }
        }
        None
    }

    /// Get a reference to the child leaf containing `coord`, if allocated.
    pub fn leaf_at(&self, coord: Coord) -> Option<&LeafNode<T>> {
        let idx = self.child_index(coord);
        match &self.children[idx] {
            ChildOrTile::Child(leaf) => Some(leaf.as_ref()),
            ChildOrTile::Tile(_) => None,
        }
    }

    /// Number of active tiles in this internal node.
    pub fn active_tile_count(&self) -> usize {
        self.value_mask
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum()
    }

    /// Remove leaves that have zero active voxels, replacing them with
    /// background tile values.
    pub fn remove_empty_leaves(&mut self) {
        for idx in 0..INTERNAL_SIZE {
            if self.has_child(idx) {
                let is_empty = match &self.children[idx] {
                    ChildOrTile::Child(leaf) => leaf.is_empty(),
                    _ => false,
                };
                if is_empty {
                    self.children[idx] = ChildOrTile::Tile(self.background);
                    // Clear child bit
                    self.child_mask[idx / 64] &= !(1u64 << (idx % 64));
                    // Mark as inactive tile
                    self.value_mask[idx / 64] &= !(1u64 << (idx % 64));
                }
            }
        }
    }

    /// Prune child leaves whose values are all within `tolerance` of a
    /// single constant, collapsing them into active tiles.
    ///
    /// `T` must support subtraction and comparison to check the spread.
    pub fn prune(&mut self, tolerance: T)
    where
        T: PartialOrd + std::ops::Sub<Output = T>,
    {
        for idx in 0..INTERNAL_SIZE {
            if !self.has_child(idx) {
                continue;
            }
            let should_collapse = match &self.children[idx] {
                ChildOrTile::Child(leaf) => {
                    // Check if all 512 values are within tolerance of the first
                    let vals = leaf.values();
                    let first = vals[0];
                    let mut all_within = true;
                    for i in 1..LEAF_SIZE {
                        let diff = if vals[i] > first {
                            vals[i] - first
                        } else {
                            first - vals[i]
                        };
                        if diff > tolerance {
                            all_within = false;
                            break;
                        }
                    }
                    if all_within {
                        Some((first, leaf.active_count() > 0))
                    } else {
                        None
                    }
                }
                _ => None,
            };

            if let Some((tile_val, was_active)) = should_collapse {
                self.children[idx] = ChildOrTile::Tile(tile_val);
                // Clear child bit
                self.child_mask[idx / 64] &= !(1u64 << (idx % 64));
                // Set active tile bit if the leaf had active voxels
                if was_active {
                    self.value_mask[idx / 64] |= 1u64 << (idx % 64);
                } else {
                    self.value_mask[idx / 64] &= !(1u64 << (idx % 64));
                }
            }
        }
    }

    /// Iterate over all child leaf nodes mutably.
    pub fn leaves_mut(&mut self) -> impl Iterator<Item = &mut LeafNode<T>> {
        self.children.iter_mut().filter_map(|slot| match slot {
            ChildOrTile::Child(leaf) => Some(leaf.as_mut()),
            ChildOrTile::Tile(_) => None,
        })
    }

    /// Get a mutable reference to the child leaf containing `coord`, if allocated.
    pub fn leaf_at_mut(&mut self, coord: Coord) -> Option<&mut LeafNode<T>> {
        let idx = self.child_index(coord);
        match &mut self.children[idx] {
            ChildOrTile::Child(leaf) => Some(leaf.as_mut()),
            ChildOrTile::Tile(_) => None,
        }
    }

    /// Iterate over all active voxels in this internal node.
    pub fn iter_active(&self) -> impl Iterator<Item = (Coord, T)> + '_ {
        self.children.iter().enumerate().flat_map(move |(idx, slot)| {
            let items: Box<dyn Iterator<Item = (Coord, T)> + '_> = match slot {
                ChildOrTile::Child(leaf) => Box::new(leaf.iter_active()),
                ChildOrTile::Tile(val) => {
                    if self.is_tile_active(idx) {
                        // Active tile: yield all voxels in the tile region
                        let val = *val;
                        let child_origin = self.child_origin(idx);
                        Box::new(
                            (0..LEAF_DIM).flat_map(move |x| {
                                (0..LEAF_DIM).flat_map(move |y| {
                                    (0..LEAF_DIM).map(move |z| {
                                        let c = Coord::new(
                                            child_origin.x + x,
                                            child_origin.y + y,
                                            child_origin.z + z,
                                        );
                                        (c, val)
                                    })
                                })
                            }),
                        )
                    } else {
                        Box::new(std::iter::empty())
                    }
                }
            };
            items
        })
    }

    /// Compute the origin of the child at slot index `idx`.
    fn child_origin(&self, idx: usize) -> Coord {
        let x = (idx / (INTERNAL_DIM * INTERNAL_DIM)) as i32;
        let y = ((idx / INTERNAL_DIM) % INTERNAL_DIM) as i32;
        let z = (idx % INTERNAL_DIM) as i32;
        Coord::new(
            self.origin.x + x * LEAF_DIM,
            self.origin.y + y * LEAF_DIM,
            self.origin.z + z * LEAF_DIM,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_new_empty() {
        let node = InternalNode::<f32>::new(Coord::origin(), 0.0);
        assert_eq!(node.origin(), Coord::origin());
        assert_eq!(node.child_count(), 0);
        assert_eq!(node.active_voxel_count(), 0);
    }

    #[test]
    fn internal_origin_aligned() {
        let node = InternalNode::<f32>::new(Coord::new(13, 200, 50), 0.0);
        // Aligned to 128 boundary
        assert_eq!(node.origin(), Coord::new(0, 128, 0));
    }

    #[test]
    fn internal_set_get() {
        let mut node = InternalNode::<f32>::new(Coord::origin(), -1.0);
        node.set(Coord::new(5, 5, 5), 42.0);
        assert_eq!(node.get(Coord::new(5, 5, 5)), 42.0);
        assert!(node.is_active(Coord::new(5, 5, 5)));
        assert_eq!(node.child_count(), 1);
    }

    #[test]
    fn internal_unset_returns_background() {
        let node = InternalNode::<f32>::new(Coord::origin(), -1.0);
        assert_eq!(node.get(Coord::new(5, 5, 5)), -1.0);
        assert!(!node.is_active(Coord::new(5, 5, 5)));
    }

    #[test]
    fn internal_different_leaves() {
        let mut node = InternalNode::<f32>::new(Coord::origin(), 0.0);
        // Two coords in different 8^3 regions within the 128^3 node
        node.set(Coord::new(0, 0, 0), 1.0);
        node.set(Coord::new(8, 0, 0), 2.0);
        assert_eq!(node.child_count(), 2);
        assert_eq!(node.get(Coord::new(0, 0, 0)), 1.0);
        assert_eq!(node.get(Coord::new(8, 0, 0)), 2.0);
    }

    #[test]
    fn internal_same_leaf() {
        let mut node = InternalNode::<f32>::new(Coord::origin(), 0.0);
        node.set(Coord::new(0, 0, 0), 1.0);
        node.set(Coord::new(7, 7, 7), 2.0);
        assert_eq!(node.child_count(), 1);
        assert_eq!(node.active_voxel_count(), 2);
    }

    #[test]
    fn internal_leaves_iter() {
        let mut node = InternalNode::<f32>::new(Coord::origin(), 0.0);
        node.set(Coord::new(0, 0, 0), 1.0);
        node.set(Coord::new(8, 0, 0), 2.0);
        node.set(Coord::new(16, 0, 0), 3.0);
        assert_eq!(node.leaves().count(), 3);
    }

    #[test]
    fn internal_leaf_mut_access() {
        let mut node = InternalNode::<f32>::new(Coord::origin(), 0.0);
        node.set(Coord::new(5, 5, 5), 42.0);
        let leaf = node.leaf_mut(Coord::new(0, 0, 0)).unwrap();
        assert_eq!(leaf.get(Coord::new(5, 5, 5)), 42.0);
    }

    #[test]
    fn internal_leaf_at() {
        let mut node = InternalNode::<f32>::new(Coord::origin(), 0.0);
        assert!(node.leaf_at(Coord::new(5, 5, 5)).is_none());
        node.set(Coord::new(5, 5, 5), 42.0);
        assert!(node.leaf_at(Coord::new(5, 5, 5)).is_some());
    }

    #[test]
    fn internal_iter_active() {
        let mut node = InternalNode::<f32>::new(Coord::origin(), 0.0);
        node.set(Coord::new(1, 2, 3), 10.0);
        node.set(Coord::new(9, 10, 11), 20.0);
        let active: Vec<_> = node.iter_active().collect();
        assert_eq!(active.len(), 2);
        assert!(active.contains(&(Coord::new(1, 2, 3), 10.0)));
        assert!(active.contains(&(Coord::new(9, 10, 11), 20.0)));
    }

    #[test]
    fn internal_wide_coord_range() {
        let mut node = InternalNode::<f32>::new(Coord::origin(), 0.0);
        // Corners of the 128^3 region
        node.set(Coord::new(0, 0, 0), 1.0);
        node.set(Coord::new(127, 127, 127), 2.0);
        assert_eq!(node.child_count(), 2);
        assert_eq!(node.get(Coord::new(0, 0, 0)), 1.0);
        assert_eq!(node.get(Coord::new(127, 127, 127)), 2.0);
    }

    #[test]
    fn internal_active_tile_count_zero() {
        let node = InternalNode::<f32>::new(Coord::origin(), 0.0);
        assert_eq!(node.active_tile_count(), 0);
    }

    #[test]
    fn internal_deactivate_voxel() {
        let mut node = InternalNode::<f32>::new(Coord::origin(), 0.0);
        node.set(Coord::new(1, 1, 1), 5.0);
        assert!(node.is_active(Coord::new(1, 1, 1)));
        node.deactivate(Coord::new(1, 1, 1));
        assert!(!node.is_active(Coord::new(1, 1, 1)));
        assert_eq!(node.get(Coord::new(1, 1, 1)), 0.0); // background
    }

    #[test]
    fn internal_remove_empty_leaves() {
        let mut node = InternalNode::<f32>::new(Coord::origin(), 0.0);
        node.set(Coord::new(0, 0, 0), 1.0);
        node.set(Coord::new(8, 0, 0), 2.0);
        assert_eq!(node.child_count(), 2);

        // Deactivate all voxels in the first leaf
        node.deactivate(Coord::new(0, 0, 0));
        // First leaf is now empty
        node.remove_empty_leaves();
        assert_eq!(node.child_count(), 1);
        // Value should now be background tile
        assert_eq!(node.get(Coord::new(0, 0, 0)), 0.0);
    }

    #[test]
    fn internal_prune_constant_leaf() {
        let mut node = InternalNode::<f32>::new(Coord::origin(), 5.0);
        // Set all voxels in one leaf to the same value (5.0 = background)
        // The leaf was created with background 5.0, so all 512 values are 5.0
        node.set(Coord::new(0, 0, 0), 5.0);
        assert_eq!(node.child_count(), 1);

        node.prune(0.0);
        // The leaf should be collapsed to a tile because all values are 5.0
        assert_eq!(node.child_count(), 0);
        assert_eq!(node.active_tile_count(), 1); // it had an active voxel
    }

    #[test]
    fn internal_prune_with_tolerance() {
        let mut node = InternalNode::<f32>::new(Coord::origin(), 5.0);
        node.set(Coord::new(0, 0, 0), 5.0);
        node.set(Coord::new(1, 1, 1), 5.001);
        assert_eq!(node.child_count(), 1);

        // Tolerance too tight: should NOT collapse
        node.prune(0.0001);
        assert_eq!(node.child_count(), 1);

        // Tolerance wide enough: should collapse
        node.prune(0.01);
        assert_eq!(node.child_count(), 0);
    }

    #[test]
    fn internal_prune_leaves_varied_leaf_alone() {
        let mut node = InternalNode::<f32>::new(Coord::origin(), 0.0);
        node.set(Coord::new(0, 0, 0), 1.0);
        node.set(Coord::new(1, 1, 1), 100.0);
        assert_eq!(node.child_count(), 1);

        node.prune(0.0);
        // Should NOT collapse because values differ significantly
        assert_eq!(node.child_count(), 1);
    }
}

//! RootNode — sparse hash-map of InternalNode tiles.
//!
//! The root uses a HashMap to map regions of index space to
//! InternalNode instances.  This gives a 3-level tree:
//! Root (HashMap) -> InternalNode (16^3) -> LeafNode (8^3).

use crate::math::Coord;
use crate::tree::internal::{InternalNode, INTERNAL_TOTAL_LOG2DIM};
use crate::tree::leaf::LeafNode;
use crate::tree::Visitor;
use std::collections::HashMap;

/// Root-level sparse container mapping tile origins to `InternalNode`s.
///
/// Three-level tree: Root -> InternalNode (16^3) -> LeafNode (8^3).
pub struct RootNode<T: Copy> {
    tiles: HashMap<Coord, InternalNode<T>>,
    background: T,
}

impl<T: Copy + Default> RootNode<T> {
    /// Create an empty root with the given background value.
    pub fn new(background: T) -> Self {
        Self {
            tiles: HashMap::new(),
            background,
        }
    }

    /// Background value used for unallocated regions.
    pub fn background(&self) -> T {
        self.background
    }

    /// Number of allocated internal nodes.
    pub fn internal_count(&self) -> usize {
        self.tiles.len()
    }

    /// Number of allocated leaf tiles across all internal nodes.
    pub fn leaf_count(&self) -> usize {
        self.tiles.values().map(|n| n.child_count()).sum()
    }

    /// Total active voxel count across all internal nodes.
    pub fn active_voxel_count(&self) -> u64 {
        self.tiles
            .values()
            .map(|n| n.active_voxel_count())
            .sum()
    }

    /// Get the value at `coord`.  Returns `background` if the internal
    /// node doesn't exist.
    pub fn get(&self, coord: Coord) -> T {
        let key = coord.aligned(INTERNAL_TOTAL_LOG2DIM);
        match self.tiles.get(&key) {
            Some(node) => node.get(coord),
            None => self.background,
        }
    }

    /// Set the value at `coord`, allocating internal and leaf nodes as needed.
    pub fn set(&mut self, coord: Coord, value: T) {
        let key = coord.aligned(INTERNAL_TOTAL_LOG2DIM);
        let bg = self.background;
        let node = self
            .tiles
            .entry(key)
            .or_insert_with(|| InternalNode::new(key, bg));
        node.set(coord, value);
    }

    /// Test if a voxel is active.
    pub fn is_active(&self, coord: Coord) -> bool {
        let key = coord.aligned(INTERNAL_TOTAL_LOG2DIM);
        self.tiles
            .get(&key)
            .is_some_and(|n| n.is_active(coord))
    }

    /// Iterate over all leaf nodes across all internal nodes.
    pub fn leaves(&self) -> impl Iterator<Item = &LeafNode<T>> {
        self.tiles.values().flat_map(|n| n.leaves())
    }

    /// Iterate over all leaf nodes mutably across all internal nodes.
    pub fn leaves_mut(&mut self) -> impl Iterator<Item = &mut LeafNode<T>> {
        self.tiles.values_mut().flat_map(|n| n.leaves_mut())
    }

    /// Collect all leaf tile origins across all internal nodes.
    pub fn leaf_origins(&self) -> Vec<Coord> {
        self.leaves().map(|l| l.origin()).collect()
    }

    /// Mutable access to a leaf at `origin` (if allocated).
    pub fn leaf_mut(&mut self, origin: Coord) -> Option<&mut LeafNode<T>> {
        let key = origin.aligned(INTERNAL_TOTAL_LOG2DIM);
        self.tiles.get_mut(&key)?.leaf_mut(origin)
    }

    /// Iterate over all active voxels across the entire tree.
    pub fn iter_active(&self) -> impl Iterator<Item = (Coord, T)> + '_ {
        self.tiles.values().flat_map(|n| n.iter_active())
    }

    /// Iterate over inactive voxels in all allocated leaf nodes.
    ///
    /// Only voxels within allocated leaves are returned — the unbounded
    /// background is not iterated.
    pub fn iter_inactive(&self) -> impl Iterator<Item = (Coord, T)> + '_ {
        self.leaves().flat_map(|leaf| leaf.iter_off())
    }

    /// Iterate over every voxel (active and inactive) in all allocated leaf nodes.
    ///
    /// Yields `(Coord, T, bool)` — coordinate, stored value, is_active.
    pub fn iter_all_leaf_voxels(&self) -> impl Iterator<Item = (Coord, T, bool)> + '_ {
        self.leaves().flat_map(|leaf| leaf.iter_all())
    }

    /// Walk all voxels in all allocated leaf nodes using a visitor.
    ///
    /// Calls `visitor.visit(coord, value, is_active)` for every voxel.
    pub fn accept_visitor<V: Visitor<T>>(&self, visitor: &mut V) {
        for leaf in self.leaves() {
            for (coord, value, is_active) in leaf.iter_all() {
                visitor.visit(coord, value, is_active);
            }
        }
    }

    /// Walk only active voxels using a visitor.
    pub fn accept_active_visitor<V: Visitor<T>>(&self, visitor: &mut V) {
        for (coord, value) in self.iter_active() {
            visitor.visit(coord, value, true);
        }
    }

    /// Deactivate a voxel at `coord`, resetting it to background.
    pub fn deactivate(&mut self, coord: Coord) {
        let key = coord.aligned(INTERNAL_TOTAL_LOG2DIM);
        if let Some(node) = self.tiles.get_mut(&key) {
            node.deactivate(coord);
        }
    }

    /// Prune all internal nodes: collapse leaves whose values are all
    /// within `tolerance` of a constant into tiles.
    pub fn prune(&mut self, tolerance: T)
    where
        T: PartialOrd + std::ops::Sub<Output = T>,
    {
        for node in self.tiles.values_mut() {
            node.prune(tolerance);
        }
    }

    /// Remove leaves with zero active voxels across all internal nodes.
    pub fn remove_empty_leaves(&mut self) {
        for node in self.tiles.values_mut() {
            node.remove_empty_leaves();
        }
    }

    /// Remove all data, resetting to background-only (empty tree).
    pub fn clear(&mut self) {
        self.tiles.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_empty_returns_background() {
        let root = RootNode::<f32>::new(1.0);
        assert_eq!(root.get(Coord::new(100, 200, 300)), 1.0);
        assert_eq!(root.leaf_count(), 0);
    }

    #[test]
    fn root_set_allocates_leaf() {
        let mut root = RootNode::<f32>::new(0.0);
        root.set(Coord::new(5, 5, 5), 3.14);
        assert_eq!(root.leaf_count(), 1);
        assert_eq!(root.get(Coord::new(5, 5, 5)), 3.14);
        assert!(root.is_active(Coord::new(5, 5, 5)));
    }

    #[test]
    fn root_different_tiles_separate_leaves() {
        let mut root = RootNode::<f32>::new(0.0);
        // Two coords in different 8^3 tiles (but same 128^3 internal node)
        root.set(Coord::new(0, 0, 0), 1.0);
        root.set(Coord::new(8, 0, 0), 2.0);
        assert_eq!(root.leaf_count(), 2);
        // Both in same internal node
        assert_eq!(root.internal_count(), 1);
    }

    #[test]
    fn root_same_tile_single_leaf() {
        let mut root = RootNode::<f32>::new(0.0);
        root.set(Coord::new(0, 0, 0), 1.0);
        root.set(Coord::new(7, 7, 7), 2.0);
        assert_eq!(root.leaf_count(), 1);
        assert_eq!(root.active_voxel_count(), 2);
    }

    #[test]
    fn root_different_internal_nodes() {
        let mut root = RootNode::<f32>::new(0.0);
        // Two coords in different 128^3 regions
        root.set(Coord::new(0, 0, 0), 1.0);
        root.set(Coord::new(128, 0, 0), 2.0);
        assert_eq!(root.internal_count(), 2);
        assert_eq!(root.leaf_count(), 2);
    }

    #[test]
    fn root_leaves_iter() {
        let mut root = RootNode::<f32>::new(0.0);
        root.set(Coord::new(0, 0, 0), 1.0);
        root.set(Coord::new(8, 0, 0), 2.0);
        root.set(Coord::new(128, 0, 0), 3.0);
        assert_eq!(root.leaves().count(), 3);
    }

    #[test]
    fn root_iter_active() {
        let mut root = RootNode::<f32>::new(0.0);
        root.set(Coord::new(1, 2, 3), 10.0);
        root.set(Coord::new(130, 130, 130), 20.0);
        let active: Vec<_> = root.iter_active().collect();
        assert_eq!(active.len(), 2);
        assert!(active.contains(&(Coord::new(1, 2, 3), 10.0)));
        assert!(active.contains(&(Coord::new(130, 130, 130), 20.0)));
    }

    #[test]
    fn root_deactivate() {
        let mut root = RootNode::<f32>::new(0.0);
        root.set(Coord::new(5, 5, 5), 42.0);
        assert!(root.is_active(Coord::new(5, 5, 5)));
        root.deactivate(Coord::new(5, 5, 5));
        assert!(!root.is_active(Coord::new(5, 5, 5)));
        assert_eq!(root.get(Coord::new(5, 5, 5)), 0.0);
    }

    #[test]
    fn root_deactivate_nonexistent_is_noop() {
        let mut root = RootNode::<f32>::new(0.0);
        // Should not panic when deactivating in a region that doesn't exist
        root.deactivate(Coord::new(999, 999, 999));
        assert!(!root.is_active(Coord::new(999, 999, 999)));
    }

    #[test]
    fn root_clear() {
        let mut root = RootNode::<f32>::new(-1.0);
        root.set(Coord::new(0, 0, 0), 1.0);
        root.set(Coord::new(128, 0, 0), 2.0);
        assert_eq!(root.internal_count(), 2);
        root.clear();
        assert_eq!(root.internal_count(), 0);
        assert_eq!(root.leaf_count(), 0);
        assert_eq!(root.active_voxel_count(), 0);
        // Background is preserved
        assert_eq!(root.get(Coord::new(0, 0, 0)), -1.0);
    }

    #[test]
    fn root_prune_collapses_constant_leaves() {
        let mut root = RootNode::<f32>::new(5.0);
        // Set a single voxel to the same value as background
        root.set(Coord::new(0, 0, 0), 5.0);
        assert_eq!(root.leaf_count(), 1);
        root.prune(0.0);
        // Leaf should be collapsed to tile
        assert_eq!(root.leaf_count(), 0);
    }

    #[test]
    fn root_remove_empty_leaves() {
        let mut root = RootNode::<f32>::new(0.0);
        root.set(Coord::new(0, 0, 0), 1.0);
        root.set(Coord::new(8, 0, 0), 2.0);
        assert_eq!(root.leaf_count(), 2);
        // Deactivate one leaf completely
        root.deactivate(Coord::new(0, 0, 0));
        root.remove_empty_leaves();
        assert_eq!(root.leaf_count(), 1);
    }
}

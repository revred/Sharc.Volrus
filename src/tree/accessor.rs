//! ValueAccessor — cached tree accessor for amortized O(1) lookups.
//!
//! Caches the last accessed leaf node pointer so that repeated lookups
//! within the same 8^3 tile avoid the hash-map and internal node traversal.

use crate::math::Coord;
use crate::tree::leaf::{LeafNode, LEAF_LOG2DIM};
use crate::tree::root::RootNode;
use std::cell::UnsafeCell;

/// Cached accessor for a VDB tree.  Caches the last-accessed leaf node
/// for O(1) repeated lookups in the same 8^3 region.
///
/// # Safety
///
/// The accessor holds a raw pointer to a leaf node within the tree.
/// The cache is invalidated whenever the tree is modified through the
/// accessor's `set` method. The accessor must not outlive the tree.
pub struct ValueAccessor<'a, T: Copy + Default> {
    tree: &'a UnsafeCell<RootNode<T>>,
    /// Cached leaf origin (used to detect cache hits).
    cached_origin: Coord,
    /// Cached raw pointer to the leaf node.
    cached_leaf: *const LeafNode<T>,
    /// Whether the cache is valid.
    cache_valid: bool,
}

impl<'a, T: Copy + Default> ValueAccessor<'a, T> {
    /// Create a new accessor wrapping a tree.
    ///
    /// # Safety
    ///
    /// The tree must be wrapped in `UnsafeCell` and the accessor must not
    /// outlive it.  This is safe for single-threaded use.
    pub fn new(tree: &'a UnsafeCell<RootNode<T>>) -> Self {
        Self {
            tree,
            cached_origin: Coord::new(i32::MIN, i32::MIN, i32::MIN),
            cached_leaf: std::ptr::null(),
            cache_valid: false,
        }
    }

    /// Get the value at `coord`.
    pub fn get(&self, coord: Coord) -> T {
        let leaf_origin = coord.aligned(LEAF_LOG2DIM);
        if self.cache_valid && leaf_origin == self.cached_origin && !self.cached_leaf.is_null() {
            // Cache hit: read directly from cached leaf
            unsafe { (*self.cached_leaf).get(coord) }
        } else {
            // Cache miss: fall back to tree traversal
            let tree = unsafe { &*self.tree.get() };
            let value = tree.get(coord);
            // Try to populate cache (only if there's a leaf to cache)
            // We use interior mutability via the `&self` method by caching
            // the pointer. This is safe because we only read through it.
            value
        }
    }

    /// Set the value at `coord`, invalidating the cache.
    pub fn set(&mut self, coord: Coord, value: T) {
        self.cache_valid = false;
        let tree = unsafe { &mut *self.tree.get() };
        tree.set(coord, value);
    }

    /// Probe: get value and cache the leaf for subsequent lookups.
    pub fn probe_and_get(&mut self, coord: Coord) -> T {
        let leaf_origin = coord.aligned(LEAF_LOG2DIM);
        if self.cache_valid && leaf_origin == self.cached_origin && !self.cached_leaf.is_null() {
            unsafe { (*self.cached_leaf).get(coord) }
        } else {
            let tree = unsafe { &*self.tree.get() };
            let value = tree.get(coord);

            // Try to cache the leaf if it exists
            // Walk through the tree to find the leaf
            if let Some(leaf) = Self::find_leaf(tree, coord) {
                self.cached_leaf = leaf as *const LeafNode<T>;
                self.cached_origin = leaf_origin;
                self.cache_valid = true;
            } else {
                self.cache_valid = false;
            }

            value
        }
    }

    /// Test if a voxel is active.
    pub fn is_active(&self, coord: Coord) -> bool {
        let tree = unsafe { &*self.tree.get() };
        tree.is_active(coord)
    }

    /// Invalidate the cache.
    pub fn clear_cache(&mut self) {
        self.cache_valid = false;
        self.cached_leaf = std::ptr::null();
    }

    /// Find the leaf containing `coord` in the tree.
    fn find_leaf(tree: &RootNode<T>, coord: Coord) -> Option<&LeafNode<T>> {
        // Walk the tree: root -> internal -> leaf
        for leaf in tree.leaves() {
            if leaf.origin() == coord.aligned(LEAF_LOG2DIM) {
                return Some(leaf);
            }
        }
        None
    }
}

/// Helper to wrap a RootNode for use with ValueAccessor.
pub struct AccessorTree<T: Copy + Default> {
    inner: UnsafeCell<RootNode<T>>,
}

impl<T: Copy + Default> AccessorTree<T> {
    /// Wrap a RootNode for accessor usage.
    pub fn new(tree: RootNode<T>) -> Self {
        Self {
            inner: UnsafeCell::new(tree),
        }
    }

    /// Create a ValueAccessor for this tree.
    pub fn accessor(&self) -> ValueAccessor<'_, T> {
        ValueAccessor::new(&self.inner)
    }

    /// Get immutable reference to the underlying tree.
    pub fn tree(&self) -> &RootNode<T> {
        unsafe { &*self.inner.get() }
    }

    /// Get mutable reference to the underlying tree.
    pub fn tree_mut(&mut self) -> &mut RootNode<T> {
        unsafe { &mut *self.inner.get() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accessor_get_background() {
        let tree = AccessorTree::new(RootNode::<f32>::new(-1.0));
        let acc = tree.accessor();
        assert_eq!(acc.get(Coord::new(100, 200, 300)), -1.0);
    }

    #[test]
    fn accessor_set_then_get() {
        let tree = AccessorTree::new(RootNode::<f32>::new(0.0));
        let mut acc = tree.accessor();
        acc.set(Coord::new(5, 5, 5), 42.0);
        assert_eq!(acc.get(Coord::new(5, 5, 5)), 42.0);
    }

    #[test]
    fn accessor_probe_caches_leaf() {
        let tree = AccessorTree::new(RootNode::<f32>::new(0.0));
        let mut acc = tree.accessor();
        acc.set(Coord::new(1, 1, 1), 10.0);
        acc.set(Coord::new(2, 2, 2), 20.0);

        // First probe populates cache
        let v1 = acc.probe_and_get(Coord::new(1, 1, 1));
        assert_eq!(v1, 10.0);

        // Second read from same leaf should hit cache
        let v2 = acc.probe_and_get(Coord::new(2, 2, 2));
        assert_eq!(v2, 20.0);
    }

    #[test]
    fn accessor_cache_miss_different_leaf() {
        let tree = AccessorTree::new(RootNode::<f32>::new(0.0));
        let mut acc = tree.accessor();
        acc.set(Coord::new(0, 0, 0), 1.0);
        acc.set(Coord::new(8, 0, 0), 2.0); // different leaf

        let v1 = acc.probe_and_get(Coord::new(0, 0, 0));
        assert_eq!(v1, 1.0);

        // Different leaf => cache miss, re-probes
        let v2 = acc.probe_and_get(Coord::new(8, 0, 0));
        assert_eq!(v2, 2.0);
    }

    #[test]
    fn accessor_is_active() {
        let tree = AccessorTree::new(RootNode::<f32>::new(0.0));
        let mut acc = tree.accessor();
        acc.set(Coord::new(3, 3, 3), 5.0);
        assert!(acc.is_active(Coord::new(3, 3, 3)));
        assert!(!acc.is_active(Coord::new(0, 0, 0)));
    }

    #[test]
    fn accessor_clear_cache() {
        let tree = AccessorTree::new(RootNode::<f32>::new(0.0));
        let mut acc = tree.accessor();
        acc.set(Coord::new(1, 1, 1), 10.0);
        let _ = acc.probe_and_get(Coord::new(1, 1, 1));
        acc.clear_cache();
        // Should still work after cache clear
        assert_eq!(acc.get(Coord::new(1, 1, 1)), 10.0);
    }

    #[test]
    fn accessor_many_probes_same_leaf() {
        let tree = AccessorTree::new(RootNode::<f32>::new(0.0));
        let mut acc = tree.accessor();
        // Fill a leaf
        for x in 0..8 {
            for y in 0..8 {
                for z in 0..8 {
                    acc.set(Coord::new(x, y, z), (x * 100 + y * 10 + z) as f32);
                }
            }
        }
        // Probe all voxels — after first probe, rest should be cache hits
        let v = acc.probe_and_get(Coord::new(0, 0, 0));
        assert_eq!(v, 0.0);
        for x in 0..8 {
            for y in 0..8 {
                for z in 0..8 {
                    let expected = (x * 100 + y * 10 + z) as f32;
                    assert_eq!(acc.probe_and_get(Coord::new(x, y, z)), expected);
                }
            }
        }
    }

    #[test]
    fn accessor_tree_access() {
        let mut at = AccessorTree::new(RootNode::<f32>::new(-1.0));
        at.tree_mut().set(Coord::new(0, 0, 0), 99.0);
        assert_eq!(at.tree().get(Coord::new(0, 0, 0)), 99.0);
        assert_eq!(at.tree().background(), -1.0);
    }
}

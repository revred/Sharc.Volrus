//! Sparse VDB tree — LeafNode, InternalNode, RootNode, Tree, ValueAccessor.
//!
//! Follows OpenVDB's tree topology: a hash-map root holding sparse
//! internal nodes, each containing dense leaf nodes.
//!
//! Three-level tree: Root (HashMap) -> InternalNode (16^3) -> LeafNode (8^3).

pub mod accessor;
pub mod internal;
pub mod leaf;
pub mod root;

pub use accessor::{AccessorTree, ValueAccessor};
pub use internal::InternalNode;
pub use leaf::{LeafNode, LEAF_LOG2DIM, LEAF_SIZE};
pub use root::RootNode;

/// Three-level VDB tree: Root -> InternalNode (16^3) -> Leaf (8^3).
pub type Tree<T> = RootNode<T>;

/// Visitor trait for tree topology traversal.
///
/// Implement this trait to collect statistics, transform values, or perform
/// any per-voxel operation during a full tree walk.
///
/// # Example
/// ```rust
/// use sharc_volrus::tree::Visitor;
/// use sharc_volrus::math::Coord;
///
/// struct Counter { pub active: usize, pub inactive: usize }
///
/// impl Visitor<f32> for Counter {
///     fn visit(&mut self, _coord: Coord, _value: f32, is_active: bool) {
///         if is_active { self.active += 1; } else { self.inactive += 1; }
///     }
/// }
/// ```
pub trait Visitor<T: Copy> {
    /// Called once per voxel during tree traversal.
    ///
    /// * `coord`     — index-space coordinate of the voxel
    /// * `value`     — stored value (background for inactive voxels)
    /// * `is_active` — whether this voxel is in the active set
    fn visit(&mut self, coord: Coord, value: T, is_active: bool);
}

use crate::math::Coord;

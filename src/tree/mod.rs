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

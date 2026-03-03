//! Sharc.Volrus
//!
//! Rust bootstrap of core sparse-volume concepts inspired by OpenVDB.
//! This crate intentionally starts small:
//! - integer voxel coordinates (`Coord`)
//! - sparse leaf storage (`LeafNode`)
//! - root-leaf tree topology (`Tree`)
//! - value grid facade (`Grid`)
//!
//! Design references from OpenVDB source tree:
//! - `openvdb/openvdb/math/Coord.h`
//! - `openvdb/openvdb/tree/LeafNode.h`
//! - `openvdb/openvdb/tree/RootNode.h`
//! - `openvdb/openvdb/tree/Tree.h`
//! - `openvdb/openvdb/Grid.h`

pub mod grid;
pub mod math;
pub mod tree;

pub use grid::Grid;
pub use math::Coord;
pub use tree::{LeafNode, RootNode, Tree};


//! Point Data Grid — particle storage binned into VDB leaf structure.
//!
//! Mirrors OpenVDB's PointDataGrid: particles are spatially hashed into
//! leaf-sized buckets (8^3 voxel regions) for efficient spatial queries.

pub mod point_grid;

pub use point_grid::{Particle, PointDataGrid, PointLeaf};

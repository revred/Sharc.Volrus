//! NanoVDB — GPU-compatible read-only linearized grid.
//!
//! Linearizes a VDB tree into a single contiguous byte buffer that can be
//! memory-mapped or uploaded to GPU. The layout is:
//!
//! `[NanoHeader][NanoLeaf 0][NanoLeaf 1]...[NanoLeaf N-1]`
//!
//! All data is little-endian, `#[repr(C)]` for GPU interop.

pub mod layout;
pub mod lookup;

pub use layout::{NanoGrid, NanoHeader, NanoLeaf};

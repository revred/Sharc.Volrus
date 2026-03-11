//! Volrus native I/O — `.vol` binary format for sparse volumetric grids.
//!
//! The `.vol` format is a simple, compact binary representation that stores
//! only the leaf-level data (origins, active masks, dense values). It avoids
//! the complexity of the OpenVDB `.vdb` format while preserving full fidelity
//! for `Grid<f32>` round-trips.

pub mod vol_format;

pub use vol_format::{load_vol, read_vol, save_vol, write_vol};

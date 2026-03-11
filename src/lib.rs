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
pub mod io;
pub mod math;
pub mod nano;
pub mod points;
pub mod render;
pub mod tools;
pub mod tree;

pub use grid::{AffineMap, Grid};
pub use nano::{NanoGrid, NanoHeader, NanoLeaf};
pub use math::{Coord, CoordBBox, Ray, Vec3d};
pub use render::{Camera, GpuMesh, GpuVertex, Image, TransferFunction, VolumeRenderConfig};
pub use tools::{
    adaptive_surface, advect_level_set, advect_particles, advect_particles_euler,
    AdaptiveMeshConfig, RenderMesh, VelocityField,
    closing, csg_difference, csg_intersection, csg_union, dilate, erode, flood_fill_sign,
    gradient_at, gradient_field, laplacian_at, make_level_set_sphere, mean_curvature_at,
    mean_filter, median_filter, mesh_to_level_set, opening, ray_intersect, sample_trilinear,
    volume_to_mesh, Mesh, RayHit, TriMeshRef,
};
pub use io::{load_vol, read_vol, save_vol, write_vol};
pub use points::{Particle, PointDataGrid, PointLeaf};
pub use tree::{AccessorTree, InternalNode, LeafNode, RootNode, Tree, ValueAccessor};

#[cfg(feature = "parallel")]
pub use tools::par_apply;

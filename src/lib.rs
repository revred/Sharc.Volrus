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

pub use grid::{AffineMap, Grid, MetaValue};
pub use nano::{NanoGrid, NanoHeader, NanoLeaf};
pub use math::{Coord, CoordBBox, Ray, Vec3d};
pub use render::{Camera, GpuMesh, GpuVertex, Image, TransferFunction, VolumeRenderConfig};
pub use tools::{
    adaptive_surface, advect_level_set, advect_particles, advect_particles_euler,
    AdaptiveMeshConfig, RenderMesh, VelocityField,
    clip_to_bbox, closing, crop_in_place, csg_difference, csg_intersection, csg_union,
    density_to_level_set, dilate, erode,
    extend_velocity, find_zero_crossings,
    flood_fill_sign,
    gradient_at, gradient_field, laplacian_at, make_level_set_sphere, mean_curvature_at,
    mean_filter, median_filter, merge_copy, merge_intersection, merge_union,
    mesh_to_level_set, opening,
    rasterize_density, rasterize_splatted, rasterize_to_sdf,
    ray_intersect, rebuild_narrow_band, sample_trilinear, trim_narrow_band,
    volume_to_mesh, Mesh, RayHit, TriMeshRef,
};
pub use io::{load_vol, read_vol, save_vol, write_vol};
pub use points::{Particle, PointDataGrid, PointLeaf};
pub use tree::{AccessorTree, InternalNode, LeafNode, RootNode, Tree, ValueAccessor, Visitor};

#[cfg(feature = "parallel")]
pub use tools::par_apply;

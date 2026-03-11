//! VDB tools — high-level operations on sparse volumetric grids.
//!
//! These tools mirror the most impactful OpenVDB utilities:
//! - Level-set CSG (union, intersection, difference)
//! - Flood-fill sign correction for level sets
//! - Trilinear interpolation sampling
//! - Level-set sphere generation
//! - Mesh-to-volume conversion
//! - Volume-to-mesh extraction (marching cubes)

pub mod adaptive_mesh;
pub mod advection;
pub mod clip;
pub mod level_set_track;
pub mod merge;
pub mod csg;
pub mod filter;
pub mod flood_fill;
pub mod gradient;
pub mod interpolation;
pub mod mesh_to_volume;
pub mod morphology;
pub mod parallel;
pub mod rasterize;
pub mod ray_intersect;
pub mod sphere;
pub mod volume_to_mesh;

pub use adaptive_mesh::{adaptive_surface, AdaptiveMeshConfig, RenderMesh};
pub use advection::{advect_level_set, advect_particles, advect_particles_euler, VelocityField};
pub use csg::{csg_difference, csg_intersection, csg_union};
pub use filter::{mean_filter, median_filter};
pub use flood_fill::flood_fill_sign;
pub use gradient::{gradient_at, gradient_field, laplacian_at, mean_curvature_at};
pub use interpolation::sample_trilinear;
pub use mesh_to_volume::{mesh_to_level_set, TriMeshRef};
pub use morphology::{closing, dilate, erode, opening};
pub use clip::{clip_to_bbox, crop_in_place};
pub use level_set_track::{extend_velocity, find_zero_crossings, rebuild_narrow_band, trim_narrow_band};
pub use merge::{merge_copy, merge_intersection, merge_union};
pub use rasterize::{density_to_level_set, rasterize_density, rasterize_splatted, rasterize_to_sdf};
pub use ray_intersect::{ray_intersect, RayHit};
pub use sphere::make_level_set_sphere;
pub use volume_to_mesh::{volume_to_mesh, Mesh};

#[cfg(feature = "parallel")]
pub use parallel::par_apply;

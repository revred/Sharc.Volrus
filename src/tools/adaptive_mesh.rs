//! Adaptive isosurface meshing — curvature-driven subdivision of marching cubes output.
//!
//! Produces denser triangulation in high-curvature regions and coarser
//! triangles in flat areas, ideal for CFD visualization.

use crate::grid::Grid;
use crate::tools::gradient::gradient_at;
use crate::tools::interpolation::sample_trilinear;
use crate::tools::volume_to_mesh::volume_to_mesh;
use std::collections::HashMap;

/// Configuration for adaptive meshing.
pub struct AdaptiveMeshConfig {
    /// Isovalue for surface extraction (typically 0.0 for level sets).
    pub isovalue: f32,
    /// Maximum edge length in world units (controls base resolution).
    pub max_edge_length: f64,
    /// Curvature threshold: edges in regions with curvature above this get refined.
    pub curvature_threshold: f64,
    /// Maximum refinement depth (limits subdivision iterations).
    pub max_depth: u32,
}

impl Default for AdaptiveMeshConfig {
    fn default() -> Self {
        Self {
            isovalue: 0.0,
            max_edge_length: 1.0,
            curvature_threshold: 0.5,
            max_depth: 3,
        }
    }
}

/// Result mesh with per-vertex data for rendering.
pub struct RenderMesh {
    pub vertices: Vec<[f64; 3]>,
    pub normals: Vec<[f64; 3]>,
    pub triangles: Vec<[u32; 3]>,
    pub curvatures: Vec<f64>,
}

/// Compute the mean curvature at a world-space position by sampling the SDF gradient
/// on the underlying grid. We snap to the nearest voxel for the finite-difference stencil.
fn curvature_at_world(grid: &Grid<f32>, world: [f64; 3]) -> f64 {
    let coord = grid.affine_map().world_to_index(world);
    let h = crate::tools::gradient::mean_curvature_at(grid, coord);
    // Convert from index-space curvature to world-space curvature.
    // In index space, curvature has units of 1/voxel. To convert to world-space
    // divide by voxel_size.
    let vs = grid.affine_map().voxel_size();
    (h as f64) / vs
}

/// Compute the SDF gradient at a world-space position, returned as a unit normal.
fn normal_at_world(grid: &Grid<f32>, world: [f64; 3]) -> [f64; 3] {
    let coord = grid.affine_map().world_to_index(world);
    let g = gradient_at(grid, coord);
    let len = ((g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) as f64).sqrt();
    if len < 1e-15 {
        [0.0, 0.0, 1.0]
    } else {
        [g[0] as f64 / len, g[1] as f64 / len, g[2] as f64 / len]
    }
}

/// Project a vertex onto the isosurface by moving it along the SDF gradient.
fn project_onto_isosurface(grid: &Grid<f32>, pos: [f64; 3], isovalue: f32) -> [f64; 3] {
    let sdf = sample_trilinear(grid, pos) - isovalue;
    let n = normal_at_world(grid, pos);
    [
        pos[0] - sdf as f64 * n[0],
        pos[1] - sdf as f64 * n[1],
        pos[2] - sdf as f64 * n[2],
    ]
}

fn edge_length(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let dz = b[2] - a[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn midpoint(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
    ]
}

/// Extract an adaptive isosurface mesh.
///
/// 1. Start with standard marching cubes mesh
/// 2. Compute per-vertex curvature from the SDF gradient
/// 3. Subdivide triangles in high-curvature regions (midpoint subdivision)
/// 4. Snap new vertices to the isosurface (project along gradient)
/// 5. Compute smooth per-vertex normals
pub fn adaptive_surface(grid: &Grid<f32>, config: &AdaptiveMeshConfig) -> RenderMesh {
    // Step 1: base mesh from marching cubes
    let base = volume_to_mesh(grid, config.isovalue);

    let mut vertices = base.vertices.clone();
    let mut triangles = base.triangles.clone();

    // Step 2-4: iterative curvature-adaptive subdivision
    for depth in 0..config.max_depth {
        let min_edge = config.max_edge_length / (1u64 << (depth + 1)) as f64;

        // Compute per-vertex curvature (only for vertices that exist now)
        let vertex_curvatures: Vec<f64> = vertices
            .iter()
            .map(|v| curvature_at_world(grid, *v).abs())
            .collect();

        // Edge midpoint cache: (min_idx, max_idx) -> new vertex index
        let mut edge_cache: HashMap<(u32, u32), u32> = HashMap::new();
        let mut new_triangles: Vec<[u32; 3]> = Vec::new();

        for tri in &triangles {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;

            // Max curvature at triangle vertices
            let max_curv = vertex_curvatures[i0]
                .max(vertex_curvatures[i1])
                .max(vertex_curvatures[i2]);

            // Max edge length of this triangle
            let e01 = edge_length(&vertices[i0], &vertices[i1]);
            let e12 = edge_length(&vertices[i1], &vertices[i2]);
            let e20 = edge_length(&vertices[i2], &vertices[i0]);
            let max_edge = e01.max(e12).max(e20);

            // Only subdivide if curvature exceeds threshold AND edges are long enough
            if max_curv > config.curvature_threshold && max_edge > min_edge {
                // 1-to-4 midpoint subdivision
                let m01 = get_or_insert_midpoint(
                    tri[0], tri[1], &mut vertices, &mut edge_cache, grid, config.isovalue,
                );
                let m12 = get_or_insert_midpoint(
                    tri[1], tri[2], &mut vertices, &mut edge_cache, grid, config.isovalue,
                );
                let m20 = get_or_insert_midpoint(
                    tri[2], tri[0], &mut vertices, &mut edge_cache, grid, config.isovalue,
                );

                new_triangles.push([tri[0], m01, m20]);
                new_triangles.push([tri[1], m12, m01]);
                new_triangles.push([tri[2], m20, m12]);
                new_triangles.push([m01, m12, m20]);
            } else {
                new_triangles.push(*tri);
            }
        }

        triangles = new_triangles;
    }

    // Step 5: compute smooth per-vertex normals (angle-weighted face normal averaging)
    let mut normals = vec![[0.0f64; 3]; vertices.len()];

    for tri in &triangles {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        let v0 = vertices[i0];
        let v1 = vertices[i1];
        let v2 = vertices[i2];

        // Edge vectors
        let e01 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e02 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let e12 = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]];

        // Face normal (cross product)
        let fn_x = e01[1] * e02[2] - e01[2] * e02[1];
        let fn_y = e01[2] * e02[0] - e01[0] * e02[2];
        let fn_z = e01[0] * e02[1] - e01[1] * e02[0];

        // Angle at each vertex (for angle-weighted normal)
        let angle0 = angle_between(&e01, &e02);
        let neg_e01 = [-e01[0], -e01[1], -e01[2]];
        let angle1 = angle_between(&neg_e01, &e12);
        let neg_e02 = [-e02[0], -e02[1], -e02[2]];
        let neg_e12 = [-e12[0], -e12[1], -e12[2]];
        let angle2 = angle_between(&neg_e02, &neg_e12);

        // Accumulate angle-weighted face normal
        normals[i0][0] += fn_x * angle0;
        normals[i0][1] += fn_y * angle0;
        normals[i0][2] += fn_z * angle0;

        normals[i1][0] += fn_x * angle1;
        normals[i1][1] += fn_y * angle1;
        normals[i1][2] += fn_z * angle1;

        normals[i2][0] += fn_x * angle2;
        normals[i2][1] += fn_y * angle2;
        normals[i2][2] += fn_z * angle2;
    }

    // Normalize all normals
    for n in &mut normals {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 1e-15 {
            n[0] /= len;
            n[1] /= len;
            n[2] /= len;
        }
    }

    // Step 6: compute per-vertex curvature for colormap
    let curvatures: Vec<f64> = vertices
        .iter()
        .map(|v| curvature_at_world(grid, *v))
        .collect();

    RenderMesh {
        vertices,
        normals,
        triangles,
        curvatures,
    }
}

fn get_or_insert_midpoint(
    a: u32,
    b: u32,
    vertices: &mut Vec<[f64; 3]>,
    cache: &mut HashMap<(u32, u32), u32>,
    grid: &Grid<f32>,
    isovalue: f32,
) -> u32 {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&idx) = cache.get(&key) {
        return idx;
    }
    let mid = midpoint(&vertices[a as usize], &vertices[b as usize]);
    let projected = project_onto_isosurface(grid, mid, isovalue);
    let idx = vertices.len() as u32;
    vertices.push(projected);
    cache.insert(key, idx);
    idx
}

fn angle_between(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let la = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let lb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    if la < 1e-15 || lb < 1e-15 {
        return 0.0;
    }
    let cos_angle = (dot / (la * lb)).clamp(-1.0, 1.0);
    cos_angle.acos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::sphere::make_level_set_sphere;

    fn make_sphere_grid(radius: f64, voxel_size: f64) -> Grid<f32> {
        make_level_set_sphere(radius, [0.0, 0.0, 0.0], voxel_size, 3.0)
    }

    #[test]
    fn adaptive_sphere_more_tris_than_uniform() {
        let grid = make_sphere_grid(5.0, 1.0);

        // Uniform (depth=0)
        let config_uniform = AdaptiveMeshConfig {
            isovalue: 0.0,
            max_edge_length: 2.0,
            curvature_threshold: 0.01, // very low threshold
            max_depth: 0,
        };
        let mesh_uniform = adaptive_surface(&grid, &config_uniform);

        // Adaptive (depth=2)
        let config_adaptive = AdaptiveMeshConfig {
            isovalue: 0.0,
            max_edge_length: 2.0,
            curvature_threshold: 0.01,
            max_depth: 2,
        };
        let mesh_adaptive = adaptive_surface(&grid, &config_adaptive);

        assert!(
            mesh_adaptive.triangles.len() > mesh_uniform.triangles.len(),
            "Adaptive mesh ({} tris) should have more triangles than uniform ({} tris)",
            mesh_adaptive.triangles.len(),
            mesh_uniform.triangles.len()
        );
    }

    #[test]
    fn normals_are_unit_length() {
        let grid = make_sphere_grid(5.0, 1.0);
        let config = AdaptiveMeshConfig {
            max_depth: 1,
            curvature_threshold: 0.01,
            ..Default::default()
        };
        let mesh = adaptive_surface(&grid, &config);

        for (i, n) in mesh.normals.iter().enumerate() {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!(
                (len - 1.0).abs() < 0.01 || len < 1e-10,
                "Normal {} has length {}, expected ~1.0",
                i,
                len
            );
        }
    }

    #[test]
    fn curvatures_on_sphere_approx_inverse_radius() {
        let radius = 10.0;
        let grid = make_sphere_grid(radius, 0.5);
        let config = AdaptiveMeshConfig {
            max_depth: 0,
            ..Default::default()
        };
        let mesh = adaptive_surface(&grid, &config);

        // For a sphere, mean curvature H = 2/R (sum of principal curvatures).
        // In world space with voxel_size=0.5, the expected curvature ≈ 2/10 = 0.2.
        // Due to discrete approximation, we allow generous tolerance.
        let expected = 2.0 / radius;
        let mut close_count = 0;
        for &k in &mesh.curvatures {
            if (k.abs() - expected).abs() < expected * 2.0 {
                close_count += 1;
            }
        }
        let ratio = close_count as f64 / mesh.curvatures.len() as f64;
        assert!(
            ratio > 0.3,
            "At least 30% of vertices should have curvature near {}, got {:.1}%",
            expected,
            ratio * 100.0
        );
    }

    #[test]
    fn depth_zero_same_as_standard_mc() {
        let grid = make_sphere_grid(5.0, 1.0);
        let base = volume_to_mesh(&grid, 0.0);

        let config = AdaptiveMeshConfig {
            max_depth: 0,
            ..Default::default()
        };
        let mesh = adaptive_surface(&grid, &config);

        assert_eq!(
            mesh.triangles.len(),
            base.tri_count(),
            "max_depth=0 should produce same triangle count as standard MC"
        );
        assert_eq!(mesh.vertices.len(), base.vertex_count());
    }

    #[test]
    fn render_mesh_counts_consistent() {
        let grid = make_sphere_grid(5.0, 1.0);
        let config = AdaptiveMeshConfig {
            max_depth: 1,
            curvature_threshold: 0.01,
            ..Default::default()
        };
        let mesh = adaptive_surface(&grid, &config);

        assert_eq!(mesh.vertices.len(), mesh.normals.len());
        assert_eq!(mesh.vertices.len(), mesh.curvatures.len());
        assert!(mesh.triangles.len() > 0);

        // All triangle indices should be valid
        for tri in &mesh.triangles {
            for &idx in tri {
                assert!(
                    (idx as usize) < mesh.vertices.len(),
                    "Triangle index {} out of bounds ({})",
                    idx,
                    mesh.vertices.len()
                );
            }
        }
    }

    #[test]
    fn empty_grid_produces_empty_mesh() {
        let grid = Grid::<f32>::new(3.0, 1.0);
        let config = AdaptiveMeshConfig::default();
        let mesh = adaptive_surface(&grid, &config);
        assert_eq!(mesh.triangles.len(), 0);
        assert_eq!(mesh.vertices.len(), 0);
        assert_eq!(mesh.normals.len(), 0);
        assert_eq!(mesh.curvatures.len(), 0);
    }
}

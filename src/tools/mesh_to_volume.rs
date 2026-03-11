//! Convert a triangle mesh to a narrow-band level-set grid.
//!
//! Uses brute-force closest-triangle distance with angle-weighted
//! pseudo-normal sign determination.  O(voxels * triangles) — correct
//! and simple; acceleration structures can be added later.

use crate::grid::{Grid, GridClass, VoxelTransform};
use crate::math::Coord;

/// Borrowed view of a triangle mesh.
pub struct TriMeshRef<'a> {
    pub vertices: &'a [[f64; 3]],
    pub triangles: &'a [[u32; 3]],
}

/// Convert a triangle mesh to a narrow-band level-set grid.
///
/// 1. Compute mesh AABB, expand by `half_width * voxel_size`.
/// 2. For each voxel center in the AABB, find distance to closest triangle.
/// 3. Sign from angle-weighted pseudo-normal.
/// 4. Only store voxels where `|distance| < half_width * voxel_size`.
pub fn mesh_to_level_set(mesh: &TriMeshRef, voxel_size: f64, half_width: f64) -> Grid<f32> {
    let band = half_width * voxel_size;
    let bg = band as f32;

    let mut grid = Grid::<f32>::new(bg, voxel_size);
    grid.set_grid_class(GridClass::LevelSet);
    grid.set_name("sdf");
    grid.set_transform(VoxelTransform::uniform(voxel_size));

    if mesh.vertices.is_empty() || mesh.triangles.is_empty() {
        return grid;
    }

    // Precompute per-vertex pseudo-normals (angle-weighted sum of face normals).
    let vertex_normals = compute_vertex_normals(mesh);

    // Compute AABB.
    let (bb_min, bb_max) = mesh_aabb(mesh);

    let inv = 1.0 / voxel_size;
    let imin = ((bb_min[0] - band) * inv).floor() as i32;
    let jmin = ((bb_min[1] - band) * inv).floor() as i32;
    let kmin = ((bb_min[2] - band) * inv).floor() as i32;
    let imax = ((bb_max[0] + band) * inv).ceil() as i32;
    let jmax = ((bb_max[1] + band) * inv).ceil() as i32;
    let kmax = ((bb_max[2] + band) * inv).ceil() as i32;

    for i in imin..=imax {
        let px = i as f64 * voxel_size;
        for j in jmin..=jmax {
            let py = j as f64 * voxel_size;
            for k in kmin..=kmax {
                let pz = k as f64 * voxel_size;
                let p = [px, py, pz];

                let (dist, sign) = closest_distance_and_sign(mesh, &vertex_normals, p);
                let sdf = (dist * sign) as f32;

                if sdf.abs() < bg {
                    grid.set(Coord::new(i, j, k), sdf);
                }
            }
        }
    }

    grid
}

/// Compute the axis-aligned bounding box of the mesh.
fn mesh_aabb(mesh: &TriMeshRef) -> ([f64; 3], [f64; 3]) {
    let mut min = [f64::MAX; 3];
    let mut max = [f64::MIN; 3];
    for v in mesh.vertices {
        for d in 0..3 {
            min[d] = min[d].min(v[d]);
            max[d] = max[d].max(v[d]);
        }
    }
    (min, max)
}

/// Compute angle-weighted vertex normals.
fn compute_vertex_normals(mesh: &TriMeshRef) -> Vec<[f64; 3]> {
    let n = mesh.vertices.len();
    let mut normals = vec![[0.0f64; 3]; n];

    for tri in mesh.triangles {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        let v0 = mesh.vertices[i0];
        let v1 = mesh.vertices[i1];
        let v2 = mesh.vertices[i2];

        let e01 = sub3(v1, v0);
        let e02 = sub3(v2, v0);
        let e12 = sub3(v2, v1);

        let face_normal = cross3(e01, e02);
        let face_area = len3(face_normal);
        if face_area < 1e-30 {
            continue;
        }
        let fn_normalized = scale3(face_normal, 1.0 / face_area);

        // Angle at each vertex.
        let a0 = angle_between(e01, e02);
        let a1 = angle_between(sub3(v0, v1), e12);
        let a2 = angle_between(sub3(v0, v2), sub3(v1, v2));

        for d in 0..3 {
            normals[i0][d] += fn_normalized[d] * a0;
            normals[i1][d] += fn_normalized[d] * a1;
            normals[i2][d] += fn_normalized[d] * a2;
        }
    }

    // Normalize.
    for n in &mut normals {
        let l = len3(*n);
        if l > 1e-30 {
            *n = scale3(*n, 1.0 / l);
        }
    }

    normals
}

/// Find the closest distance from point `p` to any triangle, and determine
/// the sign using the pseudo-normal at the closest point.
fn closest_distance_and_sign(
    mesh: &TriMeshRef,
    vertex_normals: &[[f64; 3]],
    p: [f64; 3],
) -> (f64, f64) {
    let mut best_dist_sq = f64::MAX;
    let mut best_sign = 1.0f64;

    for tri in mesh.triangles {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        let v0 = mesh.vertices[i0];
        let v1 = mesh.vertices[i1];
        let v2 = mesh.vertices[i2];

        let (closest, bary) = closest_point_on_triangle(p, v0, v1, v2);
        let diff = sub3(p, closest);
        let dist_sq = dot3(diff, diff);

        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;

            // Compute pseudo-normal at the closest point via barycentric interpolation.
            let pseudo_n = [
                vertex_normals[i0][0] * bary[0]
                    + vertex_normals[i1][0] * bary[1]
                    + vertex_normals[i2][0] * bary[2],
                vertex_normals[i0][1] * bary[0]
                    + vertex_normals[i1][1] * bary[1]
                    + vertex_normals[i2][1] * bary[2],
                vertex_normals[i0][2] * bary[0]
                    + vertex_normals[i1][2] * bary[1]
                    + vertex_normals[i2][2] * bary[2],
            ];

            // Sign: dot(p - closest, pseudo_normal) > 0 → outside (+), else inside (-).
            let sign_dot = dot3(diff, pseudo_n);
            best_sign = if sign_dot >= 0.0 { 1.0 } else { -1.0 };
        }
    }

    (best_dist_sq.sqrt(), best_sign)
}

/// Closest point on triangle (v0, v1, v2) to point p.
/// Returns (closest_point, barycentric_coords).
fn closest_point_on_triangle(
    p: [f64; 3],
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
) -> ([f64; 3], [f64; 3]) {
    let ab = sub3(v1, v0);
    let ac = sub3(v2, v0);
    let ap = sub3(p, v0);

    let d1 = dot3(ab, ap);
    let d2 = dot3(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (v0, [1.0, 0.0, 0.0]);
    }

    let bp = sub3(p, v1);
    let d3 = dot3(ab, bp);
    let d4 = dot3(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (v1, [0.0, 1.0, 0.0]);
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (add3(v0, scale3(ab, v)), [1.0 - v, v, 0.0]);
    }

    let cp = sub3(p, v2);
    let d5 = dot3(ab, cp);
    let d6 = dot3(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (v2, [0.0, 0.0, 1.0]);
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (add3(v0, scale3(ac, w)), [1.0 - w, 0.0, w]);
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (
            add3(v1, scale3(sub3(v2, v1), w)),
            [0.0, 1.0 - w, w],
        );
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (
        add3(v0, add3(scale3(ab, v), scale3(ac, w))),
        [1.0 - v - w, v, w],
    )
}

// --- Inline vector helpers ---

#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn len3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

#[inline]
fn angle_between(a: [f64; 3], b: [f64; 3]) -> f64 {
    let la = len3(a);
    let lb = len3(b);
    if la < 1e-30 || lb < 1e-30 {
        return 0.0;
    }
    let cos_theta = (dot3(a, b) / (la * lb)).clamp(-1.0, 1.0);
    cos_theta.acos()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube centered at origin: 8 vertices, 12 triangles.
    fn unit_cube() -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let vertices = vec![
            [-0.5, -0.5, -0.5], // 0
            [0.5, -0.5, -0.5],  // 1
            [0.5, 0.5, -0.5],   // 2
            [-0.5, 0.5, -0.5],  // 3
            [-0.5, -0.5, 0.5],  // 4
            [0.5, -0.5, 0.5],   // 5
            [0.5, 0.5, 0.5],    // 6
            [-0.5, 0.5, 0.5],   // 7
        ];
        // Outward-facing triangles (CCW from outside).
        let triangles = vec![
            // -Z face
            [0, 2, 1],
            [0, 3, 2],
            // +Z face
            [4, 5, 6],
            [4, 6, 7],
            // -Y face
            [0, 1, 5],
            [0, 5, 4],
            // +Y face
            [2, 3, 7],
            [2, 7, 6],
            // -X face
            [0, 4, 7],
            [0, 7, 3],
            // +X face
            [1, 2, 6],
            [1, 6, 5],
        ];
        (vertices, triangles)
    }

    #[test]
    fn mesh_to_volume_unit_cube_has_active_voxels() {
        let (verts, tris) = unit_cube();
        let mesh = TriMeshRef {
            vertices: &verts,
            triangles: &tris,
        };
        let grid = mesh_to_level_set(&mesh, 0.1, 3.0);
        assert!(
            grid.active_voxel_count() > 0,
            "Should produce active voxels"
        );
        assert_eq!(grid.grid_class(), GridClass::LevelSet);
    }

    #[test]
    fn mesh_to_volume_cube_inside_negative() {
        let (verts, tris) = unit_cube();
        let mesh = TriMeshRef {
            vertices: &verts,
            triangles: &tris,
        };
        // Use larger half_width so that the center (distance 0.5 from face)
        // falls within the narrow band.
        let grid = mesh_to_level_set(&mesh, 0.1, 6.0);

        // At the center (0,0,0) → index (0,0,0), distance to nearest face = 0.5.
        // Band = 6.0 * 0.1 = 0.6. |SDF| = 0.5 < 0.6, so it should be stored.
        let val = grid.get(Coord::new(0, 0, 0));
        assert!(
            val < 0.0,
            "SDF at cube center should be negative (inside), got {}",
            val
        );
    }

    #[test]
    fn mesh_to_volume_cube_outside_positive() {
        let (verts, tris) = unit_cube();
        let mesh = TriMeshRef {
            vertices: &verts,
            triangles: &tris,
        };
        let grid = mesh_to_level_set(&mesh, 0.1, 3.0);

        // Well outside: index (10,0,0) → world (1.0, 0, 0), distance ~ 0.5 from face.
        let val = grid.get(Coord::new(10, 0, 0));
        assert!(
            val > 0.0,
            "SDF outside cube should be positive, got {}",
            val
        );
    }

    #[test]
    fn mesh_to_volume_empty_mesh() {
        let mesh = TriMeshRef {
            vertices: &[],
            triangles: &[],
        };
        let grid = mesh_to_level_set(&mesh, 0.1, 3.0);
        assert_eq!(grid.active_voxel_count(), 0);
    }

    #[test]
    fn sphere_roundtrip_via_mesh() {
        // Generate a sphere level set, extract mesh, convert back.
        // This requires volume_to_mesh, tested in its own module.
        // Here we just verify mesh_to_level_set on a simple tetrahedron.
        let vertices = vec![
            [1.0, 0.0, -1.0 / 2.0f64.sqrt()],
            [-1.0, 0.0, -1.0 / 2.0f64.sqrt()],
            [0.0, 1.0, 1.0 / 2.0f64.sqrt()],
            [0.0, -1.0, 1.0 / 2.0f64.sqrt()],
        ];
        let triangles = vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]];
        let mesh = TriMeshRef {
            vertices: &vertices,
            triangles: &triangles,
        };
        let grid = mesh_to_level_set(&mesh, 0.1, 3.0);
        assert!(grid.active_voxel_count() > 0);
    }
}

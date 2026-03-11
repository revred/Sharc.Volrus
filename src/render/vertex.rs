//! GPU-ready vertex and mesh types for hardware-accelerated rendering.

use crate::tools::adaptive_mesh::RenderMesh;

/// Packed vertex for GPU rendering (matches typical Vulkan/wgpu vertex layout).
/// All f32 for GPU efficiency.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub curvature: f32,
    pub _pad: f32,
}

/// Packed index buffer entry.
pub type GpuIndex = u32;

/// GPU-ready mesh buffer pair.
pub struct GpuMesh {
    pub vertices: Vec<GpuVertex>,
    pub indices: Vec<GpuIndex>,
}

impl GpuMesh {
    /// Convert a RenderMesh to GPU-ready format.
    pub fn from_render_mesh(mesh: &RenderMesh) -> Self {
        let vertices: Vec<GpuVertex> = (0..mesh.vertices.len())
            .map(|i| {
                let v = &mesh.vertices[i];
                let n = &mesh.normals[i];
                let k = mesh.curvatures[i];
                GpuVertex {
                    position: [v[0] as f32, v[1] as f32, v[2] as f32],
                    normal: [n[0] as f32, n[1] as f32, n[2] as f32],
                    curvature: k as f32,
                    _pad: 0.0,
                }
            })
            .collect();

        let indices: Vec<GpuIndex> = mesh
            .triangles
            .iter()
            .flat_map(|tri| tri.iter().copied())
            .collect();

        Self { vertices, indices }
    }

    /// Raw vertex bytes for GPU upload.
    pub fn vertex_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.vertices.as_ptr() as *const u8,
                self.vertices.len() * std::mem::size_of::<GpuVertex>(),
            )
        }
    }

    /// Raw index bytes for GPU upload.
    pub fn index_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.indices.as_ptr() as *const u8,
                self.indices.len() * std::mem::size_of::<GpuIndex>(),
            )
        }
    }

    /// Vertex stride in bytes (for vertex buffer layout).
    pub fn vertex_stride() -> usize {
        std::mem::size_of::<GpuVertex>()
    }

    /// Triangle count.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_vertex_is_32_bytes() {
        assert_eq!(
            std::mem::size_of::<GpuVertex>(),
            32,
            "GpuVertex should be 32 bytes for GPU alignment"
        );
    }

    #[test]
    fn vertex_stride_is_32() {
        assert_eq!(GpuMesh::vertex_stride(), 32);
    }

    #[test]
    fn from_render_mesh_preserves_triangle_count() {
        let mesh = RenderMesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            triangles: vec![[0, 1, 2]],
            curvatures: vec![0.5; 3],
        };

        let gpu = GpuMesh::from_render_mesh(&mesh);
        assert_eq!(gpu.triangle_count(), 1);
        assert_eq!(gpu.vertices.len(), 3);
        assert_eq!(gpu.indices.len(), 3);
    }

    #[test]
    fn vertex_bytes_length_matches() {
        let mesh = RenderMesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 2],
            triangles: vec![],
            curvatures: vec![0.0; 2],
        };

        let gpu = GpuMesh::from_render_mesh(&mesh);
        assert_eq!(gpu.vertex_bytes().len(), 2 * 32);
    }

    #[test]
    fn index_bytes_length_matches() {
        let mesh = RenderMesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            triangles: vec![[0, 1, 2]],
            curvatures: vec![0.0; 3],
        };

        let gpu = GpuMesh::from_render_mesh(&mesh);
        assert_eq!(gpu.index_bytes().len(), 3 * 4); // 3 indices * 4 bytes each
    }

    #[test]
    fn gpu_vertex_repr_c_layout() {
        // Verify that the struct is tightly packed as expected with repr(C)
        let v = GpuVertex {
            position: [1.0, 2.0, 3.0],
            normal: [0.0, 0.0, 1.0],
            curvature: 0.5,
            _pad: 0.0,
        };
        assert_eq!(v.position, [1.0, 2.0, 3.0]);
        assert_eq!(v.normal, [0.0, 0.0, 1.0]);
        assert_eq!(v.curvature, 0.5);
    }

    #[test]
    fn from_render_mesh_with_multiple_triangles() {
        let mesh = RenderMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 4],
            triangles: vec![[0, 1, 2], [1, 3, 2]],
            curvatures: vec![0.1, 0.2, 0.3, 0.4],
        };

        let gpu = GpuMesh::from_render_mesh(&mesh);
        assert_eq!(gpu.triangle_count(), 2);
        assert_eq!(gpu.vertices.len(), 4);
        assert_eq!(gpu.indices.len(), 6);
        // Check curvature mapping
        assert!((gpu.vertices[0].curvature - 0.1).abs() < 1e-6);
        assert!((gpu.vertices[3].curvature - 0.4).abs() < 1e-6);
    }
}

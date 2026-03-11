//! PointDataGrid — sparse grid of particles binned into VDB leaf structure.

use crate::grid::AffineMap;
use crate::math::{Coord, CoordBBox};
use crate::tree::leaf::{LEAF_DIM, LEAF_LOG2DIM};
use std::collections::HashMap;

/// A single particle with position and optional attributes.
#[derive(Debug, Clone, Copy)]
pub struct Particle {
    /// World-space position.
    pub position: [f64; 3],
    /// Velocity vector.
    pub velocity: [f64; 3],
    /// Unique particle ID.
    pub id: u64,
}

/// A leaf bucket holding particles that fall within this leaf's voxel range.
pub struct PointLeaf {
    origin: Coord,
    particles: Vec<Particle>,
}

impl PointLeaf {
    /// Create an empty leaf at the given (aligned) origin.
    fn new(origin: Coord) -> Self {
        Self {
            origin,
            particles: Vec::new(),
        }
    }

    /// Origin coordinate of this leaf (tile-aligned).
    pub fn origin(&self) -> Coord {
        self.origin
    }

    /// Number of particles in this leaf.
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Iterate over particles in this leaf.
    pub fn iter(&self) -> impl Iterator<Item = &Particle> {
        self.particles.iter()
    }
}

/// Sparse grid of particles, binned into VDB leaf structure.
/// Each leaf covers an 8^3 voxel region and holds all particles
/// whose positions map to voxels within that region.
pub struct PointDataGrid {
    leaves: HashMap<Coord, PointLeaf>,
    voxel_size: f64,
    transform: AffineMap,
}

impl PointDataGrid {
    /// Create a new point data grid with the given voxel size.
    pub fn new(voxel_size: f64) -> Self {
        Self {
            leaves: HashMap::new(),
            voxel_size,
            transform: AffineMap::from_uniform_scale(voxel_size),
        }
    }

    /// Create a new point data grid with a full affine transform.
    pub fn with_affine(map: AffineMap) -> Self {
        Self {
            leaves: HashMap::new(),
            voxel_size: map.voxel_size(),
            transform: map,
        }
    }

    /// Voxel size of this grid.
    pub fn voxel_size(&self) -> f64 {
        self.voxel_size
    }

    /// Reference to the affine transform.
    pub fn affine_map(&self) -> &AffineMap {
        &self.transform
    }

    /// Convert a world-space position to the leaf origin it belongs to.
    fn leaf_origin_for(&self, pos: [f64; 3]) -> Coord {
        let idx = self.transform.world_to_index(pos);
        idx.aligned(LEAF_LOG2DIM)
    }

    /// Convert a world-space position to the voxel coordinate it snaps to.
    fn voxel_coord_for(&self, pos: [f64; 3]) -> Coord {
        self.transform.world_to_index(pos)
    }

    /// Insert a particle into the correct leaf.
    pub fn insert(&mut self, particle: Particle) {
        let origin = self.leaf_origin_for(particle.position);
        let leaf = self
            .leaves
            .entry(origin)
            .or_insert_with(|| PointLeaf::new(origin));
        leaf.particles.push(particle);
    }

    /// Batch insert particles.
    pub fn insert_batch(&mut self, particles: &[Particle]) {
        for &p in particles {
            self.insert(p);
        }
    }

    /// Remove a particle by ID. Returns the removed particle if found.
    pub fn remove(&mut self, id: u64) -> Option<Particle> {
        for leaf in self.leaves.values_mut() {
            if let Some(idx) = leaf.particles.iter().position(|p| p.id == id) {
                return Some(leaf.particles.swap_remove(idx));
            }
        }
        None
    }

    /// Total number of particles across all leaves.
    pub fn particle_count(&self) -> usize {
        self.leaves.values().map(|l| l.particles.len()).sum()
    }

    /// Number of occupied leaves.
    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }

    /// All particles in a specific voxel coordinate.
    pub fn particles_in_voxel(&self, coord: Coord) -> Vec<&Particle> {
        let origin = coord.aligned(LEAF_LOG2DIM);
        match self.leaves.get(&origin) {
            Some(leaf) => leaf
                .particles
                .iter()
                .filter(|p| self.voxel_coord_for(p.position) == coord)
                .collect(),
            None => Vec::new(),
        }
    }

    /// All particles within a given world-space radius of a center point.
    pub fn particles_in_radius(&self, center: [f64; 3], radius: f64) -> Vec<&Particle> {
        let r2 = radius * radius;

        // Determine which leaves could potentially overlap the sphere.
        // Convert radius to index-space extent and find all candidate leaf origins.
        let inv_vs = 1.0 / self.voxel_size;
        let r_idx = (radius * inv_vs).ceil() as i32 + LEAF_DIM;
        let center_coord = self.transform.world_to_index(center);
        let min_leaf = Coord::new(
            center_coord.x - r_idx,
            center_coord.y - r_idx,
            center_coord.z - r_idx,
        )
        .aligned(LEAF_LOG2DIM);
        let max_leaf = Coord::new(
            center_coord.x + r_idx,
            center_coord.y + r_idx,
            center_coord.z + r_idx,
        )
        .aligned(LEAF_LOG2DIM);

        let mut result = Vec::new();

        // Iterate candidate leaf origins
        let mut ox = min_leaf.x;
        while ox <= max_leaf.x {
            let mut oy = min_leaf.y;
            while oy <= max_leaf.y {
                let mut oz = min_leaf.z;
                while oz <= max_leaf.z {
                    let origin = Coord::new(ox, oy, oz);
                    if let Some(leaf) = self.leaves.get(&origin) {
                        for p in &leaf.particles {
                            let dx = p.position[0] - center[0];
                            let dy = p.position[1] - center[1];
                            let dz = p.position[2] - center[2];
                            if dx * dx + dy * dy + dz * dz <= r2 {
                                result.push(p);
                            }
                        }
                    }
                    oz += LEAF_DIM;
                }
                oy += LEAF_DIM;
            }
            ox += LEAF_DIM;
        }

        result
    }

    /// Find the nearest particle to a world-space position.
    pub fn nearest_particle(&self, pos: [f64; 3]) -> Option<&Particle> {
        let mut best: Option<(&Particle, f64)> = None;

        for leaf in self.leaves.values() {
            for p in &leaf.particles {
                let dx = p.position[0] - pos[0];
                let dy = p.position[1] - pos[1];
                let dz = p.position[2] - pos[2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if best.is_none() || d2 < best.unwrap().1 {
                    best = Some((p, d2));
                }
            }
        }

        best.map(|(p, _)| p)
    }

    /// Iterate over all particles across all leaves.
    pub fn iter_particles(&self) -> impl Iterator<Item = &Particle> {
        self.leaves.values().flat_map(|l| l.particles.iter())
    }

    /// Mutable access to all particles (used by advection for position updates).
    pub fn iter_particles_mut(&mut self) -> impl Iterator<Item = &mut Particle> {
        self.leaves.values_mut().flat_map(|l| l.particles.iter_mut())
    }

    /// Bounding box of occupied leaves in index space.
    pub fn active_bbox(&self) -> Option<CoordBBox> {
        if self.leaves.is_empty() {
            return None;
        }
        let mut bbox = CoordBBox::empty();
        for origin in self.leaves.keys() {
            bbox.expand(*origin);
            bbox.expand(Coord::new(
                origin.x + LEAF_DIM - 1,
                origin.y + LEAF_DIM - 1,
                origin.z + LEAF_DIM - 1,
            ));
        }
        Some(bbox)
    }

    /// Drain all particles out, returning them as a Vec.
    /// Leaves the grid empty.
    pub(crate) fn drain_all(&mut self) -> Vec<Particle> {
        let mut all = Vec::with_capacity(self.particle_count());
        for leaf in self.leaves.values_mut() {
            all.append(&mut leaf.particles);
        }
        self.leaves.clear();
        all
    }

    /// Re-bin particles after their positions have changed (e.g., after advection).
    /// Drains all particles and re-inserts them according to current positions.
    pub fn rebin(&mut self) {
        let particles = self.drain_all();
        for p in particles {
            self.insert(p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_particle(id: u64, x: f64, y: f64, z: f64) -> Particle {
        Particle {
            position: [x, y, z],
            velocity: [0.0, 0.0, 0.0],
            id,
        }
    }

    #[test]
    fn insert_single_particle_correct_leaf() {
        let mut grid = PointDataGrid::new(1.0);
        grid.insert(make_particle(1, 3.5, 2.1, 1.9));
        assert_eq!(grid.particle_count(), 1);
        assert_eq!(grid.leaf_count(), 1);
        // Position (3.5, 2.1, 1.9) at voxel_size=1 -> index (4, 2, 2) -> leaf origin (0, 0, 0)
        let origin = Coord::new(0, 0, 0);
        assert!(grid.leaves.contains_key(&origin));
    }

    #[test]
    fn insert_particles_different_leaves() {
        let mut grid = PointDataGrid::new(1.0);
        // Leaf 1: origin (0,0,0), covers indices 0..7
        grid.insert(make_particle(1, 1.0, 1.0, 1.0));
        // Leaf 2: origin (8,0,0), covers indices 8..15
        grid.insert(make_particle(2, 10.0, 1.0, 1.0));
        assert_eq!(grid.particle_count(), 2);
        assert_eq!(grid.leaf_count(), 2);
    }

    #[test]
    fn particles_in_voxel_returns_correct_subset() {
        let mut grid = PointDataGrid::new(1.0);
        // Both map to voxel (2,2,2) with rounding
        grid.insert(make_particle(1, 2.1, 2.2, 2.3));
        grid.insert(make_particle(2, 1.6, 1.7, 1.8)); // rounds to (2,2,2)
        // This one maps to voxel (5,5,5)
        grid.insert(make_particle(3, 5.0, 5.0, 5.0));

        let in_voxel = grid.particles_in_voxel(Coord::new(2, 2, 2));
        assert_eq!(in_voxel.len(), 2);
        let ids: Vec<u64> = in_voxel.iter().map(|p| p.id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));

        let in_voxel_5 = grid.particles_in_voxel(Coord::new(5, 5, 5));
        assert_eq!(in_voxel_5.len(), 1);
        assert_eq!(in_voxel_5[0].id, 3);

        // Empty voxel
        let empty = grid.particles_in_voxel(Coord::new(0, 0, 0));
        assert!(empty.is_empty());
    }

    #[test]
    fn particles_in_radius_spatial_query() {
        let mut grid = PointDataGrid::new(1.0);
        grid.insert(make_particle(1, 0.0, 0.0, 0.0));
        grid.insert(make_particle(2, 1.0, 0.0, 0.0));
        grid.insert(make_particle(3, 5.0, 0.0, 0.0));
        grid.insert(make_particle(4, 10.0, 0.0, 0.0));

        let near = grid.particles_in_radius([0.0, 0.0, 0.0], 2.0);
        let ids: Vec<u64> = near.iter().map(|p| p.id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(!ids.contains(&3));
        assert!(!ids.contains(&4));
    }

    #[test]
    fn nearest_particle_finds_closest() {
        let mut grid = PointDataGrid::new(1.0);
        grid.insert(make_particle(1, 0.0, 0.0, 0.0));
        grid.insert(make_particle(2, 3.0, 0.0, 0.0));
        grid.insert(make_particle(3, 10.0, 0.0, 0.0));

        let nearest = grid.nearest_particle([2.5, 0.0, 0.0]).unwrap();
        assert_eq!(nearest.id, 2);

        let nearest2 = grid.nearest_particle([9.0, 0.0, 0.0]).unwrap();
        assert_eq!(nearest2.id, 3);
    }

    #[test]
    fn nearest_particle_empty_grid() {
        let grid = PointDataGrid::new(1.0);
        assert!(grid.nearest_particle([0.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn remove_by_id() {
        let mut grid = PointDataGrid::new(1.0);
        grid.insert(make_particle(1, 0.0, 0.0, 0.0));
        grid.insert(make_particle(2, 1.0, 0.0, 0.0));
        grid.insert(make_particle(3, 2.0, 0.0, 0.0));
        assert_eq!(grid.particle_count(), 3);

        let removed = grid.remove(2).unwrap();
        assert_eq!(removed.id, 2);
        assert_eq!(grid.particle_count(), 2);

        // Remove non-existent
        assert!(grid.remove(99).is_none());
        assert_eq!(grid.particle_count(), 2);
    }

    #[test]
    fn batch_insert() {
        let mut grid = PointDataGrid::new(1.0);
        let particles = vec![
            make_particle(1, 0.0, 0.0, 0.0),
            make_particle(2, 1.0, 1.0, 1.0),
            make_particle(3, 10.0, 10.0, 10.0),
        ];
        grid.insert_batch(&particles);
        assert_eq!(grid.particle_count(), 3);
        // First two are in the same leaf (both in 0..7 range)
        // Third is in a different leaf (index 10 -> leaf origin 8)
        assert_eq!(grid.leaf_count(), 2);
    }

    #[test]
    fn empty_grid_zero_particles_and_leaves() {
        let grid = PointDataGrid::new(1.0);
        assert_eq!(grid.particle_count(), 0);
        assert_eq!(grid.leaf_count(), 0);
        assert!(grid.active_bbox().is_none());
    }

    #[test]
    fn iter_particles_all() {
        let mut grid = PointDataGrid::new(1.0);
        grid.insert(make_particle(1, 0.0, 0.0, 0.0));
        grid.insert(make_particle(2, 10.0, 0.0, 0.0));
        grid.insert(make_particle(3, 20.0, 0.0, 0.0));

        let ids: Vec<u64> = grid.iter_particles().map(|p| p.id).collect();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
    }

    #[test]
    fn active_bbox_covers_leaves() {
        let mut grid = PointDataGrid::new(1.0);
        grid.insert(make_particle(1, 0.0, 0.0, 0.0));
        grid.insert(make_particle(2, 10.0, 10.0, 10.0));

        let bbox = grid.active_bbox().unwrap();
        // Leaf 1: origin (0,0,0), Leaf 2: origin (8,8,8)
        assert_eq!(bbox.min, Coord::new(0, 0, 0));
        // Leaf 2 goes up to (8+7, 8+7, 8+7) = (15, 15, 15)
        assert_eq!(bbox.max, Coord::new(15, 15, 15));
    }

    #[test]
    fn rebin_moves_particles_to_correct_leaves() {
        let mut grid = PointDataGrid::new(1.0);
        grid.insert(make_particle(1, 1.0, 1.0, 1.0));
        assert_eq!(grid.leaf_count(), 1);
        assert!(grid.leaves.contains_key(&Coord::new(0, 0, 0)));

        // Manually move the particle's position to a different leaf region
        for p in grid.iter_particles_mut() {
            p.position = [20.0, 20.0, 20.0];
        }

        // Before rebin, still in old leaf
        assert!(grid.leaves.contains_key(&Coord::new(0, 0, 0)));

        grid.rebin();

        // After rebin, particle should be in the new leaf
        assert_eq!(grid.particle_count(), 1);
        assert_eq!(grid.leaf_count(), 1);
        // (20,20,20) -> index (20,20,20) -> leaf origin (16,16,16)
        assert!(grid.leaves.contains_key(&Coord::new(16, 16, 16)));
    }

    #[test]
    fn with_affine_constructor() {
        let map = AffineMap::from_uniform_scale(0.5);
        let mut grid = PointDataGrid::with_affine(map);
        // Position (1.0, 1.0, 1.0) at voxel_size=0.5 -> index (2, 2, 2)
        grid.insert(make_particle(1, 1.0, 1.0, 1.0));
        assert_eq!(grid.particle_count(), 1);
        assert_eq!(grid.voxel_size(), 0.5);
    }

    #[test]
    fn particles_in_radius_across_leaves() {
        let mut grid = PointDataGrid::new(1.0);
        // Put particles in different leaves but close in world space
        grid.insert(make_particle(1, 7.0, 0.0, 0.0)); // leaf origin (0,0,0)
        grid.insert(make_particle(2, 9.0, 0.0, 0.0)); // leaf origin (8,0,0)

        // Radius query centered at boundary should find both
        let near = grid.particles_in_radius([8.0, 0.0, 0.0], 2.0);
        assert_eq!(near.len(), 2);
    }
}

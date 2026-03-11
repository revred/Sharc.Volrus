//! Grid — top-level sparse volume container with transform.
//!
//! Mirrors `openvdb::Grid<TreeT>`: a tree plus an index-to-world
//! affine transform, a name, and a grid class tag.

use crate::math::{Coord, CoordBBox};
use crate::tree::Tree;
use std::collections::HashMap;

/// A typed value in a grid's metadata dictionary.
///
/// Mirrors the polymorphic metadata entries in OpenVDB's `GridBase::Meta*` API.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    /// UTF-8 string.
    String(String),
    /// 64-bit signed integer.
    Int(i64),
    /// 64-bit float.
    Float(f64),
    /// Boolean flag.
    Bool(bool),
    /// 3-component double-precision vector.
    Vec3([f64; 3]),
}

impl From<&str> for MetaValue {
    fn from(s: &str) -> Self { MetaValue::String(s.to_string()) }
}
impl From<String> for MetaValue {
    fn from(s: String) -> Self { MetaValue::String(s) }
}
impl From<i64> for MetaValue {
    fn from(v: i64) -> Self { MetaValue::Int(v) }
}
impl From<f64> for MetaValue {
    fn from(v: f64) -> Self { MetaValue::Float(v) }
}
impl From<bool> for MetaValue {
    fn from(v: bool) -> Self { MetaValue::Bool(v) }
}
impl From<[f64; 3]> for MetaValue {
    fn from(v: [f64; 3]) -> Self { MetaValue::Vec3(v) }
}

/// Classification of the grid's semantic role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridClass {
    /// Signed distance field (narrow band around surface).
    LevelSet,
    /// Density / fog volume (0 = empty, 1 = opaque).
    FogVolume,
    /// Unclassified.
    Unknown,
}

/// Uniform-scale affine transform: index-space <-> world-space.
///
/// Retained for backward compatibility. Internally, `Grid` stores an
/// `AffineMap` that subsumes this.
#[derive(Debug, Clone, Copy)]
pub struct VoxelTransform {
    /// Voxel size in world units (uniform scale).
    pub voxel_size: f64,
    /// Translation offset in world units (origin of index (0,0,0)).
    pub origin: [f64; 3],
}

impl VoxelTransform {
    /// Identity transform: voxel_size = 1.0, origin at world origin.
    pub fn identity() -> Self {
        Self {
            voxel_size: 1.0,
            origin: [0.0; 3],
        }
    }

    /// Create a uniform-scale transform.
    pub fn uniform(voxel_size: f64) -> Self {
        Self {
            voxel_size,
            origin: [0.0; 3],
        }
    }

    /// Create with both scale and origin.
    pub fn with_origin(voxel_size: f64, origin: [f64; 3]) -> Self {
        Self { voxel_size, origin }
    }

    /// Convert index-space coordinate to world-space position.
    #[inline]
    pub fn index_to_world(&self, coord: Coord) -> [f64; 3] {
        [
            coord.x as f64 * self.voxel_size + self.origin[0],
            coord.y as f64 * self.voxel_size + self.origin[1],
            coord.z as f64 * self.voxel_size + self.origin[2],
        ]
    }

    /// Convert world-space position to nearest index-space coordinate.
    #[inline]
    pub fn world_to_index(&self, world: [f64; 3]) -> Coord {
        let inv = 1.0 / self.voxel_size;
        Coord::new(
            ((world[0] - self.origin[0]) * inv).round() as i32,
            ((world[1] - self.origin[1]) * inv).round() as i32,
            ((world[2] - self.origin[2]) * inv).round() as i32,
        )
    }

    /// Convert world-space position to continuous index-space (no rounding).
    #[inline]
    pub fn world_to_index_f64(&self, world: [f64; 3]) -> [f64; 3] {
        let inv = 1.0 / self.voxel_size;
        [
            (world[0] - self.origin[0]) * inv,
            (world[1] - self.origin[1]) * inv,
            (world[2] - self.origin[2]) * inv,
        ]
    }
}

// ---------------------------------------------------------------------------
// AffineMap — full 4x4 affine transform
// ---------------------------------------------------------------------------

/// 4x4 affine transform matrix (row-major storage).
/// Maps between index-space and world-space.
#[derive(Debug, Clone, Copy)]
pub struct AffineMap {
    /// Forward matrix: index -> world (row-major).
    mat: [[f64; 4]; 4],
    /// Inverse matrix: world -> index (row-major).
    inv: [[f64; 4]; 4],
    /// Cached voxel size (column-length of the linear part).
    voxel_size: f64,
}

/// Identity 4x4 matrix.
const IDENTITY_4X4: [[f64; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

impl AffineMap {
    /// Identity transform.
    pub fn identity() -> Self {
        Self {
            mat: IDENTITY_4X4,
            inv: IDENTITY_4X4,
            voxel_size: 1.0,
        }
    }

    /// Uniform-scale transform (no rotation, no translation).
    pub fn from_uniform_scale(voxel_size: f64) -> Self {
        let s = voxel_size;
        let inv_s = 1.0 / s;
        Self {
            mat: [
                [s, 0.0, 0.0, 0.0],
                [0.0, s, 0.0, 0.0],
                [0.0, 0.0, s, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            inv: [
                [inv_s, 0.0, 0.0, 0.0],
                [0.0, inv_s, 0.0, 0.0],
                [0.0, 0.0, inv_s, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            voxel_size: s,
        }
    }

    /// Convert from the legacy `VoxelTransform`.
    pub fn from_voxel_transform(vt: &VoxelTransform) -> Self {
        let s = vt.voxel_size;
        let inv_s = 1.0 / s;
        let t = vt.origin;
        Self {
            mat: [
                [s, 0.0, 0.0, t[0]],
                [0.0, s, 0.0, t[1]],
                [0.0, 0.0, s, t[2]],
                [0.0, 0.0, 0.0, 1.0],
            ],
            inv: [
                [inv_s, 0.0, 0.0, -t[0] * inv_s],
                [0.0, inv_s, 0.0, -t[1] * inv_s],
                [0.0, 0.0, inv_s, -t[2] * inv_s],
                [0.0, 0.0, 0.0, 1.0],
            ],
            voxel_size: s,
        }
    }

    /// General constructor from non-uniform scale, 3x3 rotation, and translation.
    ///
    /// The forward matrix is:  M = T * R * S
    /// where S = diag(scale), R = rotation 3x3, T = translate.
    pub fn from_scale_rotate_translate(
        scale: [f64; 3],
        rotation: [[f64; 3]; 3],
        translate: [f64; 3],
    ) -> Self {
        // Forward = T * R * S
        // Linear part L = R * S  (column j = rotation_col_j * scale_j)
        let mut mat = [[0.0f64; 4]; 4];
        for row in 0..3 {
            for col in 0..3 {
                mat[row][col] = rotation[row][col] * scale[col];
            }
            mat[row][3] = translate[row];
        }
        mat[3][3] = 1.0;

        let inv = invert_4x4(&mat);

        // Voxel size = length of first column of the linear part
        let col0_len = (mat[0][0] * mat[0][0]
            + mat[1][0] * mat[1][0]
            + mat[2][0] * mat[2][0])
            .sqrt();

        Self {
            mat,
            inv,
            voxel_size: col0_len,
        }
    }

    /// Reconstruct from pre-computed forward and inverse matrices plus voxel size.
    /// Used by NanoGrid deserialization.
    pub fn from_matrices(mat: [[f64; 4]; 4], inv: [[f64; 4]; 4], voxel_size: f64) -> Self {
        Self {
            mat,
            inv,
            voxel_size,
        }
    }

    /// Transform an index-space coordinate to world-space.
    #[inline]
    pub fn index_to_world(&self, coord: Coord) -> [f64; 3] {
        let x = coord.x as f64;
        let y = coord.y as f64;
        let z = coord.z as f64;
        [
            self.mat[0][0] * x + self.mat[0][1] * y + self.mat[0][2] * z + self.mat[0][3],
            self.mat[1][0] * x + self.mat[1][1] * y + self.mat[1][2] * z + self.mat[1][3],
            self.mat[2][0] * x + self.mat[2][1] * y + self.mat[2][2] * z + self.mat[2][3],
        ]
    }

    /// Transform a world-space position to the nearest index-space coordinate (rounded).
    #[inline]
    pub fn world_to_index(&self, world: [f64; 3]) -> Coord {
        let f = self.world_to_index_f64(world);
        Coord::new(f[0].round() as i32, f[1].round() as i32, f[2].round() as i32)
    }

    /// Transform a world-space position to continuous index-space (no rounding).
    #[inline]
    pub fn world_to_index_f64(&self, world: [f64; 3]) -> [f64; 3] {
        let x = world[0];
        let y = world[1];
        let z = world[2];
        [
            self.inv[0][0] * x + self.inv[0][1] * y + self.inv[0][2] * z + self.inv[0][3],
            self.inv[1][0] * x + self.inv[1][1] * y + self.inv[1][2] * z + self.inv[1][3],
            self.inv[2][0] * x + self.inv[2][1] * y + self.inv[2][2] * z + self.inv[2][3],
        ]
    }

    /// Cached voxel size (length of first column of the linear part).
    #[inline]
    pub fn voxel_size(&self) -> f64 {
        self.voxel_size
    }

    /// Reference to the inverse matrix (world -> index).
    #[inline]
    pub fn inverse(&self) -> &[[f64; 4]; 4] {
        &self.inv
    }

    /// Reference to the forward matrix (index -> world).
    #[inline]
    pub fn forward(&self) -> &[[f64; 4]; 4] {
        &self.mat
    }
}

/// Compute the inverse of a 4x4 matrix using cofactor expansion.
///
/// Panics (via division) if the matrix is singular (determinant == 0).
fn invert_4x4(m: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    // Precompute 2x2 determinants from rows 2-3
    let s0 = m[0][0] * m[1][1] - m[1][0] * m[0][1];
    let s1 = m[0][0] * m[1][2] - m[1][0] * m[0][2];
    let s2 = m[0][0] * m[1][3] - m[1][0] * m[0][3];
    let s3 = m[0][1] * m[1][2] - m[1][1] * m[0][2];
    let s4 = m[0][1] * m[1][3] - m[1][1] * m[0][3];
    let s5 = m[0][2] * m[1][3] - m[1][2] * m[0][3];

    let c5 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
    let c4 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
    let c3 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
    let c2 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
    let c1 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
    let c0 = m[2][0] * m[3][1] - m[3][0] * m[2][1];

    let det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
    assert!(det.abs() > 1e-30, "Singular matrix in invert_4x4");
    let inv_det = 1.0 / det;

    let mut out = [[0.0f64; 4]; 4];

    out[0][0] = ( m[1][1] * c5 - m[1][2] * c4 + m[1][3] * c3) * inv_det;
    out[0][1] = (-m[0][1] * c5 + m[0][2] * c4 - m[0][3] * c3) * inv_det;
    out[0][2] = ( m[3][1] * s5 - m[3][2] * s4 + m[3][3] * s3) * inv_det;
    out[0][3] = (-m[2][1] * s5 + m[2][2] * s4 - m[2][3] * s3) * inv_det;

    out[1][0] = (-m[1][0] * c5 + m[1][2] * c2 - m[1][3] * c1) * inv_det;
    out[1][1] = ( m[0][0] * c5 - m[0][2] * c2 + m[0][3] * c1) * inv_det;
    out[1][2] = (-m[3][0] * s5 + m[3][2] * s2 - m[3][3] * s1) * inv_det;
    out[1][3] = ( m[2][0] * s5 - m[2][2] * s2 + m[2][3] * s1) * inv_det;

    out[2][0] = ( m[1][0] * c4 - m[1][1] * c2 + m[1][3] * c0) * inv_det;
    out[2][1] = (-m[0][0] * c4 + m[0][1] * c2 - m[0][3] * c0) * inv_det;
    out[2][2] = ( m[3][0] * s4 - m[3][1] * s2 + m[3][3] * s0) * inv_det;
    out[2][3] = (-m[2][0] * s4 + m[2][1] * s2 - m[2][3] * s0) * inv_det;

    out[3][0] = (-m[1][0] * c3 + m[1][1] * c1 - m[1][2] * c0) * inv_det;
    out[3][1] = ( m[0][0] * c3 - m[0][1] * c1 + m[0][2] * c0) * inv_det;
    out[3][2] = (-m[3][0] * s3 + m[3][1] * s1 - m[3][2] * s0) * inv_det;
    out[3][3] = ( m[2][0] * s3 - m[2][1] * s1 + m[2][2] * s0) * inv_det;

    out
}

/// Top-level sparse volume: tree + transform + metadata.
///
/// Mirrors `openvdb::Grid<Tree<T>>`.
pub struct Grid<T: Copy + Default> {
    /// The sparse VDB tree holding voxel data.
    tree: Tree<T>,
    /// Full 4x4 affine transform (index <-> world).
    affine: AffineMap,
    /// Legacy transform accessor (kept for backward compat).
    transform: VoxelTransform,
    /// Human-readable name (e.g. "density", "sdf").
    name: String,
    /// Semantic role classification.
    grid_class: GridClass,
    /// User-defined metadata dictionary.
    metadata: HashMap<String, MetaValue>,
}

impl<T: Copy + Default> Grid<T> {
    /// Create a grid with the given background value and voxel size.
    pub fn new(background: T, voxel_size: f64) -> Self {
        Self {
            tree: Tree::new(background),
            affine: AffineMap::from_uniform_scale(voxel_size),
            transform: VoxelTransform::uniform(voxel_size),
            name: String::new(),
            grid_class: GridClass::Unknown,
            metadata: HashMap::new(),
        }
    }

    /// Create a grid with a full affine transform.
    pub fn with_affine(background: T, map: AffineMap) -> Self {
        let vs = map.voxel_size();
        Self {
            tree: Tree::new(background),
            affine: map,
            transform: VoxelTransform::uniform(vs),
            name: String::new(),
            grid_class: GridClass::Unknown,
            metadata: HashMap::new(),
        }
    }

    /// Create a level-set grid with half-width in voxels.
    pub fn level_set(voxel_size: f64, half_width: f64) -> Grid<f32>
    where
        T: From<f32>,
    {
        let bg = (half_width * voxel_size) as f32;
        Grid::<f32> {
            tree: Tree::new(bg),
            affine: AffineMap::from_uniform_scale(voxel_size),
            transform: VoxelTransform::uniform(voxel_size),
            name: "sdf".to_string(),
            grid_class: GridClass::LevelSet,
            metadata: HashMap::new(),
        }
    }

    /// Set the grid name.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    /// Grid name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Set the grid class.
    pub fn set_grid_class(&mut self, class: GridClass) {
        self.grid_class = class;
    }

    /// Grid class.
    pub fn grid_class(&self) -> GridClass {
        self.grid_class
    }

    /// Reference to the legacy transform.
    pub fn transform(&self) -> &VoxelTransform {
        &self.transform
    }

    /// Set a new legacy transform (also updates the internal AffineMap).
    pub fn set_transform(&mut self, transform: VoxelTransform) {
        self.affine = AffineMap::from_voxel_transform(&transform);
        self.transform = transform;
    }

    /// Reference to the AffineMap.
    pub fn affine_map(&self) -> &AffineMap {
        &self.affine
    }

    /// Set a new AffineMap.
    pub fn set_affine_map(&mut self, map: AffineMap) {
        self.transform = VoxelTransform::uniform(map.voxel_size());
        self.affine = map;
    }

    /// Insert or update a metadata entry.
    ///
    /// ```rust,ignore
    /// grid.set_meta("author", "Alice");
    /// grid.set_meta("voxel_count", 1_000_000i64);
    /// ```
    pub fn set_meta(&mut self, key: impl Into<String>, value: impl Into<MetaValue>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Retrieve a metadata value by key, or `None` if absent.
    pub fn get_meta(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    /// Remove a metadata entry, returning its value if it existed.
    pub fn remove_meta(&mut self, key: &str) -> Option<MetaValue> {
        self.metadata.remove(key)
    }

    /// Immutable view of the full metadata dictionary.
    pub fn metadata(&self) -> &HashMap<String, MetaValue> {
        &self.metadata
    }

    /// Reference to the underlying tree.
    pub fn tree(&self) -> &Tree<T> {
        &self.tree
    }

    /// Mutable reference to the tree.
    pub fn tree_mut(&mut self) -> &mut Tree<T> {
        &mut self.tree
    }

    /// Get value at index-space coordinate.
    pub fn get(&self, coord: Coord) -> T {
        self.tree.get(coord)
    }

    /// Set value at index-space coordinate.
    pub fn set(&mut self, coord: Coord, value: T) {
        self.tree.set(coord, value);
    }

    /// Get value at world-space position (snaps to nearest voxel).
    pub fn get_world(&self, pos: [f64; 3]) -> T {
        let coord = self.affine.world_to_index(pos);
        self.tree.get(coord)
    }

    /// Set value at world-space position (snaps to nearest voxel).
    pub fn set_world(&mut self, pos: [f64; 3], value: T) {
        let coord = self.affine.world_to_index(pos);
        self.tree.set(coord, value);
    }

    /// Number of allocated leaf tiles.
    pub fn leaf_count(&self) -> usize {
        self.tree.leaf_count()
    }

    /// Total active voxel count.
    pub fn active_voxel_count(&self) -> u64 {
        self.tree.active_voxel_count()
    }
}

// ---------------------------------------------------------------------------
// Task 3: Grid statistics
// ---------------------------------------------------------------------------

impl<T: Copy + Default + PartialOrd> Grid<T> {
    /// Scan all active voxels and return (min, max) values.
    ///
    /// Returns `None` if there are no active voxels.
    pub fn eval_min_max(&self) -> Option<(T, T)> {
        let mut iter = self.tree.iter_active();
        let (_, first) = iter.next()?;
        let mut min = first;
        let mut max = first;
        for (_, val) in iter {
            if val < min {
                min = val;
            }
            if val > max {
                max = val;
            }
        }
        Some((min, max))
    }

    /// Bounding box of all active voxels in index space.
    ///
    /// Returns `None` if there are no active voxels.
    pub fn active_bbox(&self) -> Option<CoordBBox> {
        let mut bbox = CoordBBox::empty();
        let mut any = false;
        for (coord, _) in self.tree.iter_active() {
            bbox.expand(coord);
            any = true;
        }
        if any {
            Some(bbox)
        } else {
            None
        }
    }
}

impl<T: Copy + Default> Grid<T> {
    /// Estimated memory usage in bytes.
    ///
    /// Accounts for leaf value arrays and bitmasks. Does not include
    /// allocator overhead or padding.
    pub fn mem_bytes(&self) -> usize {
        use crate::tree::leaf::LEAF_SIZE;
        let leaf_count = self.leaf_count();
        let val_size = std::mem::size_of::<T>();
        // Each leaf: LEAF_SIZE * sizeof(T) values + 8 * u64 bitmask + origin Coord
        let leaf_bytes = leaf_count * (LEAF_SIZE * val_size + 8 * 8 + 12);
        // Grid struct overhead
        let grid_overhead = std::mem::size_of::<Self>();
        grid_overhead + leaf_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_new_empty() {
        let g = Grid::<f32>::new(0.0, 0.5);
        assert_eq!(g.leaf_count(), 0);
        assert_eq!(g.active_voxel_count(), 0);
        assert_eq!(g.grid_class(), GridClass::Unknown);
    }

    #[test]
    fn grid_set_get_index_space() {
        let mut g = Grid::<f32>::new(-1.0, 1.0);
        g.set(Coord::new(5, 5, 5), 42.0);
        assert_eq!(g.get(Coord::new(5, 5, 5)), 42.0);
        assert_eq!(g.get(Coord::new(0, 0, 0)), -1.0); // background
    }

    #[test]
    fn grid_world_space_roundtrip() {
        let mut g = Grid::<f32>::new(0.0, 0.25);
        g.set_world([1.0, 2.0, 3.0], 7.0);
        // [1.0, 2.0, 3.0] at voxel_size=0.25 -> index (4, 8, 12)
        assert_eq!(g.get(Coord::new(4, 8, 12)), 7.0);
        assert_eq!(g.get_world([1.0, 2.0, 3.0]), 7.0);
    }

    #[test]
    fn grid_transform_with_origin() {
        let t = VoxelTransform::with_origin(0.5, [10.0, 20.0, 30.0]);
        let world = t.index_to_world(Coord::new(2, 4, 6));
        assert_eq!(world, [11.0, 22.0, 33.0]);
        let back = t.world_to_index(world);
        assert_eq!(back, Coord::new(2, 4, 6));
    }

    #[test]
    fn grid_level_set_constructor() {
        let g = Grid::<f32>::level_set(0.1, 3.0);
        assert_eq!(g.grid_class(), GridClass::LevelSet);
        assert_eq!(g.name(), "sdf");
        // background = half_width * voxel_size = 3.0 * 0.1 = 0.3
        assert!((g.tree().background() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn grid_metadata() {
        let mut g = Grid::<f32>::new(0.0, 1.0);
        g.set_name("density");
        g.set_grid_class(GridClass::FogVolume);
        assert_eq!(g.name(), "density");
        assert_eq!(g.grid_class(), GridClass::FogVolume);
    }

    // -----------------------------------------------------------------------
    // AffineMap tests
    // -----------------------------------------------------------------------

    const EPS: f64 = 1e-10;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn affine_identity_roundtrip() {
        let m = AffineMap::identity();
        let c = Coord::new(3, -7, 11);
        let w = m.index_to_world(c);
        assert!(approx(w[0], 3.0));
        assert!(approx(w[1], -7.0));
        assert!(approx(w[2], 11.0));
        let back = m.world_to_index(w);
        assert_eq!(back, c);
    }

    #[test]
    fn affine_uniform_scale() {
        let m = AffineMap::from_uniform_scale(0.25);
        assert!(approx(m.voxel_size(), 0.25));
        let w = m.index_to_world(Coord::new(4, 8, 12));
        assert!(approx(w[0], 1.0));
        assert!(approx(w[1], 2.0));
        assert!(approx(w[2], 3.0));
        let back = m.world_to_index(w);
        assert_eq!(back, Coord::new(4, 8, 12));
    }

    #[test]
    fn affine_from_voxel_transform() {
        let vt = VoxelTransform::with_origin(0.5, [10.0, 20.0, 30.0]);
        let m = AffineMap::from_voxel_transform(&vt);
        let w = m.index_to_world(Coord::new(2, 4, 6));
        assert!(approx(w[0], 11.0));
        assert!(approx(w[1], 22.0));
        assert!(approx(w[2], 33.0));
        let back = m.world_to_index(w);
        assert_eq!(back, Coord::new(2, 4, 6));
    }

    #[test]
    fn affine_world_to_index_f64_no_rounding() {
        let m = AffineMap::from_uniform_scale(2.0);
        let f = m.world_to_index_f64([3.0, 5.0, 7.0]);
        assert!(approx(f[0], 1.5));
        assert!(approx(f[1], 2.5));
        assert!(approx(f[2], 3.5));
    }

    #[test]
    fn affine_scale_rotate_translate() {
        // 90-degree rotation around Z axis: x->y, y->-x
        let rot = [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let scale = [2.0, 2.0, 2.0];
        let translate = [10.0, 20.0, 30.0];
        let m = AffineMap::from_scale_rotate_translate(scale, rot, translate);

        // Index (1, 0, 0) -> scale -> (2, 0, 0) -> rotate -> (0, 2, 0) -> translate -> (10, 22, 30)
        let w = m.index_to_world(Coord::new(1, 0, 0));
        assert!(approx(w[0], 10.0));
        assert!(approx(w[1], 22.0));
        assert!(approx(w[2], 30.0));

        // Round-trip
        let back = m.world_to_index(w);
        assert_eq!(back, Coord::new(1, 0, 0));
    }

    #[test]
    fn affine_non_uniform_scale() {
        let scale = [1.0, 2.0, 3.0];
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let translate = [0.0, 0.0, 0.0];
        let m = AffineMap::from_scale_rotate_translate(scale, rot, translate);
        let w = m.index_to_world(Coord::new(1, 1, 1));
        assert!(approx(w[0], 1.0));
        assert!(approx(w[1], 2.0));
        assert!(approx(w[2], 3.0));
        let back = m.world_to_index(w);
        assert_eq!(back, Coord::new(1, 1, 1));
    }

    #[test]
    fn affine_inverse_is_correct() {
        let m = AffineMap::from_uniform_scale(0.5);
        let inv = m.inverse();
        // inv * mat should be ~identity
        let fwd = m.forward();
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += inv[i][k] * fwd[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx(sum, expected),
                    "inv*fwd[{}][{}] = {} expected {}",
                    i,
                    j,
                    sum,
                    expected
                );
            }
        }
    }

    #[test]
    fn affine_general_inverse_roundtrip() {
        let rot = [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let m = AffineMap::from_scale_rotate_translate(
            [0.5, 0.5, 0.5],
            rot,
            [100.0, -50.0, 25.0],
        );
        let inv = m.inverse();
        let fwd = m.forward();
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += inv[i][k] * fwd[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-9,
                    "inv*fwd[{}][{}] = {} expected {}",
                    i, j, sum, expected
                );
            }
        }
    }

    #[test]
    fn grid_with_affine_constructor() {
        let map = AffineMap::from_uniform_scale(0.25);
        let mut g = Grid::<f32>::with_affine(0.0, map);
        g.set_world([1.0, 2.0, 3.0], 7.0);
        assert_eq!(g.get(Coord::new(4, 8, 12)), 7.0);
        assert_eq!(g.get_world([1.0, 2.0, 3.0]), 7.0);
    }

    #[test]
    fn grid_with_affine_rotated() {
        // 90-degree rotation around Z: index (1,0,0) -> world (0,1,0)
        let rot = [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let map = AffineMap::from_scale_rotate_translate(
            [1.0, 1.0, 1.0],
            rot,
            [0.0, 0.0, 0.0],
        );
        let mut g = Grid::<f32>::with_affine(0.0, map);
        g.set(Coord::new(1, 0, 0), 99.0);

        // World (0, 1, 0) should map back to index (1, 0, 0)
        assert_eq!(g.get_world([0.0, 1.0, 0.0]), 99.0);
    }

    // -----------------------------------------------------------------------
    // Grid statistics tests (Task 3)
    // -----------------------------------------------------------------------

    #[test]
    fn eval_min_max_empty_grid() {
        let g = Grid::<f32>::new(0.0, 1.0);
        assert!(g.eval_min_max().is_none());
    }

    #[test]
    fn eval_min_max_single_voxel() {
        let mut g = Grid::<f32>::new(0.0, 1.0);
        g.set(Coord::new(0, 0, 0), 42.0);
        let (min, max) = g.eval_min_max().unwrap();
        assert_eq!(min, 42.0);
        assert_eq!(max, 42.0);
    }

    #[test]
    fn eval_min_max_multiple_voxels() {
        let mut g = Grid::<f32>::new(0.0, 1.0);
        g.set(Coord::new(0, 0, 0), -5.0);
        g.set(Coord::new(1, 1, 1), 10.0);
        g.set(Coord::new(2, 2, 2), 3.0);
        let (min, max) = g.eval_min_max().unwrap();
        assert_eq!(min, -5.0);
        assert_eq!(max, 10.0);
    }

    #[test]
    fn active_bbox_empty_grid() {
        let g = Grid::<f32>::new(0.0, 1.0);
        assert!(g.active_bbox().is_none());
    }

    #[test]
    fn active_bbox_single_voxel() {
        let mut g = Grid::<f32>::new(0.0, 1.0);
        g.set(Coord::new(5, 5, 5), 1.0);
        let bbox = g.active_bbox().unwrap();
        assert_eq!(bbox.min, Coord::new(5, 5, 5));
        assert_eq!(bbox.max, Coord::new(5, 5, 5));
    }

    #[test]
    fn active_bbox_multiple_voxels() {
        let mut g = Grid::<f32>::new(0.0, 1.0);
        g.set(Coord::new(1, 2, 3), 1.0);
        g.set(Coord::new(10, 20, 30), 2.0);
        let bbox = g.active_bbox().unwrap();
        assert_eq!(bbox.min, Coord::new(1, 2, 3));
        assert_eq!(bbox.max, Coord::new(10, 20, 30));
    }

    #[test]
    fn mem_bytes_empty_grid() {
        let g = Grid::<f32>::new(0.0, 1.0);
        let bytes = g.mem_bytes();
        // Should be at least the Grid struct size
        assert!(bytes >= std::mem::size_of::<Grid<f32>>());
    }

    #[test]
    fn mem_bytes_with_leaves() {
        let mut g = Grid::<f32>::new(0.0, 1.0);
        g.set(Coord::new(0, 0, 0), 1.0);
        g.set(Coord::new(8, 0, 0), 2.0);
        let bytes = g.mem_bytes();
        // 2 leaves, each ~2KB for f32 values + bitmask
        assert!(bytes > 4000);
    }

    #[test]
    fn invert_4x4_identity() {
        let inv = invert_4x4(&IDENTITY_4X4);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx(inv[i][j], expected));
            }
        }
    }

    #[test]
    fn invert_4x4_scale_translate() {
        let m = [
            [2.0, 0.0, 0.0, 5.0],
            [0.0, 3.0, 0.0, 7.0],
            [0.0, 0.0, 4.0, 9.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let inv = invert_4x4(&m);
        // Check inverse * forward = identity
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += inv[i][k] * m[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx(sum, expected),
                    "[{}][{}] = {} expected {}",
                    i, j, sum, expected
                );
            }
        }
    }
}

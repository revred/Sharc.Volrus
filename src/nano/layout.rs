//! NanoGrid linearized format — header, leaf, and serialization.

use crate::grid::{AffineMap, Grid, GridClass};
use crate::math::Coord;

/// Magic bytes identifying a NanoGrid buffer.
const NANO_MAGIC: [u8; 4] = *b"NANO";

/// Current format version.
const NANO_VERSION: u32 = 1;

/// Header at the start of the NanoGrid buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NanoHeader {
    /// Magic bytes: b"NANO"
    pub magic: [u8; 4],
    /// Format version (currently 1).
    pub version: u32,
    /// Grid class: 0=Unknown, 1=LevelSet, 2=FogVolume.
    pub grid_class: u32,
    /// Voxel size (uniform scale from the affine map).
    pub voxel_size: f64,
    /// Background value for unset voxels.
    pub background: f32,
    /// Number of leaf nodes in the buffer.
    pub leaf_count: u32,
    /// Active bounding box minimum (index space).
    pub bbox_min: [i32; 3],
    /// Active bounding box maximum (index space).
    pub bbox_max: [i32; 3],
    /// Forward affine transform (index -> world), row-major 4x4.
    pub affine_mat: [[f64; 4]; 4],
    /// Inverse affine transform (world -> index), row-major 4x4.
    pub affine_inv: [[f64; 4]; 4],
    /// Padding to align total header size to 8 bytes.
    pub _pad: [u8; 4],
}

/// Linearized leaf node in the NanoGrid buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NanoLeaf {
    /// Origin of this leaf in index space (tile-aligned).
    pub origin: [i32; 3],
    /// Padding for alignment.
    pub _pad: i32,
    /// 512-bit activity mask (8 x u64).
    pub active_mask: [u64; 8],
    /// Dense voxel values (512 x f32).
    pub values: [f32; 512],
}

/// A linearized, read-only grid stored in a single contiguous buffer.
/// Designed for GPU upload, memory mapping, and zero-copy sharing.
///
/// Memory layout (all little-endian):
/// `[NanoHeader][NanoLeaf 0][NanoLeaf 1]...[NanoLeaf N-1]`
#[derive(Debug)]
pub struct NanoGrid {
    data: Vec<u8>,
}

fn grid_class_to_u32(gc: GridClass) -> u32 {
    match gc {
        GridClass::Unknown => 0,
        GridClass::LevelSet => 1,
        GridClass::FogVolume => 2,
    }
}

fn u32_to_grid_class(v: u32) -> GridClass {
    match v {
        1 => GridClass::LevelSet,
        2 => GridClass::FogVolume,
        _ => GridClass::Unknown,
    }
}

/// Compare leaf origins lexicographically (x, then y, then z) for sorting.
pub(crate) fn cmp_origin(a: &[i32; 3], b: &[i32; 3]) -> std::cmp::Ordering {
    a[0].cmp(&b[0])
        .then(a[1].cmp(&b[1]))
        .then(a[2].cmp(&b[2]))
}

impl NanoGrid {
    /// Linearize a `Grid<f32>` into a NanoGrid buffer.
    pub fn from_grid(grid: &Grid<f32>) -> NanoGrid {
        // Collect all leaves, sorted by origin for binary search.
        let mut nano_leaves: Vec<NanoLeaf> = Vec::new();

        for leaf in grid.tree().leaves() {
            let o = leaf.origin();
            let mut nl = NanoLeaf {
                origin: [o.x, o.y, o.z],
                _pad: 0,
                active_mask: *leaf.active_mask(),
                values: [0.0f32; 512],
            };
            nl.values.copy_from_slice(leaf.values());
            nano_leaves.push(nl);
        }

        // Sort leaves by origin (lexicographic) for binary search.
        nano_leaves.sort_by(|a, b| cmp_origin(&a.origin, &b.origin));

        // Compute active bounding box.
        let mut bbox_min = [i32::MAX; 3];
        let mut bbox_max = [i32::MIN; 3];
        for (coord, _) in grid.tree().iter_active() {
            bbox_min[0] = bbox_min[0].min(coord.x);
            bbox_min[1] = bbox_min[1].min(coord.y);
            bbox_min[2] = bbox_min[2].min(coord.z);
            bbox_max[0] = bbox_max[0].max(coord.x);
            bbox_max[1] = bbox_max[1].max(coord.y);
            bbox_max[2] = bbox_max[2].max(coord.z);
        }

        // If no active voxels, use zeros for bbox.
        if bbox_min[0] == i32::MAX {
            bbox_min = [0; 3];
            bbox_max = [0; 3];
        }

        let affine = grid.affine_map();

        let header = NanoHeader {
            magic: NANO_MAGIC,
            version: NANO_VERSION,
            grid_class: grid_class_to_u32(grid.grid_class()),
            voxel_size: affine.voxel_size(),
            background: grid.tree().background(),
            leaf_count: nano_leaves.len() as u32,
            bbox_min,
            bbox_max,
            affine_mat: *affine.forward(),
            affine_inv: *affine.inverse(),
            _pad: [0; 4],
        };

        let header_size = std::mem::size_of::<NanoHeader>();
        let leaf_size = std::mem::size_of::<NanoLeaf>();
        let total = header_size + leaf_size * nano_leaves.len();

        let mut data = vec![0u8; total];

        // Write header.
        unsafe {
            std::ptr::copy_nonoverlapping(
                &header as *const NanoHeader as *const u8,
                data.as_mut_ptr(),
                header_size,
            );
        }

        // Write leaves.
        for (i, nl) in nano_leaves.iter().enumerate() {
            let offset = header_size + i * leaf_size;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    nl as *const NanoLeaf as *const u8,
                    data.as_mut_ptr().add(offset),
                    leaf_size,
                );
            }
        }

        NanoGrid { data }
    }

    /// Read the header from the buffer (zero-copy pointer cast).
    pub fn header(&self) -> &NanoHeader {
        assert!(self.data.len() >= std::mem::size_of::<NanoHeader>());
        unsafe { &*(self.data.as_ptr() as *const NanoHeader) }
    }

    /// Access a leaf by index (zero-copy pointer cast).
    ///
    /// # Panics
    /// Panics if `index >= leaf_count()`.
    pub fn leaf(&self, index: usize) -> &NanoLeaf {
        assert!(index < self.leaf_count(), "leaf index out of bounds");
        let header_size = std::mem::size_of::<NanoHeader>();
        let leaf_size = std::mem::size_of::<NanoLeaf>();
        let offset = header_size + index * leaf_size;
        assert!(offset + leaf_size <= self.data.len());
        unsafe { &*(self.data.as_ptr().add(offset) as *const NanoLeaf) }
    }

    /// Number of leaf nodes in the buffer.
    pub fn leaf_count(&self) -> usize {
        self.header().leaf_count as usize
    }

    /// Raw buffer for GPU upload or memory mapping.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Validate and wrap a raw byte buffer as a NanoGrid.
    pub fn from_bytes(data: Vec<u8>) -> Result<NanoGrid, String> {
        let header_size = std::mem::size_of::<NanoHeader>();
        if data.len() < header_size {
            return Err("Buffer too small for NanoHeader".into());
        }

        let header = unsafe { &*(data.as_ptr() as *const NanoHeader) };

        if header.magic != NANO_MAGIC {
            return Err(format!(
                "Invalid magic: expected {:?}, got {:?}",
                NANO_MAGIC, header.magic
            ));
        }

        if header.version != NANO_VERSION {
            return Err(format!(
                "Unsupported version: expected {}, got {}",
                NANO_VERSION, header.version
            ));
        }

        let leaf_size = std::mem::size_of::<NanoLeaf>();
        let expected = header_size + leaf_size * header.leaf_count as usize;
        if data.len() < expected {
            return Err(format!(
                "Buffer too small: expected {} bytes, got {}",
                expected,
                data.len()
            ));
        }

        // Verify leaves are sorted by origin.
        let nano = NanoGrid { data };
        for i in 1..nano.leaf_count() {
            let prev = nano.leaf(i - 1).origin;
            let curr = nano.leaf(i).origin;
            if cmp_origin(&prev, &curr) != std::cmp::Ordering::Less {
                return Err(format!(
                    "Leaves not sorted: leaf {} origin {:?} >= leaf {} origin {:?}",
                    i - 1,
                    prev,
                    i,
                    curr
                ));
            }
        }

        Ok(nano)
    }

    /// Convert back to a mutable `Grid<f32>`.
    pub fn to_grid(&self) -> Grid<f32> {
        let hdr = self.header();

        // Reconstruct affine map.
        let affine = AffineMap::from_matrices(hdr.affine_mat, hdr.affine_inv, hdr.voxel_size);
        let gc = u32_to_grid_class(hdr.grid_class);

        let mut grid = Grid::<f32>::with_affine(hdr.background, affine);
        grid.set_grid_class(gc);

        // Insert all leaf voxels.
        for i in 0..self.leaf_count() {
            let nl = self.leaf(i);
            let origin = Coord::new(nl.origin[0], nl.origin[1], nl.origin[2]);

            for word_idx in 0..8usize {
                let mut word = nl.active_mask[word_idx];
                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    word &= word - 1;
                    let off = word_idx * 64 + bit;
                    let x = (off >> 6) as i32;
                    let y = ((off >> 3) & 0x7) as i32;
                    let z = (off & 0x7) as i32;
                    let coord = Coord::new(origin.x + x, origin.y + y, origin.z + z);
                    grid.set(coord, nl.values[off]);
                }
            }
        }

        grid
    }
}

//! `.vol` binary format reader/writer for `Grid<f32>`.
//!
//! ## Format (little-endian)
//!
//! ```text
//! Header (32 bytes):
//!   magic:      [u8; 4]  = b"VOLR"
//!   version:    u32      = 1
//!   grid_class: u32      — 0=Unknown, 1=LevelSet, 2=FogVolume
//!   voxel_size: f64
//!   background: f32
//!   leaf_count: u32
//!   _reserved:  [u8; 4]
//!
//! Per leaf (variable):
//!   origin_x:    i32
//!   origin_y:    i32
//!   origin_z:    i32
//!   active_mask: [u64; 8]  — 512 bits
//!   values:      [f32; 512] — dense voxel data
//! ```

use crate::grid::{Grid, GridClass};
use crate::math::Coord;
use crate::tree::leaf::LEAF_SIZE;
use std::io::{self, Read, Write};

const MAGIC: [u8; 4] = *b"VOLR";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 32;

fn grid_class_to_u32(gc: GridClass) -> u32 {
    match gc {
        GridClass::Unknown => 0,
        GridClass::LevelSet => 1,
        GridClass::FogVolume => 2,
    }
}

fn u32_to_grid_class(v: u32) -> io::Result<GridClass> {
    match v {
        0 => Ok(GridClass::Unknown),
        1 => Ok(GridClass::LevelSet),
        2 => Ok(GridClass::FogVolume),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown grid class: {v}"),
        )),
    }
}

/// Write a `Grid<f32>` to a writer in `.vol` format.
pub fn write_vol<W: Write>(grid: &Grid<f32>, writer: &mut W) -> io::Result<()> {
    // --- header ---
    writer.write_all(&MAGIC)?;
    writer.write_all(&VERSION.to_le_bytes())?;
    writer.write_all(&grid_class_to_u32(grid.grid_class()).to_le_bytes())?;
    writer.write_all(&grid.transform().voxel_size.to_le_bytes())?;
    writer.write_all(&grid.tree().background().to_le_bytes())?;
    let leaf_count = grid.leaf_count() as u32;
    writer.write_all(&leaf_count.to_le_bytes())?;
    writer.write_all(&[0u8; 4])?; // reserved

    // --- per-leaf data ---
    for leaf in grid.tree().leaves() {
        let origin = leaf.origin();
        writer.write_all(&origin.x.to_le_bytes())?;
        writer.write_all(&origin.y.to_le_bytes())?;
        writer.write_all(&origin.z.to_le_bytes())?;

        for &word in leaf.active_mask() {
            writer.write_all(&word.to_le_bytes())?;
        }

        for &val in leaf.values().iter() {
            writer.write_all(&val.to_le_bytes())?;
        }
    }

    Ok(())
}

/// Read a `Grid<f32>` from a reader in `.vol` format.
pub fn read_vol<R: Read>(reader: &mut R) -> io::Result<Grid<f32>> {
    // --- header ---
    let mut header = [0u8; HEADER_SIZE];
    reader.read_exact(&mut header)?;

    if &header[0..4] != &MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a .vol file (bad magic)",
        ));
    }

    let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported .vol version: {version}"),
        ));
    }

    let grid_class = u32_to_grid_class(u32::from_le_bytes(header[8..12].try_into().unwrap()))?;
    let voxel_size = f64::from_le_bytes(header[12..20].try_into().unwrap());
    let background = f32::from_le_bytes(header[20..24].try_into().unwrap());
    let leaf_count = u32::from_le_bytes(header[24..28].try_into().unwrap());
    // bytes 28..32 reserved

    let mut grid = Grid::<f32>::new(background, voxel_size);
    grid.set_grid_class(grid_class);

    // --- per-leaf data ---
    for _ in 0..leaf_count {
        let mut origin_buf = [0u8; 12];
        reader.read_exact(&mut origin_buf)?;
        let ox = i32::from_le_bytes(origin_buf[0..4].try_into().unwrap());
        let oy = i32::from_le_bytes(origin_buf[4..8].try_into().unwrap());
        let oz = i32::from_le_bytes(origin_buf[8..12].try_into().unwrap());
        let origin = Coord::new(ox, oy, oz);

        let mut mask_buf = [0u8; 64]; // 8 * 8 bytes
        reader.read_exact(&mut mask_buf)?;
        let mut active_mask = [0u64; 8];
        for i in 0..8 {
            active_mask[i] = u64::from_le_bytes(mask_buf[i * 8..(i + 1) * 8].try_into().unwrap());
        }

        let mut val_buf = [0u8; LEAF_SIZE * 4]; // 512 * 4 bytes
        reader.read_exact(&mut val_buf)?;
        let mut values = [0.0f32; LEAF_SIZE];
        for i in 0..LEAF_SIZE {
            values[i] = f32::from_le_bytes(val_buf[i * 4..(i + 1) * 4].try_into().unwrap());
        }

        // We need to reconstruct the leaf. Use the grid's set() to allocate
        // tree nodes, then bulk-write the leaf data.
        // First, touch the origin voxel to ensure the leaf is allocated.
        grid.set(origin, values[0]);

        // Now get the leaf and write the bulk data.
        let leaf = grid
            .tree_mut()
            .leaf_mut(origin)
            .expect("leaf should exist after set()");
        *leaf.values_mut() = values;
        *leaf.active_mask_mut() = active_mask;
    }

    Ok(grid)
}

/// Write a `Grid<f32>` to a file in `.vol` format.
pub fn save_vol(grid: &Grid<f32>, path: &str) -> io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write_vol(grid, &mut file)
}

/// Read a `Grid<f32>` from a `.vol` file.
pub fn load_vol(path: &str) -> io::Result<Grid<f32>> {
    let mut file = std::fs::File::open(path)?;
    read_vol(&mut file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::make_level_set_sphere;
    use std::io::Cursor;

    #[test]
    fn roundtrip_sphere() {
        let grid = make_level_set_sphere(5.0, [0.0; 3], 0.5, 3.0);
        let mut buf = Vec::new();
        write_vol(&grid, &mut buf).unwrap();

        let loaded = read_vol(&mut Cursor::new(&buf)).unwrap();

        assert_eq!(loaded.grid_class(), grid.grid_class());
        assert_eq!(loaded.transform().voxel_size, grid.transform().voxel_size);
        assert!(
            (loaded.tree().background() - grid.tree().background()).abs() < 1e-9,
            "background mismatch"
        );
        assert_eq!(loaded.leaf_count(), grid.leaf_count());
        assert_eq!(loaded.active_voxel_count(), grid.active_voxel_count());

        // Spot-check values at a few active voxels
        for (coord, val) in grid.tree().iter_active().take(50) {
            let loaded_val = loaded.get(coord);
            assert!(
                (loaded_val - val).abs() < 1e-9,
                "value mismatch at {coord}: expected {val}, got {loaded_val}"
            );
        }
    }

    #[test]
    fn roundtrip_empty_grid() {
        let grid = Grid::<f32>::new(42.0, 1.0);
        let mut buf = Vec::new();
        write_vol(&grid, &mut buf).unwrap();

        assert_eq!(buf.len(), HEADER_SIZE); // header only, no leaf data

        let loaded = read_vol(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(loaded.leaf_count(), 0);
        assert_eq!(loaded.active_voxel_count(), 0);
        assert!((loaded.tree().background() - 42.0).abs() < 1e-9);
    }

    #[test]
    fn roundtrip_specific_voxels() {
        let mut grid = Grid::<f32>::new(-1.0, 0.25);
        grid.set_grid_class(GridClass::FogVolume);
        grid.set(Coord::new(0, 0, 0), 1.0);
        grid.set(Coord::new(7, 7, 7), 2.0);
        grid.set(Coord::new(100, 200, 50), 3.14);

        let mut buf = Vec::new();
        write_vol(&grid, &mut buf).unwrap();

        let loaded = read_vol(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(loaded.grid_class(), GridClass::FogVolume);
        assert_eq!(loaded.get(Coord::new(0, 0, 0)), 1.0);
        assert_eq!(loaded.get(Coord::new(7, 7, 7)), 2.0);
        assert!((loaded.get(Coord::new(100, 200, 50)) - 3.14).abs() < 1e-6);
        // Unset voxel should return background
        assert_eq!(loaded.get(Coord::new(99, 99, 99)), -1.0);
    }

    #[test]
    fn verify_magic_bytes() {
        let grid = Grid::<f32>::new(0.0, 1.0);
        let mut buf = Vec::new();
        write_vol(&grid, &mut buf).unwrap();

        assert_eq!(&buf[0..4], b"VOLR");
        assert_eq!(u32::from_le_bytes(buf[4..8].try_into().unwrap()), 1); // version
    }

    #[test]
    fn bad_magic_returns_error() {
        let bad_data = b"NOPE____________________________"; // 32 bytes of junk
        let result = read_vol(&mut Cursor::new(bad_data));
        assert!(result.is_err());
        let err = result.err().expect("should be an error");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("bad magic"));
    }

    #[test]
    fn bad_version_returns_error() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"VOLR");
        buf.extend_from_slice(&99u32.to_le_bytes()); // bad version
        buf.extend_from_slice(&[0u8; 24]); // rest of header
        let result = read_vol(&mut Cursor::new(&buf));
        assert!(result.is_err());
        assert!(result.err().expect("should be an error").to_string().contains("version"));
    }

    #[test]
    fn roundtrip_preserves_active_mask() {
        let mut grid = Grid::<f32>::new(0.0, 1.0);
        grid.set(Coord::new(3, 3, 3), 10.0);
        // Voxel (3,3,3) should be active; (4,4,4) should not
        assert!(grid.tree().is_active(Coord::new(3, 3, 3)));
        assert!(!grid.tree().is_active(Coord::new(4, 4, 4)));

        let mut buf = Vec::new();
        write_vol(&grid, &mut buf).unwrap();
        let loaded = read_vol(&mut Cursor::new(&buf)).unwrap();

        assert!(loaded.tree().is_active(Coord::new(3, 3, 3)));
        assert!(!loaded.tree().is_active(Coord::new(4, 4, 4)));
    }

    #[test]
    fn file_roundtrip() {
        let mut grid = Grid::<f32>::new(0.0, 1.0);
        grid.set(Coord::new(1, 2, 3), 99.0);

        let path = std::env::temp_dir().join("volrus_test.vol");
        let path_str = path.to_str().unwrap();

        save_vol(&grid, path_str).unwrap();
        let loaded = load_vol(path_str).unwrap();

        assert_eq!(loaded.get(Coord::new(1, 2, 3)), 99.0);
        assert_eq!(loaded.active_voxel_count(), 1);

        // Clean up
        let _ = std::fs::remove_file(&path);
    }
}

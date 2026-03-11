//! Integer voxel coordinate — mirrors OpenVDB `openvdb::math::Coord`.

/// Signed integer coordinate for voxel-space addressing.
///
/// All tree nodes store origins and extents as `Coord`.  Arithmetic
/// follows OpenVDB conventions (component-wise ops, bit-masking for
/// octant/tile alignment).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Coord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Coord {
    #[inline]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub const fn origin() -> Self {
        Self { x: 0, y: 0, z: 0 }
    }

    /// Component-wise minimum.
    #[inline]
    pub fn min_comp(&self, other: &Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    /// Component-wise maximum.
    #[inline]
    pub fn max_comp(&self, other: &Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }

    /// L-infinity (Chebyshev) distance to another coordinate.
    #[inline]
    pub fn linf_distance(&self, other: &Self) -> i32 {
        let dx = (self.x - other.x).abs();
        let dy = (self.y - other.y).abs();
        let dz = (self.z - other.z).abs();
        dx.max(dy).max(dz)
    }

    /// Manhattan (L1) distance to another coordinate.
    #[inline]
    pub fn l1_distance(&self, other: &Self) -> i32 {
        (self.x - other.x).abs() + (self.y - other.y).abs() + (self.z - other.z).abs()
    }

    /// Align coordinate down to a grid of size `2^log2dim` per axis.
    #[inline]
    pub fn aligned(&self, log2dim: u32) -> Self {
        let mask = !((1i32 << log2dim) - 1);
        Self {
            x: self.x & mask,
            y: self.y & mask,
            z: self.z & mask,
        }
    }

    /// Offset within a tile of size `2^log2dim` per axis.
    #[inline]
    pub fn offset_in_tile(&self, log2dim: u32) -> usize {
        let dim = 1usize << log2dim;
        let mask = (dim - 1) as i32;
        let lx = (self.x & mask) as usize;
        let ly = (self.y & mask) as usize;
        let lz = (self.z & mask) as usize;
        // ZYX order (matches OpenVDB LeafNode layout)
        (lx << (2 * log2dim)) | (ly << log2dim) | lz
    }

    /// Convert to `[f64; 3]` for world-space arithmetic.
    #[inline]
    pub fn to_f64(self) -> [f64; 3] {
        [self.x as f64, self.y as f64, self.z as f64]
    }
}

impl std::ops::Add for Coord {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::Sub for Coord {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl std::fmt::Display for Coord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coord_origin_is_zero() {
        let c = Coord::origin();
        assert_eq!(c.x, 0);
        assert_eq!(c.y, 0);
        assert_eq!(c.z, 0);
    }

    #[test]
    fn coord_add_sub() {
        let a = Coord::new(1, 2, 3);
        let b = Coord::new(4, 5, 6);
        assert_eq!(a + b, Coord::new(5, 7, 9));
        assert_eq!(b - a, Coord::new(3, 3, 3));
    }

    #[test]
    fn coord_min_max_component() {
        let a = Coord::new(1, 5, 3);
        let b = Coord::new(4, 2, 6);
        assert_eq!(a.min_comp(&b), Coord::new(1, 2, 3));
        assert_eq!(a.max_comp(&b), Coord::new(4, 5, 6));
    }

    #[test]
    fn coord_aligned_rounds_down() {
        // log2dim=3 → tile size 8, mask = !7 = ...11111000
        let c = Coord::new(13, 7, 20);
        let a = c.aligned(3);
        assert_eq!(a, Coord::new(8, 0, 16));
    }

    #[test]
    fn coord_offset_in_tile_matches_zyx_layout() {
        // 8^3 tile (log2dim=3): offset = x*64 + y*8 + z
        let c = Coord::new(2, 3, 5);
        let off = c.offset_in_tile(3);
        assert_eq!(off, 2 * 64 + 3 * 8 + 5);
    }

    #[test]
    fn coord_distances() {
        let a = Coord::new(0, 0, 0);
        let b = Coord::new(3, 4, 5);
        assert_eq!(a.linf_distance(&b), 5);
        assert_eq!(a.l1_distance(&b), 12);
    }

    #[test]
    fn coord_to_f64() {
        let c = Coord::new(-1, 2, 3);
        assert_eq!(c.to_f64(), [-1.0, 2.0, 3.0]);
    }
}

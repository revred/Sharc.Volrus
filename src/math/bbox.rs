//! Axis-aligned bounding box in index space — mirrors OpenVDB `openvdb::math::CoordBBox`.

use super::Coord;

/// Axis-aligned bounding box in index space.
///
/// Stores inclusive min/max corners. An empty box is represented by
/// `min > max` on any axis (the default state).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoordBBox {
    pub min: Coord,
    pub max: Coord,
}

impl CoordBBox {
    /// Create a bounding box from inclusive min and max corners.
    #[inline]
    pub const fn new(min: Coord, max: Coord) -> Self {
        Self { min, max }
    }

    /// Create a bounding box from an origin and dimensions.
    /// The box spans `[origin, origin + dim - 1]` on each axis.
    ///
    /// # Panics
    /// Panics if any component of `dim` is <= 0.
    #[inline]
    pub fn from_origin_and_dim(origin: Coord, dim: Coord) -> Self {
        assert!(dim.x > 0 && dim.y > 0 && dim.z > 0, "dim must be positive");
        Self {
            min: origin,
            max: Coord::new(
                origin.x + dim.x - 1,
                origin.y + dim.y - 1,
                origin.z + dim.z - 1,
            ),
        }
    }

    /// An empty bounding box (min > max).
    #[inline]
    pub const fn empty() -> Self {
        Self {
            min: Coord::new(i32::MAX, i32::MAX, i32::MAX),
            max: Coord::new(i32::MIN, i32::MIN, i32::MIN),
        }
    }

    /// Returns true if the box is empty (min > max on any axis).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }

    /// Returns the extents (dimensions) of the box.
    /// Returns `Coord::origin()` if the box is empty.
    #[inline]
    pub fn dim(&self) -> Coord {
        if self.is_empty() {
            return Coord::origin();
        }
        Coord::new(
            self.max.x - self.min.x + 1,
            self.max.y - self.min.y + 1,
            self.max.z - self.min.z + 1,
        )
    }

    /// Returns the number of voxels in the box.
    #[inline]
    pub fn volume(&self) -> u64 {
        if self.is_empty() {
            return 0;
        }
        let d = self.dim();
        d.x as u64 * d.y as u64 * d.z as u64
    }

    /// Returns true if the box contains the given coordinate (inclusive).
    #[inline]
    pub fn contains(&self, c: Coord) -> bool {
        c.x >= self.min.x
            && c.x <= self.max.x
            && c.y >= self.min.y
            && c.y <= self.max.y
            && c.z >= self.min.z
            && c.z <= self.max.z
    }

    /// Expand the box to include the given coordinate.
    #[inline]
    pub fn expand(&mut self, c: Coord) {
        self.min.x = self.min.x.min(c.x);
        self.min.y = self.min.y.min(c.y);
        self.min.z = self.min.z.min(c.z);
        self.max.x = self.max.x.max(c.x);
        self.max.y = self.max.y.max(c.y);
        self.max.z = self.max.z.max(c.z);
    }

    /// Returns true if this box intersects with `other`.
    #[inline]
    pub fn intersects(&self, other: &CoordBBox) -> bool {
        if self.is_empty() || other.is_empty() {
            return false;
        }
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Returns the intersection of two bounding boxes, or `None` if disjoint.
    #[inline]
    pub fn intersection(&self, other: &CoordBBox) -> Option<CoordBBox> {
        let result = CoordBBox {
            min: self.min.max_comp(&other.min),
            max: self.max.min_comp(&other.max),
        };
        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }

    /// Translate the bounding box by an offset.
    #[inline]
    pub fn translate(&mut self, offset: Coord) {
        self.min = self.min + offset;
        self.max = self.max + offset;
    }

    /// Returns an iterator over all coordinates in the box (ZYX order).
    #[inline]
    pub fn iter(&self) -> CoordBBoxIter {
        CoordBBoxIter {
            bbox: *self,
            cur: self.min,
            done: self.is_empty(),
        }
    }
}

/// Iterator over all coordinates in a `CoordBBox`, in ZYX scan order.
pub struct CoordBBoxIter {
    bbox: CoordBBox,
    cur: Coord,
    done: bool,
}

impl Iterator for CoordBBoxIter {
    type Item = Coord;

    fn next(&mut self) -> Option<Coord> {
        if self.done {
            return None;
        }
        let result = self.cur;

        // Advance z, then y, then x
        self.cur.z += 1;
        if self.cur.z > self.bbox.max.z {
            self.cur.z = self.bbox.min.z;
            self.cur.y += 1;
            if self.cur.y > self.bbox.max.y {
                self.cur.y = self.bbox.min.y;
                self.cur.x += 1;
                if self.cur.x > self.bbox.max.x {
                    self.done = true;
                }
            }
        }

        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done {
            return (0, Some(0));
        }
        let vol = self.bbox.volume() as usize;
        // Approximate remaining — exact would need offset calc
        (0, Some(vol))
    }
}

impl std::fmt::Display for CoordBBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{} .. {}]", self.min, self.max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_bbox() {
        let b = CoordBBox::empty();
        assert!(b.is_empty());
        assert_eq!(b.volume(), 0);
        assert_eq!(b.dim(), Coord::origin());
        assert_eq!(b.iter().count(), 0);
    }

    #[test]
    fn new_bbox() {
        let b = CoordBBox::new(Coord::new(1, 2, 3), Coord::new(3, 4, 5));
        assert!(!b.is_empty());
        assert_eq!(b.dim(), Coord::new(3, 3, 3));
        assert_eq!(b.volume(), 27);
    }

    #[test]
    fn from_origin_and_dim() {
        let b = CoordBBox::from_origin_and_dim(Coord::new(0, 0, 0), Coord::new(4, 4, 4));
        assert_eq!(b.min, Coord::new(0, 0, 0));
        assert_eq!(b.max, Coord::new(3, 3, 3));
        assert_eq!(b.volume(), 64);
    }

    #[test]
    #[should_panic]
    fn from_origin_and_dim_negative() {
        CoordBBox::from_origin_and_dim(Coord::origin(), Coord::new(0, 1, 1));
    }

    #[test]
    fn single_voxel_bbox() {
        let b = CoordBBox::new(Coord::new(5, 5, 5), Coord::new(5, 5, 5));
        assert!(!b.is_empty());
        assert_eq!(b.volume(), 1);
        assert_eq!(b.dim(), Coord::new(1, 1, 1));
        assert!(b.contains(Coord::new(5, 5, 5)));
        assert!(!b.contains(Coord::new(5, 5, 6)));
    }

    #[test]
    fn contains() {
        let b = CoordBBox::new(Coord::new(0, 0, 0), Coord::new(7, 7, 7));
        assert!(b.contains(Coord::new(0, 0, 0)));
        assert!(b.contains(Coord::new(7, 7, 7)));
        assert!(b.contains(Coord::new(3, 4, 5)));
        assert!(!b.contains(Coord::new(-1, 0, 0)));
        assert!(!b.contains(Coord::new(8, 0, 0)));
    }

    #[test]
    fn expand() {
        let mut b = CoordBBox::empty();
        b.expand(Coord::new(3, 5, 1));
        assert_eq!(b.min, Coord::new(3, 5, 1));
        assert_eq!(b.max, Coord::new(3, 5, 1));
        assert_eq!(b.volume(), 1);

        b.expand(Coord::new(0, 0, 0));
        assert_eq!(b.min, Coord::new(0, 0, 0));
        assert_eq!(b.max, Coord::new(3, 5, 1));
        assert_eq!(b.volume(), 4 * 6 * 2);
    }

    #[test]
    fn intersects() {
        let a = CoordBBox::new(Coord::new(0, 0, 0), Coord::new(4, 4, 4));
        let b = CoordBBox::new(Coord::new(3, 3, 3), Coord::new(7, 7, 7));
        let c = CoordBBox::new(Coord::new(5, 5, 5), Coord::new(9, 9, 9));
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
        assert!(!a.intersects(&CoordBBox::empty()));
    }

    #[test]
    fn intersection() {
        let a = CoordBBox::new(Coord::new(0, 0, 0), Coord::new(4, 4, 4));
        let b = CoordBBox::new(Coord::new(2, 2, 2), Coord::new(6, 6, 6));
        let isect = a.intersection(&b).unwrap();
        assert_eq!(isect.min, Coord::new(2, 2, 2));
        assert_eq!(isect.max, Coord::new(4, 4, 4));
        assert_eq!(isect.volume(), 27);

        let c = CoordBBox::new(Coord::new(5, 5, 5), Coord::new(9, 9, 9));
        assert!(a.intersection(&c).is_none());
    }

    #[test]
    fn translate() {
        let mut b = CoordBBox::new(Coord::new(0, 0, 0), Coord::new(3, 3, 3));
        b.translate(Coord::new(10, 20, 30));
        assert_eq!(b.min, Coord::new(10, 20, 30));
        assert_eq!(b.max, Coord::new(13, 23, 33));
    }

    #[test]
    fn iter_2x2x2() {
        let b = CoordBBox::new(Coord::new(0, 0, 0), Coord::new(1, 1, 1));
        let coords: Vec<Coord> = b.iter().collect();
        assert_eq!(coords.len(), 8);
        // ZYX order: x varies slowest, z varies fastest
        assert_eq!(coords[0], Coord::new(0, 0, 0));
        assert_eq!(coords[1], Coord::new(0, 0, 1));
        assert_eq!(coords[2], Coord::new(0, 1, 0));
        assert_eq!(coords[3], Coord::new(0, 1, 1));
        assert_eq!(coords[4], Coord::new(1, 0, 0));
        assert_eq!(coords[7], Coord::new(1, 1, 1));
    }

    #[test]
    fn iter_count_matches_volume() {
        let b = CoordBBox::new(Coord::new(-2, -1, 0), Coord::new(2, 1, 3));
        assert_eq!(b.iter().count(), b.volume() as usize);
    }

    #[test]
    fn display() {
        let b = CoordBBox::new(Coord::new(0, 0, 0), Coord::new(7, 7, 7));
        assert_eq!(format!("{}", b), "[(0, 0, 0) .. (7, 7, 7)]");
    }

    #[test]
    fn negative_coords() {
        let b = CoordBBox::new(Coord::new(-5, -5, -5), Coord::new(-1, -1, -1));
        assert_eq!(b.dim(), Coord::new(5, 5, 5));
        assert_eq!(b.volume(), 125);
        assert!(b.contains(Coord::new(-3, -3, -3)));
        assert!(!b.contains(Coord::new(0, 0, 0)));
    }
}

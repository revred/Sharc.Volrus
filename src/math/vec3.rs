//! 3D f64 vector — mirrors OpenVDB `openvdb::math::Vec3d`.

use super::Coord;

/// A 3-component f64 vector for world-space math.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3d {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3d {
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub const fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Create a vector with all components set to `v`.
    #[inline]
    pub const fn splat(v: f64) -> Self {
        Self { x: v, y: v, z: v }
    }

    /// Dot product.
    #[inline]
    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Cross product.
    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    /// Squared Euclidean length.
    #[inline]
    pub fn length_squared(self) -> f64 {
        self.dot(self)
    }

    /// Euclidean length.
    #[inline]
    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }

    /// Returns a unit-length vector in the same direction.
    /// Returns zero vector if the length is near zero.
    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < f64::EPSILON {
            Self::zero()
        } else {
            self * (1.0 / len)
        }
    }

    /// Component-wise minimum.
    #[inline]
    pub fn min_comp(self, other: Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    /// Component-wise maximum.
    #[inline]
    pub fn max_comp(self, other: Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }

    /// Linear interpolation: `self * (1 - t) + other * t`.
    #[inline]
    pub fn lerp(self, other: Self, t: f64) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }
}

// --- Operator impls ---

impl std::ops::Add for Vec3d {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::Sub for Vec3d {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl std::ops::Mul<f64> for Vec3d {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl std::ops::Mul<Vec3d> for f64 {
    type Output = Vec3d;
    #[inline]
    fn mul(self, rhs: Vec3d) -> Vec3d {
        Vec3d::new(self * rhs.x, self * rhs.y, self * rhs.z)
    }
}

impl std::ops::Neg for Vec3d {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

// --- Conversions ---

impl From<[f64; 3]> for Vec3d {
    #[inline]
    fn from(a: [f64; 3]) -> Self {
        Self::new(a[0], a[1], a[2])
    }
}

impl From<Vec3d> for [f64; 3] {
    #[inline]
    fn from(v: Vec3d) -> [f64; 3] {
        [v.x, v.y, v.z]
    }
}

impl From<Coord> for Vec3d {
    #[inline]
    fn from(c: Coord) -> Self {
        Self::new(c.x as f64, c.y as f64, c.z as f64)
    }
}

impl std::fmt::Display for Vec3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn approx_eq(a: Vec3d, b: Vec3d) {
        assert!((a.x - b.x).abs() < EPS, "x: {} != {}", a.x, b.x);
        assert!((a.y - b.y).abs() < EPS, "y: {} != {}", a.y, b.y);
        assert!((a.z - b.z).abs() < EPS, "z: {} != {}", a.z, b.z);
    }

    #[test]
    fn zero_and_splat() {
        assert_eq!(Vec3d::zero(), Vec3d::new(0.0, 0.0, 0.0));
        assert_eq!(Vec3d::splat(5.0), Vec3d::new(5.0, 5.0, 5.0));
    }

    #[test]
    fn dot_product() {
        let a = Vec3d::new(1.0, 2.0, 3.0);
        let b = Vec3d::new(4.0, 5.0, 6.0);
        assert!((a.dot(b) - 32.0).abs() < EPS);
    }

    #[test]
    fn cross_product() {
        let x = Vec3d::new(1.0, 0.0, 0.0);
        let y = Vec3d::new(0.0, 1.0, 0.0);
        let z = x.cross(y);
        approx_eq(z, Vec3d::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn cross_product_anticommutative() {
        let a = Vec3d::new(1.0, 2.0, 3.0);
        let b = Vec3d::new(4.0, 5.0, 6.0);
        approx_eq(a.cross(b), -b.cross(a));
    }

    #[test]
    fn length_and_normalize() {
        let v = Vec3d::new(3.0, 4.0, 0.0);
        assert!((v.length() - 5.0).abs() < EPS);
        assert!((v.length_squared() - 25.0).abs() < EPS);

        let n = v.normalize();
        assert!((n.length() - 1.0).abs() < EPS);
        approx_eq(n, Vec3d::new(0.6, 0.8, 0.0));
    }

    #[test]
    fn normalize_zero_returns_zero() {
        let n = Vec3d::zero().normalize();
        assert_eq!(n, Vec3d::zero());
    }

    #[test]
    fn min_max_comp() {
        let a = Vec3d::new(1.0, 5.0, 3.0);
        let b = Vec3d::new(4.0, 2.0, 6.0);
        assert_eq!(a.min_comp(b), Vec3d::new(1.0, 2.0, 3.0));
        assert_eq!(a.max_comp(b), Vec3d::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn lerp() {
        let a = Vec3d::new(0.0, 0.0, 0.0);
        let b = Vec3d::new(10.0, 20.0, 30.0);
        approx_eq(a.lerp(b, 0.0), a);
        approx_eq(a.lerp(b, 1.0), b);
        approx_eq(a.lerp(b, 0.5), Vec3d::new(5.0, 10.0, 15.0));
    }

    #[test]
    fn add_sub() {
        let a = Vec3d::new(1.0, 2.0, 3.0);
        let b = Vec3d::new(4.0, 5.0, 6.0);
        assert_eq!(a + b, Vec3d::new(5.0, 7.0, 9.0));
        assert_eq!(b - a, Vec3d::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn scalar_mul() {
        let v = Vec3d::new(1.0, 2.0, 3.0);
        assert_eq!(v * 2.0, Vec3d::new(2.0, 4.0, 6.0));
        assert_eq!(2.0 * v, Vec3d::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn negation() {
        let v = Vec3d::new(1.0, -2.0, 3.0);
        assert_eq!(-v, Vec3d::new(-1.0, 2.0, -3.0));
    }

    #[test]
    fn from_array() {
        let v: Vec3d = [1.0, 2.0, 3.0].into();
        assert_eq!(v, Vec3d::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn into_array() {
        let a: [f64; 3] = Vec3d::new(1.0, 2.0, 3.0).into();
        assert_eq!(a, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn from_coord() {
        let c = Coord::new(-1, 2, 3);
        let v: Vec3d = c.into();
        assert_eq!(v, Vec3d::new(-1.0, 2.0, 3.0));
    }

    #[test]
    fn display() {
        let v = Vec3d::new(1.5, 2.5, 3.5);
        assert_eq!(format!("{}", v), "(1.5, 2.5, 3.5)");
    }

    #[test]
    fn orthogonality_of_cross() {
        let a = Vec3d::new(1.0, 2.0, 3.0);
        let b = Vec3d::new(4.0, 5.0, 6.0);
        let c = a.cross(b);
        assert!(c.dot(a).abs() < EPS);
        assert!(c.dot(b).abs() < EPS);
    }
}

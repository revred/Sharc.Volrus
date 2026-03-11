//! Direct volume rendering and level-set surface rendering.

use super::camera::Camera;
use crate::grid::Grid;
use crate::tools::interpolation::sample_trilinear;
use crate::tools::ray_intersect::ray_intersect;

/// Volume rendering configuration.
pub struct VolumeRenderConfig {
    pub width: u32,
    pub height: u32,
    pub step_size: f64,
    pub density_scale: f64,
    pub transfer_fn: TransferFunction,
}

/// Transfer function mapping scalar values to color+opacity.
pub enum TransferFunction {
    /// Linear ramp from min_val to max_val, mapped to white.
    Linear { min_val: f32, max_val: f32 },
    /// Rainbow colormap (like jet).
    Jet { min_val: f32, max_val: f32 },
    /// Grayscale.
    Grayscale { min_val: f32, max_val: f32 },
    /// Cool-warm diverging (blue -> white -> red).
    CoolWarm { min_val: f32, max_val: f32 },
}

impl TransferFunction {
    /// Map a scalar value to RGBA color.
    pub fn map(&self, value: f32) -> Rgba {
        match self {
            TransferFunction::Linear { min_val, max_val } => {
                let t = remap(value, *min_val, *max_val);
                Rgba {
                    r: t,
                    g: t,
                    b: t,
                    a: t,
                }
            }
            TransferFunction::Jet { min_val, max_val } => {
                let t = remap(value, *min_val, *max_val);
                jet_colormap(t)
            }
            TransferFunction::Grayscale { min_val, max_val } => {
                let t = remap(value, *min_val, *max_val);
                Rgba {
                    r: t,
                    g: t,
                    b: t,
                    a: t,
                }
            }
            TransferFunction::CoolWarm { min_val, max_val } => {
                let t = remap(value, *min_val, *max_val);
                cool_warm_colormap(t)
            }
        }
    }
}

fn remap(value: f32, min_val: f32, max_val: f32) -> f32 {
    if (max_val - min_val).abs() < 1e-10 {
        return 0.0;
    }
    ((value - min_val) / (max_val - min_val)).clamp(0.0, 1.0)
}

fn jet_colormap(t: f32) -> Rgba {
    // Classic jet: blue -> cyan -> green -> yellow -> red
    let r = (1.5 - (t - 0.75).abs() * 4.0).clamp(0.0, 1.0);
    let g = (1.5 - (t - 0.5).abs() * 4.0).clamp(0.0, 1.0);
    let b = (1.5 - (t - 0.25).abs() * 4.0).clamp(0.0, 1.0);
    Rgba { r, g, b, a: t }
}

fn cool_warm_colormap(t: f32) -> Rgba {
    // Blue (0.0) -> White (0.5) -> Red (1.0)
    let r = if t < 0.5 {
        t * 2.0
    } else {
        1.0
    };
    let g = 1.0 - (t - 0.5).abs() * 2.0;
    let b = if t > 0.5 {
        (1.0 - t) * 2.0
    } else {
        1.0
    };
    Rgba {
        r: r.clamp(0.0, 1.0),
        g: g.clamp(0.0, 1.0),
        b: b.clamp(0.0, 1.0),
        a: t.clamp(0.0, 1.0),
    }
}

/// RGBA pixel.
#[derive(Debug, Clone, Copy)]
pub struct Rgba {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Rgba {
    pub const TRANSPARENT: Rgba = Rgba {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };

    pub const BLACK: Rgba = Rgba {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
}

/// Rendered image.
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<Rgba>,
}

impl Image {
    /// Create a new image filled with the given color.
    pub fn new(width: u32, height: u32, fill: Rgba) -> Self {
        Self {
            width,
            height,
            pixels: vec![fill; (width * height) as usize],
        }
    }

    /// Get pixel at (x, y).
    pub fn pixel(&self, x: u32, y: u32) -> Rgba {
        self.pixels[(y * self.width + x) as usize]
    }

    /// Set pixel at (x, y).
    pub fn set_pixel(&mut self, x: u32, y: u32, color: Rgba) {
        self.pixels[(y * self.width + x) as usize] = color;
    }

    /// Write as PPM file (simple uncompressed format, no external deps).
    pub fn save_ppm(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        write!(file, "P6\n{} {}\n255\n", self.width, self.height)?;
        for pixel in &self.pixels {
            let r = (pixel.r.clamp(0.0, 1.0) * 255.0) as u8;
            let g = (pixel.g.clamp(0.0, 1.0) * 255.0) as u8;
            let b = (pixel.b.clamp(0.0, 1.0) * 255.0) as u8;
            file.write_all(&[r, g, b])?;
        }
        Ok(())
    }

    /// Raw RGBA f32 bytes for GPU texture upload.
    pub fn as_rgba_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.pixels.as_ptr() as *const u8,
                self.pixels.len() * std::mem::size_of::<Rgba>(),
            )
        }
    }
}

/// Render a fog volume using ray marching with front-to-back compositing.
pub fn render_volume(
    grid: &Grid<f32>,
    camera: &Camera,
    config: &VolumeRenderConfig,
) -> Image {
    let w = config.width;
    let h = config.height;
    let mut image = Image::new(w, h, Rgba::TRANSPARENT);

    for y in 0..h {
        for x in 0..w {
            let u = (x as f64 + 0.5) / w as f64;
            let v = 1.0 - (y as f64 + 0.5) / h as f64; // flip Y

            let (origin, dir) = camera.ray_at(u, v);

            let pixel = raymarch_volume(grid, &origin, &dir, config);
            image.set_pixel(x, y, pixel);
        }
    }

    image
}

fn raymarch_volume(
    grid: &Grid<f32>,
    origin: &[f64; 3],
    dir: &[f64; 3],
    config: &VolumeRenderConfig,
) -> Rgba {
    let mut color_r = 0.0f32;
    let mut color_g = 0.0f32;
    let mut color_b = 0.0f32;
    let mut alpha = 0.0f32;

    let max_steps = 500u32;
    let step = config.step_size;

    for i in 0..max_steps {
        let t = i as f64 * step;
        let pos = [
            origin[0] + dir[0] * t,
            origin[1] + dir[1] * t,
            origin[2] + dir[2] * t,
        ];

        let value = sample_trilinear(grid, pos);

        // Skip zero/background values
        if value.abs() < 1e-8 {
            continue;
        }

        let sample = config.transfer_fn.map(value);
        let sample_alpha = (sample.a * config.density_scale as f32 * step as f32).clamp(0.0, 1.0);

        // Front-to-back compositing
        let weight = (1.0 - alpha) * sample_alpha;
        color_r += weight * sample.r;
        color_g += weight * sample.g;
        color_b += weight * sample.b;
        alpha += weight;

        // Early termination
        if alpha > 0.99 {
            break;
        }
    }

    Rgba {
        r: color_r,
        g: color_g,
        b: color_b,
        a: alpha,
    }
}

/// Render a level-set surface with basic Phong shading.
/// Uses ray_intersect for primary rays + gradient-based normals.
pub fn render_surface(
    grid: &Grid<f32>,
    camera: &Camera,
    width: u32,
    height: u32,
    light_dir: [f64; 3],
) -> Image {
    let mut image = Image::new(width, height, Rgba::BLACK);

    // Normalize light direction
    let ll = (light_dir[0] * light_dir[0]
        + light_dir[1] * light_dir[1]
        + light_dir[2] * light_dir[2])
        .sqrt();
    let light = if ll > 1e-15 {
        [light_dir[0] / ll, light_dir[1] / ll, light_dir[2] / ll]
    } else {
        [0.0, 0.0, 1.0]
    };

    let bg = Rgba {
        r: 0.1,
        g: 0.1,
        b: 0.15,
        a: 1.0,
    };

    for y in 0..height {
        for x in 0..width {
            let u = (x as f64 + 0.5) / width as f64;
            let v = 1.0 - (y as f64 + 0.5) / height as f64;

            let (origin, dir) = camera.ray_at(u, v);

            if let Some(hit) = ray_intersect(grid, origin, dir) {
                let n = hit.normal;

                // Phong shading
                let ambient = 0.15;
                let n_dot_l = (n[0] * light[0] + n[1] * light[1] + n[2] * light[2]).max(0.0);
                let diffuse = 0.7 * n_dot_l;

                // Specular: reflection of light about normal
                let r_dot = 2.0 * n_dot_l;
                let refl = [
                    r_dot * n[0] - light[0],
                    r_dot * n[1] - light[1],
                    r_dot * n[2] - light[2],
                ];
                // View direction (from hit point to eye)
                let view = [
                    origin[0] - hit.position[0],
                    origin[1] - hit.position[1],
                    origin[2] - hit.position[2],
                ];
                let vl = (view[0] * view[0] + view[1] * view[1] + view[2] * view[2]).sqrt();
                let view_n = if vl > 1e-15 {
                    [view[0] / vl, view[1] / vl, view[2] / vl]
                } else {
                    [0.0, 0.0, 0.0]
                };
                let r_dot_v =
                    (refl[0] * view_n[0] + refl[1] * view_n[1] + refl[2] * view_n[2]).max(0.0);
                let specular = 0.3 * r_dot_v.powf(32.0);

                let intensity = (ambient + diffuse + specular) as f32;
                image.set_pixel(
                    x,
                    y,
                    Rgba {
                        r: (0.8 * intensity).min(1.0),
                        g: (0.85 * intensity).min(1.0),
                        b: (0.9 * intensity).min(1.0),
                        a: 1.0,
                    },
                );
            } else {
                image.set_pixel(x, y, bg);
            }
        }
    }

    image
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Coord;
    use crate::tools::sphere::make_level_set_sphere;

    fn make_sphere() -> Grid<f32> {
        make_level_set_sphere(5.0, [0.0, 0.0, 0.0], 0.5, 3.0)
    }

    fn make_fog_sphere() -> Grid<f32> {
        // Create a fog volume: density = max(0, 1 - |r|/radius)
        let radius = 5.0f64;
        let mut grid = Grid::<f32>::new(0.0, 1.0);
        grid.set_name("density");
        use crate::grid::GridClass;
        grid.set_grid_class(GridClass::FogVolume);

        for x in -6..=6 {
            for y in -6..=6 {
                for z in -6..=6 {
                    let r = ((x * x + y * y + z * z) as f64).sqrt();
                    let density = (1.0 - r / radius).max(0.0) as f32;
                    if density > 0.0 {
                        grid.set(Coord::new(x, y, z), density);
                    }
                }
            }
        }
        grid
    }

    fn make_camera_looking_at_sphere() -> Camera {
        Camera::new(
            [0.0, 0.0, 20.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            std::f64::consts::FRAC_PI_4,
            1.0,
        )
    }

    #[test]
    fn render_surface_sphere_has_non_black_center() {
        let grid = make_sphere();
        let camera = make_camera_looking_at_sphere();
        let image = render_surface(&grid, &camera, 32, 32, [0.0, 0.0, 1.0]);

        // Center pixel should have been hit by the ray
        let center = image.pixel(16, 16);
        let brightness = center.r + center.g + center.b;
        assert!(
            brightness > 0.1,
            "Center pixel should be illuminated, got r={} g={} b={}",
            center.r,
            center.g,
            center.b
        );
    }

    #[test]
    fn render_surface_corner_is_background() {
        let grid = make_sphere();
        let camera = make_camera_looking_at_sphere();
        let image = render_surface(&grid, &camera, 32, 32, [0.0, 0.0, 1.0]);

        // Corner pixel should miss the sphere and be background
        let corner = image.pixel(0, 0);
        // Background is (0.1, 0.1, 0.15)
        assert!(
            (corner.r - 0.1).abs() < 0.01,
            "Corner r should be background (0.1), got {}",
            corner.r
        );
    }

    #[test]
    fn render_volume_fog_sphere_has_nonzero_alpha_at_center() {
        let grid = make_fog_sphere();
        let camera = make_camera_looking_at_sphere();

        let config = VolumeRenderConfig {
            width: 16,
            height: 16,
            step_size: 0.5,
            density_scale: 2.0,
            transfer_fn: TransferFunction::Linear {
                min_val: 0.0,
                max_val: 1.0,
            },
        };

        let image = render_volume(&grid, &camera, &config);
        let center = image.pixel(8, 8);
        assert!(
            center.a > 0.0,
            "Center pixel should have non-zero alpha from volume, got {}",
            center.a
        );
    }

    #[test]
    fn render_volume_empty_grid_is_transparent() {
        let grid = Grid::<f32>::new(0.0, 1.0);
        let camera = make_camera_looking_at_sphere();

        let config = VolumeRenderConfig {
            width: 8,
            height: 8,
            step_size: 1.0,
            density_scale: 1.0,
            transfer_fn: TransferFunction::Linear {
                min_val: 0.0,
                max_val: 1.0,
            },
        };

        let image = render_volume(&grid, &camera, &config);
        for y in 0..8 {
            for x in 0..8 {
                let p = image.pixel(x, y);
                assert!(
                    p.a < 0.01,
                    "Empty grid should produce transparent image, pixel ({},{}) has alpha {}",
                    x,
                    y,
                    p.a
                );
            }
        }
    }

    #[test]
    fn image_save_ppm() {
        let mut image = Image::new(4, 4, Rgba::BLACK);
        image.set_pixel(
            0,
            0,
            Rgba {
                r: 1.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
        );
        image.set_pixel(
            1,
            0,
            Rgba {
                r: 0.0,
                g: 1.0,
                b: 0.0,
                a: 1.0,
            },
        );

        let path = std::env::temp_dir().join("volrus_test_image.ppm");
        let path_str = path.to_str().unwrap();
        image.save_ppm(path_str).expect("Failed to save PPM");

        // Verify the file exists and has correct header
        let data = std::fs::read(path_str).expect("Failed to read PPM");
        assert!(data.len() > 0);

        // Check PPM header
        let header_end = data.windows(1).position(|w| w[0] == b'\n').unwrap();
        let header = std::str::from_utf8(&data[..header_end]).unwrap();
        assert_eq!(header, "P6");

        // Clean up
        let _ = std::fs::remove_file(path_str);
    }

    #[test]
    fn transfer_fn_linear_range() {
        let tf = TransferFunction::Linear {
            min_val: 0.0,
            max_val: 1.0,
        };
        let zero = tf.map(0.0);
        assert!(zero.r < 0.01);
        let one = tf.map(1.0);
        assert!(one.r > 0.99);
        let half = tf.map(0.5);
        assert!((half.r - 0.5).abs() < 0.01);
    }

    #[test]
    fn transfer_fn_jet_range() {
        let tf = TransferFunction::Jet {
            min_val: 0.0,
            max_val: 1.0,
        };
        // At t=0, jet should be blue-ish
        let low = tf.map(0.0);
        assert!(low.b > low.r, "Jet at 0 should be blue: r={} b={}", low.r, low.b);
        // At t=1, jet should be red-ish
        let high = tf.map(1.0);
        assert!(
            high.r > high.b,
            "Jet at 1 should be red: r={} b={}",
            high.r,
            high.b
        );
    }

    #[test]
    fn transfer_fn_cool_warm_range() {
        let tf = TransferFunction::CoolWarm {
            min_val: 0.0,
            max_val: 1.0,
        };
        // At t=0, should be blue
        let cool = tf.map(0.0);
        assert!(cool.b > cool.r, "CoolWarm at 0 should be blue");
        // At t=1, should be red
        let warm = tf.map(1.0);
        assert!(warm.r > warm.b, "CoolWarm at 1 should be red");
        // At t=0.5, should be white-ish
        let mid = tf.map(0.5);
        assert!(mid.g > 0.8, "CoolWarm at 0.5 should be bright");
    }

    #[test]
    fn image_as_rgba_bytes_length() {
        let image = Image::new(4, 4, Rgba::BLACK);
        let bytes = image.as_rgba_bytes();
        // 4*4 pixels * 4 floats * 4 bytes = 256
        assert_eq!(bytes.len(), 4 * 4 * 4 * 4);
    }

    #[test]
    fn image_pixel_set_get_roundtrip() {
        let mut image = Image::new(2, 2, Rgba::TRANSPARENT);
        let color = Rgba {
            r: 0.5,
            g: 0.6,
            b: 0.7,
            a: 0.8,
        };
        image.set_pixel(1, 1, color);
        let p = image.pixel(1, 1);
        assert!((p.r - 0.5).abs() < 1e-6);
        assert!((p.g - 0.6).abs() < 1e-6);
        assert!((p.b - 0.7).abs() < 1e-6);
        assert!((p.a - 0.8).abs() < 1e-6);
    }
}

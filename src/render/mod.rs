//! Render module — types and utilities for hardware-accelerated rendering pipelines.
//!
//! Provides GPU-ready mesh formats, a perspective camera, and volume/surface rendering.

pub mod camera;
pub mod vertex;
pub mod volume_render;

pub use camera::Camera;
pub use vertex::{GpuIndex, GpuMesh, GpuVertex};
pub use volume_render::{
    render_surface, render_volume, Image, Rgba, TransferFunction, VolumeRenderConfig,
};

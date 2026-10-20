// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::ffi::*;

/// Block size and dimensions of a format: bytes per block, texels per block in x, y and z.
#[derive(Clone, Copy)]
pub struct Blocks {
    pub bs: u32,
    pub bw: u32,
    pub bh: u32,
    pub bd: u32,
}

/// Bytes of mip `level` and its width in blocks.
pub fn mip_level_size(extent: [u32; 3], level: u32, b: Blocks) -> (u64, u32) {
    let dim = |e: u32, block: u32| e.checked_shr(level).unwrap_or(0).max(1).div_ceil(block);
    let w = dim(extent[0], b.bw);
    let (h, d) = (dim(extent[1], b.bh), dim(extent[2], b.bd));
    (w as u64 * h as u64 * d as u64 * b.bs as u64, w)
}

fn extent(image: &borgvk_image) -> [u32; 3] {
    let e = image.vk.extent;
    [e.width, e.height, e.depth]
}

/// Bytes of all mip levels of one sample plane.
pub fn sample_plane_size(image: &borgvk_image, b: Blocks) -> u64 {
    (0..image.vk.mip_levels).map(|l| mip_level_size(extent(image), l, b).0).sum()
}

/// Bytes of one array layer: every sample plane.
pub fn layer_size(image: &borgvk_image, b: Blocks) -> u64 {
    sample_plane_size(image, b) * (image.vk.samples as u64).max(1)
}

/// Offset of `level` in `layer`, and that level's row stride in blocks.
pub fn level_offset(image: &borgvk_image, level: u32, layer: u32, b: Blocks) -> (u64, u32) {
    let before: u64 = (0..level).map(|l| mip_level_size(extent(image), l, b).0).sum();
    let stride = mip_level_size(extent(image), level, b).1;
    (layer as u64 * layer_size(image, b) + before, stride)
}

pub fn z16_pack(z: f32) -> [u8; 2] {
    ((z.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16).to_le_bytes()
}

pub fn z16_unpack(t: [u8; 2]) -> f32 {
    u16::from_le_bytes(t) as f32 / 65535.0
}

#[no_mangle]
pub extern "C" fn borgvk_is_z16_s8(fmt: pipe_format) -> bool {
    fmt == PIPE_FORMAT_Z16_UNORM_S8_UINT
}

#[no_mangle]
pub extern "C" fn borgvk_is_r4g4_unorm(fmt: VkFormat) -> bool {
    fmt == VK_FORMAT_R4G4_UNORM_PACK8
}

/// # Safety
/// `width_blocks_out` is null or valid for a write.
#[no_mangle]
pub unsafe extern "C" fn borgvk_mip_level_size(
    width: u32,
    height: u32,
    depth: u32,
    level: u32,
    bs: u32,
    bw: u32,
    bh: u32,
    bd: u32,
    width_blocks_out: *mut u32,
) -> u64 {
    let (size, w) = mip_level_size([width, height, depth], level, Blocks { bs, bw, bh, bd });
    if let Some(out) = unsafe { width_blocks_out.as_mut() } {
        *out = w;
    }
    size
}

/// # Safety
/// `image` is a valid image.
#[no_mangle]
pub unsafe extern "C" fn borgvk_image_sample_plane_size(
    image: *const borgvk_image,
    bs: u32,
    bw: u32,
    bh: u32,
    bd: u32,
) -> u64 {
    sample_plane_size(unsafe { &*image }, Blocks { bs, bw, bh, bd })
}

/// # Safety
/// `image` is a valid image.
#[no_mangle]
pub unsafe extern "C" fn borgvk_image_layer_size(
    image: *const borgvk_image,
    bs: u32,
    bw: u32,
    bh: u32,
    bd: u32,
) -> u64 {
    layer_size(unsafe { &*image }, Blocks { bs, bw, bh, bd })
}

/// # Safety
/// `image` is a valid image; `stride_blocks_out` is null or valid for a write.
#[no_mangle]
pub unsafe extern "C" fn borgvk_image_level_offset(
    image: *const borgvk_image,
    level: u32,
    layer: u32,
    bs: u32,
    bw: u32,
    bh: u32,
    bd: u32,
    stride_blocks_out: *mut u32,
) -> u64 {
    let (offset, stride) = level_offset(unsafe { &*image }, level, layer, Blocks { bs, bw, bh, bd });
    if let Some(out) = unsafe { stride_blocks_out.as_mut() } {
        *out = stride;
    }
    offset
}

/// # Safety
/// `texel` is valid for writing two bytes.
#[no_mangle]
pub unsafe extern "C" fn borgvk_z16_s8_pack_z(texel: *mut u8, z: f32) {
    unsafe { std::ptr::copy_nonoverlapping(z16_pack(z).as_ptr(), texel, 2) };
}

/// # Safety
/// `texel` is valid for reading two bytes.
#[no_mangle]
pub unsafe extern "C" fn borgvk_z16_s8_unpack_z(texel: *const u8) -> f32 {
    z16_unpack(unsafe { [*texel, *texel.add(1)] })
}

#[cfg(test)]
mod tests {
    use super::*;

    const RGBA8: Blocks = Blocks { bs: 4, bw: 1, bh: 1, bd: 1 };

    #[test]
    fn mip_sizes() {
        assert_eq!(mip_level_size([16, 16, 1], 0, RGBA8), (1024, 16));
        assert_eq!(mip_level_size([16, 16, 1], 4, RGBA8).0, 4);
        assert_eq!(mip_level_size([16, 16, 1], 9, RGBA8).0, 4);
    }

    #[test]
    fn block_compressed_size_rounds_up() {
        assert_eq!(mip_level_size([10, 10, 1], 0, Blocks { bs: 8, bw: 4, bh: 4, bd: 1 }), (72, 3));
    }

    #[test]
    fn z16_round_trips() {
        assert_eq!(z16_pack(1.0), [0xff, 0xff]);
        assert_eq!(z16_unpack(z16_pack(1.0)), 1.0);
        assert_eq!(z16_unpack(z16_pack(-3.0)), 0.0);
        assert!((z16_unpack(z16_pack(0.5)) - 0.5).abs() < 1.0 / 65535.0);
        assert_eq!(z16_pack(f32::NAN), [0, 0]);
    }

    #[test]
    fn format_checks() {
        assert!(borgvk_is_z16_s8(PIPE_FORMAT_Z16_UNORM_S8_UINT));
        assert!(!borgvk_is_z16_s8(PIPE_FORMAT_R8G8B8A8_UNORM));
        assert!(borgvk_is_r4g4_unorm(VK_FORMAT_R4G4_UNORM_PACK8));
    }
}

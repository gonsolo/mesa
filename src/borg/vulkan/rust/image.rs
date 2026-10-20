// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::ffi::*;

use borg_layout::image::{mip_level_size, Blocks, Shape};

fn shape(image: &borgvk_image) -> Shape {
    let e = image.vk.extent;
    Shape { extent: [e.width, e.height, e.depth], levels: image.vk.mip_levels, layers: image.vk.array_layers, samples: image.vk.samples }
}

/// The blocks of a format.
fn blocks(fmt: VkFormat) -> Blocks {
    let pf = unsafe { vk_format_to_pipe_format(fmt) };
    unsafe {
        Blocks {
            bs: util_format_get_blocksize(pf),
            bw: util_format_get_blockwidth(pf),
            bh: util_format_get_blockheight(pf),
            bd: util_format_get_blockdepth(pf),
        }
    }
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
    shape(unsafe { &*image }).sample_plane_size(Blocks { bs, bw, bh, bd })
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
    shape(unsafe { &*image }).layer_size(Blocks { bs, bw, bh, bd })
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
    let (offset, stride) = shape(unsafe { &*image }).level_offset(level, layer, Blocks { bs, bw, bh, bd });
    if let Some(out) = unsafe { stride_blocks_out.as_mut() } {
        *out = stride;
    }
    offset
}

/// # Safety
/// `info` is valid.
#[no_mangle]
pub unsafe extern "C" fn borgvk_image_size(info: *const VkImageCreateInfo) -> u64 {
    let i = unsafe { &*info };
    let e = i.extent;
    let s = Shape { extent: [e.width, e.height, e.depth], levels: i.mipLevels, layers: i.arrayLayers, samples: i.samples };
    s.size(blocks(i.format))
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

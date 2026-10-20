// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Single texels in an image's format: clear colours, depth and stencil, RGBA in and out.

use crate::ffi::*;
use crate::image::{borgvk_is_r4g4_unorm, z16_pack};

pub fn unpack_rgba_float(vk: VkFormat, pf: pipe_format, texel: *const u8, out: &mut [f32; 4]) {
    if borgvk_is_r4g4_unorm(vk) {
        let t = unsafe { *texel };
        *out = [(t >> 4 & 0xF) as f32 / 15.0, (t & 0xF) as f32 / 15.0, 0.0, 1.0];
    } else {
        unsafe { util_format_unpack_rgba(pf, out.as_mut_ptr().cast(), texel.cast(), 1) };
    }
}

pub fn pack_rgba_float(vk: VkFormat, pf: pipe_format, texel: *mut u8, v: &[f32; 4]) {
    if borgvk_is_r4g4_unorm(vk) {
        let q = |x: f32| (x.clamp(0.0, 1.0) * 15.0 + 0.5) as u8;
        unsafe { *texel = q(v[0]) << 4 | q(v[1]) };
    } else {
        unsafe { util_format_pack_rgba(pf, texel.cast(), v.as_ptr().cast(), 1) };
    }
}

/// The type of the first channel of a scaled format.
fn scaled_type(pf: pipe_format) -> Option<util_format_type> {
    if !unsafe { util_format_is_scaled(pf) } {
        return None;
    }
    let ch = unsafe { util_format_get_first_non_void_channel(pf) };
    (ch >= 0).then(|| unsafe { (*util_format_description(pf)).channel[ch as usize].type_() } as util_format_type)
}

/// Packs a clear colour: integers as they are, scaled formats from their integer value.
pub fn pack_clear_color(vk: VkFormat, pf: pipe_format, pixel: &mut [u8; 16], c: &VkClearColorValue) {
    let ch = unsafe { util_format_get_first_non_void_channel(pf) };
    let wide = ch >= 0 && unsafe { util_format_is_pure_integer(pf) && (*util_format_description(pf)).channel[ch as usize].size() == 64 };
    unsafe {
        if wide {
            let n = (util_format_get_blocksize(pf) as usize).min(16);
            std::ptr::copy_nonoverlapping(c.uint32.as_ptr().cast::<u8>(), pixel.as_mut_ptr(), n);
        } else if util_format_is_pure_uint(pf) {
            util_format_pack_rgba(pf, pixel.as_mut_ptr().cast(), c.uint32.as_ptr().cast(), 1);
        } else if util_format_is_pure_sint(pf) {
            util_format_pack_rgba(pf, pixel.as_mut_ptr().cast(), c.int32.as_ptr().cast(), 1);
        } else if scaled_type(pf) == Some(UTIL_FORMAT_TYPE_UNSIGNED) {
            let v = c.uint32.map(|x| x as f32);
            util_format_pack_rgba(pf, pixel.as_mut_ptr().cast(), v.as_ptr().cast(), 1);
        } else if scaled_type(pf) == Some(UTIL_FORMAT_TYPE_SIGNED) {
            let v = c.int32.map(|x| x as f32);
            util_format_pack_rgba(pf, pixel.as_mut_ptr().cast(), v.as_ptr().cast(), 1);
        } else {
            pack_rgba_float(vk, pf, pixel.as_mut_ptr(), &c.float32);
        }
    }
}

/// Writes the depth and stencil aspects in `aspect` of one texel.
pub fn write_depth_stencil(pf: pipe_format, texel: *mut u8, aspect: VkImageAspectFlags, depth: f32, stencil: u8) {
    let (d, s) = (aspect & VK_IMAGE_ASPECT_DEPTH_BIT != 0, aspect & VK_IMAGE_ASPECT_STENCIL_BIT != 0);
    unsafe {
        if pf == PIPE_FORMAT_Z16_UNORM_S8_UINT {
            if d {
                std::ptr::copy_nonoverlapping(z16_pack(depth).as_ptr(), texel, 2);
            }
            if s {
                *texel.add(2) = stencil;
            }
            return;
        }
        if d {
            util_format_pack_z_float(pf, texel.cast(), &depth, 1);
        }
        if s {
            util_format_pack_s_8uint(pf, texel.cast(), &stencil, 1);
        }
    }
}

/// One aspect of a combined depth/stencil format, but not both.
pub fn is_partial_ds_aspect(f: VkFormat, aspect: VkImageAspectFlags) -> bool {
    let ds = unsafe { vk_format_has_depth(f) && vk_format_has_stencil(f) };
    ds && aspect != (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)
}

/// # Safety
/// `texel` holds a texel of the format, `out` has room for 4 floats.
#[no_mangle]
pub unsafe extern "C" fn borgvk_unpack_rgba_float(vk: VkFormat, pf: pipe_format, texel: *const u8, out: *mut f32) {
    unpack_rgba_float(vk, pf, texel, unsafe { &mut *out.cast() });
}

/// # Safety
/// `texel` has room for a texel of the format, `v` holds 4 floats.
#[no_mangle]
pub unsafe extern "C" fn borgvk_pack_rgba_float(vk: VkFormat, pf: pipe_format, texel: *mut u8, v: *const f32) {
    pack_rgba_float(vk, pf, texel, unsafe { &*v.cast() });
}

#[no_mangle]
pub extern "C" fn borgvk_is_partial_ds_aspect(f: VkFormat, aspect: VkImageAspectFlags) -> bool {
    is_partial_ds_aspect(f, aspect)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r4g4_packs_by_hand() {
        let mut t = 0u8;
        pack_rgba_float(VK_FORMAT_R4G4_UNORM_PACK8, PIPE_FORMAT_NONE, &mut t, &[1.0, 0.2, 0.0, 1.0]);
        assert_eq!(t, 0xF3);
        let mut v = [0.0; 4];
        unpack_rgba_float(VK_FORMAT_R4G4_UNORM_PACK8, PIPE_FORMAT_NONE, &t, &mut v);
        assert_eq!(v, [1.0, 0.2, 0.0, 1.0]);
    }

    #[test]
    fn z16_s8_writes_its_own_bytes() {
        let mut t = [0u8; 4];
        write_depth_stencil(PIPE_FORMAT_Z16_UNORM_S8_UINT, t.as_mut_ptr(), VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 1.0, 7);
        assert_eq!(t, [0xff, 0xff, 7, 0]);
    }
}

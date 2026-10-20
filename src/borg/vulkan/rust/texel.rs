// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Single texels in an image's format: clear colours, depth and stencil, RGBA in and out.

use crate::ffi::*;
use crate::image::z16_pack;

/// One texel's bytes, room for the widest format.
pub type Texel = [u8; 32];

pub fn unpack_rgba_float(vk: VkFormat, pf: pipe_format, texel: &Texel, out: &mut [f32; 4]) {
    if vk == VK_FORMAT_R4G4_UNORM_PACK8 {
        let t = texel[0];
        *out = [(t >> 4 & 0xF) as f32 / 15.0, (t & 0xF) as f32 / 15.0, 0.0, 1.0];
    } else {
        unsafe { util_format_unpack_rgba(pf, out.as_mut_ptr().cast(), texel.as_ptr().cast(), 1) };
    }
}

pub fn pack_rgba_float(vk: VkFormat, pf: pipe_format, texel: &mut Texel, v: &[f32; 4]) {
    if vk == VK_FORMAT_R4G4_UNORM_PACK8 {
        let q = |x: f32| (x.clamp(0.0, 1.0) * 15.0 + 0.5) as u8;
        texel[0] = q(v[0]) << 4 | q(v[1]);
    } else {
        unsafe { util_format_pack_rgba(pf, texel.as_mut_ptr().cast(), v.as_ptr().cast(), 1) };
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
            let mut t: Texel = [0; 32];
            pack_rgba_float(vk, pf, &mut t, &c.float32);
            pixel.copy_from_slice(&t[..16]);
        }
    }
}

/// Writes the depth and stencil aspects in `aspect` of one texel, keeping the other.
pub fn write_depth_stencil(pf: pipe_format, texel: &mut Texel, aspect: VkImageAspectFlags, depth: f32, stencil: u8) {
    let (d, s) = (aspect & VK_IMAGE_ASPECT_DEPTH_BIT != 0, aspect & VK_IMAGE_ASPECT_STENCIL_BIT != 0);
    if pf == PIPE_FORMAT_Z16_UNORM_S8_UINT {
        if d {
            texel[..2].copy_from_slice(&z16_pack(depth));
        }
        if s {
            texel[2] = stencil;
        }
        return;
    }
    if d {
        unsafe { util_format_pack_z_float(pf, texel.as_mut_ptr().cast(), &depth, 1) };
    }
    if s {
        unsafe { util_format_pack_s_8uint(pf, texel.as_mut_ptr().cast(), &stencil, 1) };
    }
}

/// One aspect of a combined depth/stencil format, but not both.
pub fn is_partial_ds_aspect(f: VkFormat, aspect: VkImageAspectFlags) -> bool {
    let ds = unsafe { vk_format_has_depth(f) && vk_format_has_stencil(f) };
    ds && aspect != (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r4g4_packs_by_hand() {
        let mut t = [0u8; 32];
        pack_rgba_float(VK_FORMAT_R4G4_UNORM_PACK8, PIPE_FORMAT_NONE, &mut t, &[1.0, 0.2, 0.0, 1.0]);
        assert_eq!(t[0], 0xF3);
        let mut v = [0.0; 4];
        unpack_rgba_float(VK_FORMAT_R4G4_UNORM_PACK8, PIPE_FORMAT_NONE, &t, &mut v);
        assert_eq!(v, [1.0, 0.2, 0.0, 1.0]);
    }

    #[test]
    fn z16_s8_writes_its_own_bytes() {
        let mut t = [0u8; 32];
        write_depth_stencil(PIPE_FORMAT_Z16_UNORM_S8_UINT, &mut t, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 1.0, 7);
        assert_eq!(t[..4], [0xff, 0xff, 7, 0]);
    }
}

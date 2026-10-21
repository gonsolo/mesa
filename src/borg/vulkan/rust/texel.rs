// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Single texels in an image's format: clear colours, depth and stencil, RGBA in and out.

use crate::ffi::*;
use crate::fmt;
use crate::image::z16_pack;

/// One texel's bytes, room for the widest format.
pub type Texel = [u8; 32];

pub fn unpack_rgba_float(vk: VkFormat, pf: pipe_format, texel: &Texel, out: &mut [f32; 4]) {
    if vk == VK_FORMAT_R4G4_UNORM_PACK8 {
        let t = texel[0];
        *out = [(t >> 4 & 0xF) as f32 / 15.0, (t & 0xF) as f32 / 15.0, 0.0, 1.0];
    } else {
        fmt::unpack_rgba(pf, texel, out);
    }
}

pub fn pack_rgba_float(vk: VkFormat, pf: pipe_format, texel: &mut Texel, v: &[f32; 4]) {
    if vk == VK_FORMAT_R4G4_UNORM_PACK8 {
        let q = |x: f32| (x.clamp(0.0, 1.0) * 15.0 + 0.5) as u8;
        texel[0] = q(v[0]) << 4 | q(v[1]);
    } else {
        fmt::pack_rgba(pf, texel, v);
    }
}

/// The type of the first channel of a scaled format.
fn scaled_type(pf: pipe_format) -> Option<util_format_type> {
    if !fmt::is_scaled(pf) {
        return None;
    }
    let ch = fmt::first_non_void_channel(pf);
    (ch >= 0).then(|| fmt::description(pf).map_or(0, |d| d.channel[ch as usize].type_()) as util_format_type)
}

/// Packs a clear colour: integers as they are, scaled formats from their integer value.
pub fn pack_clear_color(vk: VkFormat, pf: pipe_format, pixel: &mut [u8; 16], c: &VkClearColorValue) {
    let ch = fmt::first_non_void_channel(pf);
    let wide = ch >= 0 && fmt::is_pure_integer(pf) && fmt::description(pf).is_some_and(|d| d.channel[ch as usize].size() == 64);
    // Every member of the union is 16 bytes of plain data.
    let (u, i, f) = unsafe { (c.uint32, c.int32, c.float32) };
    if wide {
        let n = (fmt::blocksize(pf) as usize).min(16);
        let bytes: Vec<u8> = u.iter().flat_map(|w| w.to_ne_bytes()).collect();
        pixel[..n].copy_from_slice(&bytes[..n]);
    } else if fmt::is_pure_uint(pf) {
        fmt::pack_rgba(pf, pixel, &u);
    } else if fmt::is_pure_sint(pf) {
        fmt::pack_rgba(pf, pixel, &i);
    } else if scaled_type(pf) == Some(UTIL_FORMAT_TYPE_UNSIGNED) {
        fmt::pack_rgba(pf, pixel, &u.map(|x| x as f32));
    } else if scaled_type(pf) == Some(UTIL_FORMAT_TYPE_SIGNED) {
        fmt::pack_rgba(pf, pixel, &i.map(|x| x as f32));
    } else {
        let mut t: Texel = [0; 32];
        pack_rgba_float(vk, pf, &mut t, &f);
        pixel.copy_from_slice(&t[..16]);
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
        fmt::pack_z(pf, texel, depth);
    }
    if s {
        fmt::pack_s(pf, texel, stencil);
    }
}

/// One aspect of a combined depth/stencil format, but not both.
pub fn is_partial_ds_aspect(f: VkFormat, aspect: VkImageAspectFlags) -> bool {
    let ds = fmt::vk_has_depth(f) && fmt::vk_has_stencil(f);
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

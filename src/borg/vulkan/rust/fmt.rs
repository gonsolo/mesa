// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Mesa's format queries, safe: they read static tables. A Vulkan format the table does not
//! know maps to PIPE_FORMAT_NONE; a pipe format past the table has no description and reads
//! as false or 0 here.

use crate::ffi::*;

pub fn pipe(f: VkFormat) -> pipe_format {
    unsafe { vk_format_to_pipe_format(f) }
}

pub fn description(pf: pipe_format) -> Option<&'static util_format_description> {
    if pf >= PIPE_FORMAT_COUNT {
        return None;
    }
    unsafe { util_format_description(pf).as_ref() }
}

macro_rules! queries {
    ($($name:ident: $f:ident -> $t:ty),* $(,)?) => {
        $(pub fn $name(pf: pipe_format) -> $t {
            if description(pf).is_none() {
                return Default::default();
            }
            unsafe { $f(pf) as $t }
        })*
    };
}

queries!(
    blocksize: util_format_get_blocksize -> u32,
    blockwidth: util_format_get_blockwidth -> u32,
    blockheight: util_format_get_blockheight -> u32,
    blockdepth: util_format_get_blockdepth -> u32,
    is_pure_integer: util_format_is_pure_integer -> bool,
    is_pure_uint: util_format_is_pure_uint -> bool,
    is_pure_sint: util_format_is_pure_sint -> bool,
    is_scaled: util_format_is_scaled -> bool,
    is_compressed: util_format_is_compressed -> bool,
    is_srgb: util_format_is_srgb -> bool,
    is_depth_or_stencil: util_format_is_depth_or_stencil -> bool,
    has_alpha: util_format_has_alpha -> bool,
    first_non_void_channel: util_format_get_first_non_void_channel -> i32,
);

pub fn linear(pf: pipe_format) -> pipe_format {
    if description(pf).is_none() {
        return pf;
    }
    unsafe { util_format_linear(pf) }
}

pub fn vk_is_int(f: VkFormat) -> bool {
    unsafe { vk_format_is_int(f) }
}

pub fn vk_has_depth(f: VkFormat) -> bool {
    unsafe { vk_format_has_depth(f) }
}

pub fn vk_has_stencil(f: VkFormat) -> bool {
    unsafe { vk_format_has_stencil(f) }
}

pub fn vk_has_alpha(f: VkFormat) -> bool {
    unsafe { vk_format_has_alpha(f) }
}

pub fn aspect_format(f: VkFormat, aspect: VkImageAspectFlags) -> VkFormat {
    unsafe { vk_format_get_aspect_format(f, aspect) }
}

fn pack(pf: pipe_format) -> Option<&'static util_format_pack_description> {
    description(pf)?;
    unsafe { util_format_pack_description(pf).as_ref() }
}

fn unpack(pf: pipe_format) -> Option<&'static util_format_unpack_description> {
    description(pf)?;
    unsafe { util_format_unpack_description(pf).as_ref() }
}

/// RGBA as 16 bytes: four floats, or integers for pure-integer formats.
///
/// # Safety
/// The type is 16 bytes of plain data.
pub unsafe trait Rgba {}
unsafe impl Rgba for [f32; 4] {}
unsafe impl Rgba for [u32; 4] {}
unsafe impl Rgba for [i32; 4] {}
unsafe impl Rgba for [u8; 16] {}

// The conversions below touch one texel at the front of `t`, and do nothing when `t` is shorter
// than a block or the format has no such conversion.
fn fits(pf: pipe_format, n: usize) -> bool {
    blocksize(pf) != 0 && n >= blocksize(pf) as usize
}

pub fn unpack_rgba<R: Rgba>(pf: pipe_format, t: &[u8], out: &mut R) {
    if let Some(f) = unpack(pf).and_then(|d| d.unpack_rgba).filter(|_| fits(pf, t.len())) {
        unsafe { f(std::ptr::from_mut(out).cast(), t.as_ptr(), 1) }
    }
}

pub fn pack_rgba<R: Rgba>(pf: pipe_format, t: &mut [u8], v: &R) {
    let Some(d) = pack(pf).filter(|_| fits(pf, t.len())) else { return };
    let (dst, src) = (t.as_mut_ptr(), std::ptr::from_ref(v));
    if is_pure_uint(pf) {
        d.pack_rgba_uint.inspect(|f| unsafe { f(dst, 0, src.cast(), 0, 1, 1) });
    } else if is_pure_sint(pf) {
        d.pack_rgba_sint.inspect(|f| unsafe { f(dst, 0, src.cast(), 0, 1, 1) });
    } else {
        d.pack_rgba_float.inspect(|f| unsafe { f(dst, 0, src.cast(), 0, 1, 1) });
    }
}

pub fn unpack_z(pf: pipe_format, t: &[u8]) -> f32 {
    let mut z = 0.0;
    if let Some(f) = unpack(pf).and_then(|d| d.unpack_z_float).filter(|_| fits(pf, t.len())) {
        unsafe { f(&mut z, 0, t.as_ptr(), 0, 1, 1) }
    }
    z
}

pub fn pack_z(pf: pipe_format, t: &mut [u8], z: f32) {
    if let Some(f) = pack(pf).and_then(|d| d.pack_z_float).filter(|_| fits(pf, t.len())) {
        unsafe { f(t.as_mut_ptr(), 0, &z, 0, 1, 1) }
    }
}

pub fn unpack_s(pf: pipe_format, t: &[u8]) -> u8 {
    let mut s = 0;
    if let Some(f) = unpack(pf).and_then(|d| d.unpack_s_8uint).filter(|_| fits(pf, t.len())) {
        unsafe { f(&mut s, 0, t.as_ptr(), 0, 1, 1) }
    }
    s
}

pub fn pack_s(pf: pipe_format, t: &mut [u8], s: u8) {
    if let Some(f) = pack(pf).and_then(|d| d.pack_s_8uint).filter(|_| fits(pf, t.len())) {
        unsafe { f(t.as_mut_ptr(), 0, &s, 0, 1, 1) }
    }
}

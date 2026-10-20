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

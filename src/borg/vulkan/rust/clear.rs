// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Clears of images and of attachments, written by the host into the image memory.

use crate::cmd_buffer::cmd;
use crate::entrypoint::{entrypoint, slice, Handle};
use crate::ffi::*;
use crate::image::{blocks, shape};
use crate::texel::{pack_clear_color, write_depth_stencil};
use borg_layout::image::Blocks;

fn level_count(image: &vk_image, r: &VkImageSubresourceRange) -> u32 {
    if r.levelCount == VK_REMAINING_MIP_LEVELS as u32 { image.mip_levels.wrapping_sub(r.baseMipLevel) } else { r.levelCount }
}

fn layer_count(image: &vk_image, r: &VkImageSubresourceRange) -> u32 {
    if r.layerCount == VK_REMAINING_ARRAY_LAYERS as u32 { image.array_layers.wrapping_sub(r.baseArrayLayer) } else { r.layerCount }
}

/// The mapped image and its blocks, when it has memory and a block is one texel deep.
fn mapped(image: &borgvk_image) -> Option<(*mut u8, pipe_format, Blocks)> {
    let mem = unsafe { image.mem.as_ref() }.filter(|m| !m.map.is_null())?;
    let b = blocks(image.vk.format);
    (b.bd <= 1).then(|| (unsafe { mem.map.cast::<u8>().add(image.offset as usize) }, unsafe { vk_format_to_pipe_format(image.vk.format) }, b))
}

/// Calls `f` with the start and texel count of every sample plane of every level and layer in
/// the ranges.
fn for_each_plane(image: &borgvk_image, ranges: &[VkImageSubresourceRange], b: Blocks, mut f: impl FnMut(u64, u64)) {
    let dim = |e: u32, level: u32, block: u32| e.checked_shr(level).unwrap_or(0).max(1).div_ceil(block) as u64;
    let plane = shape(image).sample_plane_size(b);
    for r in ranges {
        for level in (0..level_count(&image.vk, r)).map(|l| r.baseMipLevel + l) {
            let (h, d) = (dim(image.vk.extent.height, level, b.bh), dim(image.vk.extent.depth, level, b.bd));
            for layer in (0..layer_count(&image.vk, r)).map(|l| r.baseArrayLayer + l) {
                let (off, stride) = shape(image).level_offset(level, layer, b);
                for s in 0..image.vk.samples.max(1) as u64 {
                    f(off + s * plane, stride as u64 * h * d);
                }
            }
        }
    }
}

entrypoint! {
    fn borgvk_CmdClearColorImage(_cmd: VkCommandBuffer, image: VkImage, _layout: VkImageLayout, color: *const VkClearColorValue, count: u32, ranges: *const VkImageSubresourceRange) {
        let Some(image) = image.get() else { return };
        let Some((base, pf, b)) = mapped(image) else { return };
        let mut pixel = [0u8; 16];
        pack_clear_color(image.vk.format, pf, &mut pixel, &*color);
        let bs = b.bs as usize;
        for_each_plane(image, slice(ranges, count), b, |off, n| {
            let dst = std::slice::from_raw_parts_mut(base.add(off as usize), n as usize * bs);
            for t in dst.chunks_exact_mut(bs) {
                t.copy_from_slice(&pixel[..bs]);
            }
        });
    }
}

entrypoint! {
    fn borgvk_CmdClearDepthStencilImage(_cmd: VkCommandBuffer, image: VkImage, _layout: VkImageLayout, ds: *const VkClearDepthStencilValue, count: u32, ranges: *const VkImageSubresourceRange) {
        let Some(image) = image.get() else { return };
        let Some((base, pf, b)) = mapped(image) else { return };
        let ds = &*ds;
        for r in slice(ranges, count) {
            for_each_plane(image, std::slice::from_ref(r), b, |off, n| {
                for t in 0..n {
                    write_depth_stencil(pf, base.add((off + t * b.bs as u64) as usize), r.aspectMask, ds.depth, ds.stencil as u8);
                }
            });
        }
    }
}

/// Clears `rect` of the attachment `view` in `aspect`, in layers from `base_layer` on.
pub fn clear_rect(view: *mut vk_image_view, aspect: VkImageAspectFlags, value: &VkClearValue, rect: VkRect2D, base_layer: u32, layers: u32) {
    let Some(view) = (unsafe { view.as_ref() }) else { return };
    let image = unsafe { &*view.image.cast::<borgvk_image>() };
    let Some((base, pf, b)) = mapped(image) else { return };
    let mut pixel = [0u8; 16];
    let ds = unsafe { value.depthStencil };
    if aspect & VK_IMAGE_ASPECT_COLOR_BIT != 0 {
        pack_clear_color(image.vk.format, pf, &mut pixel, unsafe { &value.color });
    }
    let (x0, y0, w, h) = (rect.offset.x as u32, rect.offset.y as u32, rect.extent.width, rect.extent.height);
    let bs = b.bs as u64;
    let plane = shape(image).sample_plane_size(b);
    for l in 0..layers {
        let (off, stride) = shape(image).level_offset(view.base_mip_level, view.base_array_layer + base_layer + l, b);
        for s in 0..image.vk.samples.max(1) as u64 {
            for y in y0..y0.wrapping_add(h) {
                for x in x0..x0.wrapping_add(w) {
                    let texel = unsafe { base.add((off + s * plane + (y as u64 * stride as u64 + x as u64) * bs) as usize) };
                    if aspect & VK_IMAGE_ASPECT_COLOR_BIT != 0 {
                        unsafe { std::ptr::copy_nonoverlapping(pixel.as_ptr(), texel, bs as usize) };
                    } else {
                        write_depth_stencil(pf, texel, aspect, ds.depth, ds.stencil as u8);
                    }
                }
            }
        }
    }
}

/// # Safety
/// `view` is null or valid, `value` is valid.
#[no_mangle]
pub unsafe extern "C" fn borgvk_clear_attachment_rect(
    view: *mut vk_image_view,
    aspect: VkImageAspectFlags,
    value: *const VkClearValue,
    rect: VkRect2D,
    base_layer: u32,
    layers: u32,
) {
    clear_rect(view, aspect, unsafe { &*value }, rect, base_layer, layers);
}

entrypoint! {
    fn borgvk_CmdClearAttachments(command_buffer: VkCommandBuffer, natt: u32, atts: *const VkClearAttachment, nrects: u32, rects: *const VkClearRect) {
        let c = cmd(command_buffer.ptr());
        crate::draw::flush_draws(c); // the draws so far come first
        if !c.in_rendering {
            return;
        }
        let rects = slice(rects, nrects);
        for att in slice(atts, natt) {
            let clear = |view, aspect| {
                for r in rects {
                    clear_rect(view, aspect, &att.clearValue, r.rect, r.baseArrayLayer, r.layerCount);
                }
            };
            if att.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT != 0 {
                if att.colorAttachment < c.color_attachment_count {
                    clear(c.color_views[att.colorAttachment as usize], VK_IMAGE_ASPECT_COLOR_BIT);
                }
                continue;
            }
            for r in rects {
                if att.aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT != 0 {
                    clear_rect(c.depth_view, VK_IMAGE_ASPECT_DEPTH_BIT, &att.clearValue, r.rect, r.baseArrayLayer, r.layerCount);
                }
                if att.aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT != 0 {
                    clear_rect(c.stencil_view, VK_IMAGE_ASPECT_STENCIL_BIT, &att.clearValue, r.rect, r.baseArrayLayer, r.layerCount);
                }
            }
        }
    }
}

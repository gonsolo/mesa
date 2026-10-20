// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Clears of images and of attachments, written by the host into the image memory.

use crate::cmd_buffer::CommandBuffer;
use crate::entrypoint::{entrypoint, pointee, slice};
use crate::ffi::*;
use crate::image::{blocks, shape, Image};
use crate::memory::Host;
use crate::texel::{pack_clear_color, write_depth_stencil, Texel};
use borg_layout::image::Blocks;

fn level_count(image: &vk_image, r: &VkImageSubresourceRange) -> u32 {
    if r.levelCount == VK_REMAINING_MIP_LEVELS as u32 { image.mip_levels.wrapping_sub(r.baseMipLevel) } else { r.levelCount }
}

fn layer_count(image: &vk_image, r: &VkImageSubresourceRange) -> u32 {
    if r.layerCount == VK_REMAINING_ARRAY_LAYERS as u32 { image.array_layers.wrapping_sub(r.baseArrayLayer) } else { r.layerCount }
}

/// The mapped image and its blocks, when it has memory and a block is one texel deep.
fn mapped(image: &Image) -> Option<(Host, pipe_format, Blocks)> {
    let b = blocks(image.vk.format);
    (b.bd <= 1).then_some(())?;
    Some((image.host()?, unsafe { vk_format_to_pipe_format(image.vk.format) }, b))
}

/// Calls `f` with the start and texel count of every sample plane of every level and layer in
/// the ranges.
fn for_each_plane(image: &Image, ranges: &[VkImageSubresourceRange], b: Blocks, mut f: impl FnMut(u64, u64)) {
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

pointee!(VkClearColorValue, VkClearDepthStencilValue);

entrypoint! {
    fn borgvk_CmdClearColorImage(_cmd: VkCommandBuffer, image: &Image, _layout: VkImageLayout, color: &VkClearColorValue, count: u32, ranges: *const VkImageSubresourceRange) {
        let Some((host, pf, b)) = mapped(image) else { return };
        let mut pixel = [0u8; 16];
        pack_clear_color(image.vk.format, pf, &mut pixel, color);
        let bs = b.bs as usize;
        for_each_plane(image, unsafe { slice(ranges, count) }, b, |off, n| {
            host.fill(off as usize, &pixel[..bs], n as usize);
        });
    }
}

/// Writes the depth and stencil in `aspect` of the texel at `off`, keeping the other aspect.
fn write_ds(host: &Host, off: usize, bs: usize, pf: pipe_format, aspect: VkImageAspectFlags, ds: &VkClearDepthStencilValue) {
    let mut t: Texel = [0; 32];
    if host.read(off, &mut t[..bs]) {
        write_depth_stencil(pf, &mut t, aspect, ds.depth, ds.stencil as u8);
        host.write(off, &t[..bs]);
    }
}

entrypoint! {
    fn borgvk_CmdClearDepthStencilImage(_cmd: VkCommandBuffer, image: &Image, _layout: VkImageLayout, ds: &VkClearDepthStencilValue, count: u32, ranges: *const VkImageSubresourceRange) {
        let Some((host, pf, b)) = mapped(image) else { return };
        let bs = b.bs as usize;
        for r in unsafe { slice(ranges, count) } {
            for_each_plane(image, std::slice::from_ref(r), b, |off, n| {
                for t in 0..n as usize {
                    write_ds(&host, off as usize + t * bs, bs, pf, r.aspectMask, ds);
                }
            });
        }
    }
}

/// Clears `rect` of the attachment `view` in `aspect`, in layers from `base_layer` on.
pub fn clear_rect(view: *mut vk_image_view, aspect: VkImageAspectFlags, value: &VkClearValue, rect: VkRect2D, base_layer: u32, layers: u32) {
    let Some(view) = (unsafe { view.as_ref() }) else { return };
    let image = unsafe { &*view.image.cast::<Image>() };
    let Some((host, pf, b)) = mapped(image) else { return };
    let mut pixel = [0u8; 16];
    let (color, ds) = (aspect & VK_IMAGE_ASPECT_COLOR_BIT != 0, unsafe { value.depthStencil });
    if color {
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
                    let t = (off + s * plane + (y as u64 * stride as u64 + x as u64) * bs) as usize;
                    if color {
                        host.write(t, &pixel[..bs as usize]);
                    } else {
                        write_ds(&host, t, bs as usize, pf, aspect, &ds);
                    }
                }
            }
        }
    }
}

entrypoint! {
    fn borgvk_CmdClearAttachments(c: &mut CommandBuffer, natt: u32, atts: *const VkClearAttachment, nrects: u32, rects: *const VkClearRect) {
        crate::draw::flush_draws(c); // the draws so far come first
        if !c.in_rendering {
            return;
        }
        let rects = unsafe { slice(rects, nrects) };
        for att in unsafe { slice(atts, natt) } {
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

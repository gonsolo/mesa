// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Clears of images and of attachments, written by the host into the image memory.

use crate::cmd_buffer::CommandBuffer;
use crate::entrypoint::{entrypoint, pointee};
use crate::ffi::*;
use crate::fmt;
use crate::image::{blocks, Image, ViewImage};
use crate::texels::Texels;
use crate::vk::ClearValue;
use crate::texel::{pack_clear_color, write_depth_stencil, Texel};

fn level_count(image: &vk_image, r: &VkImageSubresourceRange) -> u32 {
    if r.levelCount == VK_REMAINING_MIP_LEVELS as u32 { image.mip_levels.wrapping_sub(r.baseMipLevel) } else { r.levelCount }
}

fn layer_count(image: &vk_image, r: &VkImageSubresourceRange) -> u32 {
    if r.layerCount == VK_REMAINING_ARRAY_LAYERS as u32 { image.array_layers.wrapping_sub(r.baseArrayLayer) } else { r.layerCount }
}

/// The image's texels and format, when it has memory and a block is one texel deep.
fn mapped(image: &Image) -> Option<(Texels<'_>, pipe_format)> {
    (blocks(image.vk.format).bd <= 1).then_some(())?;
    Some((Texels::of(image)?, fmt::pipe(image.vk.format)))
}

/// Calls `f` with every texel (level, layer, block, sample) of the ranges.
fn for_each_texel(image: &Image, ranges: &[VkImageSubresourceRange], mut f: impl FnMut(u32, u32, [u32; 3], u32)) {
    let b = blocks(image.vk.format);
    let dim = |e: u32, level: u32, block: u32| e.checked_shr(level).unwrap_or(0).max(1).div_ceil(block);
    for r in ranges {
        for level in (0..level_count(&image.vk, r)).map(|l| r.baseMipLevel + l) {
            let e = image.vk.extent;
            let (w, h, d) = (dim(e.width, level, b.bw), dim(e.height, level, b.bh), dim(e.depth, level, b.bd));
            for layer in (0..layer_count(&image.vk, r)).map(|l| r.baseArrayLayer + l) {
                for s in 0..image.vk.samples.max(1) {
                    for z in 0..d {
                        for y in 0..h {
                            for x in 0..w {
                                f(level, layer, [x, y, z], s);
                            }
                        }
                    }
                }
            }
        }
    }
}

pointee!(VkClearColorValue, VkClearDepthStencilValue);

entrypoint! {
    fn borgvk_CmdClearColorImage(_cmd: VkCommandBuffer, image: &Image, _layout: VkImageLayout, color: &VkClearColorValue, _count: u32, ranges: &[VkImageSubresourceRange]) {
        let Some((t, pf)) = mapped(image) else { return };
        let mut pixel = [0u8; 16];
        pack_clear_color(image.vk.format, pf, &mut pixel, color);
        for_each_texel(image, ranges, |level, layer, xyz, s| {
            t.write(level, layer, xyz, s, &pixel);
        });
    }
}

/// Writes the depth and stencil in `aspect` of a texel, keeping the other aspect.
fn write_ds(t: &Texels, at: (u32, u32, [u32; 3], u32), pf: pipe_format, aspect: VkImageAspectFlags, ds: &VkClearDepthStencilValue) {
    let mut v: Texel = [0; 32];
    if t.read(at.0, at.1, at.2, at.3, &mut v) {
        write_depth_stencil(pf, &mut v, aspect, ds.depth, ds.stencil as u8);
        t.write(at.0, at.1, at.2, at.3, &v);
    }
}

entrypoint! {
    fn borgvk_CmdClearDepthStencilImage(_cmd: VkCommandBuffer, image: &Image, _layout: VkImageLayout, ds: &VkClearDepthStencilValue, _count: u32, ranges: &[VkImageSubresourceRange]) {
        let Some((t, pf)) = mapped(image) else { return };
        for r in ranges {
            for_each_texel(image, std::slice::from_ref(r), |level, layer, xyz, s| write_ds(&t, (level, layer, xyz, s), pf, r.aspectMask, ds));
        }
    }
}

/// Clears `rect` of the attachment `view` in `aspect`, in layers from `base_layer` on.
pub fn clear_rect(view: &vk_image_view, aspect: VkImageAspectFlags, value: &VkClearValue, rect: VkRect2D, base_layer: u32, layers: u32) {
    let Some(image) = view.image() else { return };
    let Some((t, pf)) = mapped(image) else { return };
    let mut pixel = [0u8; 16];
    let (color, ds) = (aspect & VK_IMAGE_ASPECT_COLOR_BIT != 0, value.depth_stencil());
    if color {
        // In the view's format, which may differ from the image's (mutable formats).
        pack_clear_color(view.format, fmt::pipe(view.format), &mut pixel, value.color());
    }
    let (x0, y0, w, h) = (rect.offset.x as u32, rect.offset.y as u32, rect.extent.width, rect.extent.height);
    let is3d = image.vk.image_type == VK_IMAGE_TYPE_3D;
    for l in 0..layers {
        // A layer of a 3D image's view is a slice.
        let layer = view.base_array_layer + base_layer + l;
        let (layer, z) = if is3d { (0, layer) } else { (layer, 0) };
        for s in 0..image.vk.samples.max(1) {
            for y in y0..y0.wrapping_add(h) {
                for x in x0..x0.wrapping_add(w) {
                    if color {
                        t.write(view.base_mip_level, layer, [x, y, z], s, &pixel);
                    } else {
                        write_ds(&t, (view.base_mip_level, layer, [x, y, z], s), pf, aspect, &ds);
                    }
                }
            }
        }
    }
}

entrypoint! {
    fn borgvk_CmdClearAttachments(c: &mut CommandBuffer, _natt: u32, atts: &[VkClearAttachment], _nrects: u32, rects: &[VkClearRect]) {
        crate::draw::flush_draws(c); // the draws so far come first
        if !c.in_rendering {
            return;
        }
        for att in atts {
            let has = |a| att.aspectMask & a != 0;
            if has(VK_IMAGE_ASPECT_COLOR_BIT) {
                if let Some(view) = (att.colorAttachment < c.color_attachment_count).then(|| c.color_view(att.colorAttachment as usize)).flatten() {
                    for r in rects {
                        clear_rect(view, VK_IMAGE_ASPECT_COLOR_BIT, &att.clearValue, r.rect, r.baseArrayLayer, r.layerCount);
                    }
                }
                continue;
            }
            for r in rects {
                if let Some(view) = c.depth_view().filter(|_| has(VK_IMAGE_ASPECT_DEPTH_BIT)) {
                    clear_rect(view, VK_IMAGE_ASPECT_DEPTH_BIT, &att.clearValue, r.rect, r.baseArrayLayer, r.layerCount);
                }
                if let Some(view) = c.stencil_view().filter(|_| has(VK_IMAGE_ASPECT_STENCIL_BIT)) {
                    clear_rect(view, VK_IMAGE_ASPECT_STENCIL_BIT, &att.clearValue, r.rect, r.baseArrayLayer, r.layerCount);
                }
            }
        }
    }
}

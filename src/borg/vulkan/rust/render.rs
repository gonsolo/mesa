// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Dynamic rendering: the attachments, their load clears and resolves. Render passes are the
//! runtime's, which call these.

use crate::clear::clear_rect;
use crate::cmd_buffer::{cmd, CommandBuffer};
use crate::copy::resolve_region;
use crate::entrypoint::{entrypoint, Handle};
use crate::ffi::*;
use crate::image::Image;
use crate::image::{blocks, shape};
use borg_layout::image::Blocks;
use std::ptr::null_mut;

/// A single-level, single-layer copy of an attachment's image, for rendering into one level.
pub struct Target {
    next: *mut Target,
    orig: *const vk_image_view,
    image: Image,
    view: vk_image_view,
}

pub fn free_targets(c: &mut CommandBuffer) {
    while !c.targets.is_null() {
        let t = unsafe { Box::from_raw(c.targets) };
        c.targets = t.next;
    }
}

/// A view of a mip level or layer above 0 renders as the single-level, single-layer image it
/// is: a copy of the image placed at that level, kept with the command buffer.
fn target_view(c: &mut CommandBuffer, view: *mut vk_image_view) -> *mut vk_image_view {
    let Some(v) = (unsafe { view.as_ref() }) else { return view };
    if v.image.is_null() || (v.base_mip_level == 0 && v.base_array_layer == 0) {
        return view;
    }
    let img = unsafe { &*v.image.cast::<Image>() };
    if img.vk.image_type == VK_IMAGE_TYPE_3D || v.layer_count != 1 || img.mem.is_null() {
        return view;
    }
    let mut t = c.targets;
    while let Some(target) = unsafe { t.as_mut() } {
        if target.orig == view.cast_const() {
            return &mut target.view;
        }
        t = target.next;
    }
    let b = blocks(img.vk.format);
    if b.bw != 1 || b.bh != 1 {
        return view;
    }
    let level = v.base_mip_level;
    let off = shape(img).level_offset(level, v.base_array_layer, Blocks { bw: 1, bh: 1, bd: 1, ..b }).0;
    let mut t = Box::new(Target { next: c.targets, orig: view, image: unsafe { std::ptr::read(img) }, view: unsafe { std::ptr::read(v) } });
    let i = &mut t.image;
    i.offset += off;
    i.size = img.size.saturating_sub(off);
    i.vk.extent.width = img.vk.extent.width.checked_shr(level).unwrap_or(0).max(1);
    i.vk.extent.height = img.vk.extent.height.checked_shr(level).unwrap_or(0).max(1);
    (i.vk.mip_levels, i.vk.array_layers) = (1, 1);
    t.view.image = &mut t.image.vk;
    (t.view.base_mip_level, t.view.base_array_layer) = (0, 0);
    c.targets = Box::into_raw(t);
    unsafe { &mut (*c.targets).view }
}

/// The depth or stencil attachment's view, cleared on load.
fn depth_stencil(c: &mut CommandBuffer, att: *const VkRenderingAttachmentInfo, aspect: VkImageAspectFlags, layers: u32) -> *mut vk_image_view {
    let Some(att) = (unsafe { att.as_ref() }).filter(|a| !a.imageView.is_null()) else { return null_mut() };
    let view = att.imageView.ptr();
    let target = target_view(c, view);
    if att.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR {
        clear_rect(view, aspect, &att.clearValue, c.render_area, 0, layers);
    }
    target
}

entrypoint! {
    unsafe fn borgvk_CmdBeginRendering(command_buffer: VkCommandBuffer, info: *const VkRenderingInfo) {
        let (c, ri) = (cmd(command_buffer.ptr()), &*info);
        c.in_rendering = true;
        c.render_area = ri.renderArea;
        c.layer_count = ri.layerCount.max(1);
        c.color_attachment_count = ri.colorAttachmentCount.min(BORGVK_MAX_COLOR_ATTACHMENTS);
        // A subpass without colour must not draw into the previous one's attachment.
        (c.color_views, c.color_resolve_views) = (Default::default(), Default::default());
        let layers = c.layer_count;
        for i in 0..c.color_attachment_count as usize {
            let att = &*ri.pColorAttachments.add(i);
            let view = att.imageView.ptr();
            c.color_views[i] = target_view(c, view);
            c.color_resolve_views[i] = att.resolveImageView.ptr();
            if !view.is_null() && att.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR {
                clear_rect(view, VK_IMAGE_ASPECT_COLOR_BIT, &att.clearValue, c.render_area, 0, layers);
                if i == 0 {
                    c.has_clear = true;
                    c.clear_color = att.clearValue.color.float32;
                }
            }
        }
        c.depth_view = depth_stencil(c, ri.pDepthAttachment, VK_IMAGE_ASPECT_DEPTH_BIT, layers);
        c.stencil_view = depth_stencil(c, ri.pStencilAttachment, VK_IMAGE_ASPECT_STENCIL_BIT, layers);
    }
}

fn layers(view: &vk_image_view, count: u32) -> VkImageSubresourceLayers {
    VkImageSubresourceLayers {
        aspectMask: VK_IMAGE_ASPECT_COLOR_BIT,
        mipLevel: view.base_mip_level,
        baseArrayLayer: view.base_array_layer,
        layerCount: count,
    }
}

entrypoint! {
    unsafe fn borgvk_CmdEndRendering2EXT(command_buffer: VkCommandBuffer, _info: *const VkRenderingEndInfoEXT) {
        let c = cmd(command_buffer.ptr());
        crate::draw::flush_draws(c);
        let ra = c.render_area;
        let at = VkOffset3D { x: ra.offset.x, y: ra.offset.y, z: 0 };
        for i in 0..c.color_attachment_count as usize {
            let (Some(v), Some(r)) = (c.color_views[i].as_ref(), c.color_resolve_views[i].as_ref()) else { continue };
            let region = VkImageResolve {
                srcSubresource: layers(v, c.layer_count),
                srcOffset: at,
                dstSubresource: layers(r, c.layer_count),
                dstOffset: at,
                extent: VkExtent3D { width: ra.extent.width, height: ra.extent.height, depth: 1 },
            };
            resolve_region(&*v.image.cast(), &*r.image.cast(), &region);
        }
        c.in_rendering = false;
        c.color_attachment_count = 0;
        (c.depth_view, c.stencil_view) = (null_mut(), null_mut());
    }
}

// By hand: bindgen leaves out vk_common_entrypoints.h's hidden declarations.
extern "C" {
    fn vk_common_CmdEndRendering(cb: VkCommandBuffer);
    fn vk_common_CmdBeginRenderPass(cb: VkCommandBuffer, info: *const VkRenderPassBeginInfo, contents: VkSubpassContents);
    fn vk_common_CmdBeginRenderPass2(cb: VkCommandBuffer, info: *const VkRenderPassBeginInfo, subpass: *const VkSubpassBeginInfo);
    fn vk_common_CmdNextSubpass(cb: VkCommandBuffer, contents: VkSubpassContents);
    fn vk_common_CmdNextSubpass2(cb: VkCommandBuffer, begin: *const VkSubpassBeginInfo, end: *const VkSubpassEndInfo);
    fn vk_common_CmdEndRenderPass(cb: VkCommandBuffer);
    fn vk_common_CmdEndRenderPass2(cb: VkCommandBuffer, end: *const VkSubpassEndInfo);
}

// Recorded commands replay through the device's cmd_dispatch, which holds only borgvk's entry
// points; these forward to the runtime's render passes.
entrypoint! {
    unsafe fn borgvk_CmdEndRendering(command_buffer: VkCommandBuffer) {
        vk_common_CmdEndRendering(command_buffer);
    }
}

entrypoint! {
    unsafe fn borgvk_CmdBeginRenderPass(command_buffer: VkCommandBuffer, info: *const VkRenderPassBeginInfo, contents: VkSubpassContents) {
        vk_common_CmdBeginRenderPass(command_buffer, info, contents);
    }
}

entrypoint! {
    unsafe fn borgvk_CmdBeginRenderPass2(command_buffer: VkCommandBuffer, info: *const VkRenderPassBeginInfo, subpass: *const VkSubpassBeginInfo) {
        vk_common_CmdBeginRenderPass2(command_buffer, info, subpass);
    }
}

entrypoint! {
    unsafe fn borgvk_CmdNextSubpass(command_buffer: VkCommandBuffer, contents: VkSubpassContents) {
        vk_common_CmdNextSubpass(command_buffer, contents);
    }
}

entrypoint! {
    unsafe fn borgvk_CmdNextSubpass2(command_buffer: VkCommandBuffer, begin: *const VkSubpassBeginInfo, end: *const VkSubpassEndInfo) {
        vk_common_CmdNextSubpass2(command_buffer, begin, end);
    }
}

entrypoint! {
    unsafe fn borgvk_CmdEndRenderPass(command_buffer: VkCommandBuffer) {
        vk_common_CmdEndRenderPass(command_buffer);
    }
}

entrypoint! {
    unsafe fn borgvk_CmdEndRenderPass2(command_buffer: VkCommandBuffer, end: *const VkSubpassEndInfo) {
        vk_common_CmdEndRenderPass2(command_buffer, end);
    }
}

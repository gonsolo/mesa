// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Dynamic rendering: the attachments, their load clears and resolves. Render passes are the
//! runtime's, which call these.

use crate::clear::clear_rect;
use crate::cmd_buffer::CommandBuffer;
use crate::copy::resolve_region;
use crate::entrypoint::{entrypoint, pointee, Handle};
use crate::ffi::*;
use crate::vk::*;
use crate::image::Image;
use crate::image::{blocks, shape, ViewImage};
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
        let t = c.targets;
        c.targets = unsafe { (*t).next };
        unsafe { vk_free(&(*c.vk.pool).alloc, t.cast()) };
    }
}

/// A view of a mip level or layer above 0 renders as the single-level, single-layer image it
/// is: a copy of the image placed at that level, kept with the command buffer.
fn target_view(c: &mut CommandBuffer, view: Option<&vk_image_view>) -> *mut vk_image_view {
    let Some(v) = view else { return null_mut() };
    let view = std::ptr::from_ref(v).cast_mut();
    let Some(img) = v.image().filter(|_| v.base_mip_level != 0 || v.base_array_layer != 0) else { return view };
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
    let alloc = unsafe { &(*c.vk.pool).alloc };
    let p = unsafe { vk_alloc(alloc, size_of::<Target>(), align_of::<Target>(), VK_SYSTEM_ALLOCATION_SCOPE_OBJECT) }.cast::<Target>();
    if p.is_null() {
        unsafe { __vk_command_buffer_set_error(&mut c.vk, VK_ERROR_OUT_OF_HOST_MEMORY, c"render.rs".as_ptr(), line!() as i32) };
        return view;
    }
    unsafe { p.write(Target { next: c.targets, orig: view, image: std::ptr::read(img), view: std::ptr::read(v) }) };
    let t = unsafe { &mut *p };
    let i = &mut t.image;
    i.offset += off;
    i.size = img.size.saturating_sub(off);
    i.vk.extent.width = img.vk.extent.width.checked_shr(level).unwrap_or(0).max(1);
    i.vk.extent.height = img.vk.extent.height.checked_shr(level).unwrap_or(0).max(1);
    (i.vk.mip_levels, i.vk.array_layers) = (1, 1);
    t.view.image = &mut t.image.vk;
    (t.view.base_mip_level, t.view.base_array_layer) = (0, 0);
    c.targets = p;
    &mut t.view
}

/// The depth or stencil attachment's view, cleared on load.
fn depth_stencil(c: &mut CommandBuffer, att: Option<&VkRenderingAttachmentInfo>, aspect: VkImageAspectFlags, layers: u32) -> *mut vk_image_view {
    let Some(att) = att else { return null_mut() };
    let view = unsafe { att.imageView.get_ref() };
    let target = target_view(c, view);
    if let Some(v) = view.filter(|_| att.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
        clear_rect(v, aspect, &att.clearValue, c.render_area, 0, layers);
    }
    target
}

pointee!(VkRenderingInfo);

entrypoint! {
    fn borgvk_CmdBeginRendering(c: &mut CommandBuffer, ri: &VkRenderingInfo) {
        c.in_rendering = true;
        c.render_area = ri.renderArea;
        c.layer_count = ri.layerCount.max(1);
        c.color_attachment_count = ri.colorAttachmentCount.min(BORGVK_MAX_COLOR_ATTACHMENTS);
        // A subpass without colour must not draw into the previous one's attachment.
        (c.color_views, c.color_resolve_views) = (Default::default(), Default::default());
        let layers = c.layer_count;
        let atts = &ri.color_attachments()[..c.color_attachment_count as usize];
        for (i, att) in atts.iter().enumerate() {
            let view = unsafe { att.imageView.get_ref() };
            c.color_views[i] = target_view(c, view);
            c.color_resolve_views[i] = att.resolveImageView.ptr();
            if let Some(view) = view.filter(|_| att.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
                clear_rect(view, VK_IMAGE_ASPECT_COLOR_BIT, &att.clearValue, c.render_area, 0, layers);
                if i == 0 {
                    c.has_clear = true;
                    c.clear_color = att.clearValue.color().float32();
                }
            }
        }
        c.depth_view = depth_stencil(c, ri.depth_attachment(), VK_IMAGE_ASPECT_DEPTH_BIT, layers);
        c.stencil_view = depth_stencil(c, ri.stencil_attachment(), VK_IMAGE_ASPECT_STENCIL_BIT, layers);
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
    fn borgvk_CmdEndRendering2EXT(c: &mut CommandBuffer, _info: *const VkRenderingEndInfoEXT) {
        crate::draw::flush_draws(c);
        let ra = c.render_area;
        let at = VkOffset3D { x: ra.offset.x, y: ra.offset.y, z: 0 };
        for i in 0..c.color_attachment_count as usize {
            let (Some(v), Some(r)) = (c.color_view(i), c.resolve_view(i)) else { continue };
            let (Some(src), Some(dst)) = (v.image(), r.image()) else { continue };
            let region = VkImageResolve {
                srcSubresource: layers(v, c.layer_count),
                srcOffset: at,
                dstSubresource: layers(r, c.layer_count),
                dstOffset: at,
                extent: VkExtent3D { width: ra.extent.width, height: ra.extent.height, depth: 1 },
            };
            resolve_region(src, dst, &region);
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

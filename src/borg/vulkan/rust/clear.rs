// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Clears of images and of attachments, written by the host into the image memory.

use crate::cmd_buffer::{CommandBuffer, Step};
use crate::entrypoint::{entrypoint, pointee};
use crate::ffi::*;
use crate::fmt;
use crate::image::{blocks, Image, ViewImage};
use crate::texels::Texels;
use crate::vk::{ClearColorValue, ClearValue};
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
            let one = |x: u32, y: u32| {
                if color {
                    t.write(view.base_mip_level, layer, [x, y, z], s, &pixel);
                } else {
                    write_ds(&t, (view.base_mip_level, layer, [x, y, z], s), pf, aspect, &ds);
                }
            };
            if w == 0 || h == 0 {
                continue;
            }
            one(x0, y0);
            if t.fill_from_first(view.base_mip_level, layer, [x0, y0, z], [w, h], s, aspect) {
                continue;
            }
            for y in y0..y0.wrapping_add(h) {
                for x in x0..x0.wrapping_add(w) {
                    one(x, y);
                }
            }
        }
    }
}

entrypoint! {
    fn borgvk_CmdClearAttachments(c: &mut CommandBuffer, _natt: u32, atts: &[VkClearAttachment], _nrects: u32, rects: &[VkClearRect]) {
        if !crate::record::gate(c) {
            return;
        }
        crate::draw::flush_draws(c); // the draws so far come first
        if !c.in_rendering {
            return;
        }
        let rec = c.recording();
        let clear = |c: &mut CommandBuffer, view: &vk_image_view, aspect, value: &VkClearValue, r: &VkClearRect| {
            if rec {
                let step = Step::Clear { view, aspect, value: *value, rect: r.rect, layer: r.baseArrayLayer, layers: r.layerCount };
                let r = c.rec();
                r.steps.push(crate::cmd_buffer::Placed { at: r.cur, first: r.cur, step });
            } else {
                clear_rect(view, aspect, value, r.rect, r.baseArrayLayer, r.layerCount);
            }
        };
        for att in atts {
            let has = |a| att.aspectMask & a != 0;
            if has(VK_IMAGE_ASPECT_COLOR_BIT) {
                let view = (att.colorAttachment < c.color_attachment_count).then(|| c.color_view(att.colorAttachment as usize)).flatten();
                if let Some(view) = view.map(std::ptr::from_ref) {
                    for r in rects {
                        clear(c, unsafe { &*view }, VK_IMAGE_ASPECT_COLOR_BIT, &att.clearValue, r);
                    }
                }
                continue;
            }
            for r in rects {
                if let Some(view) = c.depth_view().filter(|_| has(VK_IMAGE_ASPECT_DEPTH_BIT)).map(std::ptr::from_ref) {
                    clear(c, unsafe { &*view }, VK_IMAGE_ASPECT_DEPTH_BIT, &att.clearValue, r);
                }
                if let Some(view) = c.stencil_view().filter(|_| has(VK_IMAGE_ASPECT_STENCIL_BIT)).map(std::ptr::from_ref) {
                    clear(c, unsafe { &*view }, VK_IMAGE_ASPECT_STENCIL_BIT, &att.clearValue, r);
                }
            }
        }
    }
}

/// The flusher's UNORM8 of an FP16 channel (ColorQuantize.quantize8): round(v * 255).
fn quantize8(h: u16) -> Option<u8> {
    let (e, m) = ((h >> 10 & 31) as i32, (h & 1023) as f64);
    if e == 31 {
        return None;
    }
    let v = if e == 0 { m * 2f64.powi(-24) } else { (1024.0 + m) * 2f64.powi(e - 25) };
    Some(if h >> 15 != 0 { 0 } else { (v * 255.0 + 0.5).floor().min(255.0) as u8 })
}

/// The flusher's D16 of an FP32 depth (DepthQuantize.quantize16Fp32): round(d * 65535).
fn quantize16(d: f32) -> u16 {
    let e = d.to_bits() >> 23 & 255;
    if d.is_sign_negative() || e == 0 {
        0
    } else if e >= 127 {
        65535
    } else {
        (d as f64 * 65535.0 + 0.5).floor().min(65535.0) as u16
    }
}

/// Clears all of a large single-level, single-layer, single-sample attachment with a pass on Borg
/// when that stores the bytes the CPU's clear would: there the CPU's clear (Linux on Hutt) or the
/// upload of what it wrote (USB) costs more than the pass.
pub fn on_gpu(view: &vk_image_view, aspect: VkImageAspectFlags, value: &VkClearValue, rect: VkRect2D, base_layer: u32, layers: u32, alloc: *const VkAllocationCallbacks) -> bool {
    use crate::list::Cmd;
    let Some(img) = view.image() else { return false };
    let (w, h) = (img.vk.extent.width, img.vk.extent.height);
    let whole = rect.offset.x == 0 && rect.offset.y == 0 && rect.extent.width == w && rect.extent.height == h;
    let single = img.vk.mip_levels == 1 && img.vk.array_layers == 1 && img.vk.samples <= 1 && layers == 1 && base_layer == 0;
    if !whole || !single || w * h < 1 << 16 || view.format != img.vk.format || !crate::hw::serves_extent(w, h, false) {
        return false;
    }
    let pf = fmt::pipe(img.vk.format);
    let (cmds, color, zimg) = match aspect {
        VK_IMAGE_ASPECT_COLOR_BIT => {
            let bgra = match img.vk.format {
                VK_FORMAT_B8G8R8A8_UNORM => true,
                VK_FORMAT_R8G8B8A8_UNORM => false,
                _ => return false,
            };
            let mut px = [0u8; 16];
            pack_clear_color(img.vk.format, pf, &mut px, value.color());
            let c = value.color().float32();
            let mut rgb = [0u8; 3];
            for (k, v) in rgb.iter_mut().enumerate() {
                let Some(q) = quantize8(crate::list::f32_to_fp16(c[k].to_bits())) else { return false };
                *v = q;
            }
            let stored = if bgra { [rgb[2], rgb[1], rgb[0]] } else { rgb };
            if stored != px[..3] {
                return false;
            }
            let f = crate::format::flush_format(img.vk.format);
            (vec![Cmd::Pass(f, 0), Cmd::Target(f, c), Cmd::ClearPass(0xff00 & (px[3] as u32) << 8, 1f32.to_bits())], Some(img), None)
        }
        VK_IMAGE_ASPECT_DEPTH_BIT => {
            let d = value.depth_stencil().depth;
            let d32 = crate::format::depth_is_d32(img.vk.format);
            let mut t: Texel = [0; 32];
            write_depth_stencil(pf, &mut t, aspect, d, 0);
            let same = match img.vk.format {
                VK_FORMAT_D16_UNORM | VK_FORMAT_D16_UNORM_S8_UINT => quantize16(d).to_le_bytes() == t[..2],
                VK_FORMAT_D32_SFLOAT | VK_FORMAT_D32_SFLOAT_S8_UINT => d.to_le_bytes() == t[..4],
                _ => false,
            };
            if !same {
                return false;
            }
            (vec![Cmd::Pass(0, 1 | (d32 as u8) << 2 | 8), Cmd::ClearPass(0xff00, d.to_bits())], None, Some(img))
        }
        _ => return false,
    };
    let Some((at, pitch)) = crate::sim::placement(color, &[], zimg, None, [w, h]) else { return false };
    let mut b = crate::list::Batch::new();
    b.pending = cmds;
    b.pending.insert(1, Cmd::Place(at, pitch));
    b.build(Some(((w + 3) & !3, (h + 3) & !3)), alloc);
    let (Some(prog), _) = b.end() else { return false };
    if !prog.ok {
        return false;
    }
    crate::sim::run_pass(&prog, color, &[], zimg, None, &[], [w, h], &[], true);
    true
}

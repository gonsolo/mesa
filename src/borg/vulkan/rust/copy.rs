// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Transfers, done by the host in mapped memory: buffer fills, updates and copies, image
//! copies.

use crate::entrypoint::{entrypoint, slice, Handle};
use crate::ffi::*;
use crate::image::{aspect_blocks, blocks, shape, z16_pack, z16_unpack};
use borg_layout::image::{mip_level_size, Blocks};
use std::ffi::c_void;

/// The buffer's bytes from its binding on, when it is bound to mapped memory.
fn host(b: &borgvk_buffer) -> Option<*mut u8> {
    let m = unsafe { b.mem.as_ref() }.filter(|m| !m.map.is_null())?;
    Some(unsafe { m.map.cast::<u8>().add(b.offset as usize) })
}

entrypoint! {
    fn borgvk_CmdFillBuffer(_cmd: VkCommandBuffer, buffer: VkBuffer, offset: VkDeviceSize, size: VkDeviceSize, data: u32) {
        let Some(b) = buffer.get() else { return };
        let Some(base) = host(b) else { return };
        let size = if size == VK_WHOLE_SIZE as u64 { b.vk.size - offset } else { size };
        let dst = base.add(offset as usize).cast::<u32>();
        for i in 0..(size / 4) as usize {
            dst.add(i).write_unaligned(data);
        }
    }
}

entrypoint! {
    fn borgvk_CmdUpdateBuffer(_cmd: VkCommandBuffer, buffer: VkBuffer, offset: VkDeviceSize, size: VkDeviceSize, data: *const c_void) {
        let Some(base) = buffer.get().and_then(|b| host(b)) else { return };
        if !data.is_null() {
            std::ptr::copy_nonoverlapping(data.cast::<u8>(), base.add(offset as usize), size as usize);
        }
    }
}

fn copy_buffer(src: VkBuffer, dst: VkBuffer, regions: impl Iterator<Item = (u64, u64, u64)>) {
    let (Some(s), Some(d)) = (unsafe { (src.get().and_then(|b| host(b)), dst.get().and_then(|b| host(b))) }) else { return };
    for (from, to, size) in regions {
        unsafe { std::ptr::copy(s.add(from as usize), d.add(to as usize), size as usize) };
    }
}

entrypoint! {
    fn borgvk_CmdCopyBuffer(_cmd: VkCommandBuffer, src: VkBuffer, dst: VkBuffer, count: u32, regions: *const VkBufferCopy) {
        copy_buffer(src, dst, slice(regions, count).iter().map(|r| (r.srcOffset, r.dstOffset, r.size)));
    }
}

entrypoint! {
    fn borgvk_CmdCopyBuffer2(_cmd: VkCommandBuffer, info: *const VkCopyBufferInfo2) {
        let i = &*info;
        copy_buffer(i.srcBuffer, i.dstBuffer, slice(i.pRegions, i.regionCount).iter().map(|r| (r.srcOffset, r.dstOffset, r.size)));
    }
}

/// The image's bytes from its binding on, when it is bound to mapped memory.
fn image_host(i: &borgvk_image) -> Option<*mut u8> {
    let m = unsafe { i.mem.as_ref() }.filter(|m| !m.map.is_null())?;
    Some(unsafe { m.map.cast::<u8>().add(i.offset as usize) })
}

pub fn layers(image: &vk_image, s: &VkImageSubresourceLayers) -> u32 {
    if s.layerCount == VK_REMAINING_ARRAY_LAYERS as u32 { image.array_layers.wrapping_sub(s.baseArrayLayer) } else { s.layerCount }
}

/// Offset of a 2D layer or 3D slice of a level, and its row stride in blocks.
fn slice_offset(img: &borgvk_image, b: Blocks, level: u32, layer: u32, z: Option<u32>) -> (u64, u32) {
    let (off, stride) = shape(img).level_offset(level, if z.is_some() { 0 } else { layer }, b);
    let e = img.vk.extent;
    let z = z.map_or(0, |z| z as u64 * mip_level_size([e.width, e.height, 1], level, b).0);
    (off + z, stride)
}

/// One depth or stencil texel from one combined format to another.
fn copy_depth_stencil(spf: pipe_format, s: *const u8, dpf: pipe_format, d: *mut u8, stencil: bool) {
    let (sz16, dz16) = (spf == PIPE_FORMAT_Z16_UNORM_S8_UINT, dpf == PIPE_FORMAT_Z16_UNORM_S8_UINT);
    unsafe {
        if stencil {
            let mut v = if sz16 { *s.add(2) } else { 0 };
            if !sz16 {
                util_format_unpack_s_8uint(spf, &mut v, s.cast(), 1);
            }
            if dz16 { *d.add(2) = v } else { util_format_pack_s_8uint(dpf, d.cast(), &v, 1) }
        } else {
            let mut z = if sz16 { z16_unpack([*s, *s.add(1)]) } else { 0.0 };
            if !sz16 {
                util_format_unpack_z_float(spf, &mut z, s.cast(), 1);
            }
            if dz16 {
                std::ptr::copy_nonoverlapping(z16_pack(z).as_ptr(), d, 2);
            } else {
                util_format_pack_z_float(dpf, d.cast(), &z, 1);
            }
        }
    }
}

fn copy_image(src: &borgvk_image, dst: &borgvk_image, reg: &VkImageCopy) {
    let (Some(sbase), Some(dbase)) = (image_host(src), image_host(dst)) else { return };
    let (sb, db) = (blocks(src.vk.format), blocks(dst.vk.format));
    if sb.bd > 1 || db.bd > 1 {
        return;
    }
    let (spf, dpf) = unsafe { (vk_format_to_pipe_format(src.vk.format), vk_format_to_pipe_format(dst.vk.format)) };
    let (se, de) = (aspect_blocks(src.vk.format, reg.srcSubresource.aspectMask), aspect_blocks(dst.vk.format, reg.dstSubresource.aspectMask));
    let elem_bs = se.bs.min(de.bs) as usize;
    let w_blocks = reg.extent.width.div_ceil(se.bw);
    let h_blocks = reg.extent.height.div_ceil(se.bh);
    let (s3d, d3d) = (src.vk.image_type == VK_IMAGE_TYPE_3D, dst.vk.image_type == VK_IMAGE_TYPE_3D);
    let slice_bd = if s3d { sb.bd } else if d3d { db.bd } else { 1 };
    let slices = if s3d || d3d { reg.extent.depth.div_ceil(slice_bd) } else { layers(&src.vk, &reg.srcSubresource) };
    let (so, dof) = (reg.srcOffset, reg.dstOffset);
    for l in 0..slices {
        let sz = s3d.then(|| so.z as u32 / sb.bd + l);
        let dz = d3d.then(|| dof.z as u32 / db.bd + l);
        let (soff, sstride) = slice_offset(src, sb, reg.srcSubresource.mipLevel, reg.srcSubresource.baseArrayLayer + l, sz);
        let (doff, dstride) = slice_offset(dst, db, reg.dstSubresource.mipLevel, reg.dstSubresource.baseArrayLayer + l, dz);
        let (srow, drow) = (sstride as usize * sb.bs as usize, dstride as usize * db.bs as usize);
        let s = unsafe { sbase.add(soff as usize + (so.y as u32 / sb.bh) as usize * srow + (so.x as u32 / sb.bw) as usize * sb.bs as usize) };
        let d = unsafe { dbase.add(doff as usize + (dof.y as u32 / db.bh) as usize * drow + (dof.x as u32 / db.bw) as usize * db.bs as usize) };
        if se.bs == sb.bs && de.bs == db.bs {
            let (splane, dplane) = (shape(src).sample_plane_size(sb) as usize, shape(dst).sample_plane_size(db) as usize);
            for smp in 0..src.vk.samples.max(1) as usize {
                for row in 0..h_blocks as usize {
                    unsafe { std::ptr::copy(s.add(smp * splane + row * srow), d.add(smp * dplane + row * drow), w_blocks as usize * elem_bs) };
                }
            }
            continue;
        }
        // One aspect of a combined depth/stencil format: texel by texel.
        let stencil = reg.srcSubresource.aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT != 0;
        for row in 0..h_blocks as usize {
            for col in 0..w_blocks as usize {
                let (st, dt) = unsafe { (s.add(row * srow + col * sb.bs as usize), d.add(row * drow + col * db.bs as usize)) };
                copy_depth_stencil(spf, st, dpf, dt, stencil);
            }
        }
    }
}

entrypoint! {
    fn borgvk_CmdCopyImage(_cmd: VkCommandBuffer, src: VkImage, _sl: VkImageLayout, dst: VkImage, _dl: VkImageLayout, count: u32, regions: *const VkImageCopy) {
        let (Some(s), Some(d)) = (src.get(), dst.get()) else { return };
        for r in slice(regions, count) {
            copy_image(s, d, r);
        }
    }
}

entrypoint! {
    fn borgvk_CmdCopyImage2(_cmd: VkCommandBuffer, info: *const VkCopyImageInfo2) {
        let i = &*info;
        let (Some(s), Some(d)) = (i.srcImage.get(), i.dstImage.get()) else { return };
        for r in slice(i.pRegions, i.regionCount) {
            let r = VkImageCopy { srcSubresource: r.srcSubresource, srcOffset: r.srcOffset, dstSubresource: r.dstSubresource, dstOffset: r.dstOffset, extent: r.extent };
            copy_image(s, d, &r);
        }
    }
}

/// A block-compressed image decoded to R32G32B32A32_SFLOAT in `out`, backed by `mem`; the
/// buffer returned is malloc'd, for the caller to free.
///
/// # Safety
/// `src` is a mapped image; `out` and `mem` are valid for writes.
#[no_mangle]
pub unsafe extern "C" fn borgvk_decode_compressed_image(src: *const borgvk_image, out: *mut borgvk_image, mem: *mut borgvk_device_memory) -> *mut c_void {
    let s = unsafe { &*src };
    let pf = unsafe { vk_format_to_pipe_format(s.vk.format) };
    let sb = Blocks { bd: 1, ..blocks(s.vk.format) };
    let rgba = Blocks { bs: 16, bw: 1, bh: 1, bd: 1 };
    let layers = s.vk.array_layers.max(1);
    let o = unsafe { &mut *out };
    *o = unsafe { std::ptr::read(s) };
    o.vk.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    let total = shape(o).layer_size(rgba) * layers as u64;
    let buf = unsafe { calloc(1, total as _) }.cast::<u8>();
    if buf.is_null() {
        return std::ptr::null_mut();
    }
    let sbase = image_host(s).unwrap();
    let shr = |e: u32, l: u32| e.checked_shr(l).unwrap_or(0).max(1);
    for layer in 0..layers {
        for l in 0..s.vk.mip_levels {
            let (w, h, d) = (shr(s.vk.extent.width, l), shr(s.vk.extent.height, l), shr(s.vk.extent.depth, l));
            let (soff, wb) = shape(s).level_offset(l, layer, sb);
            let (doff, _) = shape(o).level_offset(l, layer, rgba);
            let sslice = wb as u64 * h.div_ceil(sb.bh) as u64 * sb.bs as u64;
            let dslice = w as u64 * h as u64 * 16;
            for z in 0..d as u64 {
                unsafe {
                    util_format_unpack_rgba_rect(
                        pf,
                        buf.add((doff + z * dslice) as usize).cast(),
                        w * 16,
                        sbase.add((soff + z * sslice) as usize).cast(),
                        wb * sb.bs,
                        w,
                        h,
                    )
                };
            }
        }
    }
    let m = unsafe { &mut *mem };
    *m = unsafe { std::mem::zeroed() };
    (m.map, m.size) = (buf.cast(), total as usize);
    (o.mem, o.offset) = (mem, 0);
    buf.cast()
}

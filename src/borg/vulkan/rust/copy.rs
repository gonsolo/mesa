// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Transfers, done by the host in mapped memory: buffer fills, updates and copies, image
//! copies, blits and resolves.

use crate::corpus::{borgvk_corpus_hash, borgvk_corpus_log, borgvk_corpus_on, hash};
use crate::entrypoint::{entrypoint, slice, Handle};
use crate::ffi::*;
use crate::memory::{Buffer, DeviceMemory};
use crate::image::Image;
use crate::image::{aspect_blocks, blocks, shape, z16_pack, z16_unpack};
use crate::texel::{is_partial_ds_aspect, pack_rgba_float, unpack_rgba_float};
use borg_layout::image::{mip_level_size, Blocks};
use std::ffi::c_void;

/// The buffer's bytes from its binding on, when it is bound to mapped memory.
pub fn host(b: &Buffer) -> Option<*mut u8> {
    let m = unsafe { b.mem.as_ref() }.filter(|m| !m.map.is_null())?;
    Some(unsafe { m.map.cast::<u8>().add(b.offset as usize) })
}

entrypoint! {
    unsafe fn borgvk_CmdFillBuffer(_cmd: VkCommandBuffer, buffer: VkBuffer, offset: VkDeviceSize, size: VkDeviceSize, data: u32) {
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
    unsafe fn borgvk_CmdUpdateBuffer(_cmd: VkCommandBuffer, buffer: VkBuffer, offset: VkDeviceSize, size: VkDeviceSize, data: *const c_void) {
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
    unsafe fn borgvk_CmdCopyBuffer(_cmd: VkCommandBuffer, src: VkBuffer, dst: VkBuffer, count: u32, regions: *const VkBufferCopy) {
        copy_buffer(src, dst, slice(regions, count).iter().map(|r| (r.srcOffset, r.dstOffset, r.size)));
    }
}

entrypoint! {
    unsafe fn borgvk_CmdCopyBuffer2(_cmd: VkCommandBuffer, info: *const VkCopyBufferInfo2) {
        let i = &*info;
        copy_buffer(i.srcBuffer, i.dstBuffer, slice(i.pRegions, i.regionCount).iter().map(|r| (r.srcOffset, r.dstOffset, r.size)));
    }
}

/// The image's bytes from its binding on, when it is bound to mapped memory.
pub fn image_host(i: &Image) -> Option<*mut u8> {
    let m = unsafe { i.mem.as_ref() }.filter(|m| !m.map.is_null())?;
    Some(unsafe { m.map.cast::<u8>().add(i.offset as usize) })
}

pub fn layers(image: &vk_image, s: &VkImageSubresourceLayers) -> u32 {
    if s.layerCount == VK_REMAINING_ARRAY_LAYERS as u32 { image.array_layers.wrapping_sub(s.baseArrayLayer) } else { s.layerCount }
}

/// Offset of a 2D layer or 3D slice of a level, and its row stride in blocks.
fn slice_offset(img: &Image, b: Blocks, level: u32, layer: u32, z: Option<u32>) -> (u64, u32) {
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

fn copy_image(src: &Image, dst: &Image, reg: &VkImageCopy) {
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
    unsafe fn borgvk_CmdCopyImage(_cmd: VkCommandBuffer, src: VkImage, _sl: VkImageLayout, dst: VkImage, _dl: VkImageLayout, count: u32, regions: *const VkImageCopy) {
        let (Some(s), Some(d)) = (src.get(), dst.get()) else { return };
        for r in slice(regions, count) {
            copy_image(s, d, r);
        }
    }
}

entrypoint! {
    unsafe fn borgvk_CmdCopyImage2(_cmd: VkCommandBuffer, info: *const VkCopyImageInfo2) {
        let i = &*info;
        let (Some(s), Some(d)) = (i.srcImage.get(), i.dstImage.get()) else { return };
        for r in slice(i.pRegions, i.regionCount) {
            let r = VkImageCopy { srcSubresource: r.srcSubresource, srcOffset: r.srcOffset, dstSubresource: r.dstSubresource, dstOffset: r.dstOffset, extent: r.extent };
            copy_image(s, d, &r);
        }
    }
}

/// One region between an image and a buffer, towards the buffer when `to_buffer`.
fn buffer_image(img: &Image, buf: &Buffer, reg: &VkBufferImageCopy, to_buffer: bool) {
    let (Some(ibase), Some(bbase)) = (image_host(img), host(buf)) else { return };
    let (ib, sub) = (blocks(img.vk.format), reg.imageSubresource);
    let a = aspect_blocks(img.vk.format, sub.aspectMask);
    let w_blocks = reg.imageExtent.width.div_ceil(a.bw) as usize;
    let h_blocks = reg.imageExtent.height.div_ceil(a.bh) as usize;
    // 0 means tightly packed.
    let row_len = if reg.bufferRowLength != 0 { reg.bufferRowLength.div_ceil(a.bw) as usize } else { w_blocks };
    let img_h = if reg.bufferImageHeight != 0 { reg.bufferImageHeight.div_ceil(a.bh) as usize } else { h_blocks };
    let is3d = img.vk.image_type == VK_IMAGE_TYPE_3D;
    let slices = if is3d { reg.imageExtent.depth.max(1) } else { sub.layerCount.max(1) };
    let partial = is_partial_ds_aspect(img.vk.format, sub.aspectMask);
    let (ibs, abs) = (ib.bs as usize, a.bs as usize);
    let brow = row_len * abs;
    for l in 0..slices {
        let z = is3d.then(|| reg.imageOffset.z as u32 + l);
        let (off, stride) = slice_offset(img, ib, sub.mipLevel, sub.baseArrayLayer + l, z);
        let irow = stride as usize * ibs;
        let it = unsafe { ibase.add(off as usize + (reg.imageOffset.y as u32 / ib.bh) as usize * irow + (reg.imageOffset.x as u32 / ib.bw) as usize * ibs) };
        let bt = unsafe { bbase.add(reg.bufferOffset as usize + l as usize * img_h * brow) };
        if a.bs == ib.bs && !partial {
            for row in 0..h_blocks {
                let (i, b) = unsafe { (it.add(row * irow), bt.add(row * brow)) };
                let (from, to) = if to_buffer { (i, b) } else { (b, i) };
                unsafe { std::ptr::copy(from, to, w_blocks * abs) };
            }
            continue;
        }
        let stencil = sub.aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT != 0;
        let aspect = if stencil { VK_IMAGE_ASPECT_STENCIL_BIT } else { VK_IMAGE_ASPECT_DEPTH_BIT };
        let (ipf, bpf) = unsafe { (vk_format_to_pipe_format(img.vk.format), vk_format_to_pipe_format(vk_format_get_aspect_format(img.vk.format, aspect))) };
        for row in 0..h_blocks {
            for col in 0..w_blocks {
                let (i, b) = unsafe { (it.add(row * irow + col * ibs), bt.add(row * brow + col * abs)) };
                if to_buffer { copy_depth_stencil(ipf, i, bpf, b, stencil) } else { copy_depth_stencil(bpf, b, ipf, i, stencil) }
            }
        }
    }
}

fn buffer_image_copies(image: VkImage, buffer: VkBuffer, regions: &[VkBufferImageCopy], to_buffer: bool) {
    let (Some(img), Some(buf)) = (unsafe { (image.get(), buffer.get()) }) else { return };
    if image_host(img).is_none() || host(buf).is_none() || blocks(img.vk.format).bd > 1 {
        return;
    }
    for r in regions {
        buffer_image(img, buf, r, to_buffer);
    }
    if to_buffer && borgvk_corpus_on() {
        let key = hash(BORGVK_CORPUS_SEED as u64, &img.vk.format.to_ne_bytes());
        let key = unsafe { borgvk_corpus_hash(key, regions.as_ptr().cast(), size_of_val(regions)) };
        let content = unsafe { borgvk_corpus_hash(BORGVK_CORPUS_SEED as u64, host(buf).unwrap().cast(), buf.vk.size as usize) };
        borgvk_corpus_log(b'R' as _, key, content);
    }
}

fn buffer_image_copy(r: &VkBufferImageCopy2) -> VkBufferImageCopy {
    VkBufferImageCopy {
        bufferOffset: r.bufferOffset,
        bufferRowLength: r.bufferRowLength,
        bufferImageHeight: r.bufferImageHeight,
        imageSubresource: r.imageSubresource,
        imageOffset: r.imageOffset,
        imageExtent: r.imageExtent,
    }
}

entrypoint! {
    unsafe fn borgvk_CmdCopyImageToBuffer(_cmd: VkCommandBuffer, src: VkImage, _sl: VkImageLayout, dst: VkBuffer, count: u32, regions: *const VkBufferImageCopy) {
        buffer_image_copies(src, dst, slice(regions, count), true);
    }
}

entrypoint! {
    unsafe fn borgvk_CmdCopyBufferToImage(_cmd: VkCommandBuffer, src: VkBuffer, dst: VkImage, _dl: VkImageLayout, count: u32, regions: *const VkBufferImageCopy) {
        buffer_image_copies(dst, src, slice(regions, count), false);
    }
}

entrypoint! {
    unsafe fn borgvk_CmdCopyImageToBuffer2(_cmd: VkCommandBuffer, info: *const VkCopyImageToBufferInfo2) {
        let i = &*info;
        let regions: Vec<_> = slice(i.pRegions, i.regionCount).iter().map(buffer_image_copy).collect();
        buffer_image_copies(i.srcImage, i.dstBuffer, &regions, true);
    }
}

entrypoint! {
    unsafe fn borgvk_CmdCopyBufferToImage2(_cmd: VkCommandBuffer, info: *const VkCopyBufferToImageInfo2) {
        let i = &*info;
        let regions: Vec<_> = slice(i.pRegions, i.regionCount).iter().map(buffer_image_copy).collect();
        buffer_image_copies(i.dstImage, i.srcBuffer, &regions, false);
    }
}

/// A block-compressed image decoded to R32G32B32A32_SFLOAT in `out`, backed by `mem`; the
/// buffer returned is malloc'd, for the caller to free.
///
/// # Safety
/// `src` is a mapped image; `out` and `mem` are valid for writes.
pub unsafe fn borgvk_decode_compressed_image(src: *const Image, out: *mut Image, mem: *mut DeviceMemory) -> *mut c_void {
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

fn read_z(pf: pipe_format, t: *const u8) -> f32 {
    if pf == PIPE_FORMAT_Z16_UNORM_S8_UINT {
        return unsafe { z16_unpack([*t, *t.add(1)]) };
    }
    let mut z = 0.0;
    unsafe { util_format_unpack_z_float(pf, &mut z, t.cast(), 1) };
    z
}

fn write_z(pf: pipe_format, t: *mut u8, z: f32) {
    if pf == PIPE_FORMAT_Z16_UNORM_S8_UINT {
        unsafe { std::ptr::copy_nonoverlapping(z16_pack(z).as_ptr(), t, 2) };
    } else {
        unsafe { util_format_pack_z_float(pf, t.cast(), &z, 1) };
    }
}

/// The two texels around a coordinate along one axis and their weights.
fn taps(f: f32, size: u32) -> ([i32; 2], [f32; 2]) {
    let i0 = f.floor() as i32;
    let t = f - i0 as f32;
    let c = |i: i32| i.clamp(0, size as i32 - 1);
    ([c(i0), c(i0 + 1)], [1.0 - t, t])
}

fn blit(src: &Image, dst: &Image, regions: &[VkImageBlit], filter: VkFilter) {
    if image_host(src).is_none() || image_host(dst).is_none() {
        return;
    }
    let db = blocks(dst.vk.format);
    if db.bw > 1 || db.bh > 1 {
        return;
    }
    let (mut dimg, mut dmem) = unsafe { (std::mem::zeroed::<Image>(), std::mem::zeroed::<DeviceMemory>()) };
    let mut decoded = std::ptr::null_mut();
    let sb = blocks(src.vk.format);
    let src = if sb.bw > 1 || sb.bh > 1 {
        decoded = unsafe { borgvk_decode_compressed_image(src, &mut dimg, &mut dmem) };
        if decoded.is_null() {
            return;
        }
        &dimg
    } else {
        src
    };
    let (spf, dpf) = unsafe { (vk_format_to_pipe_format(src.vk.format), vk_format_to_pipe_format(dst.vk.format)) };
    let (sbs, dbs) = (blocks(src.vk.format).bs, db.bs);
    let (is_uint, is_sint) = unsafe { (util_format_is_pure_uint(spf), util_format_is_pure_sint(spf)) };
    let (s3d, d3d) = (src.vk.image_type == VK_IMAGE_TYPE_3D, dst.vk.image_type == VK_IMAGE_TYPE_3D);
    let shr = |e: u32, l: u32| e.checked_shr(l).unwrap_or(0).max(1);
    for reg in regions {
        let aspect = reg.srcSubresource.aspectMask;
        let (depth, stencil) = (aspect & VK_IMAGE_ASPECT_DEPTH_BIT != 0, aspect & VK_IMAGE_ASPECT_STENCIL_BIT != 0);
        let ([s0, s1], [d0, d1]) = (reg.srcOffsets, reg.dstOffsets);
        let (dex, dey, dez) = ((d1.x - d0.x) as f32, (d1.y - d0.y) as f32, (d1.z - d0.z) as f32);
        let (sex, sey, sez) = ((s1.x - s0.x) as f32, (s1.y - s0.y) as f32, (s1.z - s0.z) as f32);
        let (sl, dl) = (reg.srcSubresource.mipLevel, reg.dstSubresource.mipLevel);
        let nlayers = if s3d || d3d { 1 } else { layers(&src.vk, &reg.srcSubresource) };
        for l in 0..nlayers {
            let (sw, sh) = (shr(src.vk.extent.width, sl), shr(src.vk.extent.height, sl));
            let sd = if s3d { shr(src.vk.extent.depth, sl) } else { 1 };
            let dh = shr(dst.vk.extent.height, dl);
            let one = |bs| Blocks { bs, bw: 1, bh: 1, bd: 1 };
            let (soff, sstride) = shape(src).level_offset(sl, if s3d { 0 } else { reg.srcSubresource.baseArrayLayer + l }, one(sbs));
            let (doff, dstride) = shape(dst).level_offset(dl, if d3d { 0 } else { reg.dstSubresource.baseArrayLayer + l }, one(dbs));
            let sbase = unsafe { image_host(src).unwrap().add(soff as usize) };
            let dbase = unsafe { image_host(dst).unwrap().add(doff as usize) };
            let st = |x: i32, y: i32, z: i32| unsafe {
                sbase.add((((z as u64 * sh as u64 + y as u64) * sstride as u64 + x as u64) * sbs as u64) as usize)
            };
            let w = |d: i32, o: i32, e: f32| if e != 0.0 { (d as f32 + 0.5 - o as f32) / e } else { 0.5 };
            for dz in d0.z.min(d1.z)..d0.z.max(d1.z) {
                let sz = s0.z as f32 + w(dz, d0.z, dez) * sez;
                for dy in d0.y.min(d1.y)..d0.y.max(d1.y) {
                    let sy = s0.y as f32 + w(dy, d0.y, dey) * sey;
                    let drow = (dz as i64 as u64).wrapping_mul(dh as u64).wrapping_add(dy as i64 as u64).wrapping_mul(dstride as u64 * dbs as u64);
                    for dx in d0.x.min(d1.x)..d0.x.max(d1.x) {
                        let sx = s0.x as f32 + w(dx, d0.x, dex) * sex;
                        let dt = unsafe { dbase.add(drow.wrapping_add((dx as i64 as u64).wrapping_mul(dbs as u64)) as usize) };
                        let near = || {
                            let c = |f: f32, n: u32| (f.floor() as i32).clamp(0, n as i32 - 1);
                            st(c(sx, sw), c(sy, sh), c(sz, sd))
                        };
                        let (xs, xw) = taps(sx - 0.5, sw);
                        let (ys, yw) = taps(sy - 0.5, sh);
                        let (zs, zw) = taps(sz - 0.5, sd);
                        // The weighted taps of the linear filter, z outermost.
                        let linear = || {
                            (0..8).filter_map(move |i| {
                                let (ix, iy, iz) = (i & 1, i >> 1 & 1, i >> 2);
                                let weight = xw[ix] * yw[iy] * zw[iz];
                                (weight != 0.0).then(|| (st(xs[ix], ys[iy], zs[iz]), weight))
                            })
                        };
                        if stencil {
                            copy_depth_stencil(spf, near(), dpf, dt, true);
                        }
                        if depth {
                            if filter == VK_FILTER_NEAREST {
                                copy_depth_stencil(spf, near(), dpf, dt, false);
                            } else {
                                let acc = linear().fold(0.0, |acc, (t, wt)| acc + read_z(spf, t) * wt);
                                write_z(dpf, dt, acc);
                            }
                        }
                        if depth || stencil {
                            continue;
                        }
                        if filter == VK_FILTER_NEAREST {
                            let t = near();
                            if is_uint || is_sint {
                                // The integer bit pattern either way.
                                let mut v = [0u32; 4];
                                unsafe {
                                    util_format_unpack_rgba(spf, v.as_mut_ptr().cast(), t.cast(), 1);
                                    util_format_pack_rgba(dpf, dt.cast(), v.as_ptr().cast(), 1);
                                }
                            } else {
                                let mut v = [0.0; 4];
                                unpack_rgba_float(src.vk.format, spf, t, &mut v);
                                pack_rgba_float(dst.vk.format, dpf, dt, &v);
                            }
                        } else {
                            let mut acc = [0f32; 4];
                            for (t, wt) in linear() {
                                let mut v = [0.0; 4];
                                unpack_rgba_float(src.vk.format, spf, t, &mut v);
                                for c in 0..4 {
                                    acc[c] += v[c] * wt;
                                }
                            }
                            pack_rgba_float(dst.vk.format, dpf, dt, &acc);
                        }
                    }
                }
            }
        }
    }
    unsafe { free(decoded) };
}

entrypoint! {
    unsafe fn borgvk_CmdBlitImage(_cmd: VkCommandBuffer, src: VkImage, _sl: VkImageLayout, dst: VkImage, _dl: VkImageLayout, count: u32, regions: *const VkImageBlit, filter: VkFilter) {
        let (Some(s), Some(d)) = (src.get(), dst.get()) else { return };
        blit(s, d, slice(regions, count), filter);
    }
}

entrypoint! {
    unsafe fn borgvk_CmdBlitImage2(_cmd: VkCommandBuffer, info: *const VkBlitImageInfo2) {
        let i = &*info;
        let (Some(s), Some(d)) = (i.srcImage.get(), i.dstImage.get()) else { return };
        let regions: Vec<VkImageBlit> = slice(i.pRegions, i.regionCount)
            .iter()
            .map(|r| VkImageBlit { srcSubresource: r.srcSubresource, srcOffsets: r.srcOffsets, dstSubresource: r.dstSubresource, dstOffsets: r.dstOffsets })
            .collect();
        blit(s, d, &regions, i.filter);
    }
}

/// Averages the samples of one region into a single-sampled image.
pub fn resolve_region(src: &Image, dst: &Image, reg: &VkImageResolve) {
    let (Some(sbase), Some(dbase)) = (image_host(src), image_host(dst)) else { return };
    let (spf, dpf) = unsafe { (vk_format_to_pipe_format(src.vk.format), vk_format_to_pipe_format(dst.vk.format)) };
    let texel = |f: VkFormat| Blocks { bs: blocks(f).bs, bw: 1, bh: 1, bd: 1 };
    let (sb, db) = (texel(src.vk.format), texel(dst.vk.format));
    let (sbs, dbs) = (sb.bs as usize, db.bs as usize);
    let samples = src.vk.samples.max(1);
    let plane = shape(src).sample_plane_size(sb) as usize;
    let int_copy = unsafe { util_format_is_pure_integer(spf) } && sbs == dbs;
    let (ss, ds) = (reg.srcSubresource, reg.dstSubresource);
    for l in 0..layers(&src.vk, &ss) {
        let (soff, sstride) = shape(src).level_offset(ss.mipLevel, ss.baseArrayLayer + l, sb);
        let (doff, dstride) = shape(dst).level_offset(ds.mipLevel, ds.baseArrayLayer + l, db);
        for y in 0..reg.extent.height as usize {
            let (sy, dy) = (reg.srcOffset.y as usize + y, reg.dstOffset.y as usize + y);
            for x in 0..reg.extent.width as usize {
                let (sx, dx) = (reg.srcOffset.x as usize + x, reg.dstOffset.x as usize + x);
                let st = unsafe { sbase.add(soff as usize + (sy * sstride as usize + sx) * sbs) };
                let dt = unsafe { dbase.add(doff as usize + (dy * dstride as usize + dx) * dbs) };
                if int_copy {
                    unsafe { std::ptr::copy(st, dt, sbs) };
                    continue;
                }
                let mut acc = [0.0f32; 4];
                for s in 0..samples as usize {
                    let mut v = [0.0; 4];
                    unpack_rgba_float(src.vk.format, spf, unsafe { st.add(s * plane) }, &mut v);
                    acc.iter_mut().zip(v).for_each(|(a, v)| *a += v);
                }
                acc.iter_mut().for_each(|a| *a /= samples as f32);
                pack_rgba_float(dst.vk.format, dpf, dt, &acc);
            }
        }
    }
}

entrypoint! {
    unsafe fn borgvk_CmdResolveImage(_cmd: VkCommandBuffer, src: VkImage, _sl: VkImageLayout, dst: VkImage, _dl: VkImageLayout, count: u32, regions: *const VkImageResolve) {
        let (Some(s), Some(d)) = (src.get(), dst.get()) else { return };
        for r in slice(regions, count) {
            resolve_region(s, d, r);
        }
    }
}

entrypoint! {
    unsafe fn borgvk_CmdResolveImage2(_cmd: VkCommandBuffer, info: *const VkResolveImageInfo2) {
        let i = &*info;
        let (Some(s), Some(d)) = (i.srcImage.get(), i.dstImage.get()) else { return };
        for r in slice(i.pRegions, i.regionCount) {
            let r = VkImageResolve { srcSubresource: r.srcSubresource, srcOffset: r.srcOffset, dstSubresource: r.dstSubresource, dstOffset: r.dstOffset, extent: r.extent };
            resolve_region(s, d, &r);
        }
    }
}

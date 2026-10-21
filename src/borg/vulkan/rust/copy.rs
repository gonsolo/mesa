// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Transfers, done by the host in mapped memory: buffer fills, updates and copies, image
//! copies, blits and resolves.

use crate::corpus::{borgvk_corpus_log, borgvk_corpus_on, hash};
use crate::entrypoint::{entrypoint, pointee, Handle};
use crate::ffi::*;
use crate::fmt;
use crate::image::{aspect_blocks, blocks, shape, z16_pack, z16_unpack, Image};
use crate::memory::{Buffer, DeviceMemory, Host};
use crate::vk::*;
use crate::texel::{is_partial_ds_aspect, pack_rgba_float, unpack_rgba_float, Texel};
use borg_layout::image::{mip_level_size, Blocks};
use std::ffi::c_void;

/// The texel of `n` bytes at `off`.
fn texel(h: &Host, off: usize, n: usize) -> Option<Texel> {
    let mut t: Texel = [0; 32];
    h.read(off, &mut t[..n]).then_some(t)
}

pointee!(VkCopyBufferInfo2, VkCopyImageInfo2, VkCopyImageToBufferInfo2, VkCopyBufferToImageInfo2, VkBlitImageInfo2, VkResolveImageInfo2);

entrypoint! {
    fn borgvk_CmdFillBuffer(_cmd: VkCommandBuffer, buffer: &Buffer, offset: VkDeviceSize, size: VkDeviceSize, data: u32) {
        let Some(h) = buffer.host() else { return };
        let size = if size == VK_WHOLE_SIZE as u64 { buffer.vk.size - offset } else { size };
        h.fill(offset as usize, &data.to_ne_bytes(), (size / 4) as usize);
    }
}

entrypoint! {
    fn borgvk_CmdUpdateBuffer(_cmd: VkCommandBuffer, buffer: &Buffer, offset: VkDeviceSize, size: VkDeviceSize, data: *const c_void) {
        let Some(h) = buffer.host() else { return };
        if !data.is_null() {
            h.write(offset as usize, unsafe { std::slice::from_raw_parts(data.cast(), size as usize) });
        }
    }
}

fn copy_buffer(src: &Buffer, dst: &Buffer, regions: impl Iterator<Item = (u64, u64, u64)>) {
    let (Some(s), Some(d)) = (src.host(), dst.host()) else { return };
    for (from, to, size) in regions {
        s.copy(from as usize, &d, to as usize, size as usize);
    }
}

entrypoint! {
    fn borgvk_CmdCopyBuffer(_cmd: VkCommandBuffer, src: &Buffer, dst: &Buffer, _count: u32, regions: &[VkBufferCopy]) {
        copy_buffer(src, dst, regions.iter().map(|r| (r.srcOffset, r.dstOffset, r.size)));
    }
}

entrypoint! {
    fn borgvk_CmdCopyBuffer2(_cmd: VkCommandBuffer, i: &VkCopyBufferInfo2) {
        let (Some(s), Some(d)) = (unsafe { (i.srcBuffer.get_ref(), i.dstBuffer.get_ref()) }) else { return };
        copy_buffer(s, d, i.regions().iter().map(|r| (r.srcOffset, r.dstOffset, r.size)));
    }
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

/// One depth or stencil texel from one combined format to another; `d` holds the destination
/// texel, whose other aspect stays.
fn copy_depth_stencil(spf: pipe_format, s: &Texel, dpf: pipe_format, d: &mut Texel, stencil: bool) {
    let (sz16, dz16) = (spf == PIPE_FORMAT_Z16_UNORM_S8_UINT, dpf == PIPE_FORMAT_Z16_UNORM_S8_UINT);
    if stencil {
        let mut v = if sz16 { s[2] } else { 0 };
        if !sz16 {
            unsafe { util_format_unpack_s_8uint(spf, &mut v, s.as_ptr().cast(), 1) };
        }
        if dz16 {
            d[2] = v;
        } else {
            unsafe { util_format_pack_s_8uint(dpf, d.as_mut_ptr().cast(), &v, 1) };
        }
    } else {
        let mut z = if sz16 { z16_unpack([s[0], s[1]]) } else { 0.0 };
        if !sz16 {
            unsafe { util_format_unpack_z_float(spf, &mut z, s.as_ptr().cast(), 1) };
        }
        if dz16 {
            d[..2].copy_from_slice(&z16_pack(z));
        } else {
            unsafe { util_format_pack_z_float(dpf, d.as_mut_ptr().cast(), &z, 1) };
        }
    }
}

/// A depth or stencil texel at `so` in `sh` to `doff` in `dh`.
#[allow(clippy::too_many_arguments)]
fn move_depth_stencil(sh: &Host, so: usize, sbs: usize, spf: pipe_format, dh: &Host, doff: usize, dbs: usize, dpf: pipe_format, stencil: bool) {
    let (Some(s), Some(mut d)) = (texel(sh, so, sbs), texel(dh, doff, dbs)) else { return };
    copy_depth_stencil(spf, &s, dpf, &mut d, stencil);
    dh.write(doff, &d[..dbs]);
}

fn copy_image(src: &Image, dst: &Image, reg: &VkImageCopy) {
    let (Some(sh), Some(dh)) = (src.host(), dst.host()) else { return };
    let (sb, db) = (blocks(src.vk.format), blocks(dst.vk.format));
    if sb.bd > 1 || db.bd > 1 {
        return;
    }
    let (spf, dpf) = (fmt::pipe(src.vk.format), fmt::pipe(dst.vk.format));
    let (se, de) = (aspect_blocks(src.vk.format, reg.srcSubresource.aspectMask), aspect_blocks(dst.vk.format, reg.dstSubresource.aspectMask));
    let elem_bs = se.bs.min(de.bs) as usize;
    let w_blocks = reg.extent.width.div_ceil(se.bw);
    let h_blocks = reg.extent.height.div_ceil(se.bh);
    let (s3d, d3d) = (src.vk.image_type == VK_IMAGE_TYPE_3D, dst.vk.image_type == VK_IMAGE_TYPE_3D);
    let slice_bd = if s3d { sb.bd } else if d3d { db.bd } else { 1 };
    let slices = if s3d || d3d { reg.extent.depth.div_ceil(slice_bd) } else { layers(&src.vk, &reg.srcSubresource) };
    let (so, dof) = (reg.srcOffset, reg.dstOffset);
    let (sbs, dbs) = (sb.bs as usize, db.bs as usize);
    for l in 0..slices {
        let sz = s3d.then(|| so.z as u32 / sb.bd + l);
        let dz = d3d.then(|| dof.z as u32 / db.bd + l);
        let (soff, sstride) = slice_offset(src, sb, reg.srcSubresource.mipLevel, reg.srcSubresource.baseArrayLayer + l, sz);
        let (doff, dstride) = slice_offset(dst, db, reg.dstSubresource.mipLevel, reg.dstSubresource.baseArrayLayer + l, dz);
        let (srow, drow) = (sstride as usize * sbs, dstride as usize * dbs);
        let s = soff as usize + (so.y as u32 / sb.bh) as usize * srow + (so.x as u32 / sb.bw) as usize * sbs;
        let d = doff as usize + (dof.y as u32 / db.bh) as usize * drow + (dof.x as u32 / db.bw) as usize * dbs;
        if se.bs == sb.bs && de.bs == db.bs {
            let (splane, dplane) = (shape(src).sample_plane_size(sb) as usize, shape(dst).sample_plane_size(db) as usize);
            for smp in 0..src.vk.samples.max(1) as usize {
                for row in 0..h_blocks as usize {
                    sh.copy(s + smp * splane + row * srow, &dh, d + smp * dplane + row * drow, w_blocks as usize * elem_bs);
                }
            }
            continue;
        }
        // One aspect of a combined depth/stencil format: texel by texel.
        let stencil = reg.srcSubresource.aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT != 0;
        for row in 0..h_blocks as usize {
            for col in 0..w_blocks as usize {
                move_depth_stencil(&sh, s + row * srow + col * sbs, sbs, spf, &dh, d + row * drow + col * dbs, dbs, dpf, stencil);
            }
        }
    }
}

entrypoint! {
    fn borgvk_CmdCopyImage(_cmd: VkCommandBuffer, src: &Image, _sl: VkImageLayout, dst: &Image, _dl: VkImageLayout, _count: u32, regions: &[VkImageCopy]) {
        for r in regions {
            copy_image(src, dst, r);
        }
    }
}

entrypoint! {
    fn borgvk_CmdCopyImage2(_cmd: VkCommandBuffer, i: &VkCopyImageInfo2) {
        let (Some(s), Some(d)) = (unsafe { (i.srcImage.get_ref(), i.dstImage.get_ref()) }) else { return };
        for r in i.regions() {
            let r = VkImageCopy { srcSubresource: r.srcSubresource, srcOffset: r.srcOffset, dstSubresource: r.dstSubresource, dstOffset: r.dstOffset, extent: r.extent };
            copy_image(s, d, &r);
        }
    }
}

/// One region between an image and a buffer, towards the buffer when `to_buffer`.
fn buffer_image(img: &Image, buf: &Buffer, reg: &VkBufferImageCopy, to_buffer: bool) {
    let (Some(ih), Some(bh)) = (img.host(), buf.host()) else { return };
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
        let it = off as usize + (reg.imageOffset.y as u32 / ib.bh) as usize * irow + (reg.imageOffset.x as u32 / ib.bw) as usize * ibs;
        let bt = reg.bufferOffset as usize + l as usize * img_h * brow;
        if a.bs == ib.bs && !partial {
            for row in 0..h_blocks {
                let (i, b) = (it + row * irow, bt + row * brow);
                if to_buffer { ih.copy(i, &bh, b, w_blocks * abs) } else { bh.copy(b, &ih, i, w_blocks * abs) };
            }
            continue;
        }
        let stencil = sub.aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT != 0;
        let aspect = if stencil { VK_IMAGE_ASPECT_STENCIL_BIT } else { VK_IMAGE_ASPECT_DEPTH_BIT };
        let (ipf, bpf) = (fmt::pipe(img.vk.format), fmt::pipe(fmt::aspect_format(img.vk.format, aspect)));
        for row in 0..h_blocks {
            for col in 0..w_blocks {
                let (i, b) = (it + row * irow + col * ibs, bt + row * brow + col * abs);
                if to_buffer {
                    move_depth_stencil(&ih, i, ibs, ipf, &bh, b, abs, bpf, stencil);
                } else {
                    move_depth_stencil(&bh, b, abs, bpf, &ih, i, ibs, ipf, stencil);
                }
            }
        }
    }
}

fn buffer_image_copies(img: &Image, buf: &Buffer, regions: &[VkBufferImageCopy], to_buffer: bool) {
    let (Some(_), Some(bh)) = (img.host(), buf.host()) else { return };
    if blocks(img.vk.format).bd > 1 {
        return;
    }
    for r in regions {
        buffer_image(img, buf, r, to_buffer);
    }
    if to_buffer && borgvk_corpus_on() {
        let key = hash(BORGVK_CORPUS_SEED, &img.vk.format.to_ne_bytes());
        let key = hash(key, unsafe { std::slice::from_raw_parts(regions.as_ptr().cast(), size_of_val(regions)) });
        let content = bh.with_bytes(0, buf.vk.size as usize, |b| hash(BORGVK_CORPUS_SEED, b)).unwrap_or(BORGVK_CORPUS_SEED);
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
    fn borgvk_CmdCopyImageToBuffer(_cmd: VkCommandBuffer, src: &Image, _sl: VkImageLayout, dst: &Buffer, _count: u32, regions: &[VkBufferImageCopy]) {
        buffer_image_copies(src, dst, regions, true);
    }
}

entrypoint! {
    fn borgvk_CmdCopyBufferToImage(_cmd: VkCommandBuffer, src: &Buffer, dst: &Image, _dl: VkImageLayout, _count: u32, regions: &[VkBufferImageCopy]) {
        buffer_image_copies(dst, src, regions, false);
    }
}

entrypoint! {
    fn borgvk_CmdCopyImageToBuffer2(_cmd: VkCommandBuffer, i: &VkCopyImageToBufferInfo2) {
        let (Some(img), Some(buf)) = (unsafe { (i.srcImage.get_ref(), i.dstBuffer.get_ref()) }) else { return };
        let regions: Vec<_> = i.regions().iter().map(buffer_image_copy).collect();
        buffer_image_copies(img, buf, &regions, true);
    }
}

entrypoint! {
    fn borgvk_CmdCopyBufferToImage2(_cmd: VkCommandBuffer, i: &VkCopyBufferToImageInfo2) {
        let (Some(img), Some(buf)) = (unsafe { (i.dstImage.get_ref(), i.srcBuffer.get_ref()) }) else { return };
        let regions: Vec<_> = i.regions().iter().map(buffer_image_copy).collect();
        buffer_image_copies(img, buf, &regions, false);
    }
}

/// A block-compressed image decoded to R32G32B32A32_SFLOAT, with the memory behind it.
pub struct Decoded {
    pub img: Image,
    _buf: Vec<u8>,
}

impl Drop for Decoded {
    fn drop(&mut self) {
        drop(unsafe { Box::from_raw(self.img.mem) });
    }
}

/// `s` is mapped.
pub fn decode_compressed_image(s: &Image) -> Option<Decoded> {
    let pf = fmt::pipe(s.vk.format);
    let sb = Blocks { bd: 1, ..blocks(s.vk.format) };
    let rgba = Blocks { bs: 16, bw: 1, bh: 1, bd: 1 };
    let layers = s.vk.array_layers.max(1);
    let mut o = unsafe { std::ptr::read(s) };
    o.vk.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    let total = (shape(&o).layer_size(rgba) * layers as u64) as usize;
    let mut buf = Vec::new();
    buf.try_reserve_exact(total).ok()?;
    buf.resize(total, 0u8);
    let sh = s.host().unwrap();
    let shr = |e: u32, l: u32| e.checked_shr(l).unwrap_or(0).max(1);
    for layer in 0..layers {
        for l in 0..s.vk.mip_levels {
            let (w, h, d) = (shr(s.vk.extent.width, l), shr(s.vk.extent.height, l), shr(s.vk.extent.depth, l));
            let (soff, wb) = shape(s).level_offset(l, layer, sb);
            let (doff, _) = shape(&o).level_offset(l, layer, rgba);
            let sslice = wb as u64 * h.div_ceil(sb.bh) as u64 * sb.bs as u64;
            let dslice = w as u64 * h as u64 * 16;
            for z in 0..d as u64 {
                let Some(src) = sh.ptr((soff + z * sslice) as usize, sslice as usize) else { continue };
                let at = (doff + z * dslice) as usize;
                let Some(dst) = buf.get_mut(at..at + dslice as usize) else { continue };
                unsafe {
                    util_format_unpack_rgba_rect(
                        pf,
                        dst.as_mut_ptr().cast(),
                        w * 16,
                        src.cast(),
                        wb * sb.bs,
                        w,
                        h,
                    )
                };
            }
        }
    }
    let mem = DeviceMemory { map: buf.as_mut_ptr().cast(), size: total, ..Default::default() };
    (o.mem, o.offset) = (Box::into_raw(Box::new(mem)), 0);
    Some(Decoded { img: o, _buf: buf })
}

fn read_z(pf: pipe_format, t: &Texel) -> f32 {
    if pf == PIPE_FORMAT_Z16_UNORM_S8_UINT {
        return z16_unpack([t[0], t[1]]);
    }
    let mut z = 0.0;
    unsafe { util_format_unpack_z_float(pf, &mut z, t.as_ptr().cast(), 1) };
    z
}

fn write_z(pf: pipe_format, t: &mut Texel, z: f32) {
    if pf == PIPE_FORMAT_Z16_UNORM_S8_UINT {
        t[..2].copy_from_slice(&z16_pack(z));
    } else {
        unsafe { util_format_pack_z_float(pf, t.as_mut_ptr().cast(), &z, 1) };
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
    if src.host().is_none() || dst.host().is_none() {
        return;
    }
    let db = blocks(dst.vk.format);
    if db.bw > 1 || db.bh > 1 {
        return;
    }
    let decoded;
    let sb = blocks(src.vk.format);
    let src = if sb.bw > 1 || sb.bh > 1 {
        let Some(d) = decode_compressed_image(src) else { return };
        decoded = d;
        &decoded.img
    } else {
        src
    };
    let (sh, dh) = (src.host().unwrap(), dst.host().unwrap());
    let (spf, dpf) = (fmt::pipe(src.vk.format), fmt::pipe(dst.vk.format));
    let (sbs, dbs) = (blocks(src.vk.format).bs, db.bs);
    let (is_uint, is_sint) = (fmt::is_pure_uint(spf), fmt::is_pure_sint(spf));
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
            let (sw, sh_) = (shr(src.vk.extent.width, sl), shr(src.vk.extent.height, sl));
            let sd = if s3d { shr(src.vk.extent.depth, sl) } else { 1 };
            let dhgt = shr(dst.vk.extent.height, dl);
            let one = |bs| Blocks { bs, bw: 1, bh: 1, bd: 1 };
            let (soff, sstride) = shape(src).level_offset(sl, if s3d { 0 } else { reg.srcSubresource.baseArrayLayer + l }, one(sbs));
            let (doff, dstride) = shape(dst).level_offset(dl, if d3d { 0 } else { reg.dstSubresource.baseArrayLayer + l }, one(dbs));
            let st = |x: i32, y: i32, z: i32| soff as usize + (((z as u64 * sh_ as u64 + y as u64) * sstride as u64 + x as u64) * sbs as u64) as usize;
            let rd = |off: usize| texel(&sh, off, sbs as usize);
            let w = |d: i32, o: i32, e: f32| if e != 0.0 { (d as f32 + 0.5 - o as f32) / e } else { 0.5 };
            for dz in d0.z.min(d1.z)..d0.z.max(d1.z) {
                let sz = s0.z as f32 + w(dz, d0.z, dez) * sez;
                for dy in d0.y.min(d1.y)..d0.y.max(d1.y) {
                    let sy = s0.y as f32 + w(dy, d0.y, dey) * sey;
                    let drow = (dz as i64 as u64).wrapping_mul(dhgt as u64).wrapping_add(dy as i64 as u64).wrapping_mul(dstride as u64 * dbs as u64);
                    for dx in d0.x.min(d1.x)..d0.x.max(d1.x) {
                        let sx = s0.x as f32 + w(dx, d0.x, dex) * sex;
                        let dt = (doff as usize).wrapping_add(drow.wrapping_add((dx as i64 as u64).wrapping_mul(dbs as u64)) as usize);
                        let Some(mut d) = texel(&dh, dt, dbs as usize) else { continue };
                        let near = || {
                            let c = |f: f32, n: u32| (f.floor() as i32).clamp(0, n as i32 - 1);
                            rd(st(c(sx, sw), c(sy, sh_), c(sz, sd)))
                        };
                        let (xs, xw) = taps(sx - 0.5, sw);
                        let (ys, yw) = taps(sy - 0.5, sh_);
                        let (zs, zw) = taps(sz - 0.5, sd);
                        // The weighted taps of the linear filter, z outermost.
                        let linear = || {
                            (0..8).filter_map(move |i| {
                                let (ix, iy, iz) = (i & 1, i >> 1 & 1, i >> 2);
                                let weight = xw[ix] * yw[iy] * zw[iz];
                                (weight != 0.0).then(|| (rd(st(xs[ix], ys[iy], zs[iz])).unwrap_or([0; 32]), weight))
                            })
                        };
                        let Some(n) = near() else { continue };
                        if stencil {
                            copy_depth_stencil(spf, &n, dpf, &mut d, true);
                        }
                        if depth {
                            if filter == VK_FILTER_NEAREST {
                                copy_depth_stencil(spf, &n, dpf, &mut d, false);
                            } else {
                                let acc = linear().fold(0.0, |acc, (t, wt)| acc + read_z(spf, &t) * wt);
                                write_z(dpf, &mut d, acc);
                            }
                        }
                        if !(depth || stencil) {
                            if filter == VK_FILTER_NEAREST {
                                if is_uint || is_sint {
                                    // The integer bit pattern either way.
                                    let mut v = [0u32; 4];
                                    unsafe {
                                        util_format_unpack_rgba(spf, v.as_mut_ptr().cast(), n.as_ptr().cast(), 1);
                                        util_format_pack_rgba(dpf, d.as_mut_ptr().cast(), v.as_ptr().cast(), 1);
                                    }
                                } else {
                                    let mut v = [0.0; 4];
                                    unpack_rgba_float(src.vk.format, spf, &n, &mut v);
                                    pack_rgba_float(dst.vk.format, dpf, &mut d, &v);
                                }
                            } else {
                                let mut acc = [0f32; 4];
                                for (t, wt) in linear() {
                                    let mut v = [0.0; 4];
                                    unpack_rgba_float(src.vk.format, spf, &t, &mut v);
                                    for c in 0..4 {
                                        acc[c] += v[c] * wt;
                                    }
                                }
                                pack_rgba_float(dst.vk.format, dpf, &mut d, &acc);
                            }
                        }
                        dh.write(dt, &d[..dbs as usize]);
                    }
                }
            }
        }
    }
}

entrypoint! {
    fn borgvk_CmdBlitImage(_cmd: VkCommandBuffer, src: &Image, _sl: VkImageLayout, dst: &Image, _dl: VkImageLayout, _count: u32, regions: &[VkImageBlit], filter: VkFilter) {
        blit(src, dst, regions, filter);
    }
}

entrypoint! {
    fn borgvk_CmdBlitImage2(_cmd: VkCommandBuffer, i: &VkBlitImageInfo2) {
        let (Some(s), Some(d)) = (unsafe { (i.srcImage.get_ref(), i.dstImage.get_ref()) }) else { return };
        let regions: Vec<VkImageBlit> = i.regions()
            .iter()
            .map(|r| VkImageBlit { srcSubresource: r.srcSubresource, srcOffsets: r.srcOffsets, dstSubresource: r.dstSubresource, dstOffsets: r.dstOffsets })
            .collect();
        blit(s, d, &regions, i.filter);
    }
}

/// Averages the samples of one region into a single-sampled image.
pub fn resolve_region(src: &Image, dst: &Image, reg: &VkImageResolve) {
    let (Some(sh), Some(dh)) = (src.host(), dst.host()) else { return };
    let (spf, dpf) = (fmt::pipe(src.vk.format), fmt::pipe(dst.vk.format));
    let texel_blocks = |f: VkFormat| Blocks { bs: blocks(f).bs, bw: 1, bh: 1, bd: 1 };
    let (sb, db) = (texel_blocks(src.vk.format), texel_blocks(dst.vk.format));
    let (sbs, dbs) = (sb.bs as usize, db.bs as usize);
    let samples = src.vk.samples.max(1);
    let plane = shape(src).sample_plane_size(sb) as usize;
    let int_copy = fmt::is_pure_integer(spf) && sbs == dbs;
    let (ss, ds) = (reg.srcSubresource, reg.dstSubresource);
    for l in 0..layers(&src.vk, &ss) {
        let (soff, sstride) = shape(src).level_offset(ss.mipLevel, ss.baseArrayLayer + l, sb);
        let (doff, dstride) = shape(dst).level_offset(ds.mipLevel, ds.baseArrayLayer + l, db);
        for y in 0..reg.extent.height as usize {
            let (sy, dy) = (reg.srcOffset.y as usize + y, reg.dstOffset.y as usize + y);
            for x in 0..reg.extent.width as usize {
                let (sx, dx) = (reg.srcOffset.x as usize + x, reg.dstOffset.x as usize + x);
                let st = soff as usize + (sy * sstride as usize + sx) * sbs;
                let dt = doff as usize + (dy * dstride as usize + dx) * dbs;
                if int_copy {
                    sh.copy(st, &dh, dt, sbs);
                    continue;
                }
                let Some(mut d) = texel(&dh, dt, dbs) else { continue };
                let mut acc = [0.0f32; 4];
                for s in 0..samples as usize {
                    let mut v = [0.0; 4];
                    unpack_rgba_float(src.vk.format, spf, &texel(&sh, st + s * plane, sbs).unwrap_or([0; 32]), &mut v);
                    acc.iter_mut().zip(v).for_each(|(a, v)| *a += v);
                }
                acc.iter_mut().for_each(|a| *a /= samples as f32);
                pack_rgba_float(dst.vk.format, dpf, &mut d, &acc);
                dh.write(dt, &d[..dbs]);
            }
        }
    }
}

entrypoint! {
    fn borgvk_CmdResolveImage(_cmd: VkCommandBuffer, src: &Image, _sl: VkImageLayout, dst: &Image, _dl: VkImageLayout, _count: u32, regions: &[VkImageResolve]) {
        for r in regions {
            resolve_region(src, dst, r);
        }
    }
}

entrypoint! {
    fn borgvk_CmdResolveImage2(_cmd: VkCommandBuffer, i: &VkResolveImageInfo2) {
        let (Some(s), Some(d)) = (unsafe { (i.srcImage.get_ref(), i.dstImage.get_ref()) }) else { return };
        for r in i.regions() {
            let r = VkImageResolve { srcSubresource: r.srcSubresource, srcOffset: r.srcOffset, dstSubresource: r.dstSubresource, dstOffset: r.dstOffset, extent: r.extent };
            resolve_region(s, d, &r);
        }
    }
}

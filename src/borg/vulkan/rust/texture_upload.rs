// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Textures to the Borg as wire packets: whole views in the sampler's layout and format, texel
//! buffers, and the cube's downscaled RGBA8 rows.

use crate::copy::borgvk_decode_compressed_image;
use crate::ffi::*;
use crate::fmt;
use crate::memory::{Buffer, DeviceMemory};
use crate::sampler::Sampler;
use crate::image::Image;
use crate::format::{borgvk_tex_convert, tex_format};
use crate::serial::*;
use crate::state_upload::sampler_desc;
use borg_layout::descriptor::{TexDesc, TexType};
use borg_layout::tex_format::TexFormat;
use borg_layout::tile::texture_index;
use std::ptr::null;

const CHUNK: u32 = BORGVK_TEXG_CHUNK;
const DIM: u32 = BORGVK_TEX_DIM;

fn mapped(mem: *mut DeviceMemory) -> Option<*const u8> {
    unsafe { mem.as_ref() }.filter(|m| !m.map.is_null()).map(|m| m.map.cast::<u8>().cast_const())
}

fn send(data: &[u8], desc: &TexDesc, sampler: *const Sampler) {
    let d = desc.pack();
    for off in (0..data.len() as u32).step_by(CHUNK as usize) {
        let n = (data.len() as u32 - off).min(CHUNK);
        send_texture_chunk(off, &data[off as usize..(off + n) as usize], d[..3].try_into().unwrap(), sampler_desc(unsafe { sampler.as_ref() }));
    }
}

/// One row of the cube's texture, box-filtered from the RGBA8 image down to BORGVK_TEX_DIM.
///
/// # Safety
/// `tex` is a mapped RGBA8 image, `sampler` null or valid.
pub unsafe fn borgvk_send_texture_row(tex: *mut Image, sampler: *const Sampler, dy: i32) {
    let t = unsafe { &*tex };
    let base = unsafe { (*t.mem).map.cast::<u8>().add(t.offset as usize) };
    let (sw, sh) = (t.vk.extent.width, t.vk.extent.height);
    if sw == 0 || sh == 0 {
        return;
    }
    let (sxs, sys) = ((sw / DIM).max(1), (sh / DIM).max(1));
    let mut row = [0u8; DIM as usize * 4];
    for dx in 0..DIM {
        let (mut sum, mut n) = ([0u32; 4], 0u32);
        for sy in (0..sys).map(|oy| dy as u32 * sys + oy).take_while(|&sy| sy < sh) {
            for sx in (0..sxs).map(|ox| dx * sxs + ox).take_while(|&sx| sx < sw) {
                let p = unsafe { base.add((sy * sw * 4 + sx * 4) as usize) };
                for (c, s) in sum.iter_mut().enumerate() {
                    *s += unsafe { *p.add(c) } as u32;
                }
                n += 1;
            }
        }
        let n = n.max(1);
        for c in 0..4 {
            row[dx as usize * 4 + c] = ((sum[c] + n / 2) / n) as u8;
        }
    }
    send_tex_row(dy as u8, &row, sampler_desc(unsafe { sampler.as_ref() }));
}

/// The view's levels and layers, converted and, unless it is one linear level, tiled.
fn texture(view: &vk_image_view, img: &Image, sampler: *const Sampler) -> bool {
    let Some(map) = mapped(img.mem) else { return false };
    let (mut w, mut h) = (img.vk.extent.width, img.vk.extent.height);
    if w == 0 || h == 0 || w > 4096 || h > 4096 {
        return false;
    }
    let (mut layers, mut depth) = (1, 1);
    let ty = match view.view_type {
        VK_IMAGE_VIEW_TYPE_1D_ARRAY => {
            layers = view.layer_count;
            TexType::D1
        }
        VK_IMAGE_VIEW_TYPE_1D => TexType::D1,
        VK_IMAGE_VIEW_TYPE_2D_ARRAY => {
            layers = view.layer_count;
            TexType::D2
        }
        VK_IMAGE_VIEW_TYPE_2D => TexType::D2,
        VK_IMAGE_VIEW_TYPE_CUBE => {
            layers = 6;
            TexType::Cube
        }
        VK_IMAGE_VIEW_TYPE_3D => {
            depth = img.vk.extent.depth;
            TexType::D3
        }
        _ => return false,
    };
    if layers == 0 || layers > 1024 || depth > 1024 {
        return false;
    }
    let stencil = view.aspects & VK_IMAGE_ASPECT_STENCIL_BIT != 0;
    let Some((format, tc)) = tex_format(view.format, stencil) else { return false };
    let (obpp, sbpp) = (tc.obpp, if tc.zstride != 0 { tc.zstride } else { tc.bpp });
    let base = view.base_mip_level;
    let levels = if base < img.vk.mip_levels { view.level_count.min(img.vk.mip_levels - base) } else { 0 };
    if levels == 0 || levels > 13 {
        return false;
    }
    // Every level's size and source offset within a layer.
    let (mut lw, mut lh, mut ld, mut soff, mut sum) = ([0u32; 13], [0u32; 13], [0u32; 13], [0u64; 13], 0u64);
    let shr = |e: u32, l: u32| e.checked_shr(l).unwrap_or(0).max(1);
    for l in 0..img.vk.mip_levels {
        let cw = shr(img.vk.extent.width, l);
        let ch = if ty == TexType::D1 { 1 } else { shr(img.vk.extent.height, l) };
        let cd = if ty == TexType::D3 { shr(img.vk.extent.depth, l) } else { 1 };
        if l >= base && l < base + levels {
            let i = (l - base) as usize;
            (lw[i], lh[i], ld[i], soff[i]) = (cw, ch, cd, sum);
        }
        sum += cw as u64 * ch as u64 * cd as u64 * sbpp as u64;
    }
    (w, h) = (lw[0], lh[0]);
    if ty == TexType::D3 {
        depth = ld[0];
    }
    let src = unsafe { map.add(img.offset as usize) };
    let layer_src = |layer: u32, l: usize| unsafe { src.add((view.base_array_layer as u64 + layer as u64) as usize * sum as usize + soff[l] as usize) };
    let tiled = levels > 1 || layers > 1 || depth > 1 || ty == TexType::Cube;
    let mut offs = [0u32; 12];
    let (out, pitch) = if !tiled {
        let mut out = vec![0u8; (w * h * obpp).next_multiple_of(4) as usize];
        unsafe { borgvk_tex_convert(&tc, out.as_mut_ptr(), layer_src(0, 0), w * h) };
        (out, w * obpp)
    } else {
        let mut ltotal = 0u32;
        for l in 0..levels as usize {
            if l > 0 {
                offs[l - 1] = ltotal;
            }
            let size = (lw[l].div_ceil(4) * lh[l].div_ceil(4)).wrapping_mul(16 * obpp).wrapping_mul(ld[l]);
            ltotal = ltotal.wrapping_add(size);
        }
        let total = ltotal.wrapping_mul(layers);
        // The board's texture region is 512 KB, the direct simulator's 6 MB.
        let direct = std::env::var_os("BORGVK_SIM_DIRECT").is_some_and(|d| !d.is_empty());
        if total > if direct { 0x60_0000 } else { 0x7_0000 } {
            return false;
        }
        let mut out = vec![0u8; total as usize];
        let mut lin = vec![0u8; w as usize * h as usize * depth as usize * obpp as usize];
        let ob = obpp as usize;
        for layer in 0..layers {
            for l in 0..levels as usize {
                let tw = lw[l].div_ceil(4);
                let lev = layer as usize * ltotal as usize + if l > 0 { offs[l - 1] as usize } else { 0 };
                unsafe { borgvk_tex_convert(&tc, lin.as_mut_ptr(), layer_src(layer, l), lw[l] * lh[l] * ld[l]) };
                for z in 0..ld[l] {
                    let slice = lev + z as usize * tw as usize * lh[l].div_ceil(4) as usize * 16 * ob;
                    for y in 0..lh[l] {
                        for x in 0..lw[l] {
                            let t = &lin[((z as usize * lh[l] as usize + y as usize) * lw[l] as usize + x as usize) * ob..][..ob];
                            let (tile, ti) = texture_index(lw[l], x, y);
                            if ob == 16 {
                                // 16-byte texels: bytes 0-7 of the tile's texels, then 8-15.
                                let at = slice + tile * 256 + ti * 8;
                                out[at..at + 8].copy_from_slice(&t[..8]);
                                out[at + 128..at + 136].copy_from_slice(&t[8..]);
                            } else {
                                let at = slice + (tile * 16 + ti) * ob;
                                out[at..at + ob].copy_from_slice(t);
                            }
                        }
                    }
                }
            }
        }
        (out, ltotal)
    };
    let mut sw = [view.swizzle.r, view.swizzle.g, view.swizzle.b, view.swizzle.a];
    if tc.sch != 0 && !fmt::has_alpha(fmt::pipe(view.format)) {
        for (i, s) in sw.iter_mut().enumerate() {
            let from = if *s == VK_COMPONENT_SWIZZLE_IDENTITY { VK_COMPONENT_SWIZZLE_R + i as u32 } else { *s };
            if from == VK_COMPONENT_SWIZZLE_A {
                *s = VK_COMPONENT_SWIZZLE_ONE;
            }
        }
    }
    let desc = TexDesc {
        width: w,
        height: h,
        ty,
        linear: !tiled,
        depth_or_layers: if ty == TexType::D3 { depth } else { layers },
        levels,
        format,
        swizzle: sw,
        pitch,
    };
    if levels > 1 {
        send_texture_levels(&offs);
    }
    send(&out, &desc, sampler);
    true
}

/// # Safety
/// `view` is valid; `img` and `sampler` are null or valid.
pub unsafe fn borgvk_send_generic_texture(view: *const vk_image_view, img: *const Image, sampler: *const Sampler) -> bool {
    let (view, Some(img)) = (unsafe { &*view }, unsafe { img.as_ref() }) else { return false };
    if mapped(img.mem).is_none() {
        return false;
    }
    let b = crate::image::blocks(img.vk.format);
    if b.bw <= 1 && b.bh <= 1 {
        return texture(view, img, sampler);
    }
    // Block-compressed: decoded first.
    let (mut dimg, mut dmem) = unsafe { (std::mem::zeroed::<Image>(), std::mem::zeroed::<DeviceMemory>()) };
    let buf = unsafe { borgvk_decode_compressed_image(img, &mut dimg, &mut dmem) };
    if buf.is_null() {
        return false;
    }
    let mut dview = unsafe { std::ptr::read(view) };
    dview.format = dimg.vk.format;
    let ok = texture(&dview, &dimg, sampler);
    unsafe { free(buf) };
    ok
}

/// A texel buffer as a linear 1D texture.
///
/// # Safety
/// `bv` is valid.
pub unsafe fn borgvk_send_buffer_texture(bv: *const vk_buffer_view) -> bool {
    let bv = unsafe { &*bv };
    let format = match bv.format {
        VK_FORMAT_R32_UINT => TexFormat::R32_UINT,
        VK_FORMAT_R32_SFLOAT => TexFormat::R32_SFLOAT,
        _ => return false,
    };
    let buf = unsafe { &*bv.buffer.cast::<Buffer>() };
    let Some(map) = mapped(buf.mem) else { return false };
    let w = (bv.range / 4) as u32;
    if w == 0 || w > 65536 {
        return false;
    }
    let src = unsafe { std::slice::from_raw_parts(map.add((buf.offset + bv.offset) as usize), w as usize * 4) };
    let desc = TexDesc { width: w, height: 1, ty: TexType::D1, linear: true, depth_or_layers: 1, levels: 1, format, swizzle: [0; 4], pitch: w * 4 };
    send(src, &desc, null());
    true
}

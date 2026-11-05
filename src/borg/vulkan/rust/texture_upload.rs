// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Textures to the Borg as wire packets: whole views in the sampler's layout and format, texel
//! buffers, and the cube's downscaled RGBA8 rows.

use crate::copy::decode_compressed_image;
use crate::ffi::*;
use crate::fmt;
use crate::memory::ViewBuffer;
use crate::image::Image;
use crate::format::{src_len, tex_convert, tex_format};
use crate::wire::*;
use crate::state_upload::DEFAULT_SAMPLER;
use borg_layout::descriptor::{TexDesc, TexType};
use borg_layout::tex_format::TexFormat;
use borg_layout::tile::texture_index;

const CHUNK: u32 = BORGVK_TEXG_CHUNK;

fn send(data: &[u8], desc: &TexDesc, sampler: &[u32; 4]) {
    let d = desc.pack();
    if data.is_empty() {
        send_texture_chunk(0, data, d[..3].try_into().unwrap(), sampler);   // the descriptor alone
    }
    for off in (0..data.len() as u32).step_by(CHUNK as usize) {
        let n = (data.len() as u32 - off).min(CHUNK);
        send_texture_chunk(off, &data[off as usize..(off + n) as usize], d[..3].try_into().unwrap(), sampler);
    }
}

/// The view's levels and layers, converted and, unless it is one linear level, tiled.
/// A view in the sampler's layout: its texels, descriptor and level offsets.
pub struct Texture {
    pub data: Vec<u8>,
    pub desc: TexDesc,
    pub offs: [u32; 12],
}

fn build(view: &vk_image_view, img: &Image) -> Option<Texture> {
    let mut host = img.host()?;
    let mut lin = crate::texels::linear_copy(img);
    if let Some(l) = &mut lin {
        host = crate::memory::Host::of_bytes(l);
    }
    let (mut w, mut h) = (img.vk.extent.width, img.vk.extent.height);
    if w == 0 || h == 0 || w > 4096 || h > 4096 {
        return None;
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
        _ => return None,
    };
    let samples = img.vk.samples.max(1);
    if samples > 1 && ty == TexType::D2 {
        layers *= samples;
    }
    if layers == 0 || layers > 1024 || depth > 1024 {
        return None;
    }
    let stencil = view.aspects & VK_IMAGE_ASPECT_STENCIL_BIT != 0;
    let (format, tc) = tex_format(view.format, stencil)?;
    let (obpp, sbpp) = (tc.obpp, if tc.zstride != 0 { tc.zstride } else { tc.bpp });
    let base = view.base_mip_level;
    let levels = if base < img.vk.mip_levels { view.level_count.min(img.vk.mip_levels - base) } else { 0 };
    if levels == 0 || levels > 13 {
        return None;
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
    let layer_src = |layer: u32, l: usize| (view.base_array_layer as u64 * samples as u64 + layer as u64) as usize * sum as usize + soff[l] as usize;
    let convert = |out: &mut [u8], from: usize, n: u32| {
        host.with_bytes(from, src_len(&tc, n as usize), |s| tex_convert(&tc, out, s, n as usize));
    };
    let tiled = levels > 1 || layers > 1 || depth > 1 || ty == TexType::Cube;
    let mut offs = [0u32; 12];
    let (out, pitch) = if !tiled {
        let mut out = vec![0u8; (w * h * obpp).next_multiple_of(4) as usize];
        convert(&mut out, layer_src(0, 0), w * h);
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
        // The board's texture region is 512 KB, the direct simulator's 6 MB. A larger texture sends
        // its descriptor alone, so size queries still answer; its texels read whatever is there.
        let direct = std::env::var_os("BORGVK_SIM_DIRECT").is_some_and(|d| !d.is_empty());
        let fits = total <= if direct { 0x60_0000 } else { 0x7_0000 };
        if !fits {
            crate::util::log(crate::util::MESA_LOG_WARN, format!("borgvk: a {total}-byte texture exceeds the texture region"));
        }
        let mut out = vec![0u8; if fits { total as usize } else { 0 }];
        let mut lin = vec![0u8; w as usize * h as usize * depth as usize * obpp as usize];
        let ob = obpp as usize;
        for layer in 0..if fits { layers } else { 0 } {
            for l in 0..levels as usize {
                let tw = lw[l].div_ceil(4);
                let lev = layer as usize * ltotal as usize + if l > 0 { offs[l - 1] as usize } else { 0 };
                convert(&mut lin, layer_src(layer, l), lw[l] * lh[l] * ld[l]);
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
    Some(Texture { data: out, desc, offs })
}

/// A sampled view, block-compressed images decoded first.
pub fn generic_texture(view: &vk_image_view, img: &Image) -> Option<Texture> {
    img.host()?;
    let b = crate::image::blocks(img.vk.format);
    if b.bw <= 1 && b.bh <= 1 {
        return build(view, img);
    }
    let d = decode_compressed_image(img)?;
    let mut dview = *view;
    dview.format = d.img.vk.format;
    build(&dview, &d.img)
}

/// The descriptor of a view the sampler reads where it is: a tiled colour image of one sample
/// in device memory, in a format it takes unconverted.
pub fn in_place(view: &vk_image_view, img: &Image) -> Option<[u32; 16]> {
    let gpu = img.memory()?.gpu;
    let texels = crate::texels::Texels::of(img)?;
    if gpu == 0 || img.vk.samples > 1 {
        return None;
    }
    let unconverted = || {
        let (format, tc) = tex_format(view.format, false)?;
        (tc.sch == 0 && tc.pack16 == 0 && tc.zstride == 0 && tc.obpp == crate::image::blocks(img.vk.format).bs).then_some(format)
    };
    let Some((t, planes)) = texels.tiled() else { return linear_in_place(view, img, gpu, unconverted()?) };
    // Depth planes hold D16 or float texels (D24 as float too: not the sampler's format).
    let stencil = view.aspects & VK_IMAGE_ASPECT_STENCIL_BIT != 0;
    let (p, format) = match planes {
        crate::texels::Planes::Color => (0, unconverted()?),
        crate::texels::Planes::Stencil => (0, TexFormat::R8_UINT),
        crate::texels::Planes::Depth { stencil: true, .. } if stencil => (1, TexFormat::R8_UINT),
        crate::texels::Planes::Depth { z16: true, .. } if !stencil => (0, TexFormat::D16_UNORM),
        crate::texels::Planes::Depth { .. } if !stencil && matches!(img.vk.format, VK_FORMAT_D32_SFLOAT | VK_FORMAT_D32_SFLOAT_S8_UINT) => (0, TexFormat::D32_SFLOAT),
        _ => return None,
    };
    let is3d = img.vk.image_type == VK_IMAGE_TYPE_3D;
    let (ty, layers) = match view.view_type {
        VK_IMAGE_VIEW_TYPE_1D_ARRAY => (TexType::D1, view.layer_count),
        VK_IMAGE_VIEW_TYPE_1D => (TexType::D1, 1),
        VK_IMAGE_VIEW_TYPE_2D_ARRAY => (TexType::D2, view.layer_count),
        VK_IMAGE_VIEW_TYPE_2D => (TexType::D2, 1),
        VK_IMAGE_VIEW_TYPE_CUBE => (TexType::Cube, 6),
        VK_IMAGE_VIEW_TYPE_3D => (TexType::D3, 1),
        _ => return None,
    };
    // A 2D view of a 3D image's slice is not a layer of the layout.
    if is3d != (ty == TexType::D3) {
        return None;
    }
    let base = view.base_mip_level;
    let levels = if base < img.vk.mip_levels { view.level_count.min(img.vk.mip_levels - base) } else { 0 };
    let shr = |e: u32, l: u32| e.checked_shr(l).unwrap_or(0).max(1);
    let (w, h) = (shr(img.vk.extent.width, base), if ty == TexType::D1 { 1 } else { shr(img.vk.extent.height, base) });
    let depth = if is3d { shr(img.vk.extent.depth, base) } else { layers };
    if levels == 0 || levels > 13 || w > 4096 || h > 4096 || depth == 0 || depth > 1024 {
        return None;
    }
    let sw = [view.swizzle.r, view.swizzle.g, view.swizzle.b, view.swizzle.a];
    let desc = TexDesc { width: w, height: h, ty, linear: false, depth_or_layers: depth, levels, format, swizzle: sw, pitch: t.layer_size(p) as u32 };
    let mut d = [0u32; 16];
    d[0] = gpu + (img.offset + t.level_base(p, base, if is3d { 0 } else { view.base_array_layer })) as u32;
    d[1..4].copy_from_slice(&desc.pack());
    for l in 1..levels {
        d[3 + l as usize] = (t.level_offset(p, base + l) - t.level_offset(p, base)) as u32;
    }
    Some(d)
}

/// A linear image's view of one level and layer, read where it is with its row pitch.
fn linear_in_place(view: &vk_image_view, img: &Image, gpu: u32, format: TexFormat) -> Option<[u32; 16]> {
    let base = view.base_mip_level;
    let one = base < img.vk.mip_levels && view.level_count.min(img.vk.mip_levels - base) == 1;
    let ty = match view.view_type {
        VK_IMAGE_VIEW_TYPE_1D => TexType::D1,
        VK_IMAGE_VIEW_TYPE_2D => TexType::D2,
        _ => return None,
    };
    if !one || img.vk.image_type == VK_IMAGE_TYPE_3D {
        return None;
    }
    let blk = crate::image::blocks(img.vk.format);
    let (off, stride) = crate::image::shape(img).level_offset(base, view.base_array_layer, blk);
    let shr = |e: u32| e.checked_shr(base).unwrap_or(0).max(1);
    let (w, h) = (shr(img.vk.extent.width), if ty == TexType::D1 { 1 } else { shr(img.vk.extent.height) });
    let addr = gpu + (img.offset + off) as u32;
    if w > 4096 || h > 4096 || blk.bs >= 4 && addr & 3 != 0 {
        return None;
    }
    let sw = [view.swizzle.r, view.swizzle.g, view.swizzle.b, view.swizzle.a];
    let desc = TexDesc { width: w, height: h, ty, linear: true, depth_or_layers: 1, levels: 1, format, swizzle: sw, pitch: stride * blk.bs };
    let mut d = [0u32; 16];
    d[0] = addr;
    d[1..4].copy_from_slice(&desc.pack());
    Some(d)
}

/// Sends the view's texels and descriptor, or only the descriptor of a texture in place; returns
/// the texel bytes sent.
pub fn send_generic_texture(view: &vk_image_view, img: Option<&Image>, sampler: &[u32; 4]) -> Option<u32> {
    if let Some(d) = img.and_then(|i| in_place(view, i)) {
        send_texture_desc(&d, sampler);
        return Some(0);
    }
    let t = img.and_then(|i| generic_texture(view, i))?;
    if t.desc.levels > 1 {
        send_texture_levels(&t.offs);
    }
    send(&t.data, &t.desc, sampler);
    Some(t.data.len() as u32)
}

/// A texel buffer as a linear 1D texture.
pub fn send_buffer_texture(bv: &vk_buffer_view) -> Option<u32> {
    let (format, tc) = tex_format(bv.format, false)?;
    let host = bv.buffer().and_then(|b| b.host())?;
    let bs = fmt::blocksize(fmt::pipe(bv.format)) as u64;
    let w = if bs == 0 { 0 } else { (bv.range / bs) as u32 };
    if w == 0 || w > 65536 {
        return None;
    }
    // In the device heap and unconverted: read where it is.
    let gpu = bv.buffer().and_then(|b| b.memory()).map_or(0, |m| m.gpu);
    if gpu != 0 && tc.sch == 0 && tc.pack16 == 0 && tc.zstride == 0 && tc.obpp as u64 == bs {
        let desc = TexDesc { width: w, height: 1, ty: TexType::D1, linear: true, depth_or_layers: 1, levels: 1, format, swizzle: [0; 4], pitch: w * tc.obpp };
        let mut d = [0u32; 16];
        d[0] = gpu + (bv.buffer()?.offset + bv.offset) as u32;
        d[1..4].copy_from_slice(&desc.pack());
        send_texture_desc(&d, &DEFAULT_SAMPLER);
        return Some(0);
    }
    let mut out = vec![0u8; (w * tc.obpp).next_multiple_of(4) as usize];
    host.with_bytes(bv.offset as usize, src_len(&tc, w as usize), |src| tex_convert(&tc, &mut out, src, w as usize));
    let one = tc.sch != 0 && !fmt::has_alpha(fmt::pipe(bv.format));
    let swizzle = [0, 0, 0, if one { VK_COMPONENT_SWIZZLE_ONE } else { 0 }];
    let desc = TexDesc { width: w, height: 1, ty: TexType::D1, linear: true, depth_or_layers: 1, levels: 1, format, swizzle, pitch: w * tc.obpp };
    send(&out, &desc, &DEFAULT_SAMPLER);
    Some(out.len() as u32)
}

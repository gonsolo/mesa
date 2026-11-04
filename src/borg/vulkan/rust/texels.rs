// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! An image's texels in memory, read and written in the image's format. Optimal images use the
//! GPU's tiled layout ([`Tiled`]), depth and stencil as two planes: D16 as 16 bits, D24 and D32
//! as FP32 (how the hardware tests and stores them), stencil as a byte. Linear, block-compressed
//! and odd-sized formats stay row-major.

use crate::ffi::*;
use crate::fmt;
use crate::image::{blocks, Image};
use crate::memory::Host;
use borg_layout::image::{Blocks, Shape, Tiled};

const Z24_MAX: f64 = 16777215.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Planes {
    Color,
    /// Depth in plane 0, two bytes (D16) or four (FP32); stencil in plane 1 when it has one.
    Depth { z16: bool, stencil: bool },
    Stencil,
}

#[derive(Clone, Copy)]
enum Kind {
    Linear(Shape, Blocks),
    Tiled(Tiled, Planes),
}

/// Whether and how an image of format `f` is stored tiled. Storage images stay linear: shaders
/// address them by rows.
pub fn planes(f: VkFormat, tiling: VkImageTiling, usage: VkImageUsageFlags) -> Option<Planes> {
    let pf = fmt::pipe(f);
    let b = blocks(f);
    if tiling != VK_IMAGE_TILING_OPTIMAL || usage & VK_IMAGE_USAGE_STORAGE_BIT != 0 || fmt::is_compressed(pf) || b.bw * b.bh * b.bd != 1 {
        return None;
    }
    let (z, s) = (fmt::vk_has_depth(f), fmt::vk_has_stencil(f));
    match (z, s) {
        (true, _) => Some(Planes::Depth { z16: !crate::format::depth_is_d32(f), stencil: s }),
        (false, true) => Some(Planes::Stencil),
        _ => matches!(b.bs, 1 | 2 | 4 | 8 | 16).then_some(Planes::Color),
    }
}

/// The tiled layout of an image with `planes`.
pub fn tiled_layout(shape: Shape, f: VkFormat, planes: Planes) -> Tiled {
    let one = |bs| Blocks { bs, bw: 1, bh: 1, bd: 1 };
    let p = match planes {
        Planes::Color => [blocks(f), one(0)],
        Planes::Depth { z16, stencil } => [one(if z16 { 2 } else { 4 }), one(stencil as u32)],
        Planes::Stencil => [one(1), one(0)],
    };
    Tiled { shape, planes: p, plane1: 0 }
}

pub struct Texels<'a> {
    host: Host<'a>,
    pf: pipe_format,
    bs: usize,
    is3d: bool,
    extent: [u32; 3],
    kind: Kind,
}

impl<'a> Texels<'a> {
    pub fn of(img: &'a Image) -> Option<Self> {
        let shape = crate::image::shape(img);
        let kind = match planes(img.vk.format, img.vk.tiling, img.vk.usage as VkImageUsageFlags) {
            Some(p) => Kind::Tiled(Tiled { plane1: img.plane1, ..tiled_layout(shape, img.vk.format, p) }, p),
            None => Kind::Linear(shape, blocks(img.vk.format)),
        };
        let e = img.vk.extent;
        Some(Texels {
            host: img.host()?,
            pf: fmt::pipe(img.vk.format),
            bs: blocks(img.vk.format).bs as usize,
            is3d: img.vk.image_type == VK_IMAGE_TYPE_3D,
            extent: [e.width, e.height, e.depth],
            kind,
        })
    }

    /// Bytes of one texel (block) in the image's format.
    pub fn bs(&self) -> usize {
        self.bs
    }

    pub fn tiled(&self) -> Option<(Tiled, Planes)> {
        match self.kind {
            Kind::Tiled(t, p) => Some((t, p)),
            Kind::Linear(..) => None,
        }
    }

    fn linear(&self, s: &Shape, b: Blocks, level: u32, layer: u32, [x, y, z]: [u32; 3], smp: u32) -> usize {
        let (off, stride) = s.level_offset(level, if self.is3d { 0 } else { layer }, b);
        let w = |e: u32, block: u32| e.checked_shr(level).unwrap_or(0).max(1).div_ceil(block) as u64;
        let row = stride as u64 * b.bs as u64;
        let slice = row * w(self.extent[1], b.bh);
        (off + z as u64 * slice + y as u64 * row + x as u64 * b.bs as u64 + smp as u64 * s.sample_plane_size(b)) as usize
    }

    fn at(&self, t: &Tiled, p: usize, level: u32, layer: u32, xyz: [u32; 3], smp: u32) -> usize {
        t.texel(p, level, if self.is3d { 0 } else { layer }, xyz, smp) as usize
    }

    /// Texel `xyz` (in blocks; z is the slice of a 3D image) of `level`, `layer` and sample `smp`
    /// into `out`, which holds at least `bs()` bytes.
    pub fn read(&self, level: u32, layer: u32, xyz: [u32; 3], smp: u32, out: &mut [u8]) -> bool {
        let out = &mut out[..self.bs];
        let h = &self.host;
        match &self.kind {
            Kind::Linear(s, b) => h.read(self.linear(s, *b, level, layer, xyz, smp), out),
            Kind::Tiled(t, Planes::Color) => {
                let o = self.at(t, 0, level, layer, xyz, smp);
                if self.bs == 16 {
                    let (a, b) = out.split_at_mut(8);
                    h.read(o, a) && h.read(o + 128, b)
                } else {
                    h.read(o, out)
                }
            }
            Kind::Tiled(t, Planes::Stencil) => h.read(self.at(t, 0, level, layer, xyz, smp), out),
            Kind::Tiled(t, Planes::Depth { z16, stencil }) => {
                let mut zb = [0u8; 4];
                let n = if *z16 { 2 } else { 4 };
                if !h.read(self.at(t, 0, level, layer, xyz, smp), &mut zb[..n]) {
                    return false;
                }
                let mut s = [0u8];
                if *stencil && !h.read(self.at(t, 1, level, layer, xyz, smp), &mut s) {
                    return false;
                }
                out.fill(0);
                match self.pf {
                    PIPE_FORMAT_Z16_UNORM => out.copy_from_slice(&zb[..2]),
                    PIPE_FORMAT_Z16_UNORM_S8_UINT => {
                        out[..2].copy_from_slice(&zb[..2]);
                        out[2] = s[0];
                    }
                    PIPE_FORMAT_Z32_FLOAT => out.copy_from_slice(&zb),
                    pf => {
                        let z = f32::from_le_bytes(zb);
                        if pf == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT {
                            out[..4].copy_from_slice(&zb);
                        } else {
                            let u = (z.clamp(0.0, 1.0) as f64 * Z24_MAX).round() as u32;
                            out[..4].copy_from_slice(&(if pf == PIPE_FORMAT_X8Z24_UNORM || pf == PIPE_FORMAT_S8_UINT_Z24_UNORM { u << 8 } else { u }).to_le_bytes());
                        }
                        if *stencil {
                            fmt::pack_s(pf, out, s[0]);
                        }
                    }
                }
                true
            }
        }
    }

    /// The inverse of [`read`](Self::read): `data` holds a texel in the image's format.
    pub fn write(&self, level: u32, layer: u32, xyz: [u32; 3], smp: u32, data: &[u8]) -> bool {
        let data = &data[..self.bs];
        let h = &self.host;
        match &self.kind {
            Kind::Linear(s, b) => h.write(self.linear(s, *b, level, layer, xyz, smp), data),
            Kind::Tiled(t, Planes::Color) => {
                let o = self.at(t, 0, level, layer, xyz, smp);
                if self.bs == 16 {
                    h.write(o, &data[..8]) && h.write(o + 128, &data[8..])
                } else {
                    h.write(o, data)
                }
            }
            Kind::Tiled(t, Planes::Stencil) => h.write(self.at(t, 0, level, layer, xyz, smp), data),
            Kind::Tiled(t, Planes::Depth { z16, stencil }) => {
                let (zb, s): ([u8; 4], u8) = match self.pf {
                    PIPE_FORMAT_Z16_UNORM => ([data[0], data[1], 0, 0], 0),
                    PIPE_FORMAT_Z16_UNORM_S8_UINT => ([data[0], data[1], 0, 0], data[2]),
                    PIPE_FORMAT_Z32_FLOAT => (data[..4].try_into().unwrap(), 0),
                    pf => {
                        let w = u32::from_le_bytes(data[..4].try_into().unwrap());
                        let z = if pf == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT {
                            f32::from_bits(w)
                        } else {
                            let u = if pf == PIPE_FORMAT_X8Z24_UNORM || pf == PIPE_FORMAT_S8_UINT_Z24_UNORM { w >> 8 } else { w & 0xFF_FFFF };
                            (u as f64 / Z24_MAX) as f32
                        };
                        (z.to_le_bytes(), if *stencil { fmt::unpack_s(pf, data) } else { 0 })
                    }
                };
                let n = if *z16 { 2 } else { 4 };
                h.write(self.at(t, 0, level, layer, xyz, smp), &zb[..n]) && (!*stencil || h.write(self.at(t, 1, level, layer, xyz, smp), &[s]))
            }
        }
    }
}

/// Calls `f` with every texel of a tiled image and its offset in the linear layout.
fn walk_linear(img: &Image, mut f: impl FnMut(&Texels, (u32, u32, [u32; 3], u32), usize)) -> Option<usize> {
    let t = Texels::of(img).filter(|t| t.tiled().is_some())?;
    let (s, b) = (crate::image::shape(img), blocks(img.vk.format));
    let e = img.vk.extent;
    let plane = s.sample_plane_size(b) as usize;
    for layer in 0..s.layers.max(1) {
        for level in 0..s.levels.max(1) {
            let d = |v: u32| v.checked_shr(level).unwrap_or(0).max(1);
            let (off, stride) = s.level_offset(level, layer, b);
            let (w, h, dd) = (d(e.width), d(e.height), if t.is3d { d(e.depth) } else { 1 });
            for smp in 0..s.samples.max(1) {
                for z in 0..dd {
                    for y in 0..h {
                        for x in 0..w {
                            f(&t, (level, layer, [x, y, z], smp), off as usize + smp as usize * plane + ((z * h + y) * stride + x) as usize * t.bs);
                        }
                    }
                }
            }
        }
    }
    Some(s.size(b) as usize)
}

/// A tiled image's texels in the linear layout, for code that reads linear images; None when
/// the image is linear already.
pub fn linear_copy(img: &Image) -> Option<Vec<u8>> {
    let mut lin = vec![0u8; Texels::of(img).filter(|t| t.tiled().is_some()).map(|_| crate::image::shape(img).size(blocks(img.vk.format)))? as usize];
    let mut texel = [0u8; 16];
    walk_linear(img, |t, (level, layer, xyz, smp), o| {
        if t.read(level, layer, xyz, smp, &mut texel) {
            lin[o..o + t.bs].copy_from_slice(&texel[..t.bs]);
        }
    })?;
    Some(lin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d24_survives_the_float_plane() {
        for u in [0u32, 1, 2, 0x7F_FFFF, 0x80_0000, 0xFF_FFFE, 0xFF_FFFF, 12_345_678] {
            let z = (u as f64 / Z24_MAX) as f32;
            assert_eq!((z as f64 * Z24_MAX).round() as u32, u);
        }
    }

    #[test]
    fn which_images_tile() {
        let (opt, u) = (VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT);
        assert_eq!(planes(VK_FORMAT_R8G8B8A8_UNORM, opt, u), Some(Planes::Color));
        assert_eq!(planes(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_LINEAR, u), None);
        assert_eq!(planes(VK_FORMAT_R8G8B8A8_UNORM, opt, VK_IMAGE_USAGE_STORAGE_BIT), None);
        assert_eq!(planes(VK_FORMAT_R8G8B8_UNORM, opt, u), None);
        assert_eq!(planes(VK_FORMAT_D24_UNORM_S8_UINT, opt, u), Some(Planes::Depth { z16: false, stencil: true }));
        assert_eq!(planes(VK_FORMAT_D16_UNORM, opt, u), Some(Planes::Depth { z16: true, stencil: false }));
        assert_eq!(planes(VK_FORMAT_S8_UINT, opt, u), Some(Planes::Stencil));
    }
}

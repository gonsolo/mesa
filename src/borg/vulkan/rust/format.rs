// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::ffi::*;
use crate::fmt;
use borg_layout::tex_format::TexFormat;

/// How the vertex fetch unit reads one attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fetch {
    pub code: u32,
    pub swz: u32,
    pub read_bytes: u32,
}

const A_ONE: u32 = 2 << 29;

pub fn vertex_fetch(f: VkFormat) -> Option<Fetch> {
    let (code, swz, read_bytes) = match f {
        VK_FORMAT_R8_UNORM => (1, 0, 1),
        VK_FORMAT_R8_SNORM => (2, 0, 1),
        VK_FORMAT_R8_UINT => (3, 0, 1),
        VK_FORMAT_R8_SINT => (4, 0, 1),
        VK_FORMAT_R8G8_UNORM => (6, 0, 2),
        VK_FORMAT_R8G8_SNORM => (7, 0, 2),
        VK_FORMAT_R8G8_UINT => (8, 0, 2),
        VK_FORMAT_R8G8_SINT => (9, 0, 2),
        VK_FORMAT_R8G8B8_UNORM => (11, A_ONE, 4),
        VK_FORMAT_R8G8B8_SNORM => (12, A_ONE, 4),
        VK_FORMAT_R8G8B8_UINT => (13, A_ONE, 4),
        VK_FORMAT_R8G8B8_SINT => (14, A_ONE, 4),
        VK_FORMAT_R8G8B8A8_UNORM | VK_FORMAT_A8B8G8R8_UNORM_PACK32 => (11, 0, 4),
        VK_FORMAT_R8G8B8A8_SNORM | VK_FORMAT_A8B8G8R8_SNORM_PACK32 => (12, 0, 4),
        VK_FORMAT_R8G8B8A8_UINT | VK_FORMAT_A8B8G8R8_UINT_PACK32 => (13, 0, 4),
        VK_FORMAT_R8G8B8A8_SINT | VK_FORMAT_A8B8G8R8_SINT_PACK32 => (14, 0, 4),
        VK_FORMAT_B8G8R8A8_UNORM => (16, 0, 4),
        VK_FORMAT_R16_UNORM => (18, 0, 2),
        VK_FORMAT_R16_SNORM => (19, 0, 2),
        VK_FORMAT_R16_UINT => (20, 0, 2),
        VK_FORMAT_R16_SINT => (21, 0, 2),
        VK_FORMAT_R16_SFLOAT => (22, 0, 2),
        VK_FORMAT_R16G16_UNORM => (23, 0, 4),
        VK_FORMAT_R16G16_SNORM => (24, 0, 4),
        VK_FORMAT_R16G16_UINT => (25, 0, 4),
        VK_FORMAT_R16G16_SINT => (26, 0, 4),
        VK_FORMAT_R16G16_SFLOAT => (27, 0, 4),
        VK_FORMAT_R16G16B16_UNORM => (28, A_ONE, 8),
        VK_FORMAT_R16G16B16_SNORM => (29, A_ONE, 8),
        VK_FORMAT_R16G16B16_UINT => (30, A_ONE, 8),
        VK_FORMAT_R16G16B16_SINT => (31, A_ONE, 8),
        VK_FORMAT_R16G16B16_SFLOAT => (32, A_ONE, 8),
        VK_FORMAT_R16G16B16A16_UNORM => (28, 0, 8),
        VK_FORMAT_R16G16B16A16_SNORM => (29, 0, 8),
        VK_FORMAT_R16G16B16A16_UINT => (30, 0, 8),
        VK_FORMAT_R16G16B16A16_SINT => (31, 0, 8),
        VK_FORMAT_R16G16B16A16_SFLOAT => (32, 0, 8),
        VK_FORMAT_R32_UINT => (33, 0, 4),
        VK_FORMAT_R32_SINT => (34, 0, 4),
        VK_FORMAT_R32_SFLOAT => (35, 0, 4),
        VK_FORMAT_R32G32_UINT => (36, 0, 8),
        VK_FORMAT_R32G32_SINT => (37, 0, 8),
        VK_FORMAT_R32G32_SFLOAT => (38, 0, 8),
        VK_FORMAT_R32G32B32_UINT => (39, A_ONE, 16),
        VK_FORMAT_R32G32B32_SINT => (40, A_ONE, 16),
        VK_FORMAT_R32G32B32_SFLOAT => (41, A_ONE, 16),
        VK_FORMAT_R32G32B32A32_UINT => (39, 0, 16),
        VK_FORMAT_R32G32B32A32_SINT => (40, 0, 16),
        VK_FORMAT_R32G32B32A32_SFLOAT => (41, 0, 16),
        VK_FORMAT_A2B10G10R10_UNORM_PACK32 => (42, 0, 4),
        VK_FORMAT_A2B10G10R10_UINT_PACK32 => (43, 0, 4),
        _ => return None,
    };
    Some(Fetch { code, swz, read_bytes })
}

pub fn bytes_packed(f: VkFormat) -> bool {
    f == VK_FORMAT_R8G8B8A8_UINT || f == VK_FORMAT_R8G8B8A8_SINT
}

pub fn raw32(f: VkFormat) -> bool {
    f == VK_FORMAT_R32_UINT || f == VK_FORMAT_R32_SINT || f == VK_FORMAT_R32_SFLOAT
}

pub fn half1(f: VkFormat) -> bool {
    f == VK_FORMAT_R16_SFLOAT
}

pub fn half2(f: VkFormat) -> bool {
    f == VK_FORMAT_R16G16_SFLOAT
}

/// UNORM formats with uneven channels: (format, borgc packed_layout number, bytes, alpha).
const UNEVEN: [(VkFormat, u32, u32, bool); 14] = [
    (VK_FORMAT_A1R5G5B5_UNORM_PACK16, 0, 2, true),
    (VK_FORMAT_R5G5B5A1_UNORM_PACK16, 1, 2, true),
    (VK_FORMAT_B5G5R5A1_UNORM_PACK16, 2, 2, true),
    (VK_FORMAT_R4G4B4A4_UNORM_PACK16, 3, 2, true),
    (VK_FORMAT_B4G4R4A4_UNORM_PACK16, 4, 2, true),
    (VK_FORMAT_A4R4G4B4_UNORM_PACK16, 5, 2, true),
    (VK_FORMAT_A4B4G4R4_UNORM_PACK16, 6, 2, true),
    (VK_FORMAT_A2B10G10R10_UNORM_PACK32, 7, 4, true),
    (VK_FORMAT_A2R10G10B10_UNORM_PACK32, 8, 4, true),
    (VK_FORMAT_B5G6R5_UNORM_PACK16, 9, 2, false),
    (VK_FORMAT_R4G4_UNORM_PACK8, 10, 1, false),
    (VK_FORMAT_R5G6B5_UNORM_PACK16, 11, 2, false),
    (VK_FORMAT_A2B10G10R10_UINT_PACK32, 12, 4, true),
    (VK_FORMAT_A2R10G10B10_UINT_PACK32, 13, 4, true),
];

/// The fragment shader's packing option word for a raw attachment format, and its bytes per
/// pixel; None for the natively flushed formats and those not packed.
pub fn packed_info(f: VkFormat) -> Option<(u32, u32)> {
    if f == VK_FORMAT_R8_UNORM || f == VK_FORMAT_R8G8B8A8_UNORM || f == VK_FORMAT_B8G8R8A8_UNORM
        || bytes_packed(f) || half1(f) || half2(f) || raw32(f)
    {
        return None;
    }
    if let Some(&(_, id, bytes, alpha)) = UNEVEN.iter().find(|u| u.0 == f) {
        let opt = BORGC_FS_PACKED
            | 3 << BORGC_FS_PACKED_SIZE_SHIFT
            | (id & 7) << BORGC_FS_PACKED_KIND_SHIFT
            | (id >> 3) << BORGC_FS_PACKED_N_SHIFT
            | if alpha { BORGC_FS_ALPHA_OUTPUT } else { 0 };
        return Some((opt, bytes));
    }
    let d = fmt::description(fmt::pipe(f))?;
    let n = d.nr_channels();
    if d.layout != UTIL_FORMAT_LAYOUT_PLAIN || !(1..=4).contains(&n) {
        return None;
    }
    let ch = &d.channel[..n as usize];
    let bits = ch[0].size();
    if !matches!(bits, 8 | 16 | 32) || ch.iter().any(|c| c.size() != bits) {
        return None;
    }
    let bytes = n * bits / 8;
    if bytes > 16 {
        return None;
    }
    let is_float = ch[0].type_() == UTIL_FORMAT_TYPE_FLOAT;
    if is_float && bits == 16 {
        let extra = match n {
            3 => BORGC_FS_HALF3,
            4 => BORGC_FS_HALF4 | BORGC_FS_ALPHA_OUTPUT,
            _ => 0,
        };
        return Some((BORGC_FS_HALF | extra, bytes));
    }
    let kind = if d.colorspace == UTIL_FORMAT_COLORSPACE_SRGB {
        2
    } else if ch[0].pure_integer() != 0 || is_float {
        3
    } else if bits < 32 && ch[0].normalized() != 0 {
        if ch[0].type_() == UTIL_FORMAT_TYPE_SIGNED { 1 } else { 0 }
    } else {
        return None;
    };
    let swap = n >= 3 && d.swizzle[0] as u32 == PIPE_SWIZZLE_Z;
    let size = match bits {
        8 => 0,
        16 => 1,
        _ => 2,
    };
    let opt = BORGC_FS_PACKED
        | kind << BORGC_FS_PACKED_KIND_SHIFT
        | size << BORGC_FS_PACKED_SIZE_SHIFT
        | (n - 1) << BORGC_FS_PACKED_N_SHIFT
        | (swap as u32) << BORGC_FS_PACKED_SWAP_SHIFT
        | if n == 4 { BORGC_FS_ALPHA_OUTPUT } else { 0 };
    Some((opt, bytes))
}

/// The tile flusher's format code for a colour attachment.
pub fn flush_format(f: VkFormat) -> u8 {
    if let Some((_, bytes)) = packed_info(f) {
        return match bytes {
            1 => 5,
            2 => 4,
            ..=4 => 3,
            ..=8 => 6,
            _ => 7,
        };
    }
    match f {
        VK_FORMAT_R8_UNORM => 5,
        VK_FORMAT_R8G8B8A8_UNORM => 1,
        VK_FORMAT_B8G8R8A8_UNORM => 2,
        _ if half1(f) => 4,
        _ if raw32(f) || bytes_packed(f) || half2(f) => 3,
        _ => 0,
    }
}

pub fn depth_is_d32(f: VkFormat) -> bool {
    !(f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_D16_UNORM_S8_UINT)
}

/// How a view of format `f` reaches the sampler: the hardware format and the conversion of its
/// texels; the stencil aspect samples as R8_UINT.
pub fn tex_format(f: VkFormat, stencil: bool) -> Option<(TexFormat, TexConv)> {
    use TexFormat as T;
    let raw = |t: T| (t, TexConv { bpp: t.bytes(), ..Default::default() });
    let ch = |t, cs, sch, swap_rb, alpha| (t, TexConv { bpp: cs * sch, cs, sch, alpha, swap_rb, ..Default::default() });
    let p16 = |t: T, pack16| (t, TexConv { bpp: t.bytes(), pack16, ..Default::default() });
    let (mut t, mut c) = match f {
        VK_FORMAT_R8_UNORM => raw(T::R8_UNORM),
        VK_FORMAT_R8_SNORM => raw(T::R8_SNORM),
        VK_FORMAT_R8_UINT | VK_FORMAT_S8_UINT => raw(T::R8_UINT),
        VK_FORMAT_R8_SINT => raw(T::R8_SINT),
        VK_FORMAT_R8_SRGB => raw(T::R8_SRGB),
        VK_FORMAT_R8G8_UNORM => raw(T::R8G8_UNORM),
        VK_FORMAT_R8G8_SNORM => raw(T::R8G8_SNORM),
        VK_FORMAT_R8G8_UINT => raw(T::R8G8_UINT),
        VK_FORMAT_R8G8_SINT => raw(T::R8G8_SINT),
        VK_FORMAT_R8G8_SRGB => raw(T::R8G8_SRGB),
        VK_FORMAT_R8G8B8A8_UNORM | VK_FORMAT_A8B8G8R8_UNORM_PACK32 => raw(T::R8G8B8A8_UNORM),
        VK_FORMAT_R8G8B8A8_SNORM | VK_FORMAT_A8B8G8R8_SNORM_PACK32 => raw(T::R8G8B8A8_SNORM),
        VK_FORMAT_R8G8B8A8_UINT | VK_FORMAT_A8B8G8R8_UINT_PACK32 => raw(T::R8G8B8A8_UINT),
        VK_FORMAT_R8G8B8A8_SINT | VK_FORMAT_A8B8G8R8_SINT_PACK32 => raw(T::R8G8B8A8_SINT),
        VK_FORMAT_R8G8B8A8_SRGB | VK_FORMAT_A8B8G8R8_SRGB_PACK32 => raw(T::R8G8B8A8_SRGB),
        VK_FORMAT_B8G8R8A8_UNORM => raw(T::B8G8R8A8_UNORM),
        VK_FORMAT_B8G8R8A8_SRGB => raw(T::B8G8R8A8_SRGB),
        VK_FORMAT_B8G8R8A8_SNORM => ch(T::R8G8B8A8_SNORM, 1, 4, true, 0),
        VK_FORMAT_B8G8R8A8_UINT => ch(T::R8G8B8A8_UINT, 1, 4, true, 0),
        VK_FORMAT_B8G8R8A8_SINT => ch(T::R8G8B8A8_SINT, 1, 4, true, 0),
        VK_FORMAT_R8G8B8_UNORM => ch(T::R8G8B8A8_UNORM, 1, 3, false, 255),
        VK_FORMAT_R8G8B8_SNORM => ch(T::R8G8B8A8_SNORM, 1, 3, false, 127),
        VK_FORMAT_R8G8B8_UINT => ch(T::R8G8B8A8_UINT, 1, 3, false, 1),
        VK_FORMAT_R8G8B8_SINT => ch(T::R8G8B8A8_SINT, 1, 3, false, 1),
        VK_FORMAT_R8G8B8_SRGB => ch(T::R8G8B8A8_SRGB, 1, 3, false, 255),
        VK_FORMAT_B8G8R8_UNORM => ch(T::R8G8B8A8_UNORM, 1, 3, true, 255),
        VK_FORMAT_B8G8R8_SNORM => ch(T::R8G8B8A8_SNORM, 1, 3, true, 127),
        VK_FORMAT_B8G8R8_UINT => ch(T::R8G8B8A8_UINT, 1, 3, true, 1),
        VK_FORMAT_B8G8R8_SINT => ch(T::R8G8B8A8_SINT, 1, 3, true, 1),
        VK_FORMAT_B8G8R8_SRGB => ch(T::R8G8B8A8_SRGB, 1, 3, true, 255),
        VK_FORMAT_R16_UNORM => raw(T::R16_UNORM),
        VK_FORMAT_R16_SNORM => raw(T::R16_SNORM),
        VK_FORMAT_R16_UINT => raw(T::R16_UINT),
        VK_FORMAT_R16_SINT => raw(T::R16_SINT),
        VK_FORMAT_R16_SFLOAT => raw(T::R16_SFLOAT),
        VK_FORMAT_R16G16_UNORM => raw(T::R16G16_UNORM),
        VK_FORMAT_R16G16_SNORM => raw(T::R16G16_SNORM),
        VK_FORMAT_R16G16_UINT => raw(T::R16G16_UINT),
        VK_FORMAT_R16G16_SINT => raw(T::R16G16_SINT),
        VK_FORMAT_R16G16_SFLOAT => raw(T::R16G16_SFLOAT),
        VK_FORMAT_R16G16B16_UNORM => ch(T::R16G16B16A16_UNORM, 2, 3, false, 0xffff),
        VK_FORMAT_R16G16B16_SNORM => ch(T::R16G16B16A16_SNORM, 2, 3, false, 0x7fff),
        VK_FORMAT_R16G16B16_UINT => ch(T::R16G16B16A16_UINT, 2, 3, false, 1),
        VK_FORMAT_R16G16B16_SINT => ch(T::R16G16B16A16_SINT, 2, 3, false, 1),
        VK_FORMAT_R16G16B16_SFLOAT => ch(T::R16G16B16A16_SFLOAT, 2, 3, false, 0x3c00),
        VK_FORMAT_R16G16B16A16_UNORM => raw(T::R16G16B16A16_UNORM),
        VK_FORMAT_R16G16B16A16_SNORM => raw(T::R16G16B16A16_SNORM),
        VK_FORMAT_R16G16B16A16_UINT => raw(T::R16G16B16A16_UINT),
        VK_FORMAT_R16G16B16A16_SINT => raw(T::R16G16B16A16_SINT),
        VK_FORMAT_R16G16B16A16_SFLOAT => raw(T::R16G16B16A16_SFLOAT),
        VK_FORMAT_R32_UINT => raw(T::R32_UINT),
        VK_FORMAT_R32_SINT => raw(T::R32_SINT),
        VK_FORMAT_R32_SFLOAT => raw(T::R32_SFLOAT),
        VK_FORMAT_R32G32_UINT => raw(T::R32G32_UINT),
        VK_FORMAT_R32G32_SINT => raw(T::R32G32_SINT),
        VK_FORMAT_R32G32_SFLOAT => raw(T::R32G32_SFLOAT),
        VK_FORMAT_R32G32B32_UINT => ch(T::R32G32B32A32_UINT, 4, 3, false, 1),
        VK_FORMAT_R32G32B32_SINT => ch(T::R32G32B32A32_SINT, 4, 3, false, 1),
        VK_FORMAT_R32G32B32_SFLOAT => ch(T::R32G32B32A32_SFLOAT, 4, 3, false, 0x3f800000),
        VK_FORMAT_R32G32B32A32_UINT => raw(T::R32G32B32A32_UINT),
        VK_FORMAT_R32G32B32A32_SINT => raw(T::R32G32B32A32_SINT),
        VK_FORMAT_R32G32B32A32_SFLOAT => raw(T::R32G32B32A32_SFLOAT),
        VK_FORMAT_A2B10G10R10_UNORM_PACK32 => raw(T::A2B10G10R10_UNORM_PACK32),
        VK_FORMAT_A2B10G10R10_UINT_PACK32 => raw(T::A2B10G10R10_UINT_PACK32),
        VK_FORMAT_R5G6B5_UNORM_PACK16 => raw(T::R5G6B5_UNORM_PACK16),
        VK_FORMAT_A1R5G5B5_UNORM_PACK16 => raw(T::A1R5G5B5_UNORM_PACK16),
        VK_FORMAT_B4G4R4A4_UNORM_PACK16 => raw(T::B4G4R4A4_UNORM_PACK16),
        VK_FORMAT_B10G11R11_UFLOAT_PACK32 => raw(T::B10G11R11_UFLOAT_PACK32),
        VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 => raw(T::E5B9G9R9_UFLOAT_PACK32),
        VK_FORMAT_B5G6R5_UNORM_PACK16 => p16(T::R5G6B5_UNORM_PACK16, 1),
        VK_FORMAT_R5G5B5A1_UNORM_PACK16 => p16(T::A1R5G5B5_UNORM_PACK16, 2),
        VK_FORMAT_B5G5R5A1_UNORM_PACK16 => p16(T::A1R5G5B5_UNORM_PACK16, 3),
        VK_FORMAT_R4G4B4A4_UNORM_PACK16 => p16(T::B4G4R4A4_UNORM_PACK16, 4),
        VK_FORMAT_A4R4G4B4_UNORM_PACK16 => p16(T::B4G4R4A4_UNORM_PACK16, 7),
        VK_FORMAT_A4B4G4R4_UNORM_PACK16 => p16(T::B4G4R4A4_UNORM_PACK16, 8),
        VK_FORMAT_A2R10G10B10_UNORM_PACK32 => p16(T::A2B10G10R10_UNORM_PACK32, 6),
        VK_FORMAT_A2R10G10B10_UINT_PACK32 => p16(T::A2B10G10R10_UINT_PACK32, 6),
        VK_FORMAT_R4G4_UNORM_PACK8 => (T::R8G8_UNORM, TexConv { bpp: 1, pack16: 5, ..Default::default() }),
        VK_FORMAT_D16_UNORM => raw(T::D16_UNORM),
        VK_FORMAT_X8_D24_UNORM_PACK32 | VK_FORMAT_D24_UNORM_S8_UINT => raw(T::X8_D24_UNORM_PACK32),
        VK_FORMAT_D32_SFLOAT => raw(T::D32_SFLOAT),
        VK_FORMAT_D16_UNORM_S8_UINT => (T::D16_UNORM, TexConv { bpp: 2, zstride: 3, ..Default::default() }),
        VK_FORMAT_D32_SFLOAT_S8_UINT => (T::D32_SFLOAT, TexConv { bpp: 4, zstride: 8, ..Default::default() }),
        _ => return None,
    };
    if stencil {
        (c.zstride, c.soff) = match f {
            VK_FORMAT_S8_UINT => (c.zstride, c.soff),
            VK_FORMAT_D16_UNORM_S8_UINT => (3, 2),
            VK_FORMAT_D24_UNORM_S8_UINT => (4, 3),
            VK_FORMAT_D32_SFLOAT_S8_UINT => (8, 4),
            _ => return None,
        };
        (t, c.bpp) = (T::R8_UINT, 1);
    }
    c.obpp = if c.sch != 0 { c.cs * 4 } else if c.pack16 == 5 { 2 } else { c.bpp };
    Some((t, c))
}

/// Converts `n` texels from `src` to the layout the texture unit reads.
pub fn tex_convert(c: &TexConv, dst: &mut [u8], src: &[u8], n: usize) {
    let (bpp, obpp, cs) = (c.bpp as usize, c.obpp as usize, c.cs as usize);
    if c.sch == 0 && c.pack16 == 0 && c.zstride == 0 {
        dst[..n * obpp].copy_from_slice(&src[..n * obpp]);
    }
    if c.zstride != 0 {
        let (zs, soff) = (c.zstride as usize, c.soff as usize);
        for i in 0..n {
            dst[i * obpp..][..obpp].copy_from_slice(&src[i * zs + soff..][..obpp]);
        }
    }
    if c.sch != 0 {
        let alpha = c.alpha.to_ne_bytes();
        let (r, b) = if c.swap_rb { (2 * cs, 0) } else { (0, 2 * cs) };
        for i in 0..n {
            let p = &src[i * bpp..];
            let d = &mut dst[i * obpp..];
            d[..cs].copy_from_slice(&p[r..][..cs]);
            d[cs..2 * cs].copy_from_slice(&p[cs..2 * cs]);
            d[2 * cs..3 * cs].copy_from_slice(&p[b..][..cs]);
            d[3 * cs..4 * cs].copy_from_slice(if c.sch == 4 { &p[3 * cs..4 * cs] } else { &alpha[..cs] });
        }
    }
    if c.pack16 != 0 {
        for i in 0..n {
            let mut w = [0u8; 4];
            w[..bpp].copy_from_slice(&src[i * bpp..][..bpp]);
            let v = u32::from_ne_bytes(w);
            let v = match c.pack16 {
                1 => (v >> 11) | (v & 0x07e0) | ((v & 0x1f) << 11),
                2 => ((v & 1) << 15) | (v >> 1),
                3 => ((v & 1) << 15) | ((v & 0x3e) << 9) | ((v >> 1) & 0x03e0) | (v >> 11),
                4 => ((v & 0x00f0) << 8) | (v & 0x0f0f) | ((v >> 8) & 0x00f0),
                5 => (v >> 4).wrapping_mul(17) | ((v & 0xf).wrapping_mul(17) << 8),
                7 => ((v & 0xf) << 12) | ((v & 0xf0) << 4) | ((v >> 4) & 0xf0) | (v >> 12),
                8 => ((v & 0xf00) << 4) | ((v & 0xf0) << 4) | ((v & 0xf) << 4) | (v >> 12),
                _ => (v & 0xc00ffc00) | ((v >> 20) & 0x3ff) | ((v & 0x3ff) << 20),
            };
            dst[i * obpp..][..obpp].copy_from_slice(&v.to_ne_bytes()[..obpp]);
        }
    }
}

/// # Safety
/// `c` is valid, `dst` holds `n` output texels and `src` the `n` input texels `c` describes.
pub unsafe fn borgvk_tex_convert(c: *const TexConv, dst: *mut u8, src: *const u8, n: u32) {
    let (c, n) = (unsafe { &*c }, n as usize);
    let mut src_len = n * c.bpp as usize;
    if c.sch == 0 && c.pack16 == 0 && c.zstride == 0 {
        src_len = n * c.obpp as usize;
    }
    if c.zstride != 0 && n > 0 {
        src_len = src_len.max((n - 1) * c.zstride as usize + c.soff as usize + c.obpp as usize);
    }
    let dst = unsafe { std::slice::from_raw_parts_mut(dst, n * c.obpp as usize) };
    let src = unsafe { std::slice::from_raw_parts(src, src_len) };
    tex_convert(c, dst, src, n);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tex_formats_and_their_conversions() {
        let (t, c) = tex_format(VK_FORMAT_B8G8R8_UNORM, false).unwrap();
        assert_eq!((t, c.bpp, c.obpp, c.cs, c.sch, c.alpha, c.swap_rb), (TexFormat::R8G8B8A8_UNORM, 3, 4, 1, 3, 255, true));
        let (t, c) = tex_format(VK_FORMAT_D24_UNORM_S8_UINT, true).unwrap();
        assert_eq!((t, c.bpp, c.obpp, c.zstride, c.soff), (TexFormat::R8_UINT, 1, 1, 4, 3));
        assert_eq!(tex_format(VK_FORMAT_D24_UNORM_S8_UINT, false).unwrap().0, TexFormat::X8_D24_UNORM_PACK32);
        assert_eq!(tex_format(VK_FORMAT_R4G4_UNORM_PACK8, false).unwrap().1.obpp, 2);
        assert!(tex_format(VK_FORMAT_R8_UNORM, true).is_none());
        assert!(tex_format(VK_FORMAT_BC1_RGB_UNORM_BLOCK, false).is_none());
    }

    fn conv(bpp: u32, obpp: u32, cs: u32, sch: u32, alpha: u32, swap_rb: bool) -> TexConv {
        TexConv { bpp, obpp, cs, sch, alpha, swap_rb, ..Default::default() }
    }

    #[test]
    fn tex_convert_swaps_red_and_blue() {
        let mut out = [0u8; 8];
        tex_convert(&conv(4, 4, 1, 4, 0, true), &mut out, &[1, 2, 3, 4, 5, 6, 7, 8], 2);
        assert_eq!(out, [3, 2, 1, 4, 7, 6, 5, 8]);
    }

    #[test]
    fn tex_convert_plain_copy() {
        let mut out = [0u8; 8];
        tex_convert(&conv(4, 4, 0, 0, 0, false), &mut out, &[1, 2, 3, 4, 5, 6, 7, 8], 2);
        assert_eq!(out, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn tex_convert_pads_alpha() {
        let mut out = [0u8; 4];
        tex_convert(&conv(3, 4, 1, 3, 255, false), &mut out, &[9, 8, 7], 1);
        assert_eq!(out, [9, 8, 7, 255]);
    }

    #[test]
    fn rgba32f() {
        assert_eq!(vertex_fetch(VK_FORMAT_R32G32B32A32_SFLOAT), Some(Fetch { code: 41, swz: 0, read_bytes: 16 }));
    }

    #[test]
    fn rgb32f_forces_alpha_one() {
        assert_eq!(vertex_fetch(VK_FORMAT_R32G32B32_SFLOAT), Some(Fetch { code: 41, swz: A_ONE, read_bytes: 16 }));
    }

    #[test]
    fn r8_unorm() {
        assert_eq!(vertex_fetch(VK_FORMAT_R8_UNORM), Some(Fetch { code: 1, swz: 0, read_bytes: 1 }));
    }

    #[test]
    fn flush_formats() {
        assert_eq!(flush_format(VK_FORMAT_R8G8B8A8_UNORM), 1);
        assert_eq!(flush_format(VK_FORMAT_B8G8R8A8_UNORM), 2);
        assert_eq!(flush_format(VK_FORMAT_R32_SFLOAT), 3);
        assert_eq!(flush_format(VK_FORMAT_R16_SFLOAT), 4);
        assert_eq!(flush_format(VK_FORMAT_R8_UNORM), 5);
        assert_eq!(flush_format(VK_FORMAT_R5G6B5_UNORM_PACK16), 4);
        assert_eq!(flush_format(VK_FORMAT_R4G4_UNORM_PACK8), 5);
    }

    #[test]
    fn uneven_formats_pack_by_number() {
        let (opt, bytes) = packed_info(VK_FORMAT_A2B10G10R10_UINT_PACK32).unwrap();
        assert_eq!(bytes, 4);
        assert_eq!(opt, BORGC_FS_PACKED | 3 << BORGC_FS_PACKED_SIZE_SHIFT | 4 << BORGC_FS_PACKED_KIND_SHIFT | 1 << BORGC_FS_PACKED_N_SHIFT | BORGC_FS_ALPHA_OUTPUT);
        assert_eq!(packed_info(VK_FORMAT_R8G8B8A8_UNORM), None);
    }

    #[test]
    fn no_fetch() {
        assert_eq!(vertex_fetch(VK_FORMAT_UNDEFINED), None);
        assert_eq!(vertex_fetch(VK_FORMAT_R64_SFLOAT), None);
    }
}

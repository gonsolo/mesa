// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::ffi::*;

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

/// # Safety
/// `code`, `swz` and `read_bytes` are valid for writes.
#[no_mangle]
pub unsafe extern "C" fn borgvk_vertex_format_to_fetch(
    f: VkFormat,
    code: *mut u32,
    swz: *mut u32,
    read_bytes: *mut u32,
) -> bool {
    let fetch = vertex_fetch(f);
    unsafe {
        *swz = 0;
        if let Some(v) = fetch {
            *code = v.code;
            *swz = v.swz;
            *read_bytes = v.read_bytes;
        }
    }
    fetch.is_some()
}

/// Converts `n` texels from `src` to the layout the texture unit reads.
pub fn tex_convert(c: &tex_conv, dst: &mut [u8], src: &[u8], n: usize) {
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
#[no_mangle]
pub unsafe extern "C" fn borgvk_tex_convert(c: *const tex_conv, dst: *mut u8, src: *const u8, n: u32) {
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

    fn conv(bpp: u32, obpp: u32, cs: u32, sch: u32, alpha: u32, swap_rb: bool) -> tex_conv {
        tex_conv { bpp, obpp, cs, sch, alpha, swap_rb, ..Default::default() }
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
    fn no_fetch() {
        assert_eq!(vertex_fetch(VK_FORMAT_UNDEFINED), None);
        assert_eq!(vertex_fetch(VK_FORMAT_R64_SFLOAT), None);
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

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

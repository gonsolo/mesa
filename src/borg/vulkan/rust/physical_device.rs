// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! The physical device's queries: format features and image limits.

use crate::entrypoint::{chain, entrypoint};
use crate::ffi::*;
use crate::image::blocks;

pub const SC_MSAA: VkSampleCountFlags = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT;
pub const SC_INTEGER: VkSampleCountFlags = VK_SAMPLE_COUNT_1_BIT;
pub const SC_STORAGE: VkSampleCountFlags = VK_SAMPLE_COUNT_1_BIT;

fn is_depth_stencil(f: VkFormat) -> bool {
    matches!(
        f,
        VK_FORMAT_D16_UNORM
            | VK_FORMAT_X8_D24_UNORM_PACK32
            | VK_FORMAT_D32_SFLOAT
            | VK_FORMAT_S8_UINT
            | VK_FORMAT_D16_UNORM_S8_UINT
            | VK_FORMAT_D24_UNORM_S8_UINT
            | VK_FORMAT_D32_SFLOAT_S8_UINT
    )
}

fn is_ycbcr(f: VkFormat) -> bool {
    (VK_FORMAT_G8B8G8R8_422_UNORM..=VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM).contains(&f)
        || (VK_FORMAT_G8_B8R8_2PLANE_444_UNORM..=VK_FORMAT_G16_B16R16_2PLANE_444_UNORM).contains(&f)
}

fn is_compressed(f: VkFormat) -> bool {
    (VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK..=VK_FORMAT_EAC_R11G11_SNORM_BLOCK).contains(&f)
        || (VK_FORMAT_ASTC_4x4_UNORM_BLOCK..=VK_FORMAT_ASTC_12x12_SRGB_BLOCK).contains(&f)
        || (1000288000..=1000288029).contains(&f)
        || blocks(f).bd > 1
}

fn is_bc(f: VkFormat) -> bool {
    (VK_FORMAT_BC1_RGB_UNORM_BLOCK..=VK_FORMAT_BC7_SRGB_BLOCK).contains(&f)
}

fn is_scaled(f: VkFormat) -> bool {
    unsafe { util_format_is_scaled(vk_format_to_pipe_format(f)) }
}

/// Optimal-tiling format features.
fn optimal_features(f: VkFormat) -> VkFormatFeatureFlags {
    if f == VK_FORMAT_UNDEFINED || is_ycbcr(f) || is_compressed(f) || is_scaled(f) {
        return 0;
    }
    if matches!(
        f,
        VK_FORMAT_A2R10G10B10_SNORM_PACK32 | VK_FORMAT_A2B10G10R10_SNORM_PACK32 | VK_FORMAT_A2R10G10B10_SINT_PACK32 | VK_FORMAT_A2B10G10R10_SINT_PACK32
    ) || (VK_FORMAT_R64_UINT..=VK_FORMAT_R64G64B64A64_SFLOAT).contains(&f)
    {
        return 0;
    }
    if is_bc(f) {
        return VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT
            | VK_FORMAT_FEATURE_BLIT_SRC_BIT
            | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT
            | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT
            | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    }
    let common = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT
        | VK_FORMAT_FEATURE_BLIT_SRC_BIT
        | VK_FORMAT_FEATURE_BLIT_DST_BIT
        | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT
        | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    if is_depth_stencil(f) {
        return common | VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
    }
    let mut r = common | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
    if matches!(
        f,
        VK_FORMAT_R5G6B5_UNORM_PACK16
            | VK_FORMAT_A1R5G5B5_UNORM_PACK16
            | VK_FORMAT_R8_UNORM
            | VK_FORMAT_R8G8_UNORM
            | VK_FORMAT_R8G8B8A8_UNORM
            | VK_FORMAT_R8G8B8A8_SRGB
            | VK_FORMAT_B8G8R8A8_UNORM
            | VK_FORMAT_B8G8R8A8_SRGB
            | VK_FORMAT_A8B8G8R8_UNORM_PACK32
            | VK_FORMAT_A8B8G8R8_SRGB_PACK32
            | VK_FORMAT_A2B10G10R10_UNORM_PACK32
            | VK_FORMAT_R16_SFLOAT
            | VK_FORMAT_R16G16_SFLOAT
            | VK_FORMAT_R16G16B16A16_SFLOAT
    ) {
        r |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT;
    }
    if unsafe { !vk_format_is_int(f) } {
        r |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    }
    if matches!(f, VK_FORMAT_R32_UINT | VK_FORMAT_R32_SINT) {
        r |= VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT;
    }
    r
}

fn texel_buffer_features(f: VkFormat) -> VkFormatFeatureFlags {
    match f {
        VK_FORMAT_A8B8G8R8_SINT_PACK32
        | VK_FORMAT_A8B8G8R8_SNORM_PACK32
        | VK_FORMAT_A8B8G8R8_UINT_PACK32
        | VK_FORMAT_A8B8G8R8_UNORM_PACK32
        | VK_FORMAT_R16G16B16A16_SFLOAT
        | VK_FORMAT_R16G16B16A16_SINT
        | VK_FORMAT_R16G16B16A16_UINT
        | VK_FORMAT_R32_SFLOAT
        | VK_FORMAT_R32_SINT
        | VK_FORMAT_R32_UINT
        | VK_FORMAT_R32G32_SFLOAT
        | VK_FORMAT_R32G32_SINT
        | VK_FORMAT_R32G32_UINT
        | VK_FORMAT_R32G32B32A32_SFLOAT
        | VK_FORMAT_R32G32B32A32_SINT
        | VK_FORMAT_R32G32B32A32_UINT
        | VK_FORMAT_R8G8B8A8_SINT
        | VK_FORMAT_R8G8B8A8_SNORM
        | VK_FORMAT_R8G8B8A8_UINT
        | VK_FORMAT_R8G8B8A8_UNORM => {
            let atomic = if matches!(f, VK_FORMAT_R32_UINT | VK_FORMAT_R32_SINT) { VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT } else { 0 };
            VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT | VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT | atomic
        }
        VK_FORMAT_A2B10G10R10_UINT_PACK32
        | VK_FORMAT_A2B10G10R10_UNORM_PACK32
        | VK_FORMAT_B10G11R11_UFLOAT_PACK32
        | VK_FORMAT_B8G8R8A8_UNORM
        | VK_FORMAT_R16_SFLOAT
        | VK_FORMAT_R16_SINT
        | VK_FORMAT_R16_UINT
        | VK_FORMAT_R16G16_SFLOAT
        | VK_FORMAT_R16G16_SINT
        | VK_FORMAT_R16G16_UINT
        | VK_FORMAT_R8_SINT
        | VK_FORMAT_R8_SNORM
        | VK_FORMAT_R8_UINT
        | VK_FORMAT_R8_UNORM
        | VK_FORMAT_R8G8_SINT
        | VK_FORMAT_R8G8_SNORM
        | VK_FORMAT_R8G8_UINT
        | VK_FORMAT_R8G8_UNORM => VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT,
        _ => 0,
    }
}

/// Linear-tiling, optimal-tiling and buffer features.
fn format_properties(f: VkFormat) -> VkFormatProperties {
    let opt = optimal_features(f);
    let (ds, bc) = (is_depth_stencil(f), is_bc(f));
    let mut lin = if opt != 0 && !bc { VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT } else { 0 };
    if opt & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT != 0 && !ds && !bc {
        lin |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | opt & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    }
    let buf = if (opt != 0 || is_scaled(f)) && !ds && !bc { VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT | texel_buffer_features(f) } else { 0 };
    VkFormatProperties { linearTilingFeatures: lin, optimalTilingFeatures: opt, bufferFeatures: buf }
}

entrypoint! {
    fn borgvk_GetPhysicalDeviceFormatProperties2(_pd: VkPhysicalDevice, format: VkFormat, props: *mut VkFormatProperties2) {
        let p = format_properties(format);
        (*props).formatProperties = p;
        for s in chain((*props).pNext) {
            if (*s).sType == VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3 {
                let p3 = &mut *s.cast::<VkFormatProperties3>();
                p3.linearTilingFeatures = p.linearTilingFeatures as u64;
                p3.optimalTilingFeatures = p.optimalTilingFeatures as u64;
                p3.bufferFeatures = p.bufferFeatures as u64;
                if vk_format_has_depth(format) && p.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT != 0 {
                    p3.optimalTilingFeatures |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT;
                }
            }
        }
    }
}

fn image_format_properties(info: &VkPhysicalDeviceImageFormatInfo2) -> Result<VkImageFormatProperties, VkResult> {
    let fp = format_properties(info.format);
    let feats = if info.tiling == VK_IMAGE_TILING_LINEAR { fp.linearTilingFeatures } else { fp.optimalTilingFeatures };
    let usage = info.usage;
    let need = [
        (VK_IMAGE_USAGE_SAMPLED_BIT, VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT),
        (VK_IMAGE_USAGE_STORAGE_BIT, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT),
        (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT),
        (VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT),
        (VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT, VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT),
        (VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_FORMAT_FEATURE_TRANSFER_SRC_BIT),
        (VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_FORMAT_FEATURE_TRANSFER_DST_BIT),
    ];
    let required = need.iter().filter(|(u, _)| usage & u != 0).fold(0, |r, (_, f)| r | f);
    let attach = feats & (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0;
    if feats & required != required
        || (info.type_ == VK_IMAGE_TYPE_3D && blocks(info.format).bd > 1)
        || (usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT != 0 && !attach)
    {
        return Err(VK_ERROR_FORMAT_NOT_SUPPORTED);
    }
    let cube = info.flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT != 0;
    let mut samples = VK_SAMPLE_COUNT_1_BIT;
    if info.type_ == VK_IMAGE_TYPE_2D && info.tiling == VK_IMAGE_TILING_OPTIMAL && !cube && attach {
        let sampled = if is_depth_stencil(info.format) || unsafe { !vk_format_is_int(info.format) } { SC_MSAA } else { SC_INTEGER };
        let narrow = [
            (VK_IMAGE_USAGE_STORAGE_BIT, SC_STORAGE),
            (VK_IMAGE_USAGE_SAMPLED_BIT, sampled),
            (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, SC_MSAA),
            (VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, SC_MSAA),
        ];
        let sc = narrow.iter().filter(|(u, _)| usage & u != 0).fold(!0, |s, (_, c)| s & c);
        // No usage bit narrows it: only single-sampled is required.
        samples = if sc == !0 { VK_SAMPLE_COUNT_1_BIT } else { sc | VK_SAMPLE_COUNT_1_BIT };
    }
    let (d1, d3) = (info.type_ == VK_IMAGE_TYPE_1D, info.type_ == VK_IMAGE_TYPE_3D);
    Ok(VkImageFormatProperties {
        maxExtent: VkExtent3D { width: 4096, height: if d1 { 1 } else { 4096 }, depth: if d3 { 2048 } else { 1 } },
        maxMipLevels: 13,
        maxArrayLayers: if d3 { 1 } else { 2048 }, // 3D images have one layer
        sampleCounts: samples,
        maxResourceSize: 1 << 31,
    })
}

entrypoint! {
    fn borgvk_GetPhysicalDeviceImageFormatProperties2(_pd: VkPhysicalDevice, info: *const VkPhysicalDeviceImageFormatInfo2, props: *mut VkImageFormatProperties2) -> VkResult {
        let r = image_format_properties(&*info);
        (*props).imageFormatProperties = *r.as_ref().unwrap_or(&std::mem::zeroed());
        r.map_or_else(|e| e, |_| VK_SUCCESS)
    }
}

entrypoint! {
    fn borgvk_GetPhysicalDeviceSparseImageFormatProperties2(_pd: VkPhysicalDevice, _info: *const VkPhysicalDeviceSparseImageFormatInfo2, count: *mut u32, _props: *mut VkSparseImageFormatProperties2) {
        *count = 0;
    }
}

entrypoint! {
    fn borgvk_GetPhysicalDeviceExternalBufferProperties(_pd: VkPhysicalDevice, _info: *const VkPhysicalDeviceExternalBufferInfo, props: *mut VkExternalBufferProperties) {
        (*props).externalMemoryProperties = std::mem::zeroed();
    }
}

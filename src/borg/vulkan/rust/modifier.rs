// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! VK_EXT_image_drm_format_modifier, for the display: linear, and Borg's 4x4 tiles
//! (DRM_FORMAT_MOD_BORG_TILED_4X4, borg_drm.h) for the colour formats the scan-out reads, which
//! Borg renders and the display shows in place.

use crate::ffi::*;

pub const LINEAR: u64 = 0;
pub const TILED: u64 = 0x7f << 56 | 1;

fn scanned_out(f: VkFormat) -> bool {
    matches!(f, VK_FORMAT_B8G8R8A8_UNORM | VK_FORMAT_B8G8R8A8_SRGB | VK_FORMAT_R8G8B8A8_UNORM | VK_FORMAT_R8G8B8A8_SRGB)
}

/// The modifiers of `f` and their features, from its tilings' features.
pub fn of(f: VkFormat, p: &VkFormatProperties) -> Vec<(u64, VkFormatFeatureFlags)> {
    let tiled = (scanned_out(f) && p.optimalTilingFeatures != 0).then_some((TILED, p.optimalTilingFeatures));
    let linear = (p.linearTilingFeatures != 0).then_some((LINEAR, p.linearTilingFeatures));
    tiled.into_iter().chain(linear).collect()
}

/// The tiling a modifier stands for.
pub fn tiling(m: u64) -> Option<VkImageTiling> {
    match m {
        TILED => Some(VK_IMAGE_TILING_OPTIMAL),
        LINEAR => Some(VK_IMAGE_TILING_LINEAR),
        _ => None,
    }
}

/// The tiling an image is laid out in: a modifier's.
pub fn effective(tiling: VkImageTiling, m: u64) -> VkImageTiling {
    if tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT { self::tiling(m).unwrap_or(VK_IMAGE_TILING_LINEAR) } else { tiling }
}

/// What a tiled image cannot be: storage images are linear, the scan-out reads one level, one
/// layer, one sample.
pub fn tiled_ok(usage: VkImageUsageFlags, levels: u32, layers: u32, samples: u32) -> bool {
    usage & VK_IMAGE_USAGE_STORAGE_BIT == 0 && levels <= 1 && layers <= 1 && samples <= 1
}

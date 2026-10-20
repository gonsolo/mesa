// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::entrypoint::{entrypoint, handle, slice, vk_error, Handle};
use crate::ffi::*;
use crate::memory::requirements;

use borg_layout::image::{mip_level_size, Blocks, Shape};

handle!(VkImage, borgvk_image, VK_OBJECT_TYPE_IMAGE);
handle!(VkImageView, vk_image_view, VK_OBJECT_TYPE_IMAGE_VIEW);

fn shape(image: &borgvk_image) -> Shape {
    let e = image.vk.extent;
    Shape { extent: [e.width, e.height, e.depth], levels: image.vk.mip_levels, layers: image.vk.array_layers, samples: image.vk.samples }
}

/// The blocks of a format.
fn blocks(fmt: VkFormat) -> Blocks {
    let pf = unsafe { vk_format_to_pipe_format(fmt) };
    unsafe {
        Blocks {
            bs: util_format_get_blocksize(pf),
            bw: util_format_get_blockwidth(pf),
            bh: util_format_get_blockheight(pf),
            bd: util_format_get_blockdepth(pf),
        }
    }
}

/// The blocks of one aspect: a depth or stencil aspect of a combined format has its own size,
/// the block depth stays the combined format's.
fn aspect_blocks(fmt: VkFormat, aspect: VkImageAspectFlags) -> Blocks {
    let a = if aspect.count_ones() == 1 { unsafe { vk_format_get_aspect_format(fmt, aspect) } } else { fmt };
    Blocks { bd: blocks(fmt).bd, ..blocks(a) }
}

fn size(i: &VkImageCreateInfo) -> u64 {
    let e = i.extent;
    let s = Shape { extent: [e.width, e.height, e.depth], levels: i.mipLevels, layers: i.arrayLayers, samples: i.samples };
    s.size(blocks(i.format))
}

fn subresource_layout(image: &borgvk_image, sub: &VkImageSubresource) -> VkSubresourceLayout {
    let b = aspect_blocks(image.vk.format, sub.aspectMask);
    let level = sub.mipLevel;
    let (offset, stride) = shape(image).level_offset(level, sub.arrayLayer, b);
    let dim = |e: u32, block: u32| e.checked_shr(level).unwrap_or(0).max(1).div_ceil(block) as u64;
    let row = stride as u64 * b.bs as u64;
    let slice = row * dim(image.vk.extent.height, b.bh);
    VkSubresourceLayout {
        offset,
        size: slice * dim(image.vk.extent.depth, b.bd),
        rowPitch: row,
        arrayPitch: shape(image).layer_size(b),
        depthPitch: slice,
    }
}

entrypoint! {
    fn borgvk_CreateImage(device: VkDevice, info: *const VkImageCreateInfo, alloc: *const VkAllocationCallbacks, out: *mut VkImage) -> Result<(), VkError> {
        let dev = device.get().unwrap();
        let Some(image) = vk_image_create(&mut dev.vk, info, alloc, size_of::<borgvk_image>()).cast::<borgvk_image>().as_mut() else {
            return Err(vk_error!(dev, VK_ERROR_OUT_OF_HOST_MEMORY));
        };
        image.size = size(&*info);
        *out = VkImage::from_object(image);
        Ok(())
    }
}

entrypoint! {
    fn borgvk_DestroyImage(device: VkDevice, image: VkImage, alloc: *const VkAllocationCallbacks) {
        if let Some(i) = image.get() {
            vk_image_destroy(&mut device.get().unwrap().vk, alloc, &mut i.vk);
        }
    }
}

entrypoint! {
    fn borgvk_GetImageMemoryRequirements2(_device: VkDevice, info: *const VkImageMemoryRequirementsInfo2, reqs: *mut VkMemoryRequirements2) {
        requirements((*info).image.get().unwrap().size, &mut *reqs);
    }
}

entrypoint! {
    fn borgvk_GetImageSubresourceLayout2KHR(_device: VkDevice, image: VkImage, sub: *const VkImageSubresource2KHR, layout: *mut VkSubresourceLayout2KHR) {
        (*layout).subresourceLayout = subresource_layout(image.get().unwrap(), &(*sub).imageSubresource);
    }
}

entrypoint! {
    fn borgvk_BindImageMemory2(_device: VkDevice, count: u32, infos: *const VkBindImageMemoryInfo) -> VkResult {
        for i in slice(infos, count) {
            let image = i.image.get().unwrap();
            (image.mem, image.offset) = (i.memory.ptr(), i.memoryOffset);
        }
        VK_SUCCESS
    }
}

entrypoint! {
    fn borgvk_CreateImageView(device: VkDevice, info: *const VkImageViewCreateInfo, alloc: *const VkAllocationCallbacks, out: *mut VkImageView) -> Result<(), VkError> {
        let dev = device.get().unwrap();
        let v = vk_image_view_create(&mut dev.vk, info, alloc, size_of::<vk_image_view>()).cast::<vk_image_view>();
        if v.is_null() {
            return Err(vk_error!(dev, VK_ERROR_OUT_OF_HOST_MEMORY));
        }
        *out = VkImageView::from_object(v);
        Ok(())
    }
}

entrypoint! {
    fn borgvk_DestroyImageView(device: VkDevice, view: VkImageView, alloc: *const VkAllocationCallbacks) {
        if let Some(v) = view.get() {
            vk_image_view_destroy(&mut device.get().unwrap().vk, alloc, v);
        }
    }
}

pub fn z16_pack(z: f32) -> [u8; 2] {
    ((z.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16).to_le_bytes()
}

pub fn z16_unpack(t: [u8; 2]) -> f32 {
    u16::from_le_bytes(t) as f32 / 65535.0
}

#[no_mangle]
pub extern "C" fn borgvk_is_z16_s8(fmt: pipe_format) -> bool {
    fmt == PIPE_FORMAT_Z16_UNORM_S8_UINT
}

#[no_mangle]
pub extern "C" fn borgvk_is_r4g4_unorm(fmt: VkFormat) -> bool {
    fmt == VK_FORMAT_R4G4_UNORM_PACK8
}

/// # Safety
/// `width_blocks_out` is null or valid for a write.
#[no_mangle]
pub unsafe extern "C" fn borgvk_mip_level_size(
    width: u32,
    height: u32,
    depth: u32,
    level: u32,
    bs: u32,
    bw: u32,
    bh: u32,
    bd: u32,
    width_blocks_out: *mut u32,
) -> u64 {
    let (size, w) = mip_level_size([width, height, depth], level, Blocks { bs, bw, bh, bd });
    if let Some(out) = unsafe { width_blocks_out.as_mut() } {
        *out = w;
    }
    size
}

/// # Safety
/// `image` is a valid image.
#[no_mangle]
pub unsafe extern "C" fn borgvk_image_sample_plane_size(
    image: *const borgvk_image,
    bs: u32,
    bw: u32,
    bh: u32,
    bd: u32,
) -> u64 {
    shape(unsafe { &*image }).sample_plane_size(Blocks { bs, bw, bh, bd })
}

/// # Safety
/// `image` is a valid image.
#[no_mangle]
pub unsafe extern "C" fn borgvk_image_layer_size(
    image: *const borgvk_image,
    bs: u32,
    bw: u32,
    bh: u32,
    bd: u32,
) -> u64 {
    shape(unsafe { &*image }).layer_size(Blocks { bs, bw, bh, bd })
}

/// # Safety
/// `image` is a valid image; `stride_blocks_out` is null or valid for a write.
#[no_mangle]
pub unsafe extern "C" fn borgvk_image_level_offset(
    image: *const borgvk_image,
    level: u32,
    layer: u32,
    bs: u32,
    bw: u32,
    bh: u32,
    bd: u32,
    stride_blocks_out: *mut u32,
) -> u64 {
    let (offset, stride) = shape(unsafe { &*image }).level_offset(level, layer, Blocks { bs, bw, bh, bd });
    if let Some(out) = unsafe { stride_blocks_out.as_mut() } {
        *out = stride;
    }
    offset
}

/// # Safety
/// `texel` is valid for writing two bytes.
#[no_mangle]
pub unsafe extern "C" fn borgvk_z16_s8_pack_z(texel: *mut u8, z: f32) {
    unsafe { std::ptr::copy_nonoverlapping(z16_pack(z).as_ptr(), texel, 2) };
}

/// # Safety
/// `texel` is valid for reading two bytes.
#[no_mangle]
pub unsafe extern "C" fn borgvk_z16_s8_unpack_z(texel: *const u8) -> f32 {
    z16_unpack(unsafe { [*texel, *texel.add(1)] })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subresource_layout_of_a_mip_in_a_layer() {
        let mut image = borgvk_image::default();
        image.vk.format = VK_FORMAT_R8G8B8A8_UNORM;
        image.vk.extent = VkExtent3D { width: 8, height: 8, depth: 1 };
        (image.vk.mip_levels, image.vk.array_layers, image.vk.samples) = (3, 2, 1);
        let sub = VkImageSubresource { aspectMask: VK_IMAGE_ASPECT_COLOR_BIT, mipLevel: 1, arrayLayer: 1 };
        let l = subresource_layout(&image, &sub);
        assert_eq!((l.offset, l.size, l.rowPitch, l.arrayPitch, l.depthPitch), (336 + 256, 64, 16, 336, 64));
    }

    #[test]
    fn z16_round_trips() {
        assert_eq!(z16_pack(1.0), [0xff, 0xff]);
        assert_eq!(z16_unpack(z16_pack(1.0)), 1.0);
        assert_eq!(z16_unpack(z16_pack(-3.0)), 0.0);
        assert!((z16_unpack(z16_pack(0.5)) - 0.5).abs() < 1.0 / 65535.0);
        assert_eq!(z16_pack(f32::NAN), [0, 0]);
    }

    #[test]
    fn format_checks() {
        assert!(borgvk_is_z16_s8(PIPE_FORMAT_Z16_UNORM_S8_UINT));
        assert!(!borgvk_is_z16_s8(PIPE_FORMAT_R8G8B8A8_UNORM));
        assert!(borgvk_is_r4g4_unorm(VK_FORMAT_R4G4_UNORM_PACK8));
    }
}

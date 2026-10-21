// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::device::Device;
use crate::entrypoint::{entrypoint, handle, pointee, slice, vk_error, Handle};
use crate::ffi::*;
use crate::fmt;
use crate::memory::{DeviceMemory, Host};
use crate::object::{allocate, Zeroed};
use crate::memory::requirements;

use borg_layout::image::{Blocks, Shape};

handle!(VkImage, Image, VK_OBJECT_TYPE_IMAGE);
handle!(VkImageView, vk_image_view, VK_OBJECT_TYPE_IMAGE_VIEW);

#[repr(C)]
pub struct Image {
    pub vk: vk_image,
    pub size: VkDeviceSize, // bytes needed to back this image
    pub mem: *mut DeviceMemory,
    pub offset: VkDeviceSize,
}

impl Image {
    pub fn memory(&self) -> Option<&DeviceMemory> {
        unsafe { self.mem.as_ref() }
    }

    pub fn host(&self) -> Option<Host<'_>> {
        Host::of(self.memory(), self.offset)
    }
}

/// The image a view was created on, which outlives the view.
pub trait ViewImage {
    fn image(&self) -> Option<&Image>;
}

impl ViewImage for vk_image_view {
    fn image(&self) -> Option<&Image> {
        unsafe { self.image.cast::<Image>().as_ref() }
    }
}

// Numbers, a raw pointer and the runtime's image, which it zero-allocates.
unsafe impl Zeroed for Image {}

impl Default for Image {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

pub fn shape(image: &Image) -> Shape {
    let e = image.vk.extent;
    Shape { extent: [e.width, e.height, e.depth], levels: image.vk.mip_levels, layers: image.vk.array_layers, samples: image.vk.samples }
}

/// The blocks of a format.
pub fn blocks(fmt: VkFormat) -> Blocks {
    let pf = fmt::pipe(fmt);
    Blocks { bs: fmt::blocksize(pf), bw: fmt::blockwidth(pf), bh: fmt::blockheight(pf), bd: fmt::blockdepth(pf) }
}

/// The blocks of one aspect: a depth or stencil aspect of a combined format has its own size,
/// the block depth stays the combined format's.
pub fn aspect_blocks(fmt: VkFormat, aspect: VkImageAspectFlags) -> Blocks {
    let a = if aspect.count_ones() == 1 { fmt::aspect_format(fmt, aspect) } else { fmt };
    Blocks { bd: blocks(fmt).bd, ..blocks(a) }
}

fn size(i: &VkImageCreateInfo) -> u64 {
    let e = i.extent;
    let s = Shape { extent: [e.width, e.height, e.depth], levels: i.mipLevels, layers: i.arrayLayers, samples: i.samples };
    s.size(blocks(i.format))
}

fn subresource_layout(image: &Image, sub: &VkImageSubresource) -> VkSubresourceLayout {
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

pointee!(VkImageCreateInfo, VkImageMemoryRequirementsInfo2, VkImageSubresource2KHR, VkSubresourceLayout2KHR, VkImageViewCreateInfo);

entrypoint! {
    fn borgvk_CreateImage(device: &Device, info: &VkImageCreateInfo, alloc: *const VkAllocationCallbacks, out: &mut VkImage) -> Result<(), VkError> {
        let image = unsafe { allocate::<Image>(|n| vk_image_create(device.vk_ptr(), info, alloc, n)) };
        let image = image.ok_or_else(|| vk_error!(device, VK_ERROR_OUT_OF_HOST_MEMORY))?;
        image.size = size(info);
        *out = VkImage::from_object(image);
        Ok(())
    }
}

entrypoint! {
    fn borgvk_DestroyImage(device: &Device, image: Option<&mut Image>, alloc: *const VkAllocationCallbacks) {
        if let Some(i) = image {
            unsafe { vk_image_destroy(device.vk_ptr(), alloc, &mut i.vk) };
        }
    }
}

entrypoint! {
    fn borgvk_GetImageMemoryRequirements2(_device: VkDevice, info: &VkImageMemoryRequirementsInfo2, reqs: &mut VkMemoryRequirements2) {
        requirements(unsafe { info.image.get_ref() }.unwrap().size, reqs);
    }
}

entrypoint! {
    fn borgvk_GetImageSubresourceLayout2KHR(_device: VkDevice, image: &Image, sub: &VkImageSubresource2KHR, layout: &mut VkSubresourceLayout2KHR) {
        layout.subresourceLayout = subresource_layout(image, &sub.imageSubresource);
    }
}

entrypoint! {
    fn borgvk_BindImageMemory2(_device: VkDevice, count: u32, infos: *const VkBindImageMemoryInfo) -> VkResult {
        for i in unsafe { slice(infos, count) } {
            let image = unsafe { i.image.get() }.unwrap();
            (image.mem, image.offset) = (i.memory.ptr(), i.memoryOffset);
        }
        VK_SUCCESS
    }
}

entrypoint! {
    fn borgvk_CreateImageView(device: &Device, info: &VkImageViewCreateInfo, alloc: *const VkAllocationCallbacks, out: &mut VkImageView) -> Result<(), VkError> {
        let v = unsafe { vk_image_view_create(device.vk_ptr(), info, alloc, size_of::<vk_image_view>()) }.cast::<vk_image_view>();
        if v.is_null() {
            return Err(vk_error!(device, VK_ERROR_OUT_OF_HOST_MEMORY));
        }
        *out = VkImageView::from_object(v);
        Ok(())
    }
}

entrypoint! {
    fn borgvk_DestroyImageView(device: &Device, view: Option<&mut vk_image_view>, alloc: *const VkAllocationCallbacks) {
        if let Some(v) = view {
            unsafe { vk_image_view_destroy(device.vk_ptr(), alloc, v) };
        }
    }
}

pub fn z16_pack(z: f32) -> [u8; 2] {
    ((z.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16).to_le_bytes()
}

pub fn z16_unpack(t: [u8; 2]) -> f32 {
    u16::from_le_bytes(t) as f32 / 65535.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subresource_layout_of_a_mip_in_a_layer() {
        let mut image = Image::default();
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
}

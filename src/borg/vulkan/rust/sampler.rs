// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! VkSampler: the runtime's sampler and the descriptor the texture unit reads.

use crate::device::Device;
use crate::entrypoint::{entrypoint, handle, pointee, vk_error, Handle};
use crate::ffi::*;
use crate::object::{allocate, Zeroed};
use borg_layout::descriptor::SamplerDesc;

handle!(VkSampler, Sampler, VK_OBJECT_TYPE_SAMPLER);

#[repr(C)]
pub struct Sampler {
    pub vk: vk_sampler,
    pub desc: [u32; 4],
}

// Numbers and the runtime's sampler, which it zero-allocates.
unsafe impl Zeroed for Sampler {}

pub fn desc(i: &VkSamplerCreateInfo) -> SamplerDesc {
    SamplerDesc {
        mag_linear: i.magFilter == VK_FILTER_LINEAR,
        min_linear: i.minFilter == VK_FILTER_LINEAR,
        mip_linear: i.mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR,
        address: [i.addressModeU, i.addressModeV, i.addressModeW],
        border: i.borderColor,
        compare: i.compareEnable != 0,
        compare_op: i.compareOp,
        unnormalized: i.unnormalizedCoordinates != 0,
        lod_bias: i.mipLodBias,
        min_lod: i.minLod,
        max_lod: i.maxLod,
    }
}

pointee!(VkSamplerCreateInfo);

entrypoint! {
    fn borgvk_CreateSampler(device: &Device, info: &VkSamplerCreateInfo, alloc: *const VkAllocationCallbacks, out: &mut VkSampler) -> Result<(), VkError> {
        let s = unsafe { allocate::<Sampler>(|n| vk_sampler_create(device.vk_ptr(), info, alloc, n)) };
        let s = s.ok_or_else(|| vk_error!(device, VK_ERROR_OUT_OF_HOST_MEMORY))?;
        s.desc = desc(info).pack();
        *out = VkSampler::from_object(s);
        Ok(())
    }
}

entrypoint! {
    fn borgvk_DestroySampler(device: &Device, sampler: Option<&mut Sampler>, alloc: *const VkAllocationCallbacks) {
        if let Some(s) = sampler {
            unsafe { vk_sampler_destroy(device.vk_ptr(), alloc, &mut s.vk) };
        }
    }
}

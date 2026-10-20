// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! VkSampler: the runtime's sampler and the descriptor the texture unit reads.

use crate::entrypoint::{entrypoint, handle, vk_error, Handle};
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

entrypoint! {
    fn borgvk_CreateSampler(device: VkDevice, info: *const VkSamplerCreateInfo, alloc: *const VkAllocationCallbacks, out: *mut VkSampler) -> Result<(), VkError> {
        let dev = device.get().unwrap();
        let Some(s) = allocate::<Sampler>(|n| vk_sampler_create(&mut dev.vk, info, alloc, n)) else {
            return Err(vk_error!(dev, VK_ERROR_OUT_OF_HOST_MEMORY));
        };
        s.desc = desc(&*info).pack();
        *out = VkSampler::from_object(s);
        Ok(())
    }
}

entrypoint! {
    fn borgvk_DestroySampler(device: VkDevice, sampler: VkSampler, alloc: *const VkAllocationCallbacks) {
        if let Some(s) = sampler.get() {
            vk_sampler_destroy(&mut device.get().unwrap().vk, alloc, &mut s.vk);
        }
    }
}

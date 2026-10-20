// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! VkQueryPool: no queries are counted yet, every result reads as zero.

use crate::device::Device;
use crate::entrypoint::{entrypoint, handle, vk_error, Handle};
use crate::ffi::*;
use crate::object::{allocate, Zeroed};
use std::ffi::c_void;

#[repr(C)]
pub struct QueryPool {
    base: vk_object_base,
}

handle!(VkQueryPool, QueryPool, VK_OBJECT_TYPE_QUERY_POOL);

// The runtime's object base, which it zero-allocates.
unsafe impl Zeroed for QueryPool {}

entrypoint! {
    fn borgvk_CreateQueryPool(device: &Device, _info: *const VkQueryPoolCreateInfo, alloc: *const VkAllocationCallbacks, out: &mut VkQueryPool) -> Result<(), VkError> {
        let p = unsafe { allocate::<QueryPool>(|n| vk_object_zalloc(device.vk_ptr(), alloc, n, VK_OBJECT_TYPE_QUERY_POOL)) };
        *out = VkQueryPool::from_object(p.ok_or_else(|| vk_error!(device, VK_ERROR_OUT_OF_HOST_MEMORY))?);
        Ok(())
    }
}

entrypoint! {
    fn borgvk_DestroyQueryPool(device: &Device, pool: Option<&mut QueryPool>, alloc: *const VkAllocationCallbacks) {
        if let Some(p) = pool {
            unsafe { vk_object_free(device.vk_ptr(), alloc, std::ptr::from_mut(p).cast()) };
        }
    }
}

entrypoint! {
    fn borgvk_GetQueryPoolResults(_device: VkDevice, _pool: VkQueryPool, _first: u32, _count: u32, size: usize, data: *mut c_void, _stride: VkDeviceSize, _flags: VkQueryResultFlags) -> VkResult {
        if !data.is_null() && size != 0 {
            unsafe { data.cast::<u8>().write_bytes(0, size) };
        }
        VK_SUCCESS
    }
}

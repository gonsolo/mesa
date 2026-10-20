// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! VkQueryPool: no queries are counted yet, every result reads as zero.

use crate::entrypoint::{entrypoint, handle, vk_error, Handle};
use crate::ffi::*;
use std::ffi::c_void;

#[repr(C)]
pub struct QueryPool {
    base: vk_object_base,
}

handle!(VkQueryPool, QueryPool, VK_OBJECT_TYPE_QUERY_POOL);

entrypoint! {
    fn borgvk_CreateQueryPool(device: VkDevice, _info: *const VkQueryPoolCreateInfo, alloc: *const VkAllocationCallbacks, out: *mut VkQueryPool) -> Result<(), VkError> {
        let dev = device.get().unwrap();
        let p = vk_object_zalloc(&mut dev.vk, alloc, size_of::<QueryPool>(), VK_OBJECT_TYPE_QUERY_POOL);
        if p.is_null() {
            return Err(vk_error!(dev, VK_ERROR_OUT_OF_HOST_MEMORY));
        }
        *out = VkQueryPool::from_object(p.cast());
        Ok(())
    }
}

entrypoint! {
    fn borgvk_DestroyQueryPool(device: VkDevice, pool: VkQueryPool, alloc: *const VkAllocationCallbacks) {
        if let Some(p) = pool.get() {
            vk_object_free(&mut device.get().unwrap().vk, alloc, (p as *mut QueryPool).cast());
        }
    }
}

entrypoint! {
    fn borgvk_GetQueryPoolResults(_device: VkDevice, _pool: VkQueryPool, _first: u32, _count: u32, size: usize, data: *mut c_void, _stride: VkDeviceSize, _flags: VkQueryResultFlags) -> VkResult {
        if !data.is_null() && size != 0 {
            data.cast::<u8>().write_bytes(0, size);
        }
        VK_SUCCESS
    }
}

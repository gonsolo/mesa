// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! VkQueryPool: occlusion queries count the samples of the draws between begin and end
//! (OCC_COUNT, summed per render by the firmware); other types read as zero.

use crate::cmd_buffer::CommandBuffer;
use crate::device::Device;
use crate::entrypoint::{entrypoint, handle, pointee, vk_error, Handle};
use crate::ffi::*;
use crate::memory::Buffer;
use crate::object::{allocate, Zeroed};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

#[repr(C)]
pub struct QueryPool {
    base: vk_object_base,
    count: u32,
    // `count` slots follow.
}

#[repr(C)]
pub struct Slot {
    pub result: AtomicU64,
    pub avail: AtomicBool,
}

/// An occlusion query's part in one batch: the heap word the firmware counts into.
#[derive(Clone, Copy)]
pub struct OccRun {
    pub off: u32,
    pub pool: *const QueryPool,
    pub query: u32,
    pub end: bool, // the query ends in this batch
}

handle!(VkQueryPool, QueryPool, VK_OBJECT_TYPE_QUERY_POOL);
pointee!(VkQueryPoolCreateInfo);

// The runtime's object base, which it zero-allocates, a count and atomics.
unsafe impl Zeroed for QueryPool {}

impl QueryPool {
    pub fn slots(&self) -> &[Slot] {
        unsafe { std::slice::from_raw_parts(std::ptr::from_ref(self).add(1).cast(), self.count as usize) }
    }

    pub fn slot(&self, q: u32) -> Option<&Slot> {
        self.slots().get(q as usize)
    }

    fn range(&self, first: u32, count: u32) -> &[Slot] {
        let s = self.slots();
        let a = (first as usize).min(s.len());
        &s[a..(a + count as usize).min(s.len())]
    }
}

/// The value and then availability of one query, as GetQueryPoolResults and CmdCopyQueryPoolResults
/// write them; None when neither is written.
fn result_words(s: &Slot, flags: VkQueryResultFlags) -> (bool, [u64; 2], usize) {
    let avail = s.avail.load(Relaxed);
    let n = 1 + (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT != 0) as usize;
    let value = avail || flags & VK_QUERY_RESULT_PARTIAL_BIT != 0;
    (value, [s.result.load(Relaxed), avail as u64], n)
}

fn write_result(dst: &mut [u8], s: &Slot, flags: VkQueryResultFlags) {
    let (value, w, n) = result_words(s, flags);
    let wide = flags & VK_QUERY_RESULT_64_BIT != 0;
    let sz = if wide { 8 } else { 4 };
    for (k, &v) in w[..n].iter().enumerate() {
        if (k == 0 && !value) || dst.len() < (k + 1) * sz {
            continue;
        }
        let b = &mut dst[k * sz..(k + 1) * sz];
        if wide { b.copy_from_slice(&v.to_ne_bytes()) } else { b.copy_from_slice(&(v.min(u32::MAX as u64) as u32).to_ne_bytes()) }
    }
}

entrypoint! {
    fn borgvk_CreateQueryPool(device: &Device, info: &VkQueryPoolCreateInfo, alloc: *const VkAllocationCallbacks, out: &mut VkQueryPool) -> Result<(), VkError> {
        let extra = info.queryCount as usize * size_of::<Slot>();
        let p = unsafe { allocate::<QueryPool>(|n| vk_object_zalloc(device.vk_ptr(), alloc, n + extra, VK_OBJECT_TYPE_QUERY_POOL)) };
        let p = p.ok_or_else(|| vk_error!(device, VK_ERROR_OUT_OF_HOST_MEMORY))?;
        p.count = info.queryCount;
        *out = VkQueryPool::from_object(p);
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
    fn borgvk_GetQueryPoolResults(_device: VkDevice, pool: &QueryPool, first: u32, count: u32, size: usize, data: *mut c_void, stride: VkDeviceSize, flags: VkQueryResultFlags) -> VkResult {
        if data.is_null() {
            return VK_SUCCESS;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(data.cast::<u8>(), size) };
        let mut ready = true;
        // Submits run to completion, so a query that is not available now never will be: no wait.
        for (i, s) in pool.range(first, count).iter().enumerate() {
            ready &= s.avail.load(Relaxed);
            let at = (i as u64 * stride) as usize;
            if at < out.len() {
                write_result(&mut out[at..], s, flags);
            }
        }
        if ready || flags & VK_QUERY_RESULT_WAIT_BIT != 0 { VK_SUCCESS } else { VK_NOT_READY }
    }
}

entrypoint! {
    fn borgvk_CmdResetQueryPool(c: &mut CommandBuffer, pool: &QueryPool, first: u32, count: u32) {
        crate::draw::flush_draws(c);
        for s in pool.range(first, count) {
            s.result.store(0, Relaxed);
            s.avail.store(false, Relaxed);
        }
    }
}

entrypoint! {
    fn borgvk_CmdBeginQuery(c: &mut CommandBuffer, pool: &QueryPool, query: u32, _flags: VkQueryControlFlags) {
        // The next draw opens it in its batch (draw.rs).
        (c.occ_pool, c.occ_query, c.occ_open) = (std::ptr::from_ref(pool), query, false);
    }
}

entrypoint! {
    fn borgvk_CmdEndQuery(c: &mut CommandBuffer, pool: &QueryPool, query: u32) {
        if c.occ_open {
            crate::draw::close_query(c, true);
        } else if let Some(s) = pool.slot(query) {
            s.avail.store(true, Relaxed);
        }
        c.occ_pool = std::ptr::null();
    }
}

entrypoint! {
    fn borgvk_CmdCopyQueryPoolResults(c: &mut CommandBuffer, pool: &QueryPool, first: u32, count: u32, buffer: &Buffer, offset: VkDeviceSize, stride: VkDeviceSize, flags: VkQueryResultFlags) {
        crate::draw::flush_draws(c);
        let Some(h) = buffer.host() else { return };
        for (i, s) in pool.range(first, count).iter().enumerate() {
            let mut b = [0u8; 16];
            let at = (offset + i as u64 * stride) as usize;
            let (value, _, n) = result_words(s, flags);
            let sz = if flags & VK_QUERY_RESULT_64_BIT != 0 { 8 } else { 4 };
            write_result(&mut b, s, flags);
            // An unavailable value is left as it is in the buffer.
            let skip = if value { 0 } else { sz };
            h.write(at + skip, &b[skip..n * sz]);
        }
    }
}

// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Transfers, done by the host in mapped memory: buffer fills, updates and copies.

use crate::entrypoint::{entrypoint, slice, Handle};
use crate::ffi::*;
use std::ffi::c_void;

/// The buffer's bytes from its binding on, when it is bound to mapped memory.
fn host(b: &borgvk_buffer) -> Option<*mut u8> {
    let m = unsafe { b.mem.as_ref() }.filter(|m| !m.map.is_null())?;
    Some(unsafe { m.map.cast::<u8>().add(b.offset as usize) })
}

entrypoint! {
    fn borgvk_CmdFillBuffer(_cmd: VkCommandBuffer, buffer: VkBuffer, offset: VkDeviceSize, size: VkDeviceSize, data: u32) {
        let Some(b) = buffer.get() else { return };
        let Some(base) = host(b) else { return };
        let size = if size == VK_WHOLE_SIZE as u64 { b.vk.size - offset } else { size };
        let dst = base.add(offset as usize).cast::<u32>();
        for i in 0..(size / 4) as usize {
            dst.add(i).write_unaligned(data);
        }
    }
}

entrypoint! {
    fn borgvk_CmdUpdateBuffer(_cmd: VkCommandBuffer, buffer: VkBuffer, offset: VkDeviceSize, size: VkDeviceSize, data: *const c_void) {
        let Some(base) = buffer.get().and_then(|b| host(b)) else { return };
        if !data.is_null() {
            std::ptr::copy_nonoverlapping(data.cast::<u8>(), base.add(offset as usize), size as usize);
        }
    }
}

fn copy_buffer(src: VkBuffer, dst: VkBuffer, regions: impl Iterator<Item = (u64, u64, u64)>) {
    let (Some(s), Some(d)) = (unsafe { (src.get().and_then(|b| host(b)), dst.get().and_then(|b| host(b))) }) else { return };
    for (from, to, size) in regions {
        unsafe { std::ptr::copy(s.add(from as usize), d.add(to as usize), size as usize) };
    }
}

entrypoint! {
    fn borgvk_CmdCopyBuffer(_cmd: VkCommandBuffer, src: VkBuffer, dst: VkBuffer, count: u32, regions: *const VkBufferCopy) {
        copy_buffer(src, dst, slice(regions, count).iter().map(|r| (r.srcOffset, r.dstOffset, r.size)));
    }
}

entrypoint! {
    fn borgvk_CmdCopyBuffer2(_cmd: VkCommandBuffer, info: *const VkCopyBufferInfo2) {
        let i = &*info;
        copy_buffer(i.srcBuffer, i.dstBuffer, slice(i.pRegions, i.regionCount).iter().map(|r| (r.srcOffset, r.dstOffset, r.size)));
    }
}

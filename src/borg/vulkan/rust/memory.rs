// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Device memory, buffers and buffer views. Memory is a GEM buffer of the DRM device mapped into
//! the process, or malloc'd without one; either way host-coherent.

use crate::entrypoint::{chain, entrypoint, handle, pointee, slice, vk_error, Handle};
use crate::drm::{gem_close, ioctl, GEM_CREATE, GEM_MMAP};
use crate::ffi::*;
use crate::object::{allocate, Zeroed};
use crate::device::Device;
use crate::util::mmap;
use std::ffi::c_void;
use std::ptr::null_mut;

handle!(VkDeviceMemory, DeviceMemory, VK_OBJECT_TYPE_DEVICE_MEMORY);
handle!(VkBuffer, Buffer, VK_OBJECT_TYPE_BUFFER);
handle!(VkBufferView, BufferView, VK_OBJECT_TYPE_BUFFER_VIEW);

#[repr(C)]
pub struct DeviceMemory {
    pub vk: vk_device_memory,
    pub map: *mut c_void,  // mmap(drm_fd, gem_offset) or malloc'd
    pub size: usize,       // allocation size for munmap
    pub gem_handle: u32,   // 0 = malloc path (no DRM)
}

#[repr(C)]
pub struct Buffer {
    pub vk: vk_buffer,
    pub mem: *mut DeviceMemory,
    pub offset: VkDeviceSize,
}

#[repr(C)]
pub struct BufferView {
    pub vk: vk_buffer_view,
}

// Numbers, raw pointers and runtime objects the runtime zero-allocates.
unsafe impl Zeroed for DeviceMemory {}
unsafe impl Zeroed for Buffer {}
unsafe impl Zeroed for BufferView {}

impl Default for DeviceMemory {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// Backing for `size` bytes and its GEM handle, 0 for malloc'd memory.
fn back(dev: &Device, size: usize) -> Result<(*mut c_void, u32), VkResult> {
    let fd = dev.drm_fd;
    if fd < 0 {
        let map = unsafe { malloc(size as _) };
        return if map.is_null() { Err(VK_ERROR_OUT_OF_HOST_MEMORY) } else { Ok((map, 0)) };
    }
    let mut create = drm_borg_gem_create { size: size as u64, ..Default::default() };
    if !ioctl(fd, GEM_CREATE, &mut create) {
        return Err(VK_ERROR_OUT_OF_DEVICE_MEMORY);
    }
    let mut m = drm_borg_gem_mmap { handle: create.handle, ..Default::default() };
    if !ioctl(fd, GEM_MMAP, &mut m) {
        gem_close(fd, create.handle);
        return Err(VK_ERROR_OUT_OF_DEVICE_MEMORY);
    }
    let map = unsafe { mmap(null_mut(), size, (PROT_READ | PROT_WRITE) as i32, MAP_SHARED as i32, fd, m.offset as i64) };
    if map as isize == -1 {
        gem_close(fd, create.handle);
        return Err(VK_ERROR_OUT_OF_HOST_MEMORY);
    }
    Ok((map, create.handle))
}

pointee!(VkMemoryAllocateInfo, VkMemoryMapInfo, VkBufferCreateInfo, VkBufferViewCreateInfo, VkDeviceBufferMemoryRequirements, VkMemoryRequirements2, *mut c_void);

entrypoint! {
    fn borgvk_AllocateMemory(device: &Device, info: &VkMemoryAllocateInfo, alloc: *const VkAllocationCallbacks, out: &mut VkDeviceMemory) -> Result<(), VkError> {
        let mem = unsafe { allocate::<DeviceMemory>(|n| vk_device_memory_create(device.vk_ptr(), info, alloc, n)) };
        let mem = mem.ok_or_else(|| vk_error!(device, VK_ERROR_OUT_OF_HOST_MEMORY))?;
        let size = (info.allocationSize as usize).max(1);
        match back(device, size) {
            Ok((map, gem_handle)) => (mem.map, mem.size, mem.gem_handle) = (map, size, gem_handle),
            Err(r) => {
                unsafe { vk_device_memory_destroy(device.vk_ptr(), alloc, &mut mem.vk) };
                return Err(vk_error!(device, r));
            }
        }
        *out = VkDeviceMemory::from_object(mem);
        Ok(())
    }
}

entrypoint! {
    fn borgvk_FreeMemory(device: &Device, memory: Option<&mut DeviceMemory>, alloc: *const VkAllocationCallbacks) {
        let Some(mem) = memory else { return };
        unsafe {
            if mem.gem_handle != 0 {
                munmap(mem.map, mem.size as _);
                gem_close(device.drm_fd, mem.gem_handle);
            } else {
                free(mem.map);
            }
            vk_device_memory_destroy(device.vk_ptr(), alloc, &mut mem.vk);
        }
    }
}

// No memory type is lazily allocated, so everything allocated is committed.
entrypoint! {
    fn borgvk_GetDeviceMemoryCommitment(_device: VkDevice, memory: &DeviceMemory, out: &mut VkDeviceSize) {
        *out = memory.vk.size;
    }
}

entrypoint! {
    fn borgvk_MapMemory2(_device: VkDevice, info: &VkMemoryMapInfo, out: &mut *mut c_void) -> VkResult {
        let mem = unsafe { info.memory.get() }.unwrap();
        *out = mem.map.wrapping_byte_add(info.offset as usize);
        VK_SUCCESS
    }
}

entrypoint! {
    fn borgvk_UnmapMemory2(_device: VkDevice, _info: *const VkMemoryUnmapInfo) -> VkResult {
        VK_SUCCESS
    }
}

entrypoint! {
    fn borgvk_FlushMappedMemoryRanges(_device: VkDevice, _count: u32, _ranges: *const VkMappedMemoryRange) -> VkResult {
        VK_SUCCESS
    }
}

entrypoint! {
    fn borgvk_InvalidateMappedMemoryRanges(_device: VkDevice, _count: u32, _ranges: *const VkMappedMemoryRange) -> VkResult {
        VK_SUCCESS
    }
}

entrypoint! {
    fn borgvk_CreateBuffer(device: &Device, info: &VkBufferCreateInfo, alloc: *const VkAllocationCallbacks, out: &mut VkBuffer) -> Result<(), VkError> {
        let b = unsafe { allocate::<Buffer>(|n| vk_buffer_create(device.vk_ptr(), info, alloc, n)) };
        *out = VkBuffer::from_object(b.ok_or_else(|| vk_error!(device, VK_ERROR_OUT_OF_HOST_MEMORY))?);
        Ok(())
    }
}

entrypoint! {
    fn borgvk_DestroyBuffer(device: &Device, buffer: Option<&mut Buffer>, alloc: *const VkAllocationCallbacks) {
        if let Some(b) = buffer {
            unsafe { vk_buffer_destroy(device.vk_ptr(), alloc, &mut b.vk) };
        }
    }
}

entrypoint! {
    fn borgvk_CreateBufferView(device: &Device, info: &VkBufferViewCreateInfo, alloc: *const VkAllocationCallbacks, out: &mut VkBufferView) -> Result<(), VkError> {
        let v = unsafe { allocate::<BufferView>(|n| vk_buffer_view_create(device.vk_ptr(), info, alloc, n)) };
        *out = VkBufferView::from_object(v.ok_or_else(|| vk_error!(device, VK_ERROR_OUT_OF_HOST_MEMORY))?);
        Ok(())
    }
}

entrypoint! {
    fn borgvk_DestroyBufferView(device: &Device, view: Option<&mut BufferView>, alloc: *const VkAllocationCallbacks) {
        if let Some(v) = view {
            unsafe { vk_buffer_view_destroy(device.vk_ptr(), alloc, &mut v.vk) };
        }
    }
}

/// Requirements in the one memory type, 256-byte aligned, never a dedicated allocation.
pub fn requirements(size: u64, reqs: &mut VkMemoryRequirements2) {
    let r = &mut reqs.memoryRequirements;
    (r.size, r.alignment, r.memoryTypeBits) = (size, 256, 1);
    for s in unsafe { chain(reqs.pNext) } {
        if unsafe { (*s).sType } == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS {
            let d = unsafe { &mut *s.cast::<VkMemoryDedicatedRequirements>() };
            d.prefersDedicatedAllocation = VK_FALSE;
            d.requiresDedicatedAllocation = VK_FALSE;
        }
    }
}

fn buffer_size(size: u64) -> u64 {
    if size > u64::MAX - 255 { u64::MAX } else { size.next_multiple_of(256) }
}

// The common GetBufferMemoryRequirements2 delegates here (maintenance4).
entrypoint! {
    fn borgvk_GetDeviceBufferMemoryRequirements(_device: VkDevice, info: &VkDeviceBufferMemoryRequirements, reqs: &mut VkMemoryRequirements2) {
        requirements(buffer_size(unsafe { (*info.pCreateInfo).size }), reqs);
    }
}

entrypoint! {
    fn borgvk_BindBufferMemory2(_device: VkDevice, count: u32, infos: *const VkBindBufferMemoryInfo) -> VkResult {
        for i in unsafe { slice(infos, count) } {
            let b = unsafe { i.buffer.get() }.unwrap();
            (b.mem, b.offset) = (i.memory.ptr(), i.memoryOffset);
        }
        VK_SUCCESS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffers_round_up_to_256_and_saturate() {
        let mut dedicated = VkMemoryDedicatedRequirements {
            sType: VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
            prefersDedicatedAllocation: VK_TRUE,
            ..Default::default()
        };
        let mut r = VkMemoryRequirements2 { pNext: (&mut dedicated as *mut VkMemoryDedicatedRequirements).cast(), ..Default::default() };
        requirements(buffer_size(257), &mut r);
        assert_eq!((r.memoryRequirements.size, r.memoryRequirements.alignment, r.memoryRequirements.memoryTypeBits), (512, 256, 1));
        assert_eq!(dedicated.prefersDedicatedAllocation, VK_FALSE);
        assert_eq!(buffer_size(u64::MAX - 100), u64::MAX);
    }
}

// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Device memory, buffers and buffer views. Memory is a GEM buffer of the DRM device mapped into
//! the process, or malloc'd without one; either way host-coherent.

use crate::entrypoint::{chain, entrypoint, handle, slice, vk_error, Handle};
use crate::ffi::*;
use crate::util::mmap;
use std::ffi::c_void;
use std::ptr::null_mut;

handle!(VkDeviceMemory, borgvk_device_memory, VK_OBJECT_TYPE_DEVICE_MEMORY);
handle!(VkBuffer, borgvk_buffer, VK_OBJECT_TYPE_BUFFER);
handle!(VkBufferView, borgvk_buffer_view, VK_OBJECT_TYPE_BUFFER_VIEW);

/// `_IOC` in Linux's generic ioctl encoding, which x86 and RISC-V use, for DRM's type 'd'.
const fn drm_ioc(dir: u64, nr: u32, size: usize) -> u64 {
    dir << 30 | (size as u64) << 16 | (b'd' as u64) << 8 | nr as u64
}
const GEM_CREATE: u64 = drm_ioc(3, DRM_COMMAND_BASE + DRM_BORG_GEM_CREATE, size_of::<drm_borg_gem_create>());
const GEM_MMAP: u64 = drm_ioc(3, DRM_COMMAND_BASE + DRM_BORG_GEM_MMAP, size_of::<drm_borg_gem_mmap>());
const GEM_CLOSE: u64 = drm_ioc(1, 0x09, size_of::<drm_gem_close>());

fn ioctl<T>(fd: i32, request: u64, arg: &mut T) -> bool {
    unsafe { drmIoctl(fd, request as _, (arg as *mut T).cast()) >= 0 }
}

fn gem_close(fd: i32, handle: u32) {
    ioctl(fd, GEM_CLOSE, &mut drm_gem_close { handle, ..Default::default() });
}

/// Backing for `size` bytes and its GEM handle, 0 for malloc'd memory.
fn back(dev: &borgvk_device, size: usize) -> Result<(*mut c_void, u32), VkResult> {
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

entrypoint! {
    fn borgvk_AllocateMemory(device: VkDevice, info: *const VkMemoryAllocateInfo, alloc: *const VkAllocationCallbacks, out: *mut VkDeviceMemory) -> Result<(), VkError> {
        let dev = device.get().unwrap();
        let mem = vk_device_memory_create(&mut dev.vk, info, alloc, size_of::<borgvk_device_memory>());
        let Some(mem) = mem.cast::<borgvk_device_memory>().as_mut() else {
            return Err(vk_error!(dev, VK_ERROR_OUT_OF_HOST_MEMORY));
        };
        let size = ((*info).allocationSize as usize).max(1);
        match back(dev, size) {
            Ok((map, gem_handle)) => (mem.map, mem.size, mem.gem_handle) = (map, size, gem_handle),
            Err(r) => {
                vk_device_memory_destroy(&mut dev.vk, alloc, &mut mem.vk);
                return Err(vk_error!(dev, r));
            }
        }
        *out = VkDeviceMemory::from_object(mem);
        Ok(())
    }
}

entrypoint! {
    fn borgvk_FreeMemory(device: VkDevice, memory: VkDeviceMemory, alloc: *const VkAllocationCallbacks) {
        let Some(mem) = memory.get() else {
            return;
        };
        let dev = device.get().unwrap();
        if mem.gem_handle != 0 {
            munmap(mem.map, mem.size as _);
            gem_close(dev.drm_fd, mem.gem_handle);
        } else {
            free(mem.map);
        }
        vk_device_memory_destroy(&mut dev.vk, alloc, &mut mem.vk);
    }
}

// No memory type is lazily allocated, so everything allocated is committed.
entrypoint! {
    fn borgvk_GetDeviceMemoryCommitment(_device: VkDevice, memory: VkDeviceMemory, out: *mut VkDeviceSize) {
        *out = memory.get().map_or(0, |m| m.vk.size);
    }
}

entrypoint! {
    fn borgvk_MapMemory2(_device: VkDevice, info: *const VkMemoryMapInfo, out: *mut *mut c_void) -> VkResult {
        let mem = (*info).memory.get().unwrap();
        *out = mem.map.byte_add((*info).offset as usize);
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
    fn borgvk_CreateBuffer(device: VkDevice, info: *const VkBufferCreateInfo, alloc: *const VkAllocationCallbacks, out: *mut VkBuffer) -> Result<(), VkError> {
        let dev = device.get().unwrap();
        let b = vk_buffer_create(&mut dev.vk, info, alloc, size_of::<borgvk_buffer>()).cast::<borgvk_buffer>();
        if b.is_null() {
            return Err(vk_error!(dev, VK_ERROR_OUT_OF_HOST_MEMORY));
        }
        *out = VkBuffer::from_object(b);
        Ok(())
    }
}

entrypoint! {
    fn borgvk_DestroyBuffer(device: VkDevice, buffer: VkBuffer, alloc: *const VkAllocationCallbacks) {
        if let Some(b) = buffer.get() {
            vk_buffer_destroy(&mut device.get().unwrap().vk, alloc, &mut b.vk);
        }
    }
}

entrypoint! {
    fn borgvk_CreateBufferView(device: VkDevice, info: *const VkBufferViewCreateInfo, alloc: *const VkAllocationCallbacks, out: *mut VkBufferView) -> Result<(), VkError> {
        let dev = device.get().unwrap();
        let v = vk_buffer_view_create(&mut dev.vk, info, alloc, size_of::<borgvk_buffer_view>()).cast::<borgvk_buffer_view>();
        if v.is_null() {
            return Err(vk_error!(dev, VK_ERROR_OUT_OF_HOST_MEMORY));
        }
        *out = VkBufferView::from_object(v);
        Ok(())
    }
}

entrypoint! {
    fn borgvk_DestroyBufferView(device: VkDevice, view: VkBufferView, alloc: *const VkAllocationCallbacks) {
        if let Some(v) = view.get() {
            vk_buffer_view_destroy(&mut device.get().unwrap().vk, alloc, &mut v.vk);
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
    fn borgvk_GetDeviceBufferMemoryRequirements(_device: VkDevice, info: *const VkDeviceBufferMemoryRequirements, reqs: *mut VkMemoryRequirements2) {
        requirements(buffer_size((*(*info).pCreateInfo).size), &mut *reqs);
    }
}

entrypoint! {
    fn borgvk_BindBufferMemory2(_device: VkDevice, count: u32, infos: *const VkBindBufferMemoryInfo) -> VkResult {
        for i in slice(infos, count) {
            let b = i.buffer.get().unwrap();
            (b.mem, b.offset) = (i.memory.ptr(), i.memoryOffset);
        }
        VK_SUCCESS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ioctl_numbers() {
        assert_eq!(GEM_CLOSE, 0x4008_6409);
        assert_eq!(GEM_CREATE, 0xc010_6445);
        assert_eq!(GEM_MMAP, 0xc010_6446);
    }

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

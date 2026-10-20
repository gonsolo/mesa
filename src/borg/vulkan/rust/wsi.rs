// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Window systems: Mesa's headless WSI backend, so swapchains need no display; X11 surfaces are
//! headless ones.

use crate::entrypoint::{entrypoint, handle, Handle};
use crate::ffi::*;
use std::ffi::{c_char, c_void};
use std::mem::zeroed;
use std::ptr::null;

handle!(VkPhysicalDevice, borgvk_physical_device, VK_OBJECT_TYPE_PHYSICAL_DEVICE);

unsafe extern "C" fn proc_addr(pd: VkPhysicalDevice, name: *const c_char) -> PFN_vkVoidFunction {
    unsafe { vk_instance_get_proc_addr_unchecked(pd.get().unwrap().vk.instance, name) }
}

/// # Safety
/// `pd` is a valid physical device.
#[no_mangle]
pub unsafe extern "C" fn borgvk_wsi_init(pd: *mut borgvk_physical_device) -> VkResult {
    let p = unsafe { &mut *pd };
    let opts = wsi_device_options { sw_device: true, ..unsafe { zeroed() } };
    let alloc = unsafe { &(*p.vk.instance).alloc };
    let r = unsafe { wsi_device_init(&mut p.wsi_device, VkPhysicalDevice::from_object(pd), Some(proc_addr), alloc, -1, null(), &opts) };
    if r != VK_SUCCESS {
        return r;
    }
    p.wsi_device.force_headless_swapchain = true;
    p.vk.wsi_device = &mut p.wsi_device;
    VK_SUCCESS
}

/// # Safety
/// `pd` is a valid physical device with WSI.
#[no_mangle]
pub unsafe extern "C" fn borgvk_wsi_finish(pd: *mut borgvk_physical_device) {
    let p = unsafe { &mut *pd };
    p.vk.wsi_device = std::ptr::null_mut();
    unsafe { wsi_device_finish(&mut p.wsi_device, &(*p.vk.instance).alloc) };
}

unsafe fn headless_surface(instance: VkInstance, alloc: *const VkAllocationCallbacks, out: *mut VkSurfaceKHR) -> VkResult {
    let inst = instance.cast::<vk_instance>();
    let size = size_of::<VkIcdSurfaceHeadless>();
    let s = unsafe { vk_alloc2(&(*inst).alloc, alloc, size, 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT) }.cast::<VkIcdSurfaceHeadless>();
    if s.is_null() {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    unsafe {
        (*s).base.platform = VK_ICD_WSI_PLATFORM_HEADLESS;
        *out = s.cast();
    }
    VK_SUCCESS
}

// The create infos are not read: any X11 surface becomes a headless one.
entrypoint! {
    fn borgvk_CreateXlibSurfaceKHR(instance: VkInstance, _info: *const c_void, alloc: *const VkAllocationCallbacks, out: *mut VkSurfaceKHR) -> VkResult {
        headless_surface(instance, alloc, out)
    }
}

entrypoint! {
    fn borgvk_CreateXcbSurfaceKHR(instance: VkInstance, _info: *const c_void, alloc: *const VkAllocationCallbacks, out: *mut VkSurfaceKHR) -> VkResult {
        headless_surface(instance, alloc, out)
    }
}

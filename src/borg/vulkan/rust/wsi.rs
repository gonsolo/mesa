// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Window systems: Mesa's WSI, a software device's swapchains presented by the CPU.

use crate::entrypoint::{handle, Handle};
use crate::ffi::*;
use crate::physical_device::PhysicalDevice;
use std::ffi::c_char;
use std::mem::zeroed;
use std::ptr::null;

handle!(VkPhysicalDevice, PhysicalDevice, VK_OBJECT_TYPE_PHYSICAL_DEVICE);

unsafe extern "C" fn proc_addr(pd: VkPhysicalDevice, name: *const c_char) -> PFN_vkVoidFunction {
    unsafe { vk_instance_get_proc_addr_unchecked(pd.get().unwrap().vk.instance, name) }
}

/// The display is the Borg device's own.
unsafe extern "C" fn present_on_borg(_pd: VkPhysicalDevice, _fd: i32) -> bool {
    true
}

/// With a display, swapchain images are blitted to dma-bufs the display shows; without one the CPU
/// presents them.
pub fn wsi_init(p: &mut PhysicalDevice) -> VkResult {
    let opts = wsi_device_options { sw_device: p.display_fd < 0, ..unsafe { zeroed() } };
    let alloc = unsafe { &(*p.vk.instance).alloc };
    let fd = p.display_fd;
    let r = unsafe { wsi_device_init(&mut p.wsi_device, VkPhysicalDevice::from_object(&mut *p), Some(proc_addr), alloc, fd, null(), &opts) };
    if r != VK_SUCCESS {
        return r;
    }
    if fd >= 0 {
        p.wsi_device.can_present_on_device = Some(present_on_borg);
        // Optimal images are tiled: the display gets a linear copy.
        p.wsi_device.supports_scanout = false;
        crate::scanout::KMS.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    p.vk.wsi_device = &mut p.wsi_device;
    VK_SUCCESS
}

pub fn wsi_finish(p: &mut PhysicalDevice) {
    p.vk.wsi_device = std::ptr::null_mut();
    unsafe { wsi_device_finish(&mut p.wsi_device, &(*p.vk.instance).alloc) };
}

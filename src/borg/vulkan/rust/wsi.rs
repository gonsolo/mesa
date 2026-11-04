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

pub fn wsi_init(p: &mut PhysicalDevice) -> VkResult {
    let opts = wsi_device_options { sw_device: true, ..unsafe { zeroed() } };
    let alloc = unsafe { &(*p.vk.instance).alloc };
    let r = unsafe { wsi_device_init(&mut p.wsi_device, VkPhysicalDevice::from_object(&mut *p), Some(proc_addr), alloc, -1, null(), &opts) };
    if r != VK_SUCCESS {
        return r;
    }
    p.vk.wsi_device = &mut p.wsi_device;
    VK_SUCCESS
}

pub fn wsi_finish(p: &mut PhysicalDevice) {
    p.vk.wsi_device = std::ptr::null_mut();
    unsafe { wsi_device_finish(&mut p.wsi_device, &(*p.vk.instance).alloc) };
}

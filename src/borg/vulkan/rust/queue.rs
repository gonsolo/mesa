// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Queue submission: the waits, the recorded commands replayed through the device's dispatch
//! table, the cube path, the signals.

use crate::ffi::*;

unsafe fn work(q: &mut vk_queue, submit: &mut vk_queue_submit) -> VkResult {
    let device = unsafe { &mut *q.base.device.cast::<borgvk_device>() };
    for &cb in unsafe { crate::entrypoint::slice(submit.command_buffers, submit.command_buffer_count) } {
        unsafe {
            vk_cmd_queue_execute(&mut (*cb).cmd_queue, cb.cast(), &device.cmd_dispatch);
            borgvk_flush_draws(cb.cast()); // draws outside a pass end
        }
    }
    if std::env::var_os("BORGVK_SIM").is_some() || crate::hw::enabled() {
        if unsafe { borgvk_submit_is_cube(submit) } {
            return unsafe { borgvk_submit_sim_cube(device, submit) };
        }
        return VK_SUCCESS;
    }
    unsafe { borgvk_submit_cube_live(device, submit) }
}

/// # Safety
/// The runtime's queue and submit.
#[no_mangle]
pub unsafe extern "C" fn borgvk_queue_submit(q: *mut vk_queue, submit: *mut vk_queue_submit) -> VkResult {
    let (q, s) = unsafe { (&mut *q, &mut *submit) };
    let dev = q.base.device;
    for w in unsafe { crate::entrypoint::slice(s.waits, s.wait_count) } {
        let r = unsafe { vk_sync_wait(dev, w.sync, w.wait_value, VK_SYNC_WAIT_COMPLETE, u64::MAX) };
        if r != VK_SUCCESS {
            return r;
        }
    }
    let mut result = unsafe { work(q, s) };
    for sig in unsafe { crate::entrypoint::slice(s.signals, s.signal_count) } {
        let r = unsafe { vk_sync_signal(dev, sig.sync, sig.signal_value) };
        if r != VK_SUCCESS && result == VK_SUCCESS {
            result = r;
        }
    }
    result
}

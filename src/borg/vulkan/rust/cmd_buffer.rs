// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Command buffers: the runtime's, with the recorded stream and draw state of rust/serial.rs and
//! the C recorders; push constants.

use crate::entrypoint::{entrypoint, handle, Handle};
use crate::ffi::*;
use std::ffi::c_void;
use std::mem::zeroed;
use std::ptr::null_mut;

handle!(VkCommandBuffer, vk_command_buffer, VK_OBJECT_TYPE_COMMAND_BUFFER);

/// # Safety
/// `c` is the `vk` of a `borgvk_command_buffer`, which comes first.
pub unsafe fn cmd<'a>(c: *mut vk_command_buffer) -> &'a mut borgvk_command_buffer {
    unsafe { &mut *c.cast() }
}

unsafe extern "C" fn create(pool: *mut vk_command_pool, level: VkCommandBufferLevel, out: *mut *mut vk_command_buffer) -> VkResult {
    let alloc = unsafe { &(*pool).alloc };
    let c = unsafe { vk_zalloc(alloc, size_of::<borgvk_command_buffer>(), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT) }.cast::<borgvk_command_buffer>();
    if c.is_null() {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    let mut params = vk_command_buffer_init_params { pool, ops: &CMD_BUFFER_OPS, level, needs_cmd_queue: true, ..unsafe { zeroed() } };
    let r = unsafe { vk_command_buffer_init_with_params(&mut (*c).vk, &mut params) };
    if r != VK_SUCCESS {
        unsafe { vk_free(alloc, c.cast()) };
        return r;
    }
    unsafe { *out = &mut (*c).vk };
    VK_SUCCESS
}

unsafe extern "C" fn reset(vk: *mut vk_command_buffer, _flags: VkCommandBufferResetFlags) {
    let c = unsafe { cmd(vk) };
    unsafe {
        free(c.stream.cast());
        crate::render::free_targets(c);
        free(c.state_sent.cast());
    }
    (c.stream, c.state_sent, c.state_sent_len) = (null_mut(), null_mut(), 0);
    (c.stream_len, c.stream_cap, c.heap_top, c.batch_draws) = (0, 0, 0, 0);
    (c.batch_img, c.index_ptr, c.gfx_pipeline, c.desc_set, c.cs_pipeline) = (null_mut(), null_mut(), null_mut(), null_mut(), null_mut());
    (c.in_rendering, c.has_clear, c.generic_drawn, c.dispatched) = (false, false, false, false);
    unsafe { (c.vb, c.vb_avail, c.cs_sets, c.desc_sets) = (zeroed(), zeroed(), zeroed(), zeroed()) };
    (c.color_attachment_count, c.depth_view, c.stencil_view) = (0, null_mut(), null_mut());
    unsafe { vk_command_buffer_reset(vk) };
}

unsafe extern "C" fn destroy(vk: *mut vk_command_buffer) {
    let c = unsafe { cmd(vk) };
    unsafe {
        vk_command_buffer_finish(&mut c.vk);
        free(c.stream.cast());
        free(c.state_sent.cast());
        crate::render::free_targets(c);
        vk_free(&(*c.vk.pool).alloc, (c as *mut borgvk_command_buffer).cast());
    }
}

pub static CMD_BUFFER_OPS: vk_command_buffer_ops =
    vk_command_buffer_ops { create: Some(create), reset: Some(reset), destroy: Some(destroy), ..unsafe { zeroed() } };

entrypoint! {
    fn borgvk_BeginCommandBuffer(command_buffer: VkCommandBuffer, info: *const VkCommandBufferBeginInfo) -> VkResult {
        let c = command_buffer.ptr();
        vk_command_buffer_begin(c, info);
        vk_command_buffer_get_record_result(c)
    }
}

entrypoint! {
    fn borgvk_EndCommandBuffer(command_buffer: VkCommandBuffer) -> VkResult {
        vk_command_buffer_end(command_buffer.ptr())
    }
}

/// Copies `size` bytes to `offset` of `dst` when they fit.
fn put(dst: &mut [u32], offset: u32, size: u32, values: *const c_void) -> bool {
    let end = offset.wrapping_add(size) as usize;
    if end > size_of_val(dst) {
        return false;
    }
    unsafe { std::ptr::copy_nonoverlapping(values.cast::<u8>(), dst.as_mut_ptr().cast::<u8>().add(offset as usize), size as usize) };
    true
}

entrypoint! {
    fn borgvk_CmdPushConstants(command_buffer: VkCommandBuffer, _layout: VkPipelineLayout, _stages: VkShaderStageFlags, offset: u32, size: u32, values: *const c_void) {
        let c = cmd(command_buffer.ptr());
        put(&mut c.pc, offset, size, values);
        let dev = &mut *c.vk.base.device.cast::<borgvk_device>();
        if size == 0 || !put(&mut dev.push_words, offset, size, values) {
            return;
        }
        let (lo, hi) = (offset / 4, (offset + size + 3) / 4);
        if dev.push_lo >= dev.push_hi {
            (dev.push_lo, dev.push_hi) = (lo, hi);
        } else {
            (dev.push_lo, dev.push_hi) = (dev.push_lo.min(lo), dev.push_hi.max(hi));
        }
    }
}

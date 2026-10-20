// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Command buffers: the runtime's, with the recorded stream and draw state of rust/serial.rs and
//! the C recorders; push constants.

use crate::entrypoint::{entrypoint, handle, Handle};
use crate::descriptor_set::DescriptorSet;
use crate::ffi::*;
use crate::object::{allocate, Zeroed};
use crate::render::Target;
use std::ffi::c_void;
use std::mem::zeroed;
use std::ptr::null_mut;

handle!(VkCommandBuffer, vk_command_buffer, VK_OBJECT_TYPE_COMMAND_BUFFER);

#[repr(C)]
pub struct CommandBuffer {
    pub vk: vk_command_buffer,

    pub in_rendering: bool,
    pub render_area: VkRect2D,
    pub layer_count: u32,
    pub targets: *mut Target,
    pub color_attachment_count: u32,
    pub color_views: [*mut vk_image_view; BORGVK_MAX_COLOR_ATTACHMENTS as usize],
    pub depth_view: *mut vk_image_view,
    pub stencil_view: *mut vk_image_view,
    pub color_resolve_views: [*mut vk_image_view; BORGVK_MAX_COLOR_ATTACHMENTS as usize],
    pub has_clear: bool,
    pub clear_color: [f32; 4],
    // Draw state, tracked as the recorded commands replay at submit.
    pub gfx_pipeline: *mut borgvk_pipeline,
    pub vb: [*const u8; BORGVK_MAX_VERTEX_BINDINGS as usize],
    pub vb_avail: [VkDeviceSize; BORGVK_MAX_VERTEX_BINDINGS as usize], // bytes from vb[] to the buffer's end
    pub desc_set: *mut DescriptorSet,
    pub generic_drawn: bool,
    pub cs_pipeline: *const borgvk_pipeline,
    pub dispatched: bool, // ran a compute dispatch at replay
    // Draws of the open render pass, as one wire stream the simulator runs at its end.
    pub stream: *mut u8,
    pub stream_len: usize,
    pub stream_cap: usize,
    pub heap_top: u32,
    pub batch_draws: u32,
    pub state_sent: *mut u8,
    pub state_sent_len: usize,
    pub batch_img: *mut borgvk_image,
    pub batch_extra: [*mut borgvk_image; 3], // colour attachments 1-3 of the batch
    pub batch_nextra: u32,
    pub batch_serve: bool,
    pub batch_area: VkExtent2D, // the pass size when it has no attachment
    pub batch_depth: *mut vk_image_view,
    pub batch_stencil: *mut vk_image_view,
    pub wb: [borgvk_writeback; BORGVK_MAX_WRITEBACKS as usize],
    pub nwb: u32,
    pub batch_srgb: bool,
    pub batch_pool: *const c_void,
    pub batch_vpool: *const c_void,
    pub region_set: *mut DescriptorSet, // the set whose buffer region is at region_base
    pub region_base: u32,
    pub index_ptr: *const u8, // bound index buffer, at its offset
    pub index_avail: VkDeviceSize,
    pub index_size: u32,         // 2, 4, or 0 for an unsupported type
    pub batch_push: [u32; 32],   // the push words the open batch was drawn with
    pub pc: [u32; 32],           // push-constant bytes, for compute
    pub dyn_off: [VkDeviceSize; BORGVK_MAX_BINDINGS as usize], // dynamic offsets of desc_set
    pub cs_sets: [*mut DescriptorSet; 4], // the compute bind point's own sets
    pub cs_dyn_offs: [[VkDeviceSize; BORGVK_MAX_BINDINGS as usize]; 4],
    pub desc_sets: [*mut DescriptorSet; 4], // [0] is desc_set; dyn_offs[0] is unused (dyn_off)
    pub dyn_offs: [[VkDeviceSize; BORGVK_MAX_BINDINGS as usize]; 4],
}

// Raw pointers, numbers and the runtime's command buffer, which it zero-allocates itself.
unsafe impl Zeroed for CommandBuffer {}

/// # Safety
/// `c` is the `vk` of a `CommandBuffer`, which comes first.
pub unsafe fn cmd<'a>(c: *mut vk_command_buffer) -> &'a mut CommandBuffer {
    unsafe { &mut *c.cast() }
}

unsafe extern "C" fn create(pool: *mut vk_command_pool, level: VkCommandBufferLevel, out: *mut *mut vk_command_buffer) -> VkResult {
    let alloc = unsafe { &(*pool).alloc };
    let Some(c) = (unsafe { allocate::<CommandBuffer>(|n| vk_zalloc(alloc, n, 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT)) }) else {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    };
    let mut params = vk_command_buffer_init_params { pool, ops: &CMD_BUFFER_OPS, level, needs_cmd_queue: true, ..unsafe { zeroed() } };
    let r = unsafe { vk_command_buffer_init_with_params(&mut c.vk, &mut params) };
    if r != VK_SUCCESS {
        unsafe { vk_free(alloc, std::ptr::from_mut(c).cast()) };
        return r;
    }
    unsafe { *out = &mut c.vk };
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
        vk_free(&(*c.vk.pool).alloc, (c as *mut CommandBuffer).cast());
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

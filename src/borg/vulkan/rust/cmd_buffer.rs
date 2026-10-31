// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Command buffers: the runtime's, with the recorded stream and draw state of rust/wire.rs and
//! the C recorders; push constants.

use crate::entrypoint::{entrypoint, handle, pointee};
use crate::descriptor_set::DescriptorSet;
use crate::ffi::*;
use crate::image::Image;
use crate::device::Device;
use crate::pipeline::Pipeline;
use crate::query::{OccRun, QueryPool};
use crate::object::{allocate, Owned, VkBuf, Zeroed};
use crate::render::Target;
use std::ffi::c_void;
use std::mem::zeroed;
use std::ptr::{null, null_mut};

handle!(VkCommandBuffer, vk_command_buffer, VK_OBJECT_TYPE_COMMAND_BUFFER);

/// What the draws render to, as images: set at the start of rendering.
#[derive(Clone, Copy)]
pub struct Rendering {
    pub color: [*mut Image; 4],
    pub depth: *mut Image,
    pub stencil: *mut Image,
    pub offset: [i32; 2],
    pub extent: [u32; 2],
}

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
    pub rendering: Rendering,
    pub has_clear: bool,
    pub clear_color: [f32; 4],
    // Draw state, tracked as the recorded commands replay at submit.
    pub gfx_pipeline: *mut Pipeline,
    pub vb: [*const u8; BORGVK_MAX_VERTEX_BINDINGS as usize],
    pub vb_avail: [VkDeviceSize; BORGVK_MAX_VERTEX_BINDINGS as usize], // bytes from vb[] to the buffer's end
    pub desc_set: *mut DescriptorSet,
    pub generic_drawn: bool,
    pub cs_pipeline: *const Pipeline,
    pub dispatched: bool, // ran a compute dispatch at replay
    // Draws of the open render pass, as one wire stream the simulator runs at its end.
    pub stream: VkBuf,
    pub heap_top: u32,
    pub batch_draws: u32,
    pub batch_tris: u32, // at most, over the batch's draws
    pub state_sent: VkBuf,
    pub batch_img: *mut Image,
    pub batch_extra: [*mut Image; 3], // colour attachments 1-3 of the batch
    pub batch_nextra: u32,
    pub batch_serve: bool,
    pub batch_area: [u32; 2], // the pass size when it has no attachment
    pub batch_depth: *mut Image,
    pub batch_stencil: *mut Image,
    pub wb: [Writeback; BORGVK_MAX_WRITEBACKS as usize],
    pub nwb: u32,
    pub occ_pool: *const QueryPool, // the active occlusion query
    pub occ_query: u32,
    pub occ_open: bool, // counted in the open batch, its run is the last
    pub occ: [OccRun; 16],
    pub nocc: u32,
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

impl CommandBuffer {
    /// The device the runtime created it on, which outlives it.
    pub fn device(&self) -> &Device {
        unsafe { &*self.vk.base.device.cast::<Device>() }
    }

    /// The device next to the command buffer, for as long as it is borrowed.
    pub fn split_device(&mut self) -> (&Device, &mut Self) {
        (unsafe { &*self.vk.base.device.cast::<Device>() }, self)
    }

    // What is bound stays alive while the command buffer records and runs.
    pub fn split_gfx_pipeline(&mut self) -> (Option<&Pipeline>, &mut Self) {
        (unsafe { self.gfx_pipeline.as_ref() }, self)
    }

    pub fn cs_pipeline(&self) -> Option<&Pipeline> {
        unsafe { self.cs_pipeline.as_ref() }
    }

    pub fn dyn_set(&self, s: mesa_vk_dynamic_graphics_state) -> bool {
        self.vk.dynamic_graphics_state.set[s as usize / 32] >> (s % 32) & 1 != 0
    }

    /// Viewport 0 as x, y, width, height, min and max depth, if set as dynamic state.
    pub fn dyn_viewport(&self) -> Option<[f32; 6]> {
        let v = &self.vk.dynamic_graphics_state.vp.viewports[0];
        self.dyn_set(MESA_VK_DYNAMIC_VP_VIEWPORTS).then(|| [v.x, v.y, v.width, v.height, v.minDepth, v.maxDepth])
    }

    /// Scissor 0 as x0, y0, x1, y1, clamped at 0, if set as dynamic state.
    pub fn dyn_scissor(&self) -> Option<[u32; 4]> {
        let r = &self.vk.dynamic_graphics_state.vp.scissors[0];
        let (x1, y1) = (r.offset.x.wrapping_add(r.extent.width as i32), r.offset.y.wrapping_add(r.extent.height as i32));
        self.dyn_set(MESA_VK_DYNAMIC_VP_SCISSORS).then(|| [r.offset.x.max(0) as u32, r.offset.y.max(0) as u32, x1.max(0) as u32, y1.max(0) as u32])
    }

    pub fn desc_set(&self) -> Option<&DescriptorSet> {
        unsafe { self.desc_set.as_ref() }
    }

    pub fn cs_set(&self, i: usize) -> Option<&DescriptorSet> {
        unsafe { self.cs_sets[i].as_ref() }
    }

    pub fn gfx_set(&self, i: usize) -> Option<&DescriptorSet> {
        unsafe { if i == 0 { self.desc_set } else { self.desc_sets[i] }.as_ref() }
    }

    /// The graphics sets next to the command buffer, for as long as it is borrowed.
    pub fn split_sets(&mut self) -> ([Option<&DescriptorSet>; 4], &mut Self) {
        let sets = [self.desc_set, self.desc_sets[1], self.desc_sets[2], self.desc_sets[3]];
        (sets.map(|s| unsafe { s.as_ref() }), self)
    }

    // The open pass's attachments; a view of a target lives as long as the command buffer.
    pub fn color_view(&self, i: usize) -> Option<&vk_image_view> {
        unsafe { self.color_views[i].as_ref() }
    }

    pub fn resolve_view(&self, i: usize) -> Option<&vk_image_view> {
        unsafe { self.color_resolve_views[i].as_ref() }
    }

    pub fn depth_view(&self) -> Option<&vk_image_view> {
        unsafe { self.depth_view.as_ref() }
    }

    pub fn stencil_view(&self) -> Option<&vk_image_view> {
        unsafe { self.stencil_view.as_ref() }
    }
}

impl crate::entrypoint::Pointee for CommandBuffer {
    type Raw = VkCommandBuffer;
    unsafe fn ptr(raw: VkCommandBuffer) -> *mut Self {
        raw.cast()
    }
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
    let mut c = Owned::new(c, |c| unsafe { vk_free(alloc, std::ptr::from_mut(c).cast()) });
    unsafe { (c.stream, c.state_sent) = (VkBuf::new(alloc), VkBuf::new(alloc)) };
    let mut params = vk_command_buffer_init_params { pool, ops: &CMD_BUFFER_OPS, level, needs_cmd_queue: true, ..unsafe { zeroed() } };
    let r = unsafe { vk_command_buffer_init_with_params(&mut c.vk, &mut params) };
    if r != VK_SUCCESS {
        return r;
    }
    unsafe { *out = &mut c.keep().vk };
    VK_SUCCESS
}

unsafe extern "C" fn reset(vk: *mut vk_command_buffer, _flags: VkCommandBufferResetFlags) {
    let c = unsafe { cmd(vk) };
    drop((c.stream.take(), c.state_sent.take()));
    crate::render::free_targets(c);
    (c.heap_top, c.batch_draws, c.batch_tris) = (0, 0, 0);
    (c.occ_pool, c.occ_open, c.nocc) = (null(), false, 0);
    (c.batch_img, c.index_ptr, c.gfx_pipeline, c.desc_set, c.cs_pipeline) = (null_mut(), null_mut(), null_mut(), null_mut(), null_mut());
    (c.in_rendering, c.has_clear, c.generic_drawn, c.dispatched) = (false, false, false, false);
    unsafe { (c.vb, c.vb_avail, c.cs_sets, c.desc_sets) = (zeroed(), zeroed(), zeroed(), zeroed()) };
    (c.color_attachment_count, c.depth_view, c.stencil_view) = (0, null_mut(), null_mut());
    (c.rendering.depth, c.rendering.stencil) = (null_mut(), null_mut());
    unsafe { vk_command_buffer_reset(vk) };
}

unsafe extern "C" fn destroy(vk: *mut vk_command_buffer) {
    let c = unsafe { cmd(vk) };
    unsafe {
        vk_command_buffer_finish(&mut c.vk);
        drop((c.stream.take(), c.state_sent.take()));
        crate::render::free_targets(c);
        vk_free(&(*c.vk.pool).alloc, (c as *mut CommandBuffer).cast());
    }
}

pointee!(VkCommandBufferBeginInfo);

pub static CMD_BUFFER_OPS: vk_command_buffer_ops =
    vk_command_buffer_ops { create: Some(create), reset: Some(reset), destroy: Some(destroy), ..unsafe { zeroed() } };

entrypoint! {
    fn borgvk_BeginCommandBuffer(c: &mut CommandBuffer, info: &VkCommandBufferBeginInfo) -> VkResult {
        unsafe {
            vk_command_buffer_begin(&mut c.vk, info);
            vk_command_buffer_get_record_result(&mut c.vk)
        }
    }
}

entrypoint! {
    fn borgvk_EndCommandBuffer(c: &mut CommandBuffer) -> VkResult {
        unsafe { vk_command_buffer_end(&mut c.vk) }
    }
}

/// Copies `values` to byte `offset` of `dst` when they fit.
fn put(dst: &mut [u32], offset: u32, values: &[u8]) -> bool {
    let at = offset as usize;
    let Some(d) = (at.checked_add(values.len()).filter(|&e| e <= size_of_val(dst))).map(|e| at..e) else { return false };
    for (i, &b) in d.zip(values) {
        let w = &mut dst[i / 4];
        *w = *w & !(0xff << (i % 4 * 8)) | (b as u32) << (i % 4 * 8);
    }
    true
}

entrypoint! {
    fn borgvk_CmdPushConstants(c: &mut CommandBuffer, _layout: VkPipelineLayout, _stages: VkShaderStageFlags, offset: u32, size: u32, values: *const c_void) {
        let values = if size == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(values.cast::<u8>(), size as usize) } };
        put(&mut c.pc, offset, values);
        let mut dev = c.device().scratch.lock().unwrap();
        if size == 0 || !put(&mut dev.push_words, offset, values) {
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

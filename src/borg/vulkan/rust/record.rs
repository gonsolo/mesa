// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Render passes built while recording: the graphics commands of a primary command buffer run
//! when they are recorded as well as recorded for replay. A pass whose draws read or write host
//! memory meanwhile, or that draws indirect or runs secondaries, builds at submit instead.
#![allow(non_snake_case)]

use crate::cmd_buffer::{cmd, CommandBuffer};
use crate::device::Device;
use crate::ffi::*;
use std::ffi::c_void;

extern "C" {
    static vk_cmd_enqueue_device_entrypoints: vk_device_entrypoint_table;
}

fn primary(c: &CommandBuffer) -> bool {
    c.recording() && c.vk.level == VK_COMMAND_BUFFER_LEVEL_PRIMARY
}

/// Records the command, then runs it now on a primary command buffer when `now` holds.
macro_rules! both {
    ($($name:ident($($a:ident: $t:ty),*) $now:expr;)*) => {$(both!(@ $name($($a: $t),*) $now, |d: &Device| d.cmd_dispatch.$name.unwrap());)*};
    (@ $name:ident($($a:ident: $t:ty),*) $now:expr, $run:expr) => {
        unsafe extern "C" fn $name(cb: VkCommandBuffer, $($a: $t),*) {
            unsafe { (vk_cmd_enqueue_device_entrypoints.$name.unwrap())(cb, $($a),*) };
            let c = unsafe { cmd(cb.cast()) };
            let now: fn(&mut CommandBuffer) -> bool = $now;
            if !primary(c) || !now(c) {
                return;
            }
            let run = ($run)(unsafe { &*c.vk.base.device.cast::<Device>() });
            crate::memory::touched();
            unsafe { run(cb, $($a),*) };
            if crate::memory::touched() {
                c.rec().pure = false;
            }
        }
    };
}

/// Records the command; inside a pass being built, the pass builds at submit.
macro_rules! impure {
    ($($name:ident($($a:ident: $t:ty),*);)*) => {$(
        unsafe extern "C" fn $name(cb: VkCommandBuffer, $($a: $t),*) {
            unsafe { (vk_cmd_enqueue_device_entrypoints.$name.unwrap())(cb, $($a),*) };
            let c = unsafe { cmd(cb.cast()) };
            if primary(c) {
                c.rec().pure = false;
            }
        }
    )*};
}

fn always(_: &mut CommandBuffer) -> bool {
    true
}

fn in_pass(c: &mut CommandBuffer) -> bool {
    c.rec().active
}

fn begin(c: &mut CommandBuffer) -> bool {
    // The pass is known by its recorded VkRenderingInfo, which replay hands back.
    let last = c.vk.cmd_queue.cmds.prev.cast::<vk_cmd_queue_entry>();
    let key = unsafe { last.as_ref() }.filter(|e| e.type_ == VK_CMD_BEGIN_RENDERING).map(|e| unsafe { e.u.begin_rendering.rendering_info });
    let r = c.rec();
    (r.key, r.active, r.pure) = (key.map_or(0, |k| k as usize), key.is_some(), true);
    key.is_some()
}

both! {
    CmdBindPipeline(bp: VkPipelineBindPoint, p: VkPipeline) always;
    CmdBindVertexBuffers(first: u32, n: u32, bufs: *const VkBuffer, offs: *const VkDeviceSize) always;
    CmdBindIndexBuffer(b: VkBuffer, off: VkDeviceSize, ty: VkIndexType) always;
    CmdBindDescriptorSets(bp: VkPipelineBindPoint, l: VkPipelineLayout, first: u32, n: u32, sets: *const VkDescriptorSet, nd: u32, dyn_offs: *const u32) always;
    CmdPushConstants(l: VkPipelineLayout, st: VkShaderStageFlags, off: u32, size: u32, v: *const c_void) always;
    CmdSetViewport(first: u32, n: u32, v: *const VkViewport) always;
    CmdSetScissor(first: u32, n: u32, r: *const VkRect2D) always;
    CmdSetLineWidth(w: f32) always;
    CmdSetDepthBias(k: f32, clamp: f32, slope: f32) always;
    CmdSetBlendConstants(v: *const f32) always;
    CmdSetDepthBounds(lo: f32, hi: f32) always;
    CmdSetStencilCompareMask(f: VkStencilFaceFlags, m: u32) always;
    CmdSetStencilWriteMask(f: VkStencilFaceFlags, m: u32) always;
    CmdSetStencilReference(f: VkStencilFaceFlags, r: u32) always;
    CmdDraw(n: u32, inst: u32, first: u32, first_inst: u32) in_pass;
    CmdDrawIndexed(n: u32, inst: u32, first: u32, voff: i32, first_inst: u32) in_pass;
    CmdClearAttachments(na: u32, a: *const VkClearAttachment, nr: u32, r: *const VkClearRect) in_pass;
    CmdBeginQuery(pool: VkQueryPool, q: u32, flags: VkQueryControlFlags) always;
    CmdEndQuery(pool: VkQueryPool, q: u32) always;
}

// In the replay table's union of aliases: borgvk's own entries.
extern "C" {
    fn borgvk_CmdBeginRendering(cb: VkCommandBuffer, ri: *const VkRenderingInfo);
    fn borgvk_CmdEndRendering(cb: VkCommandBuffer);
}
both!(@ CmdBeginRendering(ri: *const VkRenderingInfo) begin, |_: &Device| borgvk_CmdBeginRendering);
both!(@ CmdEndRendering() in_pass, |_: &Device| borgvk_CmdEndRendering);

impure! {
    CmdDrawIndirect(b: VkBuffer, off: VkDeviceSize, n: u32, stride: u32);
    CmdDrawIndexedIndirect(b: VkBuffer, off: VkDeviceSize, n: u32, stride: u32);
    CmdExecuteCommands(n: u32, cbs: *const VkCommandBuffer);
}

/// The record table's entries that also build passes.
pub fn install(t: &mut vk_device_entrypoint_table) {
    macro_rules! set {
        ($($n:ident),*) => { $(t.$n = Some($n);)* };
    }
    set!(
        CmdBindPipeline, CmdBindVertexBuffers, CmdBindIndexBuffer, CmdBindDescriptorSets, CmdPushConstants, CmdSetViewport,
        CmdSetScissor, CmdSetLineWidth, CmdSetDepthBias, CmdSetBlendConstants, CmdSetDepthBounds, CmdSetStencilCompareMask,
        CmdSetStencilWriteMask, CmdSetStencilReference, CmdBeginRendering, CmdEndRendering, CmdDraw, CmdDrawIndexed,
        CmdDrawIndirect, CmdDrawIndexedIndirect, CmdClearAttachments, CmdExecuteCommands, CmdBeginQuery, CmdEndQuery
    );
}

// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Render passes built while recording: the graphics commands of a primary command buffer run
//! when they are recorded as well as recorded for replay. A batch whose draws read or write host
//! memory meanwhile, draw indirect or run secondaries replays live, between the kept ones.
#![allow(non_snake_case)]

use crate::cmd_buffer::{cmd, CommandBuffer, Step};
use crate::device::Device;
use crate::ffi::*;
use std::ffi::c_void;

extern "C" {
    static vk_cmd_enqueue_device_entrypoints: vk_device_entrypoint_table;
    fn vk_common_CmdExecuteCommands(cb: VkCommandBuffer, n: u32, cbs: *const VkCommandBuffer);
}

fn primary(c: &CommandBuffer) -> bool {
    c.recording() && c.vk.level == VK_COMMAND_BUFFER_LEVEL_PRIMARY
}

/// The open batch replays live: its commands so far and the ones after it.
fn go_live(c: &mut CommandBuffer, upto: u32) {
    let r = c.rec();
    let (a, b) = (r.first as usize, (upto as usize).min(r.lives.len()));
    for l in r.lives.get_mut(a..b.max(a)).into_iter().flatten() {
        *l = true;
    }
    r.live = true;
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
            if crate::memory::touched() && c.rec().active {
                let n = c.rec().seq;
                go_live(c, n);
            }
        }
    };
}

/// Records a draw and builds it now; one that reads host memory replays live, with every batch
/// it is part of.
macro_rules! draw {
    ($($name:ident($($a:ident: $t:ty),*);)*) => {$(
        unsafe extern "C" fn $name(cb: VkCommandBuffer, $($a: $t),*) {
            unsafe { (vk_cmd_enqueue_device_entrypoints.$name.unwrap())(cb, $($a),*) };
            let c = unsafe { cmd(cb.cast()) };
            if !primary(c) || !c.rec().active {
                return;
            }
            let s = next(c);
            let r = c.rec();
            let (was_live, k0) = (r.live, r.steps.len());
            let run = unsafe { (*c.vk.base.device.cast::<Device>()).cmd_dispatch.$name.unwrap() };
            crate::memory::touched();
            unsafe { run(cb, $($a),*) };
            if crate::memory::touched() || was_live {
                // The batches taken meanwhile hold the draw's first part or the ones before it.
                let r = c.rec();
                let mut i = k0;
                while i < r.steps.len() {
                    if matches!(r.steps[i].step, Step::Run(_)) {
                        r.first = r.first.min(r.steps.remove(i).first);
                    } else {
                        i += 1;
                    }
                }
                go_live(c, s + 1);
            }
        }
    )*};
}

/// The next command of the open pass.
fn next(c: &mut CommandBuffer) -> u32 {
    let r = c.rec();
    let s = r.seq;
    (r.seq, r.cur) = (s + 1, s);
    r.lives.push(false);
    s
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
    (r.key, r.active, r.seq, r.cur, r.first, r.live) = (key.map_or(0, |k| k as usize), key.is_some(), 0, 0, 0, false);
    (r.steps.clear(), r.lives.clear());
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
    CmdBeginQuery(pool: VkQueryPool, q: u32, flags: VkQueryControlFlags) always;
    CmdEndQuery(pool: VkQueryPool, q: u32) always;
}

draw! {
    CmdDraw(n: u32, inst: u32, first: u32, first_inst: u32);
    CmdDrawIndexed(n: u32, inst: u32, first: u32, voff: i32, first_inst: u32);
    CmdDrawIndirect(b: VkBuffer, off: VkDeviceSize, n: u32, stride: u32);
    CmdDrawIndexedIndirect(b: VkBuffer, off: VkDeviceSize, n: u32, stride: u32);
}

// In the replay table's union of aliases: borgvk's own entries.
extern "C" {
    fn borgvk_CmdBeginRendering(cb: VkCommandBuffer, ri: *const VkRenderingInfo);
    fn borgvk_CmdEndRendering(cb: VkCommandBuffer);
}
both!(@ CmdBeginRendering(ri: *const VkRenderingInfo) begin, |_: &Device| borgvk_CmdBeginRendering);
both!(@ CmdEndRendering() in_pass, |_: &Device| borgvk_CmdEndRendering);

/// Ends the batch before the clear: the clear is a step of its own, never live.
unsafe extern "C" fn CmdClearAttachments(cb: VkCommandBuffer, na: u32, a: *const VkClearAttachment, nr: u32, r: *const VkClearRect) {
    unsafe { (vk_cmd_enqueue_device_entrypoints.CmdClearAttachments.unwrap())(cb, na, a, nr, r) };
    let c = unsafe { cmd(cb.cast()) };
    if !primary(c) || !c.rec().active {
        return;
    }
    let s = next(c);
    let run = unsafe { (*c.vk.base.device.cast::<Device>()).cmd_dispatch.CmdClearAttachments.unwrap() };
    unsafe { run(cb, na, a, nr, r) };
    c.rec().first = s + 1;
}

/// Secondaries replay live, after the batch before them.
unsafe extern "C" fn CmdExecuteCommands(cb: VkCommandBuffer, n: u32, cbs: *const VkCommandBuffer) {
    unsafe { (vk_cmd_enqueue_device_entrypoints.CmdExecuteCommands.unwrap())(cb, n, cbs) };
    let c = unsafe { cmd(cb.cast()) };
    if !primary(c) || !c.rec().active {
        return;
    }
    let s = next(c);
    crate::draw::flush_draws(c);
    let r = c.rec();
    (r.lives[s as usize], r.first) = (true, s + 1);
}

/// Replay's CmdExecuteCommands: the secondaries' commands are part of this one.
pub unsafe extern "C" fn replay_execute_commands(cb: VkCommandBuffer, n: u32, cbs: *const VkCommandBuffer) {
    let c = unsafe { cmd(cb.cast()) };
    if !gate(c) {
        return;
    }
    let nested = std::mem::replace(&mut c.rec().nested, true);
    unsafe { vk_common_CmdExecuteCommands(cb, n, cbs) };
    unsafe { cmd(cb.cast()) }.rec().nested = nested;
}

/// Replay of a draw or clear: in a pass built while recording, the steps before it run first;
/// it runs itself when it replays live.
pub fn gate(c: &mut CommandBuffer) -> bool {
    if !c.skipping() || c.rec().nested {
        return true;
    }
    let s = c.rec().rseq;
    c.rec().rseq = s + 1;
    steps_until(c, s);
    let r = c.rec();
    r.passes[r.skip.unwrap()].lives.get(s as usize).copied().unwrap_or(true)
}

/// Runs the pass's steps placed before command `s`, after the live draws so far.
pub fn steps_until(c: &mut CommandBuffer, s: u32) {
    let r = c.rec();
    let p = &r.passes[r.skip.unwrap()];
    if p.steps.get(r.cursor).is_none_or(|st| st.at > s) {
        return;
    }
    crate::draw::flush_draws(c);
    let r = c.rec();
    let p = &r.passes[r.skip.unwrap()];
    while let Some(st) = p.steps.get(r.cursor).filter(|st| st.at <= s) {
        st.step.run();
        r.cursor += 1;
    }
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

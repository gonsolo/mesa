// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Draws: the bound pipeline, vertex, index and descriptor state, and each draw as packets
//! into the open pass's stream, which the simulator runs at the pass's end.

use crate::cmd_buffer::cmd;
use crate::copy::host;
use crate::entrypoint::{entrypoint, slice, Handle};
use crate::ffi::*;
use std::ptr::null;

entrypoint! {
    fn borgvk_CmdBindPipeline(command_buffer: VkCommandBuffer, bind_point: VkPipelineBindPoint, pipeline: VkPipeline) {
        let c = cmd(command_buffer.ptr());
        match bind_point {
            VK_PIPELINE_BIND_POINT_GRAPHICS => c.gfx_pipeline = pipeline.ptr(),
            VK_PIPELINE_BIND_POINT_COMPUTE => c.cs_pipeline = pipeline.ptr(),
            _ => {}
        }
    }
}

entrypoint! {
    fn borgvk_CmdBindVertexBuffers(command_buffer: VkCommandBuffer, first: u32, count: u32, buffers: *const VkBuffer, offsets: *const VkDeviceSize) {
        let c = cmd(command_buffer.ptr());
        for (i, (b, &off)) in slice(buffers, count).iter().zip(slice(offsets, count)).enumerate() {
            let slot = first as usize + i;
            if slot >= BORGVK_MAX_VERTEX_BINDINGS as usize {
                continue;
            }
            let buf = b.get().map(|b| &*b);
            c.vb[slot] = buf.and_then(host).map_or(null(), |p| p.add(off as usize));
            c.vb_avail[slot] = buf.filter(|b| b.vk.size > off).map_or(0, |b| b.vk.size - off);
        }
    }
}

entrypoint! {
    fn borgvk_CmdBindDescriptorSets(command_buffer: VkCommandBuffer, bind_point: VkPipelineBindPoint, _layout: VkPipelineLayout, first_set: u32, count: u32, sets: *const VkDescriptorSet, dyn_count: u32, dyn_offsets: *const u32) {
        let c = cmd(command_buffer.ptr());
        let cs = bind_point == VK_PIPELINE_BIND_POINT_COMPUTE;
        let dyn_offsets = slice(dyn_offsets, dyn_count);
        let mut k = 0;
        for (i, &h) in slice(sets, count).iter().enumerate() {
            let si = first_set as usize + i;
            let Some(set) = h.get() else { continue };
            if si > 3 {
                continue;
            }
            let dyn_off = if cs {
                c.cs_sets[si] = h.ptr();
                &mut c.cs_dyn_offs[si]
            } else if si != 0 {
                c.desc_sets[si] = h.ptr();
                &mut c.dyn_offs[si]
            } else {
                c.desc_set = h.ptr();
                &mut c.dyn_off
            };
            *dyn_off = [0; BORGVK_MAX_BINDINGS as usize];
            for (b, d) in dyn_off.iter_mut().enumerate() {
                if k == dyn_offsets.len() {
                    break;
                }
                if set.dyn_mask >> b & 1 != 0 {
                    *d = dyn_offsets[k] as u64;
                    k += 1;
                }
            }
        }
    }
}

entrypoint! {
    fn borgvk_CmdBindIndexBuffer(command_buffer: VkCommandBuffer, buffer: VkBuffer, offset: VkDeviceSize, index_type: VkIndexType) {
        let c = cmd(command_buffer.ptr());
        (c.index_ptr, c.index_avail) = (null(), 0);
        c.index_size = match index_type {
            VK_INDEX_TYPE_UINT16 => 2,
            VK_INDEX_TYPE_UINT32 => 4,
            _ => 0,
        };
        let Some(b) = buffer.get() else { return };
        if let Some(p) = host(b).filter(|_| b.vk.size > offset) {
            (c.index_ptr, c.index_avail) = (p.add(offset as usize), b.vk.size - offset);
        }
    }
}

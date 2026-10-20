// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Draws: the bound pipeline, vertex, index and descriptor state, and each draw as packets
//! into the open pass's stream, which the simulator runs at the pass's end.

use crate::cmd_buffer::{cmd, CommandBuffer};
use crate::copy::{host, image_host};
use crate::corpus::{borgvk_corpus_hash, borgvk_corpus_image, borgvk_corpus_log, borgvk_corpus_on};
use crate::entrypoint::{entrypoint, slice, Handle};
use crate::descriptor_set::DescriptorSet;
use crate::ffi::*;
use crate::image::Image;
use crate::device::Device;
use crate::pipeline::Pipeline;
use crate::format::{depth_is_d32, flush_format, raw32, vertex_fetch, Fetch};
use crate::hw::{enabled, serves_extent};
use crate::image::blocks;
use crate::serial::{send_att, send_draw, send_mem, send_pass, send_target, send_vattr, Draw};
use crate::sim::borgvk_sim_run_pass;
use crate::sim_process::borgvk_sim_run_stream;
use crate::state_upload::{send_blend_state_serial, upload_shaders_serial};
use crate::texture_upload::{borgvk_send_buffer_texture, borgvk_send_generic_texture};
use crate::transport::{borgvk_transport_capture_begin, borgvk_transport_capture_end};
use crate::cube::borgvk_set_is_cube;
use crate::util::log;
use std::ffi::c_void;
use std::ptr::{null, null_mut};

entrypoint! {
    unsafe fn borgvk_CmdBindPipeline(command_buffer: VkCommandBuffer, bind_point: VkPipelineBindPoint, pipeline: VkPipeline) {
        let c = cmd(command_buffer.ptr());
        match bind_point {
            VK_PIPELINE_BIND_POINT_GRAPHICS => c.gfx_pipeline = pipeline.ptr(),
            VK_PIPELINE_BIND_POINT_COMPUTE => c.cs_pipeline = pipeline.ptr(),
            _ => {}
        }
    }
}

entrypoint! {
    unsafe fn borgvk_CmdBindVertexBuffers(command_buffer: VkCommandBuffer, first: u32, count: u32, buffers: *const VkBuffer, offsets: *const VkDeviceSize) {
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
    unsafe fn borgvk_CmdBindDescriptorSets(command_buffer: VkCommandBuffer, bind_point: VkPipelineBindPoint, _layout: VkPipelineLayout, first_set: u32, count: u32, sets: *const VkDescriptorSet, dyn_count: u32, dyn_offsets: *const u32) {
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
    unsafe fn borgvk_CmdBindIndexBuffer(command_buffer: VkCommandBuffer, buffer: VkBuffer, offset: VkDeviceSize, index_type: VkIndexType) {
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

const HEAP_BYTES: u32 = 0x200_0000;
const POINT_SLOT: u32 = 16;
const PUSH_WINDOW_SLOT: u32 = 33; // borgc: PUSH_WINDOW_WORDS
const SPILL_SLOT: u32 = 34; // borgc: spill.rs SCRATCH_WORDS
const SLOT_BYTES: u32 = BORGVK_SSBO_SLOT_BYTES;
const MAX_ROWS: i64 = 4096 * 4096;
const VB: usize = BORGVK_MAX_VERTEX_BINDINGS as usize;
const BINDINGS: usize = BORGVK_MAX_BINDINGS as usize;
const LINE_END: [usize; 6] = [0, 0, 1, 0, 1, 1];
const LINE_SIDE: [f32; 6] = [-1.0, 1.0, -1.0, 1.0, 1.0, -1.0];
const CORNER: [[f32; 2]; 6] = [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];

#[derive(Clone, Copy, Debug)]
enum Reject {
    Undrawable = 3,
    NoShader = 4,
    NoTargetMemory = 5,
    NoIndexBuffer = 6,
    IndexRange = 7,
    VertexInput = 8,
    VertexRange = 9,
    Upload = 10,
    MrtFormats = 12,
    MrtExtent = 13,
    MrtNotServed = 14,
}

fn reject(why: Reject) -> bool {
    if std::env::var_os("BORGVK_DEBUG").is_some() {
        log(MESA_LOG_INFO, format!("borgvk: generic sim draw rejected (reason {})", why as u32));
    }
    false
}

fn view_image(v: *const vk_image_view) -> *mut Image {
    unsafe { v.as_ref() }.map_or(null_mut(), |v| v.image.cast())
}

fn bytes_of<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), size_of_val(v)) }
}

fn corpus_state(
    h: u64,
    img: *const Image,
    extra: &[*mut Image],
    depth: *const vk_image_view,
    stencil: *const vk_image_view,
    wb: &[Writeback],
) -> u64 {
    let mut h = unsafe { borgvk_corpus_image(h, img) };
    for &e in extra.iter().chain(&[view_image(depth), view_image(stencil)]) {
        if !e.is_null() {
            h = unsafe { borgvk_corpus_image(h, e) };
        }
    }
    wb.iter().fold(h, |h, w| unsafe { borgvk_corpus_hash(h, w.host.cast(), w.size as usize) })
}

/// Runs the draws batched so far.
pub fn flush_draws(c: &mut CommandBuffer) {
    if c.stream_len == 0 {
        return;
    }
    let (bytes, n, img, extra, serve, area) = (c.stream, c.stream_len, c.batch_img, c.batch_extra, c.batch_serve, c.batch_area);
    let (wb, nwb, nextra) = (c.wb, c.nwb as usize, c.batch_nextra as usize);
    c.nwb = 0;
    c.region_set = null_mut();
    (c.batch_pool, c.batch_vpool, c.batch_srgb) = (null(), null(), false);
    (c.stream, c.stream_len, c.stream_cap) = (null_mut(), 0, 0);
    (c.heap_top, c.batch_draws, c.batch_img, c.batch_nextra) = (0, 0, null_mut(), 0);
    unsafe { free(c.state_sent.cast()) };
    (c.state_sent, c.state_sent_len) = (null_mut(), 0);
    let rec = borgvk_corpus_on();
    let seed = BORGVK_CORPUS_SEED as u64;
    let state = |h| corpus_state(h, img, &extra[..nextra], c.batch_depth, c.batch_stencil, &wb[..nwb]);
    let key = if rec { state(unsafe { borgvk_corpus_hash(seed, bytes.cast(), n) }) } else { seed };
    if serve {
        unsafe {
            borgvk_sim_run_pass(bytes, n, img, extra.as_ptr(), nextra as u32, c.batch_depth, c.batch_stencil, wb.as_ptr(), nwb as u32, area);
            free(bytes.cast());
        }
    } else {
        unsafe { borgvk_sim_run_stream(bytes, n, img, 0) }; // takes the bytes
    }
    if rec {
        borgvk_corpus_log(b'F' as _, key, state(seed));
    }
}

fn stream_append(c: &mut CommandBuffer, data: &[u8]) -> bool {
    let n = data.len();
    if c.stream_len + n > c.stream_cap {
        let cap = (c.stream_cap * 2).max(c.stream_len + n + 65536);
        let p = unsafe { realloc(c.stream.cast(), cap as _) }.cast::<u8>();
        if p.is_null() {
            return false;
        }
        (c.stream, c.stream_cap) = (p, cap);
    }
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), c.stream.add(c.stream_len), n) };
    c.stream_len += n;
    true
}

struct Heap {
    top: u32,
}

impl Heap {
    fn upload(&mut self, data: &[u8]) -> Result<u32, Reject> {
        let n = data.len() as u32;
        let padded = (n + 3) & !3;
        if self.top as u64 + padded as u64 + 16 > HEAP_BYTES as u64 {
            return Err(Reject::Upload);
        }
        let off = self.top;
        for i in (0..padded).step_by(256) {
            let mut tmp = [0u8; 256];
            let c = 256.min(n.saturating_sub(i)) as usize;
            tmp[..c].copy_from_slice(&data[i as usize..i as usize + c]);
            send_mem(off + i, &tmp[..256.min(padded - i) as usize]);
        }
        self.top += padded + 16; // slack for the typed fetch's over-read
        Ok(off)
    }
}

#[derive(Clone, Copy, Default)]
struct Params {
    indexed: bool,
    index_size: u32,
    count: u32,
    instances: u32,
    first: u32,
    vertex_offset: i32,
    first_instance: u32,
}

fn fetch(f: VkFormat) -> Result<Fetch, Reject> {
    vertex_fetch(f).ok_or(Reject::Upload)
}

struct Ctx<'a> {
    device: &'a Device,
    c: &'a mut CommandBuffer,
    p: Params,
    pl: &'a Pipeline,
    topo: u32,
    x: u32,
    points: bool,
    lines: bool,
    color: *mut Image,
    extra: [*mut Image; 3],
    nextra: usize,
    w: u32,
    h: u32,
    idx: *const u8,
    vmin: i64,
    vmax: i64,
    imin: i64,
    imax: i64,
    first_of_batch: bool,
    state: *mut u8,
    state_len: usize,
    state_same: bool,
    load: u32,
    pairs: Vec<[u32; 2]>,
    heap: Heap,
    idx_off: u32,
    index_code: u32,
    voff_out: u32,
    ubo_base: u32,
}

impl Ctx<'_> {
    fn has_ds(&self) -> bool {
        !self.c.depth_view.is_null() || !self.c.stencil_view.is_null()
    }

    fn index(&self, i: u32) -> u32 {
        unsafe {
            if self.p.index_size == 2 {
                self.idx.cast::<u16>().add(i as usize).read_unaligned() as u32
            } else {
                self.idx.cast::<u32>().add(i as usize).read_unaligned()
            }
        }
    }

    fn validate(&mut self) -> Result<(), Reject> {
        let pl = self.pl;
        if self.p.count == 0 || self.p.instances == 0 {
            return Err(Reject::Undrawable);
        }
        self.topo = match pl.topology {
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP => 1,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN => 2,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
            | VK_PRIMITIVE_TOPOLOGY_POINT_LIST
            | VK_PRIMITIVE_TOPOLOGY_LINE_LIST
            | VK_PRIMITIVE_TOPOLOGY_LINE_STRIP => 0,
            _ => return Err(Reject::Undrawable),
        };
        if pl.blob[BORGVK_STAGE_VERT as usize].len == 0 || pl.blob[BORGVK_STAGE_FRAG as usize].len == 0 {
            return Err(Reject::NoShader);
        }
        self.points = pl.topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        self.x = if self.points { 6 } else { 1 };
        // A line is a quad too, but its six rows are gathered per line (see `pairs`).
        self.lines = matches!(pl.topology, VK_PRIMITIVE_TOPOLOGY_LINE_LIST | VK_PRIMITIVE_TOPOLOGY_LINE_STRIP);

        let c = &mut *self.c;
        self.color = view_image(c.color_views[0]);
        // Depth only: no colour attachment, the depth or stencil image gives the size.
        let ds_view = if c.depth_view.is_null() { c.stencil_view } else { c.depth_view };
        let size_img = if self.color.is_null() { view_image(ds_view) } else { self.color };
        if let Some(s) = unsafe { size_img.as_ref() } {
            if image_host(s).is_none() {
                return Err(Reject::NoTargetMemory);
            }
            (self.w, self.h) = (s.vk.extent.width, s.vk.extent.height);
            let ra = c.render_area;
            if ra.offset.x == 0 && ra.offset.y == 0 && ra.extent.width != 0 && ra.extent.width <= self.w && ra.extent.height <= self.h {
                (self.w, self.h) = (ra.extent.width, ra.extent.height);
            }
        } else {
            (self.w, self.h) = (c.render_area.extent.width, c.render_area.extent.height);
        }
        if self.color.is_null() && !serves_extent(self.w, self.h, true) {
            return Err(Reject::NoTargetMemory);
        }
        self.nextra = if pl.mrt { pl.color_count as usize - 1 } else { 0 };
        if pl.color_count >= 2 && !pl.mrt {
            return Err(Reject::MrtFormats);
        }
        for k in 0..self.nextra {
            let e = view_image(c.color_views[k + 1]);
            self.extra[k] = e;
            match unsafe { e.as_ref() } {
                Some(e) if image_host(e).is_some() && e.vk.extent.width >= self.w && e.vk.extent.height >= self.h => {}
                _ => return Err(Reject::MrtExtent),
            }
        }
        let has_ds = self.has_ds();
        let c = &mut *self.c;
        if self.nextra != 0 && !serves_extent(self.w, self.h, has_ds) {
            return Err(Reject::MrtNotServed);
        }
        let same_extra = c.batch_nextra as usize == self.nextra && c.batch_extra[..self.nextra] == self.extra[..self.nextra];
        if c.batch_draws != 0
            && (!same_extra
                || c.batch_img != self.color
                || c.batch_area.width != self.w
                || c.batch_area.height != self.h
                || c.batch_depth != c.depth_view
                || c.batch_stencil != c.stencil_view)
        {
            flush_draws(c);
        }
        Ok(())
    }

    fn index_range(&mut self) -> Result<(), Reject> {
        let (p, pl) = (self.p, self.pl);
        if p.indexed {
            if self.c.index_ptr.is_null() {
                return Err(Reject::NoIndexBuffer);
            }
            self.idx = self.c.index_ptr.wrapping_add(p.first as usize * p.index_size as usize);
            if p.first.wrapping_add(p.count) as u64 * p.index_size as u64 > self.c.index_avail {
                return Err(Reject::NoIndexBuffer);
            }
        }
        let restart = if p.index_size == 2 { 0xffff } else { 0xffff_ffff };
        let (mut vmin, mut vmax) = (i64::MAX, -1);
        if p.indexed {
            for i in 0..p.count {
                let v = self.index(i);
                if pl.restart && v == restart {
                    continue;
                }
                let r = v as i64 + p.vertex_offset as i64;
                if r < 0 {
                    return Err(Reject::IndexRange);
                }
                (vmin, vmax) = (vmin.min(r), vmax.max(r));
            }
        } else {
            (vmin, vmax) = (p.first as i64, p.first as i64 + p.count as i64 - 1);
        }
        if vmax < 0 {
            return Err(Reject::IndexRange);
        }
        (self.vmin, self.vmax) = (vmin, vmax);
        // Lines: the two vertex rows of each line.
        if self.lines {
            let strip = pl.topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
            let mut prev = -1i64;
            for i in 0..p.count {
                let mut r = p.first as i64 + i as i64;
                if p.indexed {
                    let v = self.index(i);
                    if pl.restart && v == restart {
                        prev = -1;
                        continue;
                    }
                    r = v as i64 + p.vertex_offset as i64;
                }
                if prev < 0 {
                    prev = r;
                    continue;
                }
                self.pairs.push([prev as u32, r as u32]);
                prev = if strip { r } else { -1 };
            }
            if self.pairs.is_empty() || self.pairs.len() as i64 * 6 > MAX_ROWS {
                return Err(Reject::IndexRange);
            }
        }
        Ok(())
    }

    fn reserve(&mut self) {
        let (p, pl, x) = (self.p, self.pl, self.x as u64);
        let mut need = if p.indexed { p.count as u64 * p.index_size as u64 + 32 } else { 0 };
        for b in 0..VB {
            if !pl.attrs[..pl.attr_count as usize].iter().any(|a| a.binding as usize == b) {
                continue;
            }
            let (r0, r1) = if pl.binding_instance[b] { (self.imin, self.imax) } else { (self.vmin, self.vmax) };
            need = need.wrapping_add(((r1 - r0 + 1) as u64).wrapping_mul(x).wrapping_mul(pl.binding_stride[b] as u64) + 32);
        }
        if self.points {
            need = need.wrapping_add(((self.vmax - self.vmin + 1) as u64).wrapping_mul(x * 16) + 32);
        }
        if self.lines {
            need = self.pairs.len() as u64 * 6 * (16 + 2 * 16 * pl.attr_count as u64) + 64;
        }
        let c = &mut *self.c;
        if c.batch_draws != 0 && c.heap_top as u64 + need > HEAP_BYTES as u64 {
            flush_draws(c);
        }
        if c.batch_draws != 0 && pl.push_used && c.batch_push != self.device.scratch.lock().unwrap().push_words {
            flush_draws(c);
        }
        // The constant pool's slot is one per pass.
        let me: *const c_void = std::ptr::from_ref(pl).cast();
        if c.batch_draws != 0 && pl.blob[BORGVK_STAGE_FRAG as usize].npool != 0 && !c.batch_pool.is_null() && c.batch_pool != me {
            flush_draws(c);
        }
        if c.batch_draws != 0 && pl.blob[BORGVK_STAGE_VERT as usize].npool != 0 && !c.batch_vpool.is_null() && c.batch_vpool != me {
            flush_draws(c);
        }
    }

    fn state(&mut self) {
        let (c, pl) = (&*self.c, self.pl);
        let mut d = self.device.scratch.lock().unwrap();
        borgvk_transport_capture_begin();
        d.shader_blob = pl.blob;
        (d.blend_cfg, d.blend_const, d.state_reg, d.state_valid, d.vp) = (pl.blend_cfg, pl.blend_const, pl.state_reg, pl.state_valid, pl.vp);
        let dyn_state = &c.vk.dynamic_graphics_state;
        let set = |s: mesa_vk_dynamic_graphics_state| dyn_state.set[s as usize / 32] >> (s % 32) & 1 != 0;
        if pl.vp_dyn && set(MESA_VK_DYNAMIC_VP_VIEWPORTS) {
            let v = &dyn_state.vp.viewports[0];
            d.vp.v = [v.x, v.y, v.width, v.height, v.minDepth, v.maxDepth];
            d.vp.valid = true;
        }
        if pl.sc_dyn && set(MESA_VK_DYNAMIC_VP_SCISSORS) {
            let r = &dyn_state.vp.scissors[0];
            let (x1, y1) = (r.offset.x.wrapping_add(r.extent.width as i32), r.offset.y.wrapping_add(r.extent.height as i32));
            d.vp.sc = [r.offset.x.max(0) as u32, r.offset.y.max(0) as u32, x1.max(0) as u32, y1.max(0) as u32];
            d.vp.valid = true;
        }
        upload_shaders_serial(&d);
        send_blend_state_serial(&d);
        self.state = unsafe { borgvk_transport_capture_end(&mut self.state_len) };
        self.state_same = !self.state.is_null()
            && !c.state_sent.is_null()
            && self.state_len == c.state_sent_len
            && unsafe { std::slice::from_raw_parts(self.state, self.state_len) == std::slice::from_raw_parts(c.state_sent, c.state_sent_len) };
    }

    fn begin_pass(&mut self) {
        let has_ds = self.has_ds();
        let (c, first) = (&mut *self.c, self.first_of_batch);
        borgvk_transport_capture_begin();
        if first {
            c.batch_serve = serves_extent(self.w, self.h, has_ds);
            c.batch_area = VkExtent2D { width: self.w, height: self.h };
        }
        let color = unsafe { self.color.as_ref() };
        self.load = if first { 0 } else { 1 };
        if c.batch_serve {
            let (has_z, has_s) = (!c.depth_view.is_null(), !c.stencil_view.is_null());
            self.load = 1 | (has_z as u32) << 1 | (has_s as u32) << 2 | 8;
            if first {
                let fmt = if self.nextra != 0 { 5 } else { color.map_or(0, |i| flush_format(i.vk.format)) };
                let d32 = has_z && depth_is_d32(unsafe { (*(*c.depth_view).image).format });
                send_pass(fmt, has_z as u8 | (has_s as u8) << 1 | if color.is_none() { 8 } else { 0 } | if d32 { 4 } else { 0 });
                send_att(self.nextra as u8 + 1, &[5, 5, 5]);
            }
        } else if first {
            let fmt = color.map_or(0, |i| match i.vk.format {
                VK_FORMAT_R8G8B8A8_UNORM => 1,
                VK_FORMAT_B8G8R8A8_UNORM => 2,
                f if raw32(f) => 3, // RAW32
                _ => 0,
            });
            send_target(fmt, &if c.has_clear { c.clear_color } else { [0.0; 4] });
        }
    }

    fn vertex_attrs(&mut self) -> Result<(), Reject> {
        let (pl, x) = (self.pl, self.x);
        let nrows_lines = self.pairs.len() as u32 * 6;
        for at in &pl.attrs[..pl.attr_count as usize] {
            let b = at.binding as usize;
            let usable = at.location < 16 && b < VB;
            let conv = usable && converts(at.format);
            let f = if !usable {
                None
            } else if conv {
                let pf = unsafe { vk_format_to_pipe_format(at.format) };
                let code = unsafe { if util_format_is_pure_sint(pf) { 40 } else if util_format_is_pure_uint(pf) { 39 } else { 41 } };
                Some(Fetch { code, swz: 0, read_bytes: 16 })
            } else {
                vertex_fetch(at.format)
            };
            let Some(f) = f.filter(|_| !self.c.vb[b].is_null()) else { return Err(Reject::VertexInput) };
            let rb = f.read_bytes as usize;
            let (stride, inst) = (pl.binding_stride[b] as u64, pl.binding_instance[b]);
            let (vb, avail) = (self.c.vb[b], self.c.vb_avail[b]);
            let asz = blocks(at.format).bs as u64;
            let copy = |dst: &mut [u8], src: u64| {
                if src + asz <= avail {
                    vattr_copy(at.format, conv, dst, unsafe { vb.add(src as usize) }, asz as usize);
                }
            };
            if self.lines && !inst {
                for other in 0..2 {
                    let mut tight = vec![0u8; nrows_lines as usize * rb];
                    for (r, dst) in tight.chunks_exact_mut(rb).enumerate() {
                        copy(dst, self.pairs[r / 6][LINE_END[r % 6] ^ other] as u64 * stride + at.offset as u64);
                    }
                    let off = self.heap.upload(&tight)?;
                    let slot = at.location + if other != 0 { BORGVK_LINE_OTHER } else { 0 };
                    send_vattr(slot, f.code, off, nrows_lines, f.read_bytes, f.swz);
                }
                continue;
            }
            let (row0, row1) = if inst { (self.imin, self.imax) } else { (self.vmin, self.vmax) };
            let expand = self.points && !inst;
            let (e0, e1) = if expand { (row0 * x as i64, row1 * x as i64 + x as i64 - 1) } else { (row0, row1) };
            if e1 + 1 > MAX_ROWS {
                return Err(Reject::VertexRange);
            }
            let mut tight = vec![0u8; (e1 - e0 + 1) as usize * rb];
            for (r, dst) in tight.chunks_exact_mut(rb).enumerate() {
                let row = if expand { (e0 + r as i64) / x as i64 } else { row0 + r as i64 };
                copy(dst, (row as u64).wrapping_mul(stride).wrapping_add(at.offset as u64));
            }
            let off = self.heap.upload(&tight)?;
            send_vattr(at.location, f.code, off.wrapping_sub((e0 as u32).wrapping_mul(f.read_bytes)), e1 as u32 + 1, f.read_bytes, f.swz);
        }
        Ok(())
    }

    fn point_corners(&mut self) -> Result<(), Reject> {
        if !self.points {
            return Ok(());
        }
        let x = self.x as i64;
        // (dx, dy) in clip units, then the point coordinate (u, v) of the corner.
        let (e0, e1) = (self.vmin * x, self.vmax * x + x - 1);
        let f = vertex_fetch(VK_FORMAT_R32G32B32A32_SFLOAT).filter(|_| e1 + 1 <= MAX_ROWS).ok_or(Reject::VertexRange)?;
        let c: Vec<f32> = (e0..=e1)
            .flat_map(|e| {
                let k = CORNER[(e % x) as usize];
                [k[0] * 2.0 / self.w as f32, k[1] * 2.0 / self.h as f32, k[0] + 0.5, k[1] + 0.5]
            })
            .collect();
        let off = self.heap.upload(bytes_of(&c))?;
        send_vattr(POINT_SLOT, f.code, off.wrapping_sub((e0 as u32).wrapping_mul(f.read_bytes)), e1 as u32 + 1, f.read_bytes, f.swz);
        Ok(())
    }

    fn pool_stage(&mut self, vert: bool) -> Result<(), Reject> {
        let pl = self.pl;
        let me: *const c_void = std::ptr::from_ref(pl).cast();
        let (stage, slot) = if vert { (BORGVK_STAGE_VERT, BORGVK_VPOOL_SLOT) } else { (BORGVK_STAGE_FRAG, BORGVK_POOL_SLOT) };
        let batch = if vert { &mut self.c.batch_vpool } else { &mut self.c.batch_pool };
        let fb = &pl.blob[stage as usize];
        if fb.npool == 0 || *batch == me {
            return Ok(());
        }
        let mut words = [0u32; BORGVK_POOL_MAX as usize + 4];
        words[..fb.npool as usize].copy_from_slice(&fb.pool[..fb.npool as usize]);
        let rows = fb.npool.div_ceil(4);
        let off = self.heap.upload(bytes_of(&words[..rows as usize * 4]))?;
        let f = fetch(VK_FORMAT_R32G32B32A32_UINT)?;
        send_vattr(slot, f.code, off, rows, f.read_bytes, f.swz);
        *batch = me;
        Ok(())
    }

    fn srgb_table(&mut self) -> Result<(), Reject> {
        if !self.pl.srgb_table || self.c.batch_srgb {
            return Ok(());
        }
        let t: Vec<f32> = (0..=255u8).map(|i| unsafe { util_format_srgb_8unorm_to_linear_float(i) }).collect();
        let off = self.heap.upload(bytes_of(&t))?;
        let f = fetch(VK_FORMAT_R32_SFLOAT)?;
        send_vattr(BORGVK_SRGB_SLOT, f.code, off, 256, f.read_bytes, f.swz);
        self.c.batch_srgb = true;
        Ok(())
    }

    fn line_consts(&mut self) -> Result<(), Reject> {
        if !self.lines {
            return Ok(());
        }
        // The half-pixel step in clip units for x and y, then the target's size.
        let f = fetch(VK_FORMAT_R32G32B32A32_SFLOAT)?;
        let (w, h, n) = (self.w as f32, self.h as f32, self.pairs.len() as u32 * 6);
        let c: Vec<f32> = (0..n as usize).flat_map(|r| [LINE_SIDE[r % 6] / w, LINE_SIDE[r % 6] / h, w, h]).collect();
        let off = self.heap.upload(bytes_of(&c))?;
        send_vattr(BORGVK_LINE_SLOT, f.code, off, n, f.read_bytes, f.swz);
        Ok(())
    }

    fn indices(&mut self) -> Result<(), Reject> {
        let p = self.p;
        (self.idx_off, self.index_code, self.voff_out) = (0, if p.index_size == 2 { 1 } else { 2 }, p.vertex_offset as u32);
        if !p.indexed || self.lines {
            return Ok(());
        }
        if self.points {
            let x = self.x;
            let ex: Vec<u32> = (0..p.count)
                .flat_map(|i| {
                    let v = self.index(i) as i64 + p.vertex_offset as i64;
                    (0..x).map(move |k| (v * x as i64 + k as i64) as u32)
                })
                .collect();
            let off = self.heap.upload(bytes_of(&ex));
            (self.index_code, self.voff_out) = (2, 0);
            self.idx_off = off?;
        } else {
            let n = p.count as usize * p.index_size as usize;
            self.idx_off = self.heap.upload(unsafe { std::slice::from_raw_parts(self.idx, n) })?;
        }
        Ok(())
    }

    fn buffers(&mut self) -> Result<(), Reject> {
        let pl = self.pl;
        let c = &mut *self.c;
        let set = c.desc_set;
        let sets = [set, c.desc_sets[1], c.desc_sets[2], c.desc_sets[3]];
        let (mut smask, mut any_dyn) = (0u32, false);
        for (si, s) in sets.iter().enumerate() {
            let Some(s) = (unsafe { s.as_ref() }) else { continue };
            any_dyn |= s.dyn_mask != 0;
            for b in 0..BINDINGS.min(32usize.saturating_sub(pl.set_base[si] as usize)) {
                let buf = s.storage_mask >> b & 1 != 0 && unsafe { s.buffers[b].as_ref() }.is_some_and(|u| host(u).is_some());
                if buf || storage_image(s, b).is_some() {
                    smask |= 1 << (pl.set_base[si] as usize + b);
                }
            }
        }
        let umask = pl.ubo_mask & !1;
        if smask != 0 || umask != 0 || pl.push_window || pl.spills {
            let slots = if pl.spills {
                SPILL_SLOT + 1
            } else if pl.push_window {
                PUSH_WINDOW_SLOT + 1
            } else {
                1 + (32 - (smask | umask).leading_zeros())
            };
            let reuse = c.region_set == set && c.batch_draws != 0 && umask == 0 && !any_dyn && sets[1].is_null() && !pl.spills;
            let mut base = c.region_base;
            if !reuse {
                base = (self.heap.top + 3) & !3;
                if base as u64 + slots as u64 * SLOT_BYTES as u64 + 16 > HEAP_BYTES as u64 {
                    return Err(Reject::Upload);
                }
                self.heap.top = base + slots * SLOT_BYTES + 16;
                for (si, ds) in sets.iter().enumerate() {
                    let Some(ds) = (unsafe { ds.as_ref() }) else { continue };
                    for b in 0..BINDINGS {
                        let fb = pl.set_base[si] as u32 + b as u32;
                        if let Some(simg) = storage_image(ds, b).filter(|_| fb < 32) {
                            let n = (simg.vk.extent.width.wrapping_mul(simg.vk.extent.height)).min(SLOT_BYTES / 4 - 1);
                            let host = image_host(simg).unwrap();
                            let at = base + (1 + fb) * SLOT_BYTES;
                            let mut win = simg.vk.extent.width.to_ne_bytes().to_vec();
                            win.extend_from_slice(unsafe { std::slice::from_raw_parts(host, n as usize * 4) });
                            for (i, chunk) in win.chunks(256).enumerate() {
                                send_mem(at + i as u32 * 256, chunk);
                            }
                            push_writeback(c, at + 4, n * 4, host);
                            continue;
                        }
                        let Some(ub) = (unsafe { ds.buffers[b].as_ref() }) else { continue };
                        let Some(uhost) = host(ub).filter(|_| fb < 32) else { continue };
                        let st = smask >> fb & 1 != 0;
                        if !st && pl.ubo_mask != 0 && pl.ubo_mask >> fb & 1 == 0 {
                            continue;
                        }
                        let dyn_off = if si != 0 { &c.dyn_offs[si] } else { &c.dyn_off };
                        let off = ds.offsets[b].wrapping_add(if ds.dyn_mask >> b & 1 != 0 { dyn_off[b] } else { 0 });
                        let size = ub.vk.size;
                        if off >= size {
                            continue;
                        }
                        let range = ds.ranges[b];
                        let len = if range == VK_WHOLE_SIZE as u64 || range == 0 { size - off } else { range };
                        let len = len.min((size - off).min(SLOT_BYTES as u64)) as u32;
                        let h = unsafe { uhost.add(off as usize) };
                        let at = base + if st || (fb != 0 && pl.ubo_mask != 0) { (1 + fb) * SLOT_BYTES } else { 0 };
                        for i in (0..len).step_by(256) {
                            let n = 256.min(len - i);
                            let mut tmp = [0u8; 256];
                            unsafe { std::ptr::copy_nonoverlapping(h.add(i as usize), tmp.as_mut_ptr(), n as usize) };
                            send_mem(at + i, &tmp[..((n + 3) & !3) as usize]);
                        }
                        if st {
                            push_writeback(c, at, len, h);
                        }
                    }
                }
                if pl.push_window {
                    send_mem(base + PUSH_WINDOW_SLOT * SLOT_BYTES, bytes_of(&self.device.scratch.lock().unwrap().push_words));
                }
                (c.region_set, c.region_base) = (set, base);
            }
            self.ubo_base = base + 1;
        }
        let set = unsafe { set.as_ref() };
        if smask != 0 || umask != 0 || pl.spills || (pl.push_used && pl.ubo_mask == 0) {
            return Ok(());
        }
        for b in 0..BINDINGS {
            let Some(s) = set else { break };
            let Some(ub) = (unsafe { s.buffers[b].as_ref() }) else { continue };
            let Some(uhost) = host(ub) else { continue };
            if pl.ubo_mask != 0 && pl.ubo_mask >> b & 1 == 0 {
                continue;
            }
            let off = s.offsets[b].wrapping_add(if s.dyn_mask >> b & 1 != 0 { c.dyn_off[b] } else { 0 });
            let size = ub.vk.size;
            if off >= size {
                break;
            }
            let range = s.ranges[b];
            let len = if range == VK_WHOLE_SIZE as u64 || range == 0 { size - off } else { range };
            let len = len.min((size - off).min(65536)) as usize;
            let at = self.heap.upload(unsafe { std::slice::from_raw_parts(uhost.add(off as usize), len) })?;
            self.ubo_base = at + 1;
            break;
        }
        Ok(())
    }

    fn textures(&self) {
        let Some(set) = (unsafe { self.c.desc_set.as_ref() }) else { return };
        let tb = self.pl.tex_binding;
        let mut want = -1;
        if tb >= 0 {
            want = if set.map.n != 0 { -2 } else { tb };
            for j in 0..set.map.n as usize {
                if set.map.num[j] as i32 == tb {
                    want = set.map.slot[j] as i32;
                }
            }
            if want < 0 || want >= BINDINGS as i32 {
                want = -1;
            }
        }
        for b in 0..BINDINGS {
            if want >= 0 && b as i32 != want {
                continue;
            }
            if !set.views[b].is_null() && unsafe { borgvk_send_generic_texture(set.views[b], set.images[b], set.samplers[b]) } {
                break;
            }
            if !set.buffer_views[b].is_null() && unsafe { borgvk_send_buffer_texture(set.buffer_views[b]) } {
                break;
            }
        }
    }

    fn emit(&self) {
        let (p, pl) = (self.p, self.pl);
        let bits = self.load << 1 | (pl.push_stage as u32) << 5;
        let d = if self.lines {
            let words = [self.pairs.len() as u32 * 6, p.instances, 0, p.first_instance, 0, 0, self.ubo_base];
            Draw { topology: 0, index_type: 0, restart: bits, words }
        } else {
            let first = if p.indexed { 0 } else { p.first.wrapping_mul(self.x) };
            let words = [p.count.wrapping_mul(self.x), p.instances, first, p.first_instance, self.voff_out, self.idx_off, self.ubo_base];
            Draw { topology: self.topo, index_type: if p.indexed { self.index_code } else { 0 }, restart: pl.restart as u32 | bits, words }
        };
        send_draw(&d);
    }

    fn record(&mut self) -> Result<(), Reject> {
        self.vertex_attrs()?;
        self.point_corners()?;
        self.pool_stage(false)?;
        self.pool_stage(true)?;
        self.srgb_table()?;
        self.line_consts()?;
        self.indices()?;
        self.buffers()?;
        self.textures();
        self.emit();
        Ok(())
    }

    fn finish(&mut self, ok: bool) -> bool {
        let mut n = 0;
        let bytes = unsafe { borgvk_transport_capture_end(&mut n) };
        let c = &mut *self.c;
        let state = self.state;
        if !ok || bytes.is_null() || n == 0 {
            unsafe { (free(bytes.cast()), free(state.cast())) };
            return ok;
        }
        let slice = |p: *mut u8, n| unsafe { std::slice::from_raw_parts(p, n) };
        let appended = (self.state_same || state.is_null() || stream_append(c, slice(state, self.state_len))) && stream_append(c, slice(bytes, n));
        unsafe { free(bytes.cast()) };
        if !self.state_same && !state.is_null() && appended {
            unsafe { free(c.state_sent.cast()) };
            (c.state_sent, c.state_sent_len) = (state, self.state_len);
        } else {
            unsafe { free(state.cast()) };
        }
        if !appended {
            return false;
        }
        c.heap_top = self.heap.top;
        c.batch_draws += 1;
        c.batch_push = self.device.scratch.lock().unwrap().push_words;
        (c.batch_img, c.batch_nextra, c.batch_extra) = (self.color, self.nextra as u32, self.extra);
        (c.batch_depth, c.batch_stencil) = (c.depth_view, c.stencil_view);
        true
    }
}

fn push_writeback(c: &mut CommandBuffer, off: u32, size: u32, host: *mut u8) {
    if c.nwb < BORGVK_MAX_WRITEBACKS {
        c.wb[c.nwb as usize] = Writeback { off, size, host };
        c.nwb += 1;
    }
}

/// Scaled, sRGB and the rest of the plain float-converted vertex formats are expanded to RGBA32F on upload.
fn converts(f: VkFormat) -> bool {
    if vertex_fetch(f).is_some() {
        return false;
    }
    let pf = unsafe { vk_format_to_pipe_format(f) };
    pf != PIPE_FORMAT_NONE && unsafe { !util_format_is_compressed(pf) && !util_format_is_depth_or_stencil(pf) }
}

fn vattr_copy(f: VkFormat, conv: bool, dst: &mut [u8], src: *const u8, asz: usize) {
    if conv {
        // sRGB: as UNORM
        unsafe { util_format_unpack_rgba(util_format_linear(vk_format_to_pipe_format(f)), dst.as_mut_ptr().cast(), src.cast(), 1) };
    } else {
        let n = asz.min(dst.len());
        unsafe { std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), n) };
    }
}

fn storage_image(set: &DescriptorSet, b: usize) -> Option<&Image> {
    if !set.buffers[b].is_null() || !set.buffer_views[b].is_null() || set.sampled_mask >> b & 1 != 0 {
        return None;
    }
    let img = unsafe { set.images[b].as_ref() }.filter(|i| image_host(i).is_some())?;
    (img.vk.image_type == VK_IMAGE_TYPE_2D && img.vk.samples == 1 && blocks(img.vk.format).bs == 4).then_some(img)
}

fn draw_one(device: &Device, c: &mut CommandBuffer, p: &Params) -> bool {
    let Some(pl) = (unsafe { c.gfx_pipeline.as_ref() }) else { return reject(Reject::Undrawable) };
    let mut g = Ctx {
        device,
        c,
        p: *p,
        pl,
        topo: 0,
        x: 1,
        points: false,
        lines: false,
        color: null_mut(),
        extra: [null_mut(); 3],
        nextra: 0,
        w: 0,
        h: 0,
        idx: null(),
        vmin: 0,
        vmax: 0,
        imin: p.first_instance as i64,
        imax: p.first_instance as i64 + p.instances as i64 - 1,
        first_of_batch: false,
        state: null_mut(),
        state_len: 0,
        state_same: false,
        load: 0,
        pairs: Vec::new(),
        heap: Heap { top: 0 },
        idx_off: 0,
        index_code: 0,
        voff_out: 0,
        ubo_base: 0,
    };
    if let Err(why) = g.validate().and_then(|_| g.index_range()) {
        return reject(why);
    }
    g.reserve();
    g.first_of_batch = g.c.batch_draws == 0;
    g.state();
    g.begin_pass();
    g.heap = Heap { top: g.c.heap_top };
    let r = g.record();
    if let Err(why) = r {
        reject(why);
    }
    g.finish(r.is_ok())
}

/// Wide colour targets are one pipeline each, drawn in turn (no depth or stencil: a second pass would fail the test).
fn generic_draw(device: &Device, c: &mut CommandBuffer, p: &Params) -> bool {
    let first = c.gfx_pipeline;
    let Some(fp) = (unsafe { first.as_ref() }).filter(|f| f.emul_n != 0) else { return draw_one(device, c, p) };
    if !c.depth_view.is_null() || !c.stencil_view.is_null() {
        return reject(Reject::MrtFormats);
    }
    let views: [*mut vk_image_view; 4] = c.color_views[..4].try_into().unwrap();
    let mut ok = true;
    for k in 0..fp.emul_n as usize {
        if !ok {
            break;
        }
        if k != 0 {
            c.gfx_pipeline = fp.emul[k - 1].ptr();
        }
        c.color_views[0] = views[k];
        ok = draw_one(device, c, p);
    }
    c.color_views[..4].copy_from_slice(&views);
    c.gfx_pipeline = first;
    ok
}

fn sim_draw(command_buffer: VkCommandBuffer, mut p: Params) {
    let c = unsafe { cmd(command_buffer.ptr()) };
    let device = unsafe { &*c.vk.base.device.cast::<Device>() };
    // Only the simulator renders; cube.c (UBO-driven) has its own path at submit.
    if !(std::env::var_os("BORGVK_SIM").is_some() || enabled()) || (!c.desc_set.is_null() && unsafe { borgvk_set_is_cube(c.desc_set) }) {
        return;
    }
    p.index_size = c.index_size;
    if p.indexed && p.index_size == 0 {
        return;
    }
    if generic_draw(device, c, &p) {
        c.generic_drawn = true;
    }
}

entrypoint! {
    unsafe fn borgvk_CmdDraw(command_buffer: VkCommandBuffer, count: u32, instances: u32, first: u32, first_instance: u32) {
        sim_draw(command_buffer, Params { count, instances, first, first_instance, ..Default::default() });
    }
}

entrypoint! {
    unsafe fn borgvk_CmdDrawIndexed(command_buffer: VkCommandBuffer, count: u32, instances: u32, first: u32, vertex_offset: i32, first_instance: u32) {
        sim_draw(command_buffer, Params { indexed: true, count, instances, first, vertex_offset, first_instance, ..Default::default() });
    }
}

fn indirect<T: Copy>(buffer: VkBuffer, offset: VkDeviceSize, count: u32, stride: u32) -> Vec<T> {
    let Some(base) = (unsafe { buffer.get() }).and_then(|b| host(b)) else { return Vec::new() };
    let stride = if stride != 0 { stride as usize } else { size_of::<T>() };
    (0..count as usize).map(|i| unsafe { base.add(offset as usize + i * stride).cast::<T>().read_unaligned() }).collect()
}

entrypoint! {
    unsafe fn borgvk_CmdDrawIndirect(command_buffer: VkCommandBuffer, buffer: VkBuffer, offset: VkDeviceSize, count: u32, stride: u32) {
        for d in indirect::<VkDrawIndirectCommand>(buffer, offset, count, stride) {
            let p = Params { count: d.vertexCount, instances: d.instanceCount, first: d.firstVertex, first_instance: d.firstInstance, ..Default::default() };
            sim_draw(command_buffer, p);
        }
    }
}

entrypoint! {
    unsafe fn borgvk_CmdDrawIndexedIndirect(command_buffer: VkCommandBuffer, buffer: VkBuffer, offset: VkDeviceSize, count: u32, stride: u32) {
        for d in indirect::<VkDrawIndexedIndirectCommand>(buffer, offset, count, stride) {
            let p = Params {
                indexed: true,
                count: d.indexCount,
                instances: d.instanceCount,
                first: d.firstIndex,
                vertex_offset: d.vertexOffset,
                first_instance: d.firstInstance,
                ..Default::default()
            };
            sim_draw(command_buffer, p);
        }
    }
}

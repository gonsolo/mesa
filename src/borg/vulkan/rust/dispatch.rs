// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! vkCmdDispatch: the bound buffers and images as one compute job (`hw::run_job`) on the device or
//! on raw simulators, split along one axis; what the shader wrote goes back.

use crate::cmd_buffer::CommandBuffer;
use crate::corpus::{borgvk_corpus_log, borgvk_corpus_on, hash};
use crate::entrypoint::entrypoint;
use crate::ffi::*;
use crate::fmt;
use crate::memory::{Buffer, DeviceMemory, Host, ViewBuffer};
use crate::pipeline::Pipeline;
use crate::descriptor_set::{find_flat, DescriptorSet};
use crate::state_upload::DEFAULT_SAMPLER;
use crate::texture_upload::{generic_texture, in_place};
use crate::util::{log, MESA_LOG_WARN};

const SLOT_BYTES: u32 = 0x80000 * 4;
const BASE_BYTES: u32 = 0x40000 * 4;
const MAX_PARTS: u32 = 64;
// The filtered texture: its descriptor, the sampler's at +64, the texels at +256.
const WINDOWS: u32 = 40; // borgc's compute.rs
const TEX_BYTES: u32 = BASE_BYTES + (WINDOWS + 4) * SLOT_BYTES;
const TEX_DESC_BASE: u32 = 0x378 / 4;
const SAMPLER_DESC_BASE: u32 = 0x37C / 4;
/// ISA v2 root table (docs/B8_isa_v2.md): LW/SW base (0: compute addresses are absolute), the
/// texture and the sampler descriptor tables; below the windows, after the program.
const ROOT: u32 = 0x3F8 / 4;
const ROOT_BYTES: u32 = 0xFFFF0;
const SAMPLERS: u32 = 64 * BORGVK_TEX_PAIRS as u32;
const TEXELS: u32 = SAMPLERS + 16 * BORGVK_TEX_PAIRS as u32;
const HDR: usize = 12;

/// One window of the job: a buffer, a decoded image or texel buffer, the push constants or the
/// constant pool.
struct CsBuf {
    host: *mut u8,
    decoded: Option<Vec<u8>>,
    wb: bool,
    skip: u32,
    size: u32,
    addr: u32,
    texels: *mut u8,
    pf: pipe_format,
    /// A decoded image's texels per layer and the bytes from one layer to the next.
    layer: (u32, u64),
    /// A buffer the shader reads and writes in device memory: not in the job.
    in_place: bool,
}

impl CsBuf {
    fn data(&self) -> &[u8] {
        match &self.decoded {
            Some(d) => &d[..self.size as usize],
            None => unsafe { std::slice::from_raw_parts(self.host, self.size as usize) },
        }
    }

    fn padded(&self) -> u32 {
        self.size.next_multiple_of(4)
    }
}

/// imageSize's words at the end of the binding's window (borg_image_size).
fn size_header(addr: u32, size: [u32; 3]) -> CsBuf {
    decoded(addr + SLOT_BYTES - 16, size.iter().flat_map(|v| v.to_ne_bytes()).collect(), 12)
}

fn decoded(addr: u32, data: Vec<u8>, size: u32) -> CsBuf {
    CsBuf { host: std::ptr::null_mut(), decoded: Some(data), wb: false, skip: 0, size, addr, texels: std::ptr::null_mut(), pf: PIPE_FORMAT_NONE, layer: (u32::MAX, 0), in_place: false }
}

fn mapped(mem: Option<&DeviceMemory>) -> Option<*mut u8> {
    mem.filter(|m| !m.map.is_null()).map(|m| m.map.cast())
}

/// The texel at `off` of `host` as RGBA32 in the first 16 bytes of `dst`.
pub fn unpack(pf: pipe_format, dst: &mut [u8], host: &Host, off: usize) {
    let out = <&mut [u8; 16]>::try_from(&mut dst[..16]).unwrap();
    host.with_bytes(off, fmt::blocksize(pf) as usize, |t| fmt::unpack_rgba(pf, t, out));
}

/// The windows of the bound sets, the push constants and buffer sizes (slot WINDOWS) and the constant pool (slot WINDOWS + 2).
fn gather(c: &CommandBuffer, pl: &Pipeline) -> Vec<CsBuf> {
    let mut bufs = Vec::new();
    let mut sizes = [0u32; WINDOWS as usize];
    // Each binding's base and offset limit (borgc's TABLE_BASE): its buffer, else its window.
    let mut table: Vec<u32> = (0..WINDOWS).flat_map(|w| [BASE_BYTES + w * SLOT_BYTES, SLOT_BYTES - 4]).collect();
    for si in 0..4 {
        let Some(set) = c.cs_set(si) else { continue };
        let dyn_off = &c.cs_dyn_offs[si];
        let base = pl.set_base[si] as u32;
        for b in 0..BORGVK_MAX_BINDINGS as usize {
            let flat = base + b as u32;
            let window = if pl.cs_compact {
                pl.cs_windows[..pl.cs_nwindows as usize].iter().position(|&f| f as u32 == flat).map(|p| p as u32)
            } else {
                Some(flat)
            };
            let Some(window) = window.filter(|&w| w < WINDOWS) else { continue };
            if bufs.len() >= BORGVK_CS_MAX_BUFS as usize - 3 {
                break;
            }
            let addr = BASE_BYTES + window * SLOT_BYTES;
            let (buf, bv, img) = (set.buffer(b), set.buffer_view(b), set.image(b));
            if let Some(bv) = bv.filter(|_| buf.is_none()) {
                // A texel buffer, decoded to RGBA32 per texel.
                let Some(tb) = bv.buffer() else { continue };
                let pf = fmt::pipe(bv.format);
                let bs = fmt::blocksize(pf);
                let Some(map) = mapped(tb.memory()).filter(|_| pf != PIPE_FORMAT_NONE && bs != 0) else { continue };
                // In place (borgc: borg_lower_image_in_place): the descriptor, base and element count.
                if set.storage_mask >> b & 1 != 0 && unsafe { borg_lower_image_in_place(pf) } {
                    let gpu = tb.memory().map_or(0, |m| m.gpu);
                    let off = tb.offset + bv.offset;
                    let max = if gpu != 0 { u32::MAX } else { (SLOT_BYTES - 64) / bs };
                    let n = (bv.range / bs as u64).min(max as u64) as u32;
                    let base = if gpu != 0 { gpu + off as u32 } else { addr + 64 };
                    let mut data: Vec<u8> = [base, 0, 0, n, 1, 1].iter().flat_map(|v| v.to_ne_bytes()).collect();
                    data.resize(64, 0);
                    if gpu != 0 {
                        bufs.push(decoded(addr, data, 64));
                        continue;
                    }
                    let at = unsafe { map.add(off as usize) };
                    data.extend_from_slice(unsafe { std::slice::from_raw_parts(at, (n * bs) as usize) });
                    let size = data.len() as u32;
                    bufs.push(CsBuf { host: at, wb: true, skip: 64, ..decoded(addr, data, size) });
                    continue;
                }
                let n = (bv.range / bs as u64).min(SLOT_BYTES as u64 / 16 - 1) as u32;
                let mut dec = vec![0u8; n.max(1) as usize * 16];
                let src = unsafe { map.add((tb.offset + bv.offset) as usize) };
                let Some(host) = tb.host() else { continue };
                for t in 0..n as usize {
                    unpack(pf, &mut dec[t * 16..], &host, bv.offset as usize + t * bs as usize);
                }
                bufs.push(CsBuf { texels: src, pf, ..decoded(addr, dec, n * 16) });
                bufs.push(size_header(addr, [n, 1, 1]));
                continue;
            }
            if let Some(img) = img.filter(|i| buf.is_none() && bv.is_none() && i.vk.samples == 1) {
                let Some(map) = mapped(img.memory()) else { continue };
                let src = unsafe { map.add(img.offset as usize) };
                let Some(host) = img.host() else { continue };
                // Storage images are linear; a sampled one is read from a linear copy.
                let lin = crate::texels::linear_copy(img);
                let (src, host) = match &lin {
                    Some(l) => (l.as_ptr().cast_mut(), unsafe { Host::new(l.as_ptr().cast_mut(), l.len()) }),
                    None => (src, host),
                };
                let (w, h) = (img.vk.extent.width, img.vk.extent.height);
                let flat = img.vk.image_type == VK_IMAGE_TYPE_2D;
                let view = set.view(b);
                let vf = view.map_or(img.vk.format, |v| v.format);
                let layered = view.is_some_and(|v| v.view_type != VK_IMAGE_VIEW_TYPE_2D);
                // The view's mip level and layers.
                let (lvl, l0, nl) = view.map_or((0, 0, img.vk.array_layers), |v| (v.base_mip_level, v.base_array_layer, v.layer_count));
                let sh = crate::image::shape(img);
                let blk = crate::image::blocks(img.vk.format);
                let off0 = sh.level_offset(lvl, l0, blk).0;
                let mip = |e: u32| (e >> lvl).max(1);
                let (w, h, d) = (mip(w), mip(h), mip(img.vk.extent.depth));
                if set.sampled_mask >> b & 1 != 0 && flat {
                    // A sampled image: width and height, then every layer's RGBA32 texels, read as the view's format.
                    let pf = fmt::pipe(vf);
                    let bs = fmt::blocksize(pf);
                    if pf == PIPE_FORMAT_NONE || bs == 0 || fmt::is_compressed(pf) {
                        continue;
                    }
                    let (per, lb) = (w * h, sh.layer_size(blk));
                    let n = (per * nl).min(SLOT_BYTES / 16 - 1);
                    let mut dec = vec![0u8; (1 + n) as usize * 16];
                    dec[..4].copy_from_slice(&w.to_ne_bytes());
                    dec[4..8].copy_from_slice(&h.to_ne_bytes());
                    for t in 0..n {
                        let at = off0 + (t / per) as u64 * lb + (t % per) as u64 * bs as u64;
                        unpack(pf, &mut dec[16 + 16 * t as usize..], &host, at as usize);
                    }
                    bufs.push(decoded(addr, dec, (1 + n) * 16));
                    continue;
                }
                if set.sampled_mask >> b & 1 != 0 {
                    continue;
                }
                // In place (borgc: borg_lower_image_in_place): the window holds the view's descriptor,
                // base, row pitch, slice or layer stride, width, height, depth or layers.
                if unsafe { borg_lower_image_in_place(fmt::pipe(vf)) } && lin.is_none() {
                    let (off, stride) = sh.level_offset(lvl, l0, blk);
                    let pitch = stride * blk.bs;
                    let solid = img.vk.image_type == VK_IMAGE_TYPE_3D;
                    let step = if solid { pitch as u64 * h as u64 } else { sh.layer_size(blk) };
                    let n = if solid { d } else { nl };
                    let bytes = if solid { step * d as u64 } else { step * nl as u64 } as usize;
                    let gpu = img.memory().map_or(0, |m| m.gpu);
                    let base = if gpu != 0 { gpu + (img.offset + off) as u32 } else { addr + 64 };
                    let mut data: Vec<u8> = [base, pitch, step as u32, w, h, n].iter().flat_map(|v| v.to_ne_bytes()).collect();
                    data.resize(64, 0);
                    if gpu != 0 || 64 + bytes > SLOT_BYTES as usize {
                        let size = data.len() as u32;
                        bufs.push(decoded(addr, data, size));
                        continue;
                    }
                    let at = unsafe { src.add(off as usize) };
                    data.extend_from_slice(unsafe { std::slice::from_raw_parts(at, bytes) });
                    let size = data.len() as u32;
                    bufs.push(CsBuf { host: at, wb: true, skip: 64, ..decoded(addr, data, size) });
                    continue;
                }
                let src = unsafe { src.add(off0 as usize) };
                let words = [PIPE_FORMAT_R32_UINT, PIPE_FORMAT_R32_SINT, PIPE_FORMAT_R32_FLOAT, PIPE_FORMAT_R8G8B8A8_UNORM,
                             PIPE_FORMAT_R8G8B8A8_UINT, PIPE_FORMAT_R8G8B8A8_SINT];
                let solid = img.vk.image_type == VK_IMAGE_TYPE_3D && view.is_none_or(|v| v.view_type == VK_IMAGE_VIEW_TYPE_3D);
                if (flat && !layered || solid) && words.contains(&fmt::pipe(vf)) {
                    // R32 or RGBA8 texels, packed by the shader (borg_image_words): the width, then the
                    // texels, slice after slice.
                    let d = if solid { d } else { 1 };
                    let n = (w * h * d).min(SLOT_BYTES / 4 - 5);
                    let mut dec = vec![0u8; (1 + n) as usize * 4];
                    dec[..4].copy_from_slice(&w.to_ne_bytes());
                    dec[4..].copy_from_slice(unsafe { std::slice::from_raw_parts(src, n as usize * 4) });
                    bufs.push(CsBuf { host: src, wb: true, skip: 4, ..decoded(addr, dec, (1 + n) * 4) });
                    bufs.push(size_header(addr, [w, h, d]));
                    continue;
                }
                // Any other storage image: width and height, then every layer's texels as RGBA32.
                let pf = fmt::pipe(vf);
                let bs = fmt::blocksize(pf);
                if pf == PIPE_FORMAT_NONE || bs == 0 || fmt::is_compressed(pf) {
                    continue;
                }
                let per = w * h * d;
                let lb = sh.layer_size(blk);
                let n = (per * nl).min(SLOT_BYTES / 16 - 2);
                let mut dec = vec![0u8; (1 + n) as usize * 16];
                dec[..4].copy_from_slice(&w.to_ne_bytes());
                dec[4..8].copy_from_slice(&h.to_ne_bytes());
                for t in 0..n {
                    let at = off0 + (t / per) as u64 * lb + (t % per) as u64 * bs as u64;
                    unpack(pf, &mut dec[16 + 16 * t as usize..], &host, at as usize);
                }
                bufs.push(size_header(addr, [w, h, if img.vk.image_type == VK_IMAGE_TYPE_3D { d } else { nl }]));
                bufs.push(CsBuf { texels: src, pf, skip: 16, layer: (per, lb), ..decoded(addr, dec, (1 + n) * 16) });
                continue;
            }
            let Some(b_) = buf else { continue };
            let Some(map) = mapped(b_.memory()) else { continue };
            let off = set.offsets[b] + dyn_off[b];
            let avail = b_.vk.size.saturating_sub(off);
            let r = set.ranges[b];
            let whole = if r == VK_WHOLE_SIZE as u64 || r > avail { avail } else { r };
            let host = unsafe { map.add((b_.offset + off) as usize) };
            // In the device heap the shader reads and writes the buffer where it is.
            let gpu = b_.memory().map_or(0, |m| m.gpu);
            if gpu != 0 {
                let (addr, size) = (gpu + (b_.offset + off) as u32, whole.min(u32::MAX as u64 - 3) as u32);
                sizes[window as usize] = size;
                table[2 * window as usize..][..2].copy_from_slice(&[addr, (size.max(4) - 4) & !3]);
                bufs.push(CsBuf { host, decoded: None, wb: false, skip: 0, size, addr, texels: std::ptr::null_mut(), pf: PIPE_FORMAT_NONE, layer: (u32::MAX, 0), in_place: true });
                continue;
            }
            let size = whole.min(SLOT_BYTES as u64) as u32;
            sizes[window as usize] = size;
            bufs.push(CsBuf { host, decoded: None, wb: true, skip: 0, size, addr, texels: std::ptr::null_mut(), pf: PIPE_FORMAT_NONE, layer: (u32::MAX, 0), in_place: false });
        }
    }

    let pc: Vec<u8> = c.pc.iter().chain(&sizes).chain(&table).flat_map(|w| w.to_ne_bytes()).collect();
    let n = pc.len() as u32;
    bufs.push(decoded(BASE_BYTES + WINDOWS * SLOT_BYTES, pc, n));
    let root: Vec<u8> = [0, TEX_BYTES, TEX_BYTES + SAMPLERS, 0].iter().flat_map(|w: &u32| w.to_ne_bytes()).collect();
    bufs.push(decoded(ROOT_BYTES, root, 16));
    if pl.cs_npool != 0 {
        let pool: Vec<u8> = pl.cs_pool[..pl.cs_npool as usize].iter().flat_map(|w| w.to_ne_bytes()).collect();
        bufs.push(decoded(BASE_BYTES + (WINDOWS + 2) * SLOT_BYTES, pool, pl.cs_npool * 4));
    }
    if pl.tex_npairs != 0 && bufs.len() < BORGVK_CS_MAX_BUFS as usize {
        let n = textures(c, pl);
        let len = n.len() as u32;
        bufs.push(decoded(TEX_BYTES, n, len));
    }
    bufs
}

/// The texture window: descriptor k at 64 k, sampler k at SAMPLERS + 16 k, then each texture's texels.
fn textures(c: &CommandBuffer, pl: &Pipeline) -> Vec<u8> {
    let sets: [Option<&DescriptorSet>; 4] = std::array::from_fn(|i| c.cs_set(i));
    let mut data = vec![0u8; TEXELS as usize];
    for (k, &[ft, fs]) in pl.tex_pairs[..pl.tex_npairs as usize].iter().enumerate() {
        let sampler = find_flat(&sets, &pl.set_base, fs).map_or(&DEFAULT_SAMPLER, |(s, b)| s.sampler(b));
        let tex = find_flat(&sets, &pl.set_base, ft).and_then(|(s, b)| s.view(b).zip(s.image(b)));
        for (w, v) in sampler.iter().enumerate() {
            data[(SAMPLERS + 16 * k as u32) as usize + 4 * w..][..4].copy_from_slice(&v.to_ne_bytes());
        }
        let put = |data: &mut Vec<u8>, words: &mut dyn Iterator<Item = u32>| {
            for (w, v) in words.enumerate() {
                data[64 * k + 4 * w..][..4].copy_from_slice(&v.to_ne_bytes());
            }
        };
        if let Some(d) = tex.and_then(|(v, i)| in_place(v, i)) {
            put(&mut data, &mut d.into_iter());
            continue;
        }
        let Some(t) = tex.and_then(|(v, i)| generic_texture(v, i)) else { continue };
        let offs = if t.desc.levels > 1 { t.offs } else { [0; 12] };
        let d = t.desc.pack();
        let at = TEX_BYTES + data.len() as u32;
        put(&mut data, &mut [at, d[0], d[1], d[2]].into_iter().chain(offs));
        data.extend_from_slice(&t.data);
        data.resize(data.len().next_multiple_of(256), 0);
    }
    data
}

/// Header (program size, groups, local size, presets, windows, group base), program, register
/// presets with the group count where the program reads it, windows.
fn build_job(pl: &Pipeline, g: [u32; 3], bufs: &[CsBuf]) -> Vec<u8> {
    fn put(job: &mut Vec<u8>, v: u32) {
        job.extend_from_slice(&v.to_ne_bytes());
    }
    let mut job = Vec::new();
    let tex = bufs.iter().any(|b| b.addr == TEX_BYTES);
    let nregs = pl.cs_nregs + if tex { 2 } else { 0 } + 1;
    let staged = bufs.iter().filter(|b| !b.in_place).count() as u32;
    for v in [pl.cs_nwords, g[0], g[1], g[2], pl.cs_local[0], pl.cs_local[1], pl.cs_local[2], nregs, staged, 0, 0, 0] {
        put(&mut job, v);
    }
    for &w in &pl.cs_words[..pl.cs_nwords as usize] {
        put(&mut job, w);
    }
    let nwg = if pl.cs_flags & 2 != 0 { (pl.cs_flags >> 8) & 31 } else { 0 };
    for i in 0..pl.cs_nregs as usize {
        let (r, mut v) = (pl.cs_regs[2 * i], pl.cs_regs[2 * i + 1]);
        if nwg != 0 && r <= nwg && r + 2 >= nwg {
            v = g[(nwg - r) as usize];
        }
        put(&mut job, r);
        put(&mut job, v);
    }
    put(&mut job, ROOT);
    put(&mut job, ROOT_BYTES);
    if tex {
        for (r, v) in [(TEX_DESC_BASE, TEX_BYTES), (SAMPLER_DESC_BASE, TEX_BYTES + SAMPLERS)] {
            put(&mut job, r);
            put(&mut job, v);
        }
    }
    for b in bufs.iter().filter(|b| !b.in_place) {
        put(&mut job, b.addr);
        put(&mut job, b.size);
        job.extend_from_slice(b.data());
        job.resize(job.len() + (b.padded() - b.size) as usize, 0);
    }
    job
}

/// The headers of the parts: a packed program runs four one-invocation workgroups along x in a
/// quad, so gx / 4 groups of four, then the gx % 4 left over in one group of that size.
fn part_headers(pl: &Pipeline, direct: bool, g: [u32; 3], job: &[u8]) -> Vec<[u32; HDR]> {
    let [gx, gy, gz] = g;
    let packed = pl.cs_flags & 4 != 0;
    let (px, rem) = if packed { (gx / 4, gx % 4) } else { (gx, 0) };
    let axis = if gy > 1 && gy >= px { 1 } else if px > 1 { 0 } else { 2 };
    let extent = [px, gy, gz][axis];
    let mut nparts = (px != 0) as u32;
    if direct && px != 0 && pl.cs_flags & 1 == 0 && gx as u64 * gy as u64 * gz as u64 >= 64 && extent > 1 {
        let ncpu = unsafe { sysconf(_SC_NPROCESSORS_ONLN as i32) };
        nparts = (extent as i64).min(ncpu.max(1)) as u32;
        if let Some(p) = std::env::var_os("BORGVK_CS_PARTS") {
            nparts = (extent as i64).min(crate::sim::atoi(&p.to_string_lossy())).max(1) as u32;
        }
    }
    let nparts = nparts.min(MAX_PARTS - 1);
    let head: [u32; HDR] = std::array::from_fn(|i| u32::from_ne_bytes(job[i * 4..i * 4 + 4].try_into().unwrap()));
    let mut hdrs = Vec::new();
    for k in 0..nparts {
        let mut h = head;
        h[1] = px;
        h[1 + axis] = extent * (k + 1) / nparts - extent * k / nparts; // gx gy gz
        h[9 + axis] = extent * k / nparts; // bx by bz
        hdrs.push(h);
    }
    if rem != 0 {
        let mut h = head;
        (h[1], h[4], h[9]) = (1, rem, px);
        hdrs.push(h);
    }
    hdrs
}

/// Runs the parts: each on a simulator of its own, all at once, or one after the other on the
/// device. Each answer is the buffers' bytes after the part, in job order.
fn run_parts(direct: Option<&std::ffi::OsStr>, hdrs: &[[u32; HDR]], job: &[u8]) -> Vec<Option<Vec<u8>>> {
    let part = |hdr: &[u32; HDR]| {
        let mut p = job.to_vec();
        for (i, w) in hdr.iter().enumerate() {
            p[i * 4..i * 4 + 4].copy_from_slice(&w.to_ne_bytes());
        }
        p
    };
    let Some(direct) = direct else {
        return hdrs.iter().map(|h| crate::hw::compute(&part(h))).collect();
    };
    std::thread::scope(|sc| {
        let runs: Vec<_> = hdrs
            .iter()
            .map(|h| {
                let p = part(h);
                sc.spawn(move || crate::dev::spawn(direct).and_then(|mut d| crate::hw::run_job(&mut d, &p)))
            })
            .collect();
        runs.into_iter().map(|r| r.join().ok().flatten()).collect()
    })
}

/// Every window as the job had it, with each byte a part changed.
fn merge(bufs: &[CsBuf], outs: &[Option<Vec<u8>>], ran: &mut bool) -> Vec<Vec<u8>> {
    let mut merged: Vec<Vec<u8>> = bufs.iter().map(|b| b.data().to_vec()).collect();
    for out in outs {
        let Some(out) = out else {
            log(MESA_LOG_WARN, "borgvk: compute sim failed".into());
            *ran = false;
            continue;
        };
        let mut at = 0;
        for (b, m) in bufs.iter().zip(merged.iter_mut()).filter(|(b, _)| !b.in_place) {
            if let Some(new) = out.get(at..at + b.size as usize) {
                for ((t, o), m) in new.iter().zip(b.data()).zip(m.iter_mut()) {
                    if t != o {
                        *m = *t;
                    }
                }
            }
            at += b.size as usize;
        }
    }
    for (b, m) in bufs.iter().zip(merged.iter_mut()).filter(|(b, _)| b.in_place) {
        *m = b.data().to_vec();
    }
    merged
}

/// Writes back what the shader stored: buffers and storage images, texel buffers encoded back
/// into their format.
fn apply(bufs: &[CsBuf], merged: &[Vec<u8>]) {
    for (b, m) in bufs.iter().zip(merged) {
        if b.wb {
            unsafe { std::ptr::copy_nonoverlapping(m[b.skip as usize..].as_ptr(), b.host, (b.size - b.skip) as usize) };
        }
    }
    for (b, m) in bufs.iter().zip(merged).filter(|(b, _)| !b.texels.is_null()) {
        let bs = fmt::blocksize(b.pf) as usize;
        let orig = b.data();
        let (per, lb) = (b.layer.0 as usize, b.layer.1 as usize);
        for t in b.skip as usize / 16..b.size as usize / 16 {
            if m[t * 16..t * 16 + 16] != orig[t * 16..t * 16 + 16] {
                let k = t - b.skip as usize / 16;
                let texel = unsafe { std::slice::from_raw_parts_mut(b.texels.add(k / per * lb + k % per * bs), bs) };
                fmt::pack_rgba(b.pf, texel, <&[u8; 16]>::try_from(&m[t * 16..t * 16 + 16]).unwrap());
            }
        }
    }
}

entrypoint! {
    fn borgvk_CmdDispatch(c: &mut CommandBuffer, gx: u32, gy: u32, gz: u32) {
        dispatch(c, gx, gy, gz);
    }
}

entrypoint! {
    fn borgvk_CmdDispatchIndirect(c: &mut CommandBuffer, buffer: Option<&Buffer>, offset: VkDeviceSize) {
        for d in crate::draw::indirect::<VkDispatchIndirectCommand>(buffer, offset, 1, 0) {
            dispatch(c, d.x, d.y, d.z);
        }
    }
}

fn dispatch(c: &mut CommandBuffer, gx: u32, gy: u32, gz: u32) {
    let direct = std::env::var_os("BORGVK_SIM_DIRECT").filter(|d| !d.is_empty());
    let Some(pl) = c.cs_pipeline().filter(|p| p.cs_ok) else { return };
    if gx == 0 || gy == 0 || gz == 0 || (direct.is_none() && !crate::hw::enabled()) {
        return;
    }
    let bufs = gather(c, pl);
    let g = [gx, gy, gz];
    let job = build_job(pl, g, &bufs);
    let mut ran = true;
    let outs = run_parts(direct.as_deref(), &part_headers(pl, direct.is_some(), g, &job), &job);
    let merged = merge(&bufs, &outs, &mut ran);
    if borgvk_corpus_on() {
        let out = merged.iter().zip(&bufs).fold(hash(BORGVK_CORPUS_SEED as u64, &[ran as u8]), |h, (m, b)| hash(h, &m[..b.size as usize]));
        borgvk_corpus_log(b'D' as _, hash(BORGVK_CORPUS_SEED as u64, &job), out);
    }
    if ran {
        apply(&bufs, &merged);
        c.dispatched = true;
    }
}

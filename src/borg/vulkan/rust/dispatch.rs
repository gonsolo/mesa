// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! vkCmdDispatch: the bound buffers and images as one compute job for direct_sim --compute
//! (split over processes along one axis) or the device; what the shader wrote goes back.

use crate::cmd_buffer::CommandBuffer;
use crate::corpus::{borgvk_corpus_log, borgvk_corpus_on, hash};
use crate::entrypoint::entrypoint;
use crate::ffi::*;
use crate::fmt;
use crate::memory::{DeviceMemory, Host, ViewBuffer};
use crate::pipeline::Pipeline;
use crate::descriptor_set::{find_flat, DescriptorSet};
use crate::state_upload::DEFAULT_SAMPLER;
use crate::texture_upload::generic_texture;
use crate::util::{log, temp_file, MESA_LOG_WARN};
use std::io::{Read, Seek, SeekFrom, Write};
use std::process::{Child, Command};

const SLOT_BYTES: u32 = 0x80000 * 4;
const BASE_BYTES: u32 = 0x40000 * 4;
const MAX_PARTS: u32 = 64;
// The filtered texture: its descriptor, the sampler's at +64, the texels at +256.
const TEX_BYTES: u32 = BASE_BYTES + 36 * SLOT_BYTES;
const TEX_DESC_BASE: u32 = 0x378 / 4;
const SAMPLER_DESC_BASE: u32 = 0x37C / 4;
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

fn decoded(addr: u32, data: Vec<u8>, size: u32) -> CsBuf {
    CsBuf { host: std::ptr::null_mut(), decoded: Some(data), wb: false, skip: 0, size, addr, texels: std::ptr::null_mut(), pf: PIPE_FORMAT_NONE }
}

fn mapped(mem: Option<&DeviceMemory>) -> Option<*mut u8> {
    mem.filter(|m| !m.map.is_null()).map(|m| m.map.cast())
}

/// The texel at `off` of `host` as RGBA32 in the first 16 bytes of `dst`.
fn unpack(pf: pipe_format, dst: &mut [u8], host: &Host, off: usize) {
    let out = <&mut [u8; 16]>::try_from(&mut dst[..16]).unwrap();
    host.with_bytes(off, fmt::blocksize(pf) as usize, |t| fmt::unpack_rgba(pf, t, out));
}

/// The windows of the bound sets, the push constants (slot 32) and the constant pool (slot 34).
fn gather(c: &CommandBuffer, pl: &Pipeline) -> Vec<CsBuf> {
    let mut bufs = Vec::new();
    for si in 0..4 {
        let Some(set) = c.cs_set(si) else { continue };
        let dyn_off = &c.cs_dyn_offs[si];
        let base = pl.set_base[si] as u32;
        for b in 0..BORGVK_MAX_BINDINGS as usize {
            if base + b as u32 >= 32 || bufs.len() >= BORGVK_CS_MAX_BUFS as usize - 2 {
                break;
            }
            let addr = BASE_BYTES + (base + b as u32) * SLOT_BYTES;
            let (buf, bv, img) = (set.buffer(b), set.buffer_view(b), set.image(b));
            if let Some(bv) = bv.filter(|_| buf.is_none()) {
                // A texel buffer, decoded to RGBA32 per texel.
                let Some(tb) = bv.buffer() else { continue };
                let pf = fmt::pipe(bv.format);
                let bs = fmt::blocksize(pf);
                let Some(map) = mapped(tb.memory()).filter(|_| pf != PIPE_FORMAT_NONE && bs != 0) else { continue };
                let n = (bv.range / bs as u64).min(SLOT_BYTES as u64 / 16) as u32;
                let mut dec = vec![0u8; n.max(1) as usize * 16];
                let src = unsafe { map.add((tb.offset + bv.offset) as usize) };
                let Some(host) = tb.host() else { continue };
                for t in 0..n as usize {
                    unpack(pf, &mut dec[t * 16..], &host, bv.offset as usize + t * bs as usize);
                }
                bufs.push(CsBuf { texels: src, pf, ..decoded(addr, dec, n * 16) });
                continue;
            }
            if let Some(img) = img.filter(|i| buf.is_none() && bv.is_none() && i.vk.image_type == VK_IMAGE_TYPE_2D && i.vk.samples == 1) {
                let Some(map) = mapped(img.memory()) else { continue };
                let src = unsafe { map.add(img.offset as usize) };
                let Some(host) = img.host() else { continue };
                let (w, h) = (img.vk.extent.width, img.vk.extent.height);
                if set.sampled_mask >> b & 1 != 0 {
                    // A sampled image: its width, then RGBA32 texels.
                    let pf = fmt::pipe(img.vk.format);
                    let bs = fmt::blocksize(pf);
                    if pf == PIPE_FORMAT_NONE || bs == 0 || fmt::is_compressed(pf) {
                        continue;
                    }
                    let n = (w * h).min(SLOT_BYTES / 16 - 1);
                    let mut dec = vec![0u8; (1 + n) as usize * 16];
                    dec[..4].copy_from_slice(&w.to_ne_bytes());
                    for t in 0..n as usize {
                        unpack(pf, &mut dec[16 + t * 16..], &host, t * bs as usize);
                    }
                    bufs.push(decoded(addr, dec, (1 + n) * 16));
                    continue;
                }
                if fmt::blocksize(fmt::pipe(img.vk.format)) == 4 {
                    // A storage image of 4-byte texels: its width, then the texels, written back.
                    let n = (w * h).min(SLOT_BYTES / 4 - 1);
                    let mut dec = vec![0u8; (1 + n) as usize * 4];
                    dec[..4].copy_from_slice(&w.to_ne_bytes());
                    dec[4..].copy_from_slice(unsafe { std::slice::from_raw_parts(src, n as usize * 4) });
                    bufs.push(CsBuf { host: src, wb: true, skip: 4, ..decoded(addr, dec, (1 + n) * 4) });
                    continue;
                }
            }
            let Some(b_) = buf else { continue };
            let Some(map) = mapped(b_.memory()) else { continue };
            let off = set.offsets[b] + dyn_off[b];
            let avail = b_.vk.size.saturating_sub(off);
            let r = set.ranges[b];
            let size = if r == VK_WHOLE_SIZE as u64 || r > avail { avail } else { r }.min(SLOT_BYTES as u64) as u32;
            let host = unsafe { map.add((b_.offset + off) as usize) };
            bufs.push(CsBuf { host, decoded: None, wb: true, skip: 0, size, addr, texels: std::ptr::null_mut(), pf: PIPE_FORMAT_NONE });
        }
    }
    let pc: Vec<u8> = c.pc.iter().flat_map(|w| w.to_ne_bytes()).collect();
    let n = pc.len() as u32;
    bufs.push(decoded(BASE_BYTES + 32 * SLOT_BYTES, pc, n));
    if pl.cs_npool != 0 {
        let pool: Vec<u8> = pl.cs_pool[..pl.cs_npool as usize].iter().flat_map(|w| w.to_ne_bytes()).collect();
        bufs.push(decoded(BASE_BYTES + 34 * SLOT_BYTES, pool, pl.cs_npool * 4));
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
        let Some(t) = tex.and_then(|(v, i)| generic_texture(v, i)) else { continue };
        let offs = if t.desc.levels > 1 { t.offs } else { [0; 12] };
        let d = t.desc.pack();
        let words = [TEX_BYTES + data.len() as u32, d[0], d[1], d[2]].into_iter().chain(offs);
        for (w, v) in words.enumerate() {
            data[64 * k + 4 * w..][..4].copy_from_slice(&v.to_ne_bytes());
        }
        for (w, v) in sampler.iter().enumerate() {
            data[(SAMPLERS + 16 * k as u32) as usize + 4 * w..][..4].copy_from_slice(&v.to_ne_bytes());
        }
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
    let nregs = pl.cs_nregs + if tex { 2 } else { 0 };
    for v in [pl.cs_nwords, g[0], g[1], g[2], pl.cs_local[0], pl.cs_local[1], pl.cs_local[2], nregs, bufs.len() as u32, 0, 0, 0] {
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
    if tex {
        for (r, v) in [(TEX_DESC_BASE, TEX_BYTES), (SAMPLER_DESC_BASE, TEX_BYTES + SAMPLERS)] {
            put(&mut job, r);
            put(&mut job, v);
        }
    }
    for b in bufs {
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

/// A part on its way: direct_sim running, or the device done; with its files.
struct Part {
    job: Option<String>,
    out: Option<String>,
    child: Option<Child>,
    device_ok: bool,
}

fn start(direct: Option<&std::ffi::OsStr>, hdr: &[u32; HDR], job: &[u8], ran: &mut bool) -> Part {
    let mut p = Part { job: None, out: temp_file("/tmp/borgvk_cs_out_XXXXXX").map(|(path, _)| path), child: None, device_ok: false };
    let mut part = job.to_vec();
    for (i, w) in hdr.iter().enumerate() {
        part[i * 4..i * 4 + 4].copy_from_slice(&w.to_ne_bytes());
    }
    let Some(direct) = direct else {
        match &p.out {
            Some(out) => p.device_ok = crate::hw::compute(&part, std::path::Path::new(out)),
            None => *ran = false,
        }
        return p;
    };
    let jf = temp_file("/tmp/borgvk_cs_job_XXXXXX");
    p.job = jf.as_ref().map(|(path, _)| path.clone());
    let (Some((jpath, mut f)), Some(out)) = (jf, p.out.clone()) else {
        *ran = false;
        return p;
    };
    let _ = f.write_all(&part);
    drop(f);
    p.child = Command::new(direct).arg("--compute").arg(&jpath).arg(&out).env("BORG_SIM_CFG", "simt").spawn().ok();
    p
}

/// Every window as the job had it, with each byte a part changed.
fn merge(bufs: &[CsBuf], parts: &mut [Part], ran: &mut bool) -> Vec<Vec<u8>> {
    let mut merged: Vec<Vec<u8>> = bufs.iter().map(|b| b.data().to_vec()).collect();
    for p in parts.iter_mut() {
        let ok = match p.child.as_mut() {
            Some(c) => c.wait().is_ok_and(|s| s.success()),
            None => p.device_ok,
        };
        let f = p.out.as_ref().filter(|_| ok).and_then(|o| std::fs::File::open(o).ok());
        let Some(mut f) = f else {
            log(MESA_LOG_WARN, "borgvk: compute sim failed".into());
            *ran = false;
            continue;
        };
        for (b, m) in bufs.iter().zip(merged.iter_mut()) {
            let mut tmp = vec![0u8; b.size as usize];
            if f.read_exact(&mut tmp).is_ok() {
                for ((t, o), m) in tmp.iter().zip(b.data()).zip(m.iter_mut()) {
                    if t != o {
                        *m = *t;
                    }
                }
            }
            let _ = f.seek(SeekFrom::Current((b.padded() - b.size) as i64));
        }
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
        for t in 0..b.size as usize / 16 {
            if m[t * 16..t * 16 + 16] != orig[t * 16..t * 16 + 16] {
                let texel = unsafe { std::slice::from_raw_parts_mut(b.texels.add(t * bs), bs) };
                fmt::pack_rgba(b.pf, texel, <&[u8; 16]>::try_from(&m[t * 16..t * 16 + 16]).unwrap());
            }
        }
    }
}

entrypoint! {
    fn borgvk_CmdDispatch(c: &mut CommandBuffer, gx: u32, gy: u32, gz: u32) {
        let direct = std::env::var_os("BORGVK_SIM_DIRECT").filter(|d| !d.is_empty());
        let Some(pl) = c.cs_pipeline().filter(|p| p.cs_ok) else { return };
        if gx == 0 || gy == 0 || gz == 0 || (direct.is_none() && !crate::hw::enabled()) {
            return;
        }
        let bufs = gather(c, pl);
        let g = [gx, gy, gz];
        let job = build_job(pl, g, &bufs);
        let mut ran = true;
        let mut parts: Vec<Part> = part_headers(pl, direct.is_some(), g, &job)
            .iter()
            .map(|h| start(direct.as_deref(), h, &job, &mut ran))
            .collect();
        let merged = merge(&bufs, &mut parts, &mut ran);
        if borgvk_corpus_on() {
            let out = merged.iter().zip(&bufs).fold(hash(BORGVK_CORPUS_SEED as u64, &[ran as u8]), |h, (m, b)| hash(h, &m[..b.size as usize]));
            borgvk_corpus_log(b'D' as _, hash(BORGVK_CORPUS_SEED as u64, &job), out);
        }
        if ran {
            apply(&bufs, &merged);
            c.dispatched = true;
        }
        let keep = std::env::var_os("BORGVK_KEEP_JOB").is_some();
        for p in &parts {
            if let Some(j) = p.job.as_ref().filter(|_| !keep) {
                let _ = std::fs::remove_file(j);
            }
            if let Some(o) = &p.out {
                let _ = std::fs::remove_file(o);
            }
        }
    }
}

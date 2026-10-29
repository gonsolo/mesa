// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Renders a pass's packet stream on `direct_sim --serve` processes (`BORGVK_SIM_DIRECT`), or on
//! the Borg device through `borg_core` in this process. A large pass is split into horizontal
//! strips of tile rows, one simulator each.

use crate::ffi::*;
use crate::memory::Host;
use crate::texel::Texel;
use crate::fmt;
use crate::image::Image;
use crate::format::{depth_is_d32, flush_format};
use crate::util::{ftruncate, log, mmap, MESA_LOG_INFO, MESA_LOG_WARN};
use std::ffi::OsStr;
use std::io::{Read, Write};
use std::ops::Range;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

const MEM_BYTES: usize = 0x800_0000;
const MAX_PARTS: usize = 24;
const PARALLEL_BYTES: usize = 65536;

/// One simulator: a server process sharing `mem` through a memfd, or the device itself.
struct Server {
    child: Option<Child>,
    to: Option<ChildStdin>,
    from: Option<ChildStdout>,
    hw: bool,
    mem: *mut u8,
    memfd: i32,
    w: u32,
    h: u32,
    fb: u32,
    zb: u32,
    sb: u32,
    heap: u32,
    att: [u32; 4],
}

// The memory is only touched under SERVERS' lock.
unsafe impl Send for Server {}

impl Server {
    const fn new() -> Self {
        Server {
            child: None,
            to: None,
            from: None,
            hw: false,
            mem: std::ptr::null_mut(),
            memfd: -1,
            w: 0,
            h: 0,
            fb: 0,
            zb: 0,
            sb: 0,
            heap: 0,
            att: [0; 4],
        }
    }

    fn mem(&self, addr: u32) -> *mut u8 {
        unsafe { self.mem.add(addr as usize) }
    }

    /// The memory shared with the simulator, mapped from start to stop.
    fn host(&self) -> Host<'_> {
        unsafe { Host::new(self.mem, MEM_BYTES) }
    }

    fn start(&mut self, bin: Option<&OsStr>) -> bool {
        if self.child.is_some() || self.hw {
            return true;
        }
        let prot = (PROT_READ | PROT_WRITE) as i32;
        let Some(bin) = bin else {
            if !crate::hw::enabled() || unsafe { borg_hw_open() } != 0 {
                return false;
            }
            let flags = (MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE) as i32;
            let m = unsafe { mmap(std::ptr::null_mut(), MEM_BYTES as _, prot, flags, -1, 0) };
            if m as usize == usize::MAX {
                return false;
            }
            (self.mem, self.hw) = (m.cast(), true);
            return true;
        };
        if self.mem.is_null() {
            let fd = unsafe { memfd_create(c"borg-dram".as_ptr(), 0) };
            if fd < 0 || unsafe { ftruncate(fd, MEM_BYTES as _) } < 0 {
                return false;
            }
            let m = unsafe { mmap(std::ptr::null_mut(), MEM_BYTES as _, prot, MAP_SHARED as i32, fd, 0) };
            if m as usize == usize::MAX {
                unsafe { close(fd) };
                return false;
            }
            (self.mem, self.memfd) = (m.cast(), fd);
        }
        unsafe { signal(SIGPIPE as i32, std::mem::transmute::<usize, __sighandler_t>(1)) }; // SIG_IGN
        let child = Command::new(bin)
            .arg("--serve")
            .arg(self.memfd.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn();
        let Ok(mut child) = child else {
            return false;
        };
        (self.to, self.from) = (child.stdin.take(), child.stdout.take());
        self.child = Some(child);
        (self.w, self.h) = (0, 0);
        true
    }

    fn set_size(&mut self, w: u32, h: u32) -> bool {
        if (self.w, self.h) == (w, h) {
            return true;
        }
        if self.hw {
            unsafe {
                borg_core_list_flush();
                borg_core_set_flush_format(0);
                borg_core_init(w as i32, h as i32);
            }
            (self.fb, self.zb, self.sb, self.heap) = (DRAM_OUT_BASE_SPI, BORG_ZB_SPI, BORG_SB_SPI, BORG_HEAP_SPI);
            for a in 1..4u32 {
                self.att[a as usize] = 0x4A0_0000 + (a - 1) * 0x100_0000; // BORG_ATT_SPI(a)
            }
            (self.w, self.h) = (w, h);
            return true;
        }
        let cmd = [0xBC, w as u8, (w >> 8) as u8, h as u8, (h >> 8) as u8];
        let mut r = [0u8; 28];
        let (Some(to), Some(from)) = (&mut self.to, &mut self.from) else {
            return false;
        };
        if to.write_all(&cmd).is_err() || from.read_exact(&mut r).is_err() {
            return false;
        }
        let a: [u32; 7] = std::array::from_fn(|i| u32::from_ne_bytes(r[4 * i..4 * i + 4].try_into().unwrap()));
        (self.fb, self.zb, self.sb, self.heap) = (a[0], a[1], a[2], a[3]);
        self.att[1..4].copy_from_slice(&a[4..7]);
        (self.w, self.h) = (w, h);
        true
    }

    fn stop(&mut self) {
        (self.to, self.from) = (None, None);
        if let Some(mut child) = self.child.take() {
            let _ = child.wait();
        }
        (self.w, self.h) = (0, 0);
    }
}

static SERVERS: Mutex<[Server; MAX_PARTS]> = Mutex::new([const { Server::new() }; MAX_PARTS]);

fn stop_all(s: &mut [Server]) {
    s.iter_mut().for_each(Server::stop);
}

/// The server's packet loop (simulation/direct/driver.cpp, Driver::serve), in this process.
fn hw_feed(p: &[u8]) {
    let mut i = 0;
    while i < p.len() {
        let len = unsafe { borg_core_pkt_len(p[i]) } as usize;
        if len == 0 || i + len > p.len() {
            break;
        }
        if unsafe { borg_core_packet(p[i..].as_ptr()) } == BC_DRAW as i32 {
            unsafe { borg_core_list_draw() };
        }
        i += len;
    }
    unsafe { borg_core_list_flush() };
}

fn hw_upload(s: &Server, ranges: &[(u32, u32)]) {
    for &(addr, size) in ranges {
        unsafe { borg_hw_mem_write(addr, s.mem(addr).cast(), size) };
    }
}

fn hw_download(s: &Server, ranges: &[(u32, u32)]) {
    for &(addr, size) in ranges {
        unsafe { borg_hw_mem_read(addr, s.mem(addr).cast(), size) };
    }
}

/// Byte address of pixel (x, y) in a 4x4-tiled surface `w` pixels wide.
fn tiled(base: u32, w: u32, x: u32, y: u32, tile_bytes: u32, px_bytes: u32) -> u32 {
    base + borg_layout::tile::offset(w, x, y, tile_bytes, px_bytes)
}

/// A RAW128 pixel is two 8-byte halves, 128 bytes apart in its tile (two slices).
fn px_get(h: &Host, g: usize, cpx: u32) -> [u8; 16] {
    let mut px = [0u8; 16];
    if cpx == 16 {
        h.read(g, &mut px[..8]);
        h.read(g + 128, &mut px[8..]);
    } else {
        h.read(g, &mut px[..cpx as usize]);
    }
    px
}

fn px_put(h: &Host, g: usize, cpx: u32, px: &[u8; 16]) {
    if cpx == 16 {
        h.write(g, &px[..8]);
        h.write(g + 128, &px[8..]);
    } else {
        h.write(g, &px[..cpx as usize]);
    }
}

fn blocksize(f: VkFormat) -> u32 {
    fmt::blocksize(fmt::pipe(f))
}

fn image_backed(img: &Image, w: u32, h: u32) -> bool {
    let e = img.vk.extent;
    let mapped = img.host().is_some();
    mapped
        && e.width >= w
        && e.height >= h
        && e.width as u64 * e.height as u64 * blocksize(img.vk.format) as u64 * (img.vk.samples as u64).max(1) <= img.size
}

fn to565(c: &[f32; 4]) -> u16 {
    let q = |v: f32, m: f32| (v.clamp(0.0, 1.0) * m + 0.5) as u32;
    (q(c[0], 31.0) << 11 | q(c[1], 63.0) << 5 | q(c[2], 31.0)) as u16
}

fn z16(z: f32) -> u16 {
    (z.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16
}

fn unpack_z(pf: pipe_format, t: &Texel) -> f32 {
    if pf == PIPE_FORMAT_Z16_UNORM_S8_UINT {
        return u16::from_ne_bytes([t[0], t[1]]) as f32 / 65535.0;
    }
    fmt::unpack_z(pf, t)
}

fn pack_z(pf: pipe_format, t: &mut Texel, z: f32) {
    if pf == PIPE_FORMAT_Z16_UNORM_S8_UINT {
        t[..2].copy_from_slice(&z16(z).to_ne_bytes());
        return;
    }
    fmt::pack_z(pf, t, z);
}

fn unpack_s(pf: pipe_format, t: &Texel) -> u8 {
    if pf == PIPE_FORMAT_Z16_UNORM_S8_UINT {
        return t[2];
    }
    fmt::unpack_s(pf, t)
}

fn pack_s(pf: pipe_format, t: &mut Texel, s: u8) {
    if pf == PIPE_FORMAT_Z16_UNORM_S8_UINT {
        t[2] = s;
        return;
    }
    fmt::pack_s(pf, t, s);
}

/// One attachment of the pass on the host side.
struct Surface<'a> {
    pf: pipe_format,
    bs: u32,
    host: Host<'a>,
    width: usize,
    plane: usize,
    planes: u32,
}

impl<'a> Surface<'a> {
    fn new(img: &'a Image) -> Self {
        let e = img.vk.extent;
        Surface {
            pf: fmt::pipe(img.vk.format),
            bs: blocksize(img.vk.format),
            host: img.host().unwrap(),
            width: e.width as usize,
            plane: e.width as usize * e.height as usize,
            planes: img.vk.samples.max(1),
        }
    }

    /// Offset of texel (x, y) of sample plane `p`.
    fn texel(&self, p: u32, x: u32, y: u32) -> usize {
        (p as usize * self.plane + y as usize * self.width + x as usize) * self.bs as usize
    }

    /// Offset of byte (x, y): extra colour attachments are one byte per pixel.
    fn byte(&self, x: u32, y: u32) -> usize {
        y as usize * self.width + x as usize
    }

    fn read(&self, p: u32, x: u32, y: u32) -> Texel {
        let mut t: Texel = [0; 32];
        self.host.read(self.texel(p, x, y), &mut t[..self.bs as usize]);
        t
    }

    /// Changes texel (x, y) of plane `p` in place: `f` sees its current bytes.
    fn update(&self, p: u32, x: u32, y: u32, f: impl FnOnce(&mut Texel)) {
        let mut t = self.read(p, x, y);
        f(&mut t);
        self.host.write(self.texel(p, x, y), &t[..self.bs as usize]);
    }
}

/// The pass's attachments on the host.
struct Targets<'a> {
    color: Option<Surface<'a>>,
    extra: Vec<Surface<'a>>,
    z: Option<Surface<'a>>,
    s: Option<Surface<'a>>,
}

/// What the GPU holds of the pass and where.
struct Layout {
    w: u32,
    h: u32,
    wp: u32,
    fmt: u8,
    cpx: u32,
    d32: bool,
    zpx: u32,
    ms: u32, // tiles per attachment tile: 4 when stored per sample (ATTACH_MS), else 1
}

impl Layout {
    fn color(&self, s: &Server, x: u32, y: u32, smp: u32) -> usize {
        self.at(s.fb, x, y, 16 * self.cpx, if self.cpx == 16 { 8 } else { self.cpx }, smp)
    }

    fn depth(&self, s: &Server, x: u32, y: u32, smp: u32) -> usize {
        self.at(s.zb, x, y, 16 * self.zpx, self.zpx, smp)
    }

    fn stencil(&self, s: &Server, x: u32, y: u32, smp: u32) -> usize {
        self.at(s.sb, x, y, 16, 1, smp)
    }

    /// Sample `smp`'s tile follows the tile's earlier samples.
    fn at(&self, base: u32, x: u32, y: u32, tile_bytes: u32, px_bytes: u32, smp: u32) -> usize {
        let o = borg_layout::tile::offset(self.wp, x, y, tile_bytes, px_bytes);
        (base + (o / tile_bytes * self.ms + smp) * tile_bytes + o % tile_bytes) as usize
    }
}

/// A 4x pass small enough to keep every sample in each attachment region is stored per sample.
pub fn per_sample(samples: u32, wp: u32, hp: u32, extra: bool) -> bool {
    samples == 4 && !extra && wp * hp <= 1 << 16
}

/// Before the pass: the GPU's copy of each pixel, to find what the pass changed.
struct Before {
    c: Vec<u8>,
    z: Vec<u8>,
    s: Vec<u8>,
    e: Vec<u8>,
}

/// In: the application's images into the GPU's attachments, each sample in its own tile when
/// the pass stores per sample.
fn copy_in(s: &Server, l: &Layout, rows: Range<u32>, t: &Targets, b: &mut Before) {
    let npx = (l.w * l.h) as usize;
    let m = s.host();
    for y in rows {
        for x in 0..l.w {
            for p in 0..l.ms {
                let px_i = (y * l.w + x) as usize;
                let i = p as usize * npx + px_i;
                let g = l.color(s, x, y, p);
                if let Some(c) = &t.color {
                    let src = c.read(p, x, y);
                    if l.fmt != 0 {
                        let mut px = [0u8; 16];
                        let n = c.bs.min(l.cpx) as usize;
                        px[..n].copy_from_slice(&src[..n]);
                        px_put(&m, g, l.cpx, &px);
                    } else {
                        let mut f = [0f32; 4];
                        fmt::unpack_rgba(c.pf, &src, &mut f);
                        m.write(g, &to565(&f).to_ne_bytes());
                    }
                }
                b.c[i * 16..i * 16 + 16].copy_from_slice(&px_get(&m, g, l.cpx));
                for (a, e) in t.extra.iter().enumerate().filter(|_| p == 0) {
                    let ge = tiled(s.att[a + 1], l.wp, x, y, 16, 1) as usize;
                    let mut v = [0u8];
                    e.host.read(e.byte(x, y), &mut v);
                    m.write(ge, &v);
                    m.read(ge, &mut v);
                    b.e[a * npx + px_i] = v[0];
                }
                if let Some(zs) = &t.z {
                    let gz = l.depth(s, x, y, p);
                    let v = unpack_z(zs.pf, &zs.read(p, x, y));
                    let bytes = if l.d32 {
                        v.to_ne_bytes()
                    } else {
                        let h = z16(v).to_ne_bytes();
                        [h[0], h[1], 0, 0]
                    };
                    m.write(gz, &bytes[..l.zpx as usize]);
                    b.z[i * 4..i * 4 + l.zpx as usize].copy_from_slice(&bytes[..l.zpx as usize]);
                }
                if let Some(ss) = &t.s {
                    let gs = l.stencil(s, x, y, p);
                    let mut v = [unpack_s(ss.pf, &ss.read(p, x, y))];
                    m.write(gs, &v);
                    m.read(gs, &mut v);
                    b.s[i] = v[0];
                }
            }
        }
    }
}

/// Out: the pixels the pass changed, into their sample plane, or every plane of a resolved
/// pass; returns how many.
fn copy_out(s: &Server, l: &Layout, rows: Range<u32>, t: &Targets, b: &Before) -> u32 {
    let npx = (l.w * l.h) as usize;
    let cpx = l.cpx as usize;
    let m = s.host();
    let mut changed = 0;
    for y in rows {
        for x in 0..l.w {
            for p in 0..l.ms {
                let px_i = (y * l.w + x) as usize;
                let i = p as usize * npx + px_i;
                let planes = |n: u32| if l.ms > 1 { p..p + 1 } else { 0..n };
                let cur = px_get(&m, l.color(s, x, y, p), l.cpx);
                if let Some(c) = t.color.as_ref().filter(|_| cur[..cpx] != b.c[i * 16..i * 16 + cpx]) {
                    changed += 1;
                    for q in planes(c.planes) {
                        if l.fmt != 0 {
                            c.host.write(c.texel(q, x, y), &cur[..c.bs.min(l.cpx) as usize]);
                        } else {
                            let v = u16::from_ne_bytes([cur[0], cur[1]]) as u32;
                            let f = [((v >> 11) & 31) as f32 / 31.0, ((v >> 5) & 63) as f32 / 63.0, (v & 31) as f32 / 31.0, 1.0];
                            c.update(q, x, y, |d| fmt::pack_rgba(c.pf, d, &f));
                        }
                    }
                }
                for (a, e) in t.extra.iter().enumerate().filter(|_| p == 0) {
                    let mut ge = [0u8];
                    m.read(tiled(s.att[a + 1], l.wp, x, y, 16, 1) as usize, &mut ge);
                    if ge[0] != b.e[a * npx + px_i] {
                        changed += 1;
                        e.host.write(e.byte(x, y), &ge);
                    }
                }
                if let Some(zs) = &t.z {
                    let mut cur = [0u8; 4];
                    let cur = &mut cur[..l.zpx as usize];
                    m.read(l.depth(s, x, y, p), cur);
                    if *cur != b.z[i * 4..i * 4 + l.zpx as usize] {
                        let v = if l.d32 {
                            f32::from_ne_bytes((&*cur).try_into().unwrap())
                        } else {
                            u16::from_ne_bytes((&*cur).try_into().unwrap()) as f32 / 65535.0
                        };
                        for q in planes(zs.planes) {
                            zs.update(q, x, y, |d| pack_z(zs.pf, d, v));
                        }
                    }
                }
                if let Some(ss) = &t.s {
                    let mut gs = [0u8];
                    m.read(l.stencil(s, x, y, p), &mut gs);
                    if gs[0] != b.s[i] {
                        for q in planes(ss.planes) {
                            ss.update(q, x, y, |d| pack_s(ss.pf, d, gs[0]));
                        }
                    }
                }
            }
        }
    }
    changed
}

/// C's atoi: leading blanks, a sign, then digits; 0 without any.
pub(crate) fn atoi(s: &str) -> i64 {
    let s = s.trim_start();
    let (neg, digits) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let v = digits
        .chars()
        .take_while(char::is_ascii_digit)
        .fold(0i64, |v, c| v.saturating_mul(10).saturating_add(c as i64 - '0' as i64));
    if neg {
        -v
    } else {
        v
    }
}

/// Strips of tile rows: how many simulators and how many rows each.
fn strips(n: usize, hp: u32, hw: bool) -> (usize, u32) {
    let np = match std::env::var_os("BORGVK_SERVE_PARTS") {
        Some(v) => atoi(&v.to_string_lossy()),
        None => (unsafe { sysconf(_SC_NPROCESSORS_ONLN as i32) } as i64).clamp(1, 12),
    };
    let np = if !hw && n >= PARALLEL_BYTES { np.clamp(1, MAX_PARTS as i64) as u32 } else { 1 };
    let ftiles = hp >> 2;
    let rows_per = ftiles.div_ceil(np);
    (ftiles.div_ceil(rows_per) as usize, rows_per)
}

fn feed_all(servers: &mut [Server], stream: &[u8]) -> bool {
    std::thread::scope(|sc| {
        let feeds: Vec<_> = servers
            .iter_mut()
            .map(|s| sc.spawn(move || s.to.as_mut().is_some_and(|to| to.write_all(stream).is_ok() && to.write_all(&[0xBD]).is_ok())))
            .collect();
        feeds.into_iter().fold(true, |ok, f| f.join().unwrap_or(false) && ok)
    })
}

fn start_servers(all: &mut [Server], np: usize, rows_per: u32, wp: u32, hp: u32, direct: Option<&OsStr>) -> bool {
    let hw = direct.is_none();
    (0..np).all(|part| {
        let s = &mut all[part];
        let strip = [0xBE, part as u8, np as u8, rows_per as u8, (rows_per >> 8) as u8];
        s.start(direct)
            && s.set_size(wp, hp)
            && if hw {
                unsafe {
                    borg_core_list_flush();
                    borg_core_set_strip(part as i32, np as i32, rows_per as i32);
                }
                true
            } else {
                s.to.as_mut().is_some_and(|to| to.write_all(&strip).is_ok())
            }
    })
}

/// The device: upload the attachments, run the stream, download them and the writebacks.
fn run_on_device(s: &Server, l: &Layout, t: &Targets, stream: &[u8], wb: &[Writeback]) {
    let tiles16 = (l.wp >> 2) * (l.h.next_multiple_of(4) >> 2) * 16 * l.ms;
    let mut ranges = Vec::new();
    if t.color.is_some() {
        ranges.push((s.fb, tiles16 * l.cpx));
    }
    ranges.extend((0..t.extra.len()).map(|a| (s.att[a + 1], tiles16)));
    if t.z.is_some() {
        ranges.push((s.zb, tiles16 * l.zpx));
    }
    if t.s.is_some() {
        ranges.push((s.sb, tiles16));
    }
    hw_upload(s, &ranges);
    keep_stream(stream);
    hw_feed(stream);
    hw_download(s, &ranges);
    for r in wb {
        let (a0, a1) = ((s.heap + r.off) & !3, (s.heap + r.off + r.size + 3) & !3);
        hw_download(s, &[(a0, a1 - a0)]);
    }
}

/// `BORGVK_SIM_KEEP`: the stream, for replaying it by hand.
fn keep_stream(stream: &[u8]) {
    if let Some(keep) = std::env::var_os("BORGVK_SIM_KEEP").filter(|k| !k.is_empty()) {
        let _ = std::fs::write(keep, stream);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_pass(
    stream: &[u8],
    color: Option<&Image>,
    extra: &[&Image],
    zimg: Option<&Image>,
    simg: Option<&Image>,
    wb: &[Writeback],
    area: [u32; 2],
) {
    // Depth only: no colour attachment, nothing of it goes in or comes back.
    let (w, h) = match color.or(zimg).or(simg) {
        _ if area[0] != 0 => (area[0], area[1]),
        Some(size) => (size.vk.extent.width, size.vk.extent.height),
        None => return,
    };
    let (wp, hp) = ((w + 3) & !3, (h + 3) & !3);
    if color.is_some_and(|c| !image_backed(c, w, h)) || extra.iter().any(|e| !image_backed(e, w, h)) {
        return;
    }
    let zimg = zimg.filter(|z| image_backed(z, w, h));
    let simg = simg.filter(|s| image_backed(s, w, h));
    // Several colour attachments are all one byte per pixel (RAW8).
    let fmt = if !extra.is_empty() { 5 } else { color.map_or(0, |c| flush_format(c.vk.format)) };
    let cpx = match fmt {
        5 => 1,
        4 | 0 => 2,
        6 => 8,
        7 => 16,
        _ => 4,
    };
    let d32 = zimg.is_some_and(|z| depth_is_d32(z.vk.format));
    let samples = color.or(zimg).or(simg).map_or(1, |i| i.vk.samples);
    let ms = if per_sample(samples, wp, hp, !extra.is_empty()) { 4 } else { 1 };
    let l = Layout { w, h, wp, fmt, cpx, d32, zpx: if d32 { 4 } else { 2 }, ms };
    let t = Targets {
        color: color.map(Surface::new),
        extra: extra.iter().map(|e| Surface::new(e)).collect(),
        z: zimg.map(Surface::new),
        s: simg.map(Surface::new),
    };
    let npx = (w * h) as usize;
    let n = npx * ms as usize;
    let mut b = Before { c: vec![0; n * 16], z: vec![0; n * 4], s: vec![0; n], e: vec![0; npx * 3] };

    let direct = std::env::var_os("BORGVK_SIM_DIRECT");
    let hw = direct.is_none();
    let (np, rows_per) = strips(stream.len(), hp, hw);
    let rows = |part: usize| part as u32 * rows_per * 4..h.min((part as u32 + 1) * rows_per * 4);

    let mut all = SERVERS.lock().unwrap_or_else(|e| e.into_inner());
    if !start_servers(&mut all[..], np, rows_per, wp, hp, direct.as_deref()) {
        log(MESA_LOG_WARN, "borgvk: cannot start the simulator".into());
        stop_all(&mut all[..]);
        return;
    }
    for part in 0..np {
        copy_in(&all[part], &l, rows(part), &t, &mut b);
    }
    if hw {
        run_on_device(&all[0], &l, &t, stream, wb);
    } else {
        keep_stream(stream);
        let servers = &mut all[..np];
        let answered = feed_all(servers, stream)
            && servers.iter_mut().all(|s| s.from.as_mut().is_some_and(|f| f.read_exact(&mut [0u8]).is_ok()));
        if !answered {
            log(MESA_LOG_WARN, "borgvk: the simulator stopped answering".into());
            stop_all(&mut all[..]);
            return;
        }
    }
    let mut changed = 0;
    for part in 0..np {
        changed += copy_out(&all[part], &l, rows(part), &t, &b);
    }
    // The writebacks from every strip; the first strip's bytes stand.
    for r in wb {
        let host = unsafe { std::slice::from_raw_parts_mut(r.host, r.size as usize) };
        for s in all[..np].iter().rev() {
            host.copy_from_slice(unsafe { std::slice::from_raw_parts(s.mem(s.heap + r.off), r.size as usize) });
        }
    }
    if std::env::var_os("BORGVK_DEBUG").is_some() {
        log(MESA_LOG_INFO, format!("borgvk: pass of {} bytes changed {changed} of {npx} pixels (format {fmt})", stream.len()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_addresses() {
        assert_eq!(tiled(0, 8, 0, 0, 64, 4), 0);
        assert_eq!(tiled(0, 8, 1, 0, 64, 4), 4);
        assert_eq!(tiled(0, 8, 0, 1, 64, 4), 16);
        assert_eq!(tiled(0, 8, 4, 0, 64, 4), 64);
        assert_eq!(tiled(100, 8, 5, 6, 64, 4), 100 + 3 * 64 + (1 | 2 << 2) * 4);
    }

    #[test]
    fn atoi_like_c() {
        assert_eq!(atoi("12"), 12);
        assert_eq!(atoi("  7x"), 7);
        assert_eq!(atoi("-3"), -3);
        assert_eq!(atoi("abc"), 0);
    }

    #[test]
    fn strips_cover_all_tile_rows() {
        assert_eq!(strips(10, 64, true), (1, 16));
        std::env::set_var("BORGVK_SERVE_PARTS", "3");
        assert_eq!(strips(PARALLEL_BYTES, 64, false), (3, 6));
        std::env::remove_var("BORGVK_SERVE_PARTS");
    }

    #[test]
    fn rgb565() {
        assert_eq!(to565(&[1.0, 1.0, 1.0, 1.0]), 0xFFFF);
        assert_eq!(to565(&[1.0, 0.0, 0.0, 1.0]), 0xF800);
    }
}

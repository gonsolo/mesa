// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Renders a pass's packet stream on `direct_sim --raw` processes (`BORGVK_SIM_DIRECT`), the driver
//! building the render lists, or on the Borg device (`dev`). A large pass is split into horizontal
//! strips of tile rows, one simulator each.

use crate::dev::Dev;
use crate::ffi::*;
use crate::memory::Host;
use crate::texel::Texel;
use crate::texels::{Planes, Texels};
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
    /// The raw device's list builder, its state kept across passes; and the size it was set up for.
    builder: Option<Box<crate::list::Builder>>,
    built: (u32, u32),
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
            builder: None,
            built: (0, 0),
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
            if !crate::hw::enabled() || crate::dev::with(|_| ()).is_none() {
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
        let mut cmd = Command::new(bin);
        cmd.arg("--raw").arg(self.memfd.to_string());
        if let Some(h) = crate::heap::get() {
            cmd.arg(h.fd().to_string());
        }
        let child = cmd
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
        (self.fb, self.zb, self.sb, self.heap) = (DRAM_OUT_BASE_SPI, BORG_ZB_SPI, BORG_SB_SPI, BORG_HEAP_SPI);
        for a in 1..4u32 {
            self.att[a as usize] = 0x4A0_0000 + (a - 1) * 0x100_0000; // BORG_ATT_SPI(a)
        }
        (self.w, self.h) = (w, h);
        true
    }

    fn stop(&mut self) {
        (self.to, self.from) = (None, None);
        if let Some(mut child) = self.child.take() {
            let _ = child.wait();
        }
        (self.w, self.h, self.builder, self.built) = (0, 0, None, (0, 0));
    }
}

static SERVERS: Mutex<[Server; MAX_PARTS]> = Mutex::new([const { Server::new() }; MAX_PARTS]);

fn stop_all(s: &mut [Server]) {
    s.iter_mut().for_each(Server::stop);
}

/// The stream's packets through the builder: the draws are queued, then rendered.
fn feed(b: &mut crate::list::Builder, g: &mut dyn crate::list::Gpu, stream: &[u8]) {
    let mut i = 0;
    while i < stream.len() {
        let n = crate::list::packet_len(stream[i]);
        if n == 0 || i + n > stream.len() {
            break;
        }
        if b.packet(g, &stream[i..i + n]) && stream[i] == 0xBA {
            b.draw(g);
        }
        i += n;
    }
    b.flush(g);
}

fn hw_upload(d: &mut Dev, s: &Server, ranges: &[(u32, u32)]) {
    for &(addr, size) in ranges {
        d.mem_write(addr, unsafe { std::slice::from_raw_parts(s.mem(addr), size as usize) });
    }
}

fn hw_download(d: &mut Dev, s: &Server, ranges: &[(u32, u32)]) {
    for &(addr, size) in ranges {
        d.mem_read(addr, unsafe { std::slice::from_raw_parts_mut(s.mem(addr), size as usize) });
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
    img.host().is_some() && e.width >= w && e.height >= h
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
    texels: Texels<'a>,
    planes: u32,
}

impl<'a> Surface<'a> {
    fn new(img: &'a Image) -> Self {
        Surface {
            pf: fmt::pipe(img.vk.format),
            bs: blocksize(img.vk.format),
            texels: Texels::of(img).unwrap(),
            planes: img.vk.samples.max(1),
        }
    }

    fn read(&self, p: u32, x: u32, y: u32) -> Texel {
        let mut t: Texel = [0; 32];
        self.texels.read(0, 0, [x, y, 0], p, &mut t);
        t
    }

    /// Changes texel (x, y) of sample `p` in place: `f` sees its current bytes.
    fn update(&self, p: u32, x: u32, y: u32, f: impl FnOnce(&mut Texel)) {
        let mut t = self.read(p, x, y);
        f(&mut t);
        self.texels.write(0, 0, [x, y, 0], p, &t);
    }

    /// The first `src.len()` bytes of texel (x, y) of sample `p`.
    fn put(&self, p: u32, x: u32, y: u32, src: &[u8]) {
        self.update(p, x, y, |t| t[..src.len()].copy_from_slice(src));
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

/// Bytes per pixel of a flush format in the GPU's tile.
fn flush_bytes(fmt: u8) -> u32 {
    match fmt {
        5 => 1,
        4 | 0 => 2,
        6 => 8,
        7 => 16,
        _ => 4,
    }
}

/// Where a pass renders when its attachments are tiled images in device memory that hold what
/// the GPU stores: the addresses of colour, depth, stencil and attachments 1-3, and the pitch in
/// tiles they share. None: the pass copies them in and out of the fixed planes.
pub fn placement(color: Option<&Image>, extra: &[&Image], zimg: Option<&Image>, simg: Option<&Image>, area: [u32; 2]) -> Option<([u32; 6], u32)> {
    let (w, h) = match color.or(zimg).or(simg) {
        _ if area[0] != 0 => (area[0], area[1]),
        Some(i) => (i.vk.extent.width, i.vk.extent.height),
        None => return None,
    };
    let fmt = if !extra.is_empty() { 5 } else { color.map_or(0, |c| flush_format(c.vk.format)) };
    let samples = color.or(zimg).or(simg).map_or(1, |i| i.vk.samples);
    let ms = if per_sample(samples, (w + 3) & !3, (h + 3) & !3, !extra.is_empty()) { 4 } else { 1 };
    let mut pitch = None;
    let mut base = |img: &Image, p: usize| -> Option<u32> {
        let gpu = img.memory()?.gpu;
        let (t, _) = Texels::of(img)?.tiled()?;
        let tiles = img.vk.extent.width.div_ceil(4);
        if gpu == 0 || img.vk.samples.max(1) != ms || !image_backed(img, w, h) || *pitch.get_or_insert(tiles) != tiles {
            return None;
        }
        Some(gpu + (img.offset + t.texel(p, 0, 0, [0, 0, 0], 0)) as u32)
    };
    let mut at = [0u32; 6];
    if let Some(c) = color {
        let same = if fmt == 0 { c.vk.format == VK_FORMAT_R5G6B5_UNORM_PACK16 } else { fmt != 7 && blocksize(c.vk.format) == flush_bytes(fmt) };
        at[0] = base(c, 0).filter(|_| same)?;
    }
    for (k, e) in extra.iter().enumerate() {
        at[3 + k] = base(e, 0).filter(|_| blocksize(e.vk.format) == 1)?;
    }
    if let Some(z) = zimg {
        at[1] = base(z, 0)?;
    }
    if let Some(s) = simg {
        let p = match Texels::of(s)?.tiled()?.1 {
            Planes::Depth { stencil: true, .. } => 1,
            Planes::Stencil => 0,
            _ => return None,
        };
        at[2] = base(s, p)?;
    }
    Some((at, pitch?))
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
                    let mut v = [e.read(0, x, y)[0]];
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
                            c.put(q, x, y, &cur[..c.bs.min(l.cpx) as usize]);
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
                        e.put(0, x, y, &ge);
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

/// Strips of tile rows: how many simulators and how many rows each. Every strip shades all the
/// vertices, so a pass with fewer than 16 pixels per triangle stays on one.
fn strips(n: usize, wp: u32, hp: u32, hw: bool, tris: u32) -> (usize, u32) {
    let np = match std::env::var_os("BORGVK_SERVE_PARTS") {
        Some(v) => atoi(&v.to_string_lossy()),
        None => (unsafe { sysconf(_SC_NPROCESSORS_ONLN as i32) } as i64).clamp(1, 12),
    };
    let fragment_bound = (wp as u64 * hp as u64) >= 16 * tris as u64;
    let np = if !hw && n >= PARALLEL_BYTES && fragment_bound { np.clamp(1, MAX_PARTS as i64) as u32 } else { 1 };
    let ftiles = hp >> 2;
    let rows_per = ftiles.div_ceil(np);
    (ftiles.div_ceil(rows_per) as usize, rows_per)
}

/// The raw device of one server: register writes in batches, memory through the shared mappings.
struct Raw<'a> {
    s: &'a mut Server,
    regs: Vec<u8>,
    n: u32,
    ok: bool,
}

impl Raw<'_> {
    fn send(&mut self) {
        if self.n == 0 {
            return;
        }
        let mut b = vec![b'W'];
        b.extend_from_slice(&self.n.to_le_bytes());
        b.append(&mut self.regs);
        self.n = 0;
        self.ok &= self.s.to.as_mut().is_some_and(|to| to.write_all(&b).is_ok());
    }

    fn ask(&mut self, q: &[u8]) -> u32 {
        self.send();
        let mut r = [0u8; 4];
        self.ok &= self.s.to.as_mut().is_some_and(|to| to.write_all(q).is_ok()) && self.s.from.as_mut().is_some_and(|f| f.read_exact(&mut r).is_ok());
        u32::from_le_bytes(r)
    }
}

impl crate::list::Gpu for Raw<'_> {
    fn reg(&mut self, off: u32, v: u32) {
        self.regs.extend_from_slice(&off.to_le_bytes());
        self.regs.extend_from_slice(&v.to_le_bytes());
        self.n += 1;
        if self.n == 4096 {
            self.send();
        }
    }

    fn read(&mut self, off: u32) -> u32 {
        let mut q = vec![b'R'];
        q.extend_from_slice(&off.to_le_bytes());
        self.ask(&q)
    }

    fn wait(&mut self, off: u32, mask: u32, val: u32) -> u32 {
        let mut q = vec![b'P'];
        for w in [off, mask, val] {
            q.extend_from_slice(&w.to_le_bytes());
        }
        self.ask(&q)
    }

    fn mem(&mut self, addr: u32, v: u32) {
        let p = match crate::heap::get().and_then(|h| h.host(addr)) {
            Some(p) => p,
            None if (addr as usize) + 4 <= MEM_BYTES => self.s.mem(addr),
            None => return,
        };
        unsafe { p.cast::<u32>().write_unaligned(v) };
    }
}

/// Every server builds and renders the stream's lists, each its strip of tile rows.
fn build_all(servers: &mut [Server], stream: &[u8], rows_per: u32) -> bool {
    let np = servers.len() as u32;
    std::thread::scope(|sc| {
        let runs: Vec<_> = servers
            .iter_mut()
            .enumerate()
            .map(|(k, s)| {
                sc.spawn(move || {
                    let mut b = s.builder.take().unwrap_or_else(|| Box::new(crate::list::Builder::new()));
                    let size = (s.w, s.h);
                    let built = s.built;
                    let mut g = Raw { s, regs: Vec::new(), n: 0, ok: true };
                    if built != size {
                        b.flush(&mut g);
                        b.init(&mut g, size.0, size.1);
                    }
                    b.flush(&mut g);
                    b.set_strip(k as u32, np, rows_per);
                    feed(&mut b, &mut g, stream);
                    g.send();
                    let mut sync = [0u8];
                    let ok = g.ok
                        && g.s.to.as_mut().is_some_and(|to| to.write_all(b"S").is_ok())
                        && g.s.from.as_mut().is_some_and(|f| f.read_exact(&mut sync).is_ok());
                    g.s.built = size;
                    g.s.builder = Some(b);
                    ok
                })
            })
            .collect();
        runs.into_iter().fold(true, |ok, r| r.join().unwrap_or(false) && ok)
    })
}

fn start_servers(all: &mut [Server], np: usize, wp: u32, hp: u32, direct: Option<&OsStr>) -> bool {
    all[..np].iter_mut().all(|s| s.start(direct) && s.set_size(wp, hp))
}

/// The device: upload the attachments, run the stream, download them, the writebacks and the
/// occlusion counters.
fn run_on_device(d: &mut Dev, s: &Server, l: &Layout, t: &Targets, stream: &[u8], wb: &[Writeback], occ: &[u32], rows_per: u32) {
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
    hw_upload(d, s, &ranges);
    d.build(s.w, s.h, |b, d| {
        b.flush(d);
        b.set_strip(0, 1, rows_per);
        feed(b, d, stream);
    });
    hw_download(d, s, &ranges);
    for r in wb {
        let (a0, a1) = ((s.heap + r.off) & !3, (s.heap + r.off + r.size + 3) & !3);
        hw_download(d, s, &[(a0, a1 - a0)]);
    }
    hw_download(d, s, &occ.iter().map(|&o| (s.heap + o, 4)).collect::<Vec<_>>());
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
    occ: &[u32],
    tris: u32,
    placed: bool,
) -> Vec<u64> {
    let none = vec![0; occ.len()];
    // Depth only: no colour attachment, nothing of it goes in or comes back.
    let (w, h) = match color.or(zimg).or(simg) {
        _ if area[0] != 0 => (area[0], area[1]),
        Some(size) => (size.vk.extent.width, size.vk.extent.height),
        None => return none,
    };
    let (wp, hp) = ((w + 3) & !3, (h + 3) & !3);
    if color.is_some_and(|c| !image_backed(c, w, h)) || extra.iter().any(|e| !image_backed(e, w, h)) {
        return none;
    }
    let zimg = zimg.filter(|z| image_backed(z, w, h));
    let simg = simg.filter(|s| image_backed(s, w, h));
    // Several colour attachments are all one byte per pixel (RAW8).
    let fmt = if !extra.is_empty() { 5 } else { color.map_or(0, |c| flush_format(c.vk.format)) };
    let cpx = flush_bytes(fmt);
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
    // In place, the GPU renders at the attachments: nothing goes in or comes back.
    let n = if placed { 0 } else { n };
    let mut b = Before { c: vec![0; n * 16], z: vec![0; n * 4], s: vec![0; n], e: vec![0; if placed { 0 } else { npx * 3 }] };

    let direct = std::env::var_os("BORGVK_SIM_DIRECT");
    let hw = direct.is_none();
    let (np, rows_per) = strips(stream.len(), wp, hp, hw, tris);
    let rows = |part: usize| part as u32 * rows_per * 4..h.min((part as u32 + 1) * rows_per * 4);

    let mut all = SERVERS.lock().unwrap_or_else(|e| e.into_inner());
    if !start_servers(&mut all[..], np, wp, hp, direct.as_deref()) {
        log(MESA_LOG_WARN, "borgvk: cannot start the simulator".into());
        stop_all(&mut all[..]);
        return none;
    }
    for part in 0..np {
        if !placed {
            copy_in(&all[part], &l, rows(part), &t, &mut b);
        }
    }
    if hw {
        crate::dev::with(|d| run_on_device(d, &all[0], &l, &t, stream, wb, occ, rows_per));
    } else {
            if !build_all(&mut all[..np], stream, rows_per) {
            log(MESA_LOG_WARN, "borgvk: the simulator stopped answering".into());
            stop_all(&mut all[..]);
            return none;
        }
    }
    let mut changed = 0;
    for part in 0..np {
        if !placed {
            changed += copy_out(&all[part], &l, rows(part), &t, &b);
        }
    }
    // The writebacks from every strip; the first strip's bytes stand.
    for r in wb {
        let host = unsafe { std::slice::from_raw_parts_mut(r.host, r.host_size()) };
        if r.pf != PIPE_FORMAT_NONE {
            let bs = crate::fmt::blocksize(r.pf) as usize;
            for s in all[..np].iter() {
                let win = unsafe { std::slice::from_raw_parts(s.mem(s.heap + r.off), r.size as usize) };
                for (t, new) in host.chunks_exact_mut(bs).zip(win.chunks_exact(16)) {
                    let new = <&[u8; 16]>::try_from(new).unwrap();
                    let mut old = [0u8; 16];
                    crate::fmt::unpack_rgba(r.pf, t, &mut old);
                    if old != *new {
                        crate::fmt::pack_rgba(r.pf, t, new);
                    }
                }
            }
            continue;
        }
        for s in all[..np].iter().rev() {
            host.copy_from_slice(unsafe { std::slice::from_raw_parts(s.mem(s.heap + r.off), r.size as usize) });
        }
    }
    if std::env::var_os("BORGVK_DEBUG").is_some() {
        log(MESA_LOG_INFO, format!("borgvk: pass of {} bytes changed {changed} of {npx} pixels (format {fmt})", stream.len()));
    }
    // Each strip counts its own tiles.
    occ.iter().map(|&o| all[..np].iter().map(|s| unsafe { s.mem(s.heap + o).cast::<u32>().read_unaligned() } as u64).sum()).collect()
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
        assert_eq!(strips(10, 64, 64, true, 0), (1, 16));
        std::env::set_var("BORGVK_SERVE_PARTS", "3");
        assert_eq!(strips(PARALLEL_BYTES, 64, 64, false, 0), (3, 6));
        assert_eq!(strips(PARALLEL_BYTES, 64, 64, false, 4096), (1, 16));
        std::env::remove_var("BORGVK_SERVE_PARTS");
    }

    #[test]
    fn rgb565() {
        assert_eq!(to565(&[1.0, 1.0, 1.0, 1.0]), 0xFFFF);
        assert_eq!(to565(&[1.0, 0.0, 0.0, 1.0]), 0xF800);
    }
}


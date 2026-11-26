// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! The Borg device the recorded lists run on (`BORGVK_HW`): the DRM render node (`BORG_HW_NODE`,
//! default /dev/dri/renderD128), or on a host the board's USB link or a `direct_sim` process
//! (`BORG_HW_RAW`). Register writes and memory words are batched; a read or another kind of
//! access sends the batch first.

use crate::drm;
use crate::ffi::*;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::fd::IntoRawFd;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

const MAX_REGS: usize = 1024;
const LINK_READS: usize = 56;
const MAX_MEM: usize = 4096;

enum Link {
    Drm(i32),
    /// The process behind the USB link or the simulator, and its pipes.
    Pipe(Box<dyn Write + Send>, Box<dyn Read + Send>, Child),
}

pub struct Dev {
    link: Link,
    regs: Vec<[u32; 2]>,
    mem: Vec<u32>,
    mem_off: u32,
    pub ok: bool,
}

static DEV: Mutex<Option<Dev>> = Mutex::new(None);

/// `f` on the device, opened on first use; None without one.
pub fn with<R>(f: impl FnOnce(&mut Dev) -> R) -> Option<R> {
    let mut d = DEV.lock().unwrap_or_else(|e| e.into_inner());
    if d.is_none() {
        *d = open();
    }
    d.as_mut().map(f)
}

fn open() -> Option<Dev> {
    let link = if let Some(raw) = std::env::var_os("BORG_HW_RAW") {
        let mut c = Command::new(raw).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().ok()?;
        let (to, from) = (c.stdin.take()?, c.stdout.take()?);
        let mut d = Dev::new(Link::Pipe(Box::new(to), Box::new(from), c));
        d.dirty(); // what an earlier session left
        crate::litedram::calibrate(&mut d);
        return Some(d);
    } else {
        let node = std::env::var("BORG_HW_NODE").unwrap_or_else(|_| "/dev/dri/renderD128".into());
        let f = (0..100).find_map(|_| {
            OpenOptions::new().read(true).write(true).open(&node).ok().or_else(|| {
                std::thread::sleep(std::time::Duration::from_millis(100));
                None
            })
        })?;
        Link::Drm(f.into_raw_fd())
    };
    Some(Dev::new(link))
}

/// A `direct_sim` of its own: its memory private, the device heap shared.
pub fn spawn(bin: &std::ffi::OsStr) -> Option<Dev> {
    let mut c = Command::new(bin);
    c.arg("-1");
    if let Some(h) = crate::heap::get() {
        c.arg(h.fd().to_string());
    }
    let mut c = c.stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().ok()?;
    let (to, from) = (c.stdin.take()?, c.stdout.take()?);
    Some(Dev::new(Link::Pipe(Box::new(to), Box::new(from), c)))
}

impl Dev {
    fn new(link: Link) -> Self {
        Dev { link, regs: Vec::with_capacity(MAX_REGS), mem: Vec::with_capacity(MAX_MEM), mem_off: 0, ok: true }
    }

    fn put(&mut self, b: &[u8]) {
        if let Link::Pipe(to, ..) = &mut self.link {
            self.ok &= to.write_all(b).is_ok();
        }
    }

    fn get(&mut self, b: &mut [u8]) {
        if let Link::Pipe(_, from, _) = &mut self.link {
            self.ok &= from.read_exact(b).is_ok();
        }
    }

    fn ioctl<T>(&mut self, request: u64, arg: &mut T) {
        if let Link::Drm(fd) = self.link {
            if !drm::ioctl(fd, request, arg) && self.ok {
                eprintln!("borgvk: DRM ioctl {request:#x}: {}", std::io::Error::last_os_error());
                self.ok = false;
            }
        }
    }

    fn flush_regs(&mut self) {
        if self.regs.is_empty() {
            return;
        }
        let regs = std::mem::take(&mut self.regs);
        if matches!(self.link, Link::Drm(_)) {
            let mut rw = drm_borg_reg_writes { writes: regs.as_ptr() as u64, count: regs.len() as u32, pad: 0 };
            self.ioctl(drm::REG_WRITES, &mut rw);
        } else {
            let mut b = Vec::with_capacity(5 + regs.len() * 8);
            b.push(b'W');
            b.extend_from_slice(&(regs.len() as u32).to_le_bytes());
            regs.iter().flatten().for_each(|w| b.extend_from_slice(&w.to_le_bytes()));
            self.put(&b);
        }
        self.regs = regs;
        self.regs.clear();
    }

    fn flush_mem(&mut self) {
        if self.mem.is_empty() {
            return;
        }
        let mem = std::mem::take(&mut self.mem);
        let bytes = unsafe { std::slice::from_raw_parts(mem.as_ptr().cast::<u8>(), mem.len() * 4) };
        self.write(self.mem_off, bytes);
        self.mem = mem;
        self.mem.clear();
    }

    fn write(&mut self, off: u32, p: &[u8]) {
        if matches!(self.link, Link::Drm(_)) {
            let mut m = drm_borg_mem { data: p.as_ptr() as u64, offset: off, length: p.len() as u32 };
            self.ioctl(drm::MEM_WRITE, &mut m);
        } else {
            let mut b = Vec::with_capacity(9 + p.len());
            b.push(b'M');
            b.extend_from_slice(&off.to_le_bytes());
            b.extend_from_slice(&(p.len() as u32).to_le_bytes());
            b.extend_from_slice(p);
            self.put(&b);
        }
    }

    /// `p` to GPU memory at `off`.
    pub fn mem_write(&mut self, off: u32, p: &[u8]) {
        self.flush_regs();
        self.flush_mem();
        self.write(off, p);
    }

    /// `len` bytes at `off` to the word `v` ('F'); on the DRM node a write.
    pub fn mem_fill(&mut self, off: u32, len: u32, v: u32) {
        self.flush_regs();
        self.flush_mem();
        if matches!(self.link, Link::Drm(_)) {
            let words = vec![v; len as usize / 4];
            self.write(off, unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), len as usize) });
            return;
        }
        let mut b = vec![b'F'];
        for w in [off, len, v] {
            b.extend_from_slice(&w.to_le_bytes());
        }
        self.put(&b);
    }

    /// The 4 KB pages of the 256 MB window Borg wrote since the last call, a bit each ('D').
    pub fn dirty(&mut self) -> Vec<u8> {
        self.flush_regs();
        self.flush_mem();
        let mut b = vec![0u8; 8192];
        self.put(b"D");
        self.get(&mut b);
        b
    }

    /// GPU memory at `off` into `p`.
    pub fn mem_read(&mut self, off: u32, p: &mut [u8]) {
        self.mem_reads(&mut [(off, p)]);
    }

    /// Several reads of GPU memory: over a link all requests go first, so they cost one round trip.
    pub fn mem_reads(&mut self, reads: &mut [(u32, &mut [u8])]) {
        self.flush_regs();
        self.flush_mem();
        if matches!(self.link, Link::Drm(_)) {
            for (off, p) in reads.iter_mut() {
                let mut m = drm_borg_mem { data: p.as_mut_ptr() as u64, offset: *off, length: p.len() as u32 };
                self.ioctl(drm::MEM_READ, &mut m);
            }
            return;
        }
        // The link holds 1 KB of commands while the board answers a read: more deadlock.
        for batch in reads.chunks_mut(LINK_READS) {
            let mut q = Vec::with_capacity(9 * batch.len());
            for (off, p) in batch.iter() {
                q.push(b'm');
                q.extend_from_slice(&off.to_le_bytes());
                q.extend_from_slice(&(p.len() as u32).to_le_bytes());
            }
            self.put(&q);
            for (_, p) in batch.iter_mut() {
                self.get(p);
            }
        }
    }
}

impl crate::list::Gpu for Dev {
    fn reg(&mut self, off: u32, v: u32) {
        self.flush_mem();
        if self.regs.len() == MAX_REGS {
            self.flush_regs();
        }
        self.regs.push([off, v]);
    }

    fn read(&mut self, off: u32) -> u32 {
        self.flush_mem();
        self.flush_regs();
        if matches!(self.link, Link::Drm(_)) {
            let mut rr = drm_borg_reg_read { offset: off, value: 0 };
            self.ioctl(drm::REG_READ, &mut rr);
            return rr.value;
        }
        let mut q = vec![b'R'];
        q.extend_from_slice(&off.to_le_bytes());
        self.put(&q);
        let mut r = [0u8; 4];
        self.get(&mut r);
        u32::from_le_bytes(r)
    }

    /// On a simulator the poll runs there, which answers after a bounded time even if the value
    /// never comes.
    fn wait(&mut self, off: u32, mask: u32, val: u32) -> u32 {
        if matches!(self.link, Link::Drm(_)) {
            // A failed ioctl ends the wait, as a lost simulator does.
            loop {
                let v = self.read(off);
                if v & mask == val || !self.ok {
                    return v;
                }
            }
        }
        self.flush_mem();
        self.flush_regs();
        let mut q = vec![b'P'];
        for w in [off, mask, val] {
            q.extend_from_slice(&w.to_le_bytes());
        }
        // A poll answers after 200M cycles even when the condition does not hold yet.
        loop {
            self.put(&q);
            let mut r = [0u8; 4];
            self.get(&mut r);
            let v = u32::from_le_bytes(r);
            if v & mask == val || !self.ok {
                return v;
            }
            if crate::hw::env_set("BORGVK_PERF") {
                // A long run, every 200M cycles: total, frag, flush, stall, dma so far.
                let p: Vec<u32> = (0..5).map(|j| self.read(std::mem::offset_of!(borg_gpu_t, perf_total) as u32 + 4 * j)).collect();
                let issue = std::mem::offset_of!(borg_gpu_t, perf_issue) as u32;
                self.reg(issue, 15);
                // core, flusher, DMA busy; binner clearing, state; tile sequencer, walker bin, walker, list states
                eprintln!("[perf] waiting {p:?} fsm {:#x}", self.read(issue));
            }
        }
    }

    fn mem(&mut self, addr: u32, v: u32) {
        self.flush_regs();
        if !self.mem.is_empty() && (addr != self.mem_off + self.mem.len() as u32 * 4 || self.mem.len() == MAX_MEM) {
            self.flush_mem();
        }
        if self.mem.is_empty() {
            self.mem_off = addr;
        }
        self.mem.push(v);
    }
}

impl Drop for Dev {
    fn drop(&mut self) {
        if let Link::Pipe(to, _, c) = &mut self.link {
            *to = Box::new(std::io::sink());
            let _ = c.wait();
        }
    }
}

// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! The device heap at BORG_DEVICE_BASE on the GPU: one memfd the driver and every simulator
//! process map, or, over the board's USB link (BORG_HW_RAW), the host's side of the board's DDR,
//! synced at each job: what Borg writes comes back only for memory the host has read, mapped or
//! touched (`touch`). Device memory is carved from it in pages, first fit.

use crate::ffi::*;
use crate::util::{ftruncate, mmap};
use std::sync::{Mutex, OnceLock};

const PAGE: usize = 4096;
const BYTES: usize = BORG_DEVICE_BYTES as usize;

pub struct Heap {
    fd: i32,
    base: *mut u8,
    free: Mutex<Vec<(usize, usize)>>, // (offset, length), sorted, never adjacent
    board: Option<Mutex<Board>>,
}

/// What the board holds of the heap: a copy of every page it got, which pages those are, the
/// pages the host reads (mirrored after each job) and the ones Borg wrote that are not yet back.
struct Board {
    copy: *mut u8,
    sent: Vec<u64>,
    mirror: Vec<u64>,
    stale: Vec<u64>,
}

fn bit(v: &[u64], p: usize) -> bool {
    v[p / PAGE / 64] >> (p / PAGE % 64) & 1 != 0
}

fn set(v: &mut [u64], p: usize, on: bool) {
    let (w, b) = (p / PAGE / 64, 1u64 << (p / PAGE % 64));
    v[w] = if on { v[w] | b } else { v[w] & !b };
}

// The mapping lives as long as the process; the free list is behind its lock.
unsafe impl Send for Heap {}
unsafe impl Sync for Heap {}

static HEAP: OnceLock<Option<Heap>> = OnceLock::new();

/// The heap, when the driver renders on the direct simulator or over the board's USB link.
pub fn get() -> Option<&'static Heap> {
    HEAP.get_or_init(|| {
        let set = |v| std::env::var_os(v).is_some_and(|v| !v.is_empty());
        let link = !set("BORGVK_SIM_DIRECT") && set("BORG_HW_RAW");
        if !link && !set("BORGVK_SIM_DIRECT") {
            return None;
        }
        let fd = unsafe { memfd_create(c"borg-device".as_ptr(), 0) };
        if fd < 0 || unsafe { ftruncate(fd, BYTES as _) } < 0 {
            return None;
        }
        let prot = (PROT_READ | PROT_WRITE) as i32;
        let m = unsafe { mmap(std::ptr::null_mut(), BYTES, prot, MAP_SHARED as i32, fd, 0) };
        if m as usize == usize::MAX {
            return None;
        }
        let board = if link {
            let flags = (MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE) as i32;
            let c = unsafe { mmap(std::ptr::null_mut(), BYTES, prot, flags, -1, 0) };
            if c as usize == usize::MAX {
                return None;
            }
            let pages = vec![0; BYTES / PAGE / 64];
            Some(Mutex::new(Board { copy: c.cast(), sent: pages.clone(), mirror: pages.clone(), stale: pages }))
        } else {
            None
        };
        Some(Heap { fd, base: m.cast(), free: Mutex::new(vec![(0, BYTES)]), board })
    })
    .as_ref()
}

impl Heap {
    pub fn fd(&self) -> i32 {
        self.fd
    }

    /// The mapping of GPU address `gpu`, inside the heap.
    pub fn host(&self, gpu: u32) -> Option<*mut u8> {
        let off = gpu.checked_sub(BORG_DEVICE_BASE)? as usize;
        (off < BYTES).then(|| self.base.wrapping_add(off))
    }

    /// `size` bytes of zeros, page aligned: the mapping and the GPU address.
    pub fn alloc(&self, size: usize) -> Option<(*mut u8, u32)> {
        let n = size.max(1).next_multiple_of(PAGE);
        let mut f = self.free.lock().unwrap_or_else(|e| e.into_inner());
        let i = f.iter().position(|&(_, len)| len >= n)?;
        let off = f[i].0;
        if f[i].1 == n {
            f.remove(i);
        } else {
            f[i] = (off + n, f[i].1 - n);
        }
        Some((self.base.wrapping_add(off), BORG_DEVICE_BASE + off as u32))
    }

    /// The allocated (offset, length) ranges.
    fn used(&self) -> Vec<(usize, usize)> {
        let f = self.free.lock().unwrap_or_else(|e| e.into_inner());
        let mut at = 0;
        let mut out = Vec::new();
        for &(o, n) in f.iter().chain(std::iter::once(&(BYTES, 0))) {
            if o > at {
                out.push((at, o - at));
            }
            at = o + n;
        }
        out
    }

    /// Before a job on the board: every allocated page the host changed since the board got it.
    pub fn to_board(&self, d: &mut crate::dev::Dev) {
        let Some(b) = &self.board else { return };
        let mut b = b.lock().unwrap_or_else(|e| e.into_inner());
        let changed = |b: &Board, p: usize| {
            b.sent[p / PAGE / 64] >> (p / PAGE % 64) & 1 == 0
                || unsafe { std::slice::from_raw_parts(self.base.add(p), PAGE) != std::slice::from_raw_parts(b.copy.add(p), PAGE) }
        };
        for (off, len) in self.used() {
            let mut p = off;
            while p < off + len {
                let start = p;
                while p < off + len && changed(&b, p) {
                    p += PAGE;
                }
                if p == start {
                    p += PAGE;
                    continue;
                }
                let run = unsafe { std::slice::from_raw_parts(self.base.add(start), p - start) };
                d.mem_write(BORG_DEVICE_BASE + start as u32, run);
                unsafe { std::ptr::copy_nonoverlapping(run.as_ptr(), b.copy.add(start), run.len()) };
                (start..p).step_by(PAGE).for_each(|q| b.sent[q / PAGE / 64] |= 1 << (q / PAGE % 64));
            }
        }
    }

    /// After a job on the board: the allocated pages Borg wrote, into the heap and the copy where
    /// the host reads them, else marked stale.
    pub fn from_board(&self, d: &mut crate::dev::Dev) {
        let Some(b) = &self.board else { return };
        let mut b = b.lock().unwrap_or_else(|e| e.into_inner());
        let bits = d.dirty();
        let dirty = |p: usize| {
            let page = (BORG_DEVICE_BASE as usize + p) / PAGE % (bits.len() * 8);
            bits[page / 8] >> (page % 8) & 1 != 0
        };
        for (off, len) in self.used() {
            let stale: Vec<usize> = (off..off + len).step_by(PAGE).filter(|&p| dirty(p) && !bit(&b.mirror, p)).collect();
            stale.into_iter().for_each(|p| set(&mut b.stale, p, true));
            self.fetch(d, &mut b, off, len, |b, p| dirty(p) && bit(&b.mirror, p));
        }
    }

    /// The pages of `off..off + len` that `want`, from the board into the heap and the copy, in
    /// one batch of reads.
    fn fetch(&self, d: &mut crate::dev::Dev, b: &mut Board, off: usize, len: usize, want: impl Fn(&Board, usize) -> bool) {
        let mut runs = Vec::new();
        let mut p = off;
        while p < off + len {
            let start = p;
            while p < off + len && want(b, p) {
                p += PAGE;
            }
            if p == start {
                p += PAGE;
            } else {
                runs.push((start, p - start));
            }
        }
        let mut reads: Vec<(u32, &mut [u8])> = runs
            .iter()
            .map(|&(s, n)| (BORG_DEVICE_BASE + s as u32, unsafe { std::slice::from_raw_parts_mut(self.base.add(s), n) }))
            .collect();
        d.mem_reads(&mut reads);
        for &(s, n) in &runs {
            unsafe { std::ptr::copy_nonoverlapping(self.base.add(s), b.copy.add(s), n) };
            (s..s + n).step_by(PAGE).for_each(|q| {
                set(&mut b.sent, q, true);
                set(&mut b.stale, q, false);
            });
        }
    }

    /// The host reads `size` bytes at `gpu` from now on: what Borg wrote there comes back, now
    /// and after every job.
    pub fn touch(&self, gpu: u32, size: usize) {
        let Some(off) = gpu.checked_sub(BORG_DEVICE_BASE).map(|o| o as usize) else { return };
        let Some(board) = &self.board else { return };
        let n = size.max(1).next_multiple_of(PAGE).min(BYTES.saturating_sub(off));
        {
            let b = board.lock().unwrap_or_else(|e| e.into_inner());
            if (off..off + n).step_by(PAGE).all(|p| bit(&b.mirror, p)) {
                return;
            }
        }
        crate::dev::with(|d| {
            let mut b = board.lock().unwrap_or_else(|e| e.into_inner());
            self.fetch(d, &mut b, off, n, |b, p| bit(&b.stale, p));
            (off..off + n).step_by(PAGE).for_each(|p| set(&mut b.mirror, p, true));
        });
    }

    /// Gives back what `alloc` returned; the pages go back to the system, so they read as zeros.
    pub fn free(&self, gpu: u32, size: usize) {
        let (off, n) = ((gpu - BORG_DEVICE_BASE) as usize, size.max(1).next_multiple_of(PAGE));
        unsafe { madvise(self.base.add(off).cast(), n, MADV_REMOVE as i32) };
        if let Some(b) = &self.board {
            let mut b = b.lock().unwrap_or_else(|e| e.into_inner());
            (off..off + n).step_by(PAGE).for_each(|p| (set(&mut b.mirror, p, false), set(&mut b.stale, p, false)).1);
        }
        let mut f = self.free.lock().unwrap_or_else(|e| e.into_inner());
        let i = f.partition_point(|&(o, _)| o < off);
        f.insert(i, (off, n));
        if i + 1 < f.len() && f[i].0 + f[i].1 == f[i + 1].0 {
            f[i].1 += f[i + 1].1;
            f.remove(i + 1);
        }
        if i > 0 && f[i - 1].0 + f[i - 1].1 == f[i].0 {
            f[i - 1].1 += f[i].1;
            f.remove(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_are_reused_and_coalesce() {
        let flags = (MAP_SHARED | MAP_ANONYMOUS | MAP_NORESERVE) as i32;
        let m = unsafe { mmap(std::ptr::null_mut(), BYTES, (PROT_READ | PROT_WRITE) as i32, flags, -1, 0) };
        let h = Heap { fd: -1, base: m.cast(), free: Mutex::new(vec![(0, BYTES)]), board: None };
        let (_, a) = h.alloc(1).unwrap();
        let (_, b) = h.alloc(PAGE + 1).unwrap();
        let (_, c) = h.alloc(PAGE).unwrap();
        assert_eq!((a, b, c), (BORG_DEVICE_BASE, BORG_DEVICE_BASE + 4096, BORG_DEVICE_BASE + 3 * 4096));
        assert!(h.alloc(BYTES).is_none());
        h.free(a, 1);
        h.free(c, PAGE);
        h.free(b, PAGE + 1);
        assert_eq!(*h.free.lock().unwrap(), vec![(0, BYTES)]);
        assert!(h.used().is_empty());
        let (_, d) = h.alloc(2 * PAGE).unwrap();
        let (_, e) = h.alloc(PAGE).unwrap();
        h.free(d, 2 * PAGE);
        assert_eq!(h.used(), vec![(2 * PAGE, PAGE)]);
        h.free(e, PAGE);
        let (p, _) = h.alloc(BYTES).unwrap();
        assert_eq!(unsafe { *p }, 0);
    }
}

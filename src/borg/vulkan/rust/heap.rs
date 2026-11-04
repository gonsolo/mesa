// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! The simulator's device heap: one memfd the driver and every simulator process map, at
//! BORG_DEVICE_BASE on the GPU. Device memory is carved from it in pages, first fit.

use crate::ffi::*;
use crate::util::{ftruncate, mmap};
use std::sync::{Mutex, OnceLock};

const PAGE: usize = 4096;
const BYTES: usize = BORG_DEVICE_BYTES as usize;

pub struct Heap {
    fd: i32,
    base: *mut u8,
    free: Mutex<Vec<(usize, usize)>>, // (offset, length), sorted, never adjacent
}

// The mapping lives as long as the process; the free list is behind its lock.
unsafe impl Send for Heap {}
unsafe impl Sync for Heap {}

static HEAP: OnceLock<Option<Heap>> = OnceLock::new();

/// The heap, when the driver renders on the direct simulator.
pub fn get() -> Option<&'static Heap> {
    HEAP.get_or_init(|| {
        std::env::var_os("BORGVK_SIM_DIRECT").filter(|v| !v.is_empty())?;
        let fd = unsafe { memfd_create(c"borg-device".as_ptr(), 0) };
        if fd < 0 || unsafe { ftruncate(fd, BYTES as _) } < 0 {
            return None;
        }
        let m = unsafe { mmap(std::ptr::null_mut(), BYTES, (PROT_READ | PROT_WRITE) as i32, MAP_SHARED as i32, fd, 0) };
        (m as usize != usize::MAX).then(|| Heap { fd, base: m.cast(), free: Mutex::new(vec![(0, BYTES)]) })
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

    /// Gives back what `alloc` returned; the pages go back to the system, so they read as zeros.
    pub fn free(&self, gpu: u32, size: usize) {
        let (off, n) = ((gpu - BORG_DEVICE_BASE) as usize, size.max(1).next_multiple_of(PAGE));
        unsafe { madvise(self.base.add(off).cast(), n, MADV_REMOVE as i32) };
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
        let h = Heap { fd: -1, base: m.cast(), free: Mutex::new(vec![(0, BYTES)]) };
        let (_, a) = h.alloc(1).unwrap();
        let (_, b) = h.alloc(PAGE + 1).unwrap();
        let (_, c) = h.alloc(PAGE).unwrap();
        assert_eq!((a, b, c), (BORG_DEVICE_BASE, BORG_DEVICE_BASE + 4096, BORG_DEVICE_BASE + 3 * 4096));
        assert!(h.alloc(BYTES).is_none());
        h.free(a, 1);
        h.free(c, PAGE);
        h.free(b, PAGE + 1);
        assert_eq!(*h.free.lock().unwrap(), vec![(0, BYTES)]);
        let (p, _) = h.alloc(BYTES).unwrap();
        assert_eq!(unsafe { *p }, 0);
    }
}

// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

use crate::ffi::*;

pub use crate::ffi::{MESA_LOG_INFO, MESA_LOG_WARN};

#[cfg(not(test))]
pub fn log(level: mesa_log_level, msg: String) {
    let msg = std::ffi::CString::new(msg).unwrap_or_default();
    unsafe { mesa_log(level, c"MESA".as_ptr(), c"%s".as_ptr(), msg.as_ptr()) };
}

#[cfg(test)]
pub fn log(_level: mesa_log_level, _msg: String) {}

// By hand, not through bindgen: glibc's headers redirect these to the *64 names musl lacks. Borg's
// targets are all 64-bit, so off_t is 64 bits either way.
extern "C" {
    pub fn mmap(addr: *mut std::ffi::c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut std::ffi::c_void;
    pub fn ftruncate(fd: i32, len: i64) -> i32;
}

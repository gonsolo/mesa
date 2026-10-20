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

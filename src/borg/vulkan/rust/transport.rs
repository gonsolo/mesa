// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! The packets of a draw being recorded, captured into its stream.

use crate::util::{log, MESA_LOG_WARN};
use std::cell::RefCell;

struct Capture {
    active: bool,
    buf: Vec<u8>,
}

thread_local! {
    static CAPTURE: RefCell<Capture> = const { RefCell::new(Capture { active: false, buf: Vec::new() }) };
}

pub fn emit(pkt: &[u8]) {
    CAPTURE.with(|c| {
        let mut c = c.borrow_mut();
        if !c.active {
            return;
        }
        if c.buf.try_reserve(pkt.len()).is_err() {
            log(MESA_LOG_WARN, format!("borgvk: transport capture of {} bytes failed", c.buf.len() + pkt.len()));
            return;
        }
        c.buf.extend_from_slice(pkt);
    });
}

pub fn capturing() -> bool {
    CAPTURE.with(|c| c.borrow().active)
}

pub fn borgvk_transport_capture_begin() {
    CAPTURE.with(|c| {
        let mut c = c.borrow_mut();
        c.buf.clear();
        c.active = true;
    });
}

/// Ends the capture and returns what was sent.
pub fn capture_end() -> Vec<u8> {
    CAPTURE.with(|c| {
        let mut c = c.borrow_mut();
        c.active = false;
        std::mem::take(&mut c.buf)
    })
}

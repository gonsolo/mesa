// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! The commands of a draw being recorded, captured until it is complete.

use crate::list::Cmd;
use std::cell::RefCell;

struct Capture {
    active: bool,
    buf: Vec<Cmd>,
}

thread_local! {
    static CAPTURE: RefCell<Capture> = const { RefCell::new(Capture { active: false, buf: Vec::new() }) };
}

pub fn emit(c: Cmd) {
    CAPTURE.with(|cap| {
        let mut cap = cap.borrow_mut();
        if cap.active {
            cap.buf.push(c);
        }
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
pub fn capture_end() -> Vec<Cmd> {
    CAPTURE.with(|c| {
        let mut c = c.borrow_mut();
        c.active = false;
        std::mem::take(&mut c.buf)
    })
}

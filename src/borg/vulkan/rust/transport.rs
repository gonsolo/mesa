// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! Where packets go: the capture buffer while a draw is being recorded, else the simulator
//! socket in `BORGVK_SIM_SOCKET`, else the serial port in `BORGVK_SERIAL`.

use crate::ffi::*;
use crate::util::{log, MESA_LOG_INFO, MESA_LOG_WARN};
use std::cell::RefCell;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::net::UnixStream;
use std::sync::Mutex;
use std::time::Duration;

const SERIAL_DEFAULT: &str = "/dev/ttyUSB0";

/// A link opened on first use; a failed open is not retried.
enum Link<T> {
    Untried,
    Failed,
    Open(T),
}

static SERIAL: Mutex<Link<File>> = Mutex::new(Link::Untried);
static SOCKET: Mutex<Link<UnixStream>> = Mutex::new(Link::Untried);

/// A malloc'd buffer, so the C caller that takes it over can free() it.
struct Capture {
    active: bool,
    buf: *mut u8,
    len: usize,
    cap: usize,
}

thread_local! {
    static CAPTURE: RefCell<Capture> =
        const { RefCell::new(Capture { active: false, buf: std::ptr::null_mut(), len: 0, cap: 0 }) };
}

fn raw_serial(f: &File) {
    let fd = f.as_raw_fd();
    unsafe {
        let mut tio: termios = std::mem::zeroed();
        if tcgetattr(fd, &mut tio) == 0 {
            cfmakeraw(&mut tio);
            cfsetispeed(&mut tio, B115200);
            cfsetospeed(&mut tio, B115200);
            tio.c_cflag |= CLOCAL | CREAD;
            tio.c_cflag &= !CRTSCTS;
            tcsetattr(fd, TCSANOW as i32, &tio);
        }
    }
}

fn open_serial() -> Option<File> {
    let path = std::env::var("BORGVK_SERIAL").ok().filter(|p| !p.is_empty());
    let path = path.as_deref().unwrap_or(SERIAL_DEFAULT);
    match OpenOptions::new().write(true).custom_flags(O_NOCTTY as i32).open(path) {
        Ok(f) => {
            raw_serial(&f);
            log(MESA_LOG_INFO, format!("borgvk: streaming MVP to {path} @115200"));
            Some(f)
        }
        Err(_) => {
            log(
                MESA_LOG_WARN,
                format!("borgvk: cannot open serial port {path} (set $BORGVK_SERIAL); frames will not reach the FPGA"),
            );
            None
        }
    }
}

fn open_socket(path: &str) -> Option<UnixStream> {
    match UnixStream::connect(path) {
        Ok(s) => {
            log(MESA_LOG_INFO, format!("borgvk: streaming to sim socket {path}"));
            Some(s)
        }
        Err(e) => {
            log(MESA_LOG_WARN, format!("borgvk: cannot connect to sim socket {path} ({e}); is the viewer running?"));
            None
        }
    }
}

/// Writes `pkt` over the link, opening it first if needed, then gives the receiver 3 ms.
fn send<T: Write>(link: &Mutex<Link<T>>, open: impl FnOnce() -> Option<T>, pkt: &[u8], what: &str, drain: impl Fn(&T)) {
    let mut link = link.lock().unwrap_or_else(|e| e.into_inner());
    if let Link::Untried = *link {
        *link = open().map_or(Link::Failed, Link::Open);
    }
    let Link::Open(w) = &mut *link else {
        return;
    };
    if let Err(e) = w.write_all(pkt) {
        log(MESA_LOG_WARN, format!("borgvk: {what} write failed ({e})"));
        return;
    }
    drain(w);
    std::thread::sleep(Duration::from_millis(3));
}

fn capture(c: &mut Capture, pkt: &[u8]) {
    if c.len + pkt.len() > c.cap {
        let mut ncap = if c.cap > 0 { c.cap } else { 4096 };
        while ncap < c.len + pkt.len() {
            ncap *= 2;
        }
        let nbuf = unsafe { realloc(c.buf.cast(), ncap as _) }.cast::<u8>();
        if nbuf.is_null() {
            log(MESA_LOG_WARN, format!("borgvk: transport capture realloc({ncap}) failed"));
            return;
        }
        (c.buf, c.cap) = (nbuf, ncap);
    }
    unsafe { std::ptr::copy_nonoverlapping(pkt.as_ptr(), c.buf.add(c.len), pkt.len()) };
    c.len += pkt.len();
}

pub fn emit(pkt: &[u8]) {
    if CAPTURE.with(|c| {
        let mut c = c.borrow_mut();
        if c.active {
            capture(&mut c, pkt);
        }
        c.active
    }) {
        return;
    }
    if let Some(path) = std::env::var("BORGVK_SIM_SOCKET").ok().filter(|p| !p.is_empty()) {
        send(&SOCKET, || open_socket(&path), pkt, "sim socket", |_| {});
    } else {
        send(&SERIAL, open_serial, pkt, "serial", |f| unsafe {
            tcdrain(f.as_raw_fd());
        });
    }
}

pub fn capturing() -> bool {
    CAPTURE.with(|c| c.borrow().active)
}

#[no_mangle]
pub extern "C" fn borgvk_transport_capture_begin() {
    CAPTURE.with(|c| {
        let mut c = c.borrow_mut();
        c.len = 0;
        c.active = true;
    });
}

/// Ends the capture and hands its malloc'd buffer (null if nothing was sent) to the caller.
///
/// # Safety
/// `out_len` is null or valid for a write.
#[no_mangle]
pub unsafe extern "C" fn borgvk_transport_capture_end(out_len: *mut usize) -> *mut u8 {
    let (buf, len) = CAPTURE.with(|c| {
        let mut c = c.borrow_mut();
        let taken = (c.buf, c.len);
        *c = Capture { active: false, buf: std::ptr::null_mut(), len: 0, cap: 0 };
        taken
    });
    if let Some(out) = unsafe { out_len.as_mut() } {
        *out = len;
    }
    buf
}

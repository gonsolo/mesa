// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! The driver's vk_sync: a binary flag set on the host, one lock and condition variable for all.

use crate::entrypoint::slice;
use crate::ffi::*;
use std::sync::{Condvar, Mutex};
use std::time::Duration;

#[repr(C)]
struct BinarySync {
    base: vk_sync,
    signaled: bool,
}

static LOCK: Mutex<()> = Mutex::new(());
static COND: Condvar = Condvar::new();

/// # Safety
/// `s` is the base of a `BinarySync`.
unsafe fn get<'a>(s: *mut vk_sync) -> &'a mut BinarySync {
    unsafe { &mut *s.cast() }
}

fn locked() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

unsafe extern "C" fn init(_dev: *mut vk_device, s: *mut vk_sync, initial: u64) -> VkResult {
    unsafe { get(s).signaled = initial != 0 };
    VK_SUCCESS
}

unsafe extern "C" fn finish(_dev: *mut vk_device, _s: *mut vk_sync) {}

unsafe extern "C" fn signal(_dev: *mut vk_device, s: *mut vk_sync, _value: u64) -> VkResult {
    let _g = locked();
    unsafe { get(s).signaled = true };
    COND.notify_all();
    VK_SUCCESS
}

unsafe extern "C" fn reset(_dev: *mut vk_device, s: *mut vk_sync) -> VkResult {
    let _g = locked();
    unsafe { get(s).signaled = false };
    VK_SUCCESS
}

unsafe extern "C" fn move_(_dev: *mut vk_device, dst: *mut vk_sync, src: *mut vk_sync) -> VkResult {
    let _g = locked();
    unsafe { get(dst).signaled = std::mem::take(&mut get(src).signaled) };
    COND.notify_all();
    VK_SUCCESS
}

/// Waits until any or all are signaled, up to `abs_timeout` on the monotonic clock (`u64::MAX`:
/// forever).
unsafe extern "C" fn wait_many(
    _dev: *mut vk_device,
    count: u32,
    waits: *const vk_sync_wait,
    flags: vk_sync_wait_flags,
    abs_timeout: u64,
) -> VkResult {
    let waits = unsafe { slice(waits, count) };
    let any = flags & VK_SYNC_WAIT_ANY != 0;
    let mut g = locked();
    loop {
        let n = waits.iter().filter(|w| unsafe { get(w.sync).signaled }).count();
        if if any { n > 0 } else { n == waits.len() } {
            return VK_SUCCESS;
        }
        if abs_timeout == u64::MAX {
            g = COND.wait(g).unwrap_or_else(|e| e.into_inner());
            continue;
        }
        let now = unsafe { os_time_get_nano() } as u64;
        if now >= abs_timeout {
            return VK_TIMEOUT;
        }
        g = COND.wait_timeout(g, Duration::from_nanos(abs_timeout - now)).unwrap_or_else(|e| e.into_inner()).0;
    }
}

#[no_mangle]
pub static borgvk_sync_type: vk_sync_type = vk_sync_type {
    size: size_of::<BinarySync>(),
    features: VK_SYNC_FEATURE_BINARY
        | VK_SYNC_FEATURE_GPU_WAIT
        | VK_SYNC_FEATURE_GPU_MULTI_WAIT
        | VK_SYNC_FEATURE_WAIT_PENDING
        | VK_SYNC_FEATURE_CPU_WAIT
        | VK_SYNC_FEATURE_CPU_RESET
        | VK_SYNC_FEATURE_CPU_SIGNAL
        | VK_SYNC_FEATURE_WAIT_ANY,
    init: Some(init),
    finish: Some(finish),
    signal: Some(signal),
    reset: Some(reset),
    move_: Some(move_),
    wait_many: Some(wait_many),
    ..unsafe { std::mem::zeroed() }
};

#[cfg(test)]
mod tests {
    use super::*;

    fn sync() -> BinarySync {
        BinarySync { base: unsafe { std::mem::zeroed() }, signaled: false }
    }

    #[test]
    fn signal_move_and_wait() {
        let (mut a, mut b) = (sync(), sync());
        let (pa, pb) = (&mut a.base as *mut vk_sync, &mut b.base as *mut vk_sync);
        let one = [vk_sync_wait { sync: pa, ..unsafe { std::mem::zeroed() } }];
        unsafe {
            init(std::ptr::null_mut(), pa, 0);
            assert_eq!(wait_many(std::ptr::null_mut(), 1, one.as_ptr(), 0, 0), VK_TIMEOUT);
            signal(std::ptr::null_mut(), pa, 1);
            assert_eq!(wait_many(std::ptr::null_mut(), 1, one.as_ptr(), 0, 0), VK_SUCCESS);
            move_(std::ptr::null_mut(), pb, pa);
            assert!(b.signaled && !a.signaled);
        }
    }

    #[test]
    fn a_waiter_wakes_on_signal() {
        let a = Box::leak(Box::new(sync()));
        let pa = &mut a.base as *mut vk_sync as usize;
        let t = std::thread::spawn(move || {
            let w = [vk_sync_wait { sync: pa as *mut vk_sync, ..unsafe { std::mem::zeroed() } }];
            unsafe { wait_many(std::ptr::null_mut(), 1, w.as_ptr(), 0, u64::MAX) }
        });
        std::thread::sleep(Duration::from_millis(20));
        unsafe { signal(std::ptr::null_mut(), pa as *mut vk_sync, 1) };
        assert_eq!(t.join().unwrap(), VK_SUCCESS);
    }
}

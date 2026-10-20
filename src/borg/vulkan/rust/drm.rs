// Copyright © 2026 Andreas Wendleder
// SPDX-License-Identifier: MIT

//! The Borg DRM device's ioctls (drm-uapi/borg_drm.h), through drmIoctl.

use crate::ffi::*;

/// `_IOC` in Linux's generic ioctl encoding, which x86 and RISC-V use, for DRM's type 'd'.
const fn ioc(dir: u64, nr: u32, size: usize) -> u64 {
    dir << 30 | (size as u64) << 16 | (b'd' as u64) << 8 | nr as u64
}
const fn borg(dir: u64, nr: u32, size: usize) -> u64 {
    ioc(dir, DRM_COMMAND_BASE + nr, size)
}
pub const GEM_CREATE: u64 = borg(3, DRM_BORG_GEM_CREATE, size_of::<drm_borg_gem_create>());
pub const GEM_MMAP: u64 = borg(3, DRM_BORG_GEM_MMAP, size_of::<drm_borg_gem_mmap>());
pub const GEM_CLOSE: u64 = ioc(1, 0x09, size_of::<drm_gem_close>());
pub const SHADER: u64 = borg(1, DRM_BORG_SHADER, size_of::<drm_borg_shader>());
pub const BLEND: u64 = borg(1, DRM_BORG_BLEND, size_of::<drm_borg_blend>());
pub const STATE: u64 = borg(1, DRM_BORG_STATE, size_of::<drm_borg_state>());
pub const PUSH: u64 = borg(1, DRM_BORG_PUSH, size_of::<drm_borg_push>());
pub const SETUP: u64 = borg(1, DRM_BORG_SETUP, size_of::<drm_borg_setup>());
pub const SUBMIT: u64 = borg(1, DRM_BORG_SUBMIT, size_of::<drm_borg_submit>());

pub fn ioctl<T>(fd: i32, request: u64, arg: &mut T) -> bool {
    unsafe { drmIoctl(fd, request as _, (arg as *mut T).cast()) >= 0 }
}

pub fn gem_close(fd: i32, handle: u32) {
    ioctl(fd, GEM_CLOSE, &mut drm_gem_close { handle, ..Default::default() });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ioctl_numbers() {
        assert_eq!(GEM_CLOSE, 0x4008_6409);
        assert_eq!(GEM_CREATE, 0xc010_6445);
        assert_eq!(GEM_MMAP, 0xc010_6446);
        assert_eq!(BLEND, 0x4008_644a);
        assert_eq!(STATE, 0x4014_644b);
    }
}

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
pub const REG_WRITES: u64 = borg(1, DRM_BORG_REG_WRITES, size_of::<drm_borg_reg_writes>());
pub const REG_READ: u64 = borg(3, DRM_BORG_REG_READ, size_of::<drm_borg_reg_read>());
pub const MEM_WRITE: u64 = borg(1, DRM_BORG_MEM_WRITE, size_of::<drm_borg_mem>());
pub const MEM_READ: u64 = borg(1, DRM_BORG_MEM_READ, size_of::<drm_borg_mem>());

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
        assert_eq!(REG_WRITES, 0x4010_6441);
        assert_eq!(REG_READ, 0xc008_6442);
        assert_eq!(MEM_WRITE, 0x4010_6443);
        assert_eq!(MEM_READ, 0x4010_6444);
    }
}

/* SPDX-License-Identifier: MIT */
/*
 * Borg GPU DRM uAPI, the part borgvk and libborg_drm_shim.so share with the kernel
 * driver (software/linux/overlay/include/uapi/drm/borg_drm.h): register and memory access,
 * GEM buffers.
 */
#ifndef BORG_DRM_H
#define BORG_DRM_H

#include <stdint.h>

/* Allow borg_drm.h to be included before or after xf86drm.h. */
#ifndef DRM_IOWR
#  include <sys/ioctl.h>
#  define BORG_DRM_IOCTL_BASE 'd'
#  define DRM_IOCTL_BASE  BORG_DRM_IOCTL_BASE
#  define DRM_COMMAND_BASE 0x40
#  define DRM_IOW(nr, type)   _IOW(BORG_DRM_IOCTL_BASE, nr, type)
#  define DRM_IOWR(nr, type)  _IOWR(BORG_DRM_IOCTL_BASE, nr, type)
#endif

#define DRM_BORG_REG_WRITES   0x01
#define DRM_BORG_REG_READ     0x02
#define DRM_BORG_MEM_WRITE    0x03
#define DRM_BORG_MEM_READ     0x04
#define DRM_BORG_GEM_CREATE   0x05
#define DRM_BORG_GEM_MMAP     0x06

/* `writes` points to `count` pairs of uint32_t {offset, value}. */
struct drm_borg_reg_writes {
   uint64_t writes;
   uint32_t count;
   uint32_t pad;
};

struct drm_borg_reg_read {
   uint32_t offset;
   uint32_t value;
};

/* `length` bytes at GPU memory `offset` from or to `data`. */
struct drm_borg_mem {
   uint64_t data;
   uint32_t offset;
   uint32_t length;
};

/* Allocate a GEM buffer object backed by anonymous memory. */
struct drm_borg_gem_create {
   uint64_t size;     /* in:  allocation size in bytes           */
   uint32_t handle;   /* out: GEM handle                         */
   uint32_t pad;
};

/* Get the mmap offset for a GEM handle. Pass offset to mmap(drm_fd). */
struct drm_borg_gem_mmap {
   uint32_t handle;   /* in:  GEM handle                         */
   uint32_t pad;
   uint64_t offset;   /* out: mmap offset                        */
};

#define DRM_IOCTL_BORG_GEM_CREATE \
   DRM_IOWR(DRM_COMMAND_BASE + DRM_BORG_GEM_CREATE, struct drm_borg_gem_create)
#define DRM_IOCTL_BORG_GEM_MMAP \
   DRM_IOWR(DRM_COMMAND_BASE + DRM_BORG_GEM_MMAP,   struct drm_borg_gem_mmap)

#endif /* BORG_DRM_H */

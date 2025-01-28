/*
 * Copyright 2024 Andreas Wendleder
 * All Rights Reserved.
 *
 */

#ifndef __BORG_DRM_H__
#define __BORG_DRM_H__

#include "drm.h"

#if defined(__cplusplus)
extern "C" {
#endif

struct drm_borg_gem_info {
        __u32 handle;
        __u64 size;
        __u64 map_handle;
};

struct drm_borg_gem_new {
        struct drm_borg_gem_info info;
};

struct drm_borg_getparam {
        __u64 param;
        __u64 value;
};

struct drm_borg_vm_bind {
   __u32 op_count;
   __u64 op_ptr;
};

struct drm_borg_vm_bind_op {
   __u32 op;
#define DRM_BORG_VM_BIND_OP_MAP 0x0
#define DRM_BORG_VM_BIND_OP_UNMAP 0x1
   __u64 addr;
   __u64 range;
};

struct drm_borg_vm_init {
        __u64 kernel_managed_addr;
        __u64 kernel_managed_size;
};


#define DRM_BORG_VM_INIT            0x10
#define DRM_BORG_VM_BIND            0x11
#define DRM_BORG_GEM_NEW            0x40
#define DRM_BORG_GETPARAM           0x51

#define DRM_IOCTL_BORG_GEM_NEW            DRM_IOWR(DRM_COMMAND_BASE + DRM_BORG_GEM_NEW, struct drm_borg_gem_new)
#define DRM_IOCTL_BORG_GETPARAM           DRM_IOWR(DRM_COMMAND_BASE + DRM_BORG_GETPARAM, struct drm_borg_getparam)
#define DRM_IOCTL_BORG_VM_BIND            DRM_IOWR(DRM_COMMAND_BASE + DRM_BORG_VM_BIND, struct drm_borg_vm_bind)
#define DRM_IOCTL_BORG_VM_INIT            DRM_IOWR(DRM_COMMAND_BASE + DRM_BORG_VM_INIT, struct drm_borg_vm_init)

#define BORG_GETPARAM_STATUS        0


#if defined(__cplusplus)
}
#endif

#endif


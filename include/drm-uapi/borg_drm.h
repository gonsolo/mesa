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

#define DRM_BORG_GEM_NEW            0x40

#define DRM_IOCTL_BORG_GEM_NEW            DRM_IOWR(DRM_COMMAND_BASE + DRM_BORG_GEM_NEW, struct drm_borg_gem_new)

#if defined(__cplusplus)
}
#endif

#endif


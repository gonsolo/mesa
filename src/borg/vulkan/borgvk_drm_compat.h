/* SPDX-License-Identifier: MIT */
/* libdrm for the old GEM uAPI; a build without it (the board) has none of that path. */
#ifndef BORGVK_DRM_COMPAT_H
#define BORGVK_DRM_COMPAT_H

#ifdef HAVE_LIBDRM
#include <xf86drm.h>
#else
#include <errno.h>
#include "drm-uapi/drm.h"
typedef struct { int version_major; int version_minor; char *name; } *drmVersionPtr;
static inline int drmIoctl(int fd, unsigned long request, void *arg) { (void)fd; (void)request; (void)arg; errno = ENOSYS; return -1; }
static inline drmVersionPtr drmGetVersion(int fd) { (void)fd; return NULL; }
static inline void drmFreeVersion(drmVersionPtr v) { (void)v; }
#endif

#endif

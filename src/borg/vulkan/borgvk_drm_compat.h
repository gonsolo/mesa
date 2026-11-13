/* SPDX-License-Identifier: MIT */
/* libdrm, or for a static build without it (the board) the few calls borgvk makes, on the kernel's
 * ioctls directly. */
#ifndef BORGVK_DRM_COMPAT_H
#define BORGVK_DRM_COMPAT_H

#ifdef HAVE_LIBDRM
#include <xf86drm.h>
#else
#include <errno.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include "drm-uapi/drm.h"
typedef struct { int version_major; int version_minor; char *name; } *drmVersionPtr;
static inline int drmIoctl(int fd, unsigned long request, void *arg)
{
   int r;
   do
      r = ioctl(fd, request, arg);
   while (r == -1 && (errno == EINTR || errno == EAGAIN));
   return r;
}
/* The driver's name only: the first call gives its length, the second the name. */
static inline drmVersionPtr drmGetVersion(int fd)
{
   struct drm_version v = { 0 };
   if (drmIoctl(fd, DRM_IOCTL_VERSION, &v))
      return NULL;
   drmVersionPtr p = calloc(1, sizeof(*p));
   char *name = calloc(1, v.name_len + 1);
   if (!p || !name) {
      free(p);
      free(name);
      return NULL;
   }
   v.name = name;
   v.date_len = v.desc_len = 0;
   if (drmIoctl(fd, DRM_IOCTL_VERSION, &v)) {
      free(p);
      free(name);
      return NULL;
   }
   p->version_major = v.version_major;
   p->version_minor = v.version_minor;
   p->name = name;
   return p;
}
static inline void drmFreeVersion(drmVersionPtr v)
{
   if (v)
      free(v->name);
   free(v);
}
#endif

#endif

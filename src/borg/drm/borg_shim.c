/* SPDX-License-Identifier: MIT */
/*
 * Borg GPU drm-shim: a borg render node for the host, so borgvk finds a DRM device
 * without the kernel driver. GEM_CREATE, GEM_MMAP and GEM_CLOSE back each BO with
 * anonymous memory (shim_bo), mapped by borgvk as on the board.
 *
 * Usage:
 *   LD_PRELOAD=libborg_drm_shim.so VK_DRIVER_FILES=.../borg_devenv_icd.json <app>
 */
#include "drm-shim/drm_shim.h"
#include "drm-uapi/borg_drm.h"

#include <errno.h>
#include <stdlib.h>
#include <sys/mman.h>

/* ---- drm-shim ioctl handlers ------------------------------------------ */

/* drm_shim_bo_init() allocates virtual address space in shim_device.mem_fd
 * but does NOT set bo->map.  We mmap the memfd ourselves; driver_bo_free unmaps
 * on GEM_CLOSE. */
static void
borg_bo_free(struct shim_bo *bo)
{
   if (bo->map && bo->map != MAP_FAILED)
      munmap(bo->map, bo->size);
}

static int
borg_ioctl_gem_create(int fd, unsigned long req, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_borg_gem_create *c = arg;
   size_t size = (size_t)(c->size ? c->size : 1);

   struct shim_bo *bo = calloc(1, sizeof(*bo));
   if (!bo) return -ENOMEM;

   int ret = drm_shim_bo_init(bo, size);
   if (ret) { free(bo); return ret; }

   bo->map = mmap(NULL, size, PROT_READ | PROT_WRITE,
                  MAP_SHARED, shim_device.mem_fd, (off_t)bo->mem_addr);
   if (bo->map == MAP_FAILED) {
      bo->map = NULL;
      drm_shim_bo_put(bo);
      return -ENOMEM;
   }

   c->handle = drm_shim_bo_get_handle(shim_fd, bo);
   drm_shim_bo_put(bo);
   return 0;
}

static int
borg_ioctl_gem_mmap(int fd, unsigned long req, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_borg_gem_mmap *m = arg;
   struct shim_bo *bo = drm_shim_bo_lookup(shim_fd, (int)m->handle);
   if (!bo) return -ENOENT;

   m->offset = drm_shim_bo_get_mmap_offset(shim_fd, bo);
   drm_shim_bo_put(bo);
   return 0;
}

static int (*driver_ioctls[])(int, unsigned long, void *) = {
   [DRM_BORG_GEM_CREATE] = borg_ioctl_gem_create,
   [DRM_BORG_GEM_MMAP]   = borg_ioctl_gem_mmap,
};

/* ---- drm_shim_driver_init — called once on library load --------------- */

bool drm_shim_driver_prefers_first_render_node = true;

void
drm_shim_driver_init(void)
{
   shim_device.bus_type           = DRM_BUS_PLATFORM;
   shim_device.driver_name        = "borg";
   shim_device.driver_ioctls      = driver_ioctls;
   shim_device.driver_ioctl_count = ARRAY_SIZE(driver_ioctls);
   shim_device.driver_bo_free     = borg_bo_free;
   shim_device.version_major      = 1;
   shim_device.version_minor      = 0;
   shim_device.version_patchlevel = 0;
}

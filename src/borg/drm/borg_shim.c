/* SPDX-License-Identifier: MIT */
/*
 * Borg GPU drm-shim — makes the borgvk Mesa driver work without a real kernel
 * driver.  Intercepts DRM ioctls via Mesa's drm-shim framework:
 *
 *   BORG_GEM_CREATE / BORG_GEM_MMAP / GEM_CLOSE:
 *       Back each BO with anonymous memory (malloc-equivalent via shim_bo).
 *       The borgvk driver mmap()s the BO exactly as it will on real hardware.
 *
 *   BORG_SETUP:
 *       Read geometry from the UBO BO (at BORG_UBO_POS/ATTR_BYTE_OFFSET) and
 *       texture from the image BO (RGBA8).  Dedup vertices, downsample texture
 *       rows, ship everything over serial.  A sentinel file in /tmp prevents
 *       re-upload across restarts within the same FPGA power cycle.
 *
 *   BORG_SUBMIT:
 *       Read the 16-float MVP from byte 0 of the UBO BO and ship it as a 0xAD
 *       serial packet.
 *
 * Usage:
 *   LD_PRELOAD=libborg_drm_shim.so VK_DRIVER_FILES=.../borg_devenv_icd.json vkcube
 *
 * When the real Linux kernel driver exists, remove this file; nothing in
 * libvulkan_borg.so needs to change.
 */
#include "drm-shim/drm_shim.h"
#include "drm-uapi/borg_drm.h"
#include "borg_serial.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

/* ---- Sentinel (skip upload if FPGA already has the data) -------------- */

static void
sentinel_path(char *buf, size_t n)
{
   const char *dev = getenv("BORGVK_SERIAL");
   if (!dev || !dev[0]) dev = "/dev/ttyUSB0";
   const char *base = strrchr(dev, '/');
   base = base ? base + 1 : dev;
   snprintf(buf, n, "/tmp/borgvk_%s_setup", base);
}

static int
setup_already_done(void)
{
   if (getenv("BORGVK_FORCE_UPLOAD")) return 0;
   char path[256];
   sentinel_path(path, sizeof(path));
   return access(path, F_OK) == 0;
}

static void
mark_setup_done(void)
{
   char path[256];
   sentinel_path(path, sizeof(path));
   int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
   if (fd >= 0) close(fd);
}

/* ---- Vertex dedup + geometry send ------------------------------------- */

static void
send_geometry(const float *ubo)
{
   const float *pos = (const float *)((const char *)ubo + BORG_UBO_POS_BYTE_OFFSET);
   const float *att = (const float *)((const char *)ubo + BORG_UBO_ATTR_BYTE_OFFSET);

   float verts[BORG_GEOM_MAX_VERTS * 3];
   uint8_t idx[BORG_UBO_NUM_VERTS];
   float uv[BORG_UBO_NUM_VERTS * 2];
   int nverts = 0;

   for (int i = 0; i < BORG_UBO_NUM_VERTS; i++) {
      float x = pos[i*4+0], y = pos[i*4+1], z = pos[i*4+2];
      int u = -1;
      for (int j = 0; j < nverts; j++) {
         if (verts[j*3+0]==x && verts[j*3+1]==y && verts[j*3+2]==z) { u=j; break; }
      }
      if (u < 0) {
         if (nverts >= BORG_GEOM_MAX_VERTS) return;
         u = nverts++;
         verts[u*3+0]=x; verts[u*3+1]=y; verts[u*3+2]=z;
      }
      idx[i] = (uint8_t)u;
      uv[i*2+0] = att[i*4+0];
      uv[i*2+1] = att[i*4+1];
   }

   borg_serial_send_geom(verts, nverts, idx, uv, BORG_UBO_NUM_VERTS / 3);
}

/* ---- Texture downsample + send ---------------------------------------- */

static void
send_texture(const uint8_t *pixels, uint32_t sw, uint32_t sh)
{
   if (!sw || !sh) return;
   uint32_t pitch = sw * 4;
   int sxs = (int)(sw / BORG_TEX_DIM); if (sxs < 1) sxs = 1;
   int sys = (int)(sh / BORG_TEX_DIM); if (sys < 1) sys = 1;

   for (int dy = 0; dy < BORG_TEX_DIM; dy++) {
      float row[BORG_TEX_DIM * 3];
      for (int dx = 0; dx < BORG_TEX_DIM; dx++) {
         uint32_t r=0, g=0, b=0, n=0;
         for (int oy = 0; oy < sys; oy++) {
            uint32_t sy = (uint32_t)dy * sys + oy;
            if (sy >= sh) break;
            for (int ox = 0; ox < sxs; ox++) {
               uint32_t sx = (uint32_t)dx * sxs + ox;
               if (sx >= sw) break;
               const uint8_t *p = pixels + sy * pitch + sx * 4;
               r += p[0]; g += p[1]; b += p[2]; n++;
            }
         }
         if (!n) n = 1;
         row[dx*3+0] = (float)r / n / 255.0f;
         row[dx*3+1] = (float)g / n / 255.0f;
         row[dx*3+2] = (float)b / n / 255.0f;
      }
      borg_serial_send_tex_row(dy, row);
   }
}

/* ---- drm-shim ioctl handlers ------------------------------------------ */

/* drm_shim_bo_init() allocates virtual address space in shim_device.mem_fd
 * but does NOT set bo->map.  We mmap the memfd ourselves so SETUP/SUBMIT can
 * read the BO contents directly.  driver_bo_free unmaps on GEM_CLOSE. */
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

   /* Map the backing store so SETUP/SUBMIT can read CPU-written data. */
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

static int
borg_ioctl_setup(int fd, unsigned long req, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_borg_setup *s = arg;

   if (setup_already_done()) {
      fprintf(stderr, "borg-shim: skipping upload (sentinel present); "
              "set BORGVK_FORCE_UPLOAD=1 to re-upload\n");
      return 0;
   }

   struct shim_bo *ubo_bo = drm_shim_bo_lookup(shim_fd, (int)s->ubo_handle);
   if (!ubo_bo) return -ENOENT;
   struct shim_bo *tex_bo = drm_shim_bo_lookup(shim_fd, (int)s->tex_handle);
   if (!tex_bo) { drm_shim_bo_put(ubo_bo); return -ENOENT; }

   fprintf(stderr, "borg-shim: uploading geometry + texture...\n");
   send_geometry((const float *)ubo_bo->map);
   send_texture((const uint8_t *)tex_bo->map + s->tex_offset,
                s->tex_width, s->tex_height);
   mark_setup_done();
   fprintf(stderr, "borg-shim: upload complete\n");

   drm_shim_bo_put(tex_bo);
   drm_shim_bo_put(ubo_bo);
   return 0;
}

static int
borg_ioctl_submit(int fd, unsigned long req, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_borg_submit *s = arg;

   struct shim_bo *bo = drm_shim_bo_lookup(shim_fd, (int)s->ubo_handle);
   if (!bo) return -ENOENT;

   borg_serial_send_mvp((const float *)bo->map);
   drm_shim_bo_put(bo);
   return 0;
}

static int (*driver_ioctls[])(int, unsigned long, void *) = {
   [DRM_BORG_GEM_CREATE] = borg_ioctl_gem_create,
   [DRM_BORG_GEM_MMAP]   = borg_ioctl_gem_mmap,
   [DRM_BORG_SETUP]      = borg_ioctl_setup,
   [DRM_BORG_SUBMIT]     = borg_ioctl_submit,
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

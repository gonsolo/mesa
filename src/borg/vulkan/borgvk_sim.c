/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 *
 * borgvk_sim.c — the persistent simulator behind the driver.
 *
 * One `direct_sim --serve` process per driver process stands in for the GPU: it keeps
 * its state between passes, takes wire packets on a pipe and shares its memory with the
 * driver, so attachments go in and come back by copying memory instead of through a
 * process per draw. The application's images stay the truth: a pass copies its
 * attachments into GPU memory, every draw loads them (and leaves alone the tiles it does
 * not reach), and the pixels the pass changed are copied back.
 */
#include "borgvk_private.h"

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include "util/format/u_format.h"
#include "util/simple_mtx.h"
#include "vk_format.h"
#include "vk_log.h"

#define SIM_MEM_BYTES (1u << 25)   /* simulation/direct/direct_sim.h DirectSim::MEM_BYTES */

static struct {
   simple_mtx_t lock;
   pid_t pid;
   int to, from, memfd;
   uint8_t *mem;
   uint32_t w, h;
   uint32_t fb, zb, sb, heap;   /* byte addresses, from the server */
} sim = { .lock = SIMPLE_MTX_INITIALIZER, .to = -1, .from = -1 };

static bool
write_all(int fd, const uint8_t *p, size_t n)
{
   while (n) {
      ssize_t r = write(fd, p, n);
      if (r < 0 && errno == EINTR)
         continue;
      if (r <= 0)
         return false;
      p += r;
      n -= (size_t)r;
   }
   return true;
}

static bool
read_all(int fd, uint8_t *p, size_t n)
{
   while (n) {
      ssize_t r = read(fd, p, n);
      if (r < 0 && errno == EINTR)
         continue;
      if (r <= 0)
         return false;
      p += r;
      n -= (size_t)r;
   }
   return true;
}

static void
sim_stop(void)
{
   if (sim.to >= 0)
      close(sim.to);
   if (sim.from >= 0)
      close(sim.from);
   if (sim.pid > 0)
      waitpid(sim.pid, NULL, 0);
   sim.to = sim.from = -1;
   sim.pid = 0;
   sim.w = sim.h = 0;
}

static bool
sim_start(const char *bin)
{
   if (sim.pid > 0)
      return true;
   if (!sim.mem) {
      int mfd = memfd_create("borg-dram", 0);
      if (mfd < 0 || ftruncate(mfd, SIM_MEM_BYTES) < 0)
         return false;
      sim.mem = mmap(NULL, SIM_MEM_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, mfd, 0);
      if (sim.mem == MAP_FAILED) {
         sim.mem = NULL;
         close(mfd);
         return false;
      }
      sim.memfd = mfd;   /* stays open, and inheritable, for every server this process starts */
   }
   char memfd_arg[16];
   snprintf(memfd_arg, sizeof(memfd_arg), "%d", sim.memfd);
   int in[2], out[2];
   if (pipe(in) < 0)
      return false;
   if (pipe(out) < 0) {
      close(in[0]); close(in[1]);
      return false;
   }
   signal(SIGPIPE, SIG_IGN);
   pid_t pid = fork();
   if (pid < 0)
      return false;
   if (pid == 0) {
      dup2(in[0], STDIN_FILENO);
      dup2(out[1], STDOUT_FILENO);
      close(in[0]); close(in[1]); close(out[0]); close(out[1]);
      execlp(bin, bin, "--serve", memfd_arg, (char *)NULL);
      _exit(127);
   }
   close(in[0]);
   close(out[1]);
   sim.pid = pid;
   sim.to = in[1];
   sim.from = out[0];
   sim.w = sim.h = 0;
   return true;
}

static bool
sim_set_size(uint32_t w, uint32_t h)
{
   if (sim.w == w && sim.h == h)
      return true;
   uint8_t cmd[5] = { 0xBC, w & 0xff, w >> 8, h & 0xff, h >> 8 };
   uint8_t r[16];
   if (!write_all(sim.to, cmd, sizeof(cmd)) || !read_all(sim.from, r, sizeof(r)))
      return false;
   uint32_t a[4];
   memcpy(a, r, sizeof(a));
   sim.fb = a[0]; sim.zb = a[1]; sim.sb = a[2]; sim.heap = a[3];
   sim.w = w; sim.h = h;
   return true;
}

/* Colour flush format of an attachment (software/borg FlushFormat as borg_core takes it):
 * 1 R8G8B8A8, 2 B8G8R8A8, 3 RAW32; 0 = R5G6B5 for everything else. */
uint8_t
borgvk_sim_flush_format(VkFormat f)
{
   return f == VK_FORMAT_R8G8B8A8_UNORM ? 1 : f == VK_FORMAT_B8G8R8A8_UNORM ? 2 :
          f == VK_FORMAT_R32_UINT ? 3 : 0;
}

/* Whether the persistent simulator can render to this colour attachment: the direct
 * simulator, and a square power-of-two target up to 256. */
bool
borgvk_sim_serves(const struct borgvk_image *color)
{
   const char *direct = getenv("BORGVK_SIM_DIRECT");
   if (!direct || !direct[0] || !color)
      return false;
   uint32_t d = color->vk.extent.width;
   return d >= 4 && d <= 256 && (d & (d - 1)) == 0 && color->vk.extent.height == d;
}

static struct borgvk_image *
view_image(const struct vk_image_view *view)
{
   return view && view->image ? container_of(view->image, struct borgvk_image, vk) : NULL;
}

/* Byte address inside an attachment of pixel (x, y): 4x4 tiles of `tile_bytes`, row-major
 * tiles, row-major pixels of `px_bytes` in a tile. */
static inline uint32_t
tiled(uint32_t base, uint32_t w, uint32_t x, uint32_t y, uint32_t tile_bytes, uint32_t px_bytes)
{
   return base + ((y >> 2) * (w >> 2) + (x >> 2)) * tile_bytes + ((x & 3) | ((y & 3) << 2)) * px_bytes;
}

static bool
image_backed(const struct borgvk_image *img, uint32_t w, uint32_t h)
{
   if (!img || !img->mem || !img->mem->map || img->vk.extent.width != w || img->vk.extent.height != h)
      return false;
   uint32_t bs = vk_format_get_blocksize(img->vk.format);
   return (uint64_t)w * h * bs * MAX2(img->vk.samples, 1) <= img->size;
}

static inline uint16_t
to565(const float c[4])
{
   uint32_t r = (uint32_t)(CLAMP(c[0], 0.0f, 1.0f) * 31.0f + 0.5f);
   uint32_t g = (uint32_t)(CLAMP(c[1], 0.0f, 1.0f) * 63.0f + 0.5f);
   uint32_t b = (uint32_t)(CLAMP(c[2], 0.0f, 1.0f) * 31.0f + 0.5f);
   return (uint16_t)((r << 11) | (g << 5) | b);
}

static float
unpack_z(enum pipe_format pfmt, const uint8_t *texel)
{
   if (pfmt == PIPE_FORMAT_Z16_UNORM_S8_UINT) {
      uint16_t v;
      memcpy(&v, texel, 2);
      return (float)v / 65535.0f;
   }
   float z = 0;
   util_format_unpack_z_float(pfmt, &z, texel, 1);
   return z;
}

static void
pack_z(enum pipe_format pfmt, uint8_t *texel, float z)
{
   if (pfmt == PIPE_FORMAT_Z16_UNORM_S8_UINT) {
      uint16_t v = (uint16_t)(CLAMP(z, 0.0f, 1.0f) * 65535.0f + 0.5f);
      memcpy(texel, &v, 2);
      return;
   }
   util_format_pack_z_float(pfmt, texel, &z, 1);
}

static uint8_t
unpack_s(enum pipe_format pfmt, const uint8_t *texel)
{
   if (pfmt == PIPE_FORMAT_Z16_UNORM_S8_UINT)
      return texel[2];
   uint8_t s = 0;
   util_format_unpack_s_8uint(pfmt, &s, texel, 1);
   return s;
}

static void
pack_s(enum pipe_format pfmt, uint8_t *texel, uint8_t s)
{
   if (pfmt == PIPE_FORMAT_Z16_UNORM_S8_UINT) {
      texel[2] = s;
      return;
   }
   util_format_pack_s_8uint(pfmt, texel, &s, 1);
}

/* Depth as the GPU stores it for this attachment: D16_UNORM for a 16-bit host depth,
 * D32_SFLOAT for everything else. */
bool
borgvk_sim_depth_is_d32(VkFormat f)
{
   return !(f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_D16_UNORM_S8_UINT);
}

/* Run one pass: `stream` holds its packets (pass, state, draws). The attachments' content
 * goes to the GPU first and what the draws changed comes back. */
VkResult
borgvk_sim_run_pass(const uint8_t *stream, size_t n, struct borgvk_image *color,
                    const struct vk_image_view *depth_view,
                    const struct vk_image_view *stencil_view)
{
   const uint32_t w = color->vk.extent.width, h = color->vk.extent.height;
   struct borgvk_image *zimg = view_image(depth_view), *simg = view_image(stencil_view);
   if (!image_backed(color, w, h))
      return VK_SUCCESS;
   if (zimg && !image_backed(zimg, w, h))
      zimg = NULL;
   if (simg && !image_backed(simg, w, h))
      simg = NULL;

   const uint8_t fmt = borgvk_sim_flush_format(color->vk.format);
   const enum pipe_format cpf = vk_format_to_pipe_format(color->vk.format);
   const uint32_t cbs = vk_format_get_blocksize(color->vk.format);
   const uint32_t cpx = fmt ? 4 : 2;                 /* GPU bytes per colour pixel */
   const bool d32 = zimg && borgvk_sim_depth_is_d32(zimg->vk.format);
   const uint32_t zpx = d32 ? 4 : 2;
   const enum pipe_format zpf = zimg ? vk_format_to_pipe_format(zimg->vk.format) : PIPE_FORMAT_NONE;
   const enum pipe_format spf = simg ? vk_format_to_pipe_format(simg->vk.format) : PIPE_FORMAT_NONE;
   const uint32_t zbs = zimg ? vk_format_get_blocksize(zimg->vk.format) : 0;
   const uint32_t sbs = simg ? vk_format_get_blocksize(simg->vk.format) : 0;
   uint8_t *chost = (uint8_t *)color->mem->map + color->offset;
   uint8_t *zhost = zimg ? (uint8_t *)zimg->mem->map + zimg->offset : NULL;
   uint8_t *shost = simg ? (uint8_t *)simg->mem->map + simg->offset : NULL;
   const size_t npx = (size_t)w * h;

   VkResult result = VK_SUCCESS;
   uint8_t *before = malloc(npx * (4 + 4 + 1));     /* what went in: colour, depth, stencil */
   if (!before)
      return VK_SUCCESS;
   uint8_t *cb = before, *zb = before + npx * 4, *sb = before + npx * 8;

   simple_mtx_lock(&sim.lock);
   if (!sim_start(getenv("BORGVK_SIM_DIRECT")) || !sim_set_size(w, h)) {
      mesa_logw("borgvk: cannot start the simulator");
      sim_stop();
      goto out;
   }

   /* In: the application's images into the GPU's attachments. */
   for (uint32_t y = 0; y < h; y++) {
      for (uint32_t x = 0; x < w; x++) {
         const size_t i = (size_t)y * w + x;
         uint8_t *g = sim.mem + tiled(sim.fb, w, x, y, 16 * cpx, cpx);
         if (fmt) {
            memcpy(g, chost + i * cbs, 4);
         } else {
            float c[4];
            util_format_unpack_rgba(cpf, c, chost + i * cbs, 1);
            uint16_t v = to565(c);
            memcpy(g, &v, 2);
         }
         memcpy(cb + i * 4, g, cpx);
         if (zimg) {
            uint8_t *gz = sim.mem + tiled(sim.zb, w, x, y, 16 * zpx, zpx);
            float z = unpack_z(zpf, zhost + i * zbs);
            if (d32) {
               memcpy(gz, &z, 4);
            } else {
               uint16_t v = (uint16_t)(CLAMP(z, 0.0f, 1.0f) * 65535.0f + 0.5f);
               memcpy(gz, &v, 2);
            }
            memcpy(zb + i * 4, gz, zpx);
         }
         if (simg) {
            uint8_t *gs = sim.mem + tiled(sim.sb, w, x, y, 16, 1);
            *gs = unpack_s(spf, shost + i * sbs);
            sb[i] = *gs;
         }
      }
   }

   const char *keep = getenv("BORGVK_SIM_KEEP");   /* the stream, for replaying it by hand */
   if (keep && keep[0]) {
      FILE *f = fopen(keep, "wb");
      if (f) {
         fwrite(stream, 1, n, f);
         fclose(f);
      }
   }
   uint8_t sync = 0xBD, ack;
   if (!write_all(sim.to, stream, n) || !write_all(sim.to, &sync, 1) || !read_all(sim.from, &ack, 1)) {
      mesa_logw("borgvk: the simulator stopped answering");
      sim_stop();
      goto out;
   }

   /* Out: the pixels the pass changed, into every sample plane of the image. */
   const uint32_t cplanes = MAX2(color->vk.samples, 1);
   uint32_t changed = 0;
   for (uint32_t y = 0; y < h; y++) {
      for (uint32_t x = 0; x < w; x++) {
         const size_t i = (size_t)y * w + x;
         const uint8_t *g = sim.mem + tiled(sim.fb, w, x, y, 16 * cpx, cpx);
         if (memcmp(g, cb + i * 4, cpx) != 0) {
            changed++;
            for (uint32_t p = 0; p < cplanes; p++) {
               uint8_t *d = chost + (p * npx + i) * cbs;
               if (fmt) {
                  memcpy(d, g, 4);
               } else {
                  uint16_t v;
                  memcpy(&v, g, 2);
                  float c[4] = { ((v >> 11) & 31) / 31.0f, ((v >> 5) & 63) / 63.0f, (v & 31) / 31.0f, 1.0f };
                  util_format_pack_rgba(cpf, d, c, 1);
               }
            }
         }
         if (zimg) {
            const uint8_t *gz = sim.mem + tiled(sim.zb, w, x, y, 16 * zpx, zpx);
            if (memcmp(gz, zb + i * 4, zpx) != 0) {
               float z;
               if (d32) {
                  memcpy(&z, gz, 4);
               } else {
                  uint16_t v;
                  memcpy(&v, gz, 2);
                  z = (float)v / 65535.0f;
               }
               for (uint32_t p = 0; p < MAX2(zimg->vk.samples, 1); p++)
                  pack_z(zpf, zhost + (p * npx + i) * zbs, z);
            }
         }
         if (simg) {
            const uint8_t gs = sim.mem[tiled(sim.sb, w, x, y, 16, 1)];
            if (gs != sb[i])
               for (uint32_t p = 0; p < MAX2(simg->vk.samples, 1); p++)
                  pack_s(spf, shost + (p * npx + i) * sbs, gs);
         }
      }
   }
   if (getenv("BORGVK_DEBUG"))
      mesa_logi("borgvk: pass of %zu bytes changed %u of %zu pixels (format %u)", n, changed, npx, fmt);

out:
   simple_mtx_unlock(&sim.lock);
   free(before);
   return result;
}

/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"

#include <errno.h>
#include <pthread.h>
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

#define SIM_MEM_BYTES 0x8000000u
#define SIM_MAX_PARTS 24
#define SIM_PARALLEL_BYTES 65536   /* a pass this large is split over several simulators */

struct sim_srv {
   pid_t pid;
   int to, from, memfd;
   uint8_t *mem;
   uint32_t w, h;
   uint32_t fb, zb, sb, heap;   /* byte addresses, from the server */
   uint32_t att[4];             /* colour attachments 1-3 */
};

static struct {
   simple_mtx_t lock;
   struct sim_srv s[SIM_MAX_PARTS];
} sim = { .lock = SIMPLE_MTX_INITIALIZER };

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
   for (int i = 0; i < SIM_MAX_PARTS; i++) {
      struct sim_srv *S = &sim.s[i];
      if (S->to > 0)
         close(S->to);
      if (S->from > 0)
         close(S->from);
      if (S->pid > 0)
         waitpid(S->pid, NULL, 0);
      S->to = S->from = 0;
      S->pid = 0;
      S->w = S->h = 0;
   }
}

struct feed {
   int fd;
   const uint8_t *data;
   size_t n;
   const uint8_t *sync;
   bool ok;
};

static void *
feed_server(void *arg)
{
   struct feed *f = arg;
   f->ok = write_all(f->fd, f->data, f->n) && write_all(f->fd, f->sync, 1);
   return NULL;
}

static bool
sim_start(struct sim_srv *S, const char *bin)
{
   if (S->pid > 0)
      return true;
   if (!S->mem) {
      int mfd = memfd_create("borg-dram", 0);
      if (mfd < 0 || ftruncate(mfd, SIM_MEM_BYTES) < 0)
         return false;
      S->mem = mmap(NULL, SIM_MEM_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, mfd, 0);
      if (S->mem == MAP_FAILED) {
         S->mem = NULL;
         close(mfd);
         return false;
      }
      S->memfd = mfd;
   }
   char memfd_arg[16];
   snprintf(memfd_arg, sizeof(memfd_arg), "%d", S->memfd);
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
   S->pid = pid;
   S->to = in[1];
   S->from = out[0];
   S->w = S->h = 0;
   return true;
}

static bool
sim_set_size(struct sim_srv *S, uint32_t w, uint32_t h)
{
   if (S->w == w && S->h == h)
      return true;
   uint8_t cmd[5] = { 0xBC, w & 0xff, w >> 8, h & 0xff, h >> 8 };
   uint8_t r[28];
   if (!write_all(S->to, cmd, sizeof(cmd)) || !read_all(S->from, r, sizeof(r)))
      return false;
   uint32_t a[7];
   memcpy(a, r, sizeof(a));
   S->fb = a[0]; S->zb = a[1]; S->sb = a[2]; S->heap = a[3];
   S->att[1] = a[4]; S->att[2] = a[5]; S->att[3] = a[6];
   S->w = w; S->h = h;
   return true;
}

bool
borgvk_sim_bytes_packed(VkFormat f)
{
   return f == VK_FORMAT_R8G8B8A8_UINT || f == VK_FORMAT_R8G8B8A8_SINT;
}

bool
borgvk_sim_raw32(VkFormat f)
{
   return f == VK_FORMAT_R32_UINT || f == VK_FORMAT_R32_SINT || f == VK_FORMAT_R32_SFLOAT;
}

bool
borgvk_sim_half1(VkFormat f)
{
   return f == VK_FORMAT_R16_SFLOAT;
}

bool
borgvk_sim_half2(VkFormat f)
{
   return f == VK_FORMAT_R16G16_SFLOAT;
}

static bool
packed_info(VkFormat f, uint32_t *opt, unsigned *bytes)
{
   if (f == VK_FORMAT_R8_UNORM || f == VK_FORMAT_R8G8B8A8_UNORM || f == VK_FORMAT_B8G8R8A8_UNORM ||
       borgvk_sim_bytes_packed(f) || borgvk_sim_half1(f) || borgvk_sim_half2(f) || borgvk_sim_raw32(f))
      return false;
   /* UNORM formats with uneven channels, by number (borgc packed_layout). */
   static const struct { VkFormat f; uint8_t id, bytes, alpha; } uneven[] = {
      { VK_FORMAT_A1R5G5B5_UNORM_PACK16, 0, 2, 1 },  { VK_FORMAT_R5G5B5A1_UNORM_PACK16, 1, 2, 1 },
      { VK_FORMAT_B5G5R5A1_UNORM_PACK16, 2, 2, 1 },  { VK_FORMAT_R4G4B4A4_UNORM_PACK16, 3, 2, 1 },
      { VK_FORMAT_B4G4R4A4_UNORM_PACK16, 4, 2, 1 },  { VK_FORMAT_A4R4G4B4_UNORM_PACK16, 5, 2, 1 },
      { VK_FORMAT_A4B4G4R4_UNORM_PACK16, 6, 2, 1 },  { VK_FORMAT_A2B10G10R10_UNORM_PACK32, 7, 4, 1 },
      { VK_FORMAT_A2R10G10B10_UNORM_PACK32, 8, 4, 1 }, { VK_FORMAT_B5G6R5_UNORM_PACK16, 9, 2, 0 },
      { VK_FORMAT_R4G4_UNORM_PACK8, 10, 1, 0 },       { VK_FORMAT_R5G6B5_UNORM_PACK16, 11, 2, 0 },
   };
   for (unsigned i = 0; i < ARRAY_SIZE(uneven); i++) {
      if (uneven[i].f == f) {
         *opt = 0x10u | 3u << 19 | (uneven[i].id & 7u) << 16 | (uneven[i].id >> 3) << 21 | (uneven[i].alpha ? 4u : 0u);
         *bytes = uneven[i].bytes;
         return true;
      }
   }
   const struct util_format_description *d = vk_format_description(f);
   if (!d || d->layout != UTIL_FORMAT_LAYOUT_PLAIN || d->nr_channels < 1 || d->nr_channels > 4)
      return false;
   const unsigned n = d->nr_channels, bits = d->channel[0].size;
   if (bits != 8 && bits != 16 && bits != 32)
      return false;
   for (unsigned i = 0; i < n; i++)
      if (d->channel[i].size != bits)
         return false;
   *bytes = n * bits / 8;
   if (*bytes > 16)
      return false;
   const bool is_float = d->channel[0].type == UTIL_FORMAT_TYPE_FLOAT;
   if (is_float && bits == 16) {
      *opt = 0x8u | (n == 3 ? 0x40u : n == 4 ? 0x84u : 0u);
      return true;
   }
   uint32_t kind;
   if (d->colorspace == UTIL_FORMAT_COLORSPACE_SRGB)
      kind = 2;
   else if (d->channel[0].pure_integer || is_float)
      kind = 3;   /* the bits as they are */
   else if (bits < 32 && d->channel[0].normalized)
      kind = d->channel[0].type == UTIL_FORMAT_TYPE_SIGNED ? 1 : 0;
   else
      return false;
   const bool swap = n >= 3 && d->swizzle[0] == PIPE_SWIZZLE_Z;
   *opt = 0x10u | kind << 16 | (bits == 8 ? 0u : bits == 16 ? 1u : 2u) << 19 | (n - 1) << 21 | swap << 23 | (n == 4 ? 4u : 0u);
   return true;
}

uint32_t
borgvk_sim_packed_opt(VkFormat f)
{
   uint32_t opt;
   unsigned bytes;
   return packed_info(f, &opt, &bytes) ? opt : 0;
}

uint8_t
borgvk_sim_flush_format(VkFormat f)
{
   uint32_t opt;
   unsigned bytes;
   if (packed_info(f, &opt, &bytes))
      return bytes == 1 ? 5 : bytes == 2 ? 4 : bytes <= 4 ? 3 : bytes <= 8 ? 6 : 7;
   return f == VK_FORMAT_R8_UNORM ? 5 : borgvk_sim_half1(f) ? 4 :
          f == VK_FORMAT_R8G8B8A8_UNORM ? 1 : f == VK_FORMAT_B8G8R8A8_UNORM ? 2 :
          borgvk_sim_raw32(f) || borgvk_sim_bytes_packed(f) || borgvk_sim_half2(f) ? 3 : 0;
}

bool
borgvk_sim_serves(const struct borgvk_image *color, bool has_depth_stencil)
{
   const char *direct = getenv("BORGVK_SIM_DIRECT");
   if (!direct || !direct[0] || !color)
      return false;
   uint32_t w = color->vk.extent.width, h = color->vk.extent.height;
   const uint32_t max_dim = has_depth_stencil ? 1024 : 4096;
   return w >= 1 && w <= max_dim && h >= 1 && h <= max_dim;
}

static struct borgvk_image *
view_image(const struct vk_image_view *view)
{
   return view && view->image ? container_of(view->image, struct borgvk_image, vk) : NULL;
}

/* A RAW128 pixel is two 8-byte halves, 128 bytes apart in its tile (two slices). */
static inline void
px_get(const uint8_t *g, uint32_t cpx, uint8_t *out)
{
   if (cpx == 16) {
      memcpy(out, g, 8);
      memcpy(out + 8, g + 128, 8);
   } else {
      memcpy(out, g, cpx);
   }
}

static inline void
px_put(uint8_t *g, uint32_t cpx, const uint8_t *in)
{
   if (cpx == 16) {
      memcpy(g, in, 8);
      memcpy(g + 128, in + 8, 8);
   } else {
      memcpy(g, in, cpx);
   }
}

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

bool
borgvk_sim_depth_is_d32(VkFormat f)
{
   return !(f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_D16_UNORM_S8_UINT);
}

VkResult
borgvk_sim_run_pass(const uint8_t *stream, size_t n, struct borgvk_image *color,
                    struct borgvk_image *const *extra, uint32_t nextra,
                    const struct vk_image_view *depth_view,
                    const struct vk_image_view *stencil_view,
                    const struct borgvk_writeback *wb, uint32_t nwb)
{
   struct borgvk_image *zimg = view_image(depth_view), *simg = view_image(stencil_view);
   /* Depth only: no colour attachment, nothing of it goes in or comes back. */
   const struct borgvk_image *size = color ? color : zimg ? zimg : simg;
   if (!size)
      return VK_SUCCESS;
   const uint32_t w = size->vk.extent.width, h = size->vk.extent.height;
   const uint32_t hp = (h + 3) & ~3u;
   const uint32_t wp = (w + 3) & ~3u;
   if (color && !image_backed(color, w, h))
      return VK_SUCCESS;
   if (zimg && !image_backed(zimg, w, h))
      zimg = NULL;
   if (simg && !image_backed(simg, w, h))
      simg = NULL;

   for (uint32_t a = 0; a < nextra; a++)
      if (!image_backed(extra[a], w, h))
         return VK_SUCCESS;
   /* Several colour attachments are all one byte per pixel (RAW8). */
   const uint8_t fmt = nextra ? 5 : color ? borgvk_sim_flush_format(color->vk.format) : 0;
   const enum pipe_format cpf = color ? vk_format_to_pipe_format(color->vk.format) : PIPE_FORMAT_NONE;
   const uint32_t cbs = color ? vk_format_get_blocksize(color->vk.format) : 0;
   const uint32_t cpx = fmt == 5 ? 1 : fmt == 4 ? 2 : fmt == 6 ? 8 : fmt == 7 ? 16 : fmt ? 4 : 2;  /* GPU bytes per colour pixel */
   const bool d32 = zimg && borgvk_sim_depth_is_d32(zimg->vk.format);
   const uint32_t zpx = d32 ? 4 : 2;
   const enum pipe_format zpf = zimg ? vk_format_to_pipe_format(zimg->vk.format) : PIPE_FORMAT_NONE;
   const enum pipe_format spf = simg ? vk_format_to_pipe_format(simg->vk.format) : PIPE_FORMAT_NONE;
   const uint32_t zbs = zimg ? vk_format_get_blocksize(zimg->vk.format) : 0;
   const uint32_t sbs = simg ? vk_format_get_blocksize(simg->vk.format) : 0;
   uint8_t *chost = color ? (uint8_t *)color->mem->map + color->offset : NULL;
   uint8_t *zhost = zimg ? (uint8_t *)zimg->mem->map + zimg->offset : NULL;
   uint8_t *shost = simg ? (uint8_t *)simg->mem->map + simg->offset : NULL;
   const size_t npx = (size_t)w * h;

   VkResult result = VK_SUCCESS;
   uint8_t *before = malloc(npx * (16 + 4 + 1 + 3));
   if (!before)
      return VK_SUCCESS;
   uint8_t *cb = before, *zb = before + npx * 16, *sb = before + npx * 20, *eb = before + npx * 21;

   /* A large pass is split into horizontal strips of tile rows, one simulator each. */
   const char *pe = getenv("BORGVK_SERVE_PARTS");
   int np = pe ? atoi(pe) : (int)MIN2(12, MAX2(1, sysconf(_SC_NPROCESSORS_ONLN)));
   np = n >= SIM_PARALLEL_BYTES ? CLAMP(np, 1, SIM_MAX_PARTS) : 1;
   const uint32_t ftiles = hp >> 2;
   const uint32_t rows_per = (ftiles + np - 1) / np;
   np = (int)((ftiles + rows_per - 1) / rows_per);

   simple_mtx_lock(&sim.lock);
   for (int part = 0; part < np; part++) {
      uint8_t sp[5] = { 0xBE, (uint8_t)part, (uint8_t)np, (uint8_t)rows_per, (uint8_t)(rows_per >> 8) };
      if (!sim_start(&sim.s[part], getenv("BORGVK_SIM_DIRECT")) || !sim_set_size(&sim.s[part], wp, hp) ||
          !write_all(sim.s[part].to, sp, sizeof(sp))) {
         mesa_logw("borgvk: cannot start the simulator");
         sim_stop();
         goto out;
      }
   }

   /* In: the application's images into the GPU's attachments. */
   for (int part = 0; part < np; part++) {
   struct sim_srv *S = &sim.s[part];
   for (uint32_t y = part * rows_per * 4; y < MIN2(h, (part + 1) * rows_per * 4); y++) {
      for (uint32_t x = 0; x < w; x++) {
         const size_t i = (size_t)y * w + x;
         uint8_t *g = S->mem + tiled(S->fb, wp, x, y, 16 * cpx, cpx == 16 ? 8 : cpx);
         if (!color) {
         } else if (fmt) {
            uint8_t px[16] = { 0 };
            memcpy(px, chost + i * cbs, MIN2(cbs, cpx));
            px_put(g, cpx, px);
         } else {
            float c[4];
            util_format_unpack_rgba(cpf, c, chost + i * cbs, 1);
            uint16_t v = to565(c);
            memcpy(g, &v, 2);
         }
         px_get(g, cpx, cb + i * 16);
         for (uint32_t a = 0; a < nextra; a++) {
            uint8_t *ge = S->mem + tiled(S->att[a + 1], wp, x, y, 16, 1);
            *ge = ((uint8_t *)extra[a]->mem->map + extra[a]->offset)[i];
            eb[a * npx + i] = *ge;
         }
         if (zimg) {
            uint8_t *gz = S->mem + tiled(S->zb, wp, x, y, 16 * zpx, zpx);
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
            uint8_t *gs = S->mem + tiled(S->sb, wp, x, y, 16, 1);
            *gs = unpack_s(spf, shost + i * sbs);
            sb[i] = *gs;
         }
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
   struct feed feeds[SIM_MAX_PARTS];
   pthread_t th[SIM_MAX_PARTS];
   for (int part = 0; part < np; part++) {
      feeds[part] = (struct feed){ sim.s[part].to, stream, n, &sync, false };
      pthread_create(&th[part], NULL, feed_server, &feeds[part]);
   }
   bool fed = true;
   for (int part = 0; part < np; part++) {
      pthread_join(th[part], NULL);
      fed &= feeds[part].ok;
   }
   if (!fed) {
      mesa_logw("borgvk: the simulator stopped answering");
      sim_stop();
      goto out;
   }
   for (int part = 0; part < np; part++) {
      if (!read_all(sim.s[part].from, &ack, 1)) {
         mesa_logw("borgvk: the simulator stopped answering");
         sim_stop();
         goto out;
      }
   }

   /* Out: the pixels the pass changed, into every sample plane of the image. */
   const uint32_t cplanes = color ? MAX2(color->vk.samples, 1) : 0;
   uint32_t changed = 0;
   for (int part = 0; part < np; part++) {
   struct sim_srv *S = &sim.s[part];
   for (uint32_t y = part * rows_per * 4; y < MIN2(h, (part + 1) * rows_per * 4); y++) {
      for (uint32_t x = 0; x < w; x++) {
         const size_t i = (size_t)y * w + x;
         const uint8_t *g = S->mem + tiled(S->fb, wp, x, y, 16 * cpx, cpx == 16 ? 8 : cpx);
         uint8_t cur[16];
         px_get(g, cpx, cur);
         if (color && memcmp(cur, cb + i * 16, cpx) != 0) {
            changed++;
            for (uint32_t p = 0; p < cplanes; p++) {
               uint8_t *d = chost + (p * npx + i) * cbs;
               if (fmt) {
                  memcpy(d, cur, MIN2(cbs, cpx));
               } else {
                  uint16_t v;
                  memcpy(&v, cur, 2);
                  float c[4] = { ((v >> 11) & 31) / 31.0f, ((v >> 5) & 63) / 63.0f, (v & 31) / 31.0f, 1.0f };
                  util_format_pack_rgba(cpf, d, c, 1);
               }
            }
         }
         for (uint32_t a = 0; a < nextra; a++) {
            const uint8_t ge = S->mem[tiled(S->att[a + 1], wp, x, y, 16, 1)];
            if (ge != eb[a * npx + i]) {
               changed++;
               ((uint8_t *)extra[a]->mem->map + extra[a]->offset)[i] = ge;
            }
         }
         if (zimg) {
            const uint8_t *gz = S->mem + tiled(S->zb, wp, x, y, 16 * zpx, zpx);
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
            const uint8_t gs = S->mem[tiled(S->sb, wp, x, y, 16, 1)];
            if (gs != sb[i])
               for (uint32_t p = 0; p < MAX2(simg->vk.samples, 1); p++)
                  pack_s(spf, shost + (p * npx + i) * sbs, gs);
         }
      }
   }
   }
   for (uint32_t i = 0; i < nwb; i++)
      for (int part = np - 1; part >= 0; part--) {
         const uint8_t *g = sim.s[part].mem + sim.s[part].heap + wb[i].off;
         for (uint32_t k = 0; k < wb[i].size; k++)
            if (g[k] != wb[i].host[k])
               wb[i].host[k] = g[k];
      }
   if (getenv("BORGVK_DEBUG"))
      mesa_logi("borgvk: pass of %zu bytes changed %u of %zu pixels (format %u)", n, changed, npx, fmt);

out:
   simple_mtx_unlock(&sim.lock);
   free(before);
   return result;
}

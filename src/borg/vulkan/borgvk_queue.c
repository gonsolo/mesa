/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "util/format_srgb.h"
#include "borgvk_private.h"

#include "vk_command_buffer.h"
#include "vk_cmd_queue.h"
#include "vk_sync.h"
#include "vk_framebuffer.h"
#include "vk_image.h"
#include "vk_format.h"
#include "util/format/u_format.h"

#include "drm-uapi/borg_drm.h"

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include "borgvk_drm_compat.h"

/* Firmware layout constants (mailbox offsets, max verts/tris, magic). */
#include "software/borg/borg_layout.h"

/* cube.c uniform-buffer layout (floats). */
#define UBO_MVP_FLOATS   16
#define UBO_NUM_VERTS    36                  /* 12 triangles, expanded */
#define UBO_POS_FLOAT0   UBO_MVP_FLOATS      /* position[36][4] */
#define UBO_ATTR_FLOAT0  (UBO_MVP_FLOATS + UBO_NUM_VERTS * 4)  /* attr[36][4] */
#define UBO_MIN_FLOATS   (UBO_ATTR_FLOAT0 + UBO_NUM_VERTS * 4) /* 304 = 1216 B */

/* Locate the bound descriptor set (binding 0 = UBO, binding 1 = texture). */
static struct borgvk_descriptor_set *
find_set(struct vk_queue_submit *submit)
{
   for (uint32_t i = 0; i < submit->command_buffer_count; i++) {
      struct vk_command_buffer *cb = submit->command_buffers[i];
      list_for_each_entry(struct vk_cmd_queue_entry, e, &cb->cmd_queue.cmds, cmd_link) {
         if (e->type != VK_CMD_BIND_DESCRIPTOR_SETS)
            continue;
         const struct vk_cmd_bind_descriptor_sets *b = &e->u.bind_descriptor_sets;
         if (b->descriptor_set_count == 0 || b->descriptor_sets == NULL)
            continue;
         VK_FROM_HANDLE(borgvk_descriptor_set, set, b->descriptor_sets[0]);
         if (set)
            return set;
      }
   }
   return NULL;
}

/* Locate the first render-pass colour attachment image (for sim readback). */
static struct borgvk_image *
find_color_attachment(struct vk_queue_submit *submit)
{
   for (uint32_t ci = 0; ci < submit->command_buffer_count; ci++) {
      struct vk_command_buffer *cb = submit->command_buffers[ci];
      struct borgvk_command_buffer *bcb =
         container_of(cb, struct borgvk_command_buffer, vk);
      if (bcb->color_views[0] &&
          bcb->color_views[0]->image)
         return container_of(bcb->color_views[0]->image, struct borgvk_image, vk);
   }
   return NULL;
}

static void
send_geometry(const float *ubo)
{
   const float *pos = ubo + UBO_POS_FLOAT0;
   const float *att = ubo + UBO_ATTR_FLOAT0;

   float verts[BORGVK_GEOM_MAX_VERTS * 3];
   uint8_t idx[UBO_NUM_VERTS];
   float uv[UBO_NUM_VERTS * 2];
   int nverts = 0;

   for (int i = 0; i < UBO_NUM_VERTS; i++) {
      float x = pos[i*4 + 0], y = pos[i*4 + 1], z = pos[i*4 + 2];
      int u = -1;
      for (int j = 0; j < nverts; j++) {
         if (verts[j*3+0] == x && verts[j*3+1] == y && verts[j*3+2] == z) {
            u = j;
            break;
         }
      }
      if (u < 0) {
         if (nverts >= BORGVK_GEOM_MAX_VERTS)
            return;  /* mesh too complex for the fixed packet (cube fits) */
         u = nverts++;
         verts[u*3+0] = x; verts[u*3+1] = y; verts[u*3+2] = z;
      }
      idx[i] = (uint8_t)u;
      uv[i*2+0] = att[i*4 + 0];
      uv[i*2+1] = att[i*4 + 1];
   }

   borgvk_serial_send_geom(verts, nverts, idx, uv, UBO_NUM_VERTS / 3);
}

static const uint32_t default_sampler[4] = { (2u << 3) | (2u << 6) | (2u << 9), 0, 0, 0 };

static const uint32_t *
sampler_desc(const struct borgvk_sampler *sampler)
{
   return sampler ? sampler->desc : default_sampler;
}

static void
send_texture_row(struct borgvk_image *tex, const struct borgvk_sampler *sampler,
                 int dy)
{
   const uint8_t *base = (const uint8_t *)tex->mem->map + tex->offset;
   uint32_t sw = tex->vk.extent.width, sh = tex->vk.extent.height;
   if (sw == 0 || sh == 0)
      return;
   uint32_t pitch = sw * 4;                 /* RGBA8, linear tiling */
   int sxs = (int)(sw / BORGVK_TEX_DIM); if (sxs < 1) sxs = 1;
   int sys = (int)(sh / BORGVK_TEX_DIM); if (sys < 1) sys = 1;

   uint8_t row[BORGVK_TEX_DIM * 4];
   for (int dx = 0; dx < BORGVK_TEX_DIM; dx++) {
      uint32_t sum[4] = { 0, 0, 0, 0 }, n = 0;
      for (int oy = 0; oy < sys; oy++) {
         uint32_t sy = (uint32_t)dy * sys + oy;
         if (sy >= sh) break;
         for (int ox = 0; ox < sxs; ox++) {
            uint32_t sx = (uint32_t)dx * sxs + ox;
            if (sx >= sw) break;
            const uint8_t *p = base + sy * pitch + sx * 4;
            for (int c = 0; c < 4; c++)
               sum[c] += p[c];
            n++;
         }
      }
      if (n == 0) n = 1;
      for (int c = 0; c < 4; c++)
         row[dx * 4 + c] = (uint8_t)((sum[c] + n / 2) / n);
   }
   borgvk_serial_send_tex_row(dy, row, sampler_desc(sampler));
}

static void
sentinel_path(char *buf, size_t n)
{
   const char *dev = getenv("BORGVK_SERIAL");
   if (!dev || !dev[0])
      dev = "/dev/ttyUSB0";
   const char *base = strrchr(dev, '/');
   base = base ? base + 1 : dev;
   snprintf(buf, n, "/tmp/borgvk_%s_setup", base);
}

static bool
setup_already_done(void)
{
   if (getenv("BORGVK_FORCE_UPLOAD"))
      return false;
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
   if (fd >= 0)
      close(fd);
}

static void
send_blend_state(const struct borgvk_device *device)
{
   if (device->state_valid) {
      if (device->drm_fd >= 0) {
         struct drm_borg_state s;
         memcpy(s.reg, device->state_reg, sizeof(s.reg));
         if (drmIoctl(device->drm_fd, DRM_IOCTL_BORG_STATE, &s) != 0)
            mesa_logw("borgvk: DRM_IOCTL_BORG_STATE failed");
      } else {
         borgvk_serial_send_state(device->state_reg);
      }
   }
   if (device->push_lo < device->push_hi) {
      const uint32_t n = device->push_hi - device->push_lo;
      if (device->drm_fd >= 0) {
         struct drm_borg_push p = { .off_words = device->push_lo, .n_words = n };
         memcpy(p.words, device->push_words + device->push_lo, n * 4);
         if (drmIoctl(device->drm_fd, DRM_IOCTL_BORG_PUSH, &p) != 0)
            mesa_logw("borgvk: DRM_IOCTL_BORG_PUSH failed");
      } else {
         borgvk_serial_send_push_constants(device->push_lo * 4, n * 4,
                                           device->push_words + device->push_lo);
      }
   }
   if (device->vp.valid && device->drm_fd < 0)
      borgvk_serial_send_viewport(&device->vp);
   if (!device->blend_cfg)
      return;
   if (device->drm_fd >= 0) {
      struct drm_borg_blend b = { .cfg = device->blend_cfg,
                                  .constant = device->blend_const };
      if (drmIoctl(device->drm_fd, DRM_IOCTL_BORG_BLEND, &b) != 0)
         mesa_logw("borgvk: DRM_IOCTL_BORG_BLEND failed");
   } else {
      borgvk_serial_send_blend(device->blend_cfg, device->blend_const);
   }
}

static void
upload_shaders(struct borgvk_device *device)
{
   for (int st = 0; st < BORGVK_SHADER_STAGE_COUNT; st++) {
      struct borgvk_shader_blob *b = &device->shader_blob[st];
      if (b->len == 0)
         continue;   /* stage not compiled (app didn't bind it) */
      if (device->drm_fd >= 0) {
         struct drm_borg_shader s = { .stage = (uint32_t)st, .len = b->len };
         memcpy(s.data, b->data, b->len);
         if (drmIoctl(device->drm_fd, DRM_IOCTL_BORG_SHADER, &s) != 0)
            mesa_logw("borgvk: DRM_IOCTL_BORG_SHADER failed (stage %d)", st);
      } else {
         borgvk_serial_send_shader((uint8_t)st, b->data, b->len);
      }
   }
}

/* True when this descriptor set is cube.c's UBO-driven frame (vs a CTS VBO draw). */
static bool
set_is_cube(struct borgvk_descriptor_set *set)
{
   struct borgvk_buffer *ubuf = set->buffers[0];
   if (!ubuf || !ubuf->mem || !ubuf->mem->map)
      return false;
   VkDeviceSize off = ubuf->offset + set->offsets[0];
   if (off >= ubuf->vk.size)
      return false;
   VkDeviceSize avail = ubuf->vk.size - off;
   if (set->ranges[0] != VK_WHOLE_SIZE && set->ranges[0] < avail)
      avail = set->ranges[0];
   return avail / sizeof(float) >= UBO_MIN_FLOATS;
}

static bool
submit_is_cube(struct vk_queue_submit *submit)
{
   struct borgvk_descriptor_set *set = find_set(submit);
   return set && set_is_cube(set);
}

static VkResult
sim_store_rgb(struct borgvk_image *color_img, uint8_t *rgb, bool ok, bool raw32, uint32_t sdim)
{
   const uint32_t width = color_img->vk.extent.width, height = color_img->vk.extent.height;
   const size_t expected = (size_t)sdim * sdim * (raw32 ? 4 : 3);
   const size_t got = ok ? expected : 0;
   const char *ppm = getenv("BORGVK_DUMP_PPM");   /* the rendered frame as it came from the Borg */
   if (ppm && ppm[0] && ok && !raw32) {
      FILE *f = fopen(ppm, "wb");
      if (f) {
         fprintf(f, "P6\n%u %u\n255\n", sdim, sdim);
         fwrite(rgb, 1, expected, f);
         fclose(f);
      }
   }
   bool fits = color_img->mem && color_img->mem->map &&
               (uint64_t)width * height * 4 * MAX2(color_img->vk.samples, 1) <= color_img->size;
   if (rgb && got == expected && !fits) {
      /* leave the image untouched */
   } else if (rgb && got == expected && raw32) {
      memcpy((uint8_t *)color_img->mem->map + color_img->offset, rgb, expected);
   } else if (rgb && got == expected) {
      uint64_t plane_size = (uint64_t)width * height * 4;
      for (uint32_t smp = 0; smp < MAX2(color_img->vk.samples, 1); smp++) {
         uint8_t *dst = (uint8_t *)color_img->mem->map + color_img->offset + smp * plane_size;
         for (uint32_t y = 0; y < height; y++) {
            uint32_t sy = (uint32_t)((uint64_t)(2 * y + 1) * sdim / (2 * (uint64_t)height));
            for (uint32_t x = 0; x < width; x++) {
               uint32_t sx = (uint32_t)((uint64_t)(2 * x + 1) * sdim / (2 * (uint64_t)width));
               const uint8_t *s = rgb + ((size_t)sy * sdim + sx) * 3;
               uint8_t *d = dst + ((size_t)y * width + x) * 4;
               d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 255;
            }
         }
      }
   }
   free(rgb);
   return VK_SUCCESS;
}

static VkResult
sim_run_stream(uint8_t *bytes, size_t nbytes, struct borgvk_image *color_img,
               uint32_t sim_dim)
{
   const char *sim_bin = getenv("BORGVK_SIM");
   const char *sim_fw  = getenv("BORGVK_SIM_FW");
   if (!color_img || !color_img->mem || !color_img->mem->map)
      { free(bytes); return VK_SUCCESS; }
   uint32_t width  = color_img->vk.extent.width;
   uint32_t height = color_img->vk.extent.height;
   if (width == 0 || height == 0)
      { free(bytes); return VK_SUCCESS; }

   uint32_t sdim = sim_dim ? sim_dim : 128;
   const bool raw32 = borgvk_sim_raw32(color_img->vk.format) && width == height;
   if (raw32)
      sdim = width;
   const char *direct = getenv("BORGVK_SIM_DIRECT");
   if (direct && direct[0]) {
      uint32_t d = color_img->vk.extent.width;
      if (d >= 4 && d <= 256 && (d & (d - 1)) == 0 &&
          color_img->vk.extent.height == d)
         sdim = d;
   }
   const char *dim_env = getenv("BORGVK_SIM_DIM");
   if (!sim_dim && dim_env && dim_env[0]) {
      int d = atoi(dim_env);
      if (d > 0)
         sdim = (uint32_t)d;
   }

   const char *keep = getenv("BORGVK_SIM_KEEP");
   if (keep && keep[0]) {
      int kfd = open(keep, O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (kfd >= 0) {
         for (size_t w = 0; w < nbytes; ) {
            ssize_t n = write(kfd, bytes + w, nbytes - w);
            if (n <= 0) break;
            w += (size_t)n;
         }
         close(kfd);
      }
   }

   if (borgvk_hw_enabled()) {
      const size_t bpp_hw = raw32 ? 4 : 3, expected_hw = (size_t)sdim * sdim * bpp_hw;
      uint8_t *rgb_hw = malloc(expected_hw);
      size_t got_hw = rgb_hw ? borgvk_hw_render(bytes, nbytes, sdim, rgb_hw) : 0;
      free(bytes);
      return sim_store_rgb(color_img, rgb_hw, got_hw == expected_hw, raw32, sdim);
   }

   /* Hand the byte stream to arcilator_sim --cts-uart via a temp file. */
   char uart_path[] = "/tmp/borgvk_uart_XXXXXX";
   int ufd = mkstemp(uart_path);
   if (ufd < 0) {
      free(bytes);
      return VK_SUCCESS;
   }
   for (size_t w = 0; w < nbytes; ) {
      ssize_t n = write(ufd, bytes + w, nbytes - w);
      if (n <= 0)
         break;
      w += (size_t)n;
   }
   close(ufd);
   free(bytes);

   char w_str[16], h_str[16];
   snprintf(w_str, sizeof(w_str), "%u", sdim);
   snprintf(h_str, sizeof(h_str), "%u", sdim);

   const uint32_t WIN = 128;
   uint32_t nwx = sdim > WIN ? sdim / WIN : 1;
   uint32_t nparts = 1;
   if (direct && direct[0] && nwx > 1) {
      long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
      nparts = (uint32_t)(ncpu > 1 ? ncpu : 1);
      /* BORGVK_SIM_PARTS=n: at most n processes (1 under a parallel test runner). */
      const char *parts = getenv("BORGVK_SIM_PARTS");
      if (parts && atoi(parts) > 0)
         nparts = MIN2(nparts, (uint32_t)atoi(parts));
      if (nparts > nwx * nwx)
         nparts = nwx * nwx;
   }

   const size_t bpp = raw32 ? 4 : 3;
   size_t expected = (size_t)sdim * sdim * bpp;
   uint8_t *rgb = malloc(expected);
   size_t got = 0;
   int pfds[64][2];
   pid_t pids[64];
   if (nparts > 64)
      nparts = 64;
   for (uint32_t k = 0; k < nparts; k++) {
      pids[k] = -1;
      if (pipe(pfds[k]) < 0)
         continue;
      pid_t pid = fork();
      if (pid == 0) {
         close(pfds[k][0]);
         dup2(pfds[k][1], STDOUT_FILENO);
         close(pfds[k][1]);
         if (direct && direct[0]) {
            if (nwx > 1) {
               char part[32];
               snprintf(part, sizeof(part), "%u/%u", k, nparts);
               setenv("BORG_PART", part, 1);
               setenv("BORG_WINDOW_TILES", "32", 1);
            }
            execlp(direct, direct, uart_path, w_str, h_str, (char *)NULL);
         }
         execlp(sim_bin, sim_bin, "--cts-uart", uart_path, sim_fw,
                w_str, h_str, (char *)NULL);
         _exit(127);
      }
      close(pfds[k][1]);
      pids[k] = pid;
   }
   uint8_t *piece = nparts > 1 ? malloc(expected) : NULL;
   uint32_t pieces_ok = 0;
   for (uint32_t k = 0; k < nparts && rgb; k++) {
      if (pids[k] < 0)
         continue;
      uint8_t *dst = nparts > 1 ? piece : rgb;
      size_t n_got = 0;
      while (n_got < expected) {
         ssize_t n = read(pfds[k][0], dst + n_got, expected - n_got);
         if (n <= 0)
            break;
         n_got += (size_t)n;
      }
      close(pfds[k][0]);
      int wstatus = 0;
      waitpid(pids[k], &wstatus, 0);
      if (nparts > 1) {
         if (n_got == expected) {
            for (uint32_t y = 0; y < sdim; y++)
               for (uint32_t x = 0; x < sdim; x++)
                  if (((y / WIN) * nwx + x / WIN) % nparts == k)
                     memcpy(rgb + ((size_t)y * sdim + x) * bpp,
                            piece + ((size_t)y * sdim + x) * bpp, bpp);
            pieces_ok++;
         }
      } else {
         got = n_got;
      }
   }
   if (nparts > 1)
      got = pieces_ok == nparts ? expected : 0;
   free(piece);
   unlink(uart_path);
   if (getenv("BORGVK_DEBUG"))
      mesa_logi("borgvk: sim returned %zu of %zu bytes", got, expected);

   return sim_store_rgb(color_img, rgb, rgb && got == expected, raw32, sdim);
}

static VkResult
borgvk_submit_sim_cube(struct borgvk_device *device,
                       struct vk_queue_submit *submit)
{
   const char *sim_bin = getenv("BORGVK_SIM");
   const char *sim_fw  = getenv("BORGVK_SIM_FW");
   if (!borgvk_hw_enabled() && (!sim_bin || !sim_fw))
      return VK_SUCCESS;

   struct borgvk_descriptor_set *set = find_set(submit);
   if (!set)
      return VK_SUCCESS;
   struct borgvk_buffer *ubuf = set->buffers[0];
   if (!ubuf || !ubuf->mem || !ubuf->mem->map)
      return VK_SUCCESS;
   VkDeviceSize off = ubuf->offset + set->offsets[0];
   if (off >= ubuf->vk.size)
      return VK_SUCCESS;
   const float *ubo = (const float *)((const char *)ubuf->mem->map + off);
   uint32_t nfloats = (uint32_t)((ubuf->vk.size - off) / sizeof(float));
   if (nfloats < UBO_MIN_FLOATS)
      return VK_SUCCESS;
   struct borgvk_image *tex = set->images[1];

   /* Capture one frame's full wire stream (shaders + geom + texture + MVP). */
   borgvk_transport_capture_begin();
   upload_shaders(device);
   send_geometry(ubo);
   send_blend_state(device);
   if (tex && tex->mem && tex->mem->map &&
       tex->vk.extent.width && tex->vk.extent.height)
      for (int row = 0; row < BORGVK_TEX_DIM; row++)
         send_texture_row(tex, set->samplers[1], row);
   borgvk_serial_send_mvp(ubo);
   size_t nbytes = 0;
   uint8_t *bytes = borgvk_transport_capture_end(&nbytes);
   if (!bytes || nbytes == 0) {
      free(bytes);
      return VK_SUCCESS;
   }

   const char *dump = getenv("BORGVK_SIM_DUMP");
   if (dump && dump[0]) {
      int dfd = open(dump, O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (dfd >= 0) {
         for (size_t w = 0; w < nbytes; ) {
            ssize_t n = write(dfd, bytes + w, nbytes - w);
            if (n <= 0) break;
            w += (size_t)n;
         }
         close(dfd);
      }
      mesa_logi("borgvk: dumped %zu-byte sim stream to %s", nbytes, dump);
      free(bytes);
      return VK_SUCCESS;
   }

   struct borgvk_image *color_img = find_color_attachment(submit);
   const bool rec = borgvk_corpus_on();
   const uint64_t key = rec ? borgvk_corpus_hash(BORGVK_CORPUS_SEED, bytes, nbytes) : 0;
   const VkResult res = sim_run_stream(bytes, nbytes, color_img, 0);
   if (rec)
      borgvk_corpus_log('C', key, borgvk_corpus_image(BORGVK_CORPUS_SEED, color_img));
   return res;
}

struct tex_conv {
   uint32_t bpp, obpp, cs, sch, alpha, pack16, zstride, soff;
   bool swap_rb;
};

static void
tex_convert(const struct tex_conv *c, uint8_t *dst, const uint8_t *src, uint32_t n)
{
   const uint32_t bpp = c->bpp, obpp = c->obpp, cs = c->cs;
   if (!c->sch && !c->pack16 && !c->zstride)
      memcpy(dst, src, (size_t)n * obpp);
   for (uint32_t i = 0; c->zstride && i < n; i++)
      memcpy(dst + (size_t)i * obpp, src + (size_t)i * c->zstride + c->soff, obpp);
   for (uint32_t i = 0; c->sch && i < n; i++) {
      const uint8_t *p = src + (size_t)i * bpp;
      uint8_t *d = dst + (size_t)i * obpp;
      memcpy(d, p + (c->swap_rb ? 2 * cs : 0), cs);
      memcpy(d + cs, p + cs, cs);
      memcpy(d + 2 * cs, p + (c->swap_rb ? 0 : 2 * cs), cs);
      memcpy(d + 3 * cs, c->sch == 4 ? p + 3 * cs : (const uint8_t *)&c->alpha, cs);
   }
   for (uint32_t i = 0; c->pack16 && i < n; i++) {
      uint32_t v = 0;
      memcpy(&v, src + (size_t)i * bpp, bpp);
      switch (c->pack16) {
      case 1: v = (v >> 11) | (v & 0x07e0) | ((v & 0x1f) << 11); break;
      case 2: v = ((v & 1) << 15) | (v >> 1); break;
      case 3: v = ((v & 1) << 15) | ((v & 0x3e) << 9) | ((v >> 1) & 0x03e0) | (v >> 11); break;
      case 4: v = ((v & 0x00f0) << 8) | (v & 0x0f0f) | ((v >> 8) & 0x00f0); break;
      case 5: v = ((v >> 4) * 17) | ((v & 0xf) * 17) << 8; break;
      case 7: v = ((v & 0xf) << 12) | ((v & 0xf0) << 4) | ((v >> 4) & 0xf0) | (v >> 12); break;
      case 8: v = ((v & 0xf00) << 4) | ((v & 0xf0) << 4) | ((v & 0xf) << 4) | (v >> 12); break;
      default: v = (v & 0xc00ffc00) | ((v >> 20) & 0x3ff) | ((v & 0x3ff) << 20); break;
      }
      memcpy(dst + (size_t)i * obpp, &v, obpp);
   }
}

static bool
send_generic_texture(const struct vk_image_view *view,
                     const struct borgvk_image *img,
                     const struct borgvk_sampler *sampler)
{
   if (!img || !img->mem || !img->mem->map)
      return false;
   uint32_t w = img->vk.extent.width, h = img->vk.extent.height;
   if (w == 0 || h == 0 || w > 4096 || h > 4096)
      return false;
   uint32_t type;
   const bool stencil = view->aspects & VK_IMAGE_ASPECT_STENCIL_BIT;
   uint32_t layers = 1, depth = 1;
   switch (view->view_type) {
   case VK_IMAGE_VIEW_TYPE_1D_ARRAY: layers = view->layer_count; type = 0; h = 1; break;
   case VK_IMAGE_VIEW_TYPE_1D: type = 0; h = 1; break;
   case VK_IMAGE_VIEW_TYPE_2D_ARRAY: layers = view->layer_count; type = 1; break;
   case VK_IMAGE_VIEW_TYPE_2D: type = 1; break;
   case VK_IMAGE_VIEW_TYPE_CUBE: type = 3; layers = 6; break;
   case VK_IMAGE_VIEW_TYPE_3D: type = 2; depth = img->vk.extent.depth; break;
   default: return false;
   }
   if (layers == 0 || layers > 1024 || depth > 1024)
      return false;

   uint32_t bpp, fmt_code, cs = 0, sch = 0, alpha = 0, pack16 = 0, zstride = 0, sten_off = 0;
   bool swap_rb = false;
   switch (view->format) {
#define RAWFMT(vk, code, bytes) case VK_FORMAT_##vk: bpp = bytes; fmt_code = code; break;
#define CHFMT(vk, code, csz, n, swap, a) case VK_FORMAT_##vk: \
      fmt_code = code; cs = csz; sch = n; bpp = csz * n; swap_rb = swap; alpha = a; break;
#define P16FMT(vk, code, kind) case VK_FORMAT_##vk: bpp = 2; fmt_code = code; pack16 = kind; break;
   RAWFMT(R8_UNORM, 1, 1) RAWFMT(R8_SNORM, 2, 1) RAWFMT(R8_UINT, 3, 1) RAWFMT(R8_SINT, 4, 1) RAWFMT(R8_SRGB, 5, 1)
   RAWFMT(R8G8_UNORM, 6, 2) RAWFMT(R8G8_SNORM, 7, 2) RAWFMT(R8G8_UINT, 8, 2) RAWFMT(R8G8_SINT, 9, 2)
   RAWFMT(R8G8_SRGB, 10, 2)
   RAWFMT(R8G8B8A8_UNORM, 11, 4) RAWFMT(R8G8B8A8_SNORM, 12, 4) RAWFMT(R8G8B8A8_UINT, 13, 4)
   RAWFMT(R8G8B8A8_SINT, 14, 4) RAWFMT(R8G8B8A8_SRGB, 15, 4)
   RAWFMT(A8B8G8R8_UNORM_PACK32, 11, 4) RAWFMT(A8B8G8R8_SNORM_PACK32, 12, 4) RAWFMT(A8B8G8R8_UINT_PACK32, 13, 4)
   RAWFMT(A8B8G8R8_SINT_PACK32, 14, 4) RAWFMT(A8B8G8R8_SRGB_PACK32, 15, 4)
   RAWFMT(B8G8R8A8_UNORM, 16, 4) RAWFMT(B8G8R8A8_SRGB, 17, 4)
   CHFMT(B8G8R8A8_SNORM, 12, 1, 4, true, 0) CHFMT(B8G8R8A8_UINT, 13, 1, 4, true, 0)
   CHFMT(B8G8R8A8_SINT, 14, 1, 4, true, 0)
   CHFMT(R8G8B8_UNORM, 11, 1, 3, false, 255) CHFMT(R8G8B8_SNORM, 12, 1, 3, false, 127)
   CHFMT(R8G8B8_UINT, 13, 1, 3, false, 1) CHFMT(R8G8B8_SINT, 14, 1, 3, false, 1)
   CHFMT(R8G8B8_SRGB, 15, 1, 3, false, 255)
   CHFMT(B8G8R8_UNORM, 11, 1, 3, true, 255) CHFMT(B8G8R8_SNORM, 12, 1, 3, true, 127)
   CHFMT(B8G8R8_UINT, 13, 1, 3, true, 1) CHFMT(B8G8R8_SINT, 14, 1, 3, true, 1)
   CHFMT(B8G8R8_SRGB, 15, 1, 3, true, 255)
   RAWFMT(R16_UNORM, 18, 2) RAWFMT(R16_SNORM, 19, 2) RAWFMT(R16_UINT, 20, 2) RAWFMT(R16_SINT, 21, 2)
   RAWFMT(R16_SFLOAT, 22, 2) RAWFMT(R16G16_UNORM, 23, 4) RAWFMT(R16G16_SNORM, 24, 4)
   RAWFMT(R16G16_UINT, 25, 4) RAWFMT(R16G16_SINT, 26, 4) RAWFMT(R16G16_SFLOAT, 27, 4)
   CHFMT(R16G16B16_UNORM, 28, 2, 3, false, 0xffff) CHFMT(R16G16B16_SNORM, 29, 2, 3, false, 0x7fff)
   CHFMT(R16G16B16_UINT, 30, 2, 3, false, 1) CHFMT(R16G16B16_SINT, 31, 2, 3, false, 1)
   CHFMT(R16G16B16_SFLOAT, 32, 2, 3, false, 0x3c00)
   RAWFMT(R16G16B16A16_UNORM, 28, 8) RAWFMT(R16G16B16A16_SNORM, 29, 8) RAWFMT(R16G16B16A16_UINT, 30, 8)
   RAWFMT(R16G16B16A16_SINT, 31, 8) RAWFMT(R16G16B16A16_SFLOAT, 32, 8)
   RAWFMT(R32_UINT, 33, 4) RAWFMT(R32_SINT, 34, 4) RAWFMT(R32_SFLOAT, 35, 4)
   RAWFMT(R32G32_UINT, 36, 8) RAWFMT(R32G32_SINT, 37, 8) RAWFMT(R32G32_SFLOAT, 38, 8)
   CHFMT(R32G32B32_UINT, 39, 4, 3, false, 1) CHFMT(R32G32B32_SINT, 40, 4, 3, false, 1)
   CHFMT(R32G32B32_SFLOAT, 41, 4, 3, false, 0x3f800000)
   RAWFMT(R32G32B32A32_UINT, 39, 16) RAWFMT(R32G32B32A32_SINT, 40, 16) RAWFMT(R32G32B32A32_SFLOAT, 41, 16)
   RAWFMT(A2B10G10R10_UNORM_PACK32, 42, 4) RAWFMT(A2B10G10R10_UINT_PACK32, 43, 4)
   RAWFMT(R5G6B5_UNORM_PACK16, 44, 2) RAWFMT(A1R5G5B5_UNORM_PACK16, 45, 2)
   RAWFMT(B4G4R4A4_UNORM_PACK16, 46, 2) RAWFMT(B10G11R11_UFLOAT_PACK32, 47, 4)
   RAWFMT(E5B9G9R9_UFLOAT_PACK32, 48, 4)
   P16FMT(B5G6R5_UNORM_PACK16, 44, 1) P16FMT(R5G5B5A1_UNORM_PACK16, 45, 2)
   P16FMT(B5G5R5A1_UNORM_PACK16, 45, 3) P16FMT(R4G4B4A4_UNORM_PACK16, 46, 4)
   P16FMT(A4R4G4B4_UNORM_PACK16, 46, 7) P16FMT(A4B4G4R4_UNORM_PACK16, 46, 8)
   RAWFMT(D16_UNORM, 49, 2) RAWFMT(X8_D24_UNORM_PACK32, 50, 4) RAWFMT(D24_UNORM_S8_UINT, 50, 4)
   RAWFMT(D32_SFLOAT, 51, 4) RAWFMT(S8_UINT, 3, 1)
   case VK_FORMAT_D16_UNORM_S8_UINT:  bpp = 2; fmt_code = 49; zstride = 3; break;
   case VK_FORMAT_D32_SFLOAT_S8_UINT: bpp = 4; fmt_code = 51; zstride = 8; break;
   case VK_FORMAT_R4G4_UNORM_PACK8: bpp = 1; fmt_code = 6; pack16 = 5; break;
   case VK_FORMAT_A2R10G10B10_UNORM_PACK32: bpp = 4; fmt_code = 42; pack16 = 6; break;
   case VK_FORMAT_A2R10G10B10_UINT_PACK32:  bpp = 4; fmt_code = 43; pack16 = 6; break;
#undef RAWFMT
#undef CHFMT
#undef P16FMT
   default: return false;
   }
   if (stencil) {
      switch (view->format) {
      case VK_FORMAT_S8_UINT: break;   /* the bytes as they are */
      case VK_FORMAT_D16_UNORM_S8_UINT:  zstride = 3; sten_off = 2; break;
      case VK_FORMAT_D24_UNORM_S8_UINT:  zstride = 4; sten_off = 3; break;
      case VK_FORMAT_D32_SFLOAT_S8_UINT: zstride = 8; sten_off = 4; break;
      default: return false;
      }
      bpp = 1; fmt_code = 3;
   }
   const struct tex_conv tc = { bpp, sch ? cs * 4 : pack16 == 5 ? 2 : bpp, cs, sch, alpha, pack16, zstride, sten_off, swap_rb };
   const uint32_t obpp = tc.obpp, sbpp = zstride ? zstride : bpp;
   const uint32_t base = view->base_mip_level;
   const uint32_t levels = base < img->vk.mip_levels ? MIN2(view->level_count, img->vk.mip_levels - base) : 0;
   if (levels == 0 || levels > 13)
      return false;

   uint32_t lw[13], lh[13], ld[13];
   uint64_t soff[13], sum = 0;
   for (uint32_t l = 0; l < img->vk.mip_levels; l++) {
      uint32_t cw = MAX2(img->vk.extent.width >> l, 1), ch = type ? MAX2(img->vk.extent.height >> l, 1) : 1;
      uint32_t cd = type == 2 ? MAX2(img->vk.extent.depth >> l, 1) : 1;
      if (l >= base && l < base + levels) {
         lw[l - base] = cw;
         lh[l - base] = ch;
         ld[l - base] = cd;
         soff[l - base] = sum;
      }
      sum += (uint64_t)cw * ch * cd * sbpp;
   }
   w = lw[0];
   h = lh[0];
   const uint8_t *src = (const uint8_t *)img->mem->map + img->offset;

   const bool tiled = levels > 1 || layers > 1 || depth > 1 || type == 3;
   uint32_t offs[12] = { 0 }, ltotal = 0, total = 0;
   uint8_t *out = NULL;
   if (!tiled) {
      total = align(w * h * obpp, 4);
      out = calloc(1, total);
      if (out)
         tex_convert(&tc, out, src + (uint64_t)view->base_array_layer * sum + soff[0], w * h);
   } else {
      uint32_t lsz[13];
      for (uint32_t l = 0; l < levels; l++) {
         lsz[l] = ((lw[l] + 3) / 4) * ((lh[l] + 3) / 4) * 16 * obpp * ld[l];
         if (l)
            offs[l - 1] = ltotal;
         ltotal += lsz[l];
      }
      total = ltotal * layers;
      out = total <= 0x70000 ? calloc(1, total) : NULL;
      uint8_t *lin = out ? malloc((size_t)w * h * depth * obpp) : NULL;
      if (!lin) {
         free(out);
         return false;
      }
      for (uint32_t layer = 0; layer < layers; layer++)
         for (uint32_t l = 0; l < levels; l++) {
            const uint32_t tw = (lw[l] + 3) / 4, th = (lh[l] + 3) / 4;
            uint8_t *lev = out + (size_t)layer * ltotal + (l ? offs[l - 1] : 0);
            tex_convert(&tc, lin, src + (uint64_t)(view->base_array_layer + layer) * sum + soff[l],
                        lw[l] * lh[l] * ld[l]);
            for (uint32_t z = 0; z < ld[l]; z++)
               for (uint32_t y = 0; y < lh[l]; y++)
                  for (uint32_t x = 0; x < lw[l]; x++) {
                     const uint8_t *t = lin + (((size_t)z * lh[l] + y) * lw[l] + x) * obpp;
                     uint8_t *sl = lev + (size_t)z * tw * th * 16 * obpp;
                     const uint32_t tile = (y >> 2) * tw + (x >> 2), ti = (y & 3) * 4 + (x & 3);
                     if (obpp == 16) {
                        memcpy(sl + tile * 256 + ti * 8, t, 8);
                        memcpy(sl + tile * 256 + 128 + ti * 8, t + 8, 8);
                     } else {
                        memcpy(sl + ((size_t)tile * 16 + ti) * obpp, t, obpp);
                     }
                  }
         }
      free(lin);
   }
   if (!out)
      return false;

   uint32_t sw[4] = { view->swizzle.r, view->swizzle.g, view->swizzle.b, view->swizzle.a };
   const bool no_alpha = sch && !util_format_has_alpha(vk_format_to_pipe_format(view->format));
   for (uint32_t i = 0; no_alpha && i < 4; i++) {
      const uint32_t src = sw[i] == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_R + i : sw[i];
      if (src == VK_COMPONENT_SWIZZLE_A)
         sw[i] = VK_COMPONENT_SWIZZLE_ONE;
   }
   uint32_t desc[3];
   desc[0] = (w - 1) | ((h - 1) << 16) | (type << 28) | (tiled ? 0 : 1u << 30);
   desc[1] = ((type == 2 ? depth : layers) - 1) | (fmt_code << 14) | ((levels - 1) << 10) |
             ((uint32_t)(sw[0] & 7) << 20) |
             ((uint32_t)(sw[1] & 7) << 23) |
             ((uint32_t)(sw[2] & 7) << 26) |
             ((uint32_t)(sw[3] & 7) << 29);
   desc[2] = tiled ? ltotal : w * obpp;
   if (levels > 1)
      borgvk_serial_send_texture_levels(offs);
   for (uint32_t off = 0; off < total; off += BORGVK_TEXG_CHUNK) {
      uint32_t n = total - off < BORGVK_TEXG_CHUNK ? total - off : BORGVK_TEXG_CHUNK;
      borgvk_serial_send_texture_chunk(off, out + off, n, desc, sampler_desc(sampler));
   }
   free(out);
   return true;
}

static bool
send_buffer_texture(const struct vk_buffer_view *bv)
{
   uint32_t fmt_code;
   switch (bv->format) {
   case VK_FORMAT_R32_UINT:   fmt_code = 33; break;   /* BORG_TEX_FORMAT_R32_UINT */
   case VK_FORMAT_R32_SFLOAT: fmt_code = 35; break;
   default: return false;
   }
   struct borgvk_buffer *buf = container_of(bv->buffer, struct borgvk_buffer, vk);
   if (!buf->mem || !buf->mem->map)
      return false;
   uint32_t w = (uint32_t)(bv->range / 4);
   if (w == 0 || w > 65536)
      return false;
   uint32_t total = w * 4;
   const uint8_t *src = (const uint8_t *)buf->mem->map + buf->offset + bv->offset;
   uint32_t desc[3];
   desc[0] = (w - 1) | (0u << 28) | (1u << 30);       /* 1D, linear */
   desc[1] = fmt_code << 14;                          /* identity swizzle */
   desc[2] = total;
   for (uint32_t off = 0; off < total; off += BORGVK_TEXG_CHUNK) {
      uint32_t n = total - off < BORGVK_TEXG_CHUNK ? total - off : BORGVK_TEXG_CHUNK;
      borgvk_serial_send_texture_chunk(off, src + off, n, desc, sampler_desc(NULL));
   }
   return true;
}

static bool
vertex_format_to_fetch(VkFormat f, uint32_t *code, uint32_t *swz, uint32_t *read_bytes)
{
   const uint32_t A_ONE = 2u << 29;
   *swz = 0;
   switch (f) {
   case VK_FORMAT_R8_UNORM: *code = 1; *read_bytes = 1; return true;
   case VK_FORMAT_R8_SNORM: *code = 2; *read_bytes = 1; return true;
   case VK_FORMAT_R8_UINT:  *code = 3; *read_bytes = 1; return true;
   case VK_FORMAT_R8_SINT:  *code = 4; *read_bytes = 1; return true;
   case VK_FORMAT_R8G8_UNORM: *code = 6; *read_bytes = 2; return true;
   case VK_FORMAT_R8G8_SNORM: *code = 7; *read_bytes = 2; return true;
   case VK_FORMAT_R8G8_UINT:  *code = 8; *read_bytes = 2; return true;
   case VK_FORMAT_R8G8_SINT:  *code = 9; *read_bytes = 2; return true;
   case VK_FORMAT_R8G8B8_UNORM: *code = 11; *swz = A_ONE; *read_bytes = 4; return true;
   case VK_FORMAT_R8G8B8_SNORM: *code = 12; *swz = A_ONE; *read_bytes = 4; return true;
   case VK_FORMAT_R8G8B8_UINT:  *code = 13; *swz = A_ONE; *read_bytes = 4; return true;
   case VK_FORMAT_R8G8B8_SINT:  *code = 14; *swz = A_ONE; *read_bytes = 4; return true;
   case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_A8B8G8R8_UNORM_PACK32: *code = 11; *read_bytes = 4; return true;
   case VK_FORMAT_R8G8B8A8_SNORM: case VK_FORMAT_A8B8G8R8_SNORM_PACK32: *code = 12; *read_bytes = 4; return true;
   case VK_FORMAT_R8G8B8A8_UINT:  case VK_FORMAT_A8B8G8R8_UINT_PACK32:  *code = 13; *read_bytes = 4; return true;
   case VK_FORMAT_R8G8B8A8_SINT:  case VK_FORMAT_A8B8G8R8_SINT_PACK32:  *code = 14; *read_bytes = 4; return true;
   case VK_FORMAT_B8G8R8A8_UNORM: *code = 16; *read_bytes = 4; return true;
   case VK_FORMAT_R16_UNORM: *code = 18; *read_bytes = 2; return true;
   case VK_FORMAT_R16_SNORM: *code = 19; *read_bytes = 2; return true;
   case VK_FORMAT_R16_UINT:  *code = 20; *read_bytes = 2; return true;
   case VK_FORMAT_R16_SINT:  *code = 21; *read_bytes = 2; return true;
   case VK_FORMAT_R16_SFLOAT: *code = 22; *read_bytes = 2; return true;
   case VK_FORMAT_R16G16_UNORM: *code = 23; *read_bytes = 4; return true;
   case VK_FORMAT_R16G16_SNORM: *code = 24; *read_bytes = 4; return true;
   case VK_FORMAT_R16G16_UINT:  *code = 25; *read_bytes = 4; return true;
   case VK_FORMAT_R16G16_SINT:  *code = 26; *read_bytes = 4; return true;
   case VK_FORMAT_R16G16_SFLOAT: *code = 27; *read_bytes = 4; return true;
   case VK_FORMAT_R16G16B16_UNORM: *code = 28; *swz = A_ONE; *read_bytes = 8; return true;
   case VK_FORMAT_R16G16B16_SNORM: *code = 29; *swz = A_ONE; *read_bytes = 8; return true;
   case VK_FORMAT_R16G16B16_UINT:  *code = 30; *swz = A_ONE; *read_bytes = 8; return true;
   case VK_FORMAT_R16G16B16_SINT:  *code = 31; *swz = A_ONE; *read_bytes = 8; return true;
   case VK_FORMAT_R16G16B16_SFLOAT: *code = 32; *swz = A_ONE; *read_bytes = 8; return true;
   case VK_FORMAT_R16G16B16A16_UNORM: *code = 28; *read_bytes = 8; return true;
   case VK_FORMAT_R16G16B16A16_SNORM: *code = 29; *read_bytes = 8; return true;
   case VK_FORMAT_R16G16B16A16_UINT:  *code = 30; *read_bytes = 8; return true;
   case VK_FORMAT_R16G16B16A16_SINT:  *code = 31; *read_bytes = 8; return true;
   case VK_FORMAT_R16G16B16A16_SFLOAT: *code = 32; *read_bytes = 8; return true;
   case VK_FORMAT_R32_UINT:   *code = 33; *read_bytes = 4; return true;
   case VK_FORMAT_R32_SINT:   *code = 34; *read_bytes = 4; return true;
   case VK_FORMAT_R32_SFLOAT: *code = 35; *read_bytes = 4; return true;
   case VK_FORMAT_R32G32_UINT:   *code = 36; *read_bytes = 8; return true;
   case VK_FORMAT_R32G32_SINT:   *code = 37; *read_bytes = 8; return true;
   case VK_FORMAT_R32G32_SFLOAT: *code = 38; *read_bytes = 8; return true;
   case VK_FORMAT_R32G32B32_UINT:   *code = 39; *swz = A_ONE; *read_bytes = 16; return true;
   case VK_FORMAT_R32G32B32_SINT:   *code = 40; *swz = A_ONE; *read_bytes = 16; return true;
   case VK_FORMAT_R32G32B32_SFLOAT: *code = 41; *swz = A_ONE; *read_bytes = 16; return true;
   case VK_FORMAT_R32G32B32A32_UINT:   *code = 39; *read_bytes = 16; return true;
   case VK_FORMAT_R32G32B32A32_SINT:   *code = 40; *read_bytes = 16; return true;
   case VK_FORMAT_R32G32B32A32_SFLOAT: *code = 41; *read_bytes = 16; return true;
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32: *code = 42; *read_bytes = 4; return true;
   case VK_FORMAT_A2B10G10R10_UINT_PACK32:  *code = 43; *read_bytes = 4; return true;
   default: return false;
   }
}

static bool
generic_reject(int why)
{
   if (getenv("BORGVK_DEBUG"))
      mesa_logi("borgvk: generic sim draw rejected (reason %d)", why);
   return false;
}

#define BORGVK_HEAP_BYTES 0x2000000u
#define BORGVK_POINT_SLOT 16

struct draw_heap {
   uint32_t top;
};

static VkResult sim_run_stream(uint8_t *bytes, size_t nbytes, struct borgvk_image *color_img,
                               uint32_t sim_dim);

static uint64_t
corpus_state(uint64_t h, const struct borgvk_image *img, struct borgvk_image *const *extra, uint32_t nextra,
             const struct vk_image_view *depth, const struct vk_image_view *stencil,
             const struct borgvk_writeback *wb, uint32_t nwb)
{
   h = borgvk_corpus_image(h, img);
   for (uint32_t i = 0; i < nextra; i++)
      h = borgvk_corpus_image(h, extra[i]);
   if (depth && depth->image)
      h = borgvk_corpus_image(h, container_of(depth->image, struct borgvk_image, vk));
   if (stencil && stencil->image)
      h = borgvk_corpus_image(h, container_of(stencil->image, struct borgvk_image, vk));
   for (uint32_t i = 0; i < nwb; i++)
      h = borgvk_corpus_hash(h, wb[i].host, wb[i].size);
   return h;
}

void
borgvk_flush_draws(struct borgvk_command_buffer *cmd)
{
   if (cmd->stream_len == 0)
      return;
   uint8_t *bytes = cmd->stream;
   size_t n = cmd->stream_len;
   struct borgvk_image *img = cmd->batch_img;
   struct borgvk_image *extra[3] = { cmd->batch_extra[0], cmd->batch_extra[1], cmd->batch_extra[2] };
   const uint32_t nextra = cmd->batch_nextra;
   const bool serve = cmd->batch_serve;
   struct borgvk_writeback wb[BORGVK_MAX_WRITEBACKS];
   const uint32_t nwb = cmd->nwb;
   memcpy(wb, cmd->wb, sizeof(wb));
   cmd->nwb = 0;
   cmd->region_set = NULL;
   cmd->batch_pool = NULL;
   cmd->batch_srgb = false;
   cmd->stream = NULL;
   cmd->stream_len = cmd->stream_cap = 0;
   cmd->heap_top = 0;
   cmd->batch_draws = 0;
   cmd->batch_img = NULL;
   cmd->batch_nextra = 0;
   free(cmd->state_sent);
   cmd->state_sent = NULL;
   cmd->state_sent_len = 0;
   const bool rec = borgvk_corpus_on();
   uint64_t key = BORGVK_CORPUS_SEED;
   if (rec) {
      key = borgvk_corpus_hash(key, bytes, n);
      key = corpus_state(key, img, extra, nextra, cmd->batch_depth, cmd->batch_stencil, wb, nwb);
   }
   if (serve) {
      borgvk_sim_run_pass(bytes, n, img, extra, nextra, cmd->batch_depth, cmd->batch_stencil, wb, nwb);
      free(bytes);
   } else {
      sim_run_stream(bytes, n, img, 0);   /* takes ownership of bytes */
   }
   if (rec)
      borgvk_corpus_log('F', key, corpus_state(BORGVK_CORPUS_SEED, img, extra, nextra,
                                               cmd->batch_depth, cmd->batch_stencil, wb, nwb));
}

static bool
stream_append(struct borgvk_command_buffer *cmd, const uint8_t *data, size_t n)
{
   if (cmd->stream_len + n > cmd->stream_cap) {
      size_t cap = MAX2(cmd->stream_cap * 2, cmd->stream_len + n + 65536);
      uint8_t *p = realloc(cmd->stream, cap);
      if (!p)
         return false;
      cmd->stream = p;
      cmd->stream_cap = cap;
   }
   memcpy(cmd->stream + cmd->stream_len, data, n);
   cmd->stream_len += n;
   return true;
}

static uint32_t
heap_upload(struct draw_heap *h, const uint8_t *data, uint32_t n)
{
   uint32_t padded = (n + 3) & ~3u;
   if (h->top + padded + 16 > BORGVK_HEAP_BYTES)
      return UINT32_MAX;
   uint32_t off = h->top;
   for (uint32_t i = 0; i < padded; i += 256) {
      uint8_t tmp[256] = { 0 };
      uint32_t c = MIN2(256u, n > i ? n - i : 0);
      if (c)
         memcpy(tmp, data + i, c);
      borgvk_serial_send_mem(off + i, tmp, MIN2(256u, padded - i));
   }
   h->top += padded + 16;   /* slack for the typed fetch's over-read */
   return off;
}

struct draw_params {
   bool          indexed;
   uint32_t      index_size;      /* 2 or 4 */
   uint32_t      count;           /* vertices or indices */
   uint32_t      instances;
   uint32_t      first;           /* first vertex or first index */
   int32_t       vertex_offset;
   uint32_t      first_instance;
};

static bool
borgvk_sim_generic_draw(struct borgvk_device *device, struct borgvk_command_buffer *cmd,
                        const struct draw_params *dp)
{
   struct borgvk_pipeline *pipeline = cmd->gfx_pipeline;
   struct borgvk_descriptor_set *set = cmd->desc_set;

   if (!pipeline || dp->count == 0 || dp->instances == 0)
      return generic_reject(3);
   uint32_t topo;
   switch (pipeline->topology) {
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:  topo = 0; break;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: topo = 1; break;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:   topo = 2; break;
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:     topo = 0; break;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:     topo = 0; break;
   default: return generic_reject(3);
   }
   if (pipeline->blob[BORGVK_STAGE_VERT].len == 0 || pipeline->blob[BORGVK_STAGE_FRAG].len == 0)
      return generic_reject(4);
   const bool points = pipeline->topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
   const uint32_t X = points ? 6 : 1;
   /* A line is a quad too, but its six rows are gathered per line (see `pairs`). */
   const bool lines = pipeline->topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST ||
                      pipeline->topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;

   struct borgvk_image *color_img =
      cmd->color_views[0] && cmd->color_views[0]->image
         ? container_of(cmd->color_views[0]->image, struct borgvk_image, vk) : NULL;
   /* Depth only: no colour attachment, the depth or stencil image gives the size. */
   const struct vk_image_view *ds_view = cmd->depth_view ? cmd->depth_view : cmd->stencil_view;
   struct borgvk_image *size_img =
      color_img ? color_img : ds_view && ds_view->image ? container_of(ds_view->image, struct borgvk_image, vk) : NULL;
   if (!size_img || !size_img->mem || !size_img->mem->map)
      return generic_reject(5);
   if (!color_img && !borgvk_sim_serves(size_img, true))
      return generic_reject(5);
   struct borgvk_image *extra[3] = { NULL, NULL, NULL };
   const uint32_t nextra = pipeline->mrt ? pipeline->color_count - 1 : 0;
   if (pipeline->color_count >= 2 && !pipeline->mrt)
      return generic_reject(12);
   for (uint32_t k = 0; k < nextra; k++) {
      struct vk_image_view *v = cmd->color_views[k + 1];
      extra[k] = v && v->image ? container_of(v->image, struct borgvk_image, vk) : NULL;
      if (!extra[k] || !extra[k]->mem || !extra[k]->mem->map ||
          extra[k]->vk.extent.width != color_img->vk.extent.width ||
          extra[k]->vk.extent.height != color_img->vk.extent.height)
         return generic_reject(13);
   }
   if (nextra && !borgvk_sim_serves(color_img, cmd->depth_view || cmd->stencil_view))
      return generic_reject(14);
   bool same_extra = cmd->batch_nextra == nextra;
   for (uint32_t k = 0; k < nextra; k++)
      same_extra &= cmd->batch_extra[k] == extra[k];
   if (cmd->batch_draws && (!same_extra || cmd->batch_img != color_img || cmd->batch_depth != cmd->depth_view ||
                            cmd->batch_stencil != cmd->stencil_view))
      borgvk_flush_draws(cmd);

   /* The index range each binding needs. */
   const uint8_t *idx_host = NULL;
   if (dp->indexed) {
      if (!cmd->index_ptr)
         return generic_reject(6);
      idx_host = cmd->index_ptr + (size_t)dp->first * dp->index_size;
      if ((size_t)(dp->first + dp->count) * dp->index_size > cmd->index_avail)
         return generic_reject(6);
   }
   int64_t vmin = INT64_MAX, vmax = -1;
   if (dp->indexed) {
      const uint32_t restart_val = dp->index_size == 2 ? 0xFFFFu : 0xFFFFFFFFu;
      for (uint32_t i = 0; i < dp->count; i++) {
         uint32_t v = dp->index_size == 2 ? ((const uint16_t *)idx_host)[i] : ((const uint32_t *)idx_host)[i];
         if (pipeline->restart && v == restart_val)
            continue;
         int64_t r = (int64_t)v + dp->vertex_offset;
         if (r < 0)
            return generic_reject(7);
         vmin = MIN2(vmin, r);
         vmax = MAX2(vmax, r);
      }
   } else {
      vmin = dp->first;
      vmax = (int64_t)dp->first + dp->count - 1;
   }
   if (vmax < 0)
      return generic_reject(7);
   /* Lines: the two vertex rows of each line. */
   uint32_t (*pairs)[2] = NULL, nlines = 0;
   if (lines) {
      const bool strip = pipeline->topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
      const uint32_t restart_val = dp->index_size == 2 ? 0xFFFFu : 0xFFFFFFFFu;
      pairs = malloc(sizeof(*pairs) * MAX2(dp->count, 1));
      if (!pairs)
         return generic_reject(10);
      int64_t prev = -1;
      for (uint32_t i = 0; i < dp->count; i++) {
         int64_t r = (int64_t)dp->first + i;
         if (dp->indexed) {
            uint32_t v = dp->index_size == 2 ? ((const uint16_t *)idx_host)[i] : ((const uint32_t *)idx_host)[i];
            if (pipeline->restart && v == restart_val) {
               prev = -1;
               continue;
            }
            r = (int64_t)v + dp->vertex_offset;
         }
         if (prev < 0) {
            prev = r;
            continue;
         }
         pairs[nlines][0] = (uint32_t)prev;
         pairs[nlines][1] = (uint32_t)r;
         nlines++;
         prev = strip ? r : -1;
      }
      if (nlines == 0 || (uint64_t)nlines * 6 > 4096 * 4096) {
         free(pairs);
         return generic_reject(7);
      }
   }
   static const uint8_t line_end[6] = { 0, 0, 1, 0, 1, 1 };
   static const float line_side[6] = { -1, 1, -1, 1, 1, -1 };
   int64_t imin = dp->first_instance, imax = (int64_t)dp->first_instance + dp->instances - 1;

   {
      uint64_t need = dp->indexed ? (uint64_t)dp->count * dp->index_size + 32 : 0;
      for (uint32_t b = 0; b < BORGVK_MAX_VERTEX_BINDINGS; b++) {
         bool used = false;
         for (uint32_t a = 0; a < pipeline->attr_count; a++)
            used |= pipeline->attrs[a].binding == b;
         if (!used)
            continue;
         int64_t r0 = pipeline->binding_instance[b] ? imin : vmin;
         int64_t r1 = pipeline->binding_instance[b] ? imax : vmax;
         need += (uint64_t)(r1 - r0 + 1) * X * pipeline->binding_stride[b] + 32;
      }
      if (points)
         need += (uint64_t)(vmax - vmin + 1) * X * 16 + 32;
      if (lines)
         need = (uint64_t)nlines * 6 * (16 + 2 * 16 * pipeline->attr_count) + 64;
      if (cmd->batch_draws && cmd->heap_top + need > BORGVK_HEAP_BYTES)
         borgvk_flush_draws(cmd);
      /* The constant pool's slot is one per pass. */
      if (cmd->batch_draws && pipeline->blob[BORGVK_STAGE_FRAG].npool && cmd->batch_pool && cmd->batch_pool != pipeline)
         borgvk_flush_draws(cmd);
   }
   const bool first_of_batch = cmd->batch_draws == 0;

   /* The DRM shim's ioctls bypass the capture buffer; force the serial path. */
   int saved_fd = device->drm_fd;
   device->drm_fd = -1;
   borgvk_transport_capture_begin();
   memcpy(device->shader_blob, pipeline->blob, sizeof(device->shader_blob));
   device->blend_cfg = pipeline->blend_cfg;
   device->blend_const = pipeline->blend_const;
   memcpy(device->state_reg, pipeline->state_reg, sizeof(device->state_reg));
   device->state_valid = pipeline->state_valid;
   device->vp = pipeline->vp;
   {
      const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
      if (pipeline->vp_dyn && BITSET_TEST(dyn->set, MESA_VK_DYNAMIC_VP_VIEWPORTS)) {
         for (int k = 0; k < 6; k++) {
            const VkViewport *v = &dyn->vp.viewports[0];
            const float vv[6] = { v->x, v->y, v->width, v->height, v->minDepth, v->maxDepth };
            device->vp.v[k] = vv[k];
         }
         device->vp.valid = true;
      }
      if (pipeline->sc_dyn && BITSET_TEST(dyn->set, MESA_VK_DYNAMIC_VP_SCISSORS)) {
         const VkRect2D *r = &dyn->vp.scissors[0];
         device->vp.sc[0] = MAX2(r->offset.x, 0);
         device->vp.sc[1] = MAX2(r->offset.y, 0);
         device->vp.sc[2] = MAX2(r->offset.x + (int32_t)r->extent.width, 0);
         device->vp.sc[3] = MAX2(r->offset.y + (int32_t)r->extent.height, 0);
         device->vp.valid = true;
      }
   }
   upload_shaders(device);
   send_blend_state(device);
   size_t state_len = 0;
   uint8_t *state = borgvk_transport_capture_end(&state_len);
   const bool state_same = state && cmd->state_sent && state_len == cmd->state_sent_len &&
                           memcmp(state, cmd->state_sent, state_len) == 0;
   borgvk_transport_capture_begin();
   if (first_of_batch)
      cmd->batch_serve = borgvk_sim_serves(size_img, cmd->depth_view || cmd->stencil_view);
   const bool serve = cmd->batch_serve;
   uint32_t load = first_of_batch ? 0u : 1u;
   if (serve) {
      const bool has_z = cmd->depth_view != NULL, has_s = cmd->stencil_view != NULL;
      load = 1u | (has_z ? 2u : 0u) | (has_s ? 4u : 0u) | 8u;
      if (first_of_batch)
         borgvk_serial_send_pass(nextra ? 5 : color_img ? borgvk_sim_flush_format(color_img->vk.format) : 0,
                                 (has_z ? 1u : 0u) | (has_s ? 2u : 0u) | (color_img ? 0u : 8u) |
                                 (has_z && borgvk_sim_depth_is_d32(cmd->depth_view->image->format) ? 4u : 0u));
      if (first_of_batch) {
         const uint8_t raw8[3] = { 5, 5, 5 };
         borgvk_serial_send_att(nextra + 1, raw8);
      }
   } else if (first_of_batch) {
      uint8_t fmt = color_img->vk.format == VK_FORMAT_R8G8B8A8_UNORM ? 1 :
                    color_img->vk.format == VK_FORMAT_B8G8R8A8_UNORM ? 2 :
                    borgvk_sim_raw32(color_img->vk.format) ? 3 : 0;   /* 3 = RAW32 */
      float clear[4] = { 0, 0, 0, 0 };
      if (cmd->has_clear)
         memcpy(clear, cmd->clear_color, sizeof(clear));
      borgvk_serial_send_target(fmt, clear);
   }

   struct draw_heap heap = { cmd->heap_top };
   bool ok = true;
   for (uint32_t a = 0; ok && a < pipeline->attr_count; a++) {
      const struct borgvk_vertex_attr *at = &pipeline->attrs[a];
      uint32_t code, swz, rbytes;
      uint32_t b = at->binding;
      if (at->location >= 16 || b >= BORGVK_MAX_VERTEX_BINDINGS ||
          !vertex_format_to_fetch(at->format, &code, &swz, &rbytes) ||
          !cmd->vb[b]) {
         ok = generic_reject(8);
         break;
      }
      const uint32_t stride = pipeline->binding_stride[b];
      const bool inst = pipeline->binding_instance[b];
      if (lines && !inst) {
         const uint32_t asz = vk_format_get_blocksize(at->format);
         for (uint32_t other = 0; ok && other < 2; other++) {
            uint8_t *tight = calloc((size_t)nlines * 6, rbytes);
            if (!tight) {
               ok = generic_reject(10);
               break;
            }
            for (uint32_t r = 0; r < nlines * 6; r++) {
               const uint64_t src = (uint64_t)pairs[r / 6][line_end[r % 6] ^ other] * stride + at->offset;
               if (src + asz <= cmd->vb_avail[b])
                  memcpy(tight + (size_t)r * rbytes, cmd->vb[b] + src, MIN2(asz, rbytes));
            }
            const uint32_t off = heap_upload(&heap, tight, nlines * 6 * rbytes);
            free(tight);
            if (off == UINT32_MAX)
               ok = generic_reject(10);
            else
               borgvk_serial_send_vattr(at->location + (other ? BORGVK_LINE_OTHER : 0), code, off, nlines * 6, rbytes, swz);
         }
         continue;
      }
      const int64_t row0 = inst ? imin : vmin, row1 = inst ? imax : vmax;
      const bool expand = points && !inst;
      const int64_t e0 = expand ? row0 * X : row0, e1 = expand ? row1 * X + X - 1 : row1;
      if (e1 + 1 > 4096 * 4096) {
         ok = generic_reject(9);
         break;
      }
      const uint32_t nrows = (uint32_t)(e1 - e0 + 1);
      const uint32_t asz = vk_format_get_blocksize(at->format);
      uint8_t *tight = calloc(nrows, rbytes);
      if (!tight) {
         ok = generic_reject(10);
         break;
      }
      const uint64_t avail = cmd->vb_avail[b];
      for (uint32_t r = 0; r < nrows; r++) {
         const uint64_t src = (uint64_t)(expand ? (e0 + r) / X : row0 + r) * stride + at->offset;
         if (src + asz <= avail)
            memcpy(tight + (size_t)r * rbytes, cmd->vb[b] + src, MIN2(asz, rbytes));
      }
      const uint32_t off = heap_upload(&heap, tight, nrows * rbytes);
      free(tight);
      if (off == UINT32_MAX) {
         ok = generic_reject(10);
         break;
      }
      const uint32_t base = off - (uint32_t)e0 * rbytes;
      const uint32_t count = (uint32_t)e1 + 1;
      borgvk_serial_send_vattr(at->location, code, base, count, rbytes, swz);
   }
   if (ok && points) {
      static const float corner[6][2] = { { -.5f, -.5f }, { .5f, -.5f }, { -.5f, .5f },
                                          { .5f, -.5f }, { .5f, .5f }, { -.5f, .5f } };
      /* (dx, dy) in clip units, then the point coordinate (u, v) of the corner. */
      uint32_t code, swz, rbytes;
      const int64_t e0 = vmin * X, e1 = vmax * X + X - 1;
      if (e1 + 1 > 4096 * 4096 || !vertex_format_to_fetch(VK_FORMAT_R32G32B32A32_SFLOAT, &code, &swz, &rbytes)) {
         ok = generic_reject(9);
      } else {
         float *c = malloc((size_t)(e1 - e0 + 1) * 16);
         if (!c) {
            ok = generic_reject(10);
         } else {
            for (int64_t e = e0; e <= e1; e++) {
               float *o = c + 4 * (e - e0);
               o[0] = corner[e % X][0] * 2.0f / size_img->vk.extent.width;
               o[1] = corner[e % X][1] * 2.0f / size_img->vk.extent.height;
               o[2] = corner[e % X][0] + 0.5f;
               o[3] = corner[e % X][1] + 0.5f;
            }
            const uint32_t off = heap_upload(&heap, (const uint8_t *)c, (uint32_t)(e1 - e0 + 1) * 16);
            free(c);
            if (off == UINT32_MAX)
               ok = generic_reject(10);
            else
               borgvk_serial_send_vattr(BORGVK_POINT_SLOT, code, off - (uint32_t)e0 * rbytes, (uint32_t)e1 + 1, rbytes, swz);
         }
      }
   }
   if (ok && pipeline->blob[BORGVK_STAGE_FRAG].npool && cmd->batch_pool != pipeline) {
      const struct borgvk_shader_blob *fb = &pipeline->blob[BORGVK_STAGE_FRAG];
      uint32_t code, swz, rbytes, words[BORGVK_POOL_MAX + 4] = { 0 };
      memcpy(words, fb->pool, fb->npool * 4);
      const uint32_t rows = (fb->npool + 3) / 4;
      const uint32_t off = heap_upload(&heap, (const uint8_t *)words, rows * 16);
      if (off == UINT32_MAX || !vertex_format_to_fetch(VK_FORMAT_R32G32B32A32_UINT, &code, &swz, &rbytes)) {
         ok = generic_reject(10);
      } else {
         borgvk_serial_send_vattr(BORGVK_POOL_SLOT, code, off, rows, rbytes, swz);
         cmd->batch_pool = pipeline;
      }
   }
   if (ok && pipeline->srgb_table && !cmd->batch_srgb) {
      float t[256];
      uint32_t code, swz, rbytes;
      for (int i = 0; i < 256; i++)
         t[i] = util_format_srgb_8unorm_to_linear_float(i);
      const uint32_t off = heap_upload(&heap, (const uint8_t *)t, sizeof(t));
      if (off == UINT32_MAX || !vertex_format_to_fetch(VK_FORMAT_R32_SFLOAT, &code, &swz, &rbytes)) {
         ok = generic_reject(10);
      } else {
         borgvk_serial_send_vattr(BORGVK_SRGB_SLOT, code, off, 256, rbytes, swz);
         cmd->batch_srgb = true;
      }
   }
   if (ok && lines) {
      /* The half-pixel step in clip units for x and y, then the target's size. */
      uint32_t code, swz, rbytes;
      float *c = malloc((size_t)nlines * 6 * 16);
      if (!c || !vertex_format_to_fetch(VK_FORMAT_R32G32B32A32_SFLOAT, &code, &swz, &rbytes)) {
         ok = generic_reject(10);
      } else {
         const float w = size_img->vk.extent.width, h = size_img->vk.extent.height;
         for (uint32_t r = 0; r < nlines * 6; r++) {
            c[4 * r + 0] = line_side[r % 6] / w;
            c[4 * r + 1] = line_side[r % 6] / h;
            c[4 * r + 2] = w;
            c[4 * r + 3] = h;
         }
         const uint32_t off = heap_upload(&heap, (const uint8_t *)c, nlines * 6 * 16);
         if (off == UINT32_MAX)
            ok = generic_reject(10);
         else
            borgvk_serial_send_vattr(BORGVK_LINE_SLOT, code, off, nlines * 6, rbytes, swz);
      }
      free(c);
   }
   free(pairs);
   uint32_t idx_off = 0, index_code = dp->index_size == 2 ? 1 : 2, voff_out = dp->vertex_offset;
   if (ok && dp->indexed && !lines) {
      if (points) {
         uint32_t *ex = malloc((size_t)dp->count * X * 4);
         if (!ex) {
            ok = generic_reject(10);
         } else {
            for (uint32_t i = 0; i < dp->count; i++) {
               int64_t v = (dp->index_size == 2 ? ((const uint16_t *)idx_host)[i] : ((const uint32_t *)idx_host)[i]) +
                           (int64_t)dp->vertex_offset;
               for (uint32_t k = 0; k < X; k++)
                  ex[i * X + k] = (uint32_t)(v * X + k);
            }
            idx_off = heap_upload(&heap, (const uint8_t *)ex, dp->count * X * 4);
            free(ex);
            index_code = 2;
            voff_out = 0;
         }
      } else {
         idx_off = heap_upload(&heap, idx_host, dp->count * dp->index_size);
      }
      if (ok && idx_off == UINT32_MAX)
         ok = generic_reject(10);
   }
   uint32_t ubo_base = 0;
   uint32_t smask = 0;
   struct borgvk_descriptor_set *sets[4] = { set, cmd->desc_sets[1], cmd->desc_sets[2], cmd->desc_sets[3] };
   bool any_dyn = false;
   for (int si = 0; si < 4; si++) {
      if (!sets[si])
         continue;
      any_dyn |= sets[si]->dyn_mask != 0;
      for (int b = 0; b < BORGVK_MAX_BINDINGS && pipeline->set_base[si] + b < 32; b++)
         if ((sets[si]->storage_mask >> b & 1) && sets[si]->buffers[b] && sets[si]->buffers[b]->mem &&
             sets[si]->buffers[b]->mem->map)
            smask |= 1u << (pipeline->set_base[si] + b);
   }
   const uint32_t umask = pipeline->ubo_mask & ~1u;
   if (ok && (smask || umask)) {
      const uint32_t slots = 1 + util_last_bit(smask | umask);
      const bool reuse = cmd->region_set == set && cmd->batch_draws && !umask && !any_dyn && !sets[1];
      uint32_t base = cmd->region_base;
      if (!reuse) {
         base = (heap.top + 3) & ~3u;
         if (base + slots * BORGVK_SSBO_SLOT_BYTES + 16 > BORGVK_HEAP_BYTES) {
            ok = generic_reject(10);
         } else {
            heap.top = base + slots * BORGVK_SSBO_SLOT_BYTES + 16;
            for (int si = 0; si < 4; si++) {
               for (int b = 0; sets[si] && b < BORGVK_MAX_BINDINGS; b++) {
                  const struct borgvk_descriptor_set *ds = sets[si];
                  const int fb = pipeline->set_base[si] + b;
                  struct borgvk_buffer *ub = ds->buffers[b];
                  if (fb >= 32 || !ub || !ub->mem || !ub->mem->map)
                     continue;
                  const bool st = smask >> fb & 1;
                  if (!st && pipeline->ubo_mask && !(pipeline->ubo_mask >> fb & 1))
                     continue;
                  const VkDeviceSize *dyn = si ? cmd->dyn_offs[si] : cmd->dyn_off;
                  VkDeviceSize off = ds->offsets[b] + ((ds->dyn_mask >> b) & 1 ? dyn[b] : 0);
                  if (off >= ub->vk.size)
                     continue;
                  VkDeviceSize len = ds->ranges[b] == VK_WHOLE_SIZE || ds->ranges[b] == 0 ? ub->vk.size - off : ds->ranges[b];
                  len = MIN2(len, MIN2(ub->vk.size - off, BORGVK_SSBO_SLOT_BYTES));
                  uint8_t *host = (uint8_t *)ub->mem->map + ub->offset + off;
                  const uint32_t at = base + (st || (fb && pipeline->ubo_mask) ? (1 + fb) * BORGVK_SSBO_SLOT_BYTES : 0);
                  for (uint32_t i = 0; i < len; i += 256) {
                     uint8_t tmp[256] = { 0 };
                     memcpy(tmp, host + i, MIN2(256u, (uint32_t)len - i));
                     borgvk_serial_send_mem(at + i, tmp, (MIN2(256u, (uint32_t)len - i) + 3) & ~3u);
                  }
                  if (st && cmd->nwb < BORGVK_MAX_WRITEBACKS)
                     cmd->wb[cmd->nwb++] = (struct borgvk_writeback){ at, (uint32_t)len, host };
               }
            }
            cmd->region_set = set;
            cmd->region_base = base;
         }
      }
      ubo_base = base + 1;
   }
   for (int b = 0; ok && !smask && !umask && set && b < BORGVK_MAX_BINDINGS; b++) {
      struct borgvk_buffer *ub = set->buffers[b];
      if (!ub || !ub->mem || !ub->mem->map || (pipeline->ubo_mask && !(pipeline->ubo_mask >> b & 1)))
         continue;
      VkDeviceSize off = set->offsets[b] + ((set->dyn_mask >> b) & 1 ? cmd->dyn_off[b] : 0);
      VkDeviceSize len = set->ranges[b] == VK_WHOLE_SIZE || set->ranges[b] == 0 ? ub->vk.size - off : set->ranges[b];
      if (off >= ub->vk.size)
         break;
      len = MIN2(len, MIN2(ub->vk.size - off, 65536));
      uint32_t at = heap_upload(&heap, (const uint8_t *)ub->mem->map + ub->offset + off, (uint32_t)len);
      if (at == UINT32_MAX)
         ok = generic_reject(10);
      else
         ubo_base = at + 1;
      break;
   }
   if (ok) {
      int want = -1;
      if (set && pipeline->tex_binding >= 0) {
         want = set->map.n ? -2 : pipeline->tex_binding;
         for (uint32_t j = 0; j < set->map.n; j++)
            if (set->map.num[j] == pipeline->tex_binding)
               want = set->map.slot[j];
         if (want < 0 || want >= BORGVK_MAX_BINDINGS)
            want = -1;
      }
      for (int b = 0; set && b < BORGVK_MAX_BINDINGS; b++) {
         if (want >= 0 && b != want)
            continue;
         if (set->views[b] && send_generic_texture(set->views[b], set->images[b], set->samplers[b]))
            break;
         if (set->buffer_views[b] && send_buffer_texture(set->buffer_views[b]))
            break;
      }
      if (lines)
         borgvk_serial_send_draw(0, 0, load << 1, nlines * 6, dp->instances, 0, dp->first_instance, 0, 0, ubo_base);
      else
      borgvk_serial_send_draw(topo, dp->indexed ? index_code : 0,
                              (pipeline->restart ? 1u : 0u) | (load << 1), dp->count * X, dp->instances,
                              dp->indexed ? 0 : dp->first * X, dp->first_instance, voff_out, idx_off, ubo_base);
   }
   size_t nbytes = 0;
   uint8_t *bytes = borgvk_transport_capture_end(&nbytes);
   device->drm_fd = saved_fd;
   if (!ok || !bytes || nbytes == 0) {
      free(bytes);
      free(state);
      return ok;
   }
   bool appended = (state_same || !state || stream_append(cmd, state, state_len)) &&
                   stream_append(cmd, bytes, nbytes);
   free(bytes);
   if (!state_same && state && appended) {
      free(cmd->state_sent);
      cmd->state_sent = state;
      cmd->state_sent_len = state_len;
   } else {
      free(state);
   }
   if (!appended)
      return false;
   cmd->heap_top = heap.top;
   cmd->batch_draws++;
   cmd->batch_img = color_img;
   cmd->batch_nextra = nextra;
   for (uint32_t k = 0; k < 3; k++)
      cmd->batch_extra[k] = extra[k];
   cmd->batch_depth = cmd->depth_view;
   cmd->batch_stencil = cmd->stencil_view;
   return true;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdBindPipeline(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                       VkPipeline _pipeline)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct borgvk_command_buffer *cmd = container_of(vk_cmd, struct borgvk_command_buffer, vk);
   VK_FROM_HANDLE(borgvk_pipeline, pl, _pipeline);
   if (pipelineBindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS)
      cmd->gfx_pipeline = pl;
   else if (pipelineBindPoint == VK_PIPELINE_BIND_POINT_COMPUTE)
      cmd->cs_pipeline = pl;
}

#define BORGVK_CS_SLOT_BYTES (0x80000u * 4)
#define BORGVK_CS_BASE_BYTES (0x40000u * 4)
#define BORGVK_MAX_CS_PARTS 64

static void
put_u32(uint8_t **p, uint32_t v)
{
   memcpy(*p, &v, 4);
   *p += 4;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdDispatch(VkCommandBuffer commandBuffer, uint32_t gx, uint32_t gy, uint32_t gz)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct borgvk_command_buffer *cmd = container_of(vk_cmd, struct borgvk_command_buffer, vk);
   const struct borgvk_pipeline *pl = cmd->cs_pipeline;
   const char *direct = getenv("BORGVK_SIM_DIRECT");
   if (!pl || !pl->cs_ok || !gx || !gy || !gz || !direct || !direct[0])
      return;
   struct { uint8_t *host; uint8_t *decoded; bool wb; uint32_t skip, size, addr;
            uint8_t *texels; enum pipe_format pf; } bufs[34];
   memset(bufs, 0, sizeof(bufs));
   uint32_t nbufs = 0, total = 0;
   for (uint32_t si = 0; si < 4; si++) {
   struct borgvk_descriptor_set *set = cmd->cs_sets[si];
   const VkDeviceSize *dyn_off = cmd->cs_dyn_offs[si];
   for (uint32_t b = 0; set && b < BORGVK_MAX_BINDINGS && pl->set_base[si] + b < 32; b++) {
      const struct vk_buffer_view *bv = set->buffer_views[b];
      if (bv && !set->buffers[b]) {
         struct borgvk_buffer *tb = container_of(bv->buffer, struct borgvk_buffer, vk);
         enum pipe_format pf = vk_format_to_pipe_format(bv->format);
         uint32_t bs = util_format_get_blocksize(pf);
         if (!tb->mem || !tb->mem->map || pf == PIPE_FORMAT_NONE || bs == 0)
            continue;
         uint32_t n = (uint32_t)MIN2(bv->range / bs, BORGVK_CS_SLOT_BYTES / 16);
         uint8_t *dec = calloc(MAX2(n, 1), 16);
         if (!dec)
            continue;
         const uint8_t *src = (const uint8_t *)tb->mem->map + tb->offset + bv->offset;
         for (uint32_t t = 0; t < n; t++)
            util_format_unpack_rgba(pf, dec + (size_t)t * 16, src + (size_t)t * bs, 1);
         bufs[nbufs].host = NULL;
         bufs[nbufs].decoded = dec;
         bufs[nbufs].wb = false;
         bufs[nbufs].texels = (uint8_t *)tb->mem->map + tb->offset + bv->offset;
         bufs[nbufs].pf = pf;
         bufs[nbufs].skip = 0;
         bufs[nbufs].size = n * 16;
         bufs[nbufs].addr = BORGVK_CS_BASE_BYTES + (pl->set_base[si] + b) * BORGVK_CS_SLOT_BYTES;
         total += 8 + n * 16;
         nbufs++;
         continue;
      }
      struct borgvk_buffer *buf = set->buffers[b];
      struct borgvk_image *simg = set->images[b];
      if (!buf && !bv && simg && (set->sampled_mask >> b & 1) && simg->mem && simg->mem->map &&
          simg->vk.image_type == VK_IMAGE_TYPE_2D && simg->vk.samples == 1) {
         enum pipe_format pf = vk_format_to_pipe_format(simg->vk.format);
         uint32_t w = simg->vk.extent.width, h = simg->vk.extent.height;
         uint32_t bs = util_format_get_blocksize(pf);
         if (pf == PIPE_FORMAT_NONE || bs == 0 || util_format_is_compressed(pf))
            continue;
         uint32_t n = MIN2(w * h, BORGVK_CS_SLOT_BYTES / 16 - 1);
         uint8_t *dec = calloc(1 + n, 16);
         if (!dec)
            continue;
         memcpy(dec, &w, 4);
         const uint8_t *src = (const uint8_t *)simg->mem->map + simg->offset;
         for (uint32_t t = 0; t < n; t++)
            util_format_unpack_rgba(pf, dec + 16 + (size_t)t * 16, src + (size_t)t * bs, 1);
         bufs[nbufs].host = NULL;
         bufs[nbufs].decoded = dec;
         bufs[nbufs].wb = false;
         bufs[nbufs].skip = 0;
         bufs[nbufs].size = (1 + n) * 16;
         bufs[nbufs].addr = BORGVK_CS_BASE_BYTES + (pl->set_base[si] + b) * BORGVK_CS_SLOT_BYTES;
         total += 8 + (1 + n) * 16;
         nbufs++;
         continue;
      }
      if (!buf && !bv && simg && simg->mem && simg->mem->map && simg->vk.image_type == VK_IMAGE_TYPE_2D &&
          vk_format_get_blocksize(simg->vk.format) == 4 && simg->vk.samples == 1) {
         uint32_t w = simg->vk.extent.width, h = simg->vk.extent.height;
         uint32_t n = MIN2(w * h, BORGVK_CS_SLOT_BYTES / 4 - 1);
         uint8_t *dec = calloc(1 + n, 4);
         if (!dec)
            continue;
         memcpy(dec, &w, 4);
         memcpy(dec + 4, (uint8_t *)simg->mem->map + simg->offset, (size_t)n * 4);
         bufs[nbufs].host = (uint8_t *)simg->mem->map + simg->offset;
         bufs[nbufs].decoded = dec;
         bufs[nbufs].wb = true;
         bufs[nbufs].skip = 4;
         bufs[nbufs].size = (1 + n) * 4;
         bufs[nbufs].addr = BORGVK_CS_BASE_BYTES + (pl->set_base[si] + b) * BORGVK_CS_SLOT_BYTES;
         total += 8 + (1 + n) * 4;
         nbufs++;
         continue;
      }
      if (!buf || !buf->mem || !buf->mem->map)
         continue;
      VkDeviceSize boff = set->offsets[b] + dyn_off[b];
      VkDeviceSize avail = buf->vk.size > boff ? buf->vk.size - boff : 0;
      VkDeviceSize size = set->ranges[b] == VK_WHOLE_SIZE || set->ranges[b] > avail ? avail : set->ranges[b];
      if (size > BORGVK_CS_SLOT_BYTES)
         size = BORGVK_CS_SLOT_BYTES;
      bufs[nbufs].host = (uint8_t *)buf->mem->map + buf->offset + boff;
      bufs[nbufs].decoded = NULL;
      bufs[nbufs].wb = true;
      bufs[nbufs].skip = 0;
      bufs[nbufs].size = (uint32_t)size;
      bufs[nbufs].addr = BORGVK_CS_BASE_BYTES + (pl->set_base[si] + b) * BORGVK_CS_SLOT_BYTES;
      total += 8 + ((uint32_t)size + 3) / 4 * 4;
      nbufs++;
   }
   }

   /* Push constants: one more window, slot 32. */
   {
      uint8_t *dec = calloc(1, sizeof(cmd->pc));
      if (dec) {
         memcpy(dec, cmd->pc, sizeof(cmd->pc));
         bufs[nbufs].host = NULL;
         bufs[nbufs].decoded = dec;
         bufs[nbufs].wb = false;
         bufs[nbufs].skip = 0;
         bufs[nbufs].size = sizeof(cmd->pc);
         bufs[nbufs].addr = BORGVK_CS_BASE_BYTES + 32 * BORGVK_CS_SLOT_BYTES;
         total += 8 + sizeof(cmd->pc);
         nbufs++;
      }
   }

   const uint32_t HDR = 12;
   size_t cap = HDR * 4 + pl->cs_nwords * 4 + pl->cs_nregs * 8 + total;
   uint8_t *job = malloc(cap), *w = job;
   put_u32(&w, pl->cs_nwords); put_u32(&w, gx); put_u32(&w, gy); put_u32(&w, gz);
   put_u32(&w, pl->cs_local[0]); put_u32(&w, pl->cs_local[1]); put_u32(&w, pl->cs_local[2]);
   put_u32(&w, pl->cs_nregs); put_u32(&w, nbufs);
   put_u32(&w, 0); put_u32(&w, 0); put_u32(&w, 0);
   for (uint32_t i = 0; i < pl->cs_nwords; i++) put_u32(&w, pl->cs_words[i]);
   for (uint32_t i = 0; i < pl->cs_nregs; i++) { put_u32(&w, pl->cs_regs[2 * i]); put_u32(&w, pl->cs_regs[2 * i + 1]); }
   for (uint32_t i = 0; i < nbufs; i++) {
      put_u32(&w, bufs[i].addr); put_u32(&w, bufs[i].size);
      memcpy(w, bufs[i].decoded ? bufs[i].decoded : bufs[i].host, bufs[i].size);
      memset(w + bufs[i].size, 0, ((bufs[i].size + 3) & ~3u) - bufs[i].size);
      w += (bufs[i].size + 3) & ~3u;
   }
   const size_t job_len = (size_t)(w - job);

   uint32_t axis = gy > 1 && gy >= gx ? 1 : gx > 1 ? 0 : 2;
   uint32_t extent = axis == 0 ? gx : axis == 1 ? gy : gz;
   uint32_t nparts = 1;
   if (!(pl->cs_flags & 1) && (uint64_t)gx * gy * gz >= 4096 && extent > 1) {
      long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
      nparts = (uint32_t)MIN2((long)extent, MAX2(ncpu, 1));
      if (getenv("BORGVK_CS_PARTS"))
         nparts = (uint32_t)MAX2(1, MIN2((long)extent, atol(getenv("BORGVK_CS_PARTS"))));
   }

   char jpath[BORGVK_MAX_CS_PARTS][32], opath[BORGVK_MAX_CS_PARTS][32];
   pid_t pids[BORGVK_MAX_CS_PARTS];
   nparts = MIN2(nparts, BORGVK_MAX_CS_PARTS);
   bool ran = true;
   for (uint32_t k = 0; k < nparts; k++) {
      snprintf(jpath[k], sizeof jpath[k], "/tmp/borgvk_cs_job_XXXXXX");
      snprintf(opath[k], sizeof opath[k], "/tmp/borgvk_cs_out_XXXXXX");
      pids[k] = -1;
      int jfd = mkstemp(jpath[k]), ofd = mkstemp(opath[k]);
      if (jfd < 0 || ofd < 0) { ran = false; continue; }
      uint32_t lo = extent * k / nparts, hi = extent * (k + 1) / nparts;
      uint32_t hdr[HDR];
      memcpy(hdr, job, sizeof hdr);
      hdr[1 + axis] = hi - lo;        /* gx gy gz */
      hdr[9 + axis] = lo;             /* bx by bz */
      if (write(jfd, hdr, sizeof hdr) != (ssize_t)sizeof hdr) ran = false;
      for (size_t o = sizeof hdr; o < job_len; ) {
         ssize_t n = write(jfd, job + o, job_len - o);
         if (n <= 0) break;
         o += (size_t)n;
      }
      close(jfd);
      close(ofd);
      pids[k] = fork();
      if (pids[k] == 0) {
         setenv("BORG_SIM_CFG", "simt", 1);
         execlp(direct, direct, "--compute", jpath[k], opath[k], (char *)NULL);
         _exit(127);
      }
   }
   uint8_t *merged[BORGVK_MAX_BINDINGS + 1] = { 0 };
   for (uint32_t i = 0; i < nbufs; i++) {
      merged[i] = malloc(((bufs[i].size + 3) & ~3u) + 1);
      memcpy(merged[i], bufs[i].decoded ? bufs[i].decoded : bufs[i].host, bufs[i].size);
   }
   for (uint32_t k = 0; k < nparts; k++) {
      int st = 0;
      if (pids[k] > 0)
         waitpid(pids[k], &st, 0);
      FILE *f = pids[k] > 0 && WIFEXITED(st) && WEXITSTATUS(st) == 0 ? fopen(opath[k], "rb") : NULL;
      if (!f) {
         mesa_logw("borgvk: compute sim failed (status %d)", st);
         ran = false;
         continue;
      }
      for (uint32_t i = 0; i < nbufs; i++) {
         uint8_t *tmp = malloc(((bufs[i].size + 3) & ~3u) + 1);
         const uint8_t *orig = bufs[i].decoded ? bufs[i].decoded : bufs[i].host;
         if (tmp && fread(tmp, 1, bufs[i].size, f) == bufs[i].size)
            for (uint32_t b = 0; b < bufs[i].size; b++)
               if (tmp[b] != orig[b])
                  merged[i][b] = tmp[b];
         free(tmp);
         fseek(f, ((bufs[i].size + 3) & ~3u) - bufs[i].size, SEEK_CUR);
      }
      fclose(f);
   }
   if (borgvk_corpus_on()) {
      uint64_t out = borgvk_corpus_hash(BORGVK_CORPUS_SEED, &ran, sizeof(ran));
      for (uint32_t i = 0; i < nbufs; i++)
         out = borgvk_corpus_hash(out, merged[i], bufs[i].size);
      borgvk_corpus_log('D', borgvk_corpus_hash(BORGVK_CORPUS_SEED, job, job_len), out);
   }
   if (ran) {
      for (uint32_t i = 0; i < nbufs; i++)
         if (bufs[i].wb)
            memcpy(bufs[i].host, merged[i] + bufs[i].skip, bufs[i].size - bufs[i].skip);
      /* A texel buffer: the texels a shader stored, encoded back into the view's format. */
      for (uint32_t i = 0; i < nbufs; i++) {
         if (!bufs[i].texels)
            continue;
         const uint32_t bs = util_format_get_blocksize(bufs[i].pf);
         for (uint32_t t = 0; t < bufs[i].size / 16; t++)
            if (memcmp(merged[i] + t * 16, bufs[i].decoded + t * 16, 16) != 0)
               util_format_pack_rgba(bufs[i].pf, bufs[i].texels + (size_t)t * bs, merged[i] + t * 16, 1);
      }
      cmd->dispatched = true;
   }
   for (uint32_t k = 0; k < nparts; k++) {
      if (!getenv("BORGVK_KEEP_JOB"))
         unlink(jpath[k]);
      unlink(opath[k]);
   }
   for (uint32_t i = 0; i < nbufs; i++) {
      free(merged[i]);
      free(bufs[i].decoded);
   }
   free(job);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdBindVertexBuffers(VkCommandBuffer commandBuffer, uint32_t firstBinding,
                            uint32_t bindingCount, const VkBuffer *pBuffers,
                            const VkDeviceSize *pOffsets)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct borgvk_command_buffer *cmd = container_of(vk_cmd, struct borgvk_command_buffer, vk);
   for (uint32_t i = 0; i < bindingCount; i++) {
      uint32_t b = firstBinding + i;
      VK_FROM_HANDLE(borgvk_buffer, buf, pBuffers[i]);
      if (b < BORGVK_MAX_VERTEX_BINDINGS) {
         cmd->vb[b] = buf && buf->mem && buf->mem->map
            ? (const uint8_t *)buf->mem->map + buf->offset + pOffsets[i] : NULL;
         cmd->vb_avail[b] = buf && buf->vk.size > pOffsets[i] ? buf->vk.size - pOffsets[i] : 0;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdBindDescriptorSets(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                             VkPipelineLayout layout, uint32_t firstSet, uint32_t descriptorSetCount,
                             const VkDescriptorSet *pDescriptorSets, uint32_t dynamicOffsetCount,
                             const uint32_t *pDynamicOffsets)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct borgvk_command_buffer *cmd = container_of(vk_cmd, struct borgvk_command_buffer, vk);
   const bool cs = pipelineBindPoint == VK_PIPELINE_BIND_POINT_COMPUTE;
   uint32_t k = 0;
   for (uint32_t i = 0; i < descriptorSetCount && pDescriptorSets; i++) {
      VK_FROM_HANDLE(borgvk_descriptor_set, set, pDescriptorSets[i]);
      const uint32_t si = firstSet + i;
      if (!set || si > 3)
         continue;
      VkDeviceSize *dyn = cs ? cmd->cs_dyn_offs[si] : si ? cmd->dyn_offs[si] : cmd->dyn_off;
      if (cs)
         cmd->cs_sets[si] = set;
      else if (si)
         cmd->desc_sets[si] = set;
      else
         cmd->desc_set = set;
      memset(dyn, 0, sizeof(cmd->dyn_off));
      for (uint32_t b = 0; b < BORGVK_MAX_BINDINGS && k < dynamicOffsetCount; b++)
         if (set->dyn_mask & (1ull << b))
            dyn[b] = pDynamicOffsets[k++];
   }
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdBindIndexBuffer(VkCommandBuffer commandBuffer, VkBuffer _buffer, VkDeviceSize offset,
                          VkIndexType indexType)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct borgvk_command_buffer *cmd = container_of(vk_cmd, struct borgvk_command_buffer, vk);
   VK_FROM_HANDLE(borgvk_buffer, buf, _buffer);
   cmd->index_ptr = NULL;
   cmd->index_avail = 0;
   cmd->index_size = indexType == VK_INDEX_TYPE_UINT16 ? 2 : indexType == VK_INDEX_TYPE_UINT32 ? 4 : 0;
   if (buf && buf->mem && buf->mem->map && buf->vk.size > offset) {
      cmd->index_ptr = (const uint8_t *)buf->mem->map + buf->offset + offset;
      cmd->index_avail = buf->vk.size - offset;
   }
}

static void
sim_draw(VkCommandBuffer commandBuffer, const struct draw_params *dp)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct borgvk_command_buffer *cmd = container_of(vk_cmd, struct borgvk_command_buffer, vk);
   struct borgvk_device *device = container_of(vk_cmd->base.device, struct borgvk_device, vk);

   /* Only the simulator renders; cube.c (UBO-driven) has its own path at submit. */
   if (!(getenv("BORGVK_SIM") || borgvk_hw_enabled()) || (cmd->desc_set && set_is_cube(cmd->desc_set)))
      return;
   struct draw_params p = *dp;
   p.index_size = cmd->index_size;
   if (p.indexed && p.index_size == 0)
      return;
   if (borgvk_sim_generic_draw(device, cmd, &p))
      cmd->generic_drawn = true;
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount,
               uint32_t firstVertex, uint32_t firstInstance)
{
   struct draw_params dp = { .count = vertexCount, .instances = instanceCount, .first = firstVertex,
                             .first_instance = firstInstance };
   sim_draw(commandBuffer, &dp);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount, uint32_t instanceCount,
                      uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance)
{
   struct draw_params dp = { .indexed = true, .count = indexCount, .instances = instanceCount,
                             .first = firstIndex, .vertex_offset = vertexOffset,
                             .first_instance = firstInstance };
   sim_draw(commandBuffer, &dp);
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer, VkDeviceSize offset,
                       uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(borgvk_buffer, buf, _buffer);
   if (!buf || !buf->mem || !buf->mem->map)
      return;
   for (uint32_t i = 0; i < drawCount; i++) {
      VkDrawIndirectCommand c;
      memcpy(&c, (const uint8_t *)buf->mem->map + buf->offset + offset + (VkDeviceSize)i * (stride ? stride : sizeof(c)),
             sizeof(c));
      struct draw_params dp = { .count = c.vertexCount, .instances = c.instanceCount, .first = c.firstVertex,
                                .first_instance = c.firstInstance };
      sim_draw(commandBuffer, &dp);
   }
}

VKAPI_ATTR void VKAPI_CALL
borgvk_CmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer, VkDeviceSize offset,
                              uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(borgvk_buffer, buf, _buffer);
   if (!buf || !buf->mem || !buf->mem->map)
      return;
   for (uint32_t i = 0; i < drawCount; i++) {
      VkDrawIndexedIndirectCommand c;
      memcpy(&c, (const uint8_t *)buf->mem->map + buf->offset + offset + (VkDeviceSize)i * (stride ? stride : sizeof(c)),
             sizeof(c));
      struct draw_params dp = { .indexed = true, .count = c.indexCount, .instances = c.instanceCount,
                                .first = c.firstIndex, .vertex_offset = c.vertexOffset,
                                .first_instance = c.firstInstance };
      sim_draw(commandBuffer, &dp);
   }
}

static VkResult borgvk_queue_submit_work(struct vk_queue *vk_queue, struct vk_queue_submit *submit);

VkResult
borgvk_queue_submit(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct vk_device *vk_dev = vk_queue->base.device;

   for (uint32_t i = 0; i < submit->wait_count; i++) {
      VkResult r = vk_sync_wait(vk_dev, submit->waits[i].sync, submit->waits[i].wait_value,
                                VK_SYNC_WAIT_COMPLETE, UINT64_MAX);
      if (r != VK_SUCCESS)
         return r;
   }

   VkResult result = borgvk_queue_submit_work(vk_queue, submit);

   for (uint32_t i = 0; i < submit->signal_count; i++) {
      VkResult r = vk_sync_signal(vk_dev, submit->signals[i].sync, submit->signals[i].signal_value);
      if (r != VK_SUCCESS && result == VK_SUCCESS)
         result = r;
   }
   return result;
}

static VkResult
borgvk_queue_submit_work(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct borgvk_device *replay_dev =
      container_of(vk_queue->base.device, struct borgvk_device, vk);
   for (uint32_t ci = 0; ci < submit->command_buffer_count; ci++) {
      struct vk_command_buffer *cb = submit->command_buffers[ci];
      vk_cmd_queue_execute(&cb->cmd_queue, vk_command_buffer_to_handle(cb), &replay_dev->cmd_dispatch);
      borgvk_flush_draws(container_of(cb, struct borgvk_command_buffer, vk));   /* draws outside a pass end */
   }

   struct borgvk_device *device =
      container_of(vk_queue->base.device, struct borgvk_device, vk);

   if (getenv("BORGVK_SIM") || borgvk_hw_enabled()) {
      if (submit_is_cube(submit))
         return borgvk_submit_sim_cube(device, submit);
      return VK_SUCCESS;
   }

   struct borgvk_descriptor_set *set = find_set(submit);
   if (!set)
      return VK_SUCCESS;

   struct borgvk_buffer *ubuf = set->buffers[0];
   if (!ubuf || !ubuf->mem || !ubuf->mem->map)
      return VK_SUCCESS;
   VkDeviceSize off = ubuf->offset + set->offsets[0];
   if (off >= ubuf->vk.size)
      return VK_SUCCESS;
   const float *ubo = (const float *)((const char *)ubuf->mem->map + off);
   uint32_t nfloats = (uint32_t)((ubuf->vk.size - off) / sizeof(float));

   struct borgvk_image *tex = set->images[1];
   bool can_geom = nfloats >= UBO_MIN_FLOATS;
   bool can_tex  = tex && tex->mem && tex->mem->map &&
                   tex->vk.extent.width && tex->vk.extent.height;

   static bool g_setup_done = false;

   if (device->drm_fd >= 0) {
      bool has_gem = ubuf->mem->gem_handle != 0 &&
                     tex && tex->mem && tex->mem->gem_handle != 0;

      if (!g_setup_done && has_gem) {
         upload_shaders(device);   /* borgc-compiled shaders, before geom/tex */
         struct drm_borg_setup s = {
            .ubo_handle = ubuf->mem->gem_handle,
            .tex_handle = tex->mem->gem_handle,
            .tex_offset = (__u64)tex->offset,
            .tex_width  = tex->vk.extent.width,
            .tex_height = tex->vk.extent.height,
         };
         memcpy(s.sampler, sampler_desc(set->samplers[1]), sizeof(s.sampler));
         if (drmIoctl(device->drm_fd, DRM_IOCTL_BORG_SETUP, &s) == 0)
            g_setup_done = true;
         else
            mesa_logw("borgvk: DRM_IOCTL_BORG_SETUP failed");
      }

      if (g_setup_done) {
         send_blend_state(device);
         struct drm_borg_submit sub = { .ubo_handle = ubuf->mem->gem_handle };
         drmIoctl(device->drm_fd, DRM_IOCTL_BORG_SUBMIT, &sub);
      }
   } else {
      if (!g_setup_done) {
         if (setup_already_done()) {
            mesa_logi("borgvk: skipping upload (sentinel present); "
                      "set BORGVK_FORCE_UPLOAD=1 to re-upload");
            g_setup_done = true;
         } else if (can_geom && can_tex) {
            mesa_logi("borgvk: uploading shaders + geometry + texture burst...");
            upload_shaders(device);   /* borgc-compiled shaders, before geom/tex */
            send_geometry(ubo);
            for (int row = 0; row < BORGVK_TEX_DIM; row++)
               send_texture_row(tex, set->samplers[1], row);
            mark_setup_done();
            mesa_logi("borgvk: upload complete");
            g_setup_done = true;
         }
      }

      if (g_setup_done) {
         send_blend_state(device);
         borgvk_serial_send_mvp(ubo);
      }
   }

   return VK_SUCCESS;
}

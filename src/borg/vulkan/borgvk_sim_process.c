/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"

#include <fcntl.h>
#include <sys/wait.h>

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

VkResult
borgvk_sim_run_stream(uint8_t *bytes, size_t nbytes, struct borgvk_image *color_img,
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

/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 *
 * borgvk_hw.c — the Borg behind the DRM render node (or `direct_sim --raw` on a host).
 *
 * The wire stream a frame would send over the serial link is run through borg_core
 * (software/borg) in this process; its register and memory hooks (borg_hw.c) are ioctls
 * on /dev/dri/renderD128. The colour attachment comes back by reading GPU memory.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "borgvk_private.h"
#include "software/borg/borg_core.h"
#include "software/borg/borg_hw.h"
#include "software/borg/borg_layout.h"

bool
borgvk_hw_enabled(void)
{
   const char *e = getenv("BORGVK_HW");
   return e && e[0];
}

/* Render `n` bytes of wire stream on a dim x dim target; the pixels (RGB888, or the raw
 * 32-bit words of a RAW32 target) go to `out`. Returns the number of bytes, 0 on failure. */
size_t
borgvk_hw_render(const uint8_t *b, size_t n, uint32_t dim, uint8_t *out)
{
   if (borg_hw_open()) {
      mesa_logw("borgvk: cannot open the Borg device");
      return 0;
   }
   borg_core_set_flush_format(0);
   borg_core_init(dim, dim);

   int draws = 0;
   bool target_seen = false;   /* a 0xB6 packet carries the app's clear colour; otherwise the default */
   for (size_t i = 0; i < n;) {
      if (b[i] == 0xB1) { i++; continue; }   /* serial-reload trigger: not a draw packet */
      int len = borg_core_pkt_len(b[i]);
      if (!len || i + (size_t)len > n) {
         mesa_logw("borgvk: bad packet 0x%02x at byte %zu", b[i], i);
         break;
      }
      int kind = borg_core_packet(&b[i]);
      if (kind == BC_BAD) {
         mesa_logw("borgvk: rejected packet 0x%02x at byte %zu", b[i], i);
         break;
      }
      if (kind == BC_TARGET)
         target_seen = true;
      if (kind == BC_DRAW || (kind == BC_MVP && borg_core_ready())) {
         if (!target_seen)
            borg_core_set_clear(0x3266, 0x3266, 0x3266);
         if (kind == BC_DRAW)
            borg_core_list_draw();
         else
            borg_core_draw(NULL, 0);
         draws++;
      }
      i += (size_t)len;
   }
   borg_core_list_flush();
   borg_hw_flush();
   if (!draws)
      return 0;

   const int fmt = borg_core_flush_format();   /* 0 = R5G6B5, 1 = RGBA8, 2 = BGRA8, 3 = RAW32 */
   const size_t bytes = (size_t)dim * dim * (fmt ? 4 : 2);
   uint32_t *words = malloc(bytes);
   if (!words)
      return 0;
   borg_hw_mem_read(DRAM_OUT_BASE_SPI, words, (uint32_t)bytes);

   size_t ret = 0;
   for (uint32_t y = 0; y < dim; y++) {
      for (uint32_t x = 0; x < dim; x++) {
         const uint32_t tile = (y >> 2) * (dim >> 2) + (x >> 2), ti = (x & 3) | ((y & 3) << 2);
         if (fmt == 3) {
            memcpy(out + ((size_t)y * dim + x) * 4, &words[tile * 16 + ti], 4);
            continue;
         }
         uint8_t r, g, bl;
         if (fmt) {   /* tile = 16 pixels x 4 B: R,G,B,A (fmt 1) or B,G,R,A (fmt 2) */
            uint32_t px = words[tile * 16 + ti];
            uint8_t b0 = px & 0xFF, b1 = (px >> 8) & 0xFF, b2 = (px >> 16) & 0xFF;
            r = fmt == 1 ? b0 : b2; g = b1; bl = fmt == 1 ? b2 : b0;
         } else {
            uint32_t word = words[tile * 8 + (ti >> 1)];
            uint16_t px = (ti & 1) ? (uint16_t)(word >> 16) : (uint16_t)word;
            r = ((px >> 11) & 0x1F) << 3; g = ((px >> 5) & 0x3F) << 2; bl = (px & 0x1F) << 3;
            r |= r >> 5; g |= g >> 6; bl |= bl >> 5;
         }
         uint8_t *o = out + ((size_t)y * dim + x) * 3;
         o[0] = r; o[1] = g; o[2] = bl;
      }
   }
   ret = (size_t)dim * dim * (fmt == 3 ? 4 : 3);
   free(words);
   return ret;
}

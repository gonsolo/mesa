/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
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

size_t
borgvk_hw_render(const uint8_t *b, size_t n, uint32_t w, uint32_t h, uint8_t *out)
{
   if (borg_hw_open()) {
      mesa_logw("borgvk: cannot open the Borg device");
      return 0;
   }
   borg_core_set_flush_format(0);
   borg_core_init(w, h);

   int draws = 0;
   bool target_seen = false;
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
   const size_t bytes = (size_t)w * h * (fmt ? 4 : 2);
   uint32_t *words = malloc(bytes);
   if (!words)
      return 0;
   borg_hw_mem_read(DRAM_OUT_BASE_SPI, words, (uint32_t)bytes);

   size_t ret = 0;
   for (uint32_t y = 0; y < h; y++) {
      for (uint32_t x = 0; x < w; x++) {
         const uint32_t tile = (y >> 2) * (w >> 2) + (x >> 2), ti = (x & 3) | ((y & 3) << 2);
         if (fmt == 3) {
            memcpy(out + ((size_t)y * w + x) * 4, &words[tile * 16 + ti], 4);
            continue;
         }
         uint8_t r, g, bl;
         if (fmt) {
            uint32_t px = words[tile * 16 + ti];
            uint8_t b0 = px & 0xFF, b1 = (px >> 8) & 0xFF, b2 = (px >> 16) & 0xFF;
            r = fmt == 1 ? b0 : b2; g = b1; bl = fmt == 1 ? b2 : b0;
         } else {
            uint32_t word = words[tile * 8 + (ti >> 1)];
            uint16_t px = (ti & 1) ? (uint16_t)(word >> 16) : (uint16_t)word;
            r = ((px >> 11) & 0x1F) << 3; g = ((px >> 5) & 0x3F) << 2; bl = (px & 0x1F) << 3;
            r |= r >> 5; g |= g >> 6; bl |= bl >> 5;
         }
         uint8_t *o = out + ((size_t)y * w + x) * 3;
         o[0] = r; o[1] = g; o[2] = bl;
      }
   }
   ret = (size_t)w * h * (fmt == 3 ? 4 : 3);
   free(words);
   return ret;
}

/* The register/memory program of direct_sim --compute (simulation/direct/main.cpp), on the device. */
int
borgvk_hw_compute(const uint8_t *job, size_t len, const char *outf)
{
   enum { GPR = 0, IMEM = 128, CONTROL = 424, LS_BASE = 724, C_CTRL = 752, C_PC = 756, C_XY = 760,
          C_Z = 764, C_LOCAL = 768, CODE_BASE = 0x318 };
   const uint32_t IMEM_WORDS = 72, CODE_ADDR = 0xF0000;
   if (len < 48 || borg_hw_open())
      return -1;
   const uint32_t *h = (const uint32_t *)job;
   const uint32_t nprog = h[0], gx = h[1], gy = h[2], gz = h[3], lx = h[4], ly = h[5], lz = h[6],
                  nregs = h[7], nbufs = h[8], bx = h[9], by = h[10], bz = h[11];
   size_t p = 48;
   const uint32_t *prog = (const uint32_t *)(job + p);
   p += (size_t)nprog * 4;
   bh_reg_write(CONTROL, 2);
   bh_reg_write(LS_BASE, 0);
   if (nprog > IMEM_WORDS) {
      uint32_t *code = calloc(nprog + 1, 4);
      if (!code)
         return -1;
      memcpy(code, prog, (size_t)nprog * 4);
      borg_hw_mem_write(CODE_ADDR, code, (nprog + 1) * 4);
      free(code);
      bh_reg_write(CODE_BASE, CODE_ADDR);
   }
   for (uint32_t i = 0; i < nprog && i < IMEM_WORDS; i++)
      bh_reg_write(IMEM + 4 * i, prog[i]);
   if (nprog < IMEM_WORDS)
      bh_reg_write(IMEM + 4 * nprog, 0);
   for (uint32_t i = 0; i < nregs; i++, p += 8) {
      const uint32_t *rv = (const uint32_t *)(job + p);
      bh_reg_write(GPR + 4 * rv[0], rv[1]);
   }
   if (bx | by | bz) {
      bh_reg_write(GPR + 4 * 12, bx);
      bh_reg_write(GPR + 4 * 13, by);
      bh_reg_write(GPR + 4 * 14, bz);
   }
   uint32_t addr[BORGVK_CS_MAX_BUFS], size[BORGVK_CS_MAX_BUFS];
   if (nbufs > BORGVK_CS_MAX_BUFS)
      return -1;
   for (uint32_t i = 0; i < nbufs; i++) {
      const uint32_t *bh = (const uint32_t *)(job + p);
      addr[i] = bh[0];
      size[i] = bh[1];
      p += 8;
      borg_hw_mem_write(addr[i], job + p, (size[i] + 3) & ~3u);
      p += (size[i] + 3) & ~3u;
   }
   bh_reg_write(C_PC, 0);
   bh_reg_write(C_XY, gx | (gy << 16));
   bh_reg_write(C_Z, gz);
   bh_reg_write(C_LOCAL, lx | (ly << 8) | ((lx * ly * lz) << 16));
   bh_reg_write(C_CTRL, 1);
   uint32_t st = 0;
   for (int n = 0; n < 2000000 && !(st & 1); n++)
      st = bh_reg_read(C_CTRL);
   if (!(st & 1) || (st & 8))
      return -1;
   FILE *o = fopen(outf, "wb");
   if (!o)
      return -1;
   for (uint32_t i = 0; i < nbufs; i++) {
      uint32_t padded = (size[i] + 3) & ~3u;
      uint8_t *tmp = malloc(padded);
      if (!tmp) {
         fclose(o);
         return -1;
      }
      borg_hw_mem_read(addr[i], tmp, padded);
      fwrite(tmp, 1, size[i], o);
      free(tmp);
   }
   fclose(o);
   return 0;
}

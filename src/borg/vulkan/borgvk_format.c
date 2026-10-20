/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"

void
borgvk_tex_convert(const struct tex_conv *c, uint8_t *dst, const uint8_t *src, uint32_t n)
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

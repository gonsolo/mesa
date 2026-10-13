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

bool
borgvk_vertex_format_to_fetch(VkFormat f, uint32_t *code, uint32_t *swz, uint32_t *read_bytes)
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

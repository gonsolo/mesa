/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_image.h"


#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures;

static void
check(bool cond, const char *what)
{
   if (!cond) {
      fprintf(stderr, "FAIL: %s\n", what);
      failures++;
   }
}

static void
test_vertex_formats(void)
{
   uint32_t code, swz, rb;
   check(borgvk_vertex_format_to_fetch(VK_FORMAT_R32G32B32A32_SFLOAT, &code, &swz, &rb) &&
         code == 41 && swz == 0 && rb == 16, "rgba32f fetch");
   check(borgvk_vertex_format_to_fetch(VK_FORMAT_R32G32B32_SFLOAT, &code, &swz, &rb) &&
         code == 41 && swz != 0 && rb == 16, "rgb32f fetch forces alpha one");
   check(borgvk_vertex_format_to_fetch(VK_FORMAT_R8_UNORM, &code, &swz, &rb) && code == 1 && rb == 1,
         "r8 unorm fetch");
   check(!borgvk_vertex_format_to_fetch(VK_FORMAT_UNDEFINED, &code, &swz, &rb), "undefined has no fetch");
   check(!borgvk_vertex_format_to_fetch(VK_FORMAT_R64_SFLOAT, &code, &swz, &rb), "r64 has no fetch");
}

static void
test_tex_convert(void)
{
   const uint8_t bgra[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
   uint8_t out[8] = { 0 };
   struct tex_conv swap = { .bpp = 4, .obpp = 4, .cs = 1, .sch = 4, .swap_rb = true };
   borgvk_tex_convert(&swap, out, bgra, 2);
   const uint8_t rgba[8] = { 3, 2, 1, 4, 7, 6, 5, 8 };
   check(memcmp(out, rgba, 8) == 0, "tex convert swaps red and blue");

   struct tex_conv copy = { .bpp = 4, .obpp = 4 };
   borgvk_tex_convert(&copy, out, bgra, 2);
   check(memcmp(out, bgra, 8) == 0, "tex convert plain copy");

   const uint8_t rgb[3] = { 9, 8, 7 };
   struct tex_conv pad = { .bpp = 3, .obpp = 4, .cs = 1, .sch = 3, .alpha = 255 };
   borgvk_tex_convert(&pad, out, rgb, 1);
   check(out[0] == 9 && out[1] == 8 && out[2] == 7 && out[3] == 255, "tex convert pads alpha");
}

static void
test_image_helpers(void)
{
   uint32_t wb = 0;
   check(borgvk_mip_level_size(16, 16, 1, 0, 4, 1, 1, 1, &wb) == 1024 && wb == 16, "mip 0 size");
   check(borgvk_mip_level_size(16, 16, 1, 4, 4, 1, 1, 1, NULL) == 4, "last mip is one texel");
   check(borgvk_mip_level_size(16, 16, 1, 9, 4, 1, 1, 1, NULL) == 4, "mip below one texel stays one");
   check(borgvk_mip_level_size(10, 10, 1, 0, 8, 4, 4, 1, &wb) == 72 && wb == 3, "block compressed size rounds up");

   uint8_t texel[3] = { 0 };
   borgvk_z16_s8_pack_z(texel, 1.0f);
   check(texel[0] == 0xFF && texel[1] == 0xFF && borgvk_z16_s8_unpack_z(texel) == 1.0f, "z16 one round trip");
   borgvk_z16_s8_pack_z(texel, -3.0f);
   check(borgvk_z16_s8_unpack_z(texel) == 0.0f, "z16 clamps below zero");
   borgvk_z16_s8_pack_z(texel, 0.5f);
   check(fabsf(borgvk_z16_s8_unpack_z(texel) - 0.5f) < 1.0f / 65535.0f, "z16 half round trip");
   check(borgvk_is_z16_s8(PIPE_FORMAT_Z16_UNORM_S8_UINT) && !borgvk_is_z16_s8(PIPE_FORMAT_R8G8B8A8_UNORM),
         "z16s8 format test");
}

int
main(void)
{
   test_vertex_formats();
   test_tex_convert();
   test_image_helpers();
   if (failures) {
      fprintf(stderr, "%d check(s) failed\n", failures);
      return 1;
   }
   printf("borgvk formats and image helpers: all checks passed\n");
   return 0;
}

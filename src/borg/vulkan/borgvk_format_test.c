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
   test_image_helpers();
   if (failures) {
      fprintf(stderr, "%d check(s) failed\n", failures);
      return 1;
   }
   printf("borgvk formats and image helpers: all checks passed\n");
   return 0;
}

/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"

#include <stdlib.h>

bool
borgvk_sim_serves(const struct borgvk_image *color, bool has_depth_stencil)
{
   return color && borgvk_sim_serves_extent(color->vk.extent.width, color->vk.extent.height, has_depth_stencil);
}

bool
borgvk_sim_serves_extent(uint32_t w, uint32_t h, bool has_depth_stencil)
{
   const char *direct = getenv("BORGVK_SIM_DIRECT");
   if (!direct || !direct[0])
      direct = NULL;
   if (!direct && !borgvk_hw_enabled())
      return false;
   const uint32_t max_dim = has_depth_stencil ? 1024 : 4096;
   return w >= 1 && w <= max_dim && h >= 1 && h <= max_dim;
}

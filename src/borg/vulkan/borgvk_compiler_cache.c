/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"
#include "util/log.h"

void
borgvk_compile_stage(struct borgvk_device *device, uint32_t vfetch,
                     const VkPipelineShaderStageCreateInfo *stage_info)
{
   const int slot = stage_info->stage == VK_SHADER_STAGE_VERTEX_BIT ? BORGVK_STAGE_VERT :
                    stage_info->stage == VK_SHADER_STAGE_FRAGMENT_BIT ? BORGVK_STAGE_FRAG : -1;
   if (slot < 0)
      return;
   bool pntc;
   if (borgvk_shader_cache_get(stage_info, vfetch, &device->shader_blob[slot], &pntc)) {
      if (slot == BORGVK_STAGE_FRAG)
         device->frag_reads_pntc = pntc;
      return;
   }
   device->shader_blob[slot].len = 0;
   mesa_logw("borgvk: shader not in the cache and no compiler in this build");
}

void
borgvk_compile_compute_stage(struct borgvk_device *device,
                             const VkPipelineShaderStageCreateInfo *stage_info,
                             struct borgvk_pipeline *pipeline)
{
   pipeline->cs_ok = false;
}

void
borgvk_compile_empty_frag(struct borgvk_device *device, uint32_t opt)
{
   device->shader_blob[BORGVK_STAGE_FRAG].len = 0;
}

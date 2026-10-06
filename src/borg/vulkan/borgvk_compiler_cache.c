/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 *
 * The shader compiler of a borgvk without borgc (-Dborg-compiler=cache): stages come from
 * the shader cache ($BORGVK_SHADER_CACHE, borgvk_shader_cache.c) or are not compiled.
 */
#include "borgvk_private.h"
#include "util/log.h"

void
borgvk_compiler_selftest(void)
{
   mesa_logi("borgvk: shader compiler is cache-only");
}

uint32_t
borgvk_blob_num_varyings(const struct borgvk_shader_blob *b)
{
   if (b->len < 6 || !(b->data[5] & 1))
      return 0;
   const uint32_t at = 6 + 4 * b->data[0] + b->data[3] + b->data[4] + 4 * b->data[4];
   return at < b->len ? b->data[at] : 0;
}

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

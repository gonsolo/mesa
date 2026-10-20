/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */
#include "borgvk_private.h"
#include "nir_builder.h"
#include "borg_nir_passes.h"

#include "vk_pipeline.h"
#include "vk_pipeline_layout.h"

#include "nir.h"
#include "glsl_types.h"
#include "spirv/nir_spirv.h"
#include "util/hash_table.h"
#include "util/ralloc.h"
#include "util/log.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/* Turn one pipeline shader stage's SPIR-V into NIR and hand it to borgc. */
/* A pipeline without a fragment stage (depth only): a shader that writes nothing. */
void
borgvk_compile_empty_frag(struct borgvk_device *device, uint32_t opt)
{
   struct borgvk_shader_blob *b = &device->shader_blob[BORGVK_STAGE_FRAG];
   nir_builder nb = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &borg_nir_options, "borgvk_empty_fs");
   borg_lower_nir_for_borgc(nb.shader);
   uint32_t len = 0;
   borgc_compile_nir(nb.shader, b->data, BORGVK_SHADER_BLOB_MAX, &len, opt, -1, NULL);
   b->len = len <= BORGVK_SHADER_BLOB_MAX ? len : 0;
   b->npool = 0;
   device->frag_reads_pntc = false;
   device->frag_tex_binding = -1;
   device->stage_ubo[1] = 0;
   device->stage_push[0] = device->stage_push[1] = false;
   device->stage_pwin[1] = false;
   device->stage_spill[1] = false;
   ralloc_free(nb.shader);
}

void
borgvk_compile_stage(struct borgvk_device *device, uint32_t vfetch,
                     const VkPipelineShaderStageCreateInfo *stage_info)
{
   const int cslot = stage_info->stage == VK_SHADER_STAGE_VERTEX_BIT ? BORGVK_STAGE_VERT :
                     stage_info->stage == VK_SHADER_STAGE_FRAGMENT_BIT ? BORGVK_STAGE_FRAG : -1;
   const struct borg_blend *blend = stage_info->stage == VK_SHADER_STAGE_FRAGMENT_BIT ? device->blend_lower : NULL;
   if (cslot >= 0 && !blend) {   /* the cache does not know the blend state */
      bool pntc;
      if (borgvk_shader_cache_get(stage_info, vfetch, &device->shader_blob[cslot], &pntc)) {
         if (cslot == BORGVK_STAGE_FRAG) {
            device->frag_reads_pntc = pntc;
            device->frag_tex_binding = -1;
         }
         device->stage_ubo[cslot == BORGVK_STAGE_FRAG] = 0;
         device->stage_push[cslot == BORGVK_STAGE_FRAG] = false;
         device->stage_pwin[cslot == BORGVK_STAGE_FRAG] = false;
         device->stage_spill[cslot == BORGVK_STAGE_FRAG] = false;
         device->shader_blob[cslot].npool = 0;
         mesa_logi("borgvk: shader blob from the cache (%u bytes)", device->shader_blob[cslot].len);
         return;
      }
   }
   struct nir_shader *nir = NULL;
   VkResult result =
      vk_pipeline_shader_stage_to_nir(&device->vk, 0, stage_info,
                                      &borg_spirv_options, &borg_nir_options,
                                      NULL, &nir);
   if (result != VK_SUCCESS || nir == NULL) {
      mesa_logw("borgvk: SPIR-V→NIR failed for stage 0x%x (%d)",
                stage_info->stage, result);
      return;
   }

   borgvk_nir_remap_array_bindings(device, nir);
   if (stage_info->stage == VK_SHADER_STAGE_VERTEX_BIT && (vfetch & (BORGC_VS_FETCH | BORGC_VS_POINT_QUADS | BORGC_VS_LINES_BIT)) == (BORGC_VS_FETCH | BORGC_VS_LINES_BIT))
      borgvk_lower_lines(nir, vfetch & 0xFFFFu);
   nir_shader *retry = stage_info->stage == VK_SHADER_STAGE_VERTEX_BIT ? nir_shader_clone(NULL, nir) : NULL;
   borg_lower_nir_for_borgc_blend(nir, blend);
   device->frag_reads_pntc = stage_info->stage == VK_SHADER_STAGE_FRAGMENT_BIT && borgvk_nir_reads_point_coord(nir);
   if (stage_info->stage == VK_SHADER_STAGE_FRAGMENT_BIT)
      device->frag_tex_binding = borgvk_nir_sampled_binding(nir);
   if (stage_info->stage == VK_SHADER_STAGE_VERTEX_BIT || stage_info->stage == VK_SHADER_STAGE_FRAGMENT_BIT)
{
      const int fs = stage_info->stage == VK_SHADER_STAGE_FRAGMENT_BIT;
      device->stage_ubo[fs] = borgvk_nir_ubo_mask(nir);
      device->stage_push[fs] = borgvk_nir_uses_push_constants(nir);
   }

   int slot = -1;
   switch (stage_info->stage) {
   case VK_SHADER_STAGE_VERTEX_BIT:   slot = BORGVK_STAGE_VERT; break;
   case VK_SHADER_STAGE_FRAGMENT_BIT: slot = BORGVK_STAGE_FRAG; break;
   default: break;
   }

   if (slot >= 0) {
      struct borgvk_shader_blob *b = &device->shader_blob[slot];
      uint32_t len = 0;
      const int32_t out_select = slot == BORGVK_STAGE_FRAG ? (int)device->frag_select - 1 : -1;
      struct borgc_info info;
      borgc_compile_nir(nir, b->data, BORGVK_SHADER_BLOB_MAX, &len, vfetch, out_select, &info);
      if (!len && retry) {
         borg_lower_set_vertex_remat(true);
         borg_lower_nir_for_borgc_blend(retry, blend);
         borg_lower_set_vertex_remat(false);
         borgc_compile_nir(retry, b->data, BORGVK_SHADER_BLOB_MAX, &len, vfetch, out_select, &info);
      }
      device->stage_pwin[slot == BORGVK_STAGE_FRAG] = info.push_window;
      device->stage_spill[slot == BORGVK_STAGE_FRAG] = info.spill_slots > 0;
      if (len > 0 && len <= BORGVK_SHADER_BLOB_MAX) {
         b->len = len;
         b->npool = MIN2(info.npool, BORGVK_POOL_MAX);
         memcpy(b->pool, info.pool, 4 * b->npool);
         if (!blend && !b->npool)   /* the cache holds neither */
            borgvk_shader_cache_record(stage_info, vfetch, b, device->frag_reads_pntc);
         mesa_logi("borgvk: captured %s shader blob (%u bytes) for upload",
                   slot == BORGVK_STAGE_VERT ? "vertex" : "fragment", len);
      } else {
         b->len = 0;
         mesa_logw("borgvk: %s shader blob not captured (size %u, max %u)",
                   slot == BORGVK_STAGE_VERT ? "vertex" : "fragment",
                   len, BORGVK_SHADER_BLOB_MAX);
      }
   } else {
      borgc_compile_nir(nir, NULL, 0, NULL, 0, -1, NULL);   /* diagnostics only */
   }
   ralloc_free(retry);
   ralloc_free(nir);
}

/* Compute pipeline: SPIR-V -> NIR -> borgc's compute backend. */
void
borgvk_compile_compute_stage(struct borgvk_device *device,
                             const VkPipelineShaderStageCreateInfo *stage_info,
                             struct borgvk_pipeline *pipeline)
{
   struct nir_shader *nir = NULL;
   VkResult result =
      vk_pipeline_shader_stage_to_nir(&device->vk, 0, stage_info,
                                      &borg_spirv_options, &borg_nir_options,
                                      NULL, &nir);
   if (result != VK_SUCCESS || nir == NULL) {
      mesa_logw("borgvk: SPIR-V->NIR failed for compute stage (%d)", result);
      return;
   }
   borgvk_nir_remap_array_bindings(device, nir);
   borg_lower_nir_for_borgc(nir);
   uint32_t rc = borgc_compile_compute(nir, pipeline->cs_words, BORGVK_CS_WORDS_MAX, &pipeline->cs_nwords,
                                       pipeline->cs_regs, 32, &pipeline->cs_nregs,
                                       pipeline->cs_local, &pipeline->cs_flags,
                                       pipeline->cs_pool, BORGVK_CS_POOL_MAX, &pipeline->cs_npool);
   pipeline->cs_ok = rc == 0;
   borgvk_layout_set_base(device, pipeline->set_base);
   mesa_logi("borgvk: compute shader %s (%u words, %u presets)", rc == 0 ? "compiled" : "REFUSED",
             pipeline->cs_nwords, pipeline->cs_nregs);
   ralloc_free(nir);
}

/*
 * Copyright © 2026 Borg GPU project
 * SPDX-License-Identifier: MIT
 *
 * C side of the borgvk shader compiler. The NIR→Borg-ISA lowering lives in the
 * Rust crate `borgc` (src/borg/compiler/); this file turns the app's SPIR-V into
 * NIR (via the Mesa runtime) and hands each stage to Rust.
 */
#include "borgvk_private.h"
#include "borg_nir_passes.h"

#include "vk_pipeline.h"

#include "nir.h"
#include "glsl_types.h"
#include "spirv/nir_spirv.h"
#include "util/ralloc.h"
#include "util/log.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/* Implemented in Rust: src/borg/compiler/lib.rs. borgc_compile_nir compiles the
 * shader to a .borg blob: it writes up to buf_cap bytes into out_buf and sets
 * *out_len to the true blob size (so *out_len > buf_cap signals an undersized
 * buffer); the return value is the instruction count. */

void
borgvk_compiler_selftest(void)
{
   mesa_logi("borgvk: borgc (Rust compiler) selftest = 0x%x", borgc_selftest());
}

/* Turn one pipeline shader stage's SPIR-V into NIR and hand it to borgc. */
void
borgvk_compile_stage(struct borgvk_device *device,
                     const VkPipelineShaderStageCreateInfo *stage_info)
{
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

   /* Shared with the offline borgc CLI -- see borg_nir_passes.h for why this
    * must not be a second copy. */
   borg_lower_nir_for_borgc(nir);

   /* Map the Vulkan stage to a firmware shader slot. Only the vertex and
    * fragment stages are app-derived; anything else (e.g. compute) has no Borg
    * slot, so compile for diagnostics but don't capture a blob. */
   int slot = -1;
   switch (stage_info->stage) {
   case VK_SHADER_STAGE_VERTEX_BIT:   slot = BORGVK_STAGE_VERT; break;
   case VK_SHADER_STAGE_FRAGMENT_BIT: slot = BORGVK_STAGE_FRAG; break;
   default: break;
   }

   if (slot >= 0) {
      struct borgvk_shader_blob *b = &device->shader_blob[slot];
      uint32_t len = 0;
      borgc_compile_nir(nir, b->data, BORGVK_SHADER_BLOB_MAX, &len);
      if (len > 0 && len <= BORGVK_SHADER_BLOB_MAX) {
         b->len = len;
         mesa_logi("borgvk: captured %s shader blob (%u bytes) for upload",
                   slot == BORGVK_STAGE_VERT ? "vertex" : "fragment", len);
      } else {
         b->len = 0;
         mesa_logw("borgvk: %s shader blob not captured (size %u, max %u)",
                   slot == BORGVK_STAGE_VERT ? "vertex" : "fragment",
                   len, BORGVK_SHADER_BLOB_MAX);
      }
   } else {
      borgc_compile_nir(nir, NULL, 0, NULL);   /* diagnostics only */
   }
   ralloc_free(nir);
}

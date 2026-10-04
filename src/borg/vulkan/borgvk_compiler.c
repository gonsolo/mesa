/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 *
 * C side of the borgvk shader compiler. The NIR→Borg-ISA lowering lives in the
 * Rust crate `borgc` (src/borg/compiler/); this file turns the app's SPIR-V into
 * NIR (via the Mesa runtime) and hands each stage to Rust.
 */
#include "borgvk_private.h"
#include "borg_nir_passes.h"

#include "vk_pipeline.h"
#include "vk_pipeline_layout.h"

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

static bool
reads_point_coord(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_point_coord)
               return true;
         }
      }
   }
   return false;
}

/* Turn one pipeline shader stage's SPIR-V into NIR and hand it to borgc. */
/* The draw extension's varying count of a compiled blob (software/borg/borg_spirb.c layout). */
uint32_t
borgvk_blob_num_varyings(const struct borgvk_shader_blob *b)
{
   if (b->len < 6 || !(b->data[5] & 1))
      return 0;
   const uint32_t at = 6 + 4 * b->data[0] + b->data[3] + b->data[4] + 4 * b->data[4];
   return at < b->len ? b->data[at] : 0;
}

/* A buffer binding that is an array: borgc names buffers by (set, binding), so point each constant
 * array index at its flat slot (the slot the descriptor update wrote). */
static void
remap_array_bindings(struct borgvk_device *device, nir_shader *nir)
{
   VK_FROM_HANDLE(vk_pipeline_layout, pl, device->compile_layout);
   if (!pl)
      return;
   bool any = false;
   for (uint32_t s = 0; s < pl->set_count; s++) {
      struct vk_descriptor_set_layout *l = pl->set_layouts[s];
      any |= l && container_of(l, struct borgvk_descriptor_set_layout, vk)->map.arrays;
   }
   if (!any)
      return;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_vulkan_resource_index ||
                !nir_src_is_const(intr->src[0]))
               continue;
            uint32_t set = nir_intrinsic_desc_set(intr), binding = nir_intrinsic_binding(intr);
            if (set >= pl->set_count || !pl->set_layouts[set])
               continue;
            const struct borgvk_desc_map *m =
               &container_of(pl->set_layouts[set], struct borgvk_descriptor_set_layout, vk)->map;
            for (uint32_t j = 0; j < m->n; j++) {
               if (m->num[j] == binding) {
                  nir_intrinsic_set_binding(intr, m->slot[j] + MIN2(nir_src_as_uint(intr->src[0]), m->count[j] - 1u));
                  break;
               }
            }
         }
      }
   }
}

void
borgvk_compile_stage(struct borgvk_device *device, uint32_t vfetch,
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

   remap_array_bindings(device, nir);
   /* Shared with the offline borgc CLI -- see borg_nir_passes.h for why this
    * must not be a second copy. */
   borg_lower_nir_for_borgc(nir);
   device->frag_reads_pntc = stage_info->stage == VK_SHADER_STAGE_FRAGMENT_BIT && reads_point_coord(nir);

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
      borgc_compile_nir(nir, b->data, BORGVK_SHADER_BLOB_MAX, &len,
                        vfetch);
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
      borgc_compile_nir(nir, NULL, 0, NULL, 0);   /* diagnostics only */
   }
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
   remap_array_bindings(device, nir);
   borg_lower_nir_for_borgc(nir);
   uint32_t rc = borgc_compile_compute(nir, pipeline->cs_words, 512, &pipeline->cs_nwords,
                                       pipeline->cs_regs, 32, &pipeline->cs_nregs,
                                       pipeline->cs_local, &pipeline->cs_flags);
   pipeline->cs_ok = rc == 0;
   mesa_logi("borgvk: compute shader %s (%u words, %u presets)", rc == 0 ? "compiled" : "REFUSED",
             pipeline->cs_nwords, pipeline->cs_nregs);
   ralloc_free(nir);
}

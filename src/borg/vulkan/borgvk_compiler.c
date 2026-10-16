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

/* The uniform-buffer bindings the shader loads from, as a bit mask. */
static uint32_t
ubo_mask_used(nir_shader *nir)
{
   uint32_t mask = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic ||
                nir_instr_as_intrinsic(instr)->intrinsic != nir_intrinsic_load_ubo)
               continue;
            nir_def *d = nir_instr_as_intrinsic(instr)->src[0].ssa;
            for (int n = 0; n < 8 && d; n++) {
               nir_instr *di = nir_def_instr(d);
               if (di->type == nir_instr_type_alu) {
                  d = nir_instr_as_alu(di)->src[0].src.ssa;
               } else if (di->type == nir_instr_type_intrinsic) {
                  nir_intrinsic_instr *r = nir_instr_as_intrinsic(di);
                  if (r->intrinsic == nir_intrinsic_vulkan_resource_index) {
                     mask |= 1u << nir_intrinsic_binding(r);
                     break;
                  }
                  d = r->intrinsic == nir_intrinsic_load_vulkan_descriptor ? r->src[0].ssa : NULL;
               } else {
                  break;
               }
            }
         }
      }
   }
   return mask;
}

static bool
uses_push_constants(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir)
      nir_foreach_block(block, impl)
         nir_foreach_instr(instr, block)
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_push_constant)
               return true;
   return false;
}

static int
sampled_binding(nir_shader *nir)
{
   int binding = -1;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            int di = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
            nir_variable *var = di >= 0 ? nir_deref_instr_get_variable(nir_src_as_deref(tex->src[di].src)) : NULL;
            if (var && binding < 0)
               binding = var->data.binding;
         }
      }
   }
   return binding;
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
uint32_t
borgvk_blob_num_varyings(const struct borgvk_shader_blob *b)
{
   if (b->len < 6 || !(b->data[5] & 1))
      return 0;
   const uint32_t at = 6 + 4 * b->data[0] + b->data[3] + b->data[4] + 4 * b->data[4];
   return at < b->len ? b->data[at] : 0;
}

void
borgvk_layout_set_base(struct borgvk_device *device, uint8_t base[4])
{
   VK_FROM_HANDLE(vk_pipeline_layout, pl, device->compile_layout);
   uint32_t at = 0;
   memset(base, 0, 4);
   for (uint32_t s = 0; pl && s < pl->set_count && s < 4; s++) {
      base[s] = MIN2(at, 255u);
      if (pl->set_layouts[s])
         at += container_of(pl->set_layouts[s], struct borgvk_descriptor_set_layout, vk)->map.total;
   }
}

static void
remap_array_bindings(struct borgvk_device *device, nir_shader *nir)
{
   VK_FROM_HANDLE(vk_pipeline_layout, pl, device->compile_layout);
   if (!pl)
      return;
   uint8_t base[4];
   borgvk_layout_set_base(device, base);
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_vulkan_resource_index)
               continue;
            uint32_t set = nir_intrinsic_desc_set(intr), binding = nir_intrinsic_binding(intr);
            if (set >= pl->set_count || set >= 4 || !pl->set_layouts[set])
               continue;
            const struct borgvk_desc_map *m =
               &container_of(pl->set_layouts[set], struct borgvk_descriptor_set_layout, vk)->map;
            uint32_t slot = binding, elem = nir_src_is_const(intr->src[0]) ? nir_src_as_uint(intr->src[0]) : 0;
            for (uint32_t j = 0; j < m->n; j++) {
               if (m->num[j] == binding) {
                  slot = m->slot[j] + MIN2(elem, m->count[j] - 1u);
                  break;
               }
            }
            nir_intrinsic_set_binding(intr, base[set] + slot);
            nir_intrinsic_set_desc_set(intr, 0);
         }
      }
   }
}

static void
borgvk_lower_lines(nir_shader *nir, uint32_t instance_mask)
{
   nir_variable *pos = nir_find_variable_with_location(nir, nir_var_shader_out, VARYING_SLOT_POS);
   if (!pos)
      return;
   /* One function, so the copy holds every use of the inputs and of gl_Position. */
   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_function_temp);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_inline_functions);
   nir_remove_non_entrypoints(nir);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   const struct glsl_type *vec4 = glsl_vec4_type();
   nir_variable *own = nir_variable_create(nir, nir_var_shader_temp, vec4, "line_own");
   nir_variable *other = nir_variable_create(nir, nir_var_shader_temp, vec4, "line_other");
   nir_variable *line = nir_variable_create(nir, nir_var_shader_in, vec4, "line_step");
   line->data.location = VERT_ATTRIB_GENERIC0 + BORGVK_LINE_SLOT;

   nir_function *fn = nir_function_create(nir, "line_other_end");
   nir_function_impl *copy = nir_function_impl_clone(nir, impl);
   nir_function_set_impl(fn, copy);

   struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
   nir_foreach_block(block, copy) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;
         nir_deref_instr *d = nir_instr_as_deref(instr);
         if (d->deref_type != nir_deref_type_var)
            continue;
         nir_variable *v = d->var;
         if (v == pos) {
            d->var = other;
            continue;
         }
         const bool in = v->data.mode == nir_var_shader_in, out = v->data.mode == nir_var_shader_out;
         const int loc = v->data.location - VERT_ATTRIB_GENERIC0;
         if (!out && !(in && loc >= 0 && loc < 16 && !(instance_mask >> loc & 1)))
            continue;
         struct hash_entry *e = _mesa_hash_table_search(remap, v);
         if (!e) {
            nir_variable *n = nir_variable_create(nir, in ? nir_var_shader_in : nir_var_shader_temp, v->type, "line_end");
            if (in) {
               n->data = v->data;
               n->data.location = v->data.location + BORGVK_LINE_OTHER;
            }
            e = _mesa_hash_table_insert(remap, v, n);
         }
         d->var = e->data;
      }
   }
   _mesa_hash_table_destroy(remap, NULL);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;
         nir_deref_instr *d = nir_instr_as_deref(instr);
         if (d->deref_type == nir_deref_type_var && d->var == pos)
            d->var = own;
      }
   }

   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_call(&b, fn);
   b.cursor = nir_after_impl(impl);
   nir_def *a = nir_load_var(&b, own), *o = nir_load_var(&b, other), *l = nir_load_var(&b, line);
   nir_def *aw = nir_channel(&b, a, 3);
   nir_def *na = nir_fdiv(&b, nir_trim_vector(&b, a, 2), aw);
   nir_def *nb = nir_fdiv(&b, nir_trim_vector(&b, o, 2), nir_channel(&b, o, 3));
   nir_def *d = nir_fmul(&b, nir_fsub(&b, nb, na), nir_channels(&b, l, 0xc));
   nir_def *xmajor = nir_fge(&b, nir_fabs(&b, nir_channel(&b, d, 0)), nir_fabs(&b, nir_channel(&b, d, 1)));
   nir_def *zero = nir_imm_float(&b, 0.0f);
   nir_def *sx = nir_bcsel(&b, xmajor, zero, nir_channel(&b, l, 0));
   nir_def *sy = nir_bcsel(&b, xmajor, nir_channel(&b, l, 1), zero);
   nir_def *out = nir_vec4(&b, nir_fadd(&b, nir_channel(&b, a, 0), nir_fmul(&b, sx, aw)),
                               nir_fadd(&b, nir_channel(&b, a, 1), nir_fmul(&b, sy, aw)),
                               nir_channel(&b, a, 2), aw);
   nir_store_var(&b, pos, out, 0xf);
   nir_progress(true, impl, nir_metadata_none);
   nir_fixup_deref_modes(nir);
   NIR_PASS(_, nir, nir_inline_functions);
   nir_remove_non_entrypoints(nir);
}

/* A pipeline without a fragment stage (depth only): a shader that writes nothing. */
void
borgvk_compile_empty_frag(struct borgvk_device *device, uint32_t opt)
{
   struct borgvk_shader_blob *b = &device->shader_blob[BORGVK_STAGE_FRAG];
   nir_builder nb = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &borg_nir_options, "borgvk_empty_fs");
   borg_lower_nir_for_borgc(nb.shader);
   uint32_t len = 0;
   borgc_compile_nir(nb.shader, b->data, BORGVK_SHADER_BLOB_MAX, &len, opt);
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

   remap_array_bindings(device, nir);
   if (stage_info->stage == VK_SHADER_STAGE_VERTEX_BIT && (vfetch & (BORGC_VS_FETCH | BORGC_VS_POINT_QUADS | BORGC_VS_LINES_BIT)) == (BORGC_VS_FETCH | BORGC_VS_LINES_BIT))
      borgvk_lower_lines(nir, vfetch & 0xFFFFu);
   borg_lower_nir_for_borgc_blend(nir, blend);
   device->frag_reads_pntc = stage_info->stage == VK_SHADER_STAGE_FRAGMENT_BIT && reads_point_coord(nir);
   if (stage_info->stage == VK_SHADER_STAGE_FRAGMENT_BIT)
      device->frag_tex_binding = sampled_binding(nir);
   if (stage_info->stage == VK_SHADER_STAGE_VERTEX_BIT || stage_info->stage == VK_SHADER_STAGE_FRAGMENT_BIT)
{
      const int fs = stage_info->stage == VK_SHADER_STAGE_FRAGMENT_BIT;
      device->stage_ubo[fs] = ubo_mask_used(nir);
      device->stage_push[fs] = uses_push_constants(nir);
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
      borgc_set_output_select(slot == BORGVK_STAGE_FRAG ? (int)device->frag_select - 1 : -1);
      borgc_compile_nir(nir, b->data, BORGVK_SHADER_BLOB_MAX, &len,
                        vfetch);
      borgc_set_output_select(-1);
      device->stage_pwin[slot == BORGVK_STAGE_FRAG] = borgc_last_push_window();
      device->stage_spill[slot == BORGVK_STAGE_FRAG] = borgc_last_spill_slots() > 0;
      if (len > 0 && len <= BORGVK_SHADER_BLOB_MAX) {
         b->len = len;
         b->npool = slot == BORGVK_STAGE_FRAG ? borgc_last_frag_pool(b->pool, BORGVK_POOL_MAX) :
                    borgc_last_vert_pool(b->pool, BORGVK_POOL_MAX);
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
   borgvk_layout_set_base(device, pipeline->set_base);
   mesa_logi("borgvk: compute shader %s (%u words, %u presets)", rc == 0 ? "compiled" : "REFUSED",
             pipeline->cs_nwords, pipeline->cs_nregs);
   ralloc_free(nir);
}

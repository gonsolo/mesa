/*
 * Copyright © 2024 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 */

#include <stdbool.h>

#include "nir.h"
#include "nir_builder.h"

#include "borg_cmd_buffer.h"
#include "borg_descriptor_set_layout.h"
#include "borg_private.h"
#include "borg_shader.h"

struct lower_descriptors_ctx {
        const struct borg_descriptor_set_layout *set_layouts[BORG_MAX_SETS];

        nir_address_format ssbo_addr_format;
};

static bool
descriptor_type_is_ssbo(VkDescriptorType desc_type)
{
        switch (desc_type) {
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                return true;

                default:
                return false;
        }
}

static const struct borg_descriptor_set_binding_layout *
get_binding_layout(uint32_t set, uint32_t binding,
                   const struct lower_descriptors_ctx *ctx)
{
   assert(set < BORG_MAX_SETS);
   assert(ctx->set_layouts[set] != NULL);

   const struct borg_descriptor_set_layout *set_layout = ctx->set_layouts[set];

   assert(binding < set_layout->binding_count);
   return &set_layout->binding[binding];
}

static nir_def *
load_descriptor_set_addr(nir_builder *b, uint32_t set,
                         UNUSED const struct lower_descriptors_ctx *ctx)
{
   puts("load_descriptor_set_addr");

   uint32_t set_addr_offset = borg_root_descriptor_offset(sets) +
      set * sizeof(struct borg_buffer_address);

   return nir_load_ubo(b, 1, 64, nir_imm_int(b, 0),
                       nir_imm_int(b, set_addr_offset),
                       .align_mul = 8, .align_offset = 0, .range = ~0);

   return NULL;
}


static bool
lower_ssbo_resource_index(nir_builder *b, nir_intrinsic_instr *intrin,
                            const struct lower_descriptors_ctx *ctx)
{
   puts("lower_ssbo_resource_index");

   if (!descriptor_type_is_ssbo(nir_intrinsic_desc_type(intrin)))
      return false;

   puts("  descriptor type is ssbo");

   b->cursor = nir_instr_remove(&intrin->instr);

   uint32_t set = nir_intrinsic_desc_set(intrin);
   uint32_t binding = nir_intrinsic_binding(intrin);
   nir_def *index = intrin->src[0].ssa;

   const struct borg_descriptor_set_binding_layout *binding_layout = get_binding_layout(set, binding, ctx);

   nir_def *binding_addr;
   uint8_t binding_stride;

   switch (binding_layout->type) {
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: {
         nir_def *set_addr = load_descriptor_set_addr(b, set, ctx);
         binding_addr = nir_iadd_imm(b, set_addr, binding_layout->offset);
         binding_stride = binding_layout->stride;
         break;
      }
      default:
         printf("unhandled binding_layout type: %i\n", binding_layout->type);
         unreachable("Not an SSBO descriptor");
   }

   binding_addr = nir_ior_imm(b, binding_addr, (uint64_t)binding_stride << 56);

   const uint32_t binding_size = binding_layout->array_size * binding_stride;
   nir_def *offset_in_binding = nir_imul_imm(b, index, binding_stride);

   assert(binding_layout->array_size >= 1);

   nir_def *addr;
   switch (ctx->ssbo_addr_format) {
   case nir_address_format_64bit_global_32bit_offset:
   case nir_address_format_64bit_bounded_global:
      addr = nir_vec4(b, nir_unpack_64_2x32_split_x(b, binding_addr),
                         nir_unpack_64_2x32_split_y(b, binding_addr),
                         nir_imm_int(b, binding_size),
                         offset_in_binding);
      break;

   default:
      unreachable("Unknown address mode");
   }

   nir_def_rewrite_uses(&intrin->def, addr);

   return true;
}


static bool
lower_ssbo_descriptor_instr(nir_builder *b, nir_instr *instr, void *_data)
{
   puts("lower_ssbo_descriptor_instr");

   const struct lower_descriptors_ctx *ctx = _data;

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
   switch (intrin->intrinsic) {
   case nir_intrinsic_vulkan_resource_index:
      return lower_ssbo_resource_index(b, intrin, ctx);
   default:
      return false;
   }
}

bool
borg_nir_lower_descriptors(nir_shader *nir,
                           const struct borg_physical_device *pdev,
                           uint32_t set_layout_count,
                           struct vk_descriptor_set_layout * const *set_layouts)
{
   struct lower_descriptors_ctx ctx = {
      .ssbo_addr_format = nir_address_format_64bit_global,
   };

   assert(set_layout_count <= BORG_MAX_SETS);
   for (uint32_t s = 0; s < set_layout_count; s++) {
   if (set_layouts[s] != NULL)
      ctx.set_layouts[s] = vk_to_borg_descriptor_set_layout(set_layouts[s]);
   }

   return nir_shader_instructions_pass(nir, lower_ssbo_descriptor_instr,
                                       nir_metadata_control_flow,
                                       (void *)&ctx);
}


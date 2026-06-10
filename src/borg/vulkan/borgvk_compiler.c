/*
 * Copyright © 2026 Borg GPU project
 * SPDX-License-Identifier: MIT
 *
 * C side of the borgvk shader compiler. The NIR→Borg-ISA lowering lives in the
 * Rust crate `borgc` (src/borg/compiler/); this file turns the app's SPIR-V into
 * NIR (via the Mesa runtime) and hands each stage to Rust.
 */
#include "borgvk_private.h"

#include "vk_pipeline.h"

#include "nir.h"
#include "glsl_types.h"
#include "spirv/nir_spirv.h"
#include "util/ralloc.h"
#include "util/log.h"

#include <stdint.h>

/* Implemented in Rust: src/borg/compiler/lib.rs. */
uint32_t borgc_selftest(void);
uint32_t borgc_compile_nir(struct nir_shader *nir);

void
borgvk_compiler_selftest(void)
{
   mesa_logi("borgvk: borgc (Rust compiler) selftest = 0x%x", borgc_selftest());
}

/* SPIR-V → NIR options. Addressing formats mirror a software driver (lavapipe);
 * the Borg backend reads resources from the firmware's uniform/descriptor model,
 * so the exact formats only need to be self-consistent for NIR construction. */
static const struct spirv_to_nir_options borgvk_spirv_options = {
   .environment            = NIR_SPIRV_VULKAN,
   .ubo_addr_format        = nir_address_format_vec2_index_32bit_offset,
   .ssbo_addr_format       = nir_address_format_vec2_index_32bit_offset,
   .phys_ssbo_addr_format  = nir_address_format_64bit_global,
   .push_const_addr_format = nir_address_format_logical,
   .shared_addr_format     = nir_address_format_32bit_offset,
   .constant_addr_format   = nir_address_format_64bit_global,
};

/* NIR compiler options for the Borg target. Minimal for now (defaults); the
 * lowering set grows as the Rust backend dictates what it wants pre-lowered. */
static const struct nir_shader_compiler_options borgvk_nir_options = {
   .lower_fdiv = true,
};

/* Location (vec4-slot) sizing for nir_lower_io of varyings. */
static int
borgvk_type_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

/* Turn one pipeline shader stage's SPIR-V into NIR and hand it to borgc. */
void
borgvk_compile_stage(struct borgvk_device *device,
                     const VkPipelineShaderStageCreateInfo *stage_info)
{
   struct nir_shader *nir = NULL;
   VkResult result =
      vk_pipeline_shader_stage_to_nir(&device->vk, 0, stage_info,
                                      &borgvk_spirv_options, &borgvk_nir_options,
                                      NULL, &nir);
   if (result != VK_SUCCESS || nir == NULL) {
      mesa_logw("borgvk: SPIR-V→NIR failed for stage 0x%x (%d)",
                stage_info->stage, result);
      return;
   }

   /* SSA + system values (gl_VertexIndex → load_vertex_id), then lower the UBO
    * deref I/O to load_ubo with byte offsets. The Borg core now has integer ops
    * (iadd/ishl), so the offset address arithmetic is selectable — this is the
    * standard Mesa addressing path, unblocked by the new integer ALU. */
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_lower_system_values);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ubo,
            borgvk_spirv_options.ubo_addr_format);
   /* Lower varying I/O to load_input/store_output(location): the outputs
    * (gl_Position/texcoord/frag_pos) become store_output, which the backend maps
    * to the sequencer's snooped output registers; fragment inputs become
    * load_input. vec4-slot sizing. */
   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
            borgvk_type_size, 0);

   /* Lower/optimize toward the Borg ISA: scalarize, then fold and lower ALU ops
    * (fsub→fadd, fdiv→fmul·frcp via lower_fdiv, fdot→fmul+ffma, constant folding)
    * so the backend sees the small supported op set. */
   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      /* Borg has no branches: flatten if/else (e.g. cube.frag's linearToSrgb
       * piecewise) into bcsel selects. expensive_alu_ok so the pow branch doesn't
       * block flattening — the whole select then collapses to one FSRGB op. */
      static const nir_opt_peephole_select_options peephole_opts = {
         .limit = 64, .indirect_load_ok = true, .expensive_alu_ok = true,
      };
      NIR_PASS(progress, nir, nir_opt_peephole_select, &peephole_opts);
   } while (progress);

   if (getenv("BORGC_DUMP_NIR"))
      nir_print_shader(nir, stderr);

   borgc_compile_nir(nir);   /* Rust: inspects/lowers the shader */
   ralloc_free(nir);
}

/*
 * Copyright © 2026 Borg GPU project
 * SPDX-License-Identifier: MIT
 *
 * See borg_nir_passes.h for why this is shared rather than duplicated.
 */
#include "borg_nir_passes.h"

#include "glsl_types.h"
#include "util/ralloc.h"

#include <stdlib.h>

const struct spirv_to_nir_options borg_spirv_options = {
   .environment            = NIR_SPIRV_VULKAN,
   .ubo_addr_format        = nir_address_format_vec2_index_32bit_offset,
   .ssbo_addr_format       = nir_address_format_vec2_index_32bit_offset,
   .phys_ssbo_addr_format  = nir_address_format_64bit_global,
   .push_const_addr_format = nir_address_format_logical,
   .shared_addr_format     = nir_address_format_32bit_offset,
   .constant_addr_format   = nir_address_format_64bit_global,
};

const struct nir_shader_compiler_options borg_nir_options = {
   .lower_fdiv = true,
};

/* Location (vec4-slot) sizing for nir_lower_io of varyings. */
static int
borg_type_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

void
borg_lower_nir_for_borgc(struct nir_shader *nir)
{
   /* SSA + system values (gl_VertexIndex → load_vertex_id), then lower the UBO
    * deref I/O to load_ubo with byte offsets. The Borg core has integer ops
    * (iadd/ishl), so the offset address arithmetic is selectable. */
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_lower_system_values);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ubo,
            borg_spirv_options.ubo_addr_format);
   /* Varying I/O → load_input/store_output(location). vec4-slot sizing. */
   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
            borg_type_size, 0);

   /* Lower/optimize toward the Borg ISA: scalarize, then fold and lower ALU ops
    * (fsub→fadd, fdiv→fmul·frcp via lower_fdiv, fdot→fmul+ffma, constant
    * folding) so the backend sees the small supported op set. */
   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      /* Flatten if/else into bcsel selects where that is cheaper than
       * predication.
       *
       * Borg DOES have control flow now -- BRZ/BRNZ plus an execution mask
       * (EXPUSH/EXELSE/EXPOP), which borgc lowers nir_if onto. Flattening is
       * still preferable for a small `if`, since one bcsel beats five
       * instructions plus both arms executing. The limit is what decides
       * where that trade tips, and it is deliberately still high: lowering it
       * would change codegen for every existing shader, which is a separate
       * change with its own image comparison. Raising the mask's share of
       * control flow is a tuning exercise to do against measurements, not a
       * side effect of adding the capability.
       *
       * expensive_alu_ok so cube.frag's linearToSrgb pow branch does not block
       * flattening -- the whole select then collapses to one FSRGB op. */
      static const nir_opt_peephole_select_options peephole_opts = {
         .limit = 64, .indirect_load_ok = true, .expensive_alu_ok = true,
      };
      NIR_PASS(progress, nir, nir_opt_peephole_select, &peephole_opts);
   } while (progress);

   if (getenv("BORGC_DUMP_NIR"))
      nir_print_shader(nir, stderr);
}

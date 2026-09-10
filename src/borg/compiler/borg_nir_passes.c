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
   /* The Vulkan-runtime preamble.
    *
    * The driver gets this from vk_spirv_to_nir before its own passes run; the
    * offline CLI cannot, because vk_spirv_to_nir dereferences
    * device->physical->properties and standing up a device is exactly what the
    * CLI exists to avoid. So it lives here, where both front ends get it and
    * neither can drift. Every pass below is idempotent, so the driver running
    * it a second time costs a walk and changes nothing.
    *
    * Skipping it is not subtle: cube.frag calls linearToSrgb(), and WITHOUT
    * nir_inline_functions that call is never inlined, the output store stays a
    * store_deref borgc does not understand, and dead-code elimination then
    * deletes almost the whole shader -- 66 selected instructions emitting 7.
    * Local constant initializers have to be lowered before inlining or they
    * get initialized in the caller instead of the callee. */
   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_function_temp);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_inline_functions);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_deref);
   NIR_PASS(_, nir, nir_lower_variable_initializers, ~0);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_split_per_member_structs);
   NIR_PASS(_, nir, nir_remove_dead_variables,
            nir_var_shader_in | nir_var_shader_out | nir_var_system_value, NULL);
   NIR_PASS(_, nir, nir_lower_var_copies);

   /* SSA + system values (gl_VertexIndex → load_vertex_id), then lower the UBO
    * deref I/O to load_ubo with byte offsets. The Borg core has integer ops
    * (iadd/ishl), so the offset address arithmetic is selectable. */
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_lower_system_values);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ubo,
            borg_spirv_options.ubo_addr_format);
   /* Push constants. Confirmed empirically (not assumed): with
    * push_const_addr_format left at its default, a `layout(push_constant)`
    * access arrives here as a plain load_deref, same as any other pointer,
    * and the flat selection walk below has no deref handling -- the whole
    * shader silently DCE'd to nothing (0 instructions selected) the first
    * time this was tried. Lowering to an explicit byte offset turns it into
    * nir_intrinsic_load_push_constant, which IS something a flat walk can
    * select from directly. */
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const,
            nir_address_format_32bit_offset);
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

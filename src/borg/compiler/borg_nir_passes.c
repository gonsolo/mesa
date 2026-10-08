/*
 * Copyright © 2026 Andreas Wendleder
 * SPDX-License-Identifier: MIT
 *
 * See borg_nir_passes.h for why this is shared rather than duplicated.
 */
#include "borg_nir_passes.h"

#include "glsl_types.h"
#include "nir_builder.h"
#include "vulkan/vulkan_core.h"
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

/* Compute: texel buffers.  A texelFetch on a textureBuffer / imageLoad on an imageBuffer
 * becomes four word loads from the binding's window, 16 bytes per texel.  The buffer view's
 * format is only known at dispatch, so the driver decodes the view into that canonical
 * four-word layout (RGBA as the shader's declared type: float, uint or int bits) before the
 * dispatch; the shader never sees the format. */
static bool
borg_texel_buffer_filter(const nir_instr *instr, const void *data)
{
   if (instr->type == nir_instr_type_tex) {
      const nir_tex_instr *tex = nir_instr_as_tex(instr);
      return tex->op == nir_texop_txf && tex->sampler_dim == GLSL_SAMPLER_DIM_BUF;
   }
   if (instr->type == nir_instr_type_intrinsic) {
      const nir_intrinsic_instr *i = nir_instr_as_intrinsic(instr);
      return (i->intrinsic == nir_intrinsic_image_deref_load &&
              nir_intrinsic_image_dim(i) == GLSL_SAMPLER_DIM_BUF) ||
             (i->intrinsic == nir_intrinsic_image_deref_store &&
              nir_intrinsic_image_dim(i) == GLSL_SAMPLER_DIM_2D &&
              !nir_intrinsic_image_array(i));
   }
   return false;
}

static nir_def *
borg_texel_buffer_lower(nir_builder *b, nir_instr *instr, void *data)
{
   nir_deref_instr *deref;
   nir_def *coord;
   if (instr->type == nir_instr_type_intrinsic &&
       nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_image_deref_store) {
      /* A 2D image store (single 32-bit texel formats): the binding's window holds the image's
       * width in word 0, then the texels row by row; the host copies the image in and back. */
      nir_intrinsic_instr *st = nir_instr_as_intrinsic(instr);
      nir_variable *ivar = nir_deref_instr_get_variable(nir_src_as_deref(st->src[0]));
      if (!ivar)
         return NULL;
      const nir_address_format afmt = borg_spirv_options.ssbo_addr_format;
      nir_def *ridx = nir_vulkan_resource_index(b, nir_address_format_num_components(afmt),
                                                nir_address_format_bit_size(afmt), nir_imm_int(b, 0),
                                                .desc_set = ivar->data.descriptor_set,
                                                .binding = ivar->data.binding,
                                                .desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
      nir_def *rdesc = nir_load_vulkan_descriptor(b, nir_address_format_num_components(afmt),
                                                  nir_address_format_bit_size(afmt), ridx,
                                                  .desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
      nir_def *buf = nir_channel(b, rdesc, 0);
      nir_def *width = nir_load_ssbo(b, 1, 32, buf, nir_imm_int(b, 0), .align_mul = 4);
      nir_def *x = nir_channel(b, st->src[1].ssa, 0);
      nir_def *y = nir_channel(b, st->src[1].ssa, 1);
      nir_def *word = nir_iadd(b, nir_iadd(b, nir_imul(b, y, width), x), nir_imm_int(b, 1));
      nir_store_ssbo(b, nir_channel(b, st->src[3].ssa, 0), buf, nir_ishl_imm(b, word, 2),
                     .write_mask = 1, .align_mul = 4);
      return NIR_LOWER_INSTR_PROGRESS_REPLACE;
   }
   if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      int di = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
      int ci = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      if (di < 0 || ci < 0)
         return NULL;
      deref = nir_src_as_deref(tex->src[di].src);
      coord = nir_channel(b, tex->src[ci].src.ssa, 0);
   } else {
      nir_intrinsic_instr *i = nir_instr_as_intrinsic(instr);
      deref = nir_src_as_deref(i->src[0]);
      coord = nir_channel(b, i->src[1].ssa, 0);
   }
   nir_variable *var = deref ? nir_deref_instr_get_variable(deref) : NULL;
   if (!var)
      return NULL;

   const nir_address_format fmt = borg_spirv_options.ssbo_addr_format;
   nir_def *idx = nir_vulkan_resource_index(b, nir_address_format_num_components(fmt),
                                            nir_address_format_bit_size(fmt), nir_imm_int(b, 0),
                                            .desc_set = var->data.descriptor_set,
                                            .binding = var->data.binding,
                                            .desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
   nir_def *desc = nir_load_vulkan_descriptor(b, nir_address_format_num_components(fmt),
                                              nir_address_format_bit_size(fmt), idx,
                                              .desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
   nir_def *off = nir_imul_imm(b, coord, 16);
   return nir_load_ssbo(b, 4, 32, nir_channel(b, desc, 0), off, .align_mul = 16);
}

/* Cube sampling: the hardware takes the face (TEXA's layer) and the face coordinates s, t in
 * [0, 1]; the direction's major-axis projection is done here (Vulkan 16.5.4). The cube
 * becomes a 2D array sample whose layer is the face. */
static bool
borg_lower_cube(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE || tex->is_array)
      return false;
   const int ci = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (ci < 0)
      return false;
   b->cursor = nir_before_instr(instr);
   nir_def *c = tex->src[ci].src.ssa;
   nir_def *x = nir_channel(b, c, 0), *y = nir_channel(b, c, 1), *z = nir_channel(b, c, 2);
   /* Non-negative floats order like their bits: the comparisons are integer ones, two
    * instructions each. A sign test is the sign bit. */
   nir_def *ax = nir_fabs(b, x), *ay = nir_fabs(b, y), *az = nir_fabs(b, z);
   nir_def *xneg = nir_ilt_imm(b, x, 0), *yneg = nir_ilt_imm(b, y, 0), *zneg = nir_ilt_imm(b, z, 0);
   nir_def *nx = nir_fneg(b, x), *ny = nir_fneg(b, y), *nz = nir_fneg(b, z);
   /* per major axis: |ma|, sc, tc, face */
   nir_def *X[4] = { ax, nir_bcsel(b, xneg, z, nz), ny, nir_b2f32(b, xneg) };
   nir_def *Y[4] = { ay, x, nir_bcsel(b, yneg, nz, z), nir_fadd_imm(b, nir_b2f32(b, yneg), 2.0) };
   nir_def *Z[4] = { az, nir_bcsel(b, zneg, nx, x), ny, nir_fadd_imm(b, nir_b2f32(b, zneg), 4.0) };
   nir_def *x_over_y = nir_ult(b, ay, ax);                          /* X beats Y; ties go to Y */
   nir_def *not_z = nir_ior(b, nir_ult(b, az, ax), nir_ult(b, az, ay));   /* Z wins ties */
   nir_def *r[4];
   for (int i = 0; i < 4; i++)
      r[i] = nir_bcsel(b, not_z, nir_bcsel(b, x_over_y, X[i], Y[i]), Z[i]);
   nir_def *half_inv = nir_fmul_imm(b, nir_frcp(b, r[0]), 0.5f);
   nir_def *half = nir_imm_float(b, 0.5f);
   nir_def *st = nir_vec3(b, nir_ffma(b, r[1], half_inv, half), nir_ffma(b, r[2], half_inv, half), r[3]);
   nir_src_rewrite(&tex->src[ci].src, st);
   tex->coord_components = 3;
   tex->is_array = true;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   return true;
}

#ifdef BORG_SMALL_NIR
/* nir_opt_algebraic's tables are 8.6 MB; the on-board build lowers only what the ISA lacks. */
static bool
borg_basic_alu_filter(const nir_instr *instr, const void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   const nir_op op = nir_instr_as_alu(instr)->op;
   return op == nir_op_fsub || op == nir_op_fdiv || op == nir_op_fsqrt;
}

static nir_def *
borg_basic_alu_lower(nir_builder *b, nir_instr *instr, void *data)
{
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   nir_def *x = nir_ssa_for_alu_src(b, alu, 0);
   switch (alu->op) {
   case nir_op_fsub:
      return nir_fadd(b, x, nir_fneg(b, nir_ssa_for_alu_src(b, alu, 1)));
   case nir_op_fdiv:
      return nir_fmul(b, x, nir_frcp(b, nir_ssa_for_alu_src(b, alu, 1)));
   default:
      return nir_frcp(b, nir_frsq(b, x));
   }
}
#endif

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
   /* A constant array indexed by gl_VertexIndex (the usual vertex-less fullscreen
    * triangle/quad) becomes a bcsel chain. */
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees, nir_var_function_temp | nir_var_shader_temp, UINT32_MAX);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_lower_system_values);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ubo,
            borg_spirv_options.ubo_addr_format);
   /* Storage buffers (compute): load/store/atomic_ssbo intrinsics. */
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo,
            borg_spirv_options.ssbo_addr_format);
   if (nir->info.stage == MESA_SHADER_COMPUTE)
      NIR_PASS(_, nir, nir_shader_lower_instructions, borg_texel_buffer_filter,
               borg_texel_buffer_lower, NULL);
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
   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      NIR_PASS(_, nir, nir_shader_instructions_pass, borg_lower_cube, nir_metadata_control_flow, NULL);
   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
#ifdef BORG_SMALL_NIR
      NIR_PASS(progress, nir, nir_shader_lower_instructions, borg_basic_alu_filter,
               borg_basic_alu_lower, NULL);
#else
      NIR_PASS(progress, nir, nir_opt_algebraic);
#endif
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
      /* BORGC_NO_FLATTEN keeps every `if` as control flow, to exercise the mask path. */
      const nir_opt_peephole_select_options peephole_opts = {
         .limit = getenv("BORGC_NO_FLATTEN") ? 1 : 64, .indirect_load_ok = true, .expensive_alu_ok = true,
      };
      NIR_PASS(progress, nir, nir_opt_peephole_select, &peephole_opts);
   } while (progress);

   /* Compute: booleans as 32-bit integers. The backend has no select; both arms
    * of an `if` store into the phi's register. */
   if (nir->info.stage == MESA_SHADER_COMPUTE)
      NIR_PASS(_, nir, nir_lower_bool_to_int32);
   /* Phis become registers for every stage. */
   NIR_PASS(_, nir, nir_convert_from_ssa, true, false);

   if (getenv("BORGC_DUMP_NIR"))
      nir_print_shader(nir, stderr);
}
